//! One text-to-image request end to end: tokenize, encode text (streamed layers), refine, run the six PDD steps, decode.
use super::prepare::{PlanKey, PreparedDenoise, PreparedVae};
use crate::cuda_minimax_h3_image::backend::ops::GemmPolicy;
use crate::cuda_minimax_h3_image::backend::{AttentionPolicy, CudaBuffer, CudaContext};
use crate::cuda_minimax_h3_image::config::*;
use crate::cuda_minimax_h3_image::model::blocks::{refine_text, rows, TextBlocks};
use crate::cuda_minimax_h3_image::model::host::{patchify, unpatchify};
use crate::cuda_minimax_h3_image::model::noise::randn;
use crate::cuda_minimax_h3_image::model::vae_host::Chw;
use crate::cuda_minimax_h3_image::weights::{DitWeights, TextWeights};
use crate::image::{GeneratedImage, ImageRequest, ImageRuntime};
use apxinf_core::{Error, Result, Tensor};
use apxinf_tokenizer::Tokenizer;
use std::rc::Rc;

pub(in crate::cuda_minimax_h3_image) struct ModelRunner {
    ctx: Rc<CudaContext>,
    dit: Rc<DitWeights>,
    vae: PreparedVae,
    tokenizer: Tokenizer,
    text: TextWeights,
    gemm: GemmPolicy,
    attention: AttentionPolicy,
    /// Capture a CUDA Graph per request shape (default). `APXINF_H3_IMAGE_EAGER` runs the same forward eagerly instead.
    capture: bool,
    /// The plan of the most recent request shape. One plan at a time bounds device memory: a plan's arena is about
    /// 160 KiB per packed row.
    plan: Option<PreparedDenoise>,
    /// Refined text rows of recent prompts, most recent first, keyed by token ids. The text stage is a pure function of the
    /// ids, so a repeated prompt (a new seed or canvas) skips the 22.7 GiB weight stream entirely.
    conditioning: Vec<(Vec<u32>, Tensor)>,
}

/// Prompts whose refined text is kept. Each entry is at most `MAX_TEXT_TOKENS * HIDDEN * 2` bytes (5.25 MiB).
const CONDITIONING_CACHE: usize = 8;

/// What a request resolves to before any device work.
pub(super) struct Plan {
    pub canvas: Canvas,
    pub layout: Layout,
    pub ids: Vec<u32>,
}

/// Per-stage wall time of a request, printed to stderr when `APXINF_H3_IMAGE_TIMINGS` is set. Every stage ends with a
/// device synchronisation, so enabling it serialises work that would otherwise overlap; leave it off for latency numbers.
struct Stages<'a> {
    ctx: &'a CudaContext,
    enabled: bool,
    last: std::time::Instant,
}

impl<'a> Stages<'a> {
    fn new(ctx: &'a CudaContext) -> Self {
        Self {
            ctx,
            enabled: std::env::var_os("APXINF_H3_IMAGE_TIMINGS").is_some(),
            last: std::time::Instant::now(),
        }
    }

    fn mark(&mut self, name: &str) {
        if !self.enabled {
            return;
        }
        let _ = self.ctx.synchronize();
        eprintln!(
            "  stage {name:<22} {:7.3} s",
            self.last.elapsed().as_secs_f64()
        );
        self.last = std::time::Instant::now();
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(format!("MiniMax-H3-Image: {}", message.into()))
}

fn download_f32(ctx: &CudaContext, tensor: &Tensor) -> Result<Vec<f32>> {
    let mut bytes = vec![0u8; tensor.numel() * 4];
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .copy_to_host_on(&mut bytes, ctx.stream())
        .map_err(Error::Cuda)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

impl ModelRunner {
    pub fn new(
        ctx: Rc<CudaContext>,
        dit: Rc<DitWeights>,
        vae: PreparedVae,
        tokenizer: Tokenizer,
        text: TextWeights,
        gemm: GemmPolicy,
        attention: AttentionPolicy,
    ) -> Self {
        Self {
            ctx,
            dit,
            vae,
            tokenizer,
            text,
            gemm,
            attention,
            capture: Self::capture_enabled(),
            plan: None,
            conditioning: Vec::new(),
        }
    }

    /// Graph capture is the default; `APXINF_H3_IMAGE_EAGER` runs the same operators eagerly (debugging, parity checks).
    pub fn capture_enabled() -> bool {
        std::env::var_os("APXINF_H3_IMAGE_EAGER").is_none()
    }

    /// Everything that can be rejected without touching the device.
    pub(super) fn resolve(&self, request: &ImageRequest<'_>) -> Result<Plan> {
        if request.steps != PDD_STEPS {
            return Err(invalid(format!(
                "this checkpoint is a {PDD_STEPS}-step PDD release; steps must be {PDD_STEPS}, got {}",
                request.steps
            )));
        }
        let canvas = Canvas::new(request.width, request.height)?;
        if request.prompt.trim().is_empty() {
            return Err(invalid("the prompt is empty"));
        }
        // raw text, no special tokens, no chat template (both reference pipelines)
        let ids = self
            .tokenizer
            .encode_with_special_tokens(request.prompt, false)?;
        let layout = Layout::new(ids.len(), canvas)?;
        Ok(Plan {
            canvas,
            layout,
            ids,
        })
    }

    /// Text conditioning: Qwen3 stack over streamed layers, then `condition_proj` and the token refiner.
    fn text_conditioning(&self, ids: &[u32], stages: &mut Stages<'_>) -> Result<Tensor> {
        let ctx = &self.ctx;
        let real = ids.len();
        let padded = real.div_ceil(TE_ROW_BUCKET) * TE_ROW_BUCKET;
        let embeddings = self.text.embed(ids)?;
        let blocks = TextBlocks::new(ctx.clone(), padded)?;
        let x = blocks.embed(&embeddings, real)?;
        stages.mark("embed rows");
        self.text
            .run_layers(|_, layer| blocks.layer(layer, &x, &self.gemm, &self.attention))?;
        if stages.enabled {
            eprintln!(
                "  text layers: streamed by {}",
                if self.text.is_pinned() {
                    "DMA from the page-locked file"
                } else {
                    "the pageable path (file not page-locked)"
                }
            );
        }
        stages.mark("text encoder");
        let states = rows(&x, 0, real, TE_DIM)?;
        let refined = refine_text(ctx, &self.dit, &states, &self.gemm, &self.attention)?;
        stages.mark("token refiner");
        Ok(refined)
    }

    /// The plan for `key`, built (and captured) on first use or when the request shape changes.
    fn plan_for(&mut self, key: PlanKey, stages: &mut Stages<'_>) -> Result<&PreparedDenoise> {
        if self.plan.as_ref().is_none_or(|plan| plan.key != key) {
            // release the old plan's arena and graph before the new one allocates its own
            self.plan = None;
            let plan = PreparedDenoise::new(
                self.ctx.clone(),
                self.dit.clone(),
                key,
                self.capture,
                self.gemm.clone(),
                self.attention.clone(),
            )?;
            if stages.enabled {
                if let Some((used, capacity)) = plan.arena() {
                    eprintln!(
                        "  arena: {:.0} MiB used of {:.0} MiB",
                        used as f64 / 1048576.0,
                        capacity as f64 / 1048576.0
                    );
                }
            }
            stages.mark(if plan.is_captured() {
                "prepare + capture"
            } else {
                "prepare (eager)"
            });
            self.plan = Some(plan);
        }
        Ok(self.plan.as_ref().expect("plan was just installed"))
    }

    pub(super) fn run(&mut self, request: &ImageRequest<'_>) -> Result<GeneratedImage> {
        let ctx = self.ctx.clone();
        let mut stages = Stages::new(&ctx);
        let resolved = self.resolve(request)?;
        stages.mark("tokenize");
        let text = match self
            .conditioning
            .iter()
            .position(|(ids, _)| *ids == resolved.ids)
        {
            Some(index) => {
                let entry = self.conditioning.remove(index);
                let text = entry.1.clone();
                self.conditioning.insert(0, entry);
                stages.mark("text (cached)");
                text
            }
            None => {
                let text = self.text_conditioning(&resolved.ids, &mut stages)?;
                self.conditioning
                    .insert(0, (resolved.ids.clone(), text.clone()));
                self.conditioning.truncate(CONDITIONING_CACHE);
                text
            }
        };
        let (h, w) = (
            resolved.canvas.latent_height(),
            resolved.canvas.latent_width(),
        );
        let noise = randn(request.seed, LATENT_CHANNELS * h * w);
        let prepared = self.plan_for(
            PlanKey {
                layout: resolved.layout,
                canvas: resolved.canvas,
            },
            &mut stages,
        )?;
        prepared.bind(&text, &patchify(&noise, h, w))?;
        drop(text);
        for step in 0..PDD_STEPS {
            prepared.step(step)?;
            stages.mark(&format!("dit step {step}"));
        }
        let latent = download_f32(&ctx, prepared.latent())?;
        let samples = Chw {
            c: LATENT_CHANNELS,
            h,
            w,
            data: unpatchify(&latent, h, w),
        };
        stages.mark("latent download");
        let rgb = self.vae.decode_image(&samples)?;
        stages.mark("vae decode");
        Ok(GeneratedImage {
            width: resolved.canvas.width,
            height: resolved.canvas.height,
            rgb,
        })
    }
}

impl ImageRuntime for ModelRunner {
    fn generate(&mut self, request: &ImageRequest<'_>) -> Result<GeneratedImage> {
        self.run(request)
    }
}
