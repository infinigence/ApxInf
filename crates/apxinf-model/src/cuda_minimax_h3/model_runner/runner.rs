//! One text-to-video-and-audio request up to the denoised latents: tokenize, encode text (streamed layers), refine, draw the two noise
//! tensors from one seeded generator and run the sampler.
use super::audio::AudioDecoder;
use super::prepare::{PlanKey, PreparedDenoise};
use super::video::VideoDecoder;
use crate::cuda_minimax_h3::backend::ops::GemmPolicy;
use crate::cuda_minimax_h3::backend::{AttentionPolicy, CudaContext};
use crate::cuda_minimax_h3::config::*;
use crate::cuda_minimax_h3::model::blocks::{refine_text, rows, TextBlocks};
use crate::cuda_minimax_h3::model::host;
use crate::cuda_minimax_h3::model::layout::{
    align_frame_count, audio_latent_steps, video_latent_steps, Layout,
};
use crate::cuda_minimax_h3::model::noise::Generator;
use crate::cuda_minimax_h3::weights::{DitWeights, TextWeights};
use crate::video::{GeneratedVideo, VideoRequest, VideoRuntime};
use apxinf_core::{Error, Result, Tensor};
use apxinf_tokenizer::Tokenizer;
use std::rc::Rc;

/// One request. `frames` is snapped up onto the `17 k + 5` grid the model was trained on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::cuda_minimax_h3) struct GenerationRequest<'a> {
    pub prompt: &'a str,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub seed: u64,
    pub steps: usize,
}

/// The denoised latents of a request in the layout of ComfyUI's samples: video `[LATENT_CHANNELS, t, h, w]`, audio
/// `[AUDIO_CHANNELS, AUDIO_STEREO, audio_t]`, both FP32 row-major.
pub(in crate::cuda_minimax_h3) struct Latents {
    pub video: Vec<f32>,
    pub audio: Vec<f32>,
    /// Frames the video will have after decoding (the request's, snapped onto the grid).
    pub frames: usize,
    pub layout: Layout,
}

/// What a request resolves to before any device work.
pub(in crate::cuda_minimax_h3) struct Plan {
    pub layout: Layout,
    pub frames: usize,
    pub ids: Vec<u32>,
    pub steps: usize,
}

pub(in crate::cuda_minimax_h3) struct ModelRunner {
    ctx: Rc<CudaContext>,
    dit: Rc<DitWeights>,
    tokenizer: Tokenizer,
    text: TextWeights,
    video: VideoDecoder,
    audio: AudioDecoder,
    gemm: GemmPolicy,
    attention: AttentionPolicy,
    /// Capture a CUDA Graph per request shape (default). `APXINF_H3_EAGER` runs the same forward eagerly instead.
    capture: bool,
    /// The plan of the most recent request shape; one plan at a time bounds device memory.
    plan: Option<PreparedDenoise>,
    /// Refined text rows of recent prompts, most recent first, keyed by token ids. The text stage is a pure function of the ids, so
    /// a repeated prompt (a new seed or length) skips the 24 GiB weight stream entirely.
    conditioning: Vec<(Vec<u32>, Tensor)>,
}

/// Prompts whose refined text is kept. Each entry is at most `MAX_TEXT_TOKENS * HIDDEN * 2` bytes (5.25 MiB).
const CONDITIONING_CACHE: usize = 8;

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(format!("MiniMax-H3: {}", message.into()))
}

/// Per-stage wall time, printed to stderr when `APXINF_H3_TIMINGS` is set. Every stage ends with a device synchronisation, so enabling
/// it serialises work that would otherwise overlap; leave it off for latency numbers.
pub(in crate::cuda_minimax_h3) struct Stages<'a> {
    ctx: &'a CudaContext,
    enabled: bool,
    last: std::time::Instant,
}

impl<'a> Stages<'a> {
    fn new(ctx: &'a CudaContext) -> Self {
        Self {
            ctx,
            enabled: std::env::var_os("APXINF_H3_TIMINGS").is_some(),
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

impl ModelRunner {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: Rc<CudaContext>,
        dit: Rc<DitWeights>,
        tokenizer: Tokenizer,
        text: TextWeights,
        video: VideoDecoder,
        audio: AudioDecoder,
        gemm: GemmPolicy,
        attention: AttentionPolicy,
    ) -> Self {
        Self {
            ctx,
            dit,
            tokenizer,
            text,
            video,
            audio,
            gemm,
            attention,
            capture: Self::capture_enabled(),
            plan: None,
            conditioning: Vec::new(),
        }
    }

    /// Graph capture is the default; `APXINF_H3_EAGER` runs the same operators eagerly (debugging, parity checks).
    pub fn capture_enabled() -> bool {
        std::env::var_os("APXINF_H3_EAGER").is_none()
    }

    /// Everything that can be rejected without touching the device.
    pub(in crate::cuda_minimax_h3) fn resolve(
        &self,
        request: &GenerationRequest<'_>,
    ) -> Result<Plan> {
        if request.steps == 0 || request.steps > MAX_STEPS {
            return Err(invalid(format!(
                "steps must be 1..={MAX_STEPS}, got {}",
                request.steps
            )));
        }
        if request.width % CANVAS_MULTIPLE != 0
            || request.height % CANVAS_MULTIPLE != 0
            || !(MIN_SIDE..=MAX_SIDE).contains(&request.width)
            || !(MIN_SIDE..=MAX_SIDE).contains(&request.height)
        {
            return Err(invalid(format!(
                "width and height must be multiples of {CANVAS_MULTIPLE} in {MIN_SIDE}..={MAX_SIDE}, got {}x{}",
                request.width, request.height
            )));
        }
        if !(MIN_FRAMES..=MAX_FRAMES).contains(&request.frames) {
            return Err(invalid(format!(
                "frames must be {MIN_FRAMES}..={MAX_FRAMES}, got {}",
                request.frames
            )));
        }
        if request.prompt.trim().is_empty() {
            return Err(invalid("the prompt is empty"));
        }
        // raw text, no special tokens, no chat template (the reference conditioning node)
        let ids = self
            .tokenizer
            .encode_with_special_tokens(request.prompt, false)?;
        if ids.is_empty() || ids.len() > MAX_TEXT_TOKENS {
            return Err(invalid(format!(
                "the prompt must be 1..={MAX_TEXT_TOKENS} tokens, got {}",
                ids.len()
            )));
        }
        let frames = align_frame_count(request.frames);
        let layout = Layout::new(
            ids.len(),
            video_latent_steps(frames),
            request.height / VAE_RATIO,
            request.width / VAE_RATIO,
            audio_latent_steps(frames),
        )?;
        Ok(Plan {
            layout,
            frames,
            ids,
            steps: request.steps,
        })
    }

    /// Text conditioning: Qwen3 stack over streamed layers, then `condition_proj` and the token refiner.
    pub(in crate::cuda_minimax_h3) fn text_conditioning(
        &self,
        ids: &[u32],
        stages: &mut Stages<'_>,
    ) -> Result<Tensor> {
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

    /// The refined text rows of `ids`, from the cache when the prompt was seen recently.
    fn conditioning_for(&mut self, ids: &[u32], stages: &mut Stages<'_>) -> Result<Tensor> {
        if let Some(index) = self
            .conditioning
            .iter()
            .position(|(cached, _)| cached == ids)
        {
            let entry = self.conditioning.remove(index);
            let text = entry.1.clone();
            self.conditioning.insert(0, entry);
            stages.mark("text (cached)");
            return Ok(text);
        }
        let text = self.text_conditioning(ids, stages)?;
        self.conditioning.insert(0, (ids.to_vec(), text.clone()));
        self.conditioning.truncate(CONDITIONING_CACHE);
        Ok(text)
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

    /// Run the request to denoised latents.
    pub(in crate::cuda_minimax_h3) fn denoise(
        &mut self,
        request: &GenerationRequest<'_>,
    ) -> Result<Latents> {
        let ctx = self.ctx.clone();
        let mut stages = Stages::new(&ctx);
        let resolved = self.resolve(request)?;
        stages.mark("tokenize");
        let text = self.conditioning_for(&resolved.ids, &mut stages)?;
        let layout = resolved.layout;
        // comfy.sample.prepare_noise: the video tensor first, then the audio tensor, from one generator
        let mut generator = Generator::new(request.seed);
        let video_noise =
            generator.randn(LATENT_CHANNELS * layout.latent_t * layout.latent_h * layout.latent_w);
        let audio_noise = generator.randn(AUDIO_CHANNELS * AUDIO_STEREO * layout.audio_t);
        let video_rows = host::patchify_video(
            &video_noise,
            layout.latent_t,
            layout.latent_h,
            layout.latent_w,
        );
        let audio_rows = host::pack_audio(&audio_noise, layout.audio_t);
        stages.mark("noise");
        let steps = resolved.steps;
        let plan = self.plan_for(PlanKey { layout, steps }, &mut stages)?;
        plan.bind(&text, &video_rows, &audio_rows)?;
        drop(text);
        for step in 0..steps {
            plan.step(step)?;
            stages.mark(&format!("dit step {step}"));
        }
        let (video, audio) = plan.latents()?;
        stages.mark("latent download");
        Ok(Latents {
            video: host::unpatchify_video(
                &video,
                layout.latent_t,
                layout.latent_h,
                layout.latent_w,
            ),
            audio: host::unpack_audio(&audio, layout.audio_t),
            frames: resolved.frames,
            layout,
        })
    }

    /// The refined text rows of a prompt as a host copy (parity tests).
    #[cfg(test)]
    pub(in crate::cuda_minimax_h3) fn refined_text_rows(
        &mut self,
        prompt: &str,
    ) -> Result<Vec<f32>> {
        let ids = self.tokenizer.encode_with_special_tokens(prompt, false)?;
        let ctx = self.ctx.clone();
        let mut stages = Stages::new(&ctx);
        let text = self.conditioning_for(&ids, &mut stages)?;
        let mut bytes = vec![0u8; text.numel() * 2];
        crate::cuda_minimax_h3::backend::CudaBuffer::from_tensor(&text)
            .map_err(Error::Cuda)?
            .copy_to_host_on(&mut bytes, ctx.stream())
            .map_err(Error::Cuda)?;
        ctx.synchronize().map_err(Error::Cuda)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect())
    }
}

/// Float pixels in `[0, 1]`, `[3, frames, h, w]`, to row-major RGB8 `[frames, h, w, 3]`. ComfyUI converts with `(x * 255).clamp(0, 255).byte()`,
/// which truncates, so this does too: the frames equal the ones its `SaveVideo` would encode.
fn to_rgb8(video: &crate::cuda_minimax_h3::model::vae_host::Vid) -> Vec<u8> {
    let (t, h, w) = (video.t, video.h, video.w);
    let mut rgb = vec![0u8; t * h * w * 3];
    for c in 0..video.c {
        for f in 0..t {
            let plane = &video.data[(c * t + f) * h * w..][..h * w];
            for (i, v) in plane.iter().enumerate() {
                rgb[(f * h * w + i) * 3 + c] = (v * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }
    rgb
}

/// `[2, L]` channel-major samples to interleaved `L` frames of 2 channels.
fn interleave_stereo(samples: &[f32]) -> Vec<f32> {
    let l = samples.len() / 2;
    (0..l).flat_map(|i| [samples[i], samples[l + i]]).collect()
}

impl ModelRunner {
    /// Prompt to decoded frames and samples: the sampler, then both decoders.
    pub(super) fn run(&mut self, request: &VideoRequest<'_>) -> Result<GeneratedVideo> {
        let started = std::time::Instant::now();
        let latents = self.denoise(&GenerationRequest {
            prompt: request.prompt,
            width: request.width,
            height: request.height,
            frames: request.frames,
            seed: request.seed,
            steps: request.steps,
        })?;
        let layout = latents.layout;
        // The denoising plan holds its arena (about 160 KiB per packed row) and graph. Keeping it makes a repeat request of the same shape
        // about a second faster but leaves the decoder less memory (its tile batch must then be small); `APXINF_H3_DROP_PLAN` frees it first,
        // which lets `APXINF_H3_VAE_BATCH` go higher.
        if std::env::var_os("APXINF_H3_DROP_PLAN").is_some() {
            self.plan = None;
        }
        let video = self.video.decode(
            &latents.video,
            [layout.latent_t, layout.latent_h, layout.latent_w],
        )?;
        if video.t != latents.frames {
            return Err(invalid(format!(
                "the decoder produced {} frames for a clip of {}",
                video.t, latents.frames
            )));
        }
        let samples = self.audio.decode(&latents.audio, layout.audio_t)?;
        if std::env::var_os("APXINF_H3_TIMINGS").is_some() {
            eprintln!("  request total {:.3} s", started.elapsed().as_secs_f64());
        }
        Ok(GeneratedVideo {
            width: video.w,
            height: video.h,
            frames: video.t,
            frame_rate: FRAME_RATE,
            rgb: to_rgb8(&video),
            audio_sample_rate: AUDIO_SAMPLE_RATE as u32,
            audio_channels: AUDIO_STEREO,
            audio: interleave_stereo(&samples),
        })
    }
}

impl VideoRuntime for ModelRunner {
    fn generate(&mut self, request: &VideoRequest<'_>) -> Result<GeneratedVideo> {
        self.run(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_minimax_h3::model::vae_host::Vid;

    #[test]
    fn rgb_conversion_truncates_like_comfyui_and_orders_frames_rows_columns_channels() {
        let mut v = Vid::zeros(3, 2, 1, 2);
        // channel 0 frame 0: 0.0, 1.0; channel 1 frame 1 column 1: 0.5 -> 127.5 -> 127 (truncated, not rounded)
        v.data[1] = 1.0;
        v.data[7] = 0.5; // channel 1, frame 1, row 0, column 1 of [3, 2, 1, 2]
        let rgb = to_rgb8(&v);
        assert_eq!(rgb.len(), 2 * 2 * 3); // frames x columns x channels (one row)
        assert_eq!(rgb[3], 255); // frame 0, column 1, red
        assert_eq!(rgb[10], 127); // frame 1, column 1, green
        assert_eq!(rgb[0], 0);
        // out-of-range values clamp
        let mut v = Vid::zeros(3, 1, 1, 1);
        v.data[0] = 2.0;
        v.data[1] = -1.0;
        assert_eq!(to_rgb8(&v)[..2], [255, 0]);
    }

    #[test]
    fn stereo_is_interleaved_per_sample_frame() {
        assert_eq!(
            interleave_stereo(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]),
            vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]
        );
    }
}
