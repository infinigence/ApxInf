//! Prepared denoising for one request shape: persistent buffers, the per-step tables and, unless disabled, one captured
//! CUDA Graph of a whole DiT forward. Shape = (text rows, canvas); nothing here depends on the prompt's content, so a plan is
//! reused by every request of the same shape and a request only refreshes the text rows and the noise.
use crate::cuda_minimax_h3_image::backend::ops::{self, GemmPolicy};
use crate::cuda_minimax_h3_image::backend::{
    AttentionPolicy, CudaBuffer, CudaContext, ExecutionSession, PreparedPhase,
};
use crate::cuda_minimax_h3_image::config::*;
use crate::cuda_minimax_h3_image::model::blocks::{alloc_like, Int8Blocks, VaeBlocks};
use crate::cuda_minimax_h3_image::model::vae_host::{
    assemble, latent_tile_tokens, to_rgb8, unpatchify_pixels, Chw, TilePlan, TILE_TOKENS,
};
use crate::cuda_minimax_h3_image::weights::DitWeights;
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use std::rc::Rc;

/// Identity of a plan: everything the captured graph's shapes depend on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct PlanKey {
    pub layout: Layout,
    pub canvas: Canvas,
}

pub(super) struct PreparedDenoise {
    // The graph is declared first so it drops first: its nodes address the session arena and the buffers below.
    phase: Option<PreparedPhase>,
    pub key: PlanKey,
    blocks: Int8Blocks,
    ctx: Rc<CudaContext>,
    latent: Tensor,
    text: Tensor,
    velocity: Tensor,
    gemm: GemmPolicy,
    attention: AttentionPolicy,
}

fn persistent(ctx: &CudaContext, shape: &[usize], dtype: DType) -> Result<Tensor> {
    let bytes = shape.iter().product::<usize>() * dtype.size_in_bytes();
    CudaBuffer::alloc(bytes, ctx.device_id())
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape.to_vec()), dtype)
        .map_err(Error::Cuda)
}

impl PreparedDenoise {
    /// Build the tables and buffers for `key` and, when `capture` is set, prepare and capture one DiT forward.
    pub fn new(
        ctx: Rc<CudaContext>,
        dit: Rc<DitWeights>,
        key: PlanKey,
        capture: bool,
        gemm: GemmPolicy,
        attention: AttentionPolicy,
    ) -> Result<Self> {
        let blocks = Int8Blocks::new(
            ctx.clone(),
            dit,
            key.layout,
            key.canvas,
            &PDD_SIGMAS[..PDD_STEPS],
        )?;
        let rows = key.layout.image_rows;
        let latent = persistent(&ctx, &[rows, PATCH_DIM], DType::F32)?;
        let text = persistent(&ctx, &[key.layout.text_rows, HIDDEN], DType::BF16)?;
        let velocity = persistent(&ctx, &[rows, PATCH_DIM], DType::F32)?;
        let mut plan = Self {
            phase: None,
            key,
            blocks,
            ctx,
            latent,
            text,
            velocity,
            gemm,
            attention,
        };
        if capture {
            plan.capture()?;
        }
        Ok(plan)
    }

    fn capture(&mut self) -> Result<()> {
        // preparation executes the forward once, so every input it reads must hold finite data
        self.blocks.select_step(0)?;
        for tensor in [&self.latent, &self.text] {
            CudaBuffer::from_tensor(tensor)
                .map_err(Error::Cuda)?
                .zero_async(self.ctx.stream())
                .map_err(Error::Cuda)?;
        }
        let session = ExecutionSession::with_capacity(
            Int8Blocks::arena_bytes(self.key.layout),
            self.ctx.device_id(),
        )?;
        let (blocks, latent, text, velocity, gemm, attention) = (
            &self.blocks,
            &self.latent,
            &self.text,
            &self.velocity,
            &self.gemm,
            &self.attention,
        );
        let phase = PreparedPhase::prepare_and_capture(&self.ctx, session, || {
            blocks.forward_into(latent, text, velocity, gemm, attention)
        })?;
        self.ctx.synchronize().map_err(Error::Cuda)?;
        self.phase = Some(phase);
        Ok(())
    }

    pub fn is_captured(&self) -> bool {
        self.phase.is_some()
    }

    /// Bytes of the session arena the preparation traversal used, and its capacity (graph mode only).
    pub fn arena(&self) -> Option<(usize, usize)> {
        self.phase
            .as_ref()
            .map(|p| (p.workspace_used(), p.workspace_capacity()))
    }

    /// Install a request's inputs: the refined text rows and the starting latent (FP32 `[image_rows, PATCH_DIM]`).
    pub fn bind(&self, text: &Tensor, noise_rows: &[f32]) -> Result<()> {
        ops::copy_tensor(&self.ctx, text, &self.text)?;
        let bytes: Vec<u8> = noise_rows.iter().flat_map(|v| v.to_le_bytes()).collect();
        CudaBuffer::from_tensor(&self.latent)
            .map_err(Error::Cuda)?
            .copy_from_host_on(&bytes, self.ctx.stream())
            .map_err(Error::Cuda)
    }

    /// One denoising step: select the step's tables, run the DiT (graph replay or eager), then the FP32 Euler update
    /// `latent += (sigma[step] - sigma[step + 1]) * output`, where `output` is the BF16-rounded head output.
    pub fn step(&self, step: usize) -> Result<()> {
        let ctx = &*self.ctx;
        self.blocks.select_step(step)?;
        match &self.phase {
            Some(phase) => phase.replay()?,
            None => self.blocks.forward_into(
                &self.latent,
                &self.text,
                &self.velocity,
                &self.gemm,
                &self.attention,
            )?,
        }
        let dt = PDD_SIGMAS[step + 1] - PDD_SIGMAS[step];
        let narrow = alloc_like(ctx, &self.velocity, DType::BF16)?;
        ops::quantization::cast(ctx, &self.velocity, &narrow)?;
        let widened = alloc_like(ctx, &self.velocity, DType::F32)?;
        ops::quantization::cast(ctx, &narrow, &widened)?;
        let delta = alloc_like(ctx, &self.velocity, DType::F32)?;
        ops::elementwise_scale(ctx, &widened, &delta, -dt)?;
        let next = alloc_like(ctx, &self.velocity, DType::F32)?;
        ops::elementwise_add(ctx, &self.latent, &delta, &next)?;
        ops::copy_tensor(ctx, &next, &self.latent)
    }

    pub fn latent(&self) -> &Tensor {
        &self.latent
    }
}

/// The VAE decoder at its fixed batch of tiles: persistent input and output buffers and, unless disabled, one captured CUDA
/// Graph of the batch. A short last batch is padded with zero latents, so one graph serves every canvas.
pub(in crate::cuda_minimax_h3_image) struct PreparedVae {
    // Declared first so the graph drops before the buffers and arena its nodes address.
    phase: Option<PreparedPhase>,
    blocks: Rc<VaeBlocks>,
    ctx: Rc<CudaContext>,
    /// FP32 normalised latent tokens of one batch and the BF16 decoder rows it produces.
    input: Tensor,
    output: Tensor,
    gemm: GemmPolicy,
    attention: AttentionPolicy,
}

fn zeros(ctx: &CudaContext, shape: Vec<usize>, dtype: DType) -> Result<Tensor> {
    let bytes = shape.iter().product::<usize>() * dtype.size_in_bytes();
    CudaBuffer::alloc_zeros(bytes, ctx.device_id())
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape), dtype)
        .map_err(Error::Cuda)
}

impl PreparedVae {
    pub fn new(
        ctx: Rc<CudaContext>,
        blocks: Rc<VaeBlocks>,
        capture: bool,
        gemm: GemmPolicy,
        attention: AttentionPolicy,
    ) -> Result<Self> {
        let tiles = blocks.max_tiles();
        let input = zeros(
            &ctx,
            vec![tiles * VAE_TILE_LATENT * VAE_TILE_LATENT, LATENT_CHANNELS],
            DType::F32,
        )?;
        let output = zeros(&ctx, vec![tiles * TILE_TOKENS, VAE_OUT_DIM], DType::BF16)?;
        let mut prepared = Self {
            phase: None,
            blocks,
            ctx,
            input,
            output,
            gemm,
            attention,
        };
        if capture {
            let session = ExecutionSession::with_capacity(
                VaeBlocks::arena_bytes(tiles),
                prepared.ctx.device_id(),
            )?;
            let phase = PreparedPhase::prepare_and_capture(&prepared.ctx, session, || {
                prepared.decode_batch()
            })?;
            prepared.ctx.synchronize().map_err(Error::Cuda)?;
            prepared.phase = Some(phase);
        }
        Ok(prepared)
    }

    /// One batch of `max_tiles` tiles from the persistent input into the persistent output.
    fn decode_batch(&self) -> Result<()> {
        let rows = self.blocks.decode(
            &self.input,
            self.blocks.max_tiles(),
            &self.gemm,
            &self.attention,
        )?;
        ops::copy_tensor(&self.ctx, &rows, &self.output)
    }

    fn run_batch(&self) -> Result<()> {
        match &self.phase {
            Some(phase) => phase.replay(),
            None => self.decode_batch(),
        }
    }

    /// Decode a normalised latent `[LATENT_CHANNELS, h, w]` (host, FP32) into row-major RGB8 `[h * 16, w * 16, 3]`.
    /// Tiles of one row are decoded in batches of at most `max_tiles`; blending runs on the host (see doc/minimax-h3-image.md).
    pub fn decode_image(&self, latent: &Chw) -> Result<Vec<u8>> {
        let ctx = &*self.ctx;
        let tiles_per_batch = self.blocks.max_tiles();
        let (height, width) = (latent.h * VAE_RATIO, latent.w * VAE_RATIO);
        let pixels = VAE_TILE_LATENT * VAE_TILE_LATENT;
        let ys = TilePlan::new(height);
        // a tile is never larger than the image; images below one tile are rejected by `Canvas`
        let timings = std::env::var_os("APXINF_H3_IMAGE_TIMINGS").is_some();
        let (mut device_s, mut unpatch_s) = (0.0f64, 0.0f64);
        let started = std::time::Instant::now();
        let canvas = assemble(height, width, |row, xs| {
            let (tile_h, y0) = (ys.lens[row] / VAE_RATIO, ys.starts[row] / VAE_RATIO);
            let mut tiles = Vec::with_capacity(xs.count());
            for first in (0..xs.count()).step_by(tiles_per_batch) {
                let batch = (xs.count() - first).min(tiles_per_batch);
                let mut tokens = Vec::with_capacity(batch * pixels * LATENT_CHANNELS);
                for j in first..first + batch {
                    latent_tile_tokens(
                        latent,
                        y0,
                        xs.starts[j] / VAE_RATIO,
                        tile_h,
                        xs.lens[j] / VAE_RATIO,
                        &mut tokens,
                    );
                }
                if tile_h != VAE_TILE_LATENT
                    || xs.lens[first..first + batch]
                        .iter()
                        .any(|l| *l != VAE_TILE_PIXELS)
                {
                    return Err(Error::Other(
                        "VAE tiles must be full 256-pixel tiles".into(),
                    ));
                }
                let t = std::time::Instant::now();
                // a short last batch is padded with zero latents; tiles never attend to each other, so the real tiles are unchanged
                tokens.resize(tiles_per_batch * pixels * LATENT_CHANNELS, 0.0);
                let bytes_in: Vec<u8> = tokens.iter().flat_map(|v| v.to_le_bytes()).collect();
                CudaBuffer::from_tensor(&self.input)
                    .map_err(Error::Cuda)?
                    .copy_from_host_on(&bytes_in, ctx.stream())
                    .map_err(Error::Cuda)?;
                self.run_batch()?;
                let mut bytes = vec![0u8; self.output.size_in_bytes()];
                CudaBuffer::from_tensor(&self.output)
                    .map_err(Error::Cuda)?
                    .copy_to_host_on(&mut bytes, ctx.stream())
                    .map_err(Error::Cuda)?;
                device_s += t.elapsed().as_secs_f64();
                let t = std::time::Instant::now();
                let values: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect();
                for t in 0..batch {
                    let rows = &values[t * TILE_TOKENS * VAE_OUT_DIM..][..pixels * VAE_OUT_DIM];
                    tiles.push(unpatchify_pixels(rows, VAE_TILE_LATENT, VAE_TILE_LATENT));
                }
                unpatch_s += t.elapsed().as_secs_f64();
            }
            Ok(tiles)
        })?;
        let assembled = started.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        let rgb = to_rgb8(&canvas);
        if timings {
            eprintln!(
                "  vae: device {device_s:.2} s, bf16->f32 + unpatchify {unpatch_s:.2} s, blend + copy {:.2} s, rgb8 {:.2} s",
                assembled - device_s - unpatch_s,
                t.elapsed().as_secs_f64()
            );
        }
        Ok(rgb)
    }
}
