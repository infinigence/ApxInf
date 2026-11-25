//! Prepared denoising for one request shape: persistent latent and conditioning buffers, the per-step tables and, unless disabled, one
//! captured CUDA Graph of a whole DiT forward. Shape = (text rows, video latent grid, audio steps, step count); nothing here depends on the
//! prompt's content or the seed, so a plan is reused by every request of the same shape and a request only refreshes the text rows and
//! the two noise tensors.
use crate::cuda_minimax_h3::backend::ops::{self, GemmPolicy};
use crate::cuda_minimax_h3::backend::{
    AttentionPolicy, CudaBuffer, CudaContext, ExecutionSession, PreparedPhase,
};
use crate::cuda_minimax_h3::config::*;
use crate::cuda_minimax_h3::model::blocks::{alloc_like, Int8Blocks, StepTimes};
use crate::cuda_minimax_h3::model::host;
use crate::cuda_minimax_h3::model::layout::Layout;
use crate::cuda_minimax_h3::weights::DitWeights;
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use std::rc::Rc;

/// Identity of a plan: everything the captured graph's shapes and the per-step tables depend on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::cuda_minimax_h3) struct PlanKey {
    pub layout: Layout,
    pub steps: usize,
}

/// Video and audio sigmas of a request, `steps + 1` values each, ending in 0. The audio stream follows the video grid through the two
/// flow shifts (ComfyUI `time_shift_sigma`); its terminal value is exactly 0.
pub(in crate::cuda_minimax_h3) fn schedule(steps: usize) -> (Vec<f32>, Vec<f32>) {
    let video = host::simple_sigmas(steps, SIGMA_SHIFT_VIDEO);
    let mut audio: Vec<f32> = video[..steps]
        .iter()
        .map(|s| StepTimes { sigma_video: *s }.sigma_audio())
        .collect();
    audio.push(0.0);
    (video, audio)
}

pub(in crate::cuda_minimax_h3) struct PreparedDenoise {
    // The graph is declared first so it drops first: its nodes address the session arena and the buffers below.
    phase: Option<PreparedPhase>,
    pub key: PlanKey,
    blocks: Int8Blocks,
    ctx: Rc<CudaContext>,
    /// FP32 latents in packed row layout: video `[video_rows, PATCH_DIM]`, audio `[audio_rows, AUDIO_CHANNELS]`.
    video: Tensor,
    audio: Tensor,
    /// The DiT's FP32 head outputs of the current step (the model returns their negation).
    video_head: Tensor,
    audio_head: Tensor,
    /// Refined BF16 text rows `[text_rows, HIDDEN]`.
    text: Tensor,
    sigma_video: Vec<f32>,
    sigma_audio: Vec<f32>,
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

fn upload_f32(ctx: &CudaContext, tensor: &Tensor, values: &[f32]) -> Result<()> {
    if values.len() != tensor.numel() {
        return Err(Error::Other(format!(
            "expected {} values, got {}",
            tensor.numel(),
            values.len()
        )));
    }
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .copy_from_host_on(&bytes, ctx.stream())
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
        if key.steps == 0 {
            return Err(Error::Other(
                "MiniMax-H3: a request needs at least one denoising step".into(),
            ));
        }
        let (sigma_video, sigma_audio) = schedule(key.steps);
        let times: Vec<StepTimes> = sigma_video[..key.steps]
            .iter()
            .map(|s| StepTimes { sigma_video: *s })
            .collect();
        let blocks = Int8Blocks::new(ctx.clone(), dit, key.layout, &times)?;
        let layout = key.layout;
        let video = persistent(&ctx, &[layout.video_rows(), PATCH_DIM], DType::F32)?;
        let audio = persistent(&ctx, &[layout.audio_rows(), AUDIO_CHANNELS], DType::F32)?;
        let video_head = persistent(&ctx, &[layout.video_rows(), PATCH_DIM], DType::F32)?;
        let audio_head = persistent(&ctx, &[layout.audio_rows(), AUDIO_CHANNELS], DType::F32)?;
        let text = persistent(&ctx, &[layout.text_rows, HIDDEN], DType::BF16)?;
        let mut plan = Self {
            phase: None,
            key,
            blocks,
            ctx,
            video,
            audio,
            video_head,
            audio_head,
            text,
            sigma_video,
            sigma_audio,
            gemm,
            attention,
        };
        if capture {
            plan.capture()?;
        }
        Ok(plan)
    }

    fn forward(&self) -> Result<()> {
        self.blocks.forward_into(
            &self.video,
            &self.audio,
            &self.text,
            &self.video_head,
            &self.audio_head,
            &self.gemm,
            &self.attention,
        )
    }

    fn capture(&mut self) -> Result<()> {
        // preparation executes the forward once, so every input it reads must hold finite data
        self.blocks.select_step(0)?;
        for tensor in [&self.video, &self.audio, &self.text] {
            CudaBuffer::from_tensor(tensor)
                .map_err(Error::Cuda)?
                .zero_async(self.ctx.stream())
                .map_err(Error::Cuda)?;
        }
        let session = ExecutionSession::with_capacity(
            Int8Blocks::arena_bytes(self.key.layout),
            self.ctx.device_id(),
        )?;
        let phase = PreparedPhase::prepare_and_capture(&self.ctx, session, || self.forward())?;
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

    /// Install a request's inputs: the refined text rows (BF16 `[text_rows, HIDDEN]`) and the two starting latents in packed row layout.
    pub fn bind(&self, text: &Tensor, video_rows: &[f32], audio_rows: &[f32]) -> Result<()> {
        ops::copy_tensor(&self.ctx, text, &self.text)?;
        upload_f32(&self.ctx, &self.video, video_rows)?;
        upload_f32(&self.ctx, &self.audio, audio_rows)
    }

    /// `latent += (sigma_next - sigma) * (-head)` for one stream, where the model's returned velocity is the negated head output
    /// rounded to BF16 (the dtype ComfyUI's DiT returns).
    fn euler(&self, latent: &Tensor, head: &Tensor, dt: f32) -> Result<()> {
        let ctx = &*self.ctx;
        let narrow = alloc_like(ctx, head, DType::BF16)?;
        ops::quantization::cast(ctx, head, &narrow)?;
        let widened = alloc_like(ctx, head, DType::F32)?;
        ops::quantization::cast(ctx, &narrow, &widened)?;
        let delta = alloc_like(ctx, head, DType::F32)?;
        ops::elementwise_scale(ctx, &widened, &delta, -dt)?;
        let next = alloc_like(ctx, latent, DType::F32)?;
        ops::elementwise_add(ctx, latent, &delta, &next)?;
        ops::copy_tensor(ctx, &next, latent)
    }

    /// One denoising step: select the step's tables, run the DiT (graph replay or eager), then the FP32 Euler update of both streams.
    pub fn step(&self, step: usize) -> Result<()> {
        self.blocks.select_step(step)?;
        match &self.phase {
            Some(phase) => phase.replay()?,
            None => self.forward()?,
        }
        self.euler(
            &self.video,
            &self.video_head,
            self.sigma_video[step + 1] - self.sigma_video[step],
        )?;
        self.euler(
            &self.audio,
            &self.audio_head,
            self.sigma_audio[step + 1] - self.sigma_audio[step],
        )
    }

    /// The denoised packed latents, downloaded: video `[video_rows, PATCH_DIM]` and audio `[audio_rows, AUDIO_CHANNELS]` rows.
    pub fn latents(&self) -> Result<(Vec<f32>, Vec<f32>)> {
        Ok((
            download_f32(&self.ctx, &self.video)?,
            download_f32(&self.ctx, &self.audio)?,
        ))
    }
}
