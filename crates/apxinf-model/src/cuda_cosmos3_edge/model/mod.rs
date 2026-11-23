//! Cosmos3-Edge joint-policy topology and the fixed UniPC flow-matching schedule.
//!
//! Each request: VAE-encode the observation into a condition latent; run the UND
//! tower once as a per-layer K/V prefill; run the GEN diffusion tower for four
//! UniPC steps (video + action tokens denoised jointly); read out the action.
mod blocks;
use super::{
    backend::{ops::GemmPolicy, AttentionPolicy, CudaBuffer, CudaContext},
    weights::Bf16Weights,
};
use apxinf_core::{Error, Result, Tensor};
use blocks::Bf16Blocks;
use std::rc::Rc;

/// Canonical device inputs, allocated and rebound by model_runner.
pub(super) struct Inputs {
    /// Cached chat-templated token embeddings `[1, text_tokens, 2048]`.
    pub text: Tensor,
    /// Initial joint noise `[1, 48*5*34*46 + 16*64]` (FP32): the CTHW video
    /// latent followed by the packed action latent.
    pub noise: Tensor,
    /// VAE pixel input `[1, 272, 368, 12]` (BF16, THWC), the patched observation
    /// singleton — the 16-frame rollout chunk is not encoded.
    pub pixel: Tensor,
    pub rgb: CudaBuffer,
}

/// One UniPC flow-matching step. `sigma` is the current noise level.
struct SolverStep {
    timestep: f32,
    sigma: f64,
}

/// `flow_shift = 5.0`, `num_train_timesteps = 1000`. The step count is a
/// fixed DROID profile uses four steps. `sigmas =
/// linspace(999/1000, 0, steps + 1)[:-1]`, each flow-shifted; model timesteps
/// = floor(sigma * 1000).
fn schedule(steps: usize) -> Vec<SolverStep> {
    const SHIFT: f64 = 5.0;
    const TRAIN_TIMESTEPS: f64 = 1000.0;
    let sigma_max = (TRAIN_TIMESTEPS - 1.0) / TRAIN_TIMESTEPS;
    let mut sigmas = Vec::with_capacity(steps);
    for i in 0..steps {
        let raw = sigma_max + (i as f64 / steps as f64) * (0.0 - sigma_max);
        sigmas.push(SHIFT * raw / (1.0 + (SHIFT - 1.0) * raw));
    }
    (0..steps)
        .map(|i| SolverStep {
            timestep: (sigmas[i] * TRAIN_TIMESTEPS).floor() as f32,
            sigma: sigmas[i],
        })
        .collect()
}

pub(super) struct Cosmos3EdgeModel {
    blocks: Bf16Blocks,
    steps: usize,
    constants_prepared: bool,
}

impl Cosmos3EdgeModel {
    /// `convolution` selects the VAE Conv3d recipes; it is consumed at
    /// construction because every convolution is prepared up front.
    /// `num_inference_steps` is the UniPC step count from the checkpoint
    /// config (4 for the DROID real-time profile, 30 for LIBERO accuracy).
    pub fn new(
        ctx: Rc<CudaContext>,
        weights: Rc<Bf16Weights>,
        inputs: &Inputs,
        convolution: &super::backend::Conv3dPolicy,
        num_inference_steps: usize,
    ) -> Result<Self> {
        let schedule = schedule(num_inference_steps);
        Ok(Self {
            blocks: Bf16Blocks::new(ctx, weights, inputs, &schedule, convolution)?,
            steps: schedule.len(),
            constants_prepared: false,
        })
    }
    pub fn prepare_constants(
        &mut self,
        gemm: &GemmPolicy,
        attention: &AttentionPolicy,
    ) -> Result<()> {
        self.blocks.prepare_constants(gemm, attention)?;
        self.constants_prepared = true;
        Ok(())
    }
    pub fn forward(&mut self, gemm: &GemmPolicy, attention: &AttentionPolicy) -> Result<()> {
        if !self.constants_prepared {
            return Err(Error::Other(
                "Cosmos3-Edge constants have not been prepared".into(),
            ));
        }
        self.blocks.encode_observation(gemm, attention)?;
        self.blocks.und_prefill(gemm, attention)?;
        self.blocks.initial_state()?;
        for step in 0..self.steps {
            self.blocks.denoise_step(step, gemm, attention)?;
        }
        self.blocks
            .finish()
            .map_err(|e| Error::Other(format!("finish: {e}")))
    }
    pub fn actions(&self) -> &Tensor {
        self.blocks.actions()
    }
    pub fn workspace_bytes(&self) -> usize {
        self.blocks.workspace_bytes()
    }
}
