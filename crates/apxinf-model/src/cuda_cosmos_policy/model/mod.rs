//! Joint-policy topology and the fixed rectified-flow / 2ab schedule.
mod blocks;
use super::{
    backend::{ops::GemmPolicy, AttentionPolicy, CudaBuffer, CudaContext},
    weights::Bf16Weights,
};
use apxinf_core::{Error, Result, Tensor};
use blocks::Bf16Blocks;
use std::rc::Rc;

/// The single source of truth for the pinned LIBERO direct profile geometry.
/// `config.rs` validates the prepared assets against these values, the runner
/// builds its public `VlaContract` from them, and the blocks allocate from
/// them, so the numbers are not restated per module.
pub(super) const NUM_VIEWS: usize = 2;
pub(super) const IMAGE_SIZE: usize = 224;
pub(super) const STATE_DIM: usize = 9;
pub(super) const DENOISER_CALLS: usize = 5;
pub(super) const ACTION_HORIZON: usize = 16;
pub(super) const ACTION_DIM: usize = 7;
pub(super) const LATENT_FRAMES: usize = 9;
pub(super) const LATENT_SIDE: usize = 28;
pub(super) const LATENT_CHANNELS: usize = 16;
pub(super) const TEXT_TOKENS: usize = 512;
pub(super) const TEXT_WIDTH: usize = 1024;
pub(super) const NUM_TASKS: usize = 40;

/// `[1, 16, 9, 28, 28]`: the complete joint latent the sampler consumes.
pub(super) const INITIAL_LATENT_SHAPE: [usize; 5] =
    [1, LATENT_CHANNELS, LATENT_FRAMES, LATENT_SIDE, LATENT_SIDE];
/// `[16, 7]`: the public action chunk, independent of the latent geometry.
pub(super) const ACTION_SHAPE: [usize; 2] = [ACTION_HORIZON, ACTION_DIM];

/// Canonical device inputs, allocated and rebound by model_runner.
pub(super) struct Inputs {
    pub rgb: CudaBuffer,
    pub proprio: Tensor,
    pub task: Tensor,
    pub noise: Tensor,
}
struct SolverStep {
    sigma: f64,
    recurrence: Option<[f64; 4]>,
}
fn schedule() -> Vec<SolverStep> {
    let sigmas: [f64; 5] = std::array::from_fn(|i| {
        (80f64.powf(1. / 7.) + i as f64 / 4. * (4f64.powf(1. / 7.) - 80f64.powf(1. / 7.))).powi(7)
    });
    sigmas
        .iter()
        .enumerate()
        .map(|(i, &sigma)| {
            let recurrence = if i == 4 {
                None
            } else {
                let (s, t) = (sigma, sigmas[i + 1]);
                Some(if i == 0 {
                    [t / s, (s - t) / s, 0., 1.]
                } else {
                    let (s, t, m) = (-s.ln(), -t.ln(), -sigmas[i - 1].ln());
                    let dt = t - s;
                    let c2 = (m - s) / dt;
                    let phi1 = (-dt).exp_m1() / (-dt);
                    let phi2 = (phi1 - 1.) / (-dt);
                    [(-dt).exp(), phi1 - 1. / c2 * phi2, 1. / c2 * phi2, dt]
                })
            };
            SolverStep { sigma, recurrence }
        })
        .collect()
}
pub(super) struct CosmosPolicyModel {
    blocks: Bf16Blocks,
    steps: usize,
    constants_prepared: bool,
}
impl CosmosPolicyModel {
    pub fn new(
        ctx: Rc<CudaContext>,
        weights: Rc<Bf16Weights>,
        inputs: &Inputs,
        convolution: &crate::cuda_cosmos_policy::backend::Conv3dPolicy,
    ) -> Result<Self> {
        let schedule = schedule();
        Ok(Self {
            blocks: Bf16Blocks::new(ctx, weights, inputs, &schedule, convolution)?,
            steps: schedule.len(),
            constants_prepared: false,
        })
    }
    /// Prepare checkpoint-fixed blank-frame features before recording the
    /// request execution sequence. Inputs and weights are owned by this model.
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
                "Cosmos fixed features have not been prepared".into(),
            ));
        }
        self.blocks.encode_observation(gemm, attention)?;
        self.blocks.initial_state()?;
        for step in 0..self.steps {
            self.blocks.denoise_step(step, gemm, attention)?;
        }
        self.blocks.finish()
    }
    pub fn actions(&self) -> &Tensor {
        self.blocks.actions()
    }
    #[cfg(test)]
    pub fn joint(&self) -> &Tensor {
        self.blocks.joint()
    }
    pub fn workspace_bytes(&self) -> usize {
        self.blocks.workspace_bytes()
    }
}
