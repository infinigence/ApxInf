//! Joint-policy topology and the fixed rectified-flow / 2ab schedule.
mod blocks;
use super::{
    backend::{ops::GemmPolicy, AttentionPolicy, CudaBuffer, CudaContext},
    weights::Bf16Weights,
};
use apxinf_core::{Result, Tensor};
use blocks::Bf16Blocks;
use std::rc::Rc;

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
}
impl CosmosPolicyModel {
    pub fn new(ctx: Rc<CudaContext>, weights: Rc<Bf16Weights>, inputs: &Inputs) -> Result<Self> {
        let schedule = schedule();
        Ok(Self {
            blocks: Bf16Blocks::new(ctx, weights, inputs, &schedule)?,
            steps: schedule.len(),
        })
    }
    pub fn forward(&mut self, gemm: &GemmPolicy, attention: &AttentionPolicy) -> Result<()> {
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
