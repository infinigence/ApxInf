//! Model topology and fixed flow schedule. Precision details live in blocks.
mod blocks;
use super::backend::{ops, Context, DeviceBuffer};
use apxinf_core::{Result, Tensor};
use blocks::Bf16Blocks;
use std::path::Path;

pub(in crate::cuda_fastwam) struct Inputs {
    pub rgb: DeviceBuffer,
    pub image_layout: ops::ImageMemoryLayout,
    pub ids: Tensor,
    pub padding: Tensor,
    pub mask: Tensor,
    pub state: Tensor,
    pub noise: Tensor,
}

pub(in crate::cuda_fastwam) struct FastWamModel {
    blocks: Bf16Blocks,
}
impl FastWamModel {
    pub(in crate::cuda_fastwam) fn workspace_bytes(&self) -> usize {
        self.blocks.workspace_bytes()
    }
    pub(in crate::cuda_fastwam) fn forward(
        &self,
        ctx: &Context,
        inputs: &Inputs,
        autotune: bool,
    ) -> Result<Tensor> {
        let policies = self.blocks.policies(autotune);
        let image = self.blocks.encode_vision(ctx, inputs, &policies)?;
        let text = self.blocks.encode_text(ctx, inputs, &policies)?;
        let conditioning = self
            .blocks
            .condition(ctx, &image, &text, &inputs.state, &policies)?;
        let mut x = self.blocks.initial_state(ctx, &inputs.noise)?;
        for step in 0..10 {
            let dt = sigma(step + 1) as f32 - sigma(step) as f32;
            x = self
                .blocks
                .step(ctx, &x, &conditioning, step, dt, &policies)?;
        }
        self.blocks.finish(ctx, &x)
    }
}

fn sigma(step: usize) -> f64 {
    let u = 1. - step as f64 / 10.;
    5. * u / (1. + 4. * u)
}

pub(in crate::cuda_fastwam) fn build_model(
    ctx: &Context,
    path: &Path,
    text_path: &Path,
    vae_path: &Path,
    gemm: ops::GemmPolicy,
    attention: ops::AttentionPolicy,
) -> Result<FastWamModel> {
    let timesteps = (0..10)
        .map(|step| (sigma(step) * 1000.) as f32)
        .collect::<Vec<_>>();
    Ok(FastWamModel {
        blocks: Bf16Blocks::load(ctx, path, text_path, vae_path, gemm, attention, &timesteps)?,
    })
}
