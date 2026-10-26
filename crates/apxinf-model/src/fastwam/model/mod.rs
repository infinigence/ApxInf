//! Native computation only; request binding and capture live in model_runner.
pub(super) mod blocks;

use super::{
    backend::{self, ops, Context, DeviceBuffer},
    weights::{cast, host_tensor, text::TextEncoder, vae::Vae, Linear},
};
use apxinf_core::{DType, Result, Shape, Tensor};

use blocks::bf16::Expert;

pub(super) struct Inputs {
    pub rgb: DeviceBuffer,
    pub image_layout: ops::ImageMemoryLayout,
    pub ids: Tensor,
    pub padding: Tensor,
    pub mask: Tensor,
    pub state: Tensor,
    pub noise: Tensor,
}
pub(super) struct Model {
    pub(super) text: TextEncoder,
    pub(super) vae: Vae,
    pub(super) video: Expert,
    pub(super) action: Expert,
    pub(super) proprio: Linear,
    pub(super) rgb_table: Tensor,
    pub gemm: ops::GemmPolicy,
    pub attention: ops::AttentionPolicy,
}
impl Model {
    pub(super) fn workspace_bytes(&self) -> usize {
        let preprocessing = 224 * 448 * 3 * (4 + 2) + 129 * 4096 * 2 + 4096 * 2;
        let solver_outputs = 10 * (32 * 7 * 8 + 512) + 32 * 7 * 4;
        self.vae.workspace_bytes()
            + self.text.workspace_bytes()
            + self.video.workspace_bytes()
            + 10 * self.action.workspace_bytes()
            + preprocessing
            + solver_outputs
            + 16 * 256
    }

    pub(super) fn forward(&self, ctx: &Context, inputs: &Inputs, autotune: bool) -> Result<Tensor> {
        let mut gemm = self.gemm.clone();
        gemm.online_tune &= autotune;
        let mut attention = self.attention.clone();
        attention.online_tune &= autotune;
        let (image, text) = self.encode(ctx, inputs, &gemm, &attention)?;
        self.forward_encoded(ctx, inputs, &image, &text, &gemm, &attention)
    }

    fn encode(
        &self,
        ctx: &Context,
        inputs: &Inputs,
        gemm: &ops::GemmPolicy,
        attention: &ops::AttentionPolicy,
    ) -> Result<(Tensor, Tensor)> {
        let mut indices = ctx.allocate_output(Shape::new(vec![224 * 448 * 3]), DType::I32)?;
        ops::image_rgb_indices(
            ctx,
            &inputs.rgb,
            &mut indices,
            2,
            224,
            224,
            3,
            inputs.image_layout,
        )?;
        let rgb = ctx.allocate_output(Shape::new(vec![224 * 448 * 3, 1]), DType::BF16)?;
        ops::embedding_gather(ctx, &self.rgb_table, &indices, &rgb)?;
        let image = self
            .vae
            .forward(ctx, &rgb.reshape(vec![224 * 448, 3])?, gemm)?;
        let text = self.text.forward(
            ctx,
            &inputs.ids,
            &inputs.padding,
            &inputs.mask,
            gemm,
            attention,
        )?;
        Ok((image, text))
    }

    fn forward_encoded(
        &self,
        ctx: &Context,
        inputs: &Inputs,
        image: &Tensor,
        text: &Tensor,
        gemm: &ops::GemmPolicy,
        attention: &ops::AttentionPolicy,
    ) -> Result<Tensor> {
        let proprio = self.proprio.forward(ctx, &inputs.state, gemm)?;
        let context = ops::concat_rows(ctx, text, &proprio)?;
        let video_context = self.video.context(ctx, &context, gemm)?;
        let (_, cache) =
            self.video
                .forward(ctx, image, &video_context, None, 0, gemm, attention)?;
        let action_context = self.action.context(ctx, &context, gemm)?;
        // Keep iterative state, action input projection and head velocity in F32.
        // Schedule deltas retain BF16 rounding.
        let mut x = cast(ctx, &inputs.noise, DType::F32)?;
        for step in 0..10 {
            let (velocity, _) = self.action.forward(
                ctx,
                &x,
                &action_context,
                Some(&cache),
                step,
                gemm,
                attention,
            )?;
            let velocity = cast(ctx, &velocity, x.dtype())?;
            let mut next = ctx.allocate_output(x.shape().clone(), x.dtype())?;
            let mut args =
                ops::PointwiseArgs::new(ops::PointwiseSemantic::EulerUpdateRounded, &x, &mut next);
            args.secondary = Some(&velocity);
            args.dt = half::bf16::from_f32(
                blocks::bf16::sigma(step + 1) as f32 - blocks::bf16::sigma(step) as f32,
            )
            .to_f32();
            ops::pointwise(ctx, args)?;
            x = next;
        }
        cast(ctx, &x, DType::F32)
    }
}

use std::path::Path;

use super::weights::{dit::Expert as ExpertWeights, Bias, Store};

pub(super) fn build_model(
    ctx: &Context,
    path: &Path,
    text_path: &Path,
    vae_path: &Path,
    gemm: ops::GemmPolicy,
    attention: ops::AttentionPolicy,
) -> Result<Model> {
    let mut store = Store::load(path)?;
    let video = blocks::bf16::Expert::new(
        ctx,
        ExpertWeights::load(&mut store, ctx, true)?,
        &gemm,
        &[0.],
    )?;
    let action = blocks::bf16::Expert::new(
        ctx,
        ExpertWeights::load(&mut store, ctx, false)?,
        &gemm,
        &(0..10)
            .map(|step| (blocks::bf16::sigma(step) * 1000.) as f32)
            .collect::<Vec<_>>(),
    )?;
    let proprio = store.linear(ctx, "model.proprio_encoder", Bias::Required)?;
    proprio.validate("proprio encoder", 8, 4096)?;
    drop(store);
    let rgb = (0..256)
        .map(|v| {
            let unit = half::bf16::from_f32(v as f32 / 255.).to_f32();
            half::bf16::from_f32(half::bf16::from_f32(unit * 2.).to_f32() - 1.).to_f32()
        })
        .collect::<Vec<_>>();
    Ok(Model {
        video,
        action,
        proprio,
        text: TextEncoder::load(ctx, text_path)?,
        vae: Vae::load(ctx, vae_path, DType::BF16)?,
        rgb_table: backend::upload(ctx, &host_tensor(vec![256, 1], &rgb, DType::BF16)?)?,
        gemm,
        attention,
    })
}
