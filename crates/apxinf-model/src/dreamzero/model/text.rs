use super::super::{
    backend::{allocate, binary, linear, ops, Context, Policies},
    weights::{TextBlockWeights, TextWeights},
    DreamZeroConfig as C,
};
use apxinf_core::{Result, Tensor};
use ops::{PointwiseActivation as A, PointwiseSemantic as S};

/// Explicit reusable storage avoids retaining an activation arena for every
/// text layer. Allocate outside preparation/capture; all forward writes are
/// device-only and graph addresses remain stable.
pub(in crate::dreamzero) struct TextWorkspace {
    pub ids: Tensor,
    /// `[1,512]`: zero for valid keys, BF16 minimum for padding keys.
    pub mask: Tensor,
    pub output: Tensor,
    pub(super) hidden: Tensor,
    norm: Tensor,
    residual: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attention: Tensor,
    projection: Tensor,
    gate: Tensor,
    activated: Tensor,
    up: Tensor,
    product: Tensor,
    bias: Tensor,
}
impl TextWorkspace {
    pub fn new(ctx: &Context) -> Result<Self> {
        let hidden = || allocate(ctx, &[C::TEXT_LEN, C::TEXT_DIM]);
        let ffn = || allocate(ctx, &[C::TEXT_LEN, C::TEXT_FFN]);
        Ok(Self {
            ids: ctx.allocate_output(
                apxinf_core::Shape::new(vec![C::TEXT_LEN]),
                apxinf_core::DType::I32,
            )?,
            mask: allocate(ctx, &[1, C::TEXT_LEN])?,
            output: hidden()?,
            hidden: hidden()?,
            norm: hidden()?,
            residual: hidden()?,
            q: hidden()?,
            k: hidden()?,
            v: hidden()?,
            attention: hidden()?,
            projection: hidden()?,
            gate: ffn()?,
            activated: ffn()?,
            up: ffn()?,
            product: ffn()?,
            bias: allocate(ctx, &[C::TEXT_HEADS * C::TEXT_LEN, C::TEXT_LEN])?,
        })
    }
}

/// Exact BF16 UMT5 computation, including each FFN and attention rounding.
pub(in crate::dreamzero) fn encode(
    ctx: &Context,
    policies: &Policies,
    weights: &TextWeights,
    work: &mut TextWorkspace,
) -> Result<()> {
    ops::embedding_gather(ctx, &weights.embedding, &work.ids, &work.hidden)?;
    for block in &weights.blocks {
        encode_block(ctx, policies, block, work)?;
    }
    crate::dreamzero::backend::staged_rms(
        ctx,
        ops::NormArgs::rms(&work.hidden, &weights.norm, &mut work.output, 1e-6),
    )
}

pub(super) fn encode_block(
    ctx: &Context,
    policies: &Policies,
    block: &TextBlockWeights,
    work: &mut TextWorkspace,
) -> Result<()> {
    let n = C::TEXT_LEN;
    crate::dreamzero::backend::staged_rms(
        ctx,
        ops::NormArgs::rms(&work.hidden, &block.norm1, &mut work.norm, 1e-6),
    )?;
    linear(ctx, policies, &work.norm, &block.q, &mut work.q)?;
    linear(ctx, policies, &work.norm, &block.k, &mut work.k)?;
    linear(ctx, policies, &work.norm, &block.v, &mut work.v)?;
    binary(
        ctx,
        S::BroadcastAdd,
        &block.position.reshape(vec![C::TEXT_HEADS * n, n])?,
        &work.mask,
        &mut work.bias,
    )?;
    let shape = vec![1, n, C::TEXT_HEADS, C::TEXT_DIM / C::TEXT_HEADS];
    let q = work.q.reshape(shape.clone())?;
    let k = work.k.reshape(shape.clone())?;
    let v = work.v.reshape(shape.clone())?;
    let mut out = work.attention.reshape(shape)?;
    let bias = work.bias.reshape(vec![1, C::TEXT_HEADS, n, n])?;
    let mut attention = ops::AttentionArgs::new(&q, &k, &v, &mut out);
    attention.scale = 1.;
    attention.policy = policies.attention.clone();
    ops::rounded_additive_attention(
        ctx,
        ops::RoundedAttentionBiasArgs {
            attention,
            bias: &bias,
        },
    )?;
    linear(
        ctx,
        policies,
        &work.attention,
        &block.o,
        &mut work.projection,
    )?;
    binary(
        ctx,
        S::Add,
        &work.hidden,
        &work.projection,
        &mut work.residual,
    )?;
    crate::dreamzero::backend::staged_rms(
        ctx,
        ops::NormArgs::rms(&work.residual, &block.norm2, &mut work.norm, 1e-6),
    )?;
    linear(ctx, policies, &work.norm, &block.gate, &mut work.gate)?;
    let mut activation =
        ops::PointwiseArgs::new(S::BiasActivation, &work.gate, &mut work.activated);
    activation.activation = A::GeluRounded;
    ops::pointwise(ctx, activation)?;
    linear(ctx, policies, &work.norm, &block.up, &mut work.up)?;
    binary(
        ctx,
        S::Multiply,
        &work.up,
        &work.activated,
        &mut work.product,
    )?;
    linear(
        ctx,
        policies,
        &work.product,
        &block.down,
        &mut work.projection,
    )?;
    binary(
        ctx,
        S::Add,
        &work.residual,
        &work.projection,
        &mut work.hidden,
    )?;
    Ok(())
}
