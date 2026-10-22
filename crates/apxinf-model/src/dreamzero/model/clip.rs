use super::super::{
    backend::{allocate, binary, linear, ops, transfers, Context, Policies},
    weights::clip::{ClipBlock, ClipWeights, Linear, Norm, DIM, HEADS, TOKENS},
};
use crate::dreamzero::backend;
use apxinf_core::{Error, Result, Shape, Tensor};

pub(in crate::dreamzero) struct ClipWorkspace {
    pub patches: Tensor,
    padded_patches: Tensor,
    pub output: Tensor,
    pub(super) projected: Tensor,
    embedded: Tensor,
    position: Tensor,
    pub(super) hidden: Tensor,
    pub(super) normalized: Tensor,
    pub(super) residual: Tensor,
    qkv: Tensor,
    qkv_split: Tensor,
    pub(super) q: Tensor,
    pub(super) k: Tensor,
    pub(super) v: Tensor,
    pub(super) attention: Tensor,
    pub(super) projected_attention: Tensor,
    pub(super) up: Tensor,
    pub(super) activated: Tensor,
}
impl ClipWorkspace {
    pub fn new(ctx: &Context) -> Result<Self> {
        let hidden = || allocate(ctx, &[TOKENS, DIM]);
        let qkv_split = allocate(ctx, &[3, TOKENS, DIM])?;
        let flat = qkv_split.reshape(vec![3 * TOKENS, DIM])?;
        let q = rows(&flat, 0, TOKENS)?;
        let k = rows(&flat, TOKENS, TOKENS)?;
        let v = rows(&flat, 2 * TOKENS, TOKENS)?;
        Ok(Self {
            patches: allocate(ctx, &[256, 14 * 14 * 3])?,
            padded_patches: allocate(ctx, &[256, 14 * 14 * 16])?,
            output: hidden()?,
            projected: allocate(ctx, &[256, DIM])?,
            embedded: hidden()?,
            position: hidden()?,
            hidden: hidden()?,
            normalized: hidden()?,
            residual: hidden()?,
            qkv: allocate(ctx, &[TOKENS, 3 * DIM])?,
            qkv_split,
            q,
            k,
            v,
            attention: hidden()?,
            projected_attention: hidden()?,
            up: allocate(ctx, &[TOKENS, 4 * DIM])?,
            activated: allocate(ctx, &[TOKENS, 4 * DIM])?,
        })
    }
}
fn norm(ctx: &Context, x: &Tensor, w: &Norm, y: &mut Tensor) -> Result<()> {
    ops::layer_norm(ctx, ops::LayerNormArgs::new(x, &w.weight, &w.bias, y, 1e-5))
}
fn projection(ctx: &Context, p: &Policies, x: &Tensor, w: &Linear, y: &mut Tensor) -> Result<()> {
    let mut gemm = ops::GemmArgs::new(x, &w.weight, y);
    gemm.policy = p.gemm.clone();
    // Preserve the fixed BF16 reference's Lt reduction/workspace profile.
    gemm.policy.workspace_limit = gemm.policy.workspace_limit.min(1 << 20);
    gemm.b_transposed = true;
    ops::gemm_bias_single_round(
        ctx,
        ops::GemmBiasArgs {
            gemm,
            bias: &w.bias,
        },
    )
}
fn rows(x: &Tensor, offset: usize, count: usize) -> Result<Tensor> {
    let buffer = backend::CudaBuffer::from_tensor(x).map_err(Error::Cuda)?;
    let view = buffer
        .view(offset * DIM * 2, count * DIM * 2)
        .map_err(Error::Cuda)?;
    view.as_tensor(Shape::new(vec![count, DIM]), x.dtype())
        .map_err(Error::Cuda)
}
/// CLIP visual encoder from device-resident normalized 14×14 RGB patches.
/// Image resizing/normalization populates `patches` before this computation.
pub(in crate::dreamzero) fn encode_patches(
    ctx: &Context,
    p: &Policies,
    w: &ClipWeights,
    s: &mut ClipWorkspace,
) -> Result<()> {
    encode_prefix(ctx, p, w, s)?;
    for b in &w.blocks {
        encode_block(ctx, p, b, s)?;
    }
    transfers::copy_cuda_to_cuda(ctx, &s.hidden, &s.output)
}
pub(super) fn encode_prefix(
    ctx: &Context,
    p: &Policies,
    w: &ClipWeights,
    s: &mut ClipWorkspace,
) -> Result<()> {
    // Match the reference convolution's channel-aligned tensor-core reduction.
    // Zero extension changes the reduction layout, not the convolution values.
    ops::unfold3d(
        ctx,
        &s.patches.reshape(vec![256, 1, 14, 14, 3])?,
        ops::Unfold3dGeometry {
            kernel: [1, 14, 14],
            stride: [1, 1, 1],
            dilation: [1, 1, 1],
            padding_before: [0; 3],
            padding_after: [0; 3],
        },
        0,
        &mut s.padded_patches,
    )?;
    linear(ctx, p, &s.padded_patches, &w.patch, &mut s.projected)?;
    transfers::copy_cuda_to_cuda(ctx, &w.class, &rows(&s.embedded, 0, 1)?)?;
    transfers::copy_cuda_to_cuda(ctx, &s.projected, &rows(&s.embedded, 1, 256)?)?;
    binary(
        ctx,
        ops::PointwiseSemantic::Add,
        &s.embedded,
        &w.position,
        &mut s.position,
    )?;
    norm(ctx, &s.position, &w.pre_norm, &mut s.hidden)?;
    Ok(())
}
pub(super) fn encode_block(
    ctx: &Context,
    p: &Policies,
    b: &ClipBlock,
    s: &mut ClipWorkspace,
) -> Result<()> {
    norm(ctx, &s.hidden, &b.norm1, &mut s.normalized)?;
    projection(ctx, p, &s.normalized, &b.qkv, &mut s.qkv)?;
    ops::permute(
        ctx,
        &s.qkv.reshape(vec![TOKENS, 3, DIM])?,
        &[1, 0, 2],
        &mut s.qkv_split,
    )?;
    let shape = vec![1, TOKENS, HEADS, DIM / HEADS];
    let q = s.q.reshape(shape.clone())?;
    let k = s.k.reshape(shape.clone())?;
    let v = s.v.reshape(shape.clone())?;
    let mut out = s.attention.reshape(shape)?;
    let mut args = ops::AttentionArgs::new(&q, &k, &v, &mut out);
    args.policy = p.attention.clone();
    // Keep all 31 prepared layers on the existing linear-memory FA2 path.
    args.policy.workspace_limit = 2 << 20;
    ops::attention(ctx, args)?;
    projection(ctx, p, &s.attention, &b.o, &mut s.projected_attention)?;
    binary(
        ctx,
        ops::PointwiseSemantic::Add,
        &s.hidden,
        &s.projected_attention,
        &mut s.residual,
    )?;
    norm(ctx, &s.residual, &b.norm2, &mut s.normalized)?;
    projection(ctx, p, &s.normalized, &b.up, &mut s.up)?;
    let mut activation = ops::PointwiseArgs::new(
        ops::PointwiseSemantic::BiasActivation,
        &s.up,
        &mut s.activated,
    );
    activation.activation = ops::PointwiseActivation::GeluExact;
    ops::pointwise(ctx, activation)?;
    projection(ctx, p, &s.activated, &b.down, &mut s.projected_attention)?;
    binary(
        ctx,
        ops::PointwiseSemantic::Add,
        &s.residual,
        &s.projected_attention,
        &mut s.hidden,
    )?;
    Ok(())
}
