//! Text conditioning: `condition_proj` and the two-block BF16 token refiner. The result does not depend on the
//! timestep, so a request computes it once and every step reads the same rows.
use super::util::{alloc, dense_linear};
use crate::cuda_minimax_h3_image::backend::ops::{self, GemmPolicy, NormArgs};
use crate::cuda_minimax_h3_image::backend::{AttentionPolicy, CudaContext};
use crate::cuda_minimax_h3_image::config::*;
use crate::cuda_minimax_h3_image::weights::{DitWeights, RefinerBlock};
use apxinf_core::{DType, Result, Tensor};

/// Dense BF16 attention over `[rows, INNER]` projections, returned as `[rows, INNER]`.
pub(super) fn attend(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    policy: &AttentionPolicy,
) -> Result<Tensor> {
    let rows = q.shape().dims()[0];
    let shape = vec![1, rows, HEADS, HEAD_DIM];
    let (q, k, v) = (
        q.reshape(shape.clone())?,
        k.reshape(shape.clone())?,
        v.reshape(shape.clone())?,
    );
    let mut out = alloc(ctx, &shape, DType::BF16)?;
    let mut args = ops::AttentionArgs::new(&q, &k, &v, &mut out);
    args.policy = policy.clone();
    ops::attention(ctx, args)?;
    out.reshape(vec![rows, INNER])
}

/// Causal grouped-query attention over `[rows, 64 * 128]` queries and `[rows, 8 * 128]` keys and values, returned as
/// `[rows, 64 * 128]`.
pub(super) fn attend_gqa(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    rows: usize,
    policy: &AttentionPolicy,
) -> Result<Tensor> {
    let q = q.reshape(vec![1, rows, TE_HEADS, TE_HEAD_DIM])?;
    let (k, v) = (
        k.reshape(vec![1, rows, TE_KV_HEADS, TE_HEAD_DIM])?,
        v.reshape(vec![1, rows, TE_KV_HEADS, TE_HEAD_DIM])?,
    );
    let mut out = alloc(ctx, &[1, rows, TE_HEADS, TE_HEAD_DIM], DType::BF16)?;
    let mut args = ops::AttentionArgs::new(&q, &k, &v, &mut out).causal();
    args.policy = policy.clone();
    ops::attention(ctx, args)?;
    out.reshape(vec![rows, TE_Q])
}

pub(super) fn rms(ctx: &CudaContext, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let out = alloc(ctx, input.shape().dims(), DType::BF16)?;
    ops::norm::norm(ctx, NormArgs::rms(input, weight, &out, NORM_EPS))?;
    Ok(out)
}

fn refiner_block(
    ctx: &CudaContext,
    block: &RefinerBlock,
    x: &Tensor,
    gemm: &GemmPolicy,
    attention: &AttentionPolicy,
) -> Result<()> {
    let rows = x.shape().dims()[0];
    let h = rms(ctx, x, &block.norm1)?;
    let q = dense_linear(ctx, &h, &block.q, gemm)?;
    let k = dense_linear(ctx, &h, &block.k, gemm)?;
    let v = dense_linear(ctx, &h, &block.v, gemm)?;
    // per-head RMSNorm of q and k; the refiner has no rotary embedding
    let per_head = vec![rows * HEADS, HEAD_DIM];
    let q = rms(ctx, &q.reshape(per_head.clone())?, &block.q_norm)?.reshape(vec![rows, INNER])?;
    let k = rms(ctx, &k.reshape(per_head)?, &block.k_norm)?.reshape(vec![rows, INNER])?;
    let attended = attend(ctx, &q, &k, &v, attention)?;
    let branch = dense_linear(ctx, &attended, &block.out, gemm)?;
    ops::add_into(ctx, &branch, x)?;

    let h = rms(ctx, x, &block.norm2)?;
    let gate = dense_linear(ctx, &h, &block.gate, gemm)?;
    let up = dense_linear(ctx, &h, &block.up, gemm)?;
    let activated = ops::pointwise::silu_mul(ctx, &gate, &up)?;
    let branch = dense_linear(ctx, &activated, &block.fc2, gemm)?;
    ops::add_into(ctx, &branch, x)
}

/// `[tokens, TEXT_DIM]` BF16 text states to `[tokens, HIDDEN]` BF16 refined rows.
pub(in crate::cuda_minimax_h3_image) fn refine_text(
    ctx: &CudaContext,
    weights: &DitWeights,
    text_states: &Tensor,
    gemm: &GemmPolicy,
    attention: &AttentionPolicy,
) -> Result<Tensor> {
    let x = dense_linear(ctx, text_states, &weights.condition, gemm)?;
    for block in &weights.refiner {
        refiner_block(ctx, block, &x, gemm, attention)?;
    }
    rms(ctx, &x, &weights.refiner_final_norm)
}
