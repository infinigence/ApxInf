//! Legacy `kernels::rope` names over the cuda-new packed-QKV RoPE operator.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

pub use super::attention::QkvTensors;
use crate::{ops, CudaContext};

fn split_impl(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
    cache: Option<(&Tensor, &Tensor, usize)>,
) -> Result<QkvTensors> {
    let dims = qkv.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other("packed QKV must be rank 2".into()));
    }
    let tokens = dims[0];
    let q_shape = Shape::new(vec![tokens, q_heads, head_dim]);
    let mut q = ctx.allocate_output(q_shape, DType::BF16)?;
    let (mut k, mut v, kv_output_offset) = match cache {
        // Rotate K and copy V directly into caller-owned cache rows.
        Some((k_cache, v_cache, offset)) => (k_cache.clone(), v_cache.clone(), offset),
        None => {
            let kv_shape = Shape::new(vec![tokens, kv_heads, head_dim]);
            (
                ctx.allocate_output(kv_shape.clone(), DType::BF16)?,
                ctx.allocate_output(kv_shape, DType::BF16)?,
                0,
            )
        }
    };
    ops::rope(
        ctx,
        ops::RopeArgs {
            semantic: ops::RopeSemantic::SplitQkvRope,
            qkv,
            bias,
            q: &mut q,
            k: &mut k,
            v: &mut v,
            q_heads,
            kv_heads,
            head_dim,
            theta,
            position_offset,
            kv_output_offset,
        },
    )?;
    Ok(QkvTensors { q, k, v })
}

/// `split_qkv_apply_bf16`: split a packed GQA projection and rotate Q/K into
/// fresh per-call buffers.
#[allow(clippy::too_many_arguments)]
pub fn split_qkv_apply_bf16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
) -> Result<QkvTensors> {
    split_impl(
        ctx,
        qkv,
        bias,
        q_heads,
        kv_heads,
        head_dim,
        theta,
        position_offset,
        None,
    )
}

/// `apply_q_write_kv_bf16`: split and rotate, appending K/V into caller-owned
/// caches at `output_offset`, returning only the rotated Q.
#[allow(clippy::too_many_arguments)]
pub fn apply_q_write_kv_bf16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
    k_cache: &Tensor,
    v_cache: &Tensor,
    output_offset: usize,
) -> Result<Tensor> {
    Ok(split_impl(
        ctx,
        qkv,
        bias,
        q_heads,
        kv_heads,
        head_dim,
        theta,
        position_offset,
        Some((k_cache, v_cache, output_offset)),
    )?
    .q)
}
