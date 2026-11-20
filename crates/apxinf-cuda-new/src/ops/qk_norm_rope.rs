//! Per-head query/key normalization and the fused norm+rope decode stage.
//!
//! These two launches cover what the product `norm` and `rope` families
//! cannot express without an extra intermediate round trip: a weighted
//! per-head RMSNorm fused with partial rope, and a KV row append whose row
//! index is read on the device so a captured decode graph survives position
//! advance.
//!
//! Note the norm here is plain-weight RMSNorm (Qwen3 semantics: the weight
//! multiplies directly). The zero-centered `(1 + weight)` form is a different
//! numerics mode of the `norm` family (`with_weight_offset`) and does not
//! apply to q/k norm.

use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::{qk_norm_rope as abi, status};
use crate::ops::rope::contracts::{invalid, tensor_storage};
use crate::CudaContext;

/// Fused per-head RMSNorm (weighted, bf16-rounded) + partial rope, one launch.
///
/// Bit-identical to `norm` (`NormKind::Rms`) followed by `rope::partial_rope`
/// on the same buffers: same 256-thread reduction tree, same `(1 + weight)`
/// multiplier, and the rope stage re-reads the norm output through memory
/// after the bf16 round trip.
///
/// `input` is `[tokens * heads, head_dim]` (the `q`/`k` projection output
/// reshaped); `output` must match and is written in place (rope mutates it, as
/// `partial_rope` does). `positions` is `[tokens]` I32.
pub fn qk_norm_rope(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    output: &mut Tensor,
    positions: &Tensor,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
    eps: f32,
) -> Result<()> {
    let dims = input.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(invalid("qk_norm_rope expects a rank-2 [rows, head_dim] input"));
    }
    let (rows, cols) = (dims[0], dims[1]);
    if heads == 0 || rows % heads != 0 || cols != head_dim {
        return Err(invalid(
            "qk_norm_rope: rows must be tokens*heads and cols must be head_dim",
        ));
    }
    let tokens = rows / heads;
    if rotary_dim == 0 || rotary_dim > head_dim || rotary_dim % 2 != 0 {
        return Err(invalid("rotary width must be even and at most head_dim"));
    }
    if output.shape().dims() != dims.as_slice() {
        return Err(invalid("qk_norm_rope output must match input shape"));
    }
    let input_buffer = tensor_storage(ctx, input, DType::BF16, rows * cols)?;
    let output_buffer = tensor_storage(ctx, output, DType::BF16, rows * cols)?;
    let weight_buffer = tensor_storage(ctx, weight, DType::BF16, cols)?;
    let position_buffer = tensor_storage(ctx, positions, DType::I32, tokens)?;
    unsafe {
        status::check(abi::apxinf_qk_norm_rope_fused(
            input_buffer.ptr(),
            weight_buffer.ptr(),
            position_buffer.ptr(),
            output_buffer.ptr(),
            tokens as i64,
            heads as i64,
            head_dim as i64,
            rotary_dim as i64,
            theta,
            eps,
            ctx.stream().handle(),
        ))
    }
}

/// Append one BF16 row of `src` (any rank totalling `cols` elements) into the
/// row-major `cache` `[capacity, cols]` at the row index read from `position`
/// (I32 `[1]`).
///
/// The row index is read on the device, so the launch is CUDA-Graph-safe for
/// decode loops whose position advances every step: the captured graph stays
/// valid without recapture.
pub fn cache_append_row(
    ctx: &CudaContext,
    src: &Tensor,
    cache: &Tensor,
    position: &Tensor,
) -> Result<()> {
    let cache_dims = cache.shape().dims().to_vec();
    if cache_dims.len() != 2 || cache_dims.contains(&0) {
        return Err(invalid("cache append expects a rank-2 [capacity, cols] cache"));
    }
    let cols = cache_dims[1];
    let src_dims = src.shape().dims().to_vec();
    let src_elems: usize = src_dims.iter().product();
    if src_dims.is_empty() || src_elems != cols {
        return Err(invalid(format!(
            "cache append source holds {src_elems} elements, cache row is {cols}"
        )));
    }
    let src_buffer = tensor_storage(ctx, src, DType::BF16, src_elems)?;
    let cache_buffer = tensor_storage(ctx, cache, DType::BF16, cache_dims[0] * cols)?;
    let position_buffer = tensor_storage(ctx, position, DType::I32, 1)?;
    unsafe {
        status::check(abi::apxinf_qk_norm_rope_cache_append_row(
            src_buffer.ptr(),
            cache_buffer.ptr(),
            position_buffer.ptr(),
            cols as i64,
            cache_dims[0] as i64,
            ctx.stream().handle(),
        ))
    }
}
