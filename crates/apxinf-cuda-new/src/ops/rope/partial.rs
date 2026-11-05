//! Partial rotary embedding helpers.

use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::{rope as abi, status};
use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::CudaContext;

/// Rotary dimension implied by a partial rotary factor.
///
/// Rounded down to an even number because the rotation pairs elements.
pub fn rotary_dim(head_dim: usize, partial_rotary_factor: f32) -> usize {
    let raw = (head_dim as f32 * partial_rotary_factor) as usize;
    raw - raw % 2
}

/// Apply partial rotary embedding in place over `[tokens, heads, head_dim]`.
///
/// `positions` is `[tokens]` I32. For text-only input this is exactly the
/// model's mRoPE, because all three position sections share one id.
pub fn partial_rope(
    ctx: &CudaContext,
    data: &Tensor,
    positions: &Tensor,
    rotary_width: usize,
    theta: f32,
) -> Result<()> {
    let dims = data.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(invalid("RoPE expects [tokens, heads, head_dim]"));
    }
    let (tokens, heads, head_dim) = (dims[0], dims[1], dims[2]);
    if rotary_width == 0 || rotary_width > head_dim || rotary_width % 2 != 0 {
        return Err(invalid("rotary width must be even and at most head_dim"));
    }
    let data_buffer = tensor_storage(ctx, data, DType::BF16, &dims)?;
    let position_buffer = tensor_storage(ctx, positions, DType::I32, &[tokens])?;
    unsafe {
        status::check(abi::apxinf_rope_partial(
            data_buffer.ptr(),
            position_buffer.ptr(),
            tokens as i64,
            heads as i64,
            head_dim as i64,
            rotary_width as i64,
            theta,
            ctx.stream().handle(),
        ))
    }
}
