//! Full-attention primitives: partial RoPE, per-head q/k normalization, and
//! the sigmoid output gate.
//!
//! See `native/kernels/custom/attn_ops.h` for two scope limits that matter:
//! the mRoPE collapse is only valid for text-only input, and the rotary
//! pairing convention still needs confirming against a reference engine.

use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::{attn as abi, status};
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
        status::check(abi::apxinf_attn_partial_rope(
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

/// Per-head RMSNorm in place over `[rows, head_dim]`, for q_norm and k_norm.
pub fn head_rms_norm(
    ctx: &CudaContext,
    data: &Tensor,
    weight: &Tensor,
    epsilon: f32,
) -> Result<()> {
    let dims = data.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(invalid("head RMSNorm expects [rows, head_dim]"));
    }
    let data_buffer = tensor_storage(ctx, data, DType::BF16, &dims)?;
    let weight_buffer = tensor_storage(ctx, weight, DType::BF16, &[dims[1]])?;
    unsafe {
        status::check(abi::apxinf_attn_head_rms_norm(
            data_buffer.ptr(),
            weight_buffer.ptr(),
            dims[0] as i64,
            dims[1] as i64,
            epsilon,
            ctx.stream().handle(),
        ))
    }
}

/// Split q_proj's `[tokens, heads, 2*head_dim]` output into query and gate.
pub fn split_query_and_gate(
    ctx: &CudaContext,
    fused: &Tensor,
    query: &Tensor,
    gate: &Tensor,
) -> Result<()> {
    let dims = query.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(invalid("query must be [tokens, heads, head_dim]"));
    }
    let (tokens, heads, head_dim) = (dims[0], dims[1], dims[2]);
    let fused_buffer =
        tensor_storage(ctx, fused, DType::BF16, &[tokens, heads, 2 * head_dim])?;
    let query_buffer = tensor_storage(ctx, query, DType::BF16, &dims)?;
    let gate_buffer = tensor_storage(ctx, gate, DType::BF16, &dims)?;
    unsafe {
        status::check(abi::apxinf_attn_split_query_and_gate(
            fused_buffer.ptr(),
            query_buffer.ptr(),
            gate_buffer.ptr(),
            tokens as i64,
            heads as i64,
            head_dim as i64,
            ctx.stream().handle(),
        ))
    }
}

/// `data *= sigmoid(gate)`, elementwise and in place.
///
/// Sigmoid, not silu, even though `config.json` carries
/// `output_gate_type: "swish"`. That key is never read by the reference
/// implementation; `Qwen3_5Attention.forward` applies a bare
/// `torch.sigmoid(gate)`. The silu-shaped gate in this model is the GDN one
/// (`gdn_gated_norm`), which is separate and unaffected.
pub fn apply_output_gate(ctx: &CudaContext, data: &Tensor, gate: &Tensor) -> Result<()> {
    let dims = data.shape().dims().to_vec();
    if dims != gate.shape().dims() {
        return Err(invalid("output gate requires matching shapes"));
    }
    let data_buffer = tensor_storage(ctx, data, DType::BF16, &dims)?;
    let gate_buffer = tensor_storage(ctx, gate, DType::BF16, &dims)?;
    unsafe {
        status::check(abi::apxinf_attn_apply_output_gate(
            data_buffer.ptr(),
            gate_buffer.ptr(),
            dims.iter().product::<usize>() as i64,
            ctx.stream().handle(),
        ))
    }
}
