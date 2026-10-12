//! Legacy `kernels::cache` names over cuda-new cache operators.

use apxinf_core::{Result, Tensor};

use crate::{ops, CudaContext};

/// `reserve_prefix_bf16`: allocate a `[total_rows, cols]` cache seeded with
/// `prefix` and zero-padded to capacity.
pub fn reserve_prefix_bf16(
    ctx: &CudaContext,
    prefix: &Tensor,
    total_rows: usize,
) -> Result<Tensor> {
    ops::reserve_prefix(ctx, prefix, total_rows)
}

// ── Decode fast-path `_into` surface (caller-owned buffers, graph-safe) ───

/// Append one token to a caller-owned head-major
/// `[kv_heads, max_seq_len, head_dim]` KV cache using a device-resident
/// position. Ports the legacy `kernels::cache::append_at` contract over the
/// llm_decode adapter.
#[allow(clippy::too_many_arguments)]
pub fn append_at(
    ctx: &CudaContext,
    dtype: apxinf_core::DType,
    cache: &crate::CudaBuffer,
    input: &crate::CudaBuffer,
    kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    position: crate::CudaDeviceAddress,
) -> Result<()> {
    use super::contracts::{check_cuda, checked_bytes, require_address, require_buffers};
    use crate::ffi::abi::llm_decode as abi;
    use apxinf_core::DType;
    require_buffers(
        ctx,
        "KV append",
        &[
            (
                "cache",
                cache,
                checked_bytes(dtype, &[kv_heads, max_seq_len, head_dim], "KV append")?,
            ),
            (
                "input",
                input,
                checked_bytes(dtype, &[kv_heads, head_dim], "KV append")?,
            ),
        ],
    )?;
    require_address(ctx, "KV append", "position", position, 4)?;
    let status = unsafe {
        match dtype {
            DType::F32 => abi::apxinf_llmdec_kv_cache_append_decode_f32(
                cache.ptr(),
                input.ptr(),
                kv_heads as u32,
                head_dim as u32,
                max_seq_len as u32,
                position.ptr(),
                ctx.stream().handle(),
            ),
            DType::BF16 => abi::apxinf_llmdec_kv_cache_append_decode_bf16(
                cache.ptr(),
                input.ptr(),
                kv_heads as u32,
                head_dim as u32,
                max_seq_len as u32,
                position.ptr(),
                ctx.stream().handle(),
            ),
            dtype => {
                return Err(apxinf_core::Error::Other(format!(
                    "decode KV append does not support {dtype}"
                )))
            }
        }
    };
    check_cuda(status)
}
