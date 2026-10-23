//! Caller-owned cache buffer operators.

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

/// Append `[append_len, kv_heads, head_dim]` rows into a caller-owned
/// head-major `[kv_heads, max_seq_len, head_dim]` cache.
#[allow(clippy::too_many_arguments)]
pub fn append(
    ctx: &CudaContext,
    cache: &crate::CudaBuffer,
    new_data: &Tensor,
    kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    seq_len: usize,
    append_len: usize,
) -> Result<()> {
    use crate::ops::validation::{check_cuda, checked_bytes, require_buffers};
    use crate::ffi::abi::llm_decode as abi;
    use apxinf_core::{DType, Error};

    if append_len == 0
        || seq_len
            .checked_add(append_len)
            .map_or(true, |end| end > max_seq_len)
    {
        return Err(Error::Other(
            "KV append range exceeds cache capacity".into(),
        ));
    }
    let dtype = new_data.dtype();
    if !matches!(dtype, DType::F32 | DType::BF16) {
        return Err(Error::Other(format!("KV append does not support {dtype}")));
    }
    let input = crate::CudaBuffer::from_tensor(new_data).map_err(Error::Cuda)?;
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
                &input,
                checked_bytes(dtype, &[append_len, kv_heads, head_dim], "KV append")?,
            ),
        ],
    )?;
    let status = unsafe {
        match dtype {
            DType::F32 => abi::apxinf_llmdec_kv_cache_append_f32(
                cache.ptr(),
                input.ptr(),
                kv_heads as u32,
                head_dim as u32,
                max_seq_len as u32,
                seq_len as u32,
                append_len as u32,
                ctx.stream().handle(),
            ),
            DType::BF16 => abi::apxinf_llmdec_kv_cache_append_bf16(
                cache.ptr(),
                input.ptr(),
                kv_heads as u32,
                head_dim as u32,
                max_seq_len as u32,
                seq_len as u32,
                append_len as u32,
                ctx.stream().handle(),
            ),
            _ => unreachable!(),
        }
    };
    check_cuda(status)
}

/// Append one token to a caller-owned head-major
/// `[kv_heads, max_seq_len, head_dim]` KV cache using a device-resident
/// position. Ports the legacy caller-owned `append_at` contract over the
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
    use crate::ops::validation::{check_cuda, checked_bytes, require_address, require_buffers};
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
