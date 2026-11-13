//! Caller-owned cache buffer operators.

use apxinf_core::{Result, Tensor};

use crate::CudaContext;

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
    use crate::ffi::abi::cache as abi;
    use crate::ops::validation::{check_cuda, checked_bytes, require_buffers};
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
            DType::F32 => abi::apxinf_cache_fixed_decode_append_f32(
                cache.ptr(),
                input.ptr(),
                kv_heads as u32,
                head_dim as u32,
                max_seq_len as u32,
                seq_len as u32,
                append_len as u32,
                ctx.stream().handle(),
            ),
            DType::BF16 => abi::apxinf_cache_fixed_decode_append_bf16(
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
/// fixed decode adapter.
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
    use crate::ffi::abi::cache as abi;
    use crate::ops::validation::{check_cuda, checked_bytes, require_address, require_buffers};
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
            DType::F32 => abi::apxinf_cache_fixed_decode_append_decode_f32(
                cache.ptr(),
                input.ptr(),
                kv_heads as u32,
                head_dim as u32,
                max_seq_len as u32,
                position.ptr(),
                ctx.stream().handle(),
            ),
            DType::BF16 => abi::apxinf_cache_fixed_decode_append_decode_bf16(
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

/// Copy one input token into the cache row named by `meta[1]`.
///
/// A row index at or past the cache capacity writes nothing, which mirrors the
/// clamp the decode attention applies to the same metadata: the write position
/// is device-side state, so enforcing it on the host would require a sync.
/// F16 and BF16 both store one 16-bit word per element, so the copy is
/// dtype-agnostic; the contract restricts callers to those two.
pub fn write_kv_cache_token(
    ctx: &CudaContext,
    input: &Tensor,
    cache: &Tensor,
    meta: &crate::ops::KvCacheDecodeMeta,
) -> Result<()> {
    use crate::ffi::abi::cache as abi;
    use crate::ops::validation::check_cuda;
    use apxinf_core::{DType, Error};

    let dims = cache.shape().dims().to_vec();
    let input_dims = input.shape().dims().to_vec();
    if dims.len() != 4 || dims[0] != 1 || dims[2] == 0 || dims[3] == 0 {
        return Err(Error::Other(
            "dynamic KV token write requires [1, capacity, heads, head_dim] cache".into(),
        ));
    }
    if input_dims != [1, 1, dims[2], dims[3]] {
        return Err(Error::Other(
            "dynamic KV token write requires a single token row".into(),
        ));
    }
    let dtype = cache.dtype();
    if !matches!(dtype, DType::F16 | DType::BF16) || input.dtype() != dtype {
        return Err(Error::Other(
            "dynamic KV token write supports matching F16 or BF16 tensors".into(),
        ));
    }
    if meta.capacity() != dims[1] || meta.device() != ctx.device_id() {
        return Err(Error::Other("dynamic KV token write meta geometry mismatch".into()));
    }
    let capacity = dims[1];
    let row_elements = dims[2] * dims[3];
    let input_buffer = crate::CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let cache_buffer = crate::CudaBuffer::from_tensor(cache).map_err(Error::Cuda)?;
    let input_bytes = row_elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| Error::Other("dynamic KV token write size overflow".into()))?;
    let cache_bytes = capacity
        .checked_mul(row_elements)
        .and_then(|v| v.checked_mul(dtype.size_in_bytes()))
        .ok_or_else(|| Error::Other("dynamic KV token write size overflow".into()))?;
    if input_buffer.len() < input_bytes || cache_buffer.len() < cache_bytes {
        return Err(Error::Other("dynamic KV token write storage is too small".into()));
    }
    meta.retain();
    crate::workspace::retain_buffer(&input_buffer);
    crate::workspace::retain_buffer(&cache_buffer);
    crate::workspace::validate_capture_target(ctx.device_id(), ctx.stream().handle() as usize)?;
    let status = unsafe {
        abi::apxinf_cache_write_token(
            input_buffer.ptr(),
            cache_buffer.ptr(),
            meta.device_address(),
            capacity as i64,
            row_elements as i64,
            ctx.stream().handle(),
        )
    };
    check_cuda(status)
}
