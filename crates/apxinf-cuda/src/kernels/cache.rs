//! KV-cache storage operator contracts.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use super::contracts::{
    bf16_output, check_cuda, checked_bytes, f16_output, gpu_ptr, make_gpu_tensor, matrix_shape,
    matrix_tensor, require_address, require_buffers, unsupported_dtype,
};
use crate::buffer::{CudaBuffer, CudaDeviceAddress};
use crate::context::CudaContext;
use crate::ffi;

/// Append one token to caller-owned KV cache using a device position.
#[allow(clippy::too_many_arguments)]
pub fn append_at(
    ctx: &CudaContext,
    dtype: DType,
    cache: &CudaBuffer,
    input: &CudaBuffer,
    kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    position: CudaDeviceAddress,
) -> Result<()> {
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
            DType::F32 => ffi::apxinf_kv_cache_append_decode_f32(
                cache.ptr(),
                input.ptr(),
                kv_heads as u32,
                head_dim as u32,
                max_seq_len as u32,
                position.ptr(),
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_kv_cache_append_decode_bf16(
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

/// Append K/V rows to a KV cache buffer. Dispatches on new_data.dtype().
pub fn append(
    ctx: &CudaContext,
    cache_buf: &CudaBuffer,
    new_data: &Tensor,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    seq_len: usize,
    append_len: usize,
) -> Result<()> {
    unsafe {
        let res = match new_data.dtype() {
            DType::F32 => ffi::apxinf_kv_cache_append_f32(
                cache_buf.ptr(),
                gpu_ptr(new_data)?,
                n_kv_heads as u32,
                head_dim as u32,
                max_seq_len as u32,
                seq_len as u32,
                append_len as u32,
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_kv_cache_append_bf16(
                cache_buf.ptr(),
                gpu_ptr(new_data)?,
                n_kv_heads as u32,
                head_dim as u32,
                max_seq_len as u32,
                seq_len as u32,
                append_len as u32,
                ctx.stream().handle(),
            ),
            dtype => return unsupported_dtype("kv_cache_append", dtype),
        };
        ffi::check_cuda(res).map_err(Error::Cuda)?;
    }
    Ok(())
}

/// Quantize contiguous BF16 cache rows to E4M3 with one FP32 scale per row.
pub fn quantize_bf16_e4m3_rows(
    ctx: &CudaContext,
    input: &CudaBuffer,
    output: &CudaBuffer,
    scales: &CudaBuffer,
    rows: usize,
    head_dim: usize,
) -> Result<()> {
    let elements = rows
        .checked_mul(head_dim)
        .ok_or_else(|| Error::Other("E4M3 KV quantization size overflow".into()))?;
    require_buffers(
        ctx,
        "E4M3 KV quantization",
        &[
            (
                "input",
                input,
                elements
                    .checked_mul(DType::BF16.size_in_bytes())
                    .ok_or_else(|| Error::Other("BF16 KV byte size overflow".into()))?,
            ),
            ("output", output, elements),
            (
                "scales",
                scales,
                rows.checked_mul(std::mem::size_of::<f32>())
                    .ok_or_else(|| Error::Other("E4M3 KV scale size overflow".into()))?,
            ),
        ],
    )?;
    if rows == 0 || head_dim == 0 || rows > u32::MAX as usize || head_dim > u32::MAX as usize {
        return Err(Error::Other(
            "E4M3 KV quantization launch contract mismatch".into(),
        ));
    }
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qwen35_kv_quantize_bf16_e4m3(
            input.ptr(),
            output.ptr(),
            scales.ptr(),
            rows as u32,
            head_dim as u32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

/// Quantize and append 1..N token-major BF16 rows into an E4M3 KV cache.
#[allow(clippy::too_many_arguments)]
pub fn append_e4m3(
    ctx: &CudaContext,
    cache: &CudaBuffer,
    scales: &CudaBuffer,
    input: &Tensor,
    kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    seq_len: usize,
    append_len: usize,
) -> Result<()> {
    if input.dtype() != DType::BF16 || input.device() != apxinf_core::Device::Cuda(ctx.device_id())
    {
        return Err(Error::Other(
            "E4M3 KV append input must be CUDA BF16".into(),
        ));
    }
    let cache_bytes = kv_heads
        .checked_mul(max_seq_len)
        .and_then(|value| value.checked_mul(head_dim))
        .ok_or_else(|| Error::Other("E4M3 KV cache size overflow".into()))?;
    let scale_bytes = kv_heads
        .checked_mul(max_seq_len)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| Error::Other("E4M3 KV scale size overflow".into()))?;
    let input_bytes = append_len
        .checked_mul(kv_heads)
        .and_then(|value| value.checked_mul(head_dim))
        .and_then(|value| value.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("E4M3 KV input size overflow".into()))?;
    let input_buffer = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    require_buffers(
        ctx,
        "E4M3 KV append",
        &[
            ("cache", cache, cache_bytes),
            ("scales", scales, scale_bytes),
            ("input", &input_buffer, input_bytes),
        ],
    )?;
    if kv_heads == 0
        || head_dim == 0
        || max_seq_len == 0
        || append_len == 0
        || seq_len > max_seq_len
        || append_len > max_seq_len - seq_len
        || kv_heads > u32::MAX as usize
        || head_dim > u32::MAX as usize
        || max_seq_len > u32::MAX as usize
        || seq_len > u32::MAX as usize
        || append_len > u32::MAX as usize
    {
        return Err(Error::Other(
            "E4M3 KV append launch contract mismatch".into(),
        ));
    }
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qwen35_kv_append_bf16_e4m3(
            cache.ptr(),
            scales.ptr(),
            input_buffer.ptr(),
            kv_heads as u32,
            head_dim as u32,
            max_seq_len as u32,
            seq_len as u32,
            append_len as u32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

/// Quantize and append one BF16 token into an E4M3 KV cache.
#[allow(clippy::too_many_arguments)]
pub fn append_at_e4m3(
    ctx: &CudaContext,
    cache: &CudaBuffer,
    scales: &CudaBuffer,
    input: &CudaBuffer,
    kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    position: CudaDeviceAddress,
) -> Result<()> {
    require_buffers(
        ctx,
        "E4M3 KV append",
        &[
            (
                "cache",
                cache,
                kv_heads
                    .checked_mul(max_seq_len)
                    .and_then(|value| value.checked_mul(head_dim))
                    .ok_or_else(|| Error::Other("E4M3 KV cache size overflow".into()))?,
            ),
            (
                "scales",
                scales,
                kv_heads
                    .checked_mul(max_seq_len)
                    .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
                    .ok_or_else(|| Error::Other("E4M3 KV scale size overflow".into()))?,
            ),
            (
                "input",
                input,
                checked_bytes(DType::BF16, &[kv_heads, head_dim], "E4M3 KV append")?,
            ),
        ],
    )?;
    require_address(ctx, "E4M3 KV append", "position", position, 4)?;
    if kv_heads == 0
        || head_dim == 0
        || max_seq_len == 0
        || kv_heads > u32::MAX as usize
        || head_dim > u32::MAX as usize
        || max_seq_len > u32::MAX as usize
    {
        return Err(Error::Other(
            "E4M3 KV append launch contract mismatch".into(),
        ));
    }
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qwen35_kv_append_decode_bf16_e4m3(
            cache.ptr(),
            scales.ptr(),
            input.ptr(),
            kv_heads as u32,
            head_dim as u32,
            max_seq_len as u32,
            position.ptr(),
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

pub fn reserve_prefix_bf16(
    ctx: &CudaContext,
    prefix: &Tensor,
    total_rows: usize,
) -> Result<Tensor> {
    let (prefix_rows, cols) = matrix_shape(prefix, "prefix KV cache")?;
    if prefix.dtype() != DType::BF16 || total_rows < prefix_rows {
        return Err(Error::Other(
            "static inference BF16 prefix KV cache has incompatible shape".into(),
        ));
    }
    let output = bf16_output(ctx, total_rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::cudaMemcpyAsync(
            output.ptr(),
            gpu_ptr(prefix)?,
            prefix.size_in_bytes(),
            ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(matrix_tensor(ctx, total_rows, cols, output))
}
/// Allocate a persistent K/V cache and copy the prefix into its first rows.
pub fn reserve_prefix_f16(ctx: &CudaContext, prefix: &Tensor, total_rows: usize) -> Result<Tensor> {
    let (prefix_rows, cols) = matrix_shape(prefix, "prefix KV cache")?;
    if prefix.dtype() != DType::F16 || total_rows < prefix_rows {
        return Err(Error::Other(format!(
            "static inference prefix KV cache expected FP16 with total_rows >= {prefix_rows}, got {:?} and {total_rows}",
            prefix.shape().dims()
        )));
    }
    let output = f16_output(ctx, total_rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::cudaMemcpyAsync(
            output.ptr(),
            gpu_ptr(prefix)?,
            prefix.size_in_bytes(),
            ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![total_rows, cols]),
        DType::F16,
        ctx.device_id(),
        output,
    ))
}
