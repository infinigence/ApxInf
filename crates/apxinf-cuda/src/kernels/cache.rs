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
