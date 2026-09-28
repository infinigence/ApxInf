//! Persistent KV-cache allocation and prefix initialization.

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use crate::ffi;
use crate::{CudaBuffer, CudaContext};

/// Allocate a persistent row-major K/V cache and copy `prefix` into its first
/// rows. During an execution session the cache is a deterministic arena slice,
/// so prepare and capture reuse the same address without allocating inside CUDA
/// stream capture. Remaining rows are zero-initialized so callers never expose
/// stale device memory when a later validity length is wrong.
pub fn reserve_prefix(ctx: &CudaContext, prefix: &Tensor, total_rows: usize) -> Result<Tensor> {
    let dims = prefix.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other("prefix KV cache must be rank 2".into()));
    }
    let (prefix_rows, cols) = (dims[0], dims[1]);
    if !matches!(prefix.dtype(), DType::F16 | DType::BF16)
        || prefix.device() != Device::Cuda(ctx.device_id())
        || total_rows < prefix_rows
    {
        return Err(Error::Other(
            "prefix KV cache has incompatible dtype, device, or row count".into(),
        ));
    }
    let bytes = total_rows
        .checked_mul(cols)
        .and_then(|elements| elements.checked_mul(prefix.dtype().size_in_bytes()))
        .ok_or_else(|| Error::Other("prefix KV cache size overflow".into()))?;
    let shape = Shape::new(vec![total_rows, cols]);
    let output_tensor = ctx.allocate_output(shape, prefix.dtype())?;
    let output = CudaBuffer::from_tensor(&output_tensor).map_err(Error::Cuda)?;
    let source = CudaBuffer::from_tensor(prefix).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::cudaMemsetAsync(
            output.ptr(),
            0,
            bytes,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
        ffi::check_cuda(ffi::cudaMemcpyAsync(
            output.ptr(),
            source.ptr(),
            prefix.size_in_bytes(),
            ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(output_tensor)
}

/// Concatenate two contiguous row-major matrices with the same column count.
///
/// This is a graph-safe layout semantic: it records two asynchronous D2D
/// copies on the context stream and allocates the result from the active
/// execution session when present.
pub fn concat_rows(ctx: &CudaContext, first: &Tensor, second: &Tensor) -> Result<Tensor> {
    let first_dims = first.shape().dims();
    let second_dims = second.shape().dims();
    if first_dims.len() != 2
        || second_dims.len() != 2
        || first_dims[1] != second_dims[1]
        || first.dtype() != second.dtype()
        || first.device() != Device::Cuda(ctx.device_id())
        || second.device() != Device::Cuda(ctx.device_id())
    {
        return Err(Error::Other(
            "row concatenation requires same-dtype CUDA matrices with equal columns".into(),
        ));
    }
    let rows = first_dims[0]
        .checked_add(second_dims[0])
        .ok_or_else(|| Error::Other("row concatenation size overflow".into()))?;
    let shape = Shape::new(vec![rows, first_dims[1]]);
    let output = ctx.allocate_output(shape, first.dtype())?;
    let first_buffer = CudaBuffer::from_tensor(first).map_err(Error::Cuda)?;
    let second_buffer = CudaBuffer::from_tensor(second).map_err(Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::cudaMemcpyAsync(
            output_buffer.ptr(),
            first_buffer.ptr(),
            first.size_in_bytes(),
            ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
        ffi::check_cuda(ffi::cudaMemcpyAsync(
            output_buffer
                .ptr()
                .cast::<u8>()
                .add(first.size_in_bytes())
                .cast(),
            second_buffer.ptr(),
            second.size_in_bytes(),
            ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(output)
}
