//! Shared validation helpers for caller-owned kernel buffers.

use std::ffi::c_void;

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::buffer::{CudaBuffer, CudaDeviceAddress};
use crate::context::CudaContext;

pub(super) fn checked_bytes(dtype: DType, dims: &[usize], operation: &str) -> Result<usize> {
    if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Err(Error::Other(format!(
            "{operation} does not support {dtype}"
        )));
    }
    if dims.iter().any(|dimension| *dimension == 0) {
        return Err(Error::Other(format!(
            "{operation} dimensions must be non-zero: {dims:?}"
        )));
    }
    dims.iter()
        .try_fold(dtype.size_in_bytes(), |bytes, dimension| {
            bytes
                .checked_mul(*dimension)
                .ok_or_else(|| Error::Other(format!("{operation} byte size overflow")))
        })
}

pub(super) fn require_buffers(
    ctx: &CudaContext,
    operation: &str,
    buffers: &[(&str, &CudaBuffer, usize)],
) -> Result<()> {
    for (name, buffer, required) in buffers {
        if buffer.device() != ctx.device_id() {
            return Err(Error::Other(format!(
                "{operation} {name} is on CUDA{}, expected CUDA{}",
                buffer.device(),
                ctx.device_id()
            )));
        }
        if buffer.len() < *required {
            return Err(Error::Other(format!(
                "{operation} {name} requires {required} bytes, has {}",
                buffer.len()
            )));
        }
    }
    Ok(())
}

pub(super) fn require_address(
    ctx: &CudaContext,
    operation: &str,
    name: &str,
    address: CudaDeviceAddress,
    required: usize,
) -> Result<()> {
    if address.device() != ctx.device_id() || address.len() < required {
        return Err(Error::Other(format!(
            "{operation} {name} needs {required} bytes on CUDA{}, got {} bytes on CUDA{}",
            ctx.device_id(),
            address.len(),
            address.device()
        )));
    }
    Ok(())
}

pub(super) fn require_finite(operation: &str, values: &[f32]) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(Error::Other(format!(
            "{operation} received a non-finite scalar"
        )))
    }
}

pub(super) fn unsupported_dtype<T>(operation: &str, dtype: DType) -> Result<T> {
    Err(Error::Other(format!(
        "CUDA {operation} does not support {dtype}"
    )))
}

pub(super) fn tensor_ptr(tensor: &Tensor) -> Result<*mut c_void> {
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    Ok(buffer.ptr())
}

pub(super) fn gpu_ptr(tensor: &Tensor) -> Result<*mut c_void> {
    tensor_ptr(tensor)
}

pub(super) fn optional_tensor_ptr(tensor: Option<&Tensor>) -> Result<*mut c_void> {
    tensor.map_or(Ok(std::ptr::null_mut()), tensor_ptr)
}

pub(super) fn optional_ptr(tensor: Option<&Tensor>) -> Result<*mut c_void> {
    optional_tensor_ptr(tensor)
}

pub(super) fn matrix_shape(tensor: &Tensor, operation: &str) -> Result<(usize, usize)> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!(
            "CUDA {operation} expects a 2D tensor, got {dims:?}"
        )));
    }
    Ok((dims[0], dims[1]))
}

pub(super) fn output_tensor(
    ctx: &CudaContext,
    shape: Shape,
    dtype: DType,
    buffer: CudaBuffer,
) -> Tensor {
    debug_assert_eq!(buffer.device(), ctx.device_id());
    buffer.into_tensor(shape, dtype)
}

pub(super) fn make_gpu_tensor(
    shape: Shape,
    dtype: DType,
    _device_id: usize,
    buffer: CudaBuffer,
) -> Tensor {
    buffer.into_tensor(shape, dtype)
}

pub(super) fn matrix_tensor(
    ctx: &CudaContext,
    rows: usize,
    cols: usize,
    buffer: CudaBuffer,
) -> Tensor {
    output_tensor(ctx, Shape::new(vec![rows, cols]), DType::BF16, buffer)
}

pub(super) fn bf16_output(ctx: &CudaContext, rows: usize, cols: usize) -> Result<CudaBuffer> {
    crate::workspace::output_buffer(
        ctx,
        rows.checked_mul(cols)
            .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
            .ok_or_else(|| Error::Other("BF16 output size overflow".into()))?,
    )
}

pub(super) fn f16_output(ctx: &CudaContext, rows: usize, cols: usize) -> Result<CudaBuffer> {
    crate::workspace::output_buffer(
        ctx,
        rows.checked_mul(cols)
            .and_then(|elements| elements.checked_mul(DType::F16.size_in_bytes()))
            .ok_or_else(|| Error::Other("FP16 output size overflow".into()))?,
    )
}

pub(super) fn fp8_output(ctx: &CudaContext, rows: usize, cols: usize) -> Result<CudaBuffer> {
    crate::workspace::output_buffer(
        ctx,
        rows.checked_mul(cols)
            .ok_or_else(|| Error::Other("FP8 output size overflow".into()))?,
    )
}

pub(super) fn check_cuda(status: crate::ffi::cudaError_t) -> Result<()> {
    crate::ffi::check_cuda(status).map_err(Error::Cuda)
}
