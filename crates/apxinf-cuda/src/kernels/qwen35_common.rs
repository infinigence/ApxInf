use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::{ffi, CudaBuffer, CudaContext};

pub const HIDDEN: usize = 5120;

pub fn rmsnorm_offset_write(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    epsilon: f32,
) -> Result<()> {
    let rows = require_hidden(ctx, "input", input)?;
    require_weight(ctx, weight)?;
    if require_hidden(ctx, "output", output)? != rows {
        return Err(Error::Other(
            "Qwen3.5 offset RMSNorm input/output row mismatch".into(),
        ));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(Error::Other(
            "Qwen3.5 offset RMSNorm epsilon must be positive".into(),
        ));
    }
    let input = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let weight = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    let output = CudaBuffer::from_tensor(output).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qwen35_rmsnorm_offset_bf16(
            input.ptr(),
            weight.ptr(),
            output.ptr(),
            rows as i32,
            HIDDEN as i32,
            epsilon,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

pub fn residual_add_rmsnorm_offset_write(
    ctx: &CudaContext,
    residual: &Tensor,
    delta: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    epsilon: f32,
) -> Result<()> {
    let rows = require_hidden_rows(ctx, "residual", residual, 64)?;
    for (name, tensor) in [("delta", delta), ("output", output)] {
        if require_hidden_rows(ctx, name, tensor, 64)? != rows {
            return Err(Error::Other(format!(
                "Qwen3.5 residual offset RMSNorm {name} row mismatch"
            )));
        }
    }
    require_weight(ctx, weight)?;
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(Error::Other(
            "Qwen3.5 residual offset RMSNorm epsilon must be positive".into(),
        ));
    }
    let residual = CudaBuffer::from_tensor(residual).map_err(Error::Cuda)?;
    let delta = CudaBuffer::from_tensor(delta).map_err(Error::Cuda)?;
    let weight = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    let output = CudaBuffer::from_tensor(output).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_qwen35_residual_add_rmsnorm_offset_bf16(
            residual.ptr(),
            delta.ptr(),
            weight.ptr(),
            output.ptr(),
            rows as i32,
            HIDDEN as i32,
            epsilon,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

fn require_hidden(ctx: &CudaContext, name: &str, tensor: &Tensor) -> Result<usize> {
    require_hidden_rows(ctx, name, tensor, 8)
}

fn require_hidden_rows(
    ctx: &CudaContext,
    name: &str,
    tensor: &Tensor,
    max_rows: usize,
) -> Result<usize> {
    let dims = tensor.shape().dims();
    if tensor.dtype() != DType::BF16
        || tensor.device() != Device::Cuda(ctx.device_id())
        || dims.len() != 2
        || !(1..=max_rows).contains(&dims[0])
        || dims[1] != HIDDEN
    {
        return Err(Error::Other(format!(
            "Qwen3.5 offset RMSNorm {name} must be BF16 [1..=8,{HIDDEN}] on CUDA{}, got {} {:?} on {}",
            ctx.device_id(), tensor.dtype(), dims, tensor.device()
        )));
    }
    Ok(dims[0])
}

fn require_weight(ctx: &CudaContext, tensor: &Tensor) -> Result<()> {
    if tensor.dtype() != DType::BF16
        || tensor.device() != Device::Cuda(ctx.device_id())
        || tensor.shape().dims() != [HIDDEN]
    {
        return Err(Error::Other(format!(
            "Qwen3.5 offset RMSNorm weight must be BF16 [{HIDDEN}] on CUDA{}",
            ctx.device_id()
        )));
    }
    Ok(())
}
