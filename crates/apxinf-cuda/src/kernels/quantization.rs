//! Quantization operator contracts.

use apxinf_core::{DType, Error, Result, Tensor};

use super::contracts::{gpu_ptr, make_gpu_tensor};
use crate::context::CudaContext;
use crate::ffi;
use crate::workspace::output_buffer;

/// Quantize an FP16 device tensor to E4M3 using a pre-calibrated scale.
pub fn quantize_f16_e4m3(ctx: &CudaContext, input: &Tensor, scale: f32) -> Result<Tensor> {
    if input.dtype() != DType::F16 {
        return Err(Error::DTypeMismatch {
            expected: DType::F16,
            got: input.dtype(),
        });
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::Other(format!("invalid FP8 scale {scale}")));
    }
    let output = output_buffer(ctx, input.numel())?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_quantize_f16_e4m3(
            gpu_ptr(input)?,
            output.ptr(),
            input.numel() as i64,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        input.shape().clone(),
        DType::F8E4M3,
        ctx.device_id(),
        output,
    ))
}
