//! Per-tensor quantization helpers.

use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::{quantization as abi, status};
use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::CudaContext;

/// Quantize BF16 to E4M3 against a single per-tensor scale.
pub fn quantize_per_tensor(
    ctx: &CudaContext,
    input: &Tensor,
    output: &Tensor,
    input_scale: f32,
) -> Result<()> {
    let dims = input.shape().dims().to_vec();
    if dims != output.shape().dims() {
        return Err(invalid("FP8 quantization requires matching shapes"));
    }
    if !(input_scale > 0.0) || !input_scale.is_finite() {
        return Err(invalid("FP8 input_scale must be finite and positive"));
    }
    let source = tensor_storage(ctx, input, DType::BF16, &dims)?;
    let destination = tensor_storage(ctx, output, DType::F8E4M3, &dims)?;
    unsafe {
        status::check(abi::apxinf_quantization_fp8_per_tensor(
            source.ptr(),
            destination.ptr(),
            dims.iter().product::<usize>() as i64,
            input_scale,
            ctx.stream().handle(),
        ))
    }
}
