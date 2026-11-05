use apxinf_core::{DType, Result, Tensor};

use crate::context::CudaContext;
use crate::ffi::abi::activation as activation_abi;
use crate::ops::validation::{check_cuda, expect_bf16, gpu_ptr, make_gpu_tensor};
use crate::workspace::output_buffer;

/// Exact (erf) GELU with BF16 storage, owned by the pointwise family.
pub fn gelu_exact(ctx: &CudaContext, input: &Tensor) -> Result<Tensor> {
    expect_bf16(input, "gelu exact")?;
    let output = output_buffer(ctx, input.size_in_bytes())?;
    unsafe {
        check_cuda(activation_abi::apxinf_activation_gelu_exact_bf16(
            gpu_ptr(input)?,
            output.ptr(),
            input.numel() as i64,
            ctx.stream().handle(),
        ))
    }?;
    Ok(make_gpu_tensor(
        input.shape().clone(),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}
