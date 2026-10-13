//! The pointwise family's non-Spec members: BF16 SwiGLU and residual add.
//!
//! The Spec-dispatched semantics of this family are in [`super::pointwise`].
//! These two are BF16-only with a single implementation each, so they keep flat
//! parameters per `doc/cuda-new-design.md` section 5 — there is no variant axis
//! for a Spec to carry.
//!
//! Moved here from `ops/mlp.rs`, which was named after a call site rather than
//! a mathematical family (section 1.3).

use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::{pointwise as abi, status};
use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::CudaContext;

/// `y[r,c] = silu(fused[r,c]) * fused[r,width+c]`
///
/// `fused` is `[rows, 2*width]` with gate first, the layout one fused gate/up
/// GEMM produces. This is SwiGLU; [`PointwiseSemantic::Geglu`] computes the
/// GELU variant and is not interchangeable.
///
/// [`PointwiseSemantic::Geglu`]: super::PointwiseSemantic::Geglu
pub fn swiglu(ctx: &CudaContext, fused: &Tensor, output: &Tensor) -> Result<()> {
    let fused_dims = fused.shape().dims().to_vec();
    let output_dims = output.shape().dims().to_vec();
    if fused_dims.len() != 2 || output_dims.len() != 2 {
        return Err(invalid("SwiGLU requires rank-2 tensors"));
    }
    if fused_dims[0] != output_dims[0] || fused_dims[1] != output_dims[1] * 2 {
        return Err(invalid(
            "SwiGLU input must be [rows, 2*width] for an output of [rows, width]",
        ));
    }
    let fused_buffer = tensor_storage(ctx, fused, DType::BF16, &fused_dims)?;
    let output_buffer = tensor_storage(ctx, output, DType::BF16, &output_dims)?;
    unsafe {
        status::check(abi::apxinf_pointwise_swiglu(
            fused_buffer.ptr(),
            output_buffer.ptr(),
            output_dims[0] as i64,
            output_dims[1] as i64,
            ctx.stream().handle(),
        ))
    }
}

/// `accumulator += addend`, elementwise. Residual connections.
pub fn add_into(ctx: &CudaContext, addend: &Tensor, accumulator: &Tensor) -> Result<()> {
    let dims = addend.shape().dims().to_vec();
    if dims != accumulator.shape().dims() {
        return Err(invalid("residual add requires matching shapes"));
    }
    let addend_buffer = tensor_storage(ctx, addend, DType::BF16, &dims)?;
    let accumulator_buffer = tensor_storage(ctx, accumulator, DType::BF16, &dims)?;
    unsafe {
        status::check(abi::apxinf_pointwise_add_into(
            addend_buffer.ptr(),
            accumulator_buffer.ptr(),
            dims.iter().product::<usize>() as i64,
            ctx.stream().handle(),
        ))
    }
}
