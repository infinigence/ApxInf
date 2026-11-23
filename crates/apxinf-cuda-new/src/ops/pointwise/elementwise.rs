//! Fixed pointwise operations with one implementation each.

use apxinf_core::{Result, Tensor};

use crate::ops::gemm::contracts::invalid;
use crate::CudaContext;

use super::{pointwise, PointwiseArgs, PointwiseSemantic};

/// `y[r,c] = silu(fused[r,c]) * fused[r,width+c]`.
pub fn swiglu_into(ctx: &CudaContext, fused: &Tensor, output: &Tensor) -> Result<()> {
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
    let mut output = output.clone();
    pointwise(
        ctx,
        PointwiseArgs::new(PointwiseSemantic::Swiglu, fused, &mut output),
    )
}
