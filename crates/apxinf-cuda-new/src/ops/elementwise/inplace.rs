//! Caller-owned elementwise operations.

use apxinf_core::{Result, Tensor};

use crate::ops::gemm::contracts::invalid;
use crate::ops::{pointwise, PointwiseArgs, PointwiseSemantic};
use crate::CudaContext;

/// `accumulator += addend`, elementwise BF16.
pub fn add_into(ctx: &CudaContext, addend: &Tensor, accumulator: &Tensor) -> Result<()> {
    let dims = addend.shape().dims().to_vec();
    if dims != accumulator.shape().dims() {
        return Err(invalid("residual add requires matching shapes"));
    }
    let mut accumulator = accumulator.clone();
    pointwise(
        ctx,
        PointwiseArgs::new(PointwiseSemantic::AddInto, addend, &mut accumulator),
    )
}
