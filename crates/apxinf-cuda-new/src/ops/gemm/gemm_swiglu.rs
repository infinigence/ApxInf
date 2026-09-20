use apxinf_core::Result;

use super::contracts::{normalize, GemmArgs, Semantic};
use super::gemm_execution as execution;
use crate::CudaContext;

/// Fused SwiGLU over a canonical `[K, 2N]` weight. The first half is the
/// gate projection and the second half is the up projection.
pub struct GemmSwigluArgs<'a> {
    pub gemm: GemmArgs<'a>,
}

pub fn gemm_swiglu(ctx: &CudaContext, args: GemmSwigluArgs<'_>) -> Result<()> {
    execution::execute(ctx, normalize(ctx, args.gemm, Semantic::GemmSwiglu, None)?)
}
