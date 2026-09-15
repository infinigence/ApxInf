use apxinf_core::{Result, Tensor};

use super::contracts::{normalize, GemmArgs, Semantic};
use super::gemm_execution as execution;
use crate::CudaContext;

pub struct GemmBiasArgs<'a> {
    pub gemm: GemmArgs<'a>,
    pub bias: &'a Tensor,
}

pub fn gemm_bias(ctx: &CudaContext, args: GemmBiasArgs<'_>) -> Result<()> {
    execution::execute(
        ctx,
        normalize(ctx, args.gemm, Semantic::GemmBias, Some(args.bias))?,
    )
}
