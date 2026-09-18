use apxinf_core::{Result, Tensor};

use super::contracts::{normalize, GemmArgs, Semantic};
use super::gemm_execution as execution;
use crate::CudaContext;

pub struct GemmBiasReluArgs<'a> {
    pub gemm: GemmArgs<'a>,
    pub bias: &'a Tensor,
}

pub fn gemm_bias_relu(ctx: &CudaContext, args: GemmBiasReluArgs<'_>) -> Result<()> {
    execution::execute(
        ctx,
        normalize(ctx, args.gemm, Semantic::GemmBiasRelu, Some(args.bias))?,
    )
}
