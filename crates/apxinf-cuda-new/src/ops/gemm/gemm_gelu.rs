use apxinf_core::{Result, Tensor};

use super::contracts::{normalize, GemmArgs, Semantic};
use super::gemm_execution as execution;
use crate::CudaContext;

pub struct GemmBiasGeluArgs<'a> {
    pub gemm: GemmArgs<'a>,
    pub bias: &'a Tensor,
}

pub fn gemm_bias_gelu(ctx: &CudaContext, args: GemmBiasGeluArgs<'_>) -> Result<()> {
    execution::execute(
        ctx,
        normalize(ctx, args.gemm, Semantic::GemmBiasGelu, Some(args.bias))?,
    )
}
