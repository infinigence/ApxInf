use apxinf_core::{Result, Tensor};

use super::contracts::{normalize_with_residual, GemmArgs, Semantic};
use super::gemm_execution as execution;
use crate::CudaContext;

pub struct GemmBiasResidualArgs<'a> {
    pub gemm: GemmArgs<'a>,
    pub bias: &'a Tensor,
    pub residual: &'a Tensor,
}

pub fn gemm_bias_residual(ctx: &CudaContext, args: GemmBiasResidualArgs<'_>) -> Result<()> {
    execution::execute(
        ctx,
        normalize_with_residual(
            ctx,
            args.gemm,
            Semantic::GemmBiasResidual,
            Some(args.bias),
            args.residual,
        )?,
    )
}
