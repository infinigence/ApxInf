use apxinf_core::{Result, Tensor};

use super::contracts::{normalize_bias_residual, GemmArgs};
use super::gemm_execution as execution;
use crate::CudaContext;

/// Arguments for `alpha * (A @ B) + optional_bias + residual`.
pub struct GemmBiasResidualArgs<'a> {
    pub gemm: GemmArgs<'a>,
    pub bias: Option<&'a Tensor>,
    pub residual: &'a Tensor,
}

pub fn gemm_bias_residual(ctx: &CudaContext, args: GemmBiasResidualArgs<'_>) -> Result<()> {
    execution::execute(
        ctx,
        normalize_bias_residual(ctx, args.gemm, args.bias, args.residual)?,
    )
}
