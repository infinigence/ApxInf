use apxinf_core::{Result, Tensor};

use super::contracts::{normalize, GemmArgs, Semantic};
use super::gemm_execution as execution;
use crate::CudaContext;

pub struct GemmBiasArgs<'a> {
    pub gemm: GemmArgs<'a>,
    pub bias: &'a Tensor,
}

/// Accumulate the projection and bias in F32 before the sole F16/BF16 cast.
/// Unlike `gemm_bias`, this does not round the projection before adding bias.
pub fn gemm_bias_single_round(ctx: &CudaContext, args: GemmBiasArgs<'_>) -> Result<()> {
    execution::execute(
        ctx,
        normalize(
            ctx,
            args.gemm,
            Semantic::GemmBiasSingleRound,
            Some(args.bias),
        )?,
    )
}

pub fn gemm_bias(ctx: &CudaContext, args: GemmBiasArgs<'_>) -> Result<()> {
    super::observer::observe(&args.gemm)?;
    execution::execute(
        ctx,
        normalize(ctx, args.gemm, Semantic::GemmBias, Some(args.bias))?,
    )
}
