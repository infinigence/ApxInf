use apxinf_core::{Result, Tensor};

use super::contracts::{normalize, GemmArgs, Semantic};
use super::gemm_execution as execution;
use crate::CudaContext;

pub struct GemmBiasGeluArgs<'a> {
    pub gemm: GemmArgs<'a>,
    pub bias: &'a Tensor,
}

pub fn gemm_bias_gelu(ctx: &CudaContext, args: GemmBiasGeluArgs<'_>) -> Result<()> {
    super::observer::observe(&args.gemm)?;
    execution::execute(
        ctx,
        normalize(ctx, args.gemm, Semantic::GemmBiasGelu, Some(args.bias))?,
    )
}

/// BF16 GEMM followed by exact erf GELU. The FP32 accumulator is rounded to
/// BF16 before GELU; its FP32 result is rounded to BF16 again. No bias or
/// quantization is admitted, and alpha/output_scale must both be one.
pub fn gemm_gelu_exact(ctx: &CudaContext, args: GemmArgs<'_>) -> Result<()> {
    super::observer::observe(&args)?;
    execution::execute(ctx, normalize(ctx, args, Semantic::GemmGeluExact, None)?)
}
