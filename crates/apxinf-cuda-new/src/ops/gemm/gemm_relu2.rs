use apxinf_core::Result;

use super::contracts::{normalize, GemmArgs, Semantic};
use super::gemm_execution as execution;
use crate::CudaContext;

/// BF16 GEMM followed by squared ReLU (relu(x)^2). The FP32 accumulator is
/// rounded to BF16 before the activation; the FP32 result is rounded to BF16
/// again. No bias or quantization is admitted, and alpha/output_scale must be
/// one. This is the Cosmos3-Edge MoT MLP up-projection epilogue.
pub fn gemm_relu_squared(ctx: &CudaContext, args: GemmArgs<'_>) -> Result<()> {
    super::observer::observe(&args)?;
    execution::execute(ctx, normalize(ctx, args, Semantic::GemmRelu2, None)?)
}
