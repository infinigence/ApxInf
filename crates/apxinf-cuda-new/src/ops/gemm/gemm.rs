use apxinf_core::Result;

use super::contracts::{normalize, GemmArgs, Semantic};
use super::gemm_execution as execution;
use crate::CudaContext;

pub fn gemm(ctx: &CudaContext, args: GemmArgs<'_>) -> Result<()> {
    execution::execute(ctx, normalize(ctx, args, Semantic::Gemm, None)?)
}
