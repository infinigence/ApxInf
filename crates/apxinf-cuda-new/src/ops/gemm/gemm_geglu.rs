use apxinf_core::Result;

use super::contracts::{normalize, GemmArgs, Semantic};
use super::gemm_execution as execution;
use crate::CudaContext;

/// Arguments for the public GeGLU semantic.
///
/// `gemm.b` is canonical contiguous row-major `[K, 2N]`: its first `N`
/// columns are the logical `B_gate`, and its remaining `N` columns are the
/// logical `B_up`. The output is `[M, N]`. Candidates may use any internal
/// packing as long as they preserve this public meaning.
pub struct GemmGegluArgs<'a> {
    pub gemm: GemmArgs<'a>,
}

pub fn gemm_geglu(ctx: &CudaContext, args: GemmGegluArgs<'_>) -> Result<()> {
    execution::execute(ctx, normalize(ctx, args.gemm, Semantic::GemmGeglu, None)?)
}
