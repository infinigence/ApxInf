//! Semantic CUDA APIs with selection and native state behind L1.

mod attention;
mod gemm;

// Keep these crate-private aliases while graph/workspace and unit tests still
// refer to the GEMM implementation through `crate::ops`.
#[cfg(test)]
pub(crate) use attention::{
    contracts as attention_contracts, execution as attention_execution,
    normalize_kv_cache_attention, normalize_segmented_attention,
};
#[cfg(test)]
pub(crate) use gemm::contracts;
#[cfg(test)]
pub(crate) use gemm::gemm_execution as execution;

pub use crate::workspace::{ExecutionSession, GraphWorkspace};
pub use attention::{
    attention, kv_cache_attention, segmented_attention, AttentionArgs, AttentionMask,
    AttentionPolicy, KvCacheAttentionArgs, SegmentedAttentionArgs,
};
pub use gemm::{
    gemm, gemm_bias, gemm_bias_gelu, gemm_bias_relu, gemm_bias_residual, gemm_bias_silu,
    gemm_geglu, gemm_swiglu, GemmArgs, GemmBiasArgs, GemmBiasGeluArgs, GemmBiasReluArgs,
    GemmBiasResidualArgs, GemmBiasSiluArgs, GemmGegluArgs, GemmPolicy, GemmQuantization,
    GemmSwigluArgs, WeightVersion,
};

/// Run a fixed-shape forward pass that may tune and create native executions.
pub fn prepare_with_session<T>(
    session: &ExecutionSession,
    operation: impl FnOnce() -> apxinf_core::Result<T>,
) -> apxinf_core::Result<T> {
    crate::workspace::prepare_with_session(session, operation)
}

/// Run the same prepared forward pass without allocating or tuning.
/// Enqueues are asynchronous; synchronize at the outer execution boundary.
pub fn with_session<T>(
    session: &ExecutionSession,
    operation: impl FnOnce() -> apxinf_core::Result<T>,
) -> apxinf_core::Result<T> {
    crate::workspace::with_session(session, operation)
}
#[cfg(test)]
mod tests;
