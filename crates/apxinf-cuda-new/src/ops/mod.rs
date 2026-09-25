//! Semantic CUDA APIs with selection and native state behind L1.

mod attention;
mod attn_ops;
mod gdn;
mod gemm;
mod mlp;
mod model;

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
pub use attn_ops::{
    apply_output_gate, head_rms_norm, partial_rope, rotary_dim, split_query_and_gate,
};
pub use gdn::{
    gdn_causal_conv_forward, gdn_causal_conv_step, gdn_decay_and_beta,
    gdn_decay_and_beta_seq, gdn_gated_norm, gdn_gated_norm_seq,
    flashinfer_gdn_prefill, flashinfer_gdn_workspace_bytes,
    convert_f16_to_bf16, gdn_prepare_flashinfer,
    gdn_chunk_scan, gdn_chunk_scan_interleaved, gdn_l2_normalize_heads,
    gdn_recurrent_step, gdn_state_elements,
};
pub use model::{argmax, embedding_gather};
pub use mlp::{
    add_into, fp8_gemv, nvfp4_gemv, quantize_fp8_per_tensor, rms_norm, swiglu,
};
pub use gemm::{
    gemm, gemm_bias, gemm_bias_gelu, gemm_geglu, nvfp4_pack_block_scales,
    nvfp4_quantize_activation, nvfp4_quantize_rms_norm, nvfp4_quantize_swiglu,
    nvfp4_scale_buffer_bytes, ScaleLayout, GemmArgs, GemmBiasArgs, GemmBiasGeluArgs, GemmGegluArgs,
    GemmPolicy, GemmQuantization, WeightVersion,
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
