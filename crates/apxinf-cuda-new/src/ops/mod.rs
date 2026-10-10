//! Semantic CUDA APIs with selection and native state behind L1.

mod attention;
mod attn_ops;
mod cache;
mod gather;
mod gdn;
mod gemm;
pub mod mlp;
mod model;
mod norm;
mod pointwise;
mod quantization;
mod rope;

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
    attention, kv_cache_attention, packed_qkv_attention, segmented_attention, AttentionArgs,
    AttentionMask, AttentionPolicy, KvCacheAttentionArgs, KvCacheDecodeMeta,
    PackedQkvAttentionArgs, SegmentedAttentionArgs,
};
pub use attn_ops::{
    apply_output_gate, head_rms_norm, partial_rope, rotary_dim, split_query_and_gate,
};
pub use cache::{concat_rows, reserve_prefix};
pub use gather::{gather, GatherArgs, GatherSemantic, PatchGeometry as GatherPatchGeometry};
pub use gdn::{
    convert_f16_to_bf16, flashinfer_gdn_prefill, flashinfer_gdn_workspace_bytes,
    gdn_causal_conv_forward, gdn_causal_conv_step, gdn_chunk_scan, gdn_chunk_scan_interleaved,
    gdn_conv_prepare_flashinfer, gdn_decay_and_beta, gdn_decay_and_beta_seq, gdn_gated_norm,
    gdn_gated_norm_seq, gdn_gated_norm_seq_f16, gdn_l2_normalize_heads, gdn_prepare_flashinfer,
    gdn_recurrent_step, gdn_state_elements,
};
pub use gemm::{
    gemm, gemm_bias, gemm_bias_gelu, gemm_bias_residual, gemm_geglu, nvfp4_pack_block_scales,
    nvfp4_quantize_activation, nvfp4_quantize_rms_norm, nvfp4_quantize_swiglu,
    nvfp4_scale_buffer_bytes, GemmArgs, GemmBiasArgs, GemmBiasGeluArgs, GemmBiasResidualArgs,
    GemmGegluArgs, GemmPolicy, GemmQuantization, ScaleLayout, WeightVersion,
};
pub use mlp::{add_into, fp8_gemv, nvfp4_gemv, quantize_fp8_per_tensor, swiglu};
pub use model::{argmax, embedding_gather};
pub use norm::{
    ada_gate_residual, ada_gate_residual_rms_norm, adaptive_rms_norm, bias_residual,
    bias_residual_layer_norm, bias_residual_rms_norm, bias_then_residual, layer_norm, rms_norm,
    AdaGateResidualArgs, AdaGateResidualRmsNormArgs, AdaptiveRmsNormArgs, BiasResidualArgs,
    BiasResidualLayerNormArgs, BiasResidualRmsNormArgs, BiasThenResidualArgs, LayerNormArgs,
    RmsNormArgs,
};
pub use pointwise::{pointwise, PointwiseActivation, PointwiseArgs, PointwiseSemantic};
pub use quantization::{quantization, QuantizationArgs, QuantizationSemantic};
pub use rope::{decode_rope, rope, DecodeRopeArgs, RopeArgs, RopeSemantic};

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
