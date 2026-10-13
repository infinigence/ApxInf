//! Semantic CUDA APIs with selection and native state behind L1.

mod attention;
mod cache;
mod gather;
mod gdn;
mod gemm;
mod norm;
mod pointwise;
mod quantization;
mod rope;
mod sampling;

// Keep these crate-private aliases while graph/workspace and unit tests still
// refer to the GEMM implementation through `crate::ops`.
#[cfg(test)]
pub(crate) use attention::{
    contracts as attention_contracts, execution as attention_execution,
    normalize_kv_cache_attention, normalize_segmented_attention,
};
#[cfg(test)]
pub(crate) use norm::{fused_add_norm_spec, norm_spec, residual_add_spec};
#[cfg(test)]
pub(crate) use gemm::contracts;
#[cfg(test)]
pub(crate) use gemm::gemm_execution as execution;

pub use crate::workspace::{ExecutionSession, GraphWorkspace};
pub use attention::{
    apply_output_gate, attention, decode_attention, decode_attention_workspace_bytes,
    kv_cache_attention, packed_qkv_attention, segmented_attention, split_query_and_gate,
    AttentionArgs, AttentionMask, AttentionPolicy, KvCacheAttentionArgs, KvCacheDecodeMeta,
    PackedQkvAttentionArgs, SegmentedAttentionArgs,
};
pub use cache::{concat_rows, reserve_prefix};
pub use gather::{
    embedding_gather, gather, GatherArgs, GatherSemantic,
    PatchGeometry as GatherPatchGeometry,
};
pub use gdn::{
    gdn_causal_conv_forward, gdn_causal_conv_step, gdn_decay_and_beta,
    gdn_decay_and_beta_seq, gdn_gated_norm, gdn_gated_norm_quantize, gdn_gated_norm_seq, gdn_gated_norm_seq_f16,
    gdn_prefill, gdn_prefill_workspace_bytes,
    gdn_widen_f16_to_bf16, gdn_prepare_prefill, gdn_conv_prepare,
    gdn_chunk_scan, gdn_chunk_scan_interleaved, gdn_l2_normalize_heads,
    gdn_recurrent_step, gdn_state_elements,
};
pub use gemm::{
    fp8_gemv, gemm, gemm_bias, gemm_bias_gelu, gemm_bias_residual, gemm_geglu, nvfp4_dense_swiglu_aot,
    nvfp4_gemv, nvfp4_pack_block_scales, nvfp4_quantize_activation, nvfp4_quantize_rms_norm,
    nvfp4_quantize_swiglu, nvfp4_scale_buffer_bytes, GemmArgs, GemmBiasArgs, GemmBiasGeluArgs,
    GemmBiasResidualArgs, GemmGegluArgs, GemmPolicy, GemmQuantization, ScaleLayout, WeightVersion,
};
pub use norm::{
    fused_add_norm, head_rms_norm, norm, residual_add, Affine, BiasOrder, FusedAddNormArgs,
    NormArgs, NormKind, QuantOut, ResidualAddArgs,
};
pub use pointwise::{
    add_into, pointwise, swiglu, PointwiseActivation, PointwiseArgs, PointwiseSemantic,
};
pub use quantization::{
    quantization, quantize_fp8_per_tensor, QuantizationArgs, QuantizationSemantic,
};
pub use rope::{
    decode_rope, partial_rope, rope, rotary_dim, DecodeRopeArgs, RopeArgs, RopeSemantic,
};
pub use sampling::argmax;

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
