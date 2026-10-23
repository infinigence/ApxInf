//! Semantic CUDA APIs with selection and native state behind L1.

pub mod activation;
pub mod attention;
mod attn_ops;
pub mod cache;
pub mod elementwise;
pub mod embedding;
pub mod fused;
mod gather;
mod gdn;
pub(crate) mod gdn_policy;
pub mod gemm;
pub mod linear_attention;
pub mod mlp;
mod model;
pub mod norm;
pub mod pillow_bicubic;
mod pointwise;
pub mod preprocess;
pub mod quantization;
pub mod rope;
pub mod sampling;
pub(crate) mod validation;

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
    attention, decode_attention, decode_attention_workspace_bytes, kv_cache_attention,
    packed_qkv_attention, segmented_attention, AttentionArgs, AttentionMask, AttentionPolicy,
    KvCacheAttentionArgs, KvCacheDecodeMeta, PackedQkvAttentionArgs, SegmentedAttentionArgs,
};
pub use attn_ops::{
    apply_output_gate, head_rms_norm, partial_rope, rotary_dim, split_query_and_gate,
};
pub use cache::core::{append as cache_append, concat_rows, reserve_prefix};
pub use elementwise::core::{
    activation as elementwise_activation, add as elementwise_add, add_bias as elementwise_add_bias,
    mul as elementwise_mul, scale as elementwise_scale, ElementwiseActivation,
};
pub use gather::{gather, GatherArgs, GatherSemantic, PatchGeometry as GatherPatchGeometry};
pub use gdn::{
    gdn_causal_conv_forward, gdn_causal_conv_step, gdn_chunk_scan, gdn_chunk_scan_interleaved,
    gdn_conv_prepare, gdn_decay_and_beta, gdn_decay_and_beta_seq, gdn_first_prefill,
    gdn_gated_norm, gdn_gated_norm_quantize, gdn_gated_norm_seq, gdn_gated_norm_seq_f16,
    gdn_l2_normalize_heads, gdn_prefill, gdn_prefill_workspace_bytes, gdn_prepare_prefill,
    gdn_recurrent_step, gdn_state_elements, gdn_widen_f16_to_bf16, GdnFirstPrefillArgs,
};
pub use gemm::{
    gemm, gemm_bias, gemm_bias_gelu, gemm_bias_residual, gemm_geglu,
    gemm_swiglu_pair_interleaved_256, nvfp4_gemm_swiglu, nvfp4_pack_block_scales,
    nvfp4_quantize_activation, nvfp4_quantize_rms_norm, nvfp4_quantize_swiglu,
    nvfp4_scale_buffer_bytes, pack_bf16_swiglu_weight_host, GemmArgs, GemmBiasArgs,
    GemmBiasGeluArgs, GemmBiasResidualArgs, GemmGegluArgs, GemmPolicy, GemmQuantization,
    GemmSwiGluArgs, ScaleLayout, WeightVersion,
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
pub use rope::apply_vision_2d as rope_apply_vision_2d;
pub use rope::{
    decode_rope, execute_batched as rope_apply_batched, execute_mrope as rope_apply_mrope,
    multimodal_rope, rope, split_qkv_axial_rope, AxialRopeArgs, DecodeRopeArgs, MultimodalRopeArgs,
    RopeArgs, RopeSemantic,
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

/// Session-aware scratch storage for direct-launch operators.
pub fn scratch_buffer(
    ctx: &crate::CudaContext,
    bytes: usize,
) -> apxinf_core::Result<crate::CudaBuffer> {
    crate::workspace::output_buffer(ctx, bytes)
}

/// Session-aware scratch storage that is explicitly cleared before use.
pub fn scratch_buffer_zeroed(
    ctx: &crate::CudaContext,
    bytes: usize,
) -> apxinf_core::Result<crate::CudaBuffer> {
    crate::workspace::output_buffer_zeroed(ctx, bytes)
}

/// Session-aware scratch storage with only each group's unused tail cleared.
pub fn scratch_buffer_tail_zeroed(
    ctx: &crate::CudaContext,
    groups: usize,
    group_bytes: usize,
    used_bytes: usize,
) -> apxinf_core::Result<crate::CudaBuffer> {
    crate::workspace::output_buffer_tail_zeroed(ctx, groups, group_bytes, used_bytes)
}
#[cfg(test)]
mod tests;
