//! Semantic CUDA APIs with selection and native state behind L1.

pub mod attention;
pub mod cache;
pub mod elementwise;
pub mod embedding;
pub mod gather;
pub mod gdn;
pub mod gemm;
pub mod norm;
pub mod pillow_bicubic;
pub mod pointwise;
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
    apply_output_gate, split_query_and_gate, OutputGateArgs, OutputGateNumerics, OutputGateSource,
};
pub use attention::{
    attention, decode_attention, decode_attention_workspace_bytes, kv_cache_attention,
    segmented_attention, AttentionArgs, AttentionInput, AttentionMask, AttentionNumerics,
    AttentionPolicy, KvCacheAttentionArgs, KvCacheDecodeMeta, SegmentedAttentionArgs,
};
pub use cache::core::{append as cache_append, concat_rows, copy_tensor, reserve_prefix};
pub use elementwise::add_into;
pub use elementwise::core::{
    activation as elementwise_activation, add as elementwise_add, add_bias as elementwise_add_bias,
    mul as elementwise_mul, scale as elementwise_scale, ElementwiseActivation,
};
pub use gather::{
    embedding_gather, gather, EmbeddingScale, GatherArgs, GatherSemantic,
    PatchGeometry as GatherPatchGeometry,
};
pub use gdn::{
    gdn_causal_conv_forward, gdn_causal_conv_step, gdn_chunk_scan, gdn_continue, gdn_conv_prepare,
    gdn_decay_and_beta, gdn_first_prefill, gdn_gated_norm, gdn_l2_normalize_heads, gdn_prefill,
    gdn_prefill_workspace_bytes, gdn_prepare_prefill, gdn_recurrent_step, gdn_state_elements,
    GdnChunkScanArgs, GdnChunkScanInput, GdnContinuationArgs, GdnDecayAndBetaArgs,
    GdnFirstPrefillArgs, GdnGate, GdnGatedNormArgs, GdnQuantizedOutput,
};
pub use gemm::{
    gemm, gemm_bias, gemm_bias_gelu, gemm_bias_residual, gemm_geglu, gemm_swiglu, gemv, gemv_bias,
    pack_swiglu_weight_host, BlockQuantizedSwiGluArgs, GemmArgs, GemmBiasArgs, GemmBiasGeluArgs,
    GemmBiasResidualArgs, GemmGegluArgs, GemmNumerics, GemmPolicy, GemmQuantization,
    GemmSwiGluArgs, GemmWeightLayout, GemvArgs, GemvBiasArgs, GemvBlockScales, WeightVersion,
};
pub use norm::{
    fused_add_norm, residual_add, Affine, BiasOrder, FusedAddNormArgs, NormArgs, NormKind,
    NormNumerics, QuantOut, ResidualAddArgs, ResidualGate,
};
pub use pointwise::{
    pointwise, swiglu_into, PointwiseActivation, PointwiseArgs, PointwiseSemantic,
};
pub use preprocess::{vision_position, VisionPositionArgs};
pub use quantization::{
    block_quantize, block_scale_buffer_bytes, pack_block_scales, quantization, quantize_per_tensor,
    BlockQuantizationArgs, BlockQuantizationSemantic, PackBlockScalesArgs, QuantizationArgs,
    QuantizationSemantic, ScaleLayout,
};
pub use rope::apply_vision_2d as rope_apply_vision_2d;
pub use rope::{
    decode_rope, decode_rotation, execute_batched as rope_apply_batched,
    execute_mrope as rope_apply_mrope, multimodal_rope, partial_rope, rope, rotary_dim,
    split_grouped_qkv, split_qkv_axial_rope, split_qkv_cache, split_qkv_mrope_cache, AxialRopeArgs,
    AxialRopeNumerics, DecodeRopeArgs, DecodeRotationArgs, DecodeRotationPosition,
    GroupedQkvRopeArgs, MropeCacheArgs, MultimodalRopeArgs, QkvCacheRopeArgs, RopeArgs,
    RopeSemantic,
};
pub use sampling::{argmax, ArgmaxArgs};

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
