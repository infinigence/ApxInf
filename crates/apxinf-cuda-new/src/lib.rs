pub mod buffer;
pub mod context;
pub mod device_caps;
mod ffi;
mod graph;
pub mod phase;
pub mod sampling;
pub mod stream;
pub mod transfers;
mod workspace;

pub use buffer::{CudaBuffer, CudaDeviceAddress, HostMappedBuffer};
pub use context::CudaContext;
pub use device_caps::{CudaArchFamily, CudaDeviceCaps};
pub use graph::{capture, CapturedGraph};
pub use phase::PreparedPhase;
pub use ops::{
    ada_gate_residual, ada_gate_residual_rms_norm, adaptive_rms_norm, attention, bias_residual,
    bias_residual_layer_norm, bias_residual_rms_norm, bias_then_residual, concat_rows, decode_rope,
    gather, kv_cache_attention, layer_norm, packed_qkv_attention, pointwise, quantization,
    reserve_prefix, rms_norm, rope, segmented_attention, AdaGateResidualArgs,
    AdaGateResidualRmsNormArgs, AdaptiveRmsNormArgs, AttentionArgs, AttentionMask, AttentionPolicy,
    BiasResidualArgs, BiasResidualLayerNormArgs, BiasResidualRmsNormArgs, BiasThenResidualArgs,
    DecodeRopeArgs, ExecutionSession, GatherArgs, GatherPatchGeometry, GatherSemantic,
    GraphWorkspace, KvCacheAttentionArgs, KvCacheDecodeMeta, LayerNormArgs, PackedQkvAttentionArgs,
    PointwiseActivation, PointwiseArgs, PointwiseSemantic, QuantizationArgs, QuantizationSemantic,
    RmsNormArgs, RopeArgs, RopeSemantic, SegmentedAttentionArgs,
};
pub use stream::CudaStream;

pub mod ops;

/// Whether this build linked the shape-specialized Qwen3.8 NVFP4 dense SwiGLU
/// AOT object (`crates/apxinf-cuda/aot/qwen38.json`, exported through the
/// bundle named by `APXINF_CUDA_AOT_MANIFEST` at build time).
///
/// When `false`, the C ABI `apxinf_gemm_nvfp4_dense_swiglu_aot` is a stub that
/// returns `APXINF_STATUS_UNSUPPORTED`, so callers must take the generic path
/// instead.
///
/// This is a compile-time constant: `build.rs` sets the internal
/// `apxinf_qwen38_dense_swiglu_aot` cfg, which a dependent crate cannot observe
/// directly (cfgs do not cross crate boundaries), so it is folded into a
/// `pub const` that any dependent crate can branch on with zero runtime cost.
pub const QWEN38_DENSE_SWIGLU_AOT: bool = cfg!(apxinf_qwen38_dense_swiglu_aot);

/// Whether this build compiled the allocation-free FA2 decode kernel
/// (`apxinf_decode_attention`). When `false` that ABI returns
/// `APXINF_STATUS_UNSUPPORTED`, so callers must use the generic attention path.
///
/// Like [`QWEN38_DENSE_SWIGLU_AOT`], this is a zero-cost compile-time constant
/// carrying a build-time cfg across the crate boundary.
pub const FA2_DECODE: bool = cfg!(apxinf_fa2_decode);
