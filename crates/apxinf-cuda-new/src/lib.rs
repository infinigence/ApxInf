pub mod backend;
pub mod buffer;
pub mod context;
pub mod cublas;
pub mod device_caps;
mod ffi;
mod graph;
mod kv_cache;
pub mod nvtx;
pub mod phase;
pub mod profiler;
pub mod runtime;
pub mod sampling;
pub mod stream;
pub mod transfers;
mod workspace;

#[cfg(test)]
pub(crate) static CUDA_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub use backend::CudaNewBackend;
pub use buffer::{CudaBuffer, CudaDeviceAddress, HostMappedBuffer};
pub use context::CudaContext;
pub use cublas::{CublasHandle, CublasTranspose};
pub use device_caps::{CudaArchFamily, CudaDeviceCaps};
pub use graph::{capture, CapturedGraph};
pub use kv_cache::CudaKvCache;
pub use ops::norm::norm;
pub use ops::{
    attention, concat_rows, decode_rope, decode_rotation, fused_add_norm, gather,
    kv_cache_attention, multimodal_rope, pointwise, quantization, reserve_prefix, residual_add,
    rope, segmented_attention, split_grouped_qkv, split_qkv_axial_rope, split_qkv_cache,
    split_qkv_mrope_cache, vision_position, Affine, ArgmaxArgs, AttentionArgs, AttentionInput,
    AttentionMask, AttentionPolicy, AxialRopeArgs, AxialRopeNumerics, BiasOrder, DecodeRopeArgs,
    DecodeRotationArgs, DecodeRotationPosition, EmbeddingScale, ExecutionSession, FusedAddNormArgs,
    GatherArgs, GatherPatchGeometry, GatherSemantic, GraphWorkspace, GroupedQkvRopeArgs,
    KvCacheAttentionArgs, KvCacheDecodeMeta, MropeCacheArgs, MultimodalRopeArgs, NormArgs,
    NormKind, NormNumerics, PointwiseActivation, PointwiseArgs, PointwiseSemantic,
    QkvCacheRopeArgs, QuantOut, QuantizationArgs, QuantizationSemantic, ResidualAddArgs, RopeArgs,
    RopeSemantic, SegmentedAttentionArgs, VisionPositionArgs,
};
pub use phase::PreparedPhase;
pub use stream::CudaStream;

pub mod ops;

/// Whether the SM110 native NVFP4 GEMM+SwiGLU implementation is available.
///
/// The name describes the mathematical operator rather than its current model
/// consumer. Unsupported targets use the unfused GEMM + SwiGLU fallback.
pub const NVFP4_GEMM_SWIGLU: bool = cfg!(apxinf_nvfp4_gemm_swiglu);

/// Whether this build compiled the allocation-free FA2 decode kernel
/// (`apxinf_decode_attention`). When `false` that ABI returns
/// `APXINF_STATUS_UNSUPPORTED`, so callers must use the generic attention path.
///
/// Like [`NVFP4_GEMM_SWIGLU`], this is a zero-cost compile-time constant
/// carrying a build-time cfg across the crate boundary.
pub const FA2_DECODE: bool = cfg!(apxinf_fa2_decode);
