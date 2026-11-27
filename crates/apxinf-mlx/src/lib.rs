//! Native MLX arrays and backend. Enable `native` with a pinned MLX_ROOT SDK.
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod ane_tiles;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod array;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod backend;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
pub mod banded_attn;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod causal_conv;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod device_caps;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
pub mod exact_gemv;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod ffi;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
pub mod loop_glue;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
pub mod moe_qmv;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod quantized;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
pub use quantized::Quantized;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
pub use array::DEFAULT_CACHE_LIMIT_BYTES;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
pub use array::{
    clear_cache, load_safetensors, loaded_library_path, memory_stats, reset_peak_memory,
    runtime_version, set_cache_limit, set_memory_limit, Array, Compiled, MemoryStats, MetalKernel,
    MlxDType, QuantMode, Stream, StreamCounters, TemplateArg,
};
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
pub use backend::{greedy_token_arrays, MlxBackend};
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
pub use device_caps::MetalDeviceCaps;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
pub mod fusions;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod gate_swiglu;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod gated_delta;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod half_split_rope;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod interleaved_qk_norm_rope;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod mpp_f16_gemm;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod mpp_gemm;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod q4_f16;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod qk_norm_rope;
#[cfg(feature = "signpost")]
pub mod signpost;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod swiglu_gemv;
