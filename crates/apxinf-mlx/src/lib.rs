//! Native MLX arrays and backend. Enable `native` with a pinned MLX_ROOT SDK.
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod array;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod backend;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod device_caps;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod ffi;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
pub use array::{
    clear_cache, load_safetensors, memory_stats, reset_peak_memory, Array, Compiled, MemoryStats,
    MetalKernel, MlxDType, Stream, StreamCounters, TemplateArg,
};
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
pub use backend::MlxBackend;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
pub use device_caps::MetalDeviceCaps;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
pub mod fusions;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod gated_delta;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod half_split_rope;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod interleaved_qk_norm_rope;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod mpp_gemm;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod q4_f16;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod qk_norm_rope;
