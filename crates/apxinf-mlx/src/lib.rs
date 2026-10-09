//! Native MLX arrays and backend. Enable `native` with a pinned MLX_ROOT SDK.
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod array;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod backend;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod ffi;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
pub use array::{
    clear_cache, memory_stats, reset_peak_memory, Array, Compiled, MemoryStats, MetalKernel,
    MlxDType, Stream, StreamCounters,
};
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
pub use backend::MlxBackend;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
pub mod fusions;
#[cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
mod qk_norm_rope;
