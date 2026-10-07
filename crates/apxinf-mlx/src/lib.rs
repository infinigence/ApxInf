//! Native MLX arrays and backend. Enable `native` with a pinned MLX_ROOT SDK.
#[cfg(feature = "native")]
mod array;
#[cfg(feature = "native")]
mod backend;
#[cfg(feature = "native")]
mod ffi;
#[cfg(feature = "native")]
pub use array::{
    clear_cache, memory_stats, reset_peak_memory, Array, Compiled, MemoryStats, MetalKernel,
    MlxDType, Stream, StreamCounters,
};
#[cfg(feature = "native")]
pub use backend::MlxBackend;
#[cfg(feature = "native")]
pub mod fusions;
#[cfg(feature = "native")]
mod qk_norm_rope;
