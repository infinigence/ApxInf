pub mod backend;
pub mod buffer;
pub mod context;
pub mod cublas;
pub mod device_caps;
/// CUDA runtime and kernel support for apxinf.
///
/// This crate provides:
/// - Safe Rust wrappers (`CudaBuffer`, `CudaStream`, `CublasHandle`)
/// - Custom CUDA kernel wrappers (RMSNorm, RoPE, SiLU, softmax, attention)
/// - GPU matmul via cuBLAS
/// - `CudaBackend` implementing the portable `Backend` trait
mod ffi;
mod graph;
pub mod kernels;
pub mod kv_cache;
pub mod nvtx;
pub mod profiler;
mod sampling;
pub mod stream;
pub mod transfers;
pub mod tuning;
mod workspace;

#[cfg(test)]
pub(crate) mod test_util;
#[cfg(test)]
mod tests;

pub use backend::CudaBackend;
pub use buffer::HostMappedBuffer;
pub use buffer::{CudaBuffer, CudaDeviceAddress};
pub use context::{CudaContext, CudaLibraryVersions};
pub use cublas::{CublasHandle, CublasTranspose};
pub use device_caps::{CudaArchFamily, CudaDeviceCaps};
pub use kv_cache::CudaKVCache;
pub use stream::CudaStream;
