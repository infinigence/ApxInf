//! CUDA-facing seam for the SmolVLA runtime.
//!
//! SmolVLA now runs on the cuda-new runtime: `kernels` resolves to the
//! legacy-named shim over cuda-new operators, and `RuntimeBackend` is the
//! cuda-new backend. The model and runtime files are unchanged — the runtime
//! swap happens entirely in these aliases.

#[cfg(feature = "cuda")]
pub(crate) use apxinf_cuda_new::{
    kernels, transfers, CudaBuffer as DeviceBuffer, CudaContext as Context,
    CudaNewBackend as RuntimeBackend,
};
#[cfg(feature = "cuda")]
pub(crate) use apxinf_cuda_new::profiler::CudaEventTimer;
#[cfg(feature = "cuda")]
pub(crate) use crate::accelerator::downcast_cuda_new_arc as downcast_arc;
