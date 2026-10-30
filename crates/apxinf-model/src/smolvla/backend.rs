//! CUDA-facing seam for the SmolVLA runtime.
//!
//! SmolVLA uses cuda-new's public `ops` surface and runtime backend.

#[cfg(feature = "cuda")]
pub(crate) use apxinf_cuda_new::{
    ops, transfers, CudaBuffer as DeviceBuffer, CudaContext as Context,
    CudaNewBackend as RuntimeBackend,
};
#[cfg(feature = "cuda")]
pub(crate) use apxinf_cuda_new::profiler::CudaEventTimer;
#[cfg(feature = "cuda")]
pub(crate) use crate::accelerator::downcast_cuda_new_arc as downcast_arc;
