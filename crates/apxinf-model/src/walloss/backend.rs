//! CUDA-facing seam for the Walloss runtime.
//!
//! Walloss uses cuda-new's public `ops` surface and runtime backend.

pub(crate) use apxinf_cuda_new::profiler::CudaEventTimer;
pub(crate) use apxinf_cuda_new::{
    ops, transfers, CudaBuffer as DeviceBuffer, CudaContext as Context,
    CudaNewBackend as RuntimeBackend,
};
