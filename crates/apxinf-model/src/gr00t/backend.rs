//! CUDA-facing seam for the GR00T runtime.
//!
//! GR00T uses cuda-new's public `ops` surface and runtime backend.

#[cfg(feature = "cuda")]
pub(crate) use crate::accelerator::downcast_cuda_new_arc as downcast_arc;
#[cfg(feature = "cuda")]
pub(crate) use apxinf_cuda_new::CudaNewBackend as CudaBackend;
#[cfg(feature = "cuda")]
pub(crate) use apxinf_cuda_new::{
    ops, transfers, CudaBuffer as DeviceBuffer, CudaContext as Context,
    CudaNewBackend as RuntimeBackend,
};
