//! Compile-time backend seam for π0-FAST executors.
//!
//! Executor code depends on this model-local alias and the model-neutral
//! kernel contract. Adding another accelerator backend changes this seam,
//! not the layer topology.
//!
//! π0-FAST uses cuda-new's public `ops` surface and runtime backend.

pub(crate) use apxinf_cuda_new::ops;
pub(crate) use apxinf_cuda_new::ops::preprocess::ImageLayout;
pub(crate) use apxinf_cuda_new::{
    CudaBuffer as DeviceBuffer, CudaContext as Context, CudaNewBackend as RuntimeBackend,
};
