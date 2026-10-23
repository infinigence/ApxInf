//! Compile-time backend seam for the Qwen-Drive model family.
//!
//! Executor code depends on this model-local alias and the model-neutral
//! kernel contract; adding another accelerator backend changes this seam,
//! not the layer topology.
//!
//! Qwen-Drive uses cuda-new's public `ops` surface and runtime backend.

pub(crate) use apxinf_cuda_new::{
    nvtx, ops, transfers, CublasTranspose, CudaBuffer as DeviceBuffer,
    CudaContext as Context, CudaNewBackend as RuntimeBackend,
};

pub(crate) use crate::accelerator::downcast_cuda_new_arc as downcast_arc;
