//! Compile-time backend seam for the Qwen-Drive model family.
//!
//! Executor code depends on this model-local alias and the model-neutral
//! kernel contract; adding another accelerator backend changes this seam,
//! not the layer topology.
//!
//! Qwen-Drive now runs on the cuda-new runtime: `kernels` resolves to the
//! legacy-named shim over cuda-new operators, and `RuntimeBackend` is the
//! cuda-new backend. The executor and runner files are unchanged — the
//! runtime swap happens entirely in these aliases.

pub(crate) use apxinf_cuda_new::{
    kernels, nvtx, transfers, tuning, CublasTranspose, CudaBuffer as DeviceBuffer,
    CudaContext as Context, CudaNewBackend as RuntimeBackend,
};

pub(crate) use crate::accelerator::downcast_cuda_new_arc as downcast_arc;
