//! Compile-time backend seam for π0-FAST executors.
//!
//! Executor code depends on this model-local alias and the model-neutral
//! kernel contract. Adding another accelerator backend changes this seam,
//! not the layer topology.
//!
//! π0-FAST now runs on the cuda-new runtime: `kernels` resolves to the
//! legacy-named shim over cuda-new operators, and `RuntimeBackend` is the
//! cuda-new backend. The executor and runtime files are unchanged — the
//! runtime swap happens entirely in these aliases.

pub(crate) use apxinf_cuda_new::kernels;
pub(crate) use apxinf_cuda_new::kernels::preprocess::ImageLayout;
pub(crate) use apxinf_cuda_new::{
    CudaBuffer as DeviceBuffer, CudaContext as Context, CudaNewBackend as RuntimeBackend,
};
