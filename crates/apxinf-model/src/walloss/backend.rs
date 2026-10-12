//! CUDA-facing seam for the Walloss runtime.
//!
//! Walloss now runs on the cuda-new runtime: `kernels` resolves to the
//! legacy-named shim over cuda-new operators, and `RuntimeBackend` is the
//! cuda-new backend. The executor and runtime files are unchanged — the
//! runtime swap happens entirely in these aliases.

pub(crate) use apxinf_cuda_new::{
    kernels, transfers, CudaBuffer as DeviceBuffer, CudaContext as Context,
    CudaNewBackend as RuntimeBackend,
};
