//! Public, model-neutral CUDA kernel facade, organized by physical operator.
//!
//! Precision and quantization are expressed by function names or by the
//! implementation selected inside an operator module. Each safe contract owns
//! its validation, workspace policy, dispatch, and minimal raw FFI call.

pub mod activation;
pub mod attention;
pub mod cache;
mod contracts;
pub mod elementwise;
pub mod embedding;
pub mod fused;
pub mod gdn;
pub mod gemm;
pub mod norm;
pub mod preprocess;
pub mod quantization;
pub mod qwen35_attention;
pub mod qwen35_common;
pub mod rope;

pub use crate::workspace::GraphWorkspace;

/// Reset and bind a persistent, stable-address workspace around one run body.
pub fn with_workspace<T>(
    workspace: &GraphWorkspace,
    operation: impl FnOnce() -> apxinf_core::Result<T>,
) -> apxinf_core::Result<T> {
    crate::workspace::with_workspace(workspace, operation)
}

/// Run an eager preflight that prepares native plans and workspace before
/// CUDA graph capture.
pub fn prepare_with_workspace<T>(
    workspace: &GraphWorkspace,
    operation: impl FnOnce() -> apxinf_core::Result<T>,
) -> apxinf_core::Result<T> {
    crate::workspace::prepare_with_workspace(workspace, operation)
}
