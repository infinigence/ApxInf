//! Public, model-neutral CUDA kernel facade, organized by physical operator.
//!
//! Precision and quantization are expressed by function names or by the
//! implementation selected inside an operator module. Each safe contract owns
//! its validation, workspace policy, dispatch, and minimal raw FFI call.

pub mod activation;
pub mod attention;
pub mod cache;
mod contracts;
pub mod convolution;
pub mod elementwise;
pub mod embedding;
pub mod fused;
pub mod gemm;
pub mod linear_attention;
pub mod norm;
pub mod pooling;
pub mod preprocess;
pub mod quantization;
pub mod rope;
pub mod sampling;

pub use crate::workspace::GraphWorkspace;

/// Reset and bind a persistent, stable-address workspace around one run body.
pub fn with_workspace<T>(
    workspace: &GraphWorkspace,
    operation: impl FnOnce() -> apxinf_core::Result<T>,
) -> apxinf_core::Result<T> {
    crate::workspace::with_workspace(workspace, operation)
}

/// Bind a workspace for an eager (non-captured) traversal, keeping native
/// execution resources installable so GEMM plans resolve and autotune as they
/// would without a workspace.
pub fn with_workspace_eager<T>(
    workspace: &GraphWorkspace,
    operation: impl FnOnce() -> apxinf_core::Result<T>,
) -> apxinf_core::Result<T> {
    crate::workspace::with_workspace_eager(workspace, operation)
}

/// Run an eager preflight that prepares native plans and workspace before
/// CUDA graph capture.
pub fn prepare_with_workspace<T>(
    workspace: &GraphWorkspace,
    operation: impl FnOnce() -> apxinf_core::Result<T>,
) -> apxinf_core::Result<T> {
    crate::workspace::prepare_with_workspace(workspace, operation)
}

/// Scratch for a model's own intermediates, from the bound workspace when one
/// is active and from the driver otherwise.
///
/// A model that allocates its intermediates through the driver cannot be
/// captured into a CUDA graph: allocation is illegal during capture, and the
/// addresses would not survive replay. Routing a model's allocation helpers
/// through here makes its body capturable without changing what it does when
/// no workspace is bound, which is the state every caller starts in.
pub fn scratch_buffer(
    ctx: &crate::CudaContext,
    bytes: usize,
) -> apxinf_core::Result<crate::CudaBuffer> {
    crate::workspace::output_buffer(ctx, bytes)
}

/// Zeroed scratch. Inside a workspace the arena is reused, so the clear is
/// explicit rather than implied by a fresh allocation.
pub fn scratch_buffer_zeroed(
    ctx: &crate::CudaContext,
    bytes: usize,
) -> apxinf_core::Result<crate::CudaBuffer> {
    crate::workspace::output_buffer_zeroed(ctx, bytes)
}
