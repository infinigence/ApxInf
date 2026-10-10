//! Legacy-named kernel surface over cuda-new operators.
//!
//! The VLA families (walloss, pi0fast, qwen-drive) compose their executors
//! against the legacy `apxinf_cuda::kernels::*` module — a flat set of
//! fused-transformer helpers plus a `GraphWorkspace` bump arena. Rather than
//! rewrite every call site, this module re-exposes those names backed by
//! cuda-new operators and sessions, so a family migrates by pointing its
//! `backend.rs` seam here instead of at the legacy crate.
//!
//! Every function is a thin adapter: it allocates through the session arena and
//! delegates to a cuda-new operator. The arithmetic is the cuda-new operator's,
//! which is the point — the model keeps its shape and the runtime changes
//! underneath. Functions whose cuda-new operator does not exist yet are
//! deliberately absent; a family that needs one fails to compile until the
//! operator lands, which is the intended signal rather than a silent fallback.

use apxinf_core::Result;

use crate::ops::ExecutionSession;

/// The workspace an executor passes to `prepare_with_workspace` / `with_workspace`.
///
/// cuda-new folds the bump arena and the prepared-execution cache into one
/// [`ExecutionSession`]; the legacy crate split them into `GraphWorkspace` plus
/// free functions. Aliasing collapses the split so executor code is unchanged.
pub use crate::ops::ExecutionSession as GraphWorkspace;

/// Create a workspace with a fixed byte capacity.
///
/// The legacy API named the constructor on the workspace type itself; cuda-new
/// names it on the session. This keeps the call spelling.
pub fn new_workspace(capacity_bytes: usize, device: usize) -> Result<ExecutionSession> {
    ExecutionSession::with_capacity(capacity_bytes, device)
}

pub mod activation;
pub mod attention;
pub mod cache;
pub(crate) mod contracts;
pub mod elementwise;
pub mod embedding;
pub mod fixed_profile;
pub mod fused;
pub mod gdn_policy;
pub mod gemm;
pub mod linear_attention;
pub mod norm;
pub mod pillow_bicubic;
pub mod preprocess;
pub mod quantization;
pub mod rope;
pub mod sampling;

/// Workspace- or driver-backed scratch; capturable inside a session.
pub fn scratch_buffer(
    ctx: &crate::CudaContext,
    bytes: usize,
) -> apxinf_core::Result<crate::CudaBuffer> {
    crate::workspace::output_buffer(ctx, bytes)
}

/// Zeroed scratch. Inside a session the arena is reused, so the clear is
/// explicit rather than implied by a fresh allocation.
pub fn scratch_buffer_zeroed(
    ctx: &crate::CudaContext,
    bytes: usize,
) -> apxinf_core::Result<crate::CudaBuffer> {
    crate::workspace::output_buffer_zeroed(ctx, bytes)
}

/// Scratch for a `[groups, rows, cols]` region whose consumer writes every
/// row a token maps to and leaves the padding rows alone. Only that padding
/// is cleared.
pub fn scratch_buffer_tail_zeroed(
    ctx: &crate::CudaContext,
    groups: usize,
    group_bytes: usize,
    used_bytes: usize,
) -> apxinf_core::Result<crate::CudaBuffer> {
    crate::workspace::output_buffer_tail_zeroed(ctx, groups, group_bytes, used_bytes)
}

/// Prepare a fixed-shape traversal that may allocate and tune.
pub fn prepare_with_workspace<T>(
    session: &ExecutionSession,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    session.prepare(operation)
}

/// Run a prepared traversal without allocating or tuning.
pub fn with_workspace<T>(
    session: &ExecutionSession,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    session.run(operation)
}

/// Run an eager traversal (no capture). cuda-new executes eagerly whenever no
/// capture is active, so this is the same call as `with_workspace`.
pub fn with_workspace_eager<T>(
    session: &ExecutionSession,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    session.run(operation)
}
