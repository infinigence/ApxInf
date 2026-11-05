//! Public, model-neutral MUSA operator facade, organized by physical operator.
//!
//! Same contracts as `apxinf-cuda/src/kernels`, so model code written against
//! `crate::accelerator::gpu::kernels` compiles against either backend. Each
//! operator owns its validation, workspace policy, and dispatch; the dispatch
//! is a native kernel or mublas call, and an operator without one reports
//! [`not_ported`] by name.
//!
//! Operators that no supported model reaches yet report
//! [`not_ported`] rather than pretending to work.

use apxinf_core::{Error, Result};

pub mod activation;
pub mod attention;
pub mod cache;
mod contracts;
#[cfg(test)]
pub(crate) mod test_support;
pub mod elementwise;
pub mod embedding;
pub mod fused;
pub mod gemm;
pub mod norm;
pub mod preprocess;
pub mod rope;

pub use crate::workspace::{ExecutionSession, GraphWorkspace};

/// The error a facade entry point returns when the operator has no MUSA path.
///
/// `Error::UnsupportedOp` is core's own classification for exactly this, and
/// the other backends (`apxinf-mlx`, `apxinf-hip`) already answer with it, so a
/// caller can tell "this backend does not implement that" from a failure
/// without reading the message.
pub(crate) fn not_ported(operation: &'static str) -> Error {
    Error::UnsupportedOp(operation)
}

/// Whether an operator may substitute a tactic when nothing covers its exact
/// shape.
///
/// `AllowFallback` is the default for every kernel entry point. `RequireExact`
/// is a caller's `allow_fallback = false`: the dispatcher must find a record
/// measured at this exact shape (or measure one) instead of extrapolating from
/// a neighbouring record or silently using the vendor default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TacticDispatch {
    AllowFallback,
    RequireExact,
}

impl TacticDispatch {
    pub(crate) const fn requires_exact(self) -> bool {
        matches!(self, Self::RequireExact)
    }
}

/// Map a native kernel's C ABI status into an error, naming the operator.
///
/// Every native operator in this crate reaches its kernel the same way, so the
/// two lines it takes are shared rather than repeated thirteen times.
#[cfg(apxinf_musa_available)]
pub(crate) fn check_native(
    status: crate::ffi::musa::musaError_t,
    operation: &str,
) -> Result<()> {
    crate::ffi::check_runtime(status)
        .map_err(|message| crate::error::musa_error(format!("{operation}: {message}")))
}

/// The error a native operator reports in a build with no MUSA toolkit.
///
/// Named rather than silent, and never falling back to a host path: a
/// toolkit-less build exists so the workspace compiles everywhere, and an
/// operator that "worked" there would report a capability the binary does not
/// have.
#[cfg(not(apxinf_musa_available))]
pub(crate) fn native_unavailable(operation: &str) -> Error {
    crate::error::musa_error(format!(
        "{operation} is a native MUSA kernel, but this build found no MUSA toolkit \
         (APXINF_MUSA_ARCH={}); set MUSA_PATH and rebuild",
        crate::MUSA_ARCH
    ))
}

/// Scratch for a model's own intermediates, from the bound workspace when one
/// is active and from the driver otherwise.
///
/// Routing a model's allocation helpers through here is what makes its body
/// capturable later without changing what it does when no workspace is bound,
/// which is the state every caller starts in.
pub fn scratch_buffer(ctx: &crate::MusaContext, bytes: usize) -> Result<crate::MusaBuffer> {
    crate::workspace::output_buffer(ctx, bytes)
}

/// Zeroed scratch. Inside a workspace the arena is reused, so the clear is
/// explicit rather than implied by a fresh allocation.
pub fn scratch_buffer_zeroed(
    ctx: &crate::MusaContext,
    bytes: usize,
) -> Result<crate::MusaBuffer> {
    crate::workspace::output_buffer_zeroed(ctx, bytes)
}
