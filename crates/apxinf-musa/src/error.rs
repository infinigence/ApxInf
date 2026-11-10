//! Error mapping for the MUSA backend.

use apxinf_core::Error;

/// Wrap a MUSA runtime or driver failure.
///
/// The message already names the failing call and its status (see
/// [`crate::ffi::runtime_error`]), so this only adds the backend tag.
///
/// Why `Error::Other` and not a dedicated variant: `apxinf_core::Error` has a
/// `Cuda(String)` variant the CUDA backend maps onto, and a symmetric
/// `Musa(String)` would be the tidier counterpart. Adding it means auditing
/// every `match` over `Error` in the workspace, which is outside this port's
/// scope — so MUSA failures are distinguishable by the `MUSA:` prefix today,
/// and promoting them to a variant is a deliberate follow-up rather than a
/// silent omission.
pub(crate) fn musa_error(message: impl Into<String>) -> Error {
    Error::Other(format!("MUSA: {}", message.into()))
}
