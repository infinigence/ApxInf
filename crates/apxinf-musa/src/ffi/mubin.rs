//! Raw bindings for the MP22 attention assembly adapter.
//!
//! The kernels are hand-written assembly (see `kernels/asm/README.md`), loaded
//! as driver modules by `adapters/mubin_fa_adapter.mu`; that adapter owns the
//! module handles, the device metadata and the split-KV workspace. This module
//! is only the ABI.
//!
//! No bindgen, as in `ffi::mublas`: the surface is two functions, and a
//! generated binding would be larger than the thing it describes.

use std::ffi::{c_char, c_void};

/// Which assembled kernel to run.
///
/// The three images export the same kernel symbol, so the choice reaches the
/// adapter as an argument rather than through a lookup. The discriminants are
/// the adapter's `Variant` enum.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AsmFaVariant {
    /// Any query length, one query head per block over a 32-row tile.
    General = 0,
    /// Single batch, exactly ten query rows: the token and head dimensions are
    /// flattened into 80 MMA rows and D=256 into four output partitions.
    PackedDecode = 1,
    /// As `PackedDecode`, with the KV axis cut into four and merged afterwards.
    SplitKv = 2,
}

extern "C" {
    /// One packed BF16 attention launch over a dense BF16 KV.
    ///
    /// `tokens` is the query length and Q's first dimension; `capacity` is the
    /// number of KV rows present and `key_tokens` how many of them take part.
    /// The scale is fixed at `1/sqrt(256)` inside the adapter.
    ///
    /// Returns 0 on success. On failure it returns non-zero and
    /// [`apxinf_musa_fa_d256_last_error`] names the reason; there is no
    /// fallback between variants, so a caller that asked for one and got
    /// another cannot happen silently.
    pub fn apxinf_musa_fa_d256(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        tokens: i32,
        capacity: i32,
        key_tokens: i32,
        variant: i32,
        stream: *mut c_void,
    ) -> i32;

    /// Why the last call on this thread failed. A static string.
    pub fn apxinf_musa_fa_d256_last_error() -> *const c_char;
}

/// Turn a status into the adapter's message, naming the operator.
pub fn check_fa_d256(status: i32, operation: &str) -> Result<(), String> {
    crate::ffi::check_status(status, || {
        // SAFETY: the adapter returns a pointer to a thread-local static buffer
        // that lives until the next call on this thread.
        let text = unsafe { crate::ffi::error_text(apxinf_musa_fa_d256_last_error()) };
        format!(
            "{operation}: {}",
            text.unwrap_or_else(|| format!("the attention adapter returned {status}"))
        )
    })
}

// MP22 BF16 GEMM and load-time KN -> NK packing. All launches use the supplied
// engine stream; modules must have been warmed before capture.
extern "C" {
    pub fn apxinf_musa_asm_encode_variant(m: i32, n: i32, k: i32) -> i32;
    pub fn apxinf_musa_asm_gemm(
        a: *const c_void,
        b: *const c_void,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        nt: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn apxinf_musa_asm_transpose(
        input: *const c_void,
        out: *mut c_void,
        k: i32,
        n: i32,
        stream: *mut c_void,
    ) -> i32;
    pub fn apxinf_musa_asm_gemm_last_error() -> *const c_char;
}

/// Turn the assembly GEMM adapter's status into a message naming `operation`.
///
/// The same adapter answers the transpose packing step, so the caller names
/// which launch failed rather than the message hardcoding one of them.
pub fn check_asm_gemm(status: i32, operation: &str) -> Result<(), String> {
    crate::ffi::check_status(status, || {
        // SAFETY: the adapter returns a thread-local static error buffer that
        // stays valid until the next call on this thread.
        let detail = unsafe { crate::ffi::error_text(apxinf_musa_asm_gemm_last_error()) }
            .unwrap_or_else(|| format!("the asm GEMM adapter returned {status}"));
        format!("{operation}: {detail}")
    })
}
