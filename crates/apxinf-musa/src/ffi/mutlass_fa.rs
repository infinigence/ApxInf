//! Raw bindings for the vendored mp_31 D256 FMHA kernels' C ABI.
//!
//! The symbols are defined in `adapters/mutlass_fa_adapter.mu`, which owns the
//! metadata cache and the launch validation; the device code it drives is
//! `kernels/mutlass_fmha/fmha_fwd/*.mu`, vendored from MATE. As in `ffi::musa`,
//! only the symbols ApxInf calls are declared, by hand.
//!
//! Unlike `ffi::custom`, the definitions are absent from most builds: the
//! vendored tree is compiled only for mp_31 on a toolkit the render was
//! measured against (`build.rs` gates it, and `kernels::attention` gates the
//! call sites on the same `apxinf_musa_mutlass_fmha` cfg). An unreferenced
//! `extern` declaration costs nothing at link time, so the declarations here
//! are unconditional while the calls are not.
//!
//! This file is also one of the kernel-build-id inputs (`build.rs`'s
//! `KERNEL_INPUT_FILES`), so a signature change here re-keys the build even
//! when no `.mu` moved.

use std::ffi::c_void;

/// Which vendored instantiation of the D256 attention to run.
///
/// The two tile the same arithmetic differently, and MATE's own rule picks
/// between them from the query length; `kernels::attention::mp31_fa_variant`
/// is that rule. They are variants rather than an automatic choice here for the
/// same reason the MP22 assembly has three: which kernel ran has to be legible
/// in a report, not inferred from a shape.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutlassFaVariant {
    /// MATE 0.2.7 config `fmha_fwd_4386cd8f…`: `TileM` 128, packed GQA, two
    /// QK/PV consumer warpgroups. MATE's tile rule selects it for 9 to 16 query
    /// rows, which is PI0.5's ten-row action pass.
    PackedDecodeTile128 = 0,
    /// MATE 0.2.7 config `fmha_fwd_be3f5142…`: `TileM` 192, unpacked, three
    /// consumer warpgroups. Selected for 257 or more query rows, which is
    /// PI0.5's prefix pass; every longer prefix lands on the same instantiation.
    GeneralTile192 = 1,
}

extern "C" {
    /// Packed BF16 non-causal attention over a dense BF16 KV.
    ///
    /// `q` and `out` are `[tokens, 8, 256]` contiguous BF16; `k` and `v` are
    /// `[key_tokens, 256]` contiguous BF16, of which all rows take part. The
    /// scale is `1/sqrt(256)` and is not a parameter.
    ///
    /// Returns 0 on success. On failure it returns non-zero and
    /// `apxinf_musa_mutlass_fa_d256_last_error` names the reason; the launch is
    /// asynchronous on `stream`.
    pub fn apxinf_musa_mutlass_fa_d256(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        tokens: i32,
        key_tokens: i32,
        variant: i32,
        stream: *mut c_void,
    ) -> i32;

    /// The message behind the last non-zero return, as a static string.
    pub fn apxinf_musa_mutlass_fa_d256_last_error() -> *const std::os::raw::c_char;
}

/// Turn a status into the adapter's message, naming the operator.
pub fn check_mutlass_fa(status: i32, operation: &str) -> Result<(), String> {
    crate::ffi::check_status(status, || {
        // SAFETY: the adapter returns a pointer to a thread-local `char` array
        // that lives at least until the next call on this thread, and this
        // reads it before any call can happen.
        let detail = unsafe { crate::ffi::error_text(apxinf_musa_mutlass_fa_d256_last_error()) }
            .filter(|detail| !detail.is_empty())
            .unwrap_or_else(|| format!("the mutlass FA adapter returned {status}"));
        format!("MUSA {operation}: {detail}")
    })
}
