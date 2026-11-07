//! Raw bindings for the MP22 INT8 (w8a8) GEMM tile adapter.
//!
//! The kernels are the rendered muDNN `s8mma` assembly in
//! `kernels/asm/qint8_*.asm` (see that directory's README), loaded as driver
//! modules by `adapters/qint8_gemm_adapter.mu`; that adapter owns the module
//! handles, the 67-argument convolution ABI and the launch shape. This module is
//! only the ABI.
//!
//! As in `ffi::asm_gemm`, there is no bindgen: the surface is two functions and a
//! generated binding would be larger than the thing it describes.

use std::ffi::{c_char, c_void};

extern "C" {
    /// Short-row GEMM: KN weights when `nn == 1`, NK when `nn == 0`.
    /// Both INT8 operands use padded FP16 scales; output is BF16.
    pub fn apxinf_musa_asm_qint8_decode(
        a: *const c_void,
        b: *const c_void,
        d: *mut c_void,
        a_scale: *const c_void,
        b_scale: *const c_void,
        a_scale_bytes: u32,
        b_scale_bytes: u32,
        m: i32,
        n: i32,
        k: i32,
        nn: i32,
        stream: *mut c_void,
    ) -> i32;

    /// One INT8 GEMM tile launch: `[m,k] @ [n,k]^T -> [m,n]` (BF16), with FP16
    /// row/column scales.
    ///
    /// `a` is `[m,k]` row-major INT8, `b` is `[n,k]` row-major INT8
    /// (**output-major**), `d` is `[m,n]` row-major BF16. `a_scale` is `[m]` FP16
    /// and `b_scale` is `[n]` FP16; both carry a 512-element zero pad past their
    /// length, whose byte counts are `a_scale_bytes`/`b_scale_bytes`. The written
    /// value is `a_scale[i] * b_scale[j] * sum_k a[i,k] * b[j,k]`.
    ///
    /// `tile` indexes the adapter's tile table, which is the same order as
    /// `build.rs`'s `QINT8_ASM_VARIANTS` and `kernels::gemm::qint8_tiles::QINT8_TILES`.
    /// The adapter rejects a tile the shape cannot use rather than running
    /// something adjacent.
    ///
    /// Returns 0 on success. On failure it returns non-zero and
    /// [`apxinf_musa_asm_qint8_gemm_last_error`] names the reason; the caller
    /// reports it, because a failure here happens before anything is written.
    pub fn apxinf_musa_asm_qint8_gemm(
        a: *const c_void,
        b: *const c_void,
        d: *mut c_void,
        a_scale: *const c_void,
        b_scale: *const c_void,
        a_scale_bytes: u32,
        b_scale_bytes: u32,
        m: i32,
        n: i32,
        k: i32,
        tile: i32,
        stream: *mut c_void,
    ) -> i32;

    /// Why the last call on this thread failed. A static string.
    pub fn apxinf_musa_asm_qint8_gemm_last_error() -> *const c_char;
}

fn adapter_detail(status: i32) -> String {
    // SAFETY: the adapter returns a pointer to a thread-local static buffer that
    // lives until the next call on this thread.
    let text = unsafe { crate::ffi::error_text(apxinf_musa_asm_qint8_gemm_last_error()) };
    text.unwrap_or_else(|| format!("the asm INT8 GEMM adapter returned {status}"))
}

/// Turn a status into the adapter's message; `operation` runs only on failure,
/// so a successful launch does not format a string it would throw away.
pub fn check_qint8_gemm(status: i32, operation: impl FnOnce() -> String) -> Result<(), String> {
    crate::ffi::check_status(status, || format!("{}: {}", operation(), adapter_detail(status)))
}
