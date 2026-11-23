//! Raw bindings for the mublas GEMM adapter.
//!
//! The adapter owns the handle and the launch (`adapters/mublas_adapter.mu`);
//! this module is only the ABI, plus the one vendor symbol that turns a status
//! into words.
//!
//! No bindgen and no vendor crate, as in `ffi::musa`: the surface is two
//! functions, and a generated binding would be larger than the thing it
//! describes.

use std::ffi::{c_char, c_void};

/// `mublasStatus_t`. `MUBLAS_STATUS_SUCCESS` is 0, and every failure the
/// adapter can report is one of the library's own codes rather than a private
/// channel, so the string from [`mublasGetStatusString`] is the whole
/// message.
pub type mublasStatus_t = i32;

extern "C" {
    /// `[m,k] @ [k,n] -> [m,n]`, BF16 in and out, f32 accumulators, one
    /// rounding at the output.
    ///
    /// Every pointer is a device address and `stream` is a `musaStream_t`. The
    /// adapter keys its handle by stream, because `mublasGemmEx` takes no
    /// stream of its own -- naming the wrong one is
    /// neither a compile error nor a status code, it just orders the GEMM
    /// against nothing.
    pub fn apxinf_musa_bf16_gemm(
        activation: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        stream: *mut c_void,
    ) -> mublasStatus_t;

    /// mublas's own name for a status. Returns a static string.
    pub fn mublasGetStatusString(status: mublasStatus_t) -> *const c_char;

    /// A strided batch of column-major f32 GEMMs, in mublas's argument
    /// convention rather than the row-major one above: `m`, `n`, `k` and the
    /// leading dimensions describe the column-major operands. Strides are in
    /// elements, and `trans_a`/`trans_b` are non-zero for `MUBLAS_OP_T`.
    ///
    /// The vision attention needs this because its operands are
    /// `[tokens, heads, 72]` — contiguous in neither the contraction nor the
    /// batch, so the row-major entry point cannot express them.
    pub fn apxinf_musa_f32_gemm_batched_colmajor(
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        trans_a: i32,
        trans_b: i32,
        lda: i32,
        ldb: i32,
        ldc: i32,
        stride_a: i64,
        stride_b: i64,
        stride_c: i64,
        batch_count: i32,
        alpha: f32,
        stream: *mut c_void,
    ) -> mublasStatus_t;

    /// The same strided batch with BF16 A and B and an f32 C, which is the
    /// shape the vision attention's scores product wants: its operands come out
    /// of the QKV split already BF16, and the library does not mix operand
    /// types, so an f32 C is what lets them stay that way.
    pub fn apxinf_musa_bf16in_f32out_gemm_batched_colmajor(
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        trans_a: i32,
        trans_b: i32,
        lda: i32,
        ldb: i32,
        ldc: i32,
        stride_a: i64,
        stride_b: i64,
        stride_c: i64,
        batch_count: i32,
        alpha: f32,
        stream: *mut c_void,
    ) -> mublasStatus_t;
}

/// Turn a status into a readable error, as `ffi::check_runtime` does for the
/// runtime.
pub fn check_mublas(status: mublasStatus_t) -> Result<(), String> {
    crate::ffi::check_status(status, || {
        // SAFETY: `mublasGetStatusString` returns a static string for any status.
        let text = unsafe { crate::ffi::error_text(mublasGetStatusString(status)) };
        format!("mublas error {status}: {}", text.unwrap_or_default())
    })
}
