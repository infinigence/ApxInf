//! Raw bindings for the bit-exact packed-BF16 weight adapter.

use std::ffi::c_void;

use super::cuda::{cudaError_t, cudaStream_t};

extern "C" {
    /// Pack a `[k, n]` row-major BF16 weight into the four `[n, k]` planes.
    /// `reject` is a device `i32` the kernel sets to 1 if any 128-weight block
    /// has an exponent range wider than five bits, in which case the packed
    /// form is not a faithful copy and the caller must discard it.
    pub fn apxinf_pack_bf16_transposed(
        src: *const c_void,
        lo: *mut c_void,
        off_lo: *mut c_void,
        off_hi: *mut c_void,
        base: *mut c_void,
        reject: *mut c_void,
        k: i32,
        n: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    /// `y[n] = sum_k w[n][k] * x[k]` over the packed planes.
    pub fn apxinf_packed_gemv_bf16(
        lo: *const c_void,
        off_lo: *const c_void,
        off_hi: *const c_void,
        base: *const c_void,
        x: *const c_void,
        y: *mut c_void,
        n: i32,
        k: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}
