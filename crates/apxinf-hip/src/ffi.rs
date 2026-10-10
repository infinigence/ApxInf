//! The C ABI exported by `kernels/apxinf_hip.hip`.
//!
//! Each function is declared once, through [`hip_abi!`]. With the ROCm runtime
//! compiled in it becomes an `extern "C"` import; without it, a stub returning
//! [`NOT_COMPILED`]. One declaration means the two builds cannot drift apart,
//! and the stub build never references a symbol that would fail to link.

use std::ffi::c_void;

use apxinf_core::{Error, Result};

/// Status the stub ABI returns when the crate was built without ROCm.
pub(crate) const NOT_COMPILED: i32 = -1;

/// Shim dtype codes. Mirror `kDTypeF32` / `kDTypeBF16` in the device code.
pub(crate) const DTYPE_F32: i32 = 0;
pub(crate) const DTYPE_BF16: i32 = 1;

macro_rules! hip_abi {
    ($( fn $name:ident($($arg:ident: $ty:ty),* $(,)?); )*) => {
        #[cfg(apxinf_hip_runtime)]
        extern "C" {
            $( pub(crate) fn $name($($arg: $ty),*) -> i32; )*
        }

        $(
            #[cfg(not(apxinf_hip_runtime))]
            #[allow(unused_variables, clippy::too_many_arguments)]
            pub(crate) unsafe fn $name($($arg: $ty),*) -> i32 {
                NOT_COMPILED
            }
        )*
    };
}

hip_abi! {
    fn apxinf_hip_device_count(count: *mut i32);
    fn apxinf_hip_set_device(device: i32);
    fn apxinf_hip_device_info(
        device: i32,
        arch: *mut u8,
        arch_len: i32,
        name: *mut u8,
        name_len: i32,
        warp_size: *mut i32,
        compute_units: *mut i32,
        total_memory: *mut u64,
        memory_pools: *mut i32,
    );
    fn apxinf_hip_stream_create(stream: *mut *mut c_void);
    fn apxinf_hip_stream_destroy(stream: *mut c_void);
    fn apxinf_hip_stream_synchronize(stream: *mut c_void);
    fn apxinf_hip_malloc(ptr: *mut *mut c_void, bytes: u64, stream_ordered: i32, stream: *mut c_void);
    fn apxinf_hip_free(ptr: *mut c_void, stream_ordered: i32, stream: *mut c_void);
    fn apxinf_hip_memcpy_htod(dst: *mut c_void, src: *const c_void, bytes: u64, stream: *mut c_void);
    fn apxinf_hip_memcpy_dtoh(dst: *mut c_void, src: *const c_void, bytes: u64, stream: *mut c_void);
    fn apxinf_hip_memcpy_dtod(dst: *mut c_void, src: *const c_void, bytes: u64, stream: *mut c_void);
    fn apxinf_hip_error_string(code: i32, buf: *mut u8, len: i32);

    fn apxinf_hip_blas_create(handle: *mut *mut c_void, stream: *mut c_void);
    fn apxinf_hip_blas_destroy(handle: *mut c_void);
    fn apxinf_hip_gemm(
        handle: *mut c_void,
        dtype: i32,
        m: i32,
        n: i32,
        k: i32,
        a: *const c_void,
        b: *const c_void,
        c: *mut c_void,
    );

    fn apxinf_hip_silu(dtype: i32, x: *const c_void, y: *mut c_void, n: i64, stream: *mut c_void);
    fn apxinf_hip_binary(
        dtype: i32,
        op: i32,
        a: *const c_void,
        b: *const c_void,
        y: *mut c_void,
        n: i64,
        stream: *mut c_void,
    );
    fn apxinf_hip_scale(
        dtype: i32,
        x: *const c_void,
        y: *mut c_void,
        n: i64,
        factor: f32,
        stream: *mut c_void,
    );
    fn apxinf_hip_rms_norm(
        dtype: i32,
        x: *const c_void,
        weight: *const c_void,
        y: *mut c_void,
        rows: i64,
        cols: i64,
        eps: f32,
        stream: *mut c_void,
    );
    fn apxinf_hip_rope(
        dtype: i32,
        x: *const c_void,
        y: *mut c_void,
        seq: i32,
        heads: i32,
        head_dim: i32,
        theta: f32,
        pos_offset: u32,
        stream: *mut c_void,
    );
    fn apxinf_hip_embedding(
        element_bytes: i32,
        table: *const c_void,
        ids: *const c_void,
        y: *mut c_void,
        n_ids: i64,
        dim: i64,
        stream: *mut c_void,
    );
    fn apxinf_hip_attention(
        dtype: i32,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        q_len: i32,
        n_heads: i32,
        n_kv_heads: i32,
        head_dim: i32,
        kv_len: i32,
        kv_offset: i32,
        scale: f32,
        stream: *mut c_void,
    );
}

/// Turn a shim status into a `Result`, naming the operation that failed.
pub(crate) fn check(op: &str, status: i32) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    Err(Error::Other(format!(
        "apxinf-hip: {op} failed: {}",
        describe(status)
    )))
}

fn describe(status: i32) -> String {
    if status == NOT_COMPILED {
        return "apxinf-hip was built without ROCm".into();
    }
    let mut buf = [0u8; 256];
    // SAFETY: the shim writes at most `len` bytes, NUL-terminated.
    unsafe { apxinf_hip_error_string(status, buf.as_mut_ptr(), buf.len() as i32) };
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    format!("{} (status {status})", String::from_utf8_lossy(&buf[..end]))
}
