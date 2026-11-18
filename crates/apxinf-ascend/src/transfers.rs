//! Host ↔ Device memory transfers.
//!
//! Wraps `aclrtMemcpy` to move tensor data between CPU and NPU.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::ffi::acl::{self, aclrtMemcpyKind, aclrtStream};

// Global diagnostics for weight upload timing
static H2D_MALLOC_US: AtomicU64 = AtomicU64::new(0);
static H2D_COPY_US: AtomicU64 = AtomicU64::new(0);
static H2D_COUNT: AtomicU64 = AtomicU64::new(0);
static H2D_BYTES: AtomicU64 = AtomicU64::new(0);

/// Reset H2D diagnostics.
pub fn reset_h2d_diagnostics() {
    H2D_MALLOC_US.store(0, Ordering::Relaxed);
    H2D_COPY_US.store(0, Ordering::Relaxed);
    H2D_COUNT.store(0, Ordering::Relaxed);
    H2D_BYTES.store(0, Ordering::Relaxed);
}

/// Print H2D diagnostics to stderr.
pub fn print_h2d_diagnostics() {
    let malloc_ms = H2D_MALLOC_US.load(Ordering::Relaxed) as f64 / 1000.0;
    let copy_ms = H2D_COPY_US.load(Ordering::Relaxed) as f64 / 1000.0;
    let count = H2D_COUNT.load(Ordering::Relaxed);
    let bytes = H2D_BYTES.load(Ordering::Relaxed);
    eprintln!(
        "[diag-h2d] count={count} bytes={:.1}MB | malloc={malloc_ms:.1}ms copy={copy_ms:.1}ms total={:.1}ms",
        bytes as f64 / (1024.0 * 1024.0),
        malloc_ms + copy_ms,
    );
}

/// Copy `len` bytes from host memory to device memory.
///
/// `dst_dev` is a raw device pointer (from `AscendBuffer::raw_ptr`).
/// `src_host` is a host pointer (from a `&[u8]` slice).
pub fn h2d(dst_dev: *mut c_void, src_host: *const c_void, len: usize) -> Result<(), String> {
    crate::context::restore_context();
    let t0 = Instant::now();
    let ret = unsafe {
        acl::aclrtMemcpy(
            dst_dev,
            len,
            src_host,
            len,
            aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE,
        )
    };
    let copy_us = t0.elapsed().as_micros() as u64;
    H2D_COPY_US.fetch_add(copy_us, Ordering::Relaxed);
    H2D_COUNT.fetch_add(1, Ordering::Relaxed);
    H2D_BYTES.fetch_add(len as u64, Ordering::Relaxed);
    acl::check(ret, "aclrtMemcpy H2D")
}

/// Record malloc timing for diagnostics.
pub fn record_malloc_us(us: u64) {
    H2D_MALLOC_US.fetch_add(us, Ordering::Relaxed);
}

/// Asynchronous copy from host to device on a stream.
///
/// Does not block the host. Caller must `aclrtSynchronizeStream` before
/// using the destination buffer.
pub fn h2d_async(
    dst_dev: *mut c_void,
    src_host: *const c_void,
    len: usize,
    stream: aclrtStream,
) -> Result<(), String> {
    crate::context::restore_context();
    let ret = unsafe {
        acl::aclrtMemcpyAsync(
            dst_dev,
            len,
            src_host,
            len,
            aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE,
            stream,
        )
    };
    H2D_COUNT.fetch_add(1, Ordering::Relaxed);
    H2D_BYTES.fetch_add(len as u64, Ordering::Relaxed);
    acl::check(ret, "aclrtMemcpyAsync H2D")
}

/// Copy `len` bytes from device memory to host memory.
///
/// `dst_host` is a writable host buffer.
/// `src_dev` is a raw device pointer.
pub fn d2h(dst_host: *mut c_void, src_dev: *const c_void, len: usize) -> Result<(), String> {
    crate::context::restore_context();
    let ret = unsafe {
        acl::aclrtMemcpy(
            dst_host,
            len,
            src_dev,
            len,
            aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_HOST,
        )
    };
    acl::check(ret, "aclrtMemcpy D2H")
}

/// Copy `len` bytes between device buffers (same device).
pub fn d2d(dst_dev: *mut c_void, src_dev: *const c_void, len: usize) -> Result<(), String> {
    crate::context::restore_context();
    let ret = unsafe {
        acl::aclrtMemcpy(
            dst_dev,
            len,
            src_dev,
            len,
            aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE,
        )
    };
    acl::check(ret, "aclrtMemcpy D2D")
}
