//! ACL Runtime FFI declarations.
//!
//! Signatures follow the CANN 9.1.0 headers in
//! `include/acl/acl.h`, `include/acl/acl_rt.h`, and `include/acl/acl_mdl.h`.
//! All functions return `aclError` (i32), where 0 = ACL_SUCCESS.

use std::ffi::c_void;
use std::os::raw::{c_char, c_int, c_uint};

// ── Type aliases ──────────────────────────────────────────────────────

/// ACL error code. 0 = success.
pub type aclError = c_int;

/// Opaque ACL context handle.
pub type aclrtContext = *mut c_void;

/// Opaque ACL stream handle.
pub type aclrtStream = *mut c_void;

/// Opaque ACL event handle.
pub type aclrtEvent = *mut c_void;

/// Opaque ACL model ID (from `aclmdlLoadFromMem` etc.).
pub type aclmdlModelId = c_uint;

/// Opaque ACL RI (Reusable Instance) handle — captured graph.
pub type aclmdlRI = *mut c_void;

/// Memory allocation policy for `aclrtMalloc`.
#[repr(u32)]
pub enum aclrtMemMallocPolicy {
    /// Try huge-page first, fall back to normal.
    ACL_MEM_MALLOC_HUGE_FIRST = 0,
    /// Only huge-page.
    ACL_MEM_MALLOC_HUGE_ONLY = 1,
    /// Only normal-page.
    ACL_MEM_MALLOC_NORMAL_ONLY = 2,
}

/// Memory attribute for `aclrtGetMemInfo`.
#[repr(u32)]
pub enum aclrtMemAttr {
    /// Default DDR memory.
    ACL_DDR_MEM = 0,
    /// HBM memory.
    ACL_HBM_MEM = 1,
    /// Deep query of DDR memory.
    ACL_DDR_MEM_DEEP = 2,
    /// Deep query of HBM memory.
    ACL_HBM_MEM_DEEP = 3,
}

/// Memory copy direction for `aclrtMemcpy`.
#[repr(u32)]
pub enum aclrtMemcpyKind {
    /// Host → Host
    ACL_MEMCPY_HOST_TO_HOST = 0,
    /// Host → Device
    ACL_MEMCPY_HOST_TO_DEVICE = 1,
    /// Device → Host
    ACL_MEMCPY_DEVICE_TO_HOST = 2,
    /// Device → Device
    ACL_MEMCPY_DEVICE_TO_DEVICE = 3,
}

// ── Initialization / Finalization ─────────────────────────────────────

extern "C" {
    /// Initialize ACL. `config_path` may be null for default config.
    pub fn aclInit(config_path: *const c_char) -> aclError;

    /// Finalize ACL. Must be called exactly once after all other ACL calls.
    pub fn aclFinalize() -> aclError;
}

// ── Device management ─────────────────────────────────────────────────

extern "C" {
    /// Query the number of available NPU devices.
    pub fn aclrtGetDeviceCount(count: *mut u32) -> aclError;

    /// Set the current device for subsequent runtime calls.
    pub fn aclrtSetDevice(device_id: c_int) -> aclError;

    /// Reset a device, releasing its resources.
    pub fn aclrtResetDevice(device_id: c_int) -> aclError;

    /// Get the currently set device ID.
    pub fn aclrtGetDevice(device_id: *mut c_int) -> aclError;
}

// ── Context management ────────────────────────────────────────────────

extern "C" {
    /// Create a context on the given device.
    pub fn aclrtCreateContext(context: *mut aclrtContext, device_id: c_int) -> aclError;

    /// Destroy a context.
    pub fn aclrtDestroyContext(context: aclrtContext) -> aclError;

    /// Set the current context for this thread.
    pub fn aclrtSetCurrentContext(context: aclrtContext) -> aclError;

    /// Get the current context.
    pub fn aclrtGetCurrentContext(context: *mut aclrtContext) -> aclError;
}

// ── Stream management ─────────────────────────────────────────────────

extern "C" {
    /// Create a stream on the current context.
    pub fn aclrtCreateStream(stream: *mut aclrtStream) -> aclError;

    /// Destroy a stream.
    pub fn aclrtDestroyStream(stream: aclrtStream) -> aclError;

    /// Block until all ops on a stream complete.
    pub fn aclrtSynchronizeStream(stream: aclrtStream) -> aclError;

    /// Block until all ops on the current device complete.
    pub fn aclrtSynchronizeDevice() -> aclError;
}

// ── Event management ──────────────────────────────────────────────────

extern "C" {
    /// Create an event.
    pub fn aclrtCreateEvent(event: *mut aclrtEvent) -> aclError;

    /// Destroy an event.
    pub fn aclrtDestroyEvent(event: aclrtEvent) -> aclError;

    /// Record an event on a stream.
    pub fn aclrtRecordEvent(event: aclrtEvent, stream: aclrtStream) -> aclError;

    /// Block the host until an event is recorded.
    pub fn aclrtSynchronizeEvent(event: aclrtEvent) -> aclError;

    /// Query whether an event has been recorded (non-blocking).
    /// `status`: 0 = not done, 1 = done.
    pub fn aclrtQueryEvent(event: aclrtEvent, status: *mut u32) -> aclError;
}

// ── Memory management ─────────────────────────────────────────────────

extern "C" {
    /// Allocate device memory.
    pub fn aclrtMalloc(
        dev_ptr: *mut *mut c_void,
        size: usize,
        policy: aclrtMemMallocPolicy,
    ) -> aclError;

    /// Free device memory.
    pub fn aclrtFree(dev_ptr: *mut c_void) -> aclError;

    /// Query device memory free/total for `attr`. Either pointer may be null
    /// if the corresponding value is not needed.
    pub fn aclrtGetMemInfo(attr: aclrtMemAttr, free: *mut usize, total: *mut usize) -> aclError;

    /// Allocate host-pinned memory.
    pub fn aclrtMallocHost(host_ptr: *mut *mut c_void, size: usize) -> aclError;

    /// Free host-pinned memory.
    pub fn aclrtFreeHost(host_ptr: *mut c_void) -> aclError;

    /// Copy memory. `count` is bytes.
    pub fn aclrtMemcpy(
        dst: *mut c_void,
        dest_max: usize,
        src: *const c_void,
        count: usize,
        kind: aclrtMemcpyKind,
    ) -> aclError;

    /// Set device memory to a value. `count` is bytes.
    pub fn aclrtMemset(dst: *mut c_void, dest_max: usize, value: i32, count: usize) -> aclError;

    /// Copy a 2D matrix between host and device.
    pub fn aclrtMemcpy2d(
        dst: *mut c_void,
        dpitch: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: aclrtMemcpyKind,
    ) -> aclError;

    /// Asynchronous memory copy on a stream. `count` is bytes.
    pub fn aclrtMemcpyAsync(
        dst: *mut c_void,
        dest_max: usize,
        src: *const c_void,
        count: usize,
        kind: aclrtMemcpyKind,
        stream: aclrtStream,
    ) -> aclError;
}

// ── Graph capture / replay (aclmdlRI*) ────────────────────────────────

extern "C" {
    /// Begin graph capture on a stream.
    pub fn aclmdlRICaptureBegin(stream: aclrtStream, mode: u32) -> aclError;

    /// End graph capture. Returns an opaque RI handle for the captured graph.
    pub fn aclmdlRICaptureEnd(stream: aclrtStream, model_ri: *mut aclmdlRI) -> aclError;

    /// Execute a captured graph asynchronously on a stream.
    pub fn aclmdlRIExecuteAsync(model_ri: aclmdlRI, stream: aclrtStream) -> aclError;

    /// Destroy a captured graph and free its resources.
    pub fn aclmdlRIDestroy(model_ri: aclmdlRI) -> aclError;
}

// ── Error checking helper ─────────────────────────────────────────────

/// Convert an ACL error code to a Result, mapping non-zero to an error message.
pub fn check(ret: aclError, ctx: &str) -> Result<(), String> {
    if ret == 0 {
        Ok(())
    } else {
        Err(format!("ACL error {ret} in {ctx}"))
    }
}
