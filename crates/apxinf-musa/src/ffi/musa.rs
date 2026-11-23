//! Hand-written MUSA runtime and driver bindings.
//!
//! Only the symbols ApxInf actually calls are declared, the same policy as
//! `apxinf-cuda`'s `ffi` module: no bindgen, no cudarc/musa-rs, no generated
//! bindings to keep in sync with a toolkit upgrade.

use std::ffi::c_void;

/// `musaError_t`; `musaSuccess` is 0.
pub type musaError_t = i32;
/// `MUresult`, the driver API's status code; `MUDA_SUCCESS` is 0.
pub type MUresult = i32;
/// `MUdevice` is an int ordinal, not a pointer.
pub type MUdevice = i32;

pub const MUSA_SUCCESS: musaError_t = 0;
pub const MUDA_SUCCESS: MUresult = 0;

pub const MUSA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub const MUSA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
pub const MUSA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

/// `musaDeviceAttr` values used by the backend.
///
/// These are CUDA's `cudaDeviceAttr` numbers. MUSA accepts them — `device_caps`
/// reads M1000's documented 2.2 / 8 SMs through them — but the vendor does not
/// document the correspondence, so this is an observation about the 5.1.0
/// driver rather than a contract. A device reporting implausible capabilities
/// is the first thing to check against `musaInfo` if that ever changes.
pub mod attribute {
    /// The *reported* warp size, 128 on M1000. Shuffle groups are 32 wide
    /// regardless — see [`crate::device_caps::MusaDeviceCaps::hardware_warp_size`].
    pub const WARP_SIZE: i32 = 10;
    pub const MULTIPROCESSOR_COUNT: i32 = 16;
    pub const COMPUTE_CAPABILITY_MAJOR: i32 = 75;
    pub const COMPUTE_CAPABILITY_MINOR: i32 = 76;
}

/// `musaStreamCaptureMode` values. CUDA's numbering, same caveat as
/// [`attribute`]: the vendor does not document the correspondence.
pub mod capture_mode {
    /// Only this thread's operations participate, and this thread may not run
    /// an operation unsafe for capture. CUDA's one-argument
    /// `cudaStreamBeginCapture` defaults here, and it is the strictest of the
    /// three: everything [`crate::graph::capture`] does happens on the capture
    /// thread, so the relaxed variants (`GLOBAL` 0, `RELAXED` 2) buy nothing.
    pub const THREAD_LOCAL: i32 = 1;
}

extern "C" {
    // ── Runtime ──────────────────────────────────────────────────────────
    pub fn musaSetDevice(device: i32) -> musaError_t;
    pub fn musaGetLastError() -> musaError_t;
    pub fn musaGetErrorString(error: musaError_t) -> *const std::ffi::c_char;
    pub fn musaDeviceGetAttribute(value: *mut i32, attribute: i32, device: i32) -> musaError_t;
    pub fn musaDeviceSynchronize() -> musaError_t;
    pub fn musaStreamBeginCapture(stream: *mut c_void, mode: i32) -> musaError_t;
    pub fn musaStreamEndCapture(stream: *mut c_void, graph: *mut *mut c_void) -> musaError_t;
    pub fn musaGraphInstantiate(
        exec: *mut *mut c_void,
        graph: *mut c_void,
        flags: u64,
    ) -> musaError_t;
    pub fn musaGraphLaunch(exec: *mut c_void, stream: *mut c_void) -> musaError_t;
    pub fn musaGraphExecDestroy(exec: *mut c_void) -> musaError_t;
    pub fn musaGraphDestroy(graph: *mut c_void) -> musaError_t;

    pub fn musaMalloc(pointer: *mut *mut c_void, bytes: usize) -> musaError_t;
    pub fn musaFree(pointer: *mut c_void) -> musaError_t;
    pub fn musaMemcpy(
        destination: *mut c_void,
        source: *const c_void,
        bytes: usize,
        kind: i32,
    ) -> musaError_t;
    pub fn musaMemcpyAsync(
        destination: *mut c_void,
        source: *const c_void,
        bytes: usize,
        kind: i32,
        stream: *mut c_void,
    ) -> musaError_t;
    pub fn musaMemset(pointer: *mut c_void, value: i32, bytes: usize) -> musaError_t;
    pub fn musaMemsetAsync(
        pointer: *mut c_void,
        value: i32,
        bytes: usize,
        stream: *mut c_void,
    ) -> musaError_t;

    pub fn musaStreamCreate(stream: *mut *mut c_void) -> musaError_t;
    pub fn musaStreamDestroy(stream: *mut c_void) -> musaError_t;
    pub fn musaStreamSynchronize(stream: *mut c_void) -> musaError_t;

    // ── Library versions ─────────────────────────────────────────────────
    pub fn musaRuntimeGetVersion(version: *mut i32) -> musaError_t;
    /// mublas exposes no version query of its own; only mublasLt does.
    pub fn mublasLtGetVersion() -> usize;

    // ── Driver ───────────────────────────────────────────────────────────
    pub fn muInit(flags: u32) -> MUresult;
    pub fn muDeviceGet(device: *mut MUdevice, ordinal: i32) -> MUresult;
    pub fn muDeviceGetName(name: *mut std::ffi::c_char, length: i32, device: MUdevice) -> MUresult;
}
