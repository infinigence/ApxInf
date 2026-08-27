//! Raw CUDA Runtime and Graph API bindings.
//!
//! These are unsafe extern "C" declarations. Safe wrappers are in sibling modules.

use std::ffi::c_void;

// ── CUDA Runtime types ──────────────────────────────────────────────

pub type cudaError_t = i32;
pub type cudaStream_t = *mut c_void;
pub type cudaEvent_t = *mut c_void;

pub const CUDA_SUCCESS: cudaError_t = 0;

#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum cudaMemcpyKind {
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
}

/// Stream capture mode for cudaStreamBeginCapture.
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2,
}

// Stable values from CUDA's `cudaDeviceAttr` ABI.
pub const CUDA_DEV_ATTR_MULTIPROCESSOR_COUNT: i32 = 16;
pub const CUDA_DEV_ATTR_L2_CACHE_SIZE: i32 = 38;
pub const CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
pub const CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MINOR: i32 = 76;

// Flags for `cudaHostAlloc`. (Outside the extern block — these are values,
// not symbols.)
pub const cudaHostAllocDefault: u32 = 0;
pub const cudaHostAllocPortable: u32 = 1;
pub const cudaHostAllocMapped: u32 = 2;
pub const cudaHostAllocWriteCombined: u32 = 4;

extern "C" {
    pub fn cudaMalloc(devPtr: *mut *mut c_void, size: usize) -> cudaError_t;
    pub fn cudaFree(devPtr: *mut c_void) -> cudaError_t;
    pub fn cudaMallocAsync(
        devPtr: *mut *mut c_void,
        size: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaFreeAsync(devPtr: *mut c_void, stream: cudaStream_t) -> cudaError_t;

    /// Allocate page-locked (pinned) host memory. With `cudaHostAllocMapped`
    /// the buffer is also directly accessible from the GPU via
    /// `cudaHostGetDevicePointer` — on unified-memory GPUs (Tegra/Thor) this
    /// is zero-copy: a CPU store is visible to kernels with no `cudaMemcpy`.
    pub fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> cudaError_t;
    pub fn cudaFreeHost(ptr: *mut c_void) -> cudaError_t;
    pub fn cudaHostGetDevicePointer(
        devPtr: *mut *mut c_void,
        hostPtr: *const c_void,
        flags: u32,
    ) -> cudaError_t;

    pub fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;

    pub fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cudaMemcpy2DAsync(
        dst: *mut c_void,
        dpitch: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cudaMemset(devPtr: *mut c_void, value: i32, count: usize) -> cudaError_t;
    pub fn cudaMemsetAsync(
        devPtr: *mut c_void,
        value: i32,
        count: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cudaStreamCreate(stream: *mut cudaStream_t) -> cudaError_t;
    pub fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;

    pub fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t;
    pub fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;
    pub fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventElapsedTime(
        milliseconds: *mut f32,
        start: cudaEvent_t,
        end: cudaEvent_t,
    ) -> cudaError_t;

    pub fn cudaDeviceSynchronize() -> cudaError_t;

    /// Explicit collection boundaries consumed by tools such as Nsight
    /// Systems when launched with `--capture-range=cudaProfilerApi`.
    pub fn cudaProfilerStart() -> cudaError_t;
    pub fn cudaProfilerStop() -> cudaError_t;

    pub fn cudaStreamBeginCapture(stream: cudaStream_t, mode: cudaStreamCaptureMode)
        -> cudaError_t;
    pub fn cudaStreamEndCapture(stream: cudaStream_t, pGraph: *mut *mut c_void) -> cudaError_t;

    pub fn cudaGetLastError() -> cudaError_t;
    pub fn cudaGetErrorString(error: cudaError_t) -> *const std::ffi::c_char;

    pub fn cudaSetDevice(device: i32) -> cudaError_t;
    pub fn cudaGetDeviceCount(count: *mut i32) -> cudaError_t;
    pub fn cudaDeviceGetAttribute(value: *mut i32, attribute: i32, device: i32) -> cudaError_t;
    pub fn cudaRuntimeGetVersion(runtimeVersion: *mut i32) -> cudaError_t;
}

/// Check a CUDA runtime call and return a descriptive error.
pub fn check_cuda(err: cudaError_t) -> std::result::Result<(), String> {
    if err == CUDA_SUCCESS {
        Ok(())
    } else {
        let msg = unsafe {
            let ptr = cudaGetErrorString(err);
            std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
        };
        Err(format!("CUDA error {err}: {msg}"))
    }
}

// ── CUDA Graph API ────────────────────────────────────────────────────

pub type cudaGraph_t = *mut c_void;
pub type cudaGraphExec_t = *mut c_void;

extern "C" {
    pub fn cudaGraphInstantiate(
        pGraphExec: *mut cudaGraphExec_t,
        graph: cudaGraph_t,
        pErrorNode: *mut *mut c_void,
        pLogBuffer: *mut std::ffi::c_char,
        bufferSize: usize,
    ) -> cudaError_t;

    pub fn cudaGraphLaunch(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t;
    pub fn cudaGraphExecDestroy(graphExec: cudaGraphExec_t) -> cudaError_t;
    pub fn cudaGraphDestroy(graph: cudaGraph_t) -> cudaError_t;
}
