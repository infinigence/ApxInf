//! CUDA stream wrapper.

use std::ffi::c_void;

use crate::ffi;

/// Owns a CUDA stream for async kernel execution.
pub struct CudaStream {
    handle: ffi::cudaStream_t,
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    pub fn new() -> Result<Self, String> {
        let mut handle: ffi::cudaStream_t = std::ptr::null_mut();
        unsafe {
            ffi::check_cuda(ffi::cudaStreamCreate(&mut handle))?;
        }
        Ok(Self { handle })
    }

    /// Block until all operations on this stream are complete.
    pub fn synchronize(&self) -> Result<(), String> {
        unsafe { ffi::check_cuda(ffi::cudaStreamSynchronize(self.handle)) }
    }

    /// Raw stream handle for passing to CUDA APIs.
    pub fn handle(&self) -> ffi::cudaStream_t {
        self.handle
    }

    /// Queue a dependency on an event recorded by another stream.
    pub fn wait_event(&self, event: &CudaEvent) -> Result<(), String> {
        unsafe { ffi::check_cuda(ffi::cudaStreamWaitEvent(self.handle, event.handle, 0)) }
    }

    /// Default (null) stream.
    pub fn default_stream() -> Self {
        Self {
            handle: std::ptr::null_mut::<c_void>(),
        }
    }
}

/// Reusable CUDA completion event for waiting on a precise stream dependency.
pub struct CudaEvent {
    handle: ffi::cudaEvent_t,
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl CudaEvent {
    pub fn new() -> Result<Self, String> {
        let mut handle: ffi::cudaEvent_t = std::ptr::null_mut();
        unsafe {
            ffi::check_cuda(ffi::cudaEventCreateWithFlags(
                &mut handle,
                ffi::cudaEventDisableTiming,
            ))?;
        }
        Ok(Self { handle })
    }

    /// Record completion of all work already submitted to `stream`.
    pub fn record(&self, stream: &CudaStream) -> Result<(), String> {
        unsafe { ffi::check_cuda(ffi::cudaEventRecord(self.handle, stream.handle())) }
    }

    /// Block until the recorded stream dependency has completed.
    pub fn synchronize(&self) -> Result<(), String> {
        unsafe { ffi::check_cuda(ffi::cudaEventSynchronize(self.handle)) }
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = ffi::cudaEventDestroy(self.handle);
            }
        }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = ffi::cudaStreamDestroy(self.handle);
            }
        }
    }
}
