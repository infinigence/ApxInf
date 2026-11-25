//! CUDA stream wrapper.

use std::ffi::c_void;

use crate::ffi;

/// Owns a CUDA stream for async kernel execution.
pub struct CudaStream {
    handle: ffi::cudaStream_t,
    device: usize,
}

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    pub fn new() -> Result<Self, String> {
        let mut device = 0;
        unsafe {
            ffi::check_cuda(ffi::cudaGetDevice(&mut device))?;
        }
        Self::new_on(device as usize)
    }

    /// A non-blocking stream on `device`, for work that must overlap the context stream (for example weight copies).
    pub fn new_on(device: usize) -> Result<Self, String> {
        unsafe {
            ffi::check_cuda(ffi::cudaSetDevice(device as i32))?;
        }
        let mut handle: ffi::cudaStream_t = std::ptr::null_mut();
        unsafe {
            ffi::check_cuda(ffi::cudaStreamCreateWithFlags(
                &mut handle,
                ffi::CUDA_STREAM_NON_BLOCKING,
            ))?;
        }
        Ok(Self { handle, device })
    }

    /// Block until all operations on this stream are complete.
    pub fn synchronize(&self) -> Result<(), String> {
        self.with_current_device(|| unsafe {
            ffi::check_cuda(ffi::cudaStreamSynchronize(self.handle))
        })
    }

    /// Make later work on this stream wait for everything `event` recorded.
    /// An event that was never recorded is already complete.
    pub fn wait_for(&self, event: &CudaEvent) -> Result<(), String> {
        self.with_current_device(|| unsafe {
            ffi::check_cuda(ffi::cudaStreamWaitEvent(self.handle, event.handle, 0))
        })
    }

    /// Record `event` after the work already queued on this stream.
    pub fn record(&self, event: &CudaEvent) -> Result<(), String> {
        self.with_current_device(|| unsafe {
            ffi::check_cuda(ffi::cudaEventRecord(event.handle, self.handle))
        })
    }

    /// Raw stream handle for passing to CUDA APIs.
    pub fn handle(&self) -> ffi::cudaStream_t {
        self.handle
    }

    pub fn device(&self) -> usize {
        self.device
    }

    pub(crate) fn set_current_device(&self) -> Result<(), String> {
        unsafe { ffi::check_cuda(ffi::cudaSetDevice(self.device as i32)) }
    }

    pub(crate) fn with_current_device<T>(
        &self,
        operation: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        let mut previous = 0;
        unsafe {
            ffi::check_cuda(ffi::cudaGetDevice(&mut previous))?;
            if previous != self.device as i32 {
                ffi::check_cuda(ffi::cudaSetDevice(self.device as i32))?;
            }
        }
        let result = operation();
        if previous != self.device as i32 {
            let restore = unsafe { ffi::check_cuda(ffi::cudaSetDevice(previous)) };
            if result.is_ok() {
                restore?;
            }
        }
        result
    }

    /// Default (null) stream.
    pub fn default_stream() -> Self {
        let mut device = 0;
        unsafe {
            let _ = ffi::cudaGetDevice(&mut device);
        }
        Self {
            handle: std::ptr::null_mut::<c_void>(),
            device: device.max(0) as usize,
        }
    }
}

/// A CUDA event without timing, used to order work between two streams.
pub struct CudaEvent {
    handle: ffi::cudaEvent_t,
    device: usize,
}

unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl CudaEvent {
    pub fn new(device: usize) -> Result<Self, String> {
        let mut handle: ffi::cudaEvent_t = std::ptr::null_mut();
        unsafe {
            ffi::check_cuda(ffi::cudaSetDevice(device as i32))?;
            ffi::check_cuda(ffi::cudaEventCreateWithFlags(
                &mut handle,
                ffi::cudaEventDisableTiming,
            ))?;
        }
        Ok(Self { handle, device })
    }

    pub fn device(&self) -> usize {
        self.device
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::cudaEventDestroy(self.handle);
        }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let _ = self.with_current_device(|| unsafe {
                ffi::check_cuda(ffi::cudaStreamDestroy(self.handle))
            });
        }
    }
}
