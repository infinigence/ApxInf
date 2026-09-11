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

    pub(crate) fn new_on(device: usize) -> Result<Self, String> {
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

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let _ = self.with_current_device(|| unsafe {
                ffi::check_cuda(ffi::cudaStreamDestroy(self.handle))
            });
        }
    }
}
