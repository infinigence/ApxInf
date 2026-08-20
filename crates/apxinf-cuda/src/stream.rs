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

    /// Default (null) stream.
    pub fn default_stream() -> Self {
        Self {
            handle: std::ptr::null_mut::<c_void>(),
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
