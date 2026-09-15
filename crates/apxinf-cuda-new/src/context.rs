//! CUDA device and stream context.

use std::sync::Arc;

use crate::ffi;
use crate::stream::CudaStream;

/// Owns the stream and opaque native runtime used by GEMM executions.
pub struct CudaContext {
    device_id: usize,
    stream: Arc<CudaStream>,
    runtime: crate::ffi::abi::types::Runtime,
}

impl CudaContext {
    /// Create a context for the specified CUDA device.
    pub fn new(device_id: usize) -> Result<Self, String> {
        unsafe {
            ffi::check_cuda(ffi::cudaSetDevice(device_id as i32))?;
        }

        let stream = Arc::new(CudaStream::new_on(device_id)?);
        let mut runtime = std::ptr::null_mut();
        unsafe {
            crate::ffi::abi::status::check(crate::ffi::abi::runtime::apxinf_runtime_create(
                device_id as i32,
                &mut runtime,
            ))
            .map_err(|error| error.to_string())?;
        }

        Ok(Self {
            device_id,
            stream,
            runtime,
        })
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }
    pub(crate) fn shared_stream(&self) -> Arc<CudaStream> {
        Arc::clone(&self.stream)
    }
    pub(crate) fn runtime(&self) -> crate::ffi::abi::types::Runtime {
        self.runtime
    }

    pub fn synchronize(&self) -> Result<(), String> {
        self.stream.synchronize()
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe { crate::ffi::abi::runtime::apxinf_runtime_destroy(self.runtime) }
    }
}
