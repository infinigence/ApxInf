//! CUDA device and stream context.

use std::sync::Arc;

use apxinf_core::{DType, Error, Result as CoreResult, Shape, Tensor};

use crate::ffi;
use crate::stream::CudaStream;
use crate::CudaDeviceCaps;

/// Owns the stream and opaque native runtime used by GEMM executions.
pub struct CudaContext {
    device_id: usize,
    stream: Arc<CudaStream>,
    runtime: crate::ffi::abi::types::Runtime,
    caps: CudaDeviceCaps,
    cublas: crate::cublas::CublasHandle,
    tuning: std::sync::RwLock<Arc<crate::tuning::TuningSession>>,
}

impl CudaContext {
    /// Create a context for the specified CUDA device.
    pub fn new(device_id: usize) -> Result<Self, String> {
        Self::new_with_autotune(device_id, true)
    }

    /// Create a context with an immutable permission for online tuning.
    ///
    /// Tuning requires both this permission and the operator policy's
    /// `online_tune`. Disabling it still permits reuse of persisted recipes.
    pub fn new_with_autotune(device_id: usize, allow_online_tune: bool) -> Result<Self, String> {
        let device = i32::try_from(device_id)
            .map_err(|_| format!("CUDA device id {device_id} does not fit in i32"))?;
        unsafe {
            ffi::check_cuda(ffi::cudaSetDevice(device))?;
        }

        let stream = Arc::new(CudaStream::new_on(device_id)?);
        let mut runtime = std::ptr::null_mut();
        unsafe {
            crate::ffi::abi::status::check(
                crate::ffi::abi::runtime::apxinf_runtime_create_with_autotune(
                    device,
                    u32::from(allow_online_tune),
                    &mut runtime,
                ),
            )
            .map_err(|error| error.to_string())?;
        }
        let caps = match CudaDeviceCaps::query(runtime) {
            Ok(caps) => caps,
            Err(error) => {
                unsafe { crate::ffi::abi::runtime::apxinf_runtime_destroy(runtime) };
                return Err(error);
            }
        };

        let cublas = crate::cublas::CublasHandle::new()?;
        cublas.set_stream(&stream)?;

        Ok(Self {
            device_id,
            stream,
            runtime,
            caps,
            cublas,
            tuning: std::sync::RwLock::new(crate::tuning::default_session()),
        })
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }
    /// Default recipe directory, relative to the process working directory.
    /// An explicit operator `cache_dir` overrides this hardware/toolkit path.
    pub fn default_cache_dir(&self) -> &str {
        unsafe {
            std::ffi::CStr::from_ptr(crate::ffi::abi::runtime::apxinf_runtime_default_cache_dir(
                self.runtime,
            ))
        }
        .to_str()
        .expect("native default cache directory is ASCII")
    }
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }
    /// Raw cuBLAS handle bound to the context stream, for the direct-launch
    /// GEMM helpers that predate the tuned GEMM operator.
    pub fn cublas(&self) -> &crate::cublas::CublasHandle {
        &self.cublas
    }
    /// The installed tuning session (plan-invalidation identity; cuda-new
    /// operators tune through recipes, not through this session).
    pub fn tuning(&self) -> Arc<crate::tuning::TuningSession> {
        self.tuning
            .read()
            .expect("CUDA tuning session lock is poisoned")
            .clone()
    }
    /// Install a session before model prepare. Prepared plans keyed on the
    /// previous session observe the identity change and rebuild.
    pub fn install_tuning(&self, session: crate::tuning::TuningSession) -> Result<(), String> {
        *self
            .tuning
            .write()
            .map_err(|_| "CUDA tuning session lock is poisoned".to_string())? = Arc::new(session);
        Ok(())
    }
    pub fn caps(&self) -> &CudaDeviceCaps {
        &self.caps
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

    /// Allocate caller-owned output storage for an L3 semantic operation.
    ///
    /// Inside an [`crate::ExecutionSession`] this sub-allocates from the
    /// deterministic graph arena. Outside a session it creates an ordinary
    /// zero-initialized device allocation. Operator selection remains in L3;
    /// this method only provides stable runtime-owned storage for its output
    /// binding.
    pub fn allocate_output(&self, shape: Shape, dtype: DType) -> CoreResult<Tensor> {
        let bytes = shape
            .numel()
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| Error::Other("CUDA output size overflow".into()))?;
        let buffer = crate::workspace::output_buffer(self, bytes)?;
        Ok(buffer.into_tensor(shape, dtype))
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe { crate::ffi::abi::runtime::apxinf_runtime_destroy(self.runtime) }
    }
}
