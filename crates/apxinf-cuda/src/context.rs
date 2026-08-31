//! CUDA device and stream context.

use std::sync::{Arc, RwLock};

use crate::cublas::CublasHandle;
use crate::device_caps::CudaDeviceCaps;
use crate::ffi;
use crate::kernels::gemm::GemmPlanCache;
use crate::stream::CudaStream;
use crate::tuning::{TacticStore, TuningSession};

/// CUDA libraries whose versions constrain persisted tuning results.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaLibraryVersions {
    pub cuda: String,
    pub cublas: String,
}

impl CudaLibraryVersions {
    fn query(cublas: &CublasHandle) -> Result<Self, String> {
        let mut cuda = 0;
        unsafe {
            ffi::check_cuda(ffi::cudaRuntimeGetVersion(&mut cuda))?;
        }
        Ok(Self {
            cuda: format_cuda_runtime_version(cuda)?,
            cublas: format_cublas_version(cublas.version()?)?,
        })
    }
}

/// Holds the CUDA device, stream, and cuBLAS handle.
pub struct CudaContext {
    device_id: usize,
    stream: CudaStream,
    cublas: CublasHandle,
    caps: CudaDeviceCaps,
    library_versions: CudaLibraryVersions,
    tuning: RwLock<Arc<TuningSession>>,
    gemm_plans: GemmPlanCache,
}

impl CudaContext {
    /// Create a context for the specified CUDA device.
    pub fn new(device_id: usize) -> Result<Self, String> {
        unsafe {
            ffi::check_cuda(ffi::cudaSetDevice(device_id as i32))?;
        }

        let caps = CudaDeviceCaps::query(device_id)?;
        let stream = CudaStream::new()?;
        let cublas = CublasHandle::new()?;
        cublas.set_stream(&stream)?;
        let library_versions = CudaLibraryVersions::query(&cublas)?;

        Ok(Self {
            device_id,
            stream,
            cublas,
            caps,
            library_versions,
            tuning: RwLock::new(Arc::new(TuningSession::inference(TacticStore::default()))),
            gemm_plans: GemmPlanCache::default(),
        })
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }
    pub fn cublas(&self) -> &CublasHandle {
        &self.cublas
    }
    pub fn caps(&self) -> &CudaDeviceCaps {
        &self.caps
    }
    pub fn library_versions(&self) -> &CudaLibraryVersions {
        &self.library_versions
    }

    /// Runtime-owned tuning state. Cloning the `Arc` is limited to prepare
    /// paths; captured graph replay never calls this method.
    pub fn tuning(&self) -> Arc<TuningSession> {
        self.tuning
            .read()
            .expect("CUDA tuning session lock is poisoned")
            .clone()
    }

    /// Install the mode and immutable tactic snapshot before model prepare.
    pub fn install_tuning(&self, session: TuningSession) -> Result<(), String> {
        *self
            .tuning
            .write()
            .map_err(|_| "CUDA tuning session lock is poisoned".to_string())? = Arc::new(session);
        self.gemm_plans.clear().map_err(|error| error.to_string())?;
        Ok(())
    }

    pub(crate) fn gemm_plans(&self) -> &GemmPlanCache {
        &self.gemm_plans
    }

    pub fn synchronize(&self) -> Result<(), String> {
        unsafe { ffi::check_cuda(ffi::cudaStreamSynchronize(self.stream.handle())) }
    }
}

fn format_cuda_runtime_version(version: i32) -> Result<String, String> {
    if version <= 0 {
        return Err(format!("CUDA runtime returned invalid version {version}"));
    }
    let major = version / 1000;
    let minor = (version % 1000) / 10;
    let patch = version % 10;
    Ok(format_version(major, minor, patch))
}

fn format_cublas_version(version: i32) -> Result<String, String> {
    if version <= 0 {
        return Err(format!("cuBLAS returned invalid version {version}"));
    }
    let (major, minor, patch) = if version >= 10_000 {
        (version / 10_000, (version % 10_000) / 100, version % 100)
    } else {
        (version / 1000, (version % 1000) / 100, version % 100)
    };
    Ok(format_version(major, minor, patch))
}

fn format_version(major: i32, minor: i32, patch: i32) -> String {
    if patch == 0 {
        format!("{major}.{minor}")
    } else {
        format!("{major}.{minor}.{patch}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_cuda_runtime_versions() {
        assert_eq!(format_cuda_runtime_version(12_060).unwrap(), "12.6");
        assert_eq!(format_cuda_runtime_version(13_001).unwrap(), "13.0.1");
    }

    #[test]
    fn formats_cublas_versions() {
        assert_eq!(format_cublas_version(120_604).unwrap(), "12.6.4");
        assert_eq!(format_cublas_version(110_208).unwrap(), "11.2.8");
    }
}
