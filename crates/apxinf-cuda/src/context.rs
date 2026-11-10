//! CUDA device and stream context.

use crate::cublas::CublasHandle;
use crate::device_caps::CudaDeviceCaps;
use crate::ffi;
use crate::stream::CudaStream;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaMemoryInfo {
    pub free_bytes: usize,
    pub total_bytes: usize,
}

impl CudaMemoryInfo {
    pub const fn from_bytes(free_bytes: usize, total_bytes: usize) -> Result<Self, &'static str> {
        if free_bytes > total_bytes {
            return Err("free CUDA memory cannot exceed total memory");
        }
        Ok(Self {
            free_bytes,
            total_bytes,
        })
    }

    pub fn calibrated_budget_bytes(self, safety_margin_percent: u8) -> Result<usize, &'static str> {
        if safety_margin_percent > 100 {
            return Err("safety margin must be at most 100 percent");
        }
        let usable_percent = 100usize - safety_margin_percent as usize;
        self.free_bytes
            .checked_mul(usable_percent)
            .and_then(|bytes| bytes.checked_div(100))
            .ok_or("calibrated CUDA budget overflow")
    }
}

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
    device_uuid: String,
    stream: CudaStream,
    cublas: CublasHandle,
    caps: CudaDeviceCaps,
    library_versions: CudaLibraryVersions,
}

impl CudaContext {
    /// Create a context for the specified CUDA device.
    pub fn new(device_id: usize) -> Result<Self, String> {
        Self::new_inner(device_id, None)
    }

    /// Create a context and fail closed unless CUDA reports the expected
    /// physical device UUID for the selected ordinal.
    pub fn new_attested(device_id: usize, expected_uuid: &str) -> Result<Self, String> {
        if expected_uuid.trim().is_empty() {
            return Err("expected CUDA device UUID must be non-empty".into());
        }
        Self::new_inner(device_id, Some(expected_uuid.trim()))
    }

    fn new_inner(device_id: usize, expected_uuid: Option<&str>) -> Result<Self, String> {
        unsafe {
            ffi::check_cuda(ffi::cudaSetDevice(device_id as i32))?;
        }

        let caps = CudaDeviceCaps::query(device_id)?;
        let device_uuid = query_device_uuid(device_id)?;
        if let Some(expected_uuid) = expected_uuid {
            if device_uuid != expected_uuid {
                return Err(format!(
                    "CUDA ordinal {device_id} reports UUID {device_uuid}, expected {expected_uuid}"
                ));
            }
        }
        let stream = CudaStream::new()?;
        let cublas = CublasHandle::new()?;
        cublas.set_stream(&stream)?;
        let library_versions = CudaLibraryVersions::query(&cublas)?;

        Ok(Self {
            device_id,
            device_uuid,
            stream,
            cublas,
            caps,
            library_versions,
        })
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }
    pub fn device_uuid(&self) -> &str {
        &self.device_uuid
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

    pub fn memory_info(&self) -> Result<CudaMemoryInfo, String> {
        unsafe {
            ffi::check_cuda(ffi::cudaSetDevice(self.device_id as i32))?;
        }
        let (mut free_bytes, mut total_bytes) = (0usize, 0usize);
        unsafe {
            ffi::check_cuda(ffi::cudaMemGetInfo(&mut free_bytes, &mut total_bytes))?;
        }
        CudaMemoryInfo::from_bytes(free_bytes, total_bytes).map_err(str::to_owned)
    }

    pub fn synchronize(&self) -> Result<(), String> {
        unsafe { ffi::check_cuda(ffi::cudaStreamSynchronize(self.stream.handle())) }
    }
}

fn query_device_uuid(device_id: usize) -> Result<String, String> {
    let ordinal = i32::try_from(device_id)
        .map_err(|_| format!("CUDA device id {device_id} does not fit in i32"))?;
    unsafe {
        ffi::check_cuda_driver(ffi::cuInit(0))?;
        let mut device = 0;
        ffi::check_cuda_driver(ffi::cuDeviceGet(&mut device, ordinal))?;
        let mut uuid = ffi::CUuuid { bytes: [0; 16] };
        ffi::check_cuda_driver(ffi::cuDeviceGetUuid(&mut uuid, device))?;
        Ok(format_cuda_uuid(uuid.bytes))
    }
}

fn format_cuda_uuid(bytes: [u8; 16]) -> String {
    let hex = bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!(
        "GPU-{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
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
    fn calibrates_free_memory_with_eight_percent_safety_margin() {
        let info = CudaMemoryInfo::from_bytes(1_000, 2_000).unwrap();
        assert_eq!(info.calibrated_budget_bytes(8).unwrap(), 920);
    }

    #[test]
    fn rejects_invalid_memory_ranges_and_margin() {
        assert!(CudaMemoryInfo::from_bytes(2, 1).is_err());
        let info = CudaMemoryInfo::from_bytes(1, 1).unwrap();
        assert!(info.calibrated_budget_bytes(101).is_err());
    }

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

    #[test]
    fn formats_cuda_uuid_bytes_as_nvidia_uuid() {
        assert_eq!(
            format_cuda_uuid([0u8, 1, 2, 3, 0xab, 0xcd, 0xef, 0xff, 8, 9, 10, 11, 12, 13, 14, 15,]),
            "GPU-00010203-abcd-efff-0809-0a0b0c0d0e0f"
        );
    }

    #[test]
    fn attested_context_accepts_expected_uuid_and_rejects_mismatch() {
        let Some(expected) = std::env::var_os("APXINF_TEST_GPU_UUID") else {
            return;
        };
        let expected = expected.to_string_lossy();
        let context = CudaContext::new_attested(0, &expected).unwrap();
        assert_eq!(context.device_uuid(), expected);
        let mismatch = CudaContext::new_attested(0, "GPU-00000000-0000-0000-0000-000000000000");
        assert!(matches!(mismatch, Err(error) if error.contains("reports UUID")));
    }
}
