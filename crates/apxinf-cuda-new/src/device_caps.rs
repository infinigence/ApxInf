//! Immutable CUDA device facts used by kernel dispatch and model policy.

use std::ffi::CStr;
use std::fmt;

use crate::ffi::abi::{runtime, status, types::Runtime};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CudaArchFamily {
    Sm80,
    Sm100,
    Other(u32),
}

impl fmt::Display for CudaArchFamily {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sm80 => formatter.write_str("sm80-family"),
            Self::Sm100 => formatter.write_str("sm100-family"),
            Self::Other(sm) => write!(formatter, "sm{sm}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaDeviceCaps {
    pub device_name: String,
    pub compute_major: u32,
    pub compute_minor: u32,
    pub sm: u32,
    pub multiprocessor_count: u32,
    pub arch_family: CudaArchFamily,
}

impl CudaDeviceCaps {
    pub(crate) fn query(runtime_handle: Runtime) -> Result<Self, String> {
        let mut info = runtime::DeviceInfo {
            version: runtime::DEVICE_INFO_VERSION,
            compute_major: 0,
            compute_minor: 0,
            multiprocessor_count: 0,
            device_name: [0; 256],
        };
        unsafe {
            status::check(runtime::apxinf_runtime_device_info(
                runtime_handle,
                &mut info,
            ))
            .map_err(|error| error.to_string())?;
        }
        let sm = info
            .compute_major
            .checked_mul(10)
            .and_then(|major| major.checked_add(info.compute_minor))
            .ok_or_else(|| "CUDA compute capability overflow".to_string())?;
        let device_name = unsafe { CStr::from_ptr(info.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        Ok(Self {
            device_name,
            compute_major: info.compute_major,
            compute_minor: info.compute_minor,
            sm,
            multiprocessor_count: info.multiprocessor_count,
            arch_family: Self::classify(sm),
        })
    }

    pub const fn classify(sm: u32) -> CudaArchFamily {
        match sm {
            80 | 86 | 87 | 89 => CudaArchFamily::Sm80,
            100 | 101 | 110 | 120 | 121 => CudaArchFamily::Sm100,
            other => CudaArchFamily::Other(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_supported_architecture_families() {
        for sm in [80, 86, 87, 89] {
            assert_eq!(CudaDeviceCaps::classify(sm), CudaArchFamily::Sm80);
        }
        for sm in [100, 101, 110, 120, 121] {
            assert_eq!(CudaDeviceCaps::classify(sm), CudaArchFamily::Sm100);
        }
    }
}
