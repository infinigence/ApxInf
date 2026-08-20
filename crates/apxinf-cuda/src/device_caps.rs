//! Immutable CUDA device facts used by kernel hardware dispatch.

use std::ffi::CStr;
use std::fmt;

use crate::ffi;

/// NVIDIA architecture families which share ApxInf kernel implementations.
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

/// Hardware capabilities queried once when a [`crate::CudaContext`] is made.
///
/// This type deliberately contains hardware facts only. Model names,
/// precision policy, and selected tactics belong to higher or lower layers.
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
    pub fn query(device_id: usize) -> Result<Self, String> {
        let device = i32::try_from(device_id)
            .map_err(|_| format!("CUDA device id {device_id} does not fit in i32"))?;
        let compute_major = query_attribute(device, ffi::CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MAJOR)?;
        let compute_minor = query_attribute(device, ffi::CUDA_DEV_ATTR_COMPUTE_CAPABILITY_MINOR)?;
        let multiprocessor_count =
            query_attribute(device, ffi::CUDA_DEV_ATTR_MULTIPROCESSOR_COUNT)?;
        let sm = compute_major
            .checked_mul(10)
            .and_then(|major| major.checked_add(compute_minor))
            .ok_or_else(|| "CUDA compute capability overflow".to_string())?;
        let mut name = [0 as std::ffi::c_char; 256];
        unsafe {
            ffi::check_cuda_driver(ffi::cuInit(0))?;
            let mut driver_device = 0;
            ffi::check_cuda_driver(ffi::cuDeviceGet(&mut driver_device, device))?;
            ffi::check_cuda_driver(ffi::cuDeviceGetName(
                name.as_mut_ptr(),
                name.len() as i32,
                driver_device,
            ))?;
        }
        let device_name = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        Ok(Self {
            device_name,
            compute_major,
            compute_minor,
            sm,
            multiprocessor_count,
            arch_family: Self::classify(sm),
        })
    }

    pub const fn classify(sm: u32) -> CudaArchFamily {
        match sm {
            80 | 86 | 87 | 89 => CudaArchFamily::Sm80,
            100 | 101 | 110 | 120 => CudaArchFamily::Sm100,
            other => CudaArchFamily::Other(other),
        }
    }
}

fn query_attribute(device: i32, attribute: i32) -> Result<u32, String> {
    let mut value = 0i32;
    unsafe {
        ffi::check_cuda(ffi::cudaDeviceGetAttribute(&mut value, attribute, device))?;
    }
    u32::try_from(value).map_err(|_| {
        format!("CUDA device {device} returned negative attribute {attribute}: {value}")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_supported_architecture_families() {
        for sm in [80, 86, 87, 89] {
            assert_eq!(CudaDeviceCaps::classify(sm), CudaArchFamily::Sm80);
        }
        for sm in [100, 101, 110, 120] {
            assert_eq!(CudaDeviceCaps::classify(sm), CudaArchFamily::Sm100);
        }
    }

    #[test]
    fn preserves_unknown_architecture() {
        assert_eq!(CudaDeviceCaps::classify(75), CudaArchFamily::Other(75));
    }
}
