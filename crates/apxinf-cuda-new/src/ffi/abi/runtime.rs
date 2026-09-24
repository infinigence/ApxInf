use std::ffi::c_char;

use super::types::Runtime;

pub(crate) const DEVICE_INFO_VERSION: u32 = 1;

#[repr(C)]
pub(crate) struct DeviceInfo {
    pub version: u32,
    pub compute_major: u32,
    pub compute_minor: u32,
    pub multiprocessor_count: u32,
    pub device_name: [c_char; 256],
}

unsafe extern "C" {
    pub(crate) fn apxinf_runtime_create(device: i32, runtime: *mut Runtime) -> i32;
    pub(crate) fn apxinf_runtime_device_info(runtime: Runtime, info: *mut DeviceInfo) -> i32;
    pub(crate) fn apxinf_runtime_destroy(runtime: Runtime);
}
