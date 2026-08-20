//! Raw CUDA Driver API bindings used for device identity.

pub type CUdevice = i32;
pub type CUresult = i32;

pub const CUDA_DRIVER_SUCCESS: CUresult = 0;

// ── CUDA Driver API (device identity only) ─────────────────────────

extern "C" {
    pub fn cuInit(flags: u32) -> CUresult;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: i32) -> CUresult;
    pub fn cuDeviceGetName(name: *mut std::ffi::c_char, len: i32, device: CUdevice) -> CUresult;
    pub fn cuGetErrorString(error: CUresult, message: *mut *const std::ffi::c_char) -> CUresult;
}

/// Check a CUDA driver call and return a descriptive error.
pub fn check_cuda_driver(status: CUresult) -> std::result::Result<(), String> {
    if status == CUDA_DRIVER_SUCCESS {
        Ok(())
    } else {
        let mut message = std::ptr::null();
        let description = unsafe {
            if cuGetErrorString(status, &mut message) == CUDA_DRIVER_SUCCESS && !message.is_null() {
                std::ffi::CStr::from_ptr(message)
                    .to_string_lossy()
                    .into_owned()
            } else {
                "unknown driver error".to_string()
            }
        };
        Err(format!("CUDA driver error {status}: {description}"))
    }
}
