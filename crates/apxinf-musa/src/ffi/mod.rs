//! Raw MUSA bindings. The only module allowed to declare `extern "C"`.

#![allow(non_camel_case_types)]

pub mod custom;
pub mod mubin;
pub mod mublas;
pub mod musa;
pub mod mutlass_fa;
pub mod qint8_gemm;

/// Turn a status into a readable error, naming the operator.
///
/// `message` runs only on failure, so a successful launch does not format a
/// string it would throw away — which matters on the operators that launch
/// thousands of times per inference.
pub fn check_status(status: i32, message: impl FnOnce() -> String) -> Result<(), String> {
    if status == 0 {
        Ok(())
    } else {
        Err(message())
    }
}

/// Read an adapter's thread-local error string, if it wrote one.
///
/// # Safety
/// `pointer` must be null or a NUL-terminated string that stays valid on this
/// thread until the next adapter call.
pub(crate) unsafe fn error_text(pointer: *const std::os::raw::c_char) -> Option<String> {
    if pointer.is_null() {
        None
    } else {
        Some(
            unsafe { std::ffi::CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned(),
        )
    }
}

/// Turn a runtime status into a readable error.
pub fn runtime_error(status: musa::musaError_t) -> String {
    if status == musa::MUSA_SUCCESS {
        return "musaSuccess".to_string();
    }
    // SAFETY: `musaGetErrorString` returns a static string for any status.
    let text = unsafe { error_text(musa::musaGetErrorString(status)) };
    format!("MUSA error {status}: {}", text.unwrap_or_default())
}

/// Turn a driver status into a readable error.
pub fn driver_error(status: musa::MUresult) -> String {
    format!("MUSA driver error {status}")
}

pub fn check_runtime(status: musa::musaError_t) -> Result<(), String> {
    check_status(status, || runtime_error(status))
}
