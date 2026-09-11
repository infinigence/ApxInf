use apxinf_core::{Error, Result};
use std::ffi::{c_char, CStr};

unsafe extern "C" {
    fn apxinf_last_error() -> *const c_char;
}

pub(crate) fn check(status: i32) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let pointer = unsafe { apxinf_last_error() };
    let message = if pointer.is_null() {
        "native runtime returned no error message".into()
    } else {
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    };
    Err(Error::Cuda(format!(
        "ApxInf native status {status}: {message}"
    )))
}
