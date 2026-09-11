use std::ffi::c_void;

pub(crate) type Runtime = *mut c_void;
pub(crate) type CudaStream = *mut c_void;
