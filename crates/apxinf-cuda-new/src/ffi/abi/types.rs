use std::ffi::{c_char, c_void};

pub(crate) type Runtime = *mut c_void;
pub(crate) type CudaStream = *mut c_void;

#[repr(C)]
pub(crate) struct Policy {
    pub workspace_limit: u64,
    pub online_tune: u32,
    pub allow_fallback: u32,
    pub graph_safe: u32,
    pub deterministic: u32,
    pub cache_dir: *const c_char,
}
