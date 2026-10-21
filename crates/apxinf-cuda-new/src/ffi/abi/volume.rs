use super::types::{CudaStream, Runtime};
use std::ffi::c_void;

#[repr(C)]
pub(crate) struct Spec {
    pub version: u32,
    pub reserved: u32,
    pub input: [i64; 4],
    pub history_frames: i64,
    pub kernel: [i64; 3],
    pub stride: [i64; 3],
    pub padding_before: [i64; 3],
    pub padding_after: [i64; 3],
    pub output: [i64; 3],
    pub row_start: i64,
    pub rows: i64,
}

#[repr(C)]
pub(crate) struct Bindings {
    pub input: *const c_void,
    pub history: *const c_void,
    pub output: *mut c_void,
    pub stream: CudaStream,
}

unsafe extern "C" {
    pub(crate) fn apxinf_unfold3d_prepare(
        runtime: Runtime,
        spec: *const Spec,
        bindings: *const Bindings,
        execution: *mut *mut c_void,
    ) -> i32;
    pub(crate) fn apxinf_unfold3d_enqueue(execution: *mut c_void) -> i32;
    pub(crate) fn apxinf_unfold3d_destroy(execution: *mut c_void);
}
