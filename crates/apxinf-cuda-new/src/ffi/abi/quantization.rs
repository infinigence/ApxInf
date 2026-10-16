use std::ffi::c_void;

use super::types::Runtime;
pub(crate) use super::types::CudaStream;

pub(crate) const SPEC_VERSION: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Spec {
    pub version: u32,
    pub semantic: u32,
    pub input_dtype: u32,
    pub output_dtype: u32,
    pub scale_dtype: u32,
    pub input_alignment: u32,
    pub output_alignment: u32,
    pub scales_alignment: u32,
    pub rows: i64,
    pub input_cols: i64,
    pub output_cols: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bindings {
    pub input: *const c_void,
    pub output: *mut c_void,
    pub scales: *mut f32,
    pub stream: CudaStream,
    pub scale: f32,
}

unsafe extern "C" {
    pub(crate) fn apxinf_quantization_launch(
        runtime: Runtime,
        spec: *const Spec,
        bindings: *const Bindings,
    ) -> i32;
}
