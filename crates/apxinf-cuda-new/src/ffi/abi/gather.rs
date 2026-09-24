use std::ffi::c_void;

use super::types::Runtime;
pub(crate) use super::types::CudaStream;

pub(crate) const SPEC_VERSION: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Spec {
    pub version: u32,
    pub semantic: u32,
    pub dtype: u32,
    pub has_bias: u32,
    pub vocab_size: u32,
    pub tokens_per_view: u32,
    pub views: u32,
    pub image_size: u32,
    pub patch_size: u32,
    pub nhwc: u32,
    pub input_alignment: u32,
    pub bias_alignment: u32,
    pub output_alignment: u32,
    pub rows: i64,
    pub cols: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bindings {
    pub input: *const c_void,
    pub ids: *const u32,
    pub bias: *const c_void,
    pub position: *const c_void,
    pub output: *mut c_void,
    pub stream: CudaStream,
}

unsafe extern "C" {
    pub(crate) fn apxinf_gather_launch(
        runtime: Runtime,
        spec: *const Spec,
        bindings: *const Bindings,
    ) -> i32;
}
