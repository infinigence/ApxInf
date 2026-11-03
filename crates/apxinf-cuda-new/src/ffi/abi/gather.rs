use std::ffi::c_void;

pub(crate) use super::types::CudaStream;
use super::types::Runtime;

pub(crate) const SPEC_VERSION: u32 = 6;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct Spec {
    pub version: u32,
    pub semantic: u32,
    pub dtype: u32,
    pub has_bias: u32,
    pub embedding_scale: u32,
    pub vocab_size: u32,
    pub replacement_rows: u32,
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
    pub rank: u32,
    pub layout_shape: [i64; 8],
    pub layout_strides: [i64; 8],
    pub row_start: i64,
    pub output_spatial: [i64; 3],
    pub kernel: [i64; 3],
    pub stride: [i64; 3],
    pub dilation: [i64; 3],
    pub padding: [i64; 3],
    pub channel_block: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bindings {
    pub input: *const c_void,
    pub ids: *const u32,
    pub replacement: *const c_void,
    pub row_map: *const u32,
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
