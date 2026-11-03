use super::types::{CudaStream, Runtime};
use std::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Spec {
    pub version: u32,
    pub semantic: u32,
    pub dtype: u32,
    pub output_dtype: u32,
    pub height: u32,
    pub width: u32,
    pub channels: u32,
    pub output_height: u32,
    pub output_width: u32,
    pub output_channels: u32,
    pub kernel: u32,
    pub stride: u32,
    pub padding: u32,
    pub temporal_factor: u32,
}

#[repr(C)]
pub(crate) struct Bindings {
    pub input: *const c_void,
    pub parameter: *const c_void,
    pub parameter2: *const c_void,
    pub output: *mut c_void,
    pub scale: f32,
    pub epsilon: f32,
    pub stream: CudaStream,
}

#[repr(C)]
pub(crate) struct RgbIndicesSpec {
    pub version: u32,
    pub layout: u32,
    pub views: u32,
    pub height: u32,
    pub width: u32,
    pub channels: u32,
}

#[repr(C)]
pub(crate) struct RgbIndicesBindings {
    pub input: *const u8,
    pub output: *mut i32,
    pub stream: CudaStream,
}

unsafe extern "C" {
    pub(crate) fn apxinf_image_rgb_indices_launch(
        runtime: Runtime,
        spec: *const RgbIndicesSpec,
        bindings: *const RgbIndicesBindings,
    ) -> i32;
    pub(crate) fn apxinf_image_launch(
        runtime: Runtime,
        spec: *const Spec,
        bindings: *const Bindings,
    ) -> i32;
}
