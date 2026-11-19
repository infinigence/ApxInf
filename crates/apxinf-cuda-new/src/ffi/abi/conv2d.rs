use std::ffi::c_void;

pub(crate) use super::types::CudaStream;
use super::types::Runtime;

pub(crate) const SPEC_VERSION: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Spec {
    pub version: u32,
    pub dtype: u32,
    pub has_bias: u32,
    pub input_alignment: u32,
    pub weight_alignment: u32,
    pub bias_alignment: u32,
    pub output_alignment: u32,
    pub reserved: u32,
    pub batch: i64,
    pub in_h: i64,
    pub in_w: i64,
    pub in_channels: i64,
    pub out_channels: i64,
    pub kernel_h: i64,
    pub kernel_w: i64,
    pub stride_h: i64,
    pub stride_w: i64,
    pub pad_h: i64,
    pub pad_w: i64,
    pub out_h: i64,
    pub out_w: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bindings {
    pub input: *const c_void,
    pub weight: *const c_void,
    pub bias: *const c_void,
    pub output: *mut c_void,
    pub stream: CudaStream,
}

pub(crate) type Execution = *mut c_void;

unsafe extern "C" {
    pub(crate) fn apxinf_conv2d_prepare(
        runtime: Runtime,
        spec: *const Spec,
        bindings: *const Bindings,
        execution: *mut Execution,
    ) -> i32;
    pub(crate) fn apxinf_conv2d_enqueue(execution: Execution) -> i32;
    pub(crate) fn apxinf_conv2d_destroy(execution: Execution);
    #[cfg(test)]
    pub(crate) fn apxinf_conv2d_summary(execution: Execution) -> *const std::ffi::c_char;

    #[allow(dead_code)]
    pub(crate) fn apxinf_conv2d_launch(
        runtime: Runtime,
        spec: *const Spec,
        bindings: *const Bindings,
    ) -> i32;
}
