use std::ffi::c_void;

pub(crate) use super::types::CudaStream;
use super::types::Runtime;

pub(crate) const SPEC_VERSION: u32 = 1;

pub(crate) const SEMANTIC_CONV1D: u32 = 0;
pub(crate) const SEMANTIC_CONV_TRANSPOSE1D: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Spec {
    pub version: u32,
    pub semantic: u32,
    pub dtype: u32,
    pub has_bias: u32,
    pub input_alignment: u32,
    pub weight_alignment: u32,
    pub bias_alignment: u32,
    pub output_alignment: u32,
    pub batch: i64,
    pub in_length: i64,
    pub in_channels: i64,
    pub out_channels: i64,
    pub kernel: i64,
    pub stride: i64,
    pub padding: i64,
    pub dilation: i64,
    pub groups: i64,
    pub output_padding: i64,
    pub out_length: i64,
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
    pub(crate) fn apxinf_conv1d_prepare(
        runtime: Runtime,
        spec: *const Spec,
        bindings: *const Bindings,
        workspace_limit: u64,
        ordered_accumulation: u32,
        execution: *mut Execution,
    ) -> i32;
    pub(crate) fn apxinf_conv1d_enqueue(execution: Execution) -> i32;
    pub(crate) fn apxinf_conv1d_destroy(execution: Execution);
    #[cfg(test)]
    pub(crate) fn apxinf_conv1d_summary(execution: Execution) -> *const std::ffi::c_char;

    pub(crate) fn apxinf_conv1d_launch(
        runtime: Runtime,
        spec: *const Spec,
        bindings: *const Bindings,
    ) -> i32;
}
