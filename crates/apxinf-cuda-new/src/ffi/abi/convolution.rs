use super::types::{CudaStream, Runtime};
use std::ffi::{c_char, c_void};
pub(crate) type Execution = *mut c_void;
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Spec {
    pub version: u32,
    pub dtype: u32,
    pub input: [i64; 5],
    pub weight: [i64; 5],
    pub output: [i64; 5],
    pub stride: [i64; 3],
    pub alignment: [u32; 3],
}
#[repr(C)]
pub(crate) struct Bindings {
    pub input: *const c_void,
    pub weight: *const c_void,
    pub output: *mut c_void,
    pub workspace: *mut c_void,
    pub workspace_bytes: u64,
    pub stream: CudaStream,
}
unsafe extern "C" {
    pub(crate) fn apxinf_convolution_cudnn_version() -> u64;
    pub(crate) fn apxinf_convolution_prepare(
        runtime: Runtime,
        spec: *const Spec,
        bindings: *const Bindings,
        configuration: i32,
        out: *mut Execution,
    ) -> i32;
    pub(crate) fn apxinf_convolution_enqueue(execution: Execution) -> i32;
    pub(crate) fn apxinf_convolution_summary(execution: Execution) -> *const c_char;
    pub(crate) fn apxinf_convolution_configuration_count(execution: Execution) -> i32;
    pub(crate) fn apxinf_convolution_destroy(execution: Execution);
}
