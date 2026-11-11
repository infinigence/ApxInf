use std::ffi::c_void;

pub(crate) use super::types::CudaStream;
use super::types::Runtime;

pub(crate) const SPEC_VERSION: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Spec {
    pub version: u32,
    pub dtype: u32,
    pub heads_share_relative: u32,
    pub reserved: u32,
    pub batch: i64,
    pub length: i64,
    pub heads: i64,
    pub head_dim: i64,
    pub window: i64,
    pub scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bindings {
    pub query: *const c_void,
    pub key: *const c_void,
    pub value: *const c_void,
    pub relative_key: *const c_void,
    pub relative_value: *const c_void,
    pub output: *mut c_void,
    pub stream: CudaStream,
}

unsafe extern "C" {
    pub(crate) fn apxinf_relative_attention_launch(
        runtime: Runtime,
        spec: *const Spec,
        bindings: *const Bindings,
    ) -> i32;
}
