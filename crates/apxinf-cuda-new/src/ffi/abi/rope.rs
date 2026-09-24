use std::ffi::c_void;

use super::types::Runtime;
pub(crate) use super::types::CudaStream;

pub(crate) const SPEC_VERSION: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Spec {
    pub version: u32,
    pub semantic: u32,
    pub dtype: u32,
    pub has_bias: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub qkv_alignment: u32,
    pub bias_alignment: u32,
    pub q_alignment: u32,
    pub kv_alignment: u32,
    pub position_alignment: u32,
    pub tokens: i64,
    pub cache_capacity: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bindings {
    pub qkv: *const c_void,
    pub bias: *const c_void,
    pub q: *mut c_void,
    pub k: *mut c_void,
    pub v: *mut c_void,
    pub key_input: *const c_void,
    pub value_input: *const c_void,
    pub position: *const u32,
    pub stream: CudaStream,
    pub theta: f32,
    pub position_offset: i32,
    pub kv_output_offset: i32,
}

unsafe extern "C" {
    pub(crate) fn apxinf_rope_launch(
        runtime: Runtime,
        spec: *const Spec,
        bindings: *const Bindings,
    ) -> i32;
}
