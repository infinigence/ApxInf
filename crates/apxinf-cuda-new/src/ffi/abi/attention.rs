use std::ffi::{c_char, c_void};

pub(crate) use super::types::Policy;
use super::types::{CudaStream, Runtime};

pub(crate) const SPEC_VERSION: u32 = 3;
pub(crate) const SEMANTIC_DENSE: u32 = 0;
pub(crate) const SEMANTIC_KV_CACHE: u32 = 1;
pub(crate) const SEMANTIC_SEGMENTED: u32 = 2;
pub(crate) const SEMANTIC_PACKED_QKV: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Spec {
    pub version: u32,
    pub semantic: u32,
    pub dtype: u32,
    pub output_dtype: u32,
    pub mask: u32,
    pub q_alignment: u32,
    pub k_alignment: u32,
    pub v_alignment: u32,
    pub output_alignment: u32,
    pub offsets_alignment: u32,
    pub batch: i64,
    pub query_tokens: i64,
    pub key_tokens: i64,
    pub key_capacity: i64,
    pub query_heads: i64,
    pub kv_heads: i64,
    pub head_dim: i64,
    pub query_start: i64,
    pub segments: i64,
    pub max_segment_tokens: i64,
    pub offsets_hash: u64,
    pub scale_is_default: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bindings {
    pub query: *const c_void,
    pub key: *const c_void,
    pub value: *const c_void,
    pub offsets: *const c_void,
    pub output: *mut c_void,
    pub stream: CudaStream,
    pub scale: f32,
    pub output_scale: f32,
}

pub(crate) type Execution = *mut c_void;

unsafe extern "C" {
    pub(crate) fn apxinf_attention_prepare(
        runtime: Runtime,
        spec: *const Spec,
        policy: *const Policy,
        bindings: *const Bindings,
        execution: *mut Execution,
    ) -> i32;
    pub(crate) fn apxinf_attention_enqueue(execution: Execution) -> i32;
    pub(crate) fn apxinf_attention_destroy(execution: Execution);
    #[cfg(test)]
    pub(crate) fn apxinf_attention_summary(execution: Execution) -> *const c_char;
    #[cfg(test)]
    pub(crate) fn apxinf_attention_test_validate_candidates(
        runtime: Runtime,
        spec: *const Spec,
        policy: *const Policy,
        bindings: *const Bindings,
        expected_output: *const f32,
        expected_output_len: u64,
    ) -> i32;
}
