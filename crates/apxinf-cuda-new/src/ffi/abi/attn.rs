use std::ffi::c_void;

use super::types::CudaStream;

unsafe extern "C" {
    pub(crate) fn apxinf_attn_partial_rope(
        data: *mut c_void,
        positions: *const c_void,
        tokens: i64,
        heads: i64,
        head_dim: i64,
        rotary_dim: i64,
        theta: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_attn_head_rms_norm(
        data: *mut c_void,
        weight: *const c_void,
        rows: i64,
        head_dim: i64,
        epsilon: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_attn_split_query_and_gate(
        fused: *const c_void,
        query: *mut c_void,
        gate: *mut c_void,
        tokens: i64,
        heads: i64,
        head_dim: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_attn_apply_output_gate(
        data: *mut c_void,
        gate: *const c_void,
        count: i64,
        stream: CudaStream,
    ) -> i32;
}
