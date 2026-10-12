use std::ffi::c_void;

use super::types::CudaStream;

unsafe extern "C" {
    pub(crate) fn apxinf_vla_gqa_qkv_mrope_cache_bf16(
        qkv: *const c_void,
        bias: *const c_void,
        position_ids: *const c_void,
        q: *mut c_void,
        k_cache: *mut c_void,
        v_cache: *mut c_void,
        tokens: i32,
        q_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        theta: f32,
        section_h: i32,
        section_w: i32,
        cache_offset: i32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_vla_vision_qkv_rope_bf16(
        qkv: *const c_void,
        bias: *const c_void,
        position_ids: *const c_void,
        q: *mut c_void,
        k: *mut c_void,
        v: *mut c_void,
        tokens: i32,
        heads: i32,
        head_dim: i32,
        theta: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_vla_segmented_mha_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        offsets: *const c_void,
        output: *mut c_void,
        segments: i32,
        max_tokens: i32,
        heads: i32,
        head_dim: i32,
        stream: CudaStream,
    ) -> i32;
}
