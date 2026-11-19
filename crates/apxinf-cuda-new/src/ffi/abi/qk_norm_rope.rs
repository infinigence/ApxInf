use std::ffi::c_void;

pub(crate) use super::types::CudaStream;

unsafe extern "C" {
    pub(crate) fn apxinf_qk_norm_rope_head_rms_norm(
        data: *mut c_void,
        weight: *const c_void,
        rows: i64,
        head_dim: i64,
        epsilon: f32,
        stream: CudaStream,
    ) -> i32;

    pub(crate) fn apxinf_qk_norm_rope_fused(
        input: *const c_void,
        weight: *const c_void,
        positions: *const c_void,
        output: *mut c_void,
        tokens: i64,
        heads: i64,
        head_dim: i64,
        rotary_dim: i64,
        theta: f32,
        epsilon: f32,
        stream: CudaStream,
    ) -> i32;

    pub(crate) fn apxinf_qk_norm_rope_cache_append_row(
        src: *const c_void,
        cache: *mut c_void,
        position: *const c_void,
        row_elems: i64,
        capacity: i64,
        stream: CudaStream,
    ) -> i32;
}
