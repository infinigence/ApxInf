//! Raw bindings for the SmolVLA FP16 kernel surface, ported from the legacy
//! crate with the apxinf_smolvla_ prefix. Defined in
//! native/adapters/smolvla/ops.cu.

use std::ffi::c_void;

use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

unsafe extern "C" {
    pub(crate) fn apxinf_smolvla_rms_norm_f16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_smolvla_layer_norm_f16(
        input: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_smolvla_swiglu_f16(
        gate_up: *const c_void,
        output: *mut c_void,
        rows: i32,
        inner: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_smolvla_bias_gelu_f16(
        input: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_smolvla_bias_silu_f16(
        input: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_smolvla_bias_f16(
        input: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_smolvla_euler_update_f16(
        state: *const c_void,
        velocity: *const c_void,
        output: *mut c_void,
        count: i64,
        dt: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_smolvla_add_f16(
        a: *const c_void,
        b: *const c_void,
        output: *mut c_void,
        count: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_smolvla_scale_f16(
        input: *const c_void,
        output: *mut c_void,
        count: u32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_smolvla_pixel_shuffle_4_bf16(
        input: *const c_void,
        output: *mut c_void,
        views: i32,
        tokens_per_view: i32,
        width: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_smolvla_pixel_shuffle_4_f16(
        input: *const c_void,
        output: *mut c_void,
        views: i32,
        tokens_per_view: i32,
        width: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_smolvla_rope_half_split_f16(
        input: *const c_void,
        output: *mut c_void,
        head_dim: u32,
        n_heads: u32,
        seq_len: u32,
        rope_theta: f32,
        pos_offset: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_smolvla_cublas_gqa_f16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        query_tokens: i32,
        key_tokens: i32,
        q_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        stream: cudaStream_t,
    ) -> i32;
    pub(crate) fn apxinf_smolvla_cublas_gqa_causal_f16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        query_tokens: i32,
        key_tokens: i32,
        q_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        key_offset: i32,
        stream: cudaStream_t,
    ) -> i32;
}
