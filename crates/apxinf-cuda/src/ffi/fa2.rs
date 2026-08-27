//! Raw bindings for the FlashAttention 2 adapter.

use std::ffi::c_void;

use super::cuda::{cudaError_t, cudaStream_t};

extern "C" {
    #[cfg(apxinf_fa2_f16_sm100)]
    pub fn apxinf_static_fa2_f16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        softmax_lse: *mut c_void,
        batches: i32,
        query_tokens: i32,
        key_tokens: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    #[cfg(apxinf_fa2_direct_e4m3_sm100)]
    pub fn apxinf_static_fa2_f16_direct_e4m3_522(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        softmax_lse: *mut c_void,
        batches: i32,
        query_tokens: i32,
        key_tokens: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        output_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    #[cfg(any(apxinf_fa2_sm80, apxinf_fa2_vision_sm80))]
    pub fn apxinf_static_fa2_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        softmax_lse: *mut c_void,
        batches: i32,
        query_tokens: i32,
        key_tokens: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    #[cfg(any(apxinf_fa2_sm80, apxinf_fa2_vision_sm80))]
    pub fn apxinf_static_fa2_varlen_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        softmax_lse: *mut c_void,
        cu_seqlens: *const c_void,
        batch: i32,
        total_tokens: i32,
        max_tokens: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    #[cfg(apxinf_fa2_causal_sm80)]
    pub fn apxinf_static_fa2_causal_strided_kv_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        softmax_lse: *mut c_void,
        query_tokens: i32,
        key_tokens: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        max_seq_len: i32,
        softmax_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    #[cfg(apxinf_fa2_sm80)]
    pub fn apxinf_static_fa2_bf16_splitkv(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        softmax_lse: *mut c_void,
        softmax_lse_accum: *mut c_void,
        o_accum: *mut c_void,
        batches: i32,
        query_tokens: i32,
        key_tokens: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        num_sms: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}
