use std::ffi::c_void;

use super::types::CudaStream;
use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

unsafe extern "C" {
    pub(crate) fn apxinf_cache_append_bf16(
        new_data: *const c_void,
        cache: *mut c_void,
        n_kv_heads: i32,
        head_dim: i32,
        max_seq_len: i32,
        seq_len: i32,
        append_len: i32,
        stream: CudaStream,
    ) -> i32;
}

unsafe extern "C" {
    pub(crate) fn apxinf_cache_fixed_decode_append_decode_f32(
        cache: *mut c_void,
        new_data: *const c_void,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq_len: u32,
        pos_ptr: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_cache_fixed_decode_append_f32(
        cache: *mut c_void,
        new_data: *const c_void,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq_len: u32,
        seq_len: u32,
        append_len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_cache_fixed_decode_append_bf16(
        cache: *mut c_void,
        new_data: *const c_void,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq_len: u32,
        seq_len: u32,
        append_len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_cache_fixed_decode_append_decode_bf16(
        cache: *mut c_void,
        new_data: *const c_void,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq_len: u32,
        pos_ptr: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_cache_fixed_decode_gather_bf16(
        src: *const c_void,
        dst: *mut c_void,
        tokens: u32,
        n_kv_heads: u32,
        head_dim: u32,
        max_seq_len: u32,
        kv_offset: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}
