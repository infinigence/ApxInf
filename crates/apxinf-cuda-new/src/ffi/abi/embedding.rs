use std::ffi::c_void;

use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

unsafe extern "C" {
    pub(crate) fn apxinf_embedding_fixed_decode_lookup_f32(
        table: *const c_void,
        ids: *const c_void,
        output: *mut c_void,
        embed_dim: u32,
        seq_len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_embedding_fixed_decode_lookup_bf16(
        table: *const c_void,
        ids: *const c_void,
        output: *mut c_void,
        embed_dim: u32,
        seq_len: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

mod fixed_launchers {
    use std::ffi::c_void;

    use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

    unsafe extern "C" {
        pub(crate) fn apxinf_embedding_sinusoidal_embedding_bf16(
            positions: *const c_void,
            output: *mut c_void,
            rows: i32,
            dim: i32,
            scale: f32,
            frequency_step: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
    }
}
pub(crate) use fixed_launchers::*;
