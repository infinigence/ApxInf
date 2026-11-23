use std::ffi::c_void;

use super::types::CudaStream;
use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

unsafe extern "C" {
    pub(crate) fn apxinf_elementwise_activation_bf16(
        input: *const c_void,
        output: *mut c_void,
        count: i64,
        activation: i32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_elementwise_mul_bf16(
        a: *const c_void,
        b: *const c_void,
        output: *mut c_void,
        count: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_elementwise_add_bf16(
        a: *const c_void,
        b: *const c_void,
        output: *mut c_void,
        count: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_elementwise_scale_bf16(
        input: *const c_void,
        output: *mut c_void,
        count: i64,
        factor: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_elementwise_add_bias_bf16(
        input: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        rows: i64,
        cols: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_elementwise_gather_rows_bf16(
        input: *const c_void,
        indices: *const c_void,
        output: *mut c_void,
        rows: i64,
        cols: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_elementwise_replace_rows_bf16(
        base: *const c_void,
        replacement: *const c_void,
        row_map: *const c_void,
        output: *mut c_void,
        rows: i64,
        cols: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_elementwise_bias_position_f32_bf16(
        projection: *const c_void,
        bias: *const c_void,
        position: *const c_void,
        output: *mut c_void,
        count: i64,
        cols: i32,
        tokens_per_view: i32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_elementwise_argmax_remap_bf16(
        logits: *const c_void,
        n: u32,
        remap: *const c_void,
        out: *mut c_void,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_elementwise_broadcast_bf16_f32_rows(
        bias: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        stream: CudaStream,
    ) -> i32;
}
unsafe extern "C" {
    pub(crate) fn apxinf_elementwise_fixed_decode_add_f32(
        a: *const c_void,
        b: *const c_void,
        output: *mut c_void,
        count: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_elementwise_fixed_decode_add_bf16(
        a: *const c_void,
        b: *const c_void,
        output: *mut c_void,
        count: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_elementwise_fixed_decode_mul_f32(
        a: *const c_void,
        b: *const c_void,
        output: *mut c_void,
        count: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_elementwise_fixed_decode_mul_bf16(
        a: *const c_void,
        b: *const c_void,
        output: *mut c_void,
        count: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_elementwise_fixed_decode_scale_f32(
        input: *const c_void,
        output: *mut c_void,
        count: u32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_elementwise_fixed_decode_scale_bf16(
        input: *const c_void,
        output: *mut c_void,
        count: u32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

mod fixed_launchers {
    use std::ffi::c_void;

    use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

    unsafe extern "C" {
        pub(crate) fn apxinf_elementwise_scatter_rows_bf16(
            source: *const c_void,
            rows: *const c_void,
            output: *mut c_void,
            row_count: i32,
            cols: i32,
            add: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_elementwise_bias_qkv_in_place_bf16(
            query: *mut c_void,
            key: *mut c_void,
            value: *mut c_void,
            query_bias: *const c_void,
            key_bias: *const c_void,
            value_bias: *const c_void,
            rows: i32,
            cols: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_elementwise_bias_f16(
            input: *const c_void,
            bias: *const c_void,
            output: *mut c_void,
            rows: i32,
            cols: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_elementwise_euler_update_f16(
            state: *const c_void,
            velocity: *const c_void,
            output: *mut c_void,
            count: i64,
            dt: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_elementwise_add_f16(
            a: *const c_void,
            b: *const c_void,
            output: *mut c_void,
            count: u32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_elementwise_scale_f16(
            input: *const c_void,
            output: *mut c_void,
            count: u32,
            scale: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
    }
}
pub(crate) use fixed_launchers::*;
