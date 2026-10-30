unsafe extern "C" {
    pub(crate) fn apxinf_activation_fixed_decode_silu_f32(
        input: *const c_void,
        output: *mut c_void,
        count: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_activation_fixed_decode_silu_bf16(
        input: *const c_void,
        output: *mut c_void,
        count: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_activation_fixed_decode_silu_mul_bf16(
        gate_up: *const c_void,
        output: *mut c_void,
        inter: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}
use std::ffi::c_void;

use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

mod fixed_launchers {
    use std::ffi::c_void;

    use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

    unsafe extern "C" {
        pub(crate) fn apxinf_activation_gelu_exact_bf16(
            input: *const c_void,
            output: *mut c_void,
            count: i64,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_activation_swiglu_bf16_rounded(
            gate_up: *const c_void,
            output: *mut c_void,
            rows: i32,
            inner: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_activation_swiglu_bf16_rounded_pair_interleaved(
            gate_up: *const c_void,
            output: *mut c_void,
            rows: i32,
            inner: i32,
            pair: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_activation_swiglu_bf16(
            gate_up: *const c_void,
            output: *mut c_void,
            rows: i32,
            inner: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_activation_silu_mul_separate_bf16(
            gate: *const c_void,
            up: *const c_void,
            output: *mut c_void,
            count: u32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_activation_silu_mul_quant_bf16_e4m3(
            gate: *const c_void,
            up: *const c_void,
            output: *mut c_void,
            count: i64,
            scale: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_activation_bias_gelu_quant_bf16_e4m3(
            input: *const c_void,
            bias: *const c_void,
            output: *mut c_void,
            rows: i32,
            cols: i32,
            scale: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_activation_swiglu_f16(
            gate_up: *const c_void,
            output: *mut c_void,
            rows: i32,
            inner: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_activation_bias_gelu_f16(
            input: *const c_void,
            bias: *const c_void,
            output: *mut c_void,
            rows: i32,
            cols: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_activation_bias_silu_f16(
            input: *const c_void,
            bias: *const c_void,
            output: *mut c_void,
            rows: i32,
            cols: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
    }
}
pub(crate) use fixed_launchers::*;
