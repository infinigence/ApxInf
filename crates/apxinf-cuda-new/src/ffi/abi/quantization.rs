use std::ffi::c_void;

pub(crate) use super::types::CudaStream;
use super::types::Runtime;

pub(crate) const SPEC_VERSION: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Spec {
    pub version: u32,
    pub semantic: u32,
    pub input_dtype: u32,
    pub output_dtype: u32,
    pub scale_dtype: u32,
    pub input_alignment: u32,
    pub secondary_alignment: u32,
    pub output_alignment: u32,
    pub scales_alignment: u32,
    pub rows: i64,
    pub input_cols: i64,
    pub output_cols: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bindings {
    pub input: *const c_void,
    pub secondary: *const c_void,
    pub output: *mut c_void,
    pub scales: *mut f32,
    pub stream: CudaStream,
    pub scale: f32,
}

unsafe extern "C" {
    pub(crate) fn apxinf_quantization_fp8_per_tensor(
        input: *const c_void,
        output: *mut c_void,
        count: i64,
        input_scale: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_quantization_launch(
        runtime: Runtime,
        spec: *const Spec,
        bindings: *const Bindings,
    ) -> i32;
}

mod fixed_launchers {
    use std::ffi::c_void;

    use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

    unsafe extern "C" {
        pub(crate) fn apxinf_quantization_concat_rows_quantize_bf16_e4m3(
            first: *const c_void,
            second: *const c_void,
            output: *mut c_void,
            first_rows: i32,
            second_rows: i32,
            cols: i32,
            scale: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_quantization_dequantize_int32_bf16(
            accumulators: *const c_void,
            row_scales: *const c_void,
            column_scales: *const c_void,
            output: *mut c_void,
            rows: i32,
            cols: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_quantization_quantize_rows_bf16_int8(
            input: *const c_void,
            output: *mut c_void,
            scales: *mut c_void,
            rows: i32,
            cols: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_quantization_quantize_rows_bf16_int8_packed4(
            input: *const c_void,
            output: *mut c_void,
            scales: *mut c_void,
            rows: i32,
            cols: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_quantization_bias_gelu_quantize_rows_bf16_int8(
            input: *const c_void,
            bias: *const c_void,
            output: *mut c_void,
            scales: *mut c_void,
            rows: i32,
            cols: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[cfg(test)]
        pub(crate) fn apxinf_quantization_silu_mul_quantize_rows_bf16_int8(
            gate: *const c_void,
            up: *const c_void,
            output: *mut c_void,
            scales: *mut c_void,
            rows: i32,
            cols: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[cfg(test)]
        pub(crate) fn apxinf_quantization_silu_mul_quantize_rows_bf16_int8_packed4(
            gate: *const c_void,
            up: *const c_void,
            output: *mut c_void,
            scales: *mut c_void,
            rows: i32,
            cols: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
    }
}
pub(crate) use fixed_launchers::*;
