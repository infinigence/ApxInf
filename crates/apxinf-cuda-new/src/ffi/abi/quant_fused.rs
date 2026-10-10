use std::ffi::c_void;

use super::types::CudaStream;

unsafe extern "C" {
    pub(crate) fn apxinf_quant_rms_norm_rows_bf16_e4m3(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        scales: *mut c_void,
        rows: i32,
        input_cols: i32,
        output_cols: i32,
        eps: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_quant_bias_residual_rms_norm_rows_bf16_e4m3(
        projection: *const c_void,
        bias: *const c_void,
        residual: *const c_void,
        weight: *const c_void,
        hidden: *mut c_void,
        normalized: *mut c_void,
        scales: *mut c_void,
        rows: i32,
        cols: i32,
        output_cols: i32,
        eps: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_quant_swiglu_rows_bf16_e4m3(
        gate_up: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        scales: *mut c_void,
        rows: i32,
        input_cols: i32,
        inner: i32,
        output_cols: i32,
        stream: CudaStream,
    ) -> i32;
}
