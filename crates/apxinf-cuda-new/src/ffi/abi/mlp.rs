use std::ffi::c_void;

use super::types::CudaStream;

unsafe extern "C" {
    pub(crate) fn apxinf_rms_norm_bf16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        rows: i64,
        width: i64,
        epsilon: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_swiglu_bf16(
        fused_gate_up: *const c_void,
        output: *mut c_void,
        rows: i64,
        width: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_quantize_fp8_per_tensor(
        input: *const c_void,
        output: *mut c_void,
        count: i64,
        input_scale: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_fp8_gemv(
        weight: *const c_void,
        activation: *const c_void,
        output: *mut c_void,
        n: i64,
        k: i64,
        alpha: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_nvfp4_gemv(
        weight: *const c_void,
        weight_scales: *const c_void,
        activation: *const c_void,
        activation_scales: *const c_void,
        output: *mut c_void,
        n: i64,
        k: i64,
        alpha: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_add_bf16(
        addend: *const c_void,
        accumulator: *mut c_void,
        count: i64,
        stream: CudaStream,
    ) -> i32;
}
