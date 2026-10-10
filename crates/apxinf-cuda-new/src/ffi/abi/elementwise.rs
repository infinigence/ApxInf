use std::ffi::c_void;

use super::types::CudaStream;

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
}
