use std::ffi::c_void;

pub(crate) use super::types::{CudaStream, Runtime};

pub(crate) const ACTIVATION_NONE: u32 = 0;
pub(crate) const ACTIVATION_GELU: u32 = 1;
pub(crate) const ACTIVATION_SILU: u32 = 2;
pub(crate) const ACTIVATION_GELU_EXACT: u32 = 3;
pub(crate) const ACTIVATION_MISH: u32 = 4;
pub(crate) const ACTIVATION_LEAKY_RELU: u32 = 5;
pub(crate) const ACTIVATION_TANH: u32 = 6;
pub(crate) const ACTIVATION_SIGMOID: u32 = 7;
pub(crate) const ACTIVATION_RELU: u32 = 8;
pub(crate) const ACTIVATION_LEAKY_RELU_ONE_PERCENT: u32 = 9;

unsafe extern "C" {
    pub(crate) fn apxinf_tts_bias_activation(
        runtime: Runtime,
        input: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        count: i64,
        cols: i64,
        dtype: u32,
        activation: u32,
        round_bias: u32,
        stream: CudaStream,
    ) -> i32;

    pub(crate) fn apxinf_tts_residual_add_bias(
        runtime: Runtime,
        input: *const c_void,
        bias: *const c_void,
        residual: *const c_void,
        output: *mut c_void,
        count: i64,
        cols: i64,
        dtype: u32,
        stream: CudaStream,
    ) -> i32;

    pub(crate) fn apxinf_tts_affine(
        runtime: Runtime,
        input: *const c_void,
        style: *const c_void,
        output: *mut c_void,
        count: i64,
        cols: i64,
        dtype: u32,
        stream: CudaStream,
    ) -> i32;

    pub(crate) fn apxinf_tts_glu_separate(
        runtime: Runtime,
        input: *const c_void,
        gate: *const c_void,
        output: *mut c_void,
        count: i64,
        dtype: u32,
        stream: CudaStream,
    ) -> i32;

    pub(crate) fn apxinf_tts_tanh_sigmoid_gate(
        runtime: Runtime,
        input: *const c_void,
        secondary: *const c_void,
        output: *mut c_void,
        count: i64,
        cols: i64,
        dtype: u32,
        has_secondary: u32,
        stream: CudaStream,
    ) -> i32;

    pub(crate) fn apxinf_tts_grn(
        runtime: Runtime,
        input: *const c_void,
        style: *const c_void,
        output: *mut c_void,
        rows: i64,
        cols: i64,
        dtype: u32,
        stream: CudaStream,
    ) -> i32;

    pub(crate) fn apxinf_tts_column_norm(
        runtime: Runtime,
        input: *const c_void,
        style: *const c_void,
        output: *mut c_void,
        rows: i64,
        cols: i64,
        dtype: u32,
        eps: f32,
        stream: CudaStream,
    ) -> i32;

    pub(crate) fn apxinf_tts_rope_apply(
        runtime: Runtime,
        input: *const c_void,
        cos_values: *const c_void,
        sin_values: *const c_void,
        output: *mut c_void,
        tokens: i64,
        time: i64,
        heads: i64,
        head_dim: i64,
        dtype: u32,
        stream: CudaStream,
    ) -> i32;

    pub(crate) fn apxinf_tts_sinusoidal(
        runtime: Runtime,
        times: *const c_void,
        output: *mut c_void,
        rows: i64,
        width: i64,
        scale: f32,
        log_step: f32,
        dtype: u32,
        stream: CudaStream,
    ) -> i32;

    pub(crate) fn apxinf_tts_mean_rows(
        runtime: Runtime,
        input: *const c_void,
        output: *mut c_void,
        rows: i64,
        cols: i64,
        dtype: u32,
        stream: CudaStream,
    ) -> i32;

    pub(crate) fn apxinf_tts_argmax_rows(
        runtime: Runtime,
        logits: *const c_void,
        index: *mut c_void,
        rows: i64,
        cols: i64,
        dtype: u32,
        stream: CudaStream,
    ) -> i32;

    pub(crate) fn apxinf_tts_write_kv_cache_token(
        runtime: Runtime,
        input: *const c_void,
        cache: *mut c_void,
        meta: *const u32,
        capacity: i64,
        row_elements: i64,
        dtype: u32,
        stream: CudaStream,
    ) -> i32;

    pub(crate) fn apxinf_tts_fill_zero(
        runtime: Runtime,
        tensor: *mut c_void,
        elements: i64,
        dtype: u32,
        stream: CudaStream,
    ) -> i32;
}
