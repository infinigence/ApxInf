use std::ffi::c_void;

pub(crate) use super::types::CudaStream;
use super::types::Runtime;

pub(crate) const SPEC_VERSION: u32 = 7;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Spec {
    pub version: u32,
    pub operation: u32,
    pub statistic: u32,
    pub affine: u32,
    pub gate: u32,
    pub quantization: u32,
    pub numerics: u32,
    pub dtype: u32,
    pub output_dtype: u32,
    pub has_native_output: u32,
    pub has_bias: u32,
    pub bias_order: u32,
    pub weight_offset: u32,
    pub input_alignment: u32,
    pub weight_alignment: u32,
    pub bias_alignment: u32,
    pub residual_alignment: u32,
    pub style_alignment: u32,
    pub hidden_alignment: u32,
    pub normalized_alignment: u32,
    pub quantized_alignment: u32,
    pub quantized_scales_alignment: u32,
    pub rows: i64,
    pub cols: i64,
    pub quantized_cols: i64,
    /// Structural predicate only: the scale value lives in [`Bindings`].
    pub output_scale_is_unit: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bindings {
    pub input: *const c_void,
    pub bias: *const c_void,
    pub residual: *const c_void,
    pub weight: *const c_void,
    pub norm_bias: *const c_void,
    pub norm_style: *const c_void,
    pub gate_style: *const c_void,
    pub hidden: *mut c_void,
    pub normalized: *mut c_void,
    pub quantized: *mut c_void,
    pub quantized_scales: *mut c_void,
    pub stream: CudaStream,
    pub eps: f32,
    pub output_scale: f32,
}

unsafe extern "C" {
    pub(crate) fn apxinf_norm_launch(
        runtime: Runtime,
        spec: *const Spec,
        bindings: *const Bindings,
    ) -> i32;
}

mod fixed_launchers {
    use std::ffi::c_void;

    use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

    unsafe extern "C" {
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_norm_gated_rms_silu_bf16(
            x: *const c_void,
            z: *const c_void,
            weight: *const c_void,
            out: *mut c_void,
            rows: i32,
            cols: i32,
            z_heads: i32,
            z_row_stride: i64,
            z_col_offset: i64,
            eps: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_norm_adaln_rms_norm_bf16(
            x: *const c_void,
            weight: *const c_void,
            scale: *const c_void,
            shift: *const c_void,
            out: *mut c_void,
            rows: i32,
            cols: i32,
            eps: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_norm_adaln_gate_residual_bf16(
            proj: *const c_void,
            residual: *const c_void,
            gate: *const c_void,
            out: *mut c_void,
            count: i64,
            cols: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_norm_adaln_gate_residual_rms_bf16(
            projection: *const c_void,
            residual: *const c_void,
            gate: *const c_void,
            weight: *const c_void,
            scale: *const c_void,
            shift: *const c_void,
            hidden: *mut c_void,
            normalized: *mut c_void,
            rows: i32,
            cols: i32,
            eps: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
    }
}
pub(crate) use fixed_launchers::*;
