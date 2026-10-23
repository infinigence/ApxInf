//! Internal bitwise-regression references. This module is compiled only in tests.
//! The `apxinf_reference_*` C symbols are not a supported production ABI.
use std::ffi::c_void;

use super::cuda::{cudaError_t, cudaStream_t};

extern "C" {
    pub fn apxinf_reference_layer_norm_quant_bf16_e4m3(
        input: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_reference_bias_gelu_quant_bf16_e4m3(
        input: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_reference_bias_residual_layer_norm_quant_bf16_e4m3(
        projection: *const c_void,
        projection_bias: *const c_void,
        residual: *const c_void,
        norm_weight: *const c_void,
        norm_bias: *const c_void,
        hidden: *mut c_void,
        normalized: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn apxinf_reference_qkv_split_bias_vision_rope_precomputed_bf16(
        qkv: *const c_void,
        bias: *const c_void,
        q_out: *mut c_void,
        k_out: *mut c_void,
        v_out: *mut c_void,
        head_dim: u32,
        n_heads: u32,
        seq_len: u32,
        rotation_table: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[cfg(apxinf_cutlass_gemm)]
    pub fn apxinf_reference_cutlass_fp8_gemm_bf16_m41(
        activation: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        stream: cudaStream_t,
    ) -> i32;
}
