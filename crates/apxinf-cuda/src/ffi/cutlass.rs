//! Raw bindings for CUTLASS adapters.

use std::ffi::c_void;

use super::cuda::{cudaError_t, cudaStream_t};

extern "C" {
    #[cfg(apxinf_cutlass_fmha)]
    pub fn apxinf_static_prepare_cutlass_mha_f16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        batches: i32,
        query_tokens: i32,
        key_tokens: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        stream: cudaStream_t,
    ) -> i32;
    #[cfg(apxinf_cutlass_fmha)]
    pub fn apxinf_static_cutlass_mha_f16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        batches: i32,
        query_tokens: i32,
        key_tokens: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        stream: cudaStream_t,
    ) -> i32;
    #[cfg(apxinf_cutlass_fmha)]
    pub fn apxinf_static_prepare_cutlass_mha_packed_qkv_f16(
        qkv: *const c_void,
        output: *mut c_void,
        batches: i32,
        tokens: i32,
        heads: i32,
        head_dim: i32,
        stream: cudaStream_t,
    ) -> i32;
    #[cfg(apxinf_cutlass_fmha)]
    pub fn apxinf_static_cutlass_mha_packed_qkv_f16(
        qkv: *const c_void,
        output: *mut c_void,
        batches: i32,
        tokens: i32,
        heads: i32,
        head_dim: i32,
        stream: cudaStream_t,
    ) -> i32;
    #[cfg(apxinf_cutlass_fmha)]
    pub fn apxinf_static_prepare_cutlass_mha_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        batches: i32,
        query_tokens: i32,
        key_tokens: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        stream: cudaStream_t,
    ) -> i32;
    #[cfg(apxinf_cutlass_fmha)]
    pub fn apxinf_static_cutlass_mha_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        batches: i32,
        query_tokens: i32,
        key_tokens: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        stream: cudaStream_t,
    ) -> i32;
    #[cfg(apxinf_cutlass_gemm)]
    pub fn apxinf_static_cutlass_fp8_gemm_f16(
        activation: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        tactic: i32,
        stream: cudaStream_t,
    ) -> i32;
    #[cfg(apxinf_cutlass_gemm)]
    pub fn apxinf_static_cutlass_fp8_gemm_geglu_e4m3(
        activation: *const c_void,
        packed_weight: *const c_void,
        gate: *const c_void,
        output: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        full_n: i32,
        alpha: f32,
        output_scale: f32,
        tactic: i32,
        stream: cudaStream_t,
    ) -> i32;
    #[cfg(apxinf_cutlass_gemm)]
    pub fn apxinf_static_cutlass_fp8_dual_gemm_geglu_e4m3(
        activation: *const c_void,
        interleaved_weight: *const c_void,
        output: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        full_n: i32,
        alpha: f32,
        output_scale: f32,
        stream: cudaStream_t,
    ) -> i32;
    #[cfg(apxinf_cutlass_bf16_sm89)]
    pub fn apxinf_static_cutlass_bf16_interleaved_geglu_sm89(
        activation: *const c_void,
        interleaved_weight: *const c_void,
        output: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        full_n: i32,
        tactic: i32,
        stream: cudaStream_t,
    ) -> i32;
    #[cfg(any(apxinf_cutlass_gemm, apxinf_cutlass_bf16_sm89))]
    pub fn apxinf_static_cutlass_bf16_gemm_geglu(
        activation: *const c_void,
        packed_weight: *const c_void,
        gate: *const c_void,
        output: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        full_n: i32,
        tactic: i32,
        stream: cudaStream_t,
    ) -> i32;
    #[cfg(apxinf_cutlass_gemm)]
    pub fn apxinf_static_cutlass_bf16_dual_gemm_geglu(
        activation: *const c_void,
        interleaved_weight: *const c_void,
        output: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        full_n: i32,
        stream: cudaStream_t,
    ) -> i32;
    #[cfg(apxinf_cutlass_int8_sm80)]
    pub fn apxinf_static_cutlass_int8_gemm_bf16(
        activation: *const c_void,
        weight_output_major: *const c_void,
        row_scales: *const c_void,
        column_scales: *const c_void,
        output: *mut c_void,
        rows: i32,
        output_dim: i32,
        input_dim: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}
