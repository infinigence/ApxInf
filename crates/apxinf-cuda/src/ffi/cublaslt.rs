//! Raw bindings for the cuBLASLt FP8 adapter.

use std::ffi::c_void;

use super::cublas::cublasStatus_t;
use super::cuda::{cudaError_t, cudaStream_t};

extern "C" {
    pub fn apxinf_static_prepare_bf16_gemm(m: i32, n: i32, k: i32) -> cublasStatus_t;
    pub fn apxinf_static_set_cublaslt_bf16_gemm_heuristic(
        m: i32,
        n: i32,
        k: i32,
        heuristic_rank: i32,
    ) -> cublasStatus_t;
    pub fn apxinf_static_set_cublaslt_bf16_gemm_custom(
        m: i32,
        n: i32,
        k: i32,
        tile_id: i32,
        custom_option: i32,
        stages_id: i32,
        cluster_shape_id: i32,
    ) -> cublasStatus_t;
    pub fn apxinf_static_set_cublaslt_bf16_gemm_split_custom(
        m: i32,
        n: i32,
        k: i32,
        tile_id: i32,
        custom_option: i32,
        stages_id: i32,
        cluster_shape_id: i32,
    ) -> cublasStatus_t;
    pub fn apxinf_static_prepare_bf16_gemm_split(m: i32, n: i32, k: i32) -> cublasStatus_t;
    pub fn apxinf_static_bf16_gemm_split(
        activation: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        stream: cudaStream_t,
    ) -> cublasStatus_t;
    pub fn apxinf_static_bf16_gemm_split_first(
        activation: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        stream: cudaStream_t,
    ) -> cublasStatus_t;
    pub fn apxinf_static_bf16_gemm(
        activation: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        stream: cudaStream_t,
    ) -> cublasStatus_t;
    pub fn apxinf_static_autotune_cublaslt_bf16_gemm(
        activation: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        max_algorithms: i32,
        warmup_iterations: i32,
        benchmark_iterations: i32,
        did_tune: *mut i32,
        returned_algorithms: *mut i32,
        best_rank: *mut i32,
        default_ms: *mut f32,
        best_ms: *mut f32,
        stream: cudaStream_t,
    ) -> cublasStatus_t;
    /// Report whether the selected CUDA device has native E4M3 Tensor Core
    /// GEMM support. The implementation uses CUDA's symbolic device
    /// attributes so Rust does not depend on their numeric ABI values.
    pub fn apxinf_static_native_fp8_supported(device: i32, supported: *mut i32) -> cudaError_t;
    /// Install immutable cuBLASLt resources for one FP8 GEMM shape.
    pub fn apxinf_static_prepare_fp8_gemm_f16(m: i32, n: i32, k: i32) -> cublasStatus_t;
    pub fn apxinf_static_prepare_fp8_gemm_split_f16(m: i32, n: i32, k: i32) -> cublasStatus_t;
    /// Install a fused GELU plan and bind its stable bias/scale resources.
    pub fn apxinf_static_prepare_fp8_gemm_bias_gelu_e4m3(
        bias: *const c_void,
        m: i32,
        n: i32,
        k: i32,
        output_scale: f32,
    ) -> cublasStatus_t;
    /// Install a fused residual plan and bind its stable bias resource.
    pub fn apxinf_static_prepare_fp8_gemm_bias_residual_f16(
        bias: *const c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> cublasStatus_t;
    pub fn apxinf_static_prepare_fp8_gemm_bias_f16(
        bias: *const c_void,
        m: i32,
        n: i32,
        k: i32,
    ) -> cublasStatus_t;
    /// Static E4M3 x E4M3 GEMM with FP16 output. Returns cublasStatus_t.
    pub fn apxinf_static_fp8_gemm_f16(
        activation: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        stream: cudaStream_t,
    ) -> cublasStatus_t;
    pub fn apxinf_static_fp8_gemm_split_f16(
        activation: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        stream: cudaStream_t,
    ) -> cublasStatus_t;
    pub fn apxinf_static_fp8_gemm_split_first_f16(
        activation: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        stream: cudaStream_t,
    ) -> cublasStatus_t;
    /// Static E4M3 GEMM with fused FP16 bias, GELU, and E4M3 output.
    pub fn apxinf_static_fp8_gemm_bias_gelu_e4m3(
        activation: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        output_scale: f32,
        stream: cudaStream_t,
    ) -> cublasStatus_t;
    /// Static E4M3 GEMM with fused FP16 bias and residual accumulation.
    pub fn apxinf_static_fp8_gemm_bias_residual_f16(
        activation: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        residual: *const c_void,
        output: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        stream: cudaStream_t,
    ) -> cublasStatus_t;
    pub fn apxinf_static_fp8_gemm_bias_f16(
        activation: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        stream: cudaStream_t,
    ) -> cublasStatus_t;
    pub fn apxinf_static_set_cublaslt_gemm_heuristic(
        m: i32,
        n: i32,
        k: i32,
        heuristic_rank: i32,
    ) -> cublasStatus_t;
    pub fn apxinf_static_set_cublaslt_fp8_fused_heuristic(
        m: i32,
        n: i32,
        k: i32,
        epilogue: i32,
        heuristic_rank: i32,
    ) -> cublasStatus_t;
    pub fn apxinf_static_set_cublaslt_fp8_gemm_custom(
        m: i32,
        n: i32,
        k: i32,
        tile_id: i32,
        custom_option: i32,
        stages_id: i32,
        cluster_shape_id: i32,
    ) -> cublasStatus_t;
    pub fn apxinf_static_set_cublaslt_fp8_gemm_split_custom(
        m: i32,
        n: i32,
        k: i32,
        tile_id: i32,
        custom_option: i32,
        stages_id: i32,
        cluster_shape_id: i32,
    ) -> cublasStatus_t;
    pub fn apxinf_static_set_cublaslt_fp8_gemm_bias_custom(
        m: i32,
        n: i32,
        k: i32,
        epilogue: i32,
        tile_id: i32,
        custom_option: i32,
        stages_id: i32,
        cluster_shape_id: i32,
    ) -> cublasStatus_t;
    pub fn apxinf_static_autotune_cublaslt_fp8_gemm_f16(
        activation: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        l2_eviction_buffer: *mut c_void,
        l2_eviction_bytes: usize,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        max_algorithms: i32,
        warmup_iterations: i32,
        benchmark_iterations: i32,
        returned_algorithms: *mut i32,
        milliseconds: *mut f32,
        stream: cudaStream_t,
    ) -> cublasStatus_t;
}
