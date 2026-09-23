// Copyright 2026 apxinf contributors.
// Stable C ABI adapter for the CUTLASS SM100/SM110 FP8 GEMM operator.

#include "../kernels/cutlass/fp8_operators_sm100.h"

extern "C" int apxinf_static_cutlass_fp8_gemm_f16(
    const void* activation, const void* weight, void* output,
    int m, int n, int k, float alpha, int tactic, cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::fp8_gemm_f16(
      activation, weight, output, m, n, k, alpha, tactic, stream);
}

extern "C" int apxinf_static_cutlass_fp8_gemm_bf16(
    const void* activation, const void* weight, void* output,
    int m, int n, int k, float alpha, int tactic, cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::fp8_gemm_bf16(
      activation, weight, output, m, n, k, alpha, tactic, stream);
}

extern "C" int apxinf_dynamic_cutlass_fp8_gemm_bf16(
    const void* activation, const void* weight_nk,
    const float* activation_scales, const float* weight_scales,
    const void* bias, void* output, int m, int n, int k, int tactic,
    cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::fp8_rowwise_gemm_bf16(
      activation, weight_nk, activation_scales, weight_scales, bias, output,
      m, n, k, tactic, stream);
}

extern "C" int apxinf_static_cutlass_fp8_gemm_geglu_e4m3(
    const void* activation, const void* packed_weight, const void* gate,
    void* output, int m, int n, int k, int full_n, float alpha,
    float output_scale, int tactic, cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::fp8_gemm_geglu_e4m3(
      activation, packed_weight, gate, output, m, n, k, full_n, alpha,
      output_scale, tactic, stream);
}

extern "C" int apxinf_static_cutlass_fp8_dual_gemm_geglu_e4m3(
    const void* activation, const void* interleaved_weight, void* output,
    int m, int n, int k, int full_n, float alpha, float output_scale,
    cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::fp8_dual_geglu_detail::production_dual_geglu(
      activation, interleaved_weight, output, m, n, k, full_n, alpha,
      output_scale, stream);
}
