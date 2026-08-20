// Copyright 2026 apxinf contributors.
// Stable C ABI adapter for the CUTLASS SM100/SM110 FP8 GEMM operator.

#include "../kernels/cutlass/fp8_gemm_sm100.cu"

extern "C" int apxinf_static_cutlass_fp8_gemm_f16(
    const void* activation, const void* weight, void* output,
    int m, int n, int k, float alpha, int tactic, cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::fp8_gemm_f16(
      activation, weight, output, m, n, k, alpha, tactic, stream);
}
