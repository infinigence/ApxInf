// Copyright 2026 apxinf contributors.
// Stable C ABI adapter for the CUTLASS SM80-family W8A8 GEMM operator.

#include "../kernels/cutlass/w8a8_gemm_sm80.cu"

extern "C" cudaError_t apxinf_static_cutlass_int8_gemm_bf16(
    const void* activation, const void* weight_output_major,
    const void* row_scales, const void* column_scales, void* output,
    int m, int n, int k, cudaStream_t stream) {
  return apxinf::cuda::cutlass_ops::w8a8_gemm_bf16(
      activation, weight_output_major, row_scales, column_scales, output,
      m, n, k, stream);
}
