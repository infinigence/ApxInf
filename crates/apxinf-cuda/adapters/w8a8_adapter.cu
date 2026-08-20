// Copyright 2026 apxinf contributors.
// Stable C ABI and CUDA launch policy for dynamic W8A8 operators.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {
#include "../kernels/custom/math.cuh"
#include "../kernels/custom/reduction.cuh"
#include "../kernels/custom/quantization.cuh"
}  // namespace

extern "C" cudaError_t apxinf_static_quantize_rows_bf16_int8(
    const void* input, void* output, void* scales,
    int rows, int cols, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr ||
      rows <= 0 || cols <= 0) {
    return cudaErrorInvalidValue;
  }
  quantize_rows_bf16_int8_kernel<<<rows, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<int8_t*>(output), static_cast<float*>(scales), rows, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_dequantize_int32_bf16(
    const void* accumulators, const void* row_scales,
    const void* column_scales, void* output,
    int rows, int cols, cudaStream_t stream) {
  if (accumulators == nullptr || row_scales == nullptr ||
      column_scales == nullptr || output == nullptr || rows <= 0 || cols <= 0) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid((cols + kThreads - 1) / kThreads, rows);
  dequantize_int32_bf16_kernel<<<grid, kThreads, 0, stream>>>(
      static_cast<const int32_t*>(accumulators),
      static_cast<const float*>(row_scales),
      static_cast<const float*>(column_scales),
      static_cast<__nv_bfloat16*>(output), rows, cols);
  return cudaGetLastError();
}
