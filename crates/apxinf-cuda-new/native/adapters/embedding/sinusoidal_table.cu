// Copyright 2026 ApxInf contributors.
// Fixed launcher for the reference sinusoidal table; the kernel lives in
// `kernels/embedding/sinusoidal_table.cuh`.
#include "../../kernels/embedding/sinusoidal_table.cuh"

// dtype: 0 f32, 1 f16, 2 bf16.
extern "C" cudaError_t apxinf_embedding_sinusoidal_table(
    const void* times, void* output, int64_t rows, int64_t width,
    float scale, float log_step, uint32_t dtype, cudaStream_t stream) {
  if (times == nullptr || output == nullptr || rows <= 0 || width < 4 ||
      width % 2 != 0 || rows > INT32_MAX || width > INT32_MAX) {
    return cudaErrorInvalidValue;
  }
  const int64_t blocks = (rows * width + kTableThreads - 1) / kTableThreads;
  if (blocks > INT32_MAX) return cudaErrorInvalidValue;
  switch (dtype) {
    case 0:
      sinusoidal_table_kernel<float>
          <<<static_cast<int>(blocks), kTableThreads, 0, stream>>>(
              static_cast<const float*>(times), static_cast<float*>(output),
              rows, width, scale, log_step);
      break;
    case 1:
      sinusoidal_table_kernel<__half>
          <<<static_cast<int>(blocks), kTableThreads, 0, stream>>>(
              static_cast<const __half*>(times), static_cast<__half*>(output),
              rows, width, scale, log_step);
      break;
    case 2:
      sinusoidal_table_kernel<__nv_bfloat16>
          <<<static_cast<int>(blocks), kTableThreads, 0, stream>>>(
              static_cast<const __nv_bfloat16*>(times),
              static_cast<__nv_bfloat16*>(output), rows, width, scale,
              log_step);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}
