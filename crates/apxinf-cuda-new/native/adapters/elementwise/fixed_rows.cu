// Copyright 2026 ApxInf contributors.
// Fixed launchers for the row-reduction elementwise operators; the kernels
// live in `kernels/elementwise/fixed_rows.cuh`.
#include "../../kernels/elementwise/fixed_rows.cuh"

// dtype: 0 f32, 1 f16, 2 bf16.
extern "C" cudaError_t apxinf_elementwise_mean_rows(
    const void* input, void* output, int64_t rows, int64_t cols,
    uint32_t dtype, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || rows <= 0 || cols <= 0 ||
      rows > INT32_MAX || cols > INT32_MAX) {
    return cudaErrorInvalidValue;
  }
  switch (dtype) {
    case 0:
      mean_rows_kernel<float><<<1, kRowsThreads, 0, stream>>>(
          static_cast<const float*>(input), static_cast<float*>(output),
          static_cast<int>(rows), static_cast<int>(cols));
      break;
    case 1:
      mean_rows_kernel<__half><<<1, kRowsThreads, 0, stream>>>(
          static_cast<const __half*>(input), static_cast<__half*>(output),
          static_cast<int>(rows), static_cast<int>(cols));
      break;
    case 2:
      mean_rows_kernel<__nv_bfloat16><<<1, kRowsThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(input),
          static_cast<__nv_bfloat16*>(output), static_cast<int>(rows),
          static_cast<int>(cols));
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

// dtype: 0 f32, 1 f16, 2 bf16.
extern "C" cudaError_t apxinf_elementwise_argmax_rows(
    const void* logits, void* index, int64_t rows, int64_t cols,
    uint32_t dtype, cudaStream_t stream) {
  if (logits == nullptr || index == nullptr || rows <= 0 || cols <= 0 ||
      rows > INT32_MAX || cols > INT32_MAX) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid(static_cast<unsigned>(rows));
  switch (dtype) {
    case 0:
      argmax_rows_kernel<float><<<grid, kRowsThreads, 0, stream>>>(
          static_cast<const float*>(logits), static_cast<int32_t*>(index),
          static_cast<int>(rows), static_cast<int>(cols));
      break;
    case 1:
      argmax_rows_kernel<__half><<<grid, kRowsThreads, 0, stream>>>(
          static_cast<const __half*>(logits), static_cast<int32_t*>(index),
          static_cast<int>(rows), static_cast<int>(cols));
      break;
    case 2:
      argmax_rows_kernel<__nv_bfloat16><<<grid, kRowsThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(logits),
          static_cast<int32_t*>(index), static_cast<int>(rows),
          static_cast<int>(cols));
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

// dtype: 0 f32, 1 f16, 2 bf16; zero bytes are zero in every supported dtype,
// so the fill is a memset.
extern "C" cudaError_t apxinf_elementwise_fill_zero(
    void* tensor, int64_t elements, uint32_t dtype, cudaStream_t stream) {
  if (tensor == nullptr || elements <= 0 || elements > INT32_MAX) {
    return cudaErrorInvalidValue;
  }
  size_t element_bytes = 0;
  switch (dtype) {
    case 0: element_bytes = 4; break;
    case 1: element_bytes = 2; break;
    case 2: element_bytes = 2; break;
    case 3: element_bytes = 4; break;
    case 4: element_bytes = 1; break;
    default: return cudaErrorInvalidValue;
  }
  return cudaMemsetAsync(tensor, 0,
                         static_cast<size_t>(elements) * element_bytes, stream);
}
