#pragma once

#include <cuda_bf16.h>
#include <cstdint>

__global__ void cast_f32_bf16_kernel(
    const float* input, __nv_bfloat16* output, int64_t count) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = index; i < count; i += stride) {
    output[i] = __float2bfloat16(input[i]);
  }
}

__global__ void cast_bf16_f32_kernel(
    const __nv_bfloat16* input, float* output, int64_t count) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = index; i < count; i += stride) {
    output[i] = __bfloat162float(input[i]);
  }
}
