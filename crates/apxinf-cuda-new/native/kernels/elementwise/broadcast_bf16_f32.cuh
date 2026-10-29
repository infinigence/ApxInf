#pragma once

#include <cuda_bf16.h>
#include <cstdint>

// Broadcast a BF16 bias vector to FP32 accumulator rows without rounding.
static __global__ void broadcast_bf16_f32_rows_kernel(
    const __nv_bfloat16* bias, float* output, int64_t count, int cols) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t i = index; i < count; i += stride) {
    output[i] = __bfloat162float(bias[i % cols]);
  }
}
