#pragma once
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace apxinf::attention::kernels {
__global__ void wide_bf16_to_float(const __nv_bfloat16* input, float* output, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) output[i] = __bfloat162float(input[i]);
}
__global__ void wide_float_to_bf16(const float* input, __nv_bfloat16* output, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) output[i] = __float2bfloat16(input[i]);
}
__global__ void wide_softmax_serial(float* probabilities, int rows, int keys) {
  const int64_t row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  float* scores = probabilities + int64_t(row) * keys;
  float maximum = -CUDART_INF_F;
  for (int key = 0; key < keys; ++key) maximum = fmaxf(maximum, scores[key]);
  float sum = 0.0f;
  for (int key = 0; key < keys; ++key) {
    const float value = expf(scores[key] - maximum);
    scores[key] = value;
    sum += value;
  }
  for (int key = 0; key < keys; ++key) scores[key] /= sum;
}

// Full FP32 probabilities; only the reduction tree differs from the serial path.
__global__ void wide_softmax_parallel(float* probabilities, int keys) {
  __shared__ float partial[8];
  const int row = blockIdx.x, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  float* p = probabilities + int64_t(row) * keys;
  float maximum = -CUDART_INF_F;
  for (int k = threadIdx.x; k < keys; k += 256) maximum = fmaxf(maximum, p[k]);
  for (int delta = 16; delta; delta /= 2)
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, delta));
  if (lane == 0) partial[warp] = maximum;
  __syncthreads();
  if (warp == 0) {
    maximum = lane < 8 ? partial[lane] : -CUDART_INF_F;
    for (int delta = 16; delta; delta /= 2)
      maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, delta));
    if (lane == 0) partial[0] = maximum;
  }
  __syncthreads();
  maximum = partial[0];
  float sum = 0.f;
  for (int k = threadIdx.x; k < keys; k += 256) {
    const float value = expf(p[k] - maximum);
    p[k] = value;
    sum += value;
  }
  __syncthreads();
  for (int delta = 16; delta; delta /= 2) sum += __shfl_down_sync(0xffffffff, sum, delta);
  if (lane == 0) partial[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    sum = lane < 8 ? partial[lane] : 0.f;
    for (int delta = 16; delta; delta /= 2) sum += __shfl_down_sync(0xffffffff, sum, delta);
    if (lane == 0) partial[0] = sum;
  }
  __syncthreads();
  sum = partial[0];
  for (int k = threadIdx.x; k < keys; k += 256) p[k] /= sum;
}
}  // namespace apxinf::attention::kernels
