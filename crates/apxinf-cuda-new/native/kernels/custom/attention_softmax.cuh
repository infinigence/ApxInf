#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

namespace apxinf::attention::kernels {

// Block reduction for the F32 provider. All threads consume shared scratch
// before another reduction can reuse it.
template <bool Maximum>
__device__ inline float f32_softmax_reduce(float value, float* scratch) {
  for (int delta = 16; delta; delta /= 2) {
    const float other = __shfl_down_sync(0xffffffff, value, delta);
    value = Maximum ? fmaxf(value, other) : value + other;
  }
  if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = value;
  __syncthreads();
  if (threadIdx.x == 0) {
    float result = scratch[0];
    for (int warp = 1; warp < 4; ++warp)
      result = Maximum ? fmaxf(result, scratch[warp]) : result + scratch[warp];
    scratch[0] = result;
  }
  __syncthreads();
  const float result = scratch[0];
  __syncthreads();
  return result;
}

__global__ void attention_f32_softmax(float* scores, int queries, int keys,
                                     bool causal, int query_start) {
  __shared__ float scratch[4];
  const int64_t row = blockIdx.x;
  const int64_t valid = causal
      ? min(static_cast<int64_t>(keys), static_cast<int64_t>(query_start) + row % queries + 1)
      : keys;
  float* values = scores + row * keys;
  float maximum = -CUDART_INF_F;
  for (int key = threadIdx.x; key < valid; key += 128)
    maximum = fmaxf(maximum, values[key]);
  maximum = f32_softmax_reduce<true>(maximum, scratch);
  float sum = 0.0F;
  for (int key = threadIdx.x; key < keys; key += 128) {
    const float value = key < valid && maximum != -CUDART_INF_F
        ? expf(values[key] - maximum) : 0.0F;
    values[key] = value;
    sum += value;
  }
  sum = f32_softmax_reduce<false>(sum, scratch);
  for (int key = threadIdx.x; key < keys; key += 128)
    values[key] = sum == 0.0F ? 0.0F : values[key] / sum;
}

inline cudaError_t launch_attention_f32_softmax(float* scores, int batch,
    int heads, int queries, int keys, bool causal, int query_start,
    cudaStream_t stream) {
  const int64_t rows = static_cast<int64_t>(batch) * heads * queries;
  attention_f32_softmax<<<static_cast<unsigned>(rows), 128, 0, stream>>>(
      scores, queries, keys, causal, query_start);
  return cudaGetLastError();
}

}  // namespace apxinf::attention::kernels
