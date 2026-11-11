#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace apxinf::relative_attention::kernels {

template <class T>
__device__ __forceinline__ float to_float(T value);
template <>
__device__ __forceinline__ float to_float<float>(float value) { return value; }
template <>
__device__ __forceinline__ float to_float(__half value) { return __half2float(value); }
template <>
__device__ __forceinline__ float to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <class T>
__device__ __forceinline__ T from_float(float value);
template <>
__device__ __forceinline__ float from_float<float>(float value) { return value; }
template <>
__device__ __forceinline__ __half from_float<__half>(float value) {
  return __float2half(value);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

/* One block per (batch, head, query position).  `scores` is dynamic shared
 * storage of `length` floats. */
template <class T>
__global__ void relative_attention_kernel(
    const T* __restrict__ query, const T* __restrict__ key,
    const T* __restrict__ value, const T* __restrict__ relative_key,
    const T* __restrict__ relative_value, T* __restrict__ output,
    int64_t batch, int64_t length, int64_t heads, int64_t head_dim,
    int64_t window, float scale) {
  extern __shared__ float scores[];
  const int64_t block = blockIdx.x;
  const int64_t head_index = block % heads;
  const int64_t rest = block / heads;
  const int64_t position = rest % length;
  const int64_t batch_index = rest / length;

  const int64_t row_base =
      (batch_index * length + position) * heads * head_dim +
      head_index * head_dim;
  const T* q = query + row_base;
  const T* k_base = key + (batch_index * length * heads + head_index) * head_dim;
  const T* v_base = value + (batch_index * length * heads + head_index) * head_dim;
  T* out = output + row_base;
  const int64_t relative_width = 2 * window + 1;

  float local_max = -INFINITY;
  for (int64_t j = threadIdx.x; j < length; j += blockDim.x) {
    float dot = 0.0f;
    const T* k = k_base + j * heads * head_dim;
    for (int64_t d = 0; d < head_dim; ++d) {
      dot += to_float<T>(q[d]) * to_float<T>(k[d]);
    }
    const int64_t relative = j - position + window;
    if (relative >= 0 && relative < relative_width) {
      const T* rk = relative_key + relative * head_dim;
      for (int64_t d = 0; d < head_dim; ++d) {
        dot += to_float<T>(q[d]) * to_float<T>(rk[d]);
      }
    }
    const float score = dot * scale;
    scores[j] = score;
    if (score > local_max) local_max = score;
  }

  __shared__ float reduction[32];
  float warp_max = local_max;
  for (int offset = 16; offset > 0; offset >>= 1) {
    warp_max = fmaxf(warp_max, __shfl_down_sync(0xFFFFFFFFu, warp_max, offset));
  }
  const int warp = threadIdx.x >> 5;
  const int warps = (blockDim.x + 31) >> 5;
  if ((threadIdx.x & 31) == 0) reduction[warp] = warp_max;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = reduction[0];
    for (int i = 1; i < warps; ++i) total = fmaxf(total, reduction[i]);
    reduction[0] = total;
  }
  __syncthreads();
  const float maximum = reduction[0];

  float local_sum = 0.0f;
  for (int64_t j = threadIdx.x; j < length; j += blockDim.x) {
    const float e = expf(scores[j] - maximum);
    scores[j] = e;
    local_sum += e;
  }
  float warp_sum = local_sum;
  for (int offset = 16; offset > 0; offset >>= 1) {
    warp_sum += __shfl_down_sync(0xFFFFFFFFu, warp_sum, offset);
  }
  if ((threadIdx.x & 31) == 0) reduction[warp] = warp_sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int i = 0; i < warps; ++i) total += reduction[i];
    reduction[0] = total;
  }
  __syncthreads();
  const float denominator = reduction[0];

  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float accumulator = 0.0f;
    for (int64_t j = 0; j < length; ++j) {
      const float probability = scores[j] / denominator;
      float contribution = to_float<T>(v_base[j * heads * head_dim + d]);
      const int64_t relative = j - position + window;
      if (relative >= 0 && relative < relative_width) {
        contribution += to_float<T>(relative_value[relative * head_dim + d]);
      }
      accumulator += probability * contribution;
    }
    out[d] = from_float<T>(accumulator);
  }
}

}  // namespace apxinf::relative_attention::kernels
