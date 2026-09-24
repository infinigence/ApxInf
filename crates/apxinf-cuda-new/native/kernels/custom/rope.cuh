#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace apxinf::rope::kernels {

template <class T>
__device__ float to_float(T value);

template <>
__device__ inline float to_float(__half value) {
  return __half2float(value);
}

template <>
__device__ inline float to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <class T>
__device__ T from_float(float value);

template <>
__device__ inline __half from_float(float value) {
  return __float2half(value);
}

template <>
__device__ inline __nv_bfloat16 from_float(float value) {
  return __float2bfloat16(value);
}

// Splits a packed [tokens, q_heads*head_dim + 2*kv_heads*head_dim] projection,
// adds the optional fused bias, and rotates Q and K.  V is copied unrotated.
//
// `kv_output_offset` is what makes this one kernel serve two call sites: at 0
// it writes K/V into fresh buffers (prefill), and at the current prefix length
// it appends them into a KV cache (decode).  The caller owns which buffer the
// k/v pointers refer to.
//
// Launch: grid (tokens, q_heads + 2 * kv_heads), block >= head_dim / 2.
template <class T>
__global__ void split_qkv_rope(const T* qkv, const T* bias, T* q, T* k, T* v,
                               int tokens, int q_heads, int kv_heads,
                               int head_dim, float theta, int position_offset,
                               int kv_output_offset) {
  const int token = blockIdx.x;
  const int projection_head = blockIdx.y;
  const int half_dim = head_dim / 2;
  const int pair = threadIdx.x;
  if (token >= tokens || pair >= half_dim) return;
  const int q_width = q_heads * head_dim;
  const int kv_width = kv_heads * head_dim;
  const int fused_width = q_width + 2 * kv_width;
  const int position = position_offset + token;
  const float frequency = powf(theta, -static_cast<float>(pair) / half_dim);
  float sine, cosine;
  sincosf(position * frequency, &sine, &cosine);

  if (projection_head < q_heads) {
    const int source = token * fused_width + projection_head * head_dim;
    float first = to_float(qkv[source + pair]);
    float second = to_float(qkv[source + half_dim + pair]);
    if (bias != nullptr) {
      first += to_float(bias[projection_head * head_dim + pair]);
      second += to_float(bias[projection_head * head_dim + half_dim + pair]);
    }
    const int destination = (token * q_heads + projection_head) * head_dim;
    q[destination + pair] = from_float<T>(first * cosine - second * sine);
    q[destination + half_dim + pair] =
        from_float<T>(second * cosine + first * sine);
  } else if (projection_head < q_heads + kv_heads) {
    const int head = projection_head - q_heads;
    const int source = token * fused_width + q_width + head * head_dim;
    float first = to_float(qkv[source + pair]);
    float second = to_float(qkv[source + half_dim + pair]);
    if (bias != nullptr) {
      first += to_float(bias[q_width + head * head_dim + pair]);
      second += to_float(bias[q_width + head * head_dim + half_dim + pair]);
    }
    const int destination =
        ((kv_output_offset + token) * kv_heads + head) * head_dim;
    k[destination + pair] = from_float<T>(first * cosine - second * sine);
    k[destination + half_dim + pair] =
        from_float<T>(second * cosine + first * sine);
  } else {
    const int head = projection_head - q_heads - kv_heads;
    const int source =
        token * fused_width + q_width + kv_width + head * head_dim;
    const int destination =
        ((kv_output_offset + token) * kv_heads + head) * head_dim;
    float first = to_float(qkv[source + pair]);
    float second = to_float(qkv[source + half_dim + pair]);
    if (bias != nullptr) {
      first += to_float(bias[q_width + kv_width + head * head_dim + pair]);
      second +=
          to_float(bias[q_width + kv_width + head * head_dim + half_dim + pair]);
    }
    v[destination + pair] = from_float<T>(first);
    v[destination + half_dim + pair] = from_float<T>(second);
  }
}

// Splits a packed [tokens, 3 * projection_width] projection with an optional
// fused bias and no rotation; this is the vision tower's MHA layout, where Q,
// K and V all have the same width.
//
// Launch: grid (tokens), block any.
template <class T>
__global__ void split_qkv_bias(const T* qkv, const T* bias, T* q, T* k, T* v,
                               int tokens, int projection_width) {
  const int token = blockIdx.x;
  if (token >= tokens) return;
  const int fused_width = 3 * projection_width;
  for (int col = threadIdx.x; col < fused_width; col += blockDim.x) {
    float value = to_float(qkv[static_cast<int64_t>(token) * fused_width + col]);
    if (bias != nullptr) value += to_float(bias[col]);
    const int64_t row = static_cast<int64_t>(token) * projection_width;
    if (col < projection_width) {
      q[row + col] = from_float<T>(value);
    } else if (col < 2 * projection_width) {
      k[row + col - projection_width] = from_float<T>(value);
    } else {
      v[row + col - 2 * projection_width] = from_float<T>(value);
    }
  }
}

}  // namespace apxinf::rope::kernels
