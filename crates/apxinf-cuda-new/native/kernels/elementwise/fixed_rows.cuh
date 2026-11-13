// Copyright 2026 ApxInf contributors.
// Fixed implementations for the row-reduction elementwise operators the
// speech models need: row mean and per-row argmax. Ported verbatim from the
// legacy family without numerical changes.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kRowsThreads = 256;

template <class T>
__device__ float rows_to_float(T value);

template <>
__device__ inline float rows_to_float(float value) {
  return value;
}

template <>
__device__ inline float rows_to_float(__half value) {
  return __half2float(value);
}

template <>
__device__ inline float rows_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <class T>
__device__ T rows_from_float(float value);

template <>
__device__ inline float rows_from_float(float value) {
  return value;
}

template <>
__device__ inline __half rows_from_float(float value) {
  return __float2half(value);
}

template <>
__device__ inline __nv_bfloat16 rows_from_float(float value) {
  return __float2bfloat16(value);
}

// Single-block column reduction: a sequential f32 row sum per column, then
// one division. The row order is fixed, so the sum is reproducible.
template <class T>
__global__ void mean_rows_kernel(const T* __restrict__ input,
                                 T* __restrict__ output,
                                 int rows, int cols) {
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float sum = 0.0f;
    for (int row = 0; row < rows; ++row) {
      sum += rows_to_float<T>(input[static_cast<int64_t>(row) * cols + col]);
    }
    output[col] = rows_from_float<T>(sum / static_cast<float>(rows));
  }
}

// Per-row argmax with first-index tie breaking: warp shuffle down, then one
// thread merges the per-warp winners.
template <class T>
__global__ void argmax_rows_kernel(const T* __restrict__ logits,
                                   int32_t* __restrict__ index, int rows,
                                   int cols) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const T* values = logits + static_cast<int64_t>(row) * cols;
  float local_value = -INFINITY;
  int local_index = 0;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = rows_to_float<T>(values[col]);
    if (value > local_value || (value == local_value && col < local_index)) {
      local_value = value;
      local_index = col;
    }
  }
  __shared__ float best_value[32];
  __shared__ int best_index[32];
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other_value =
        __shfl_down_sync(0xFFFFFFFFu, local_value, offset);
    const int other_index = __shfl_down_sync(0xFFFFFFFFu, local_index, offset);
    if (other_value > local_value ||
        (other_value == local_value && other_index < local_index)) {
      local_value = other_value;
      local_index = other_index;
    }
  }
  const int warp = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) {
    best_value[warp] = local_value;
    best_index[warp] = local_index;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int warps = (blockDim.x + 31) >> 5;
    float value = best_value[0];
    int selected = best_index[0];
    for (int i = 1; i < warps; ++i) {
      if (best_value[i] > value ||
          (best_value[i] == value && best_index[i] < selected)) {
        value = best_value[i];
        selected = best_index[i];
      }
    }
    index[row] = selected;
  }
}

}  // namespace
