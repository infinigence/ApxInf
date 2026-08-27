#pragma once

// Copyright 2026 apxinf contributors.
// Shared warp/block reduction helpers for custom CUDA operators.

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
}

__device__ __forceinline__ float warp_sum_all(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffff, value, offset);
  return value;
}

__device__ __forceinline__ float warp_max(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
  return value;
}

__device__ __forceinline__ float block_sum(float value, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  value = warp_sum(value);
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < warps ? scratch[lane] : 0.0f;
    value = warp_sum(value);
    if (lane == 0) scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}

__device__ __forceinline__ float2 block_sum_pair(
    float left, float right, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  left = warp_sum(left);
  right = warp_sum(right);
  if (lane == 0) {
    scratch[warp] = left;
    scratch[warps + warp] = right;
  }
  __syncthreads();
  if (warp == 0) {
    left = lane < warps ? scratch[lane] : 0.0f;
    right = lane < warps ? scratch[warps + lane] : 0.0f;
    left = warp_sum(left);
    right = warp_sum(right);
    if (lane == 0) {
      scratch[0] = left;
      scratch[1] = right;
    }
  }
  __syncthreads();
  return make_float2(scratch[0], scratch[1]);
}

__device__ __forceinline__ float block_max(float value, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  value = warp_max(value);
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < warps ? scratch[lane] : 0.0f;
    value = warp_max(value);
    if (lane == 0) scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}
