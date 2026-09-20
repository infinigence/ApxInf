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

// Block-wide sum; UNSAFE to reuse/overwrite scratch without a caller barrier.
// All threads in a 1D block must participate; blockDim.x must be a multiple
// of 32, and scratch must hold at least blockDim.x / 32 shared floats.
//
// WARNING: the final barrier publishes scratch[0], but does NOT protect the
// reads that follow it. A fast thread can overwrite the result while another
// thread is still reading it, producing incorrect sums/normalization/scales.
// This applies to consecutive sums, sum -> max, loop iterations, and ANY
// other write to overlapping scratch, even within a single warp. Arithmetic
// between calls and shuffle intrinsics do not provide a shared-memory fence.
//
// Each thread must first consume the returned value into a local variable;
// then ALL block threads must execute __syncthreads() before scratch is
// written again. Do not combine two calls using the same scratch in one
// expression or put the barrier inside a per-thread conditional.
//   float sum = block_sum_parallel_unsafe(value, scratch);
//   __syncthreads();  // Protect all reads before the next scratch writer.
//   float max = block_max_parallel_unsafe(other, scratch);
// No trailing barrier is needed when scratch is never written again.
__device__ __forceinline__ float block_sum_parallel_unsafe(
    float value, float* scratch) {
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

// Block-wide maximum (unused lanes are padded with negative infinity),
// with the SAME unsafe scratch-reuse contract as above.
// WARNING: max -> max, max -> sum, or any overlapping scratch write can race
// with the final result reads. Consume the result, then __syncthreads() in
// ALL block threads before reusing scratch. No barrier is needed after its
// last use. The same full-warp, 1D-block and scratch-size requirements apply.
__device__ __forceinline__ float block_max_parallel_unsafe(
    float value, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  value = warp_max(value);
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < warps ? scratch[lane] : -INFINITY;
    value = warp_max(value);
    if (lane == 0) scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}
