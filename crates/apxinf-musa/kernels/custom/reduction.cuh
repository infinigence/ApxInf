#pragma once

// Derived in part from `crates/apxinf-cuda/kernels/custom/` (Apache-2.0,
// Copyright 2026 apxinf contributors); the differences are explained below.

// Shared warp/block reduction helpers, a port of
// `crates/apxinf-cuda/kernels/custom/reduction.cuh`.
//
// The two helpers the ported kernels call are carried over, plus `warp_max` and
// `block_max_parallel_unsafe`, which the W8A8 row quantizer needs for its per-row
// absmax. The CUDA file also has `warp_sum_all` and `warp_max`; `warp_max` is
// here, `warp_sum_all` is not, because nothing here reaches it.
//
// The 32-lane contract below is the reason this file needs no rewriting for
// MUSA. M1000's hardware warp is 128 lanes, but `__shfl_*_sync` groups are 32,
// which was measured on an M1000
// (`__shfl_xor_sync(mask, t, 32)` is the identity there). So `threadIdx.x & 31`
// and `threadIdx.x >> 5` mean the same thing they mean on CUDA, and a
// 256-thread block is 8 software warps.

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffffu, value, offset);
  return value;
}

__device__ __forceinline__ float warp_max(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
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
__device__ __forceinline__ float block_sum_parallel_unsafe(float value,
                                                           float* scratch) {
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

// Block-wide maximum; the same contract, barriers and hazards as
// `block_sum_parallel_unsafe` -- read the returned value into a register before
// ANY later writer of `scratch`, including a following max/sum in the same
// thread.
//
// Lanes above `warps` are padded with negative infinity, not zero: with a zero
// pad an all-negative row would report 0 as its maximum. The W8A8 row quantizer
// only takes this over non-negative values, so the two agree today -- which is
// exactly why the padding is the upstream one and not a value that happens to
// work for its current caller.
__device__ __forceinline__ float block_max_parallel_unsafe(float value,
                                                           float* scratch) {
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
