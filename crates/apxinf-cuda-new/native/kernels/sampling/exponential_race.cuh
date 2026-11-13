#pragma once

// Copyright 2026 apxinf contributors.
// Exponential-race categorical sampling: penalty-adjusted logits, adjusted
// argmax, temperature, top-k truncation, softmax, and the exponential race,
// all resolved inside one block so the host only reads back the winning token
// id, the adjusted argmax, and a status word.
//
// The numeric contract mirrors the reference host sampler exactly: FP16
// rounding at the same boundaries, a strictly sequential FP32 softmax sum in
// ascending index order, first-index tie-breaking, and one Philox4x32-10 draw
// per candidate with counter words [step_lo, step_hi, elem_lo, elem_hi] and
// key [seed_lo, seed_hi].

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "../custom/sampling.cuh"

constexpr int kExponentialRaceThreads = 1024;
constexpr int kExponentialRaceMaxVocab = 2048;

__device__ __forceinline__ float apxinf_fp16_round(float value) {
  return __half2float(__float2half_rn(value));
}

__device__ __forceinline__ bool apxinf_race_better(float value, uint32_t index,
                                                   float best_value,
                                                   uint32_t best_index) {
  return value > best_value || (value == best_value && index < best_index);
}

// Tree reduction over per-thread slots picking the largest value and, among
// ties, the lowest index — the same winner a sequential strict-greater scan
// over ascending indices produces.
template <int kThreads>
__device__ __forceinline__ void apxinf_race_block_reduce(
    float* shared_values, uint32_t* shared_indices, float* best_value,
    uint32_t* best_index) {
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (static_cast<int>(threadIdx.x) < stride &&
        apxinf_race_better(shared_values[threadIdx.x + stride],
                           shared_indices[threadIdx.x + stride],
                           shared_values[threadIdx.x],
                           shared_indices[threadIdx.x])) {
      shared_values[threadIdx.x] = shared_values[threadIdx.x + stride];
      shared_indices[threadIdx.x] = shared_indices[threadIdx.x + stride];
    }
    __syncthreads();
  }
  *best_value = shared_values[0];
  *best_index = shared_indices[0];
}

// Output layout: [0] sampled token, [1] adjusted argmax, [2] status, [3] 0.
// Status: 0 ok, 1 filtered logits invalid, 2 probability mass invalid.
template <typename T>
__global__ void apxinf_sampling_exponential_race_kernel(
    const T* logits, uint8_t* seen, uint32_t* output, uint32_t vocab,
    uint64_t seed, uint64_t step, uint32_t top_k, float temperature,
    float repetition) {
  __shared__ float values[kExponentialRaceMaxVocab];
  __shared__ float scratch[kExponentialRaceMaxVocab];
  __shared__ float reductions[kExponentialRaceThreads];
  __shared__ uint32_t reduction_indices[kExponentialRaceThreads];
  const uint32_t tid = threadIdx.x;

  // Penalty adjustment: tokens flagged in `seen` are rescaled once, with the
  // FP16 rounding the reference applies to every adjusted score.
  float best_value = -CUDART_INF_F;
  uint32_t best_index = 0xffffffffU;
  for (uint32_t index = tid; index < vocab; index += kExponentialRaceThreads) {
    float value = apxinf_logit_to_float(logits[index]);
    if (seen[index] != 0) {
      value = value < 0.0f ? apxinf_fp16_round(value * repetition)
                           : apxinf_fp16_round(value / repetition);
    }
    values[index] = value;
    if (apxinf_race_better(value, index, best_value, best_index)) {
      best_value = value;
      best_index = index;
    }
  }
  reductions[tid] = best_value;
  reduction_indices[tid] = best_index;
  __syncthreads();
  apxinf_race_block_reduce<kExponentialRaceThreads>(
      reductions, reduction_indices, &best_value, &best_index);
  const uint32_t adjusted_argmax = best_index;

  const float scaled_temperature = fmaxf(temperature, 1.0e-5f);
  for (uint32_t index = tid; index < vocab; index += kExponentialRaceThreads) {
    values[index] = apxinf_fp16_round(values[index] / scaled_temperature);
  }

  // Top-k pivot: the k-th largest temperature-scaled score, duplicates
  // counted. One masked reduction pass per rank keeps the multiset semantics
  // of the reference descending sort.
  float pivot = -CUDART_INF_F;
  if (top_k != 0 && top_k < vocab) {
    for (uint32_t index = tid; index < vocab; index += kExponentialRaceThreads) {
      scratch[index] = values[index];
    }
    __syncthreads();
    for (uint32_t pass = 0; pass < top_k; ++pass) {
      best_value = -CUDART_INF_F;
      best_index = 0xffffffffU;
      for (uint32_t index = tid; index < vocab; index += kExponentialRaceThreads) {
        if (apxinf_race_better(scratch[index], index, best_value, best_index)) {
          best_value = scratch[index];
          best_index = index;
        }
      }
      reductions[tid] = best_value;
      reduction_indices[tid] = best_index;
      __syncthreads();
      apxinf_race_block_reduce<kExponentialRaceThreads>(
          reductions, reduction_indices, &best_value, &best_index);
      pivot = best_value;
      if (tid == 0 && best_index != 0xffffffffU) {
        scratch[best_index] = -CUDART_INF_F;
      }
      __syncthreads();
    }
    for (uint32_t index = tid; index < vocab; index += kExponentialRaceThreads) {
      if (values[index] < pivot) values[index] = -CUDART_INF_F;
    }
  }

  // Softmax maximum; every thread observes the reduced value, so the early
  // exits below are uniform across the block.
  best_value = -CUDART_INF_F;
  best_index = 0xffffffffU;
  for (uint32_t index = tid; index < vocab; index += kExponentialRaceThreads) {
    if (apxinf_race_better(values[index], index, best_value, best_index)) {
      best_value = values[index];
      best_index = index;
    }
  }
  reductions[tid] = best_value;
  reduction_indices[tid] = best_index;
  __syncthreads();
  apxinf_race_block_reduce<kExponentialRaceThreads>(
      reductions, reduction_indices, &best_value, &best_index);
  const float maximum = best_value;
  if (!isfinite(maximum)) {
    if (tid == 0) {
      output[0] = 0;
      output[1] = adjusted_argmax;
      output[2] = 1;
      output[3] = 0;
    }
    return;
  }

  for (uint32_t index = tid; index < vocab; index += kExponentialRaceThreads) {
    scratch[index] =
        isfinite(values[index]) ? expf(values[index] - maximum) : 0.0f;
  }
  __shared__ float total;
  __syncthreads();
  if (tid == 0) {
    // The reference accumulates the sum sequentially in ascending index
    // order; replicate that exactly rather than using a parallel reduction.
    float sum = 0.0f;
    for (uint32_t index = 0; index < vocab; ++index) {
      sum += scratch[index];
    }
    total = sum;
  }
  __syncthreads();
  if (!isfinite(total) || total <= 0.0f) {
    if (tid == 0) {
      output[0] = 0;
      output[1] = adjusted_argmax;
      output[2] = 2;
      output[3] = 0;
    }
    return;
  }

  for (uint32_t index = tid; index < vocab; index += kExponentialRaceThreads) {
    values[index] = apxinf_fp16_round(scratch[index] / total);
  }

  // Exponential race: one Philox draw per candidate; the winner maximizes
  // FP16(probability / noise), strict greater, lowest index on ties.
  float race_best = -CUDART_INF_F;
  uint32_t race_index = 0xffffffffU;
  for (uint32_t index = tid; index < vocab; index += kExponentialRaceThreads) {
    const uint4 words = apxinf_philox4x32_10(
        make_uint4(static_cast<uint32_t>(step),
                   static_cast<uint32_t>(step >> 32),
                   static_cast<uint32_t>(index),
                   static_cast<uint32_t>(static_cast<uint64_t>(index) >> 32)),
        make_uint2(static_cast<uint32_t>(seed),
                   static_cast<uint32_t>(seed >> 32)));
    const float uniform =
        static_cast<float>(static_cast<double>(words.x) *
                               2.3283064365386963e-10 +
                           1.1641532182693481e-10);
    const float exponential =
        uniform >= 1.0f - FLT_EPSILON * 0.5f ? FLT_EPSILON * 0.5f
                                             : -logf(uniform);
    const float noise = apxinf_fp16_round(exponential);
    const float score = apxinf_fp16_round(values[index] / noise);
    if (apxinf_race_better(score, index, race_best, race_index)) {
      race_best = score;
      race_index = index;
    }
  }
  reductions[tid] = race_best;
  reduction_indices[tid] = race_index;
  __syncthreads();
  apxinf_race_block_reduce<kExponentialRaceThreads>(
      reductions, reduction_indices, &best_value, &best_index);
  if (tid == 0) {
    output[0] = best_index;
    output[1] = adjusted_argmax;
    output[2] = 0;
    output[3] = 0;
    if (best_index != 0xffffffffU) {
      seen[best_index] = 1;
    }
  }
}
