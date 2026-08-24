#pragma once

// Copyright 2026 apxinf contributors.
// Categorical token sampling and counter-based random generation.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

struct ApxInfSamplingOutput {
  uint32_t token_id;
  uint32_t status;
  float logprob;
  uint32_t reserved;
};

__device__ __forceinline__ uint64_t apxinf_splitmix64(uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

__device__ __forceinline__ uint2 apxinf_mul_hi_lo(uint32_t left,
                                                  uint32_t right) {
  const uint64_t product = static_cast<uint64_t>(left) * right;
  return make_uint2(static_cast<uint32_t>(product >> 32),
                    static_cast<uint32_t>(product));
}

__device__ __forceinline__ uint4 apxinf_philox4x32_10(uint4 counter,
                                                       uint2 key) {
  constexpr uint32_t m0 = 0xd2511f53U;
  constexpr uint32_t m1 = 0xcd9e8d57U;
  constexpr uint32_t w0 = 0x9e3779b9U;
  constexpr uint32_t w1 = 0xbb67ae85U;
  for (int round = 0; round < 10; ++round) {
    const uint2 p0 = apxinf_mul_hi_lo(m0, counter.x);
    const uint2 p1 = apxinf_mul_hi_lo(m1, counter.z);
    counter = make_uint4(p1.x ^ counter.y ^ key.x, p1.y,
                         p0.x ^ counter.w ^ key.y, p0.y);
    key.x += w0;
    key.y += w1;
  }
  return counter;
}

__device__ __forceinline__ uint4 apxinf_rng_words(
    uint64_t seed, uint64_t sequence, uint64_t draw, uint64_t group) {
  const uint64_t rotated_draw = (draw << 32) | (draw >> 32);
  const uint64_t stream = apxinf_splitmix64(sequence ^ rotated_draw);
  return apxinf_philox4x32_10(
      make_uint4(static_cast<uint32_t>(group),
                 static_cast<uint32_t>(group >> 32),
                 static_cast<uint32_t>(stream),
                 static_cast<uint32_t>(stream >> 32)),
      make_uint2(static_cast<uint32_t>(seed),
                 static_cast<uint32_t>(seed >> 32)));
}

__device__ __forceinline__ float apxinf_unit_open(uint32_t word) {
  const float value = __uint_as_float(0x3f800000U | (word >> 9)) - 1.0f;
  return fmaxf(value, __uint_as_float(0x33800000U));
}

template <typename T>
__device__ __forceinline__ float apxinf_logit_to_float(T value);

template <>
__device__ __forceinline__ float apxinf_logit_to_float(float value) {
  return value;
}

template <>
__device__ __forceinline__ float apxinf_logit_to_float(half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float apxinf_logit_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__global__ void apxinf_prepare_logits_kernel(
    const T* logits, const uint32_t* counts, float* adjusted,
    uint32_t* token_ids, uint32_t vocab_size, float repetition,
    float frequency, float presence, float inverse_temperature) {
  for (uint32_t token = blockIdx.x * blockDim.x + threadIdx.x;
       token < vocab_size; token += blockDim.x * gridDim.x) {
    float value = apxinf_logit_to_float(logits[token]);
    if (isnan(value)) value = -CUDART_INF_F;
    if (value == CUDART_INF_F) value = FLT_MAX;
    const uint32_t occurrences = counts[token];
    if (occurrences != 0) {
      if (repetition != 1.0f) {
        value = value < 0.0f ? value * repetition : value / repetition;
      }
      value -= frequency * static_cast<float>(occurrences);
      value -= presence;
    }
    adjusted[token] = value * inverse_temperature;
    token_ids[token] = token;
  }
}

__device__ __forceinline__ bool apxinf_better_pair(
    float value, uint32_t token, float best_value, uint32_t best_token) {
  return value > best_value ||
         (value == best_value && token < best_token);
}

__global__ void apxinf_argmax_stage1_kernel(
    const float* logits, uint32_t vocab_size, float* partial_values,
    uint32_t* partial_tokens) {
  float best_value = -CUDART_INF_F;
  uint32_t best_token = UINT32_MAX;
  for (uint32_t token = blockIdx.x * blockDim.x + threadIdx.x;
       token < vocab_size; token += blockDim.x * gridDim.x) {
    const float value = logits[token];
    if (apxinf_better_pair(value, token, best_value, best_token)) {
      best_value = value;
      best_token = token;
    }
  }
  __shared__ float values[256];
  __shared__ uint32_t tokens[256];
  values[threadIdx.x] = best_value;
  tokens[threadIdx.x] = best_token;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride &&
        apxinf_better_pair(values[threadIdx.x + stride],
                           tokens[threadIdx.x + stride],
                           values[threadIdx.x], tokens[threadIdx.x])) {
      values[threadIdx.x] = values[threadIdx.x + stride];
      tokens[threadIdx.x] = tokens[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    partial_values[blockIdx.x] = values[0];
    partial_tokens[blockIdx.x] = tokens[0];
  }
}

__global__ void apxinf_argmax_stage2_kernel(
    const float* partial_values, const uint32_t* partial_tokens,
    uint32_t partial_count, uint32_t* counts, ApxInfSamplingOutput* output) {
  float best_value = -CUDART_INF_F;
  uint32_t best_token = UINT32_MAX;
  for (uint32_t index = threadIdx.x; index < partial_count;
       index += blockDim.x) {
    if (apxinf_better_pair(partial_values[index], partial_tokens[index],
                           best_value, best_token)) {
      best_value = partial_values[index];
      best_token = partial_tokens[index];
    }
  }
  __shared__ float values[256];
  __shared__ uint32_t tokens[256];
  values[threadIdx.x] = best_value;
  tokens[threadIdx.x] = best_token;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride &&
        apxinf_better_pair(values[threadIdx.x + stride],
                           tokens[threadIdx.x + stride],
                           values[threadIdx.x], tokens[threadIdx.x])) {
      values[threadIdx.x] = values[threadIdx.x + stride];
      tokens[threadIdx.x] = tokens[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output->token_id = tokens[0];
    output->status = isfinite(values[0]) && tokens[0] != UINT32_MAX ? 0U : 1U;
    output->logprob = CUDART_NAN_F;
    output->reserved = 0;
    if (output->status == 0) atomicAdd(&counts[tokens[0]], 1U);
  }
}

__global__ void apxinf_softmax_weights_kernel(
    const float* sorted_logits, float* weights, uint32_t vocab_size,
    uint32_t candidate_limit) {
  const float maximum = sorted_logits[0];
  for (uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < vocab_size; index += blockDim.x * gridDim.x) {
    weights[index] = index < candidate_limit && isfinite(maximum) &&
                             isfinite(sorted_logits[index])
                         ? expf(sorted_logits[index] - maximum)
                         : 0.0f;
  }
}

__device__ __forceinline__ uint32_t apxinf_lower_bound_cdf(
    const float* cdf, uint32_t count, float target) {
  uint32_t first = 0;
  uint32_t length = count;
  while (length != 0) {
    const uint32_t half = length >> 1;
    const uint32_t middle = first + half;
    if (cdf[middle] < target) {
      first = middle + 1;
      length -= half + 1;
    } else {
      length = half;
    }
  }
  return first < count ? first : count - 1;
}

__global__ void apxinf_select_cdf_kernel(
    const uint32_t* sorted_tokens, const float* weights, const float* cdf,
    uint32_t candidate_limit, float top_p, uint32_t random_selection,
    uint32_t return_logprob, uint64_t seed, uint64_t sequence, uint64_t draw,
    uint32_t* counts, ApxInfSamplingOutput* output) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const float total = cdf[candidate_limit - 1];
  if (!(total > 0.0f) || !isfinite(total)) {
    output->token_id = 0;
    output->status = 2;
    output->logprob = CUDART_NAN_F;
    output->reserved = 0;
    return;
  }
  const uint32_t nucleus_end =
      apxinf_lower_bound_cdf(cdf, candidate_limit, top_p * total);
  const float nucleus_total = cdf[nucleus_end];
  uint32_t selected_position = 0;
  if (random_selection != 0) {
    const uint4 words = apxinf_rng_words(seed, sequence, draw, 0);
    const float target = apxinf_unit_open(words.x) * nucleus_total;
    selected_position =
        apxinf_lower_bound_cdf(cdf, nucleus_end + 1, target);
  }
  const uint32_t token = sorted_tokens[selected_position];
  output->token_id = token;
  output->status = 0;
  output->logprob = return_logprob != 0
                        ? logf(weights[selected_position] / nucleus_total)
                        : CUDART_NAN_F;
  output->reserved = 0;
  atomicAdd(&counts[token], 1U);
}

template <typename T>
__device__ __forceinline__ void apxinf_store_normal(T* output, uint64_t index,
                                                    float value);

template <>
__device__ __forceinline__ void apxinf_store_normal(float* output,
                                                    uint64_t index,
                                                    float value) {
  output[index] = value;
}

template <>
__device__ __forceinline__ void apxinf_store_normal(half* output,
                                                    uint64_t index,
                                                    float value) {
  output[index] = __float2half_rn(value);
}

template <>
__device__ __forceinline__ void apxinf_store_normal(__nv_bfloat16* output,
                                                    uint64_t index,
                                                    float value) {
  output[index] = __float2bfloat16_rn(value);
}

template <typename T>
__global__ void apxinf_standard_normal_kernel(
    T* output, uint64_t count, uint64_t seed, uint64_t sequence,
    uint64_t draw) {
  for (uint64_t pair = blockIdx.x * static_cast<uint64_t>(blockDim.x) +
                       threadIdx.x;
       pair < (count + 1) / 2;
       pair += static_cast<uint64_t>(blockDim.x) * gridDim.x) {
    const uint4 words = apxinf_rng_words(seed, sequence, draw, pair);
    const float u1 = apxinf_unit_open(words.x);
    const float u2 = apxinf_unit_open(words.y);
    const float radius = sqrtf(-2.0f * logf(u1));
    const float angle = 6.2831853071795864769f * u2;
    float sine;
    float cosine;
    sincosf(angle, &sine, &cosine);
    const uint64_t first = pair * 2;
    apxinf_store_normal(output, first, radius * cosine);
    if (first + 1 < count)
      apxinf_store_normal(output, first + 1, radius * sine);
  }
}
