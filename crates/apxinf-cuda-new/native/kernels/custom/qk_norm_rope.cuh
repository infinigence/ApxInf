#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace apxinf::qk_norm_rope::kernels {

/* Local copies of the three norm-kernel helpers this file needs. norm.cuh
 * defines whole __global__ kernels and must stay single-TU, so it cannot be
 * included from a second adapter. */
__device__ __forceinline__ float to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

__device__ __forceinline__ __nv_bfloat16 from_float(float value) {
  return __float2bfloat16(value);
}

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
}

// One block per row; `scratch` must hold blockDim.x / 32 floats. The final
// barrier publishes scratch[0], but does not protect the reads that follow it.
// Callers must consume the result and synchronize all block threads before
// reading scratch[0].
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

/* Per-head RMSNorm without a gate, for q_norm and k_norm.
 * data is [rows, head_dim] in place; weight is [head_dim] and enters as
 * (1 + weight), the Qwen3 zero-centered form. */
__global__ void head_rms_norm_kernel(__nv_bfloat16* __restrict__ data,
                                     const __nv_bfloat16* __restrict__ weight,
                                     int rows, int head_dim, float epsilon) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const long long base = (long long)row * head_dim;

  float sum = 0.0f;
  for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
    const float value = __bfloat162float(data[base + index]);
    sum += value * value;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
  }
  __shared__ float partial[32];
  const int warp = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) partial[warp] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    const int warps = (blockDim.x + 31) / 32;
    float total = 0.0f;
    for (int index = 0; index < warps; ++index) total += partial[index];
    partial[0] = total;
  }
  __syncthreads();

  const float scale =
      rsqrtf(partial[0] / static_cast<float>(head_dim) + epsilon);
  for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
    data[base + index] = __float2bfloat16(
        __bfloat162float(data[base + index]) * scale *
        (1.0f + __bfloat162float(weight[index])));
  }
}

/* Fused per-head RMSNorm + partial rope, one launch per side (q / k).
 *
 * Bit-identical to norm::kernels::rms_norm<bf16> (plain weight, 256-thread
 * block_sum_parallel_unsafe reduction) followed by partial rope on the norm
 * output: the same reduction tree, the same plain-weight multiplier, and the
 * rope stage re-reads the bf16-rounded norm result through memory after a
 * barrier (the volatile qualifier keeps the compiler from forwarding the
 * pre-barrier registers; the binding gate is the bitwise diff itself --
 * dropping the round-trip would be more accurate and would show as a non-zero
 * diff).
 *
 * Only the first `rotary_dim` elements rotate, in the rotate-half pairing
 * (x[i], x[i + rotary_dim / 2]). Grid is dim3(tokens, heads), so the decode
 * (tokens == 1) and prefill (tokens > 1) callers share one code path. */
__global__ void qk_norm_rope_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    const int32_t* __restrict__ positions,
    /* No __restrict__ on output: the rope stage reads it again (round_trip)
     * after the norm stage writes it -- the same object through two paths. */
    __nv_bfloat16* output, int tokens, int heads, int head_dim, int rotary_dim,
    float theta, float eps) {
  __shared__ float scratch[32];
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  if (token >= tokens || head >= heads) return;
  const int half = rotary_dim / 2;
  const long long base = ((long long)token * heads + head) * head_dim;
  const float position = static_cast<float>(positions[token]);

  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
    const float value = to_float(input[base + col]);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / head_dim + eps);
  for (int col = threadIdx.x; col < head_dim; col += blockDim.x) {
    const int64_t index = base + col;
    output[index] = from_float(
        to_float(input[index]) * inverse_rms * to_float(weight[col]));
  }

  __syncthreads();
  __nv_bfloat16 volatile* round_trip = output;
  for (int index = threadIdx.x; index < half; index += blockDim.x) {
    const float inverse_frequency =
        __powf(theta, -2.0f * static_cast<float>(index) /
                          static_cast<float>(rotary_dim));
    const float angle = position * inverse_frequency;
    float sine;
    float cosine;
    __sincosf(angle, &sine, &cosine);

    const float low = __bfloat162float(round_trip[base + index]);
    const float high = __bfloat162float(round_trip[base + half + index]);
    output[base + index] = __float2bfloat16(low * cosine - high * sine);
    output[base + half + index] = __float2bfloat16(high * cosine + low * sine);
  }
}

/* Scatter one `row_elems`-wide BF16 row of `src` into the row-major `cache`
 * [capacity, row_elems] at the row index read from position[0].
 *
 * The row index is a device-side value (the same persistent positions tensor
 * the rope kernel reads), so a captured graph stays valid as the decode
 * position advances without recapture. One block, grid-stride over the row;
 * out-of-range rows are dropped rather than corrupting neighbouring rows. */
__global__ void cache_append_row_kernel(
    const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ cache,
    const int32_t* __restrict__ position, int row_elems, long long capacity) {
  const int row = position[0];
  if (row < 0 || static_cast<long long>(row) >= capacity) return;
  const long long base = static_cast<long long>(row) * row_elems;
  for (int index = threadIdx.x; index < row_elems; index += blockDim.x) {
    cache[base + index] = src[index];
  }
}

}  // namespace apxinf::qk_norm_rope::kernels
