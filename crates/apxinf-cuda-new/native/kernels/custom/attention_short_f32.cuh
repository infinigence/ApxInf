#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

namespace apxinf::attention::kernels {

// Short dense F32 MHA, D128. Four warps share QueryTile query rows and reuse
// each vectorized K/V load across those rows. QK uses independent eight-lane groups;
// PV splits keys over the four warps before a shared-memory reduction. Scores
// remain on chip. Products, probabilities and accumulators all remain F32.
// The provider checks B1, Q16..128, K1..160, equal heads and 16-byte alignment.
template <int QueryTile>
__global__ void attention_short_f32(
    const float* query, const float* key, const float* value, float* output,
    int query_tokens, int key_tokens, int heads, float scale) {
  constexpr int kWarps = 4;
  constexpr int kGroup = 8;
  __shared__ float scores[QueryTile][160];
  __shared__ float partial[kWarps][QueryTile][128];
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  const int group_lane = threadIdx.x % kGroup;
  const unsigned group_mask = 0xffu << ((lane / kGroup) * kGroup);
  const int head = blockIdx.y;
  const int first_query = blockIdx.x * QueryTile;

  float q[QueryTile][16];
#pragma unroll
  for (int row = 0; row < QueryTile; ++row) {
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      const float4 x = first_query + row < query_tokens
          ? reinterpret_cast<const float4*>(
                query + ((first_query + row) * heads + head) * 128)
                [group_lane + part * kGroup]
          : make_float4(0, 0, 0, 0);
      q[row][part * 4] = x.x;
      q[row][part * 4 + 1] = x.y;
      q[row][part * 4 + 2] = x.z;
      q[row][part * 4 + 3] = x.w;
    }
  }
  for (int token = threadIdx.x / kGroup; token < key_tokens;
       token += kWarps * 32 / kGroup) {
    float k[16];
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      const float4 x = reinterpret_cast<const float4*>(
          key + (token * heads + head) * 128)[group_lane + part * kGroup];
      k[part * 4] = x.x;
      k[part * 4 + 1] = x.y;
      k[part * 4 + 2] = x.z;
      k[part * 4 + 3] = x.w;
    }
#pragma unroll
    for (int row = 0; row < QueryTile; ++row) {
      float dot = 0;
#pragma unroll
      for (int dim = 0; dim < 16; ++dim)
        dot = fmaf(q[row][dim], k[dim], dot);
      // The final key iteration may contain a partially active warp.
      // Naming inactive groups in the mask would violate shuffle convergence.
#pragma unroll
      for (int delta = 4; delta > 0; delta /= 2)
        dot += __shfl_down_sync(group_mask, dot, delta, kGroup);
      if (group_lane == 0) scores[row][token] = dot * scale;
    }
  }
  __syncthreads();

  for (int row = warp; row < QueryTile; row += kWarps) {
    float maximum = -CUDART_INF_F;
    for (int token = lane; token < key_tokens; token += 32)
      maximum = fmaxf(maximum, scores[row][token]);
    for (int delta = 16; delta > 0; delta /= 2)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, delta));
    float sum = 0;
    for (int token = lane; token < key_tokens; token += 32) {
      const float p = expf(scores[row][token] - maximum);
      scores[row][token] = p;
      sum += p;
    }
    for (int delta = 16; delta > 0; delta /= 2)
      sum += __shfl_xor_sync(0xffffffff, sum, delta);
    for (int token = lane; token < key_tokens; token += 32)
      scores[row][token] /= sum;
  }
  __syncthreads();

  float accum[QueryTile][4] = {};
  for (int token = warp; token < key_tokens; token += kWarps) {
    const float4 x = reinterpret_cast<const float4*>(
        value + (token * heads + head) * 128)[lane];
    const float v[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
    for (int row = 0; row < QueryTile; ++row) {
      const float p = scores[row][token];
#pragma unroll
      for (int dim = 0; dim < 4; ++dim)
        accum[row][dim] = fmaf(p, v[dim], accum[row][dim]);
    }
  }
#pragma unroll
  for (int row = 0; row < QueryTile; ++row) {
#pragma unroll
    for (int dim = 0; dim < 4; ++dim)
      partial[warp][row][lane * 4 + dim] = accum[row][dim];
  }
  __syncthreads();
  for (int index = threadIdx.x; index < QueryTile * 128;
       index += kWarps * 32) {
    const int row = index / 128;
    const int dim = index % 128;
    float sum = 0;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) sum += partial[w][row][dim];
    if (first_query + row < query_tokens)
      output[((first_query + row) * heads + head) * 128 + dim] = sum;
  }
}

inline cudaError_t launch_attention_short_f32(
    const void* query, const void* key, const void* value, void* output,
    int query_tokens, int key_tokens, int heads, float scale,
    cudaStream_t stream) {
  attention_short_f32<4><<<dim3((query_tokens + 3) / 4, heads), 128, 0, stream>>>(
      static_cast<const float*>(query), static_cast<const float*>(key),
      static_cast<const float*>(value), static_cast<float*>(output),
      query_tokens, key_tokens, heads, scale);
  return cudaGetLastError();
}

}  // namespace apxinf::attention::kernels
