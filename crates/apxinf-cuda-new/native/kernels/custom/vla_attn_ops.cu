// Copyright 2026 ApxInf contributors.
//
// VLA attention primitives ported bit-identically from the legacy walloss
// kernels (crates/apxinf-cuda/kernels/custom/fused.cuh and attention.cuh):
// the fused QKV split + mRoPE with cache write, the fused vision QKV split
// + 2D RoPE, and the segmented dense MHA. Arithmetic order, rounding points
// and tie-breaks are preserved exactly.

#include "vla_attn_ops.h"

#include <cuda_bf16.h>

namespace apxinf::cuda_new::vla_attn_ops {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
}

// Axis a rotary pair draws its position from: pairs are striped modulo 3
// across temporal/height/width, bounded by the section widths.
__device__ __forceinline__ int fused_mrope_axis(int pair, int section_h,
                                                int section_w) {
  const int remainder = pair % 3;
  if (remainder == 1 && pair < section_h * 3) return 1;
  if (remainder == 2 && pair < section_w * 3) return 2;
  return 0;
}

// One block per (token, projection head); one thread per rotary pair.
// Q/K rotate by the mRoPE angle for their axis; V copies straight through.
// K/V land at `cache_offset + token` in the caches.
__global__ void gqa_qkv_mrope_cache_kernel(
    const __nv_bfloat16* qkv, const __nv_bfloat16* bias,
    const uint32_t* position_ids, __nv_bfloat16* q, __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache, int tokens, int q_heads, int kv_heads,
    int head_dim, float theta, int section_h, int section_w,
    int cache_offset) {
  const int token = blockIdx.x;
  const int projection_head = blockIdx.y;
  const int pair = threadIdx.x;
  const int half_dim = head_dim / 2;
  if (pair >= half_dim) return;

  const int q_width = q_heads * head_dim;
  const int kv_width = kv_heads * head_dim;
  const int fused_width = q_width + 2 * kv_width;

  if (projection_head < q_heads + kv_heads) {
    const bool is_query = projection_head < q_heads;
    const int head = is_query ? projection_head : projection_head - q_heads;
    const int source_base = token * fused_width +
        (is_query ? head * head_dim : q_width + head * head_dim);
    float first = __bfloat162float(qkv[source_base + pair]);
    float second = __bfloat162float(qkv[source_base + half_dim + pair]);
    if (bias != nullptr) {
      const int bias_base =
          is_query ? head * head_dim : q_width + head * head_dim;
      first += __bfloat162float(bias[bias_base + pair]);
      second += __bfloat162float(bias[bias_base + half_dim + pair]);
    }
    const int axis = fused_mrope_axis(pair, section_h, section_w);
    const float position = static_cast<float>(position_ids[token * 3 + axis]);
    const float frequency = powf(theta, -static_cast<float>(pair) / half_dim);
    float sine, cosine;
    sincosf(position * frequency, &sine, &cosine);
    __nv_bfloat16* destination = is_query
        ? q + (token * q_heads + head) * head_dim
        : k_cache + ((cache_offset + token) * kv_heads + head) * head_dim;
    destination[pair] = __float2bfloat16(first * cosine - second * sine);
    destination[half_dim + pair] =
        __float2bfloat16(second * cosine + first * sine);
    return;
  }

  const int head = projection_head - q_heads - kv_heads;
  const int source_base =
      token * fused_width + q_width + kv_width + head * head_dim;
  const int bias_base = q_width + kv_width + head * head_dim;
  const int destination_base =
      ((cache_offset + token) * kv_heads + head) * head_dim;
  float first = __bfloat162float(qkv[source_base + pair]);
  float second = __bfloat162float(qkv[source_base + half_dim + pair]);
  if (bias != nullptr) {
    first += __bfloat162float(bias[bias_base + pair]);
    second += __bfloat162float(bias[bias_base + half_dim + pair]);
  }
  v_cache[destination_base + pair] = __float2bfloat16(first);
  v_cache[destination_base + half_dim + pair] = __float2bfloat16(second);
}

// One block per token. The rotation table (two axes of head_dim/4 angles) is
// built once per block in shared memory; each biased value rounds to BF16
// before rotation, exactly as the legacy kernel does.
__global__ void vision_qkv_rope_kernel(
    const __nv_bfloat16* qkv, const __nv_bfloat16* bias,
    const uint32_t* position_ids, __nv_bfloat16* q, __nv_bfloat16* k,
    __nv_bfloat16* v, int tokens, int heads, int head_dim, float theta) {
  const int token = blockIdx.x;
  const int half_dim = head_dim / 2;
  const int projection_width = heads * head_dim;
  const int fused_width = 3 * projection_width;
  const int pairs_per_projection = heads * half_dim;

  extern __shared__ float rope_smem[];
  const int quarter = half_dim / 2;
  float* rope_sin = rope_smem;  // [2][quarter]
  float* rope_cos = rope_smem + 2 * quarter;
  for (int slot = threadIdx.x; slot < 2 * quarter; slot += blockDim.x) {
    const int axis = slot / quarter;
    const int pair_in_axis = slot - axis * quarter;
    const float position = static_cast<float>(position_ids[token * 2 + axis]);
    const float frequency =
        powf(theta, -2.0f * static_cast<float>(pair_in_axis) / half_dim);
    sincosf(position * frequency, &rope_sin[slot], &rope_cos[slot]);
  }
  __syncthreads();

  for (int work = threadIdx.x; work < 2 * pairs_per_projection;
       work += blockDim.x) {
    const bool is_key = work >= pairs_per_projection;
    const int local = is_key ? work - pairs_per_projection : work;
    const int head = local / half_dim;
    const int pair = local - head * half_dim;
    const int projection_offset = is_key ? projection_width : 0;
    const int source =
        token * fused_width + projection_offset + head * head_dim;
    const int bias_base = projection_offset + head * head_dim;
    float first = __bfloat162float(qkv[source + pair]);
    float second = __bfloat162float(qkv[source + half_dim + pair]);
    if (bias != nullptr) {
      first += __bfloat162float(bias[bias_base + pair]);
      second += __bfloat162float(bias[bias_base + half_dim + pair]);
    }
    first = __bfloat162float(__float2bfloat16(first));
    second = __bfloat162float(__float2bfloat16(second));
    const int axis = pair < quarter ? 0 : 1;
    const int slot = axis * quarter + (pair < quarter ? pair : pair - quarter);
    const float sine = rope_sin[slot];
    const float cosine = rope_cos[slot];
    __nv_bfloat16* destination =
        (is_key ? k : q) + (token * heads + head) * head_dim;
    destination[pair] = __float2bfloat16(first * cosine - second * sine);
    destination[half_dim + pair] =
        __float2bfloat16(first * sine + second * cosine);
  }

  for (int col = threadIdx.x; col < projection_width; col += blockDim.x) {
    const int source = token * fused_width + 2 * projection_width + col;
    float value = __bfloat162float(qkv[source]);
    if (bias != nullptr)
      value += __bfloat162float(bias[2 * projection_width + col]);
    v[token * projection_width + col] = __float2bfloat16(value);
  }
}

// One block per (query, head, segment). F32 scores in shared memory; the
// softmax runs serially on thread 0 so the summation order is fixed.
__global__ void segmented_mha_bf16_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const uint32_t* offsets, __nv_bfloat16* output, int heads, int head_dim) {
  extern __shared__ float shared[];
  const int segment = blockIdx.z;
  const int begin = static_cast<int>(offsets[segment]);
  const int end = static_cast<int>(offsets[segment + 1]);
  const int tokens = end - begin;
  const int query = blockIdx.x;
  if (query >= tokens) return;
  float* scores = shared;
  float* warp_sums = scores + tokens;
  const int head = blockIdx.y;
  const int global_query = begin + query;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warps = blockDim.x >> 5;
  const __nv_bfloat16* query_ptr = q + (global_query * heads + head) * head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  for (int token = 0; token < tokens; ++token) {
    const __nv_bfloat16* key = k + ((begin + token) * heads + head) * head_dim;
    float dot = tid < head_dim
        ? __bfloat162float(query_ptr[tid]) * __bfloat162float(key[tid])
        : 0.0f;
    dot = warp_sum(dot);
    if (lane == 0) warp_sums[warp] = dot;
    __syncthreads();
    if (warp == 0) {
      float total = lane < warps ? warp_sums[lane] : 0.0f;
      total = warp_sum(total);
      if (lane == 0) scores[token] = total * scale;
    }
    __syncthreads();
  }
  if (tid == 0) {
    float maximum = -3.402823466e+38F;
    for (int token = 0; token < tokens; ++token)
      maximum = fmaxf(maximum, scores[token]);
    float denominator = 0.0f;
    for (int token = 0; token < tokens; ++token) {
      scores[token] = expf(scores[token] - maximum);
      denominator += scores[token];
    }
    for (int token = 0; token < tokens; ++token)
      scores[token] /= denominator;
  }
  __syncthreads();
  if (tid < head_dim) {
    float accumulator = 0.0f;
    for (int token = 0; token < tokens; ++token) {
      accumulator += scores[token] *
          __bfloat162float(v[((begin + token) * heads + head) * head_dim + tid]);
    }
    output[(global_query * heads + head) * head_dim + tid] =
        __float2bfloat16(accumulator);
  }
}

}  // namespace

int gqa_qkv_mrope_cache_bf16(const void* qkv, const void* bias,
                             const uint32_t* position_ids, void* q,
                             void* k_cache, void* v_cache, int tokens,
                             int q_heads, int kv_heads, int head_dim,
                             float theta, int section_h, int section_w,
                             int cache_offset, cudaStream_t stream) {
  if (qkv == nullptr || position_ids == nullptr || q == nullptr ||
      k_cache == nullptr || v_cache == nullptr || tokens <= 0 ||
      q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0 ||
      head_dim <= 0 || head_dim > 256 || head_dim % 2 != 0 ||
      !(theta > 0.0f) || section_h < 0 || section_w < 0 ||
      section_h + section_w > head_dim / 2 || cache_offset < 0) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  dim3 grid(tokens, q_heads + 2 * kv_heads, 1);
  gqa_qkv_mrope_cache_kernel<<<grid, head_dim / 2, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<const __nv_bfloat16*>(bias), position_ids,
      static_cast<__nv_bfloat16*>(q), static_cast<__nv_bfloat16*>(k_cache),
      static_cast<__nv_bfloat16*>(v_cache), tokens, q_heads, kv_heads,
      head_dim, theta, section_h, section_w, cache_offset);
  return static_cast<int>(cudaGetLastError());
}

int vision_qkv_rope_bf16(const void* qkv, const void* bias,
                         const uint32_t* position_ids, void* q, void* k,
                         void* v, int tokens, int heads, int head_dim,
                         float theta, cudaStream_t stream) {
  if (qkv == nullptr || position_ids == nullptr || q == nullptr ||
      k == nullptr || v == nullptr || tokens <= 0 || heads <= 0 ||
      head_dim <= 0 || head_dim > 256 || head_dim % 4 != 0 ||
      !(theta > 0.0f)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  // Two axes of head_dim/4 rotations, held as sine then cosine.
  const size_t rope_smem = static_cast<size_t>(head_dim) * sizeof(float);
  vision_qkv_rope_kernel<<<tokens, kThreads, rope_smem, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<const __nv_bfloat16*>(bias), position_ids,
      static_cast<__nv_bfloat16*>(q), static_cast<__nv_bfloat16*>(k),
      static_cast<__nv_bfloat16*>(v), tokens, heads, head_dim, theta);
  return static_cast<int>(cudaGetLastError());
}

int segmented_mha_bf16(const void* q, const void* k, const void* v,
                       const void* offsets, void* output, int segments,
                       int max_tokens, int heads, int head_dim,
                       cudaStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || offsets == nullptr ||
      output == nullptr || segments <= 0 || max_tokens <= 0 || heads <= 0 ||
      head_dim <= 0 || head_dim > kThreads) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  dim3 grid(max_tokens, heads, segments);
  const size_t shared = static_cast<size_t>(max_tokens + 8) * sizeof(float);
  segmented_mha_bf16_kernel<<<grid, kThreads, shared, stream>>>(
      static_cast<const __nv_bfloat16*>(q),
      static_cast<const __nv_bfloat16*>(k),
      static_cast<const __nv_bfloat16*>(v),
      static_cast<const uint32_t*>(offsets),
      static_cast<__nv_bfloat16*>(output), heads, head_dim);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace apxinf::cuda_new::vla_attn_ops
