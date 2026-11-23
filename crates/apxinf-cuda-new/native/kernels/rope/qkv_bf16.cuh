// Copyright 2026 ApxInf contributors.
// Q/K normalization, RoPE, QKV split, and cache-write kernels moved
// mechanically from the historical linear-attention aggregate.
#pragma once

#include "../custom/rms_reduction.cuh"

// Fused full-attention input preparation for the (q|gate)-per-head layout:
// per (token, head): per-head RMSNorm (1 + w semantics) on q/k, partial rotary
// from tables, K/V cache append. The gate half of each q head stays in the
// fused buffer for the post-attention sigmoid gate.
// Grid y: [0, q_heads) -> q, [q_heads, q_heads+kv_heads) -> k, rest -> v.
__global__ void full_attn_prepare_bf16_kernel(
    const __nv_bfloat16* fused, const __nv_bfloat16* q_norm_w, const __nv_bfloat16* k_norm_w,
    const __nv_bfloat16* cos, const __nv_bfloat16* sin,
    __nv_bfloat16* q_out, __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
    int cache_offset, int q_heads, int kv_heads, int head_dim,
    int rotary_dim, int64_t fused_width, int64_t cache_width, float eps) {
  const int token = blockIdx.x;
  const int slot = blockIdx.y;
  const int half = rotary_dim / 2;
  extern __shared__ float la_attn[];
  const int64_t row = static_cast<int64_t>(token) * fused_width;
  const int64_t table_base = static_cast<int64_t>(token) * rotary_dim;
  if (slot < q_heads + kv_heads) {
    const bool is_q = slot < q_heads;
    const int head = is_q ? slot : slot - q_heads;
    const __nv_bfloat16* norm_w = is_q ? q_norm_w : k_norm_w;
    const int64_t src = row + (is_q
        ? static_cast<int64_t>(head) * 2 * head_dim
        : static_cast<int64_t>(q_heads) * 2 * head_dim + head * head_dim);
    float partial = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
      const float v = __bfloat162float(fused[src + i]);
      la_attn[i] = v;
      partial += v * v;
    }
    __shared__ float warp_sums[32];
    for (int offset = 16; offset > 0; offset >>= 1)
      partial += __shfl_xor_sync(0xffffffff, partial, offset);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = partial;
    __syncthreads();
    if (warp == 0) {
      float v = (lane < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
      if (lane == 0) warp_sums[0] = v;
    }
    __syncthreads();
    const float rms = rsqrtf(warp_sums[0] / head_dim + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
      la_attn[i] = __bfloat162float(__float2bfloat16(
          la_attn[i] * rms * (1.0f + __bfloat162float(norm_w[i]))));
    }
    __syncthreads();
    __nv_bfloat16* dst = is_q
        ? q_out + (static_cast<int64_t>(token) * q_heads + head) * head_dim
        : k_cache + (static_cast<int64_t>(cache_offset + token)) * cache_width + head * head_dim;
    for (int p = threadIdx.x; p < half; p += blockDim.x) {
      const float a = la_attn[p];
      const float b = la_attn[half + p];
      const float c = __bfloat162float(cos[table_base + p]);
      const float s = __bfloat162float(sin[table_base + p]);
      const float t1 = __bfloat162float(__float2bfloat16(a * c));
      const float t2 = __bfloat162float(__float2bfloat16(-b * s));
      const float t3 = __bfloat162float(__float2bfloat16(b * c));
      const float t4 = __bfloat162float(__float2bfloat16(a * s));
      dst[p] = __float2bfloat16(t1 + t2);
      dst[half + p] = __float2bfloat16(t3 + t4);
    }
    for (int d = rotary_dim + threadIdx.x; d < head_dim; d += blockDim.x) {
      dst[d] = __float2bfloat16(la_attn[d]);
    }
  } else {
    const int head = slot - q_heads - kv_heads;
    const int64_t src = row + static_cast<int64_t>(q_heads) * 2 * head_dim
        + static_cast<int64_t>(kv_heads) * head_dim + head * head_dim;
    __nv_bfloat16* dst = v_cache + (static_cast<int64_t>(cache_offset + token)) * cache_width
        + head * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
      dst[i] = fused[src + i];
    }
  }
}

// Expert fused-QKV preparation: per (token, group-major slot) split the fused
// projection into query/gate/key/value, apply per-head RMSNorm (plain weight)
// and partial rotary to q/k. Grid y covers q_heads + kv_heads + kv_heads slots.
__global__ void expert_qkv_prepare_bf16_kernel(
    const __nv_bfloat16* fused, const __nv_bfloat16* q_norm_w, const __nv_bfloat16* k_norm_w,
    const __nv_bfloat16* cos, const __nv_bfloat16* sin,
    __nv_bfloat16* q_out, __nv_bfloat16* gate_out, __nv_bfloat16* k_out, __nv_bfloat16* v_out,
    int q_heads, int kv_heads, int head_dim, int rotary_dim,
    int64_t fused_width, float eps) {
  const int token = blockIdx.x;
  const int slot = blockIdx.y;
  const int half = rotary_dim / 2;
  const int heads_per_group = q_heads / kv_heads;
  const int group_width = (2 * heads_per_group + 2) * head_dim;
  const int64_t row = static_cast<int64_t>(token) * fused_width;
  const int64_t table_base = static_cast<int64_t>(token) * rotary_dim;
  extern __shared__ float la_expert[];
  if (slot < q_heads + kv_heads) {
    const bool is_q = slot < q_heads;
    const int head = is_q ? slot : slot - q_heads;
    const int group = is_q ? head / heads_per_group : head;
    const int64_t src = row + static_cast<int64_t>(group) * group_width
        + (is_q ? static_cast<int64_t>(head - group * heads_per_group) * head_dim
                : static_cast<int64_t>(2 * heads_per_group) * head_dim);
    const __nv_bfloat16* norm_w = is_q ? q_norm_w : k_norm_w;
    float partial = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
      const float v = __bfloat162float(fused[src + i]);
      la_expert[i] = v;
      partial += v * v;
    }
    __shared__ float warp_sums[32];
    for (int offset = 16; offset > 0; offset >>= 1)
      partial += __shfl_xor_sync(0xffffffff, partial, offset);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = partial;
    __syncthreads();
    if (warp == 0) {
      float v = (lane < (blockDim.x + 31) / 32) ? warp_sums[lane] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
      if (lane == 0) warp_sums[0] = v;
    }
    __syncthreads();
    float mean = warp_sums[0] / head_dim;
    if (static_cast<int64_t>(gridDim.x) * (is_q ? q_heads : kv_heads) >= 16
        && head_dim > 128 && head_dim % 4 == 0)
      mean = rms_vector_square_mean_bf16(fused + src, head_dim);
    const float rms = rsqrtf(__fadd_rn(mean, eps));
    __syncthreads();
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
      la_expert[i] = __bfloat162float(__float2bfloat16(
          la_expert[i] * rms * __bfloat162float(norm_w[i])));
    }
    __syncthreads();
    __nv_bfloat16* dst = is_q
        ? q_out + (static_cast<int64_t>(token) * q_heads + head) * head_dim
        : k_out + (static_cast<int64_t>(token) * kv_heads + head) * head_dim;
    for (int p = threadIdx.x; p < half; p += blockDim.x) {
      const float a = la_expert[p];
      const float b = la_expert[half + p];
      const float c = __bfloat162float(cos[table_base + p]);
      const float s = __bfloat162float(sin[table_base + p]);
      const float t1 = __bfloat162float(__float2bfloat16(a * c));
      const float t2 = __bfloat162float(__float2bfloat16(-b * s));
      const float t3 = __bfloat162float(__float2bfloat16(b * c));
      const float t4 = __bfloat162float(__float2bfloat16(a * s));
      dst[p] = __float2bfloat16(t1 + t2);
      dst[half + p] = __float2bfloat16(t3 + t4);
    }
    for (int d = rotary_dim + threadIdx.x; d < head_dim; d += blockDim.x) {
      dst[d] = __float2bfloat16(la_expert[d]);
    }
    if (is_q) {
      __nv_bfloat16* gate_dst = gate_out + (static_cast<int64_t>(token) * q_heads + head) * head_dim;
      const int64_t gate_src = row + static_cast<int64_t>(group) * group_width
          + static_cast<int64_t>(heads_per_group + head - group * heads_per_group) * head_dim;
      for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        gate_dst[i] = fused[gate_src + i];
      }
    }
  } else {
    const int head = slot - q_heads - kv_heads;
    const int64_t src = row + static_cast<int64_t>(head) * group_width
        + static_cast<int64_t>(2 * heads_per_group + 1) * head_dim;
    __nv_bfloat16* dst = v_out + (static_cast<int64_t>(token) * kv_heads + head) * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
      dst[i] = fused[src + i];
    }
  }
}

__global__ void full_attn_prepare_bf16_warp_kernel(
    const __nv_bfloat16* fused, const __nv_bfloat16* q_norm_w, const __nv_bfloat16* k_norm_w,
    const __nv_bfloat16* cos, const __nv_bfloat16* sin,
    __nv_bfloat16* q_out, __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
    int cache_offset, int q_heads, int kv_heads, int64_t fused_width,
    int64_t cache_width, float eps) {
  const int token = blockIdx.x;
  const int slot = blockIdx.y * 4 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (slot >= q_heads + 2 * kv_heads) return;
  const int64_t row = static_cast<int64_t>(token) * fused_width;
  if (slot >= q_heads + kv_heads) {
    const int head = slot - q_heads - kv_heads;
    const int64_t src = row + static_cast<int64_t>(q_heads) * 512
        + static_cast<int64_t>(kv_heads) * 256 + head * 256;
    __nv_bfloat16* dst = v_cache + static_cast<int64_t>(cache_offset + token) * cache_width
        + head * 256;
    #pragma unroll
    for (int j = 0; j < 8; ++j) dst[lane + 32 * j] = fused[src + lane + 32 * j];
    return;
  }
  const bool is_q = slot < q_heads;
  const int head = is_q ? slot : slot - q_heads;
  const __nv_bfloat16* norm_w = is_q ? q_norm_w : k_norm_w;
  const int64_t src = row + (is_q ? static_cast<int64_t>(head) * 512
      : static_cast<int64_t>(q_heads) * 512 + head * 256);
  float x[8], partial[4];
  #pragma unroll
  for (int j = 0; j < 8; ++j) x[j] = __bfloat162float(fused[src + lane + 32 * j]);
  // Match the original four virtual warps: each original thread owns d,d+128.
  #pragma unroll
  for (int w = 0; w < 4; ++w) partial[w] = x[w] * x[w] + x[w+4] * x[w+4];
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    #pragma unroll
    for (int w = 0; w < 4; ++w)
      partial[w] += __shfl_xor_sync(0xffffffff, partial[w], off);
  }
  #pragma unroll
  for (int w = 0; w < 4; ++w)
    partial[w] = __shfl_sync(0xffffffff, partial[w], 0);
  // Original CTA's second XOR tree combines lanes 0..3 as (0+2)+(1+3).
  const float sum = (partial[0] + partial[2]) + (partial[1] + partial[3]);
  const float rms = rsqrtf(sum / 256.0f + eps);
  float y[8];
  #pragma unroll
  for (int j = 0; j < 8; ++j)
    y[j] = __bfloat162float(__float2bfloat16(
        x[j] * rms * (1.0f + __bfloat162float(norm_w[lane + 32*j]))));
  __nv_bfloat16* dst = is_q
      ? q_out + (static_cast<int64_t>(token) * q_heads + head) * 256
      : k_cache + static_cast<int64_t>(cache_offset + token) * cache_width + head * 256;
  const int64_t table_base = static_cast<int64_t>(token) * 64;
  const float a = y[0], b = y[1];
  const float c = __bfloat162float(cos[table_base + lane]);
  const float s = __bfloat162float(sin[table_base + lane]);
  const float t1 = __bfloat162float(__float2bfloat16(a * c));
  const float t2 = __bfloat162float(__float2bfloat16(-b * s));
  const float t3 = __bfloat162float(__float2bfloat16(b * c));
  const float t4 = __bfloat162float(__float2bfloat16(a * s));
  dst[lane] = __float2bfloat16(t1 + t2);
  dst[32 + lane] = __float2bfloat16(t3 + t4);
  #pragma unroll
  for (int j = 2; j < 8; ++j) dst[lane + 32*j] = __float2bfloat16(y[j]);
}
