// Copyright 2026 ApxInf contributors.
// Fixed-shape mixture-of-experts kernels: token router and weighted SwiGLU.
// Included by adapters/moe/fixed_bf16.cu, which owns the C ABI launchers.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace {

constexpr int kRouteThreads = 128;
constexpr int kRouteWarps = kRouteThreads / 32;
constexpr int kRouteMaxExperts = 64;
constexpr int kRouteMaxTopK = 8;

// One block per token. Warps split the experts; each warp reduces one dot
// product at a time with FP32 accumulation over BF16 inputs and a fixed
// butterfly order, so the result is deterministic. Thread 0 then selects the
// top-k of (sigmoid(logit) + bias) serially: E <= 64, k <= 8, and a serial scan
// gives lower-index-first tie-breaking without any atomics.
__global__ void moe_route_kernel(const __nv_bfloat16* __restrict__ x,
                                 const __nv_bfloat16* __restrict__ gate,
                                 const float* __restrict__ bias,
                                 float* __restrict__ out, int experts, int dim,
                                 int top_k, float routed_scale) {
  __shared__ float scores[kRouteMaxExperts];
  const int token = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const __nv_bfloat16* row = x + static_cast<int64_t>(token) * dim;
  const int vectors = dim / 8;

  for (int e = warp; e < experts; e += kRouteWarps) {
    const __nv_bfloat16* weight = gate + static_cast<int64_t>(e) * dim;
    float sum = 0.0f;
    for (int v = lane; v < vectors; v += 32) {
      const uint4 a = reinterpret_cast<const uint4*>(row)[v];
      const uint4 b = reinterpret_cast<const uint4*>(weight)[v];
      const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&a);
      const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float2 fa = __bfloat1622float2(pa[i]);
        const float2 fb = __bfloat1622float2(pb[i]);
        sum = fmaf(fa.x, fb.x, sum);
        sum = fmaf(fa.y, fb.y, sum);
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) scores[e] = 1.0f / (1.0f + expf(-sum));
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    bool taken[kRouteMaxExperts];
    int selected[kRouteMaxTopK];
    for (int e = 0; e < experts; ++e) taken[e] = false;
    float total = 0.0f;
    for (int k = 0; k < top_k; ++k) {
      int best = -1;
      float best_value = -FLT_MAX;
      for (int e = 0; e < experts; ++e) {
        const float value = scores[e] + bias[e];
        // Strict '>' keeps the lowest index on ties.
        if (!taken[e] && (best < 0 || value > best_value)) {
          best = e;
          best_value = value;
        }
      }
      taken[best] = true;
      selected[k] = best;
      total += scores[best];
    }
    float* result = out + static_cast<int64_t>(token) * experts;
    for (int e = 0; e < experts; ++e) result[e] = 0.0f;
    const float scale = routed_scale / (total + 1e-20f);
    for (int k = 0; k < top_k; ++k) {
      result[selected[k]] = scores[selected[k]] * scale;
    }
  }
}

__device__ __forceinline__ float silu_f32(float value) {
  return value / (1.0f + expf(-value));
}

// out[t, c] = bf16(silu(g) * u * w[t, c / block]); one rounding, FP32 inside.
__global__ void moe_swiglu_scaled_vec8_kernel(
    const uint4* __restrict__ gate, const uint4* __restrict__ up,
    const float* __restrict__ weights, uint4* __restrict__ out,
    int64_t vectors_per_row, int experts, int block_vectors,
    int64_t total_vectors) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < total_vectors; index += stride) {
    const int64_t token = index / vectors_per_row;
    const int64_t column = index - token * vectors_per_row;
    const float w = weights[token * experts + column / block_vectors];
    const uint4 g = gate[index];
    const uint4 u = up[index];
    const __nv_bfloat162* pg = reinterpret_cast<const __nv_bfloat162*>(&g);
    const __nv_bfloat162* pu = reinterpret_cast<const __nv_bfloat162*>(&u);
    uint4 result;
    __nv_bfloat162* pr = reinterpret_cast<__nv_bfloat162*>(&result);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 fg = __bfloat1622float2(pg[i]);
      const float2 fu = __bfloat1622float2(pu[i]);
      pr[i] = __floats2bfloat162_rn(silu_f32(fg.x) * fu.x * w,
                                    silu_f32(fg.y) * fu.y * w);
    }
    out[index] = result;
  }
}

__global__ void moe_swiglu_scaled_scalar_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up, const float* __restrict__ weights,
    __nv_bfloat16* __restrict__ out, int64_t columns, int experts,
    int block_columns, int64_t total) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < total; index += stride) {
    const int64_t token = index / columns;
    const int64_t column = index - token * columns;
    const float w = weights[token * experts + column / block_columns];
    out[index] = __float2bfloat16(silu_f32(__bfloat162float(gate[index])) *
                                  __bfloat162float(up[index]) * w);
  }
}

}  // namespace
