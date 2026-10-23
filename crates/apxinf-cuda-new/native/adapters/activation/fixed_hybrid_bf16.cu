// Copyright 2026 ApxInf contributors.
// Fixed implementation for this operator family; ported without numerical changes.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {
#include "../../kernels/custom/gdn_policy.h"
#include "../../kernels/custom/math.cuh"
#include "../../kernels/custom/reduction.cuh"
#include "../../kernels/custom/pillow_bicubic_u8.cuh"
#include "../../kernels/custom/linear_attention.cuh"
#include "../../kernels/custom/gdn_raw_inverse_f1.cuh"
#include "../../kernels/custom/gdn_chunk_state_wmma.cuh"
#include "../../kernels/custom/gdn_chunk_gemm_tri.cuh"
template <bool RoundSilu = false>
__global__ void swiglu_bf16_kernel(
    const __nv_bfloat16* gate_up, __nv_bfloat16* output,
    int rows, int inner) {
  const int64_t count = static_cast<int64_t>(rows) * inner;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int row = static_cast<int>(index / inner);
    const int col = static_cast<int>(index % inner);
    const float gate = __bfloat162float(gate_up[static_cast<int64_t>(row) * 2 * inner + col]);
    const float up = __bfloat162float(gate_up[static_cast<int64_t>(row) * 2 * inner + inner + col]);
    float silu = gate / (1.0f + expf(-gate));
    if (RoundSilu) silu = __bfloat162float(__float2bfloat16(silu));
    output[index] = __float2bfloat16(silu * up);
  }
}

// The projection produced by the native fused SwiGLU weight has columns in
// [gate pair, up pair] tiles. Keep the same two BF16 rounding boundaries as
// `swiglu_bf16_kernel<true>` while undoing only that physical column order.
template <bool RoundSilu = false>
__global__ void swiglu_bf16_pair_interleaved_kernel(
    const __nv_bfloat16* gate_up, __nv_bfloat16* output,
    int rows, int inner, int pair) {
  const int64_t count = static_cast<int64_t>(rows) * inner;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int row = static_cast<int>(index / inner);
    const int col = static_cast<int>(index % inner);
    const int tile = col / pair;
    const int within = col - tile * pair;
    const int gate_col = tile * 2 * pair + within;
    const __nv_bfloat16* row_input =
        gate_up + static_cast<int64_t>(row) * 2 * inner;
    const float gate = __bfloat162float(row_input[gate_col]);
    const float up = __bfloat162float(row_input[gate_col + pair]);
    float silu = gate / (1.0f + expf(-gate));
    if (RoundSilu) silu = __bfloat162float(__float2bfloat16(silu));
    output[index] = __float2bfloat16(silu * up);
  }
}

// Eight output columns per thread through 16-byte accesses. The scalar loop
// also pays an integer divide and a modulo per element to recover (row, col);
// with inner a multiple of eight all eight land in one row, so that arithmetic
// happens once per eight. Per-element expressions are unchanged.
template <bool RoundSilu = false>
__global__ void swiglu_bf16_vec8_kernel(
    const __nv_bfloat16* __restrict__ gate_up, __nv_bfloat16* __restrict__ output,
    int rows, int inner) {
  const int64_t vec_per_row = inner / 8;
  const int64_t vec_count = static_cast<int64_t>(rows) * vec_per_row;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t v = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       v < vec_count; v += stride) {
    const int64_t row = v / vec_per_row;
    const int64_t col = (v - row * vec_per_row) * 8;
    const int64_t base = row * 2 * inner + col;
    const float4 g4 = *reinterpret_cast<const float4*>(gate_up + base);
    const float4 u4 = *reinterpret_cast<const float4*>(gate_up + base + inner);
    const __nv_bfloat16* gl = reinterpret_cast<const __nv_bfloat16*>(&g4);
    const __nv_bfloat16* ul = reinterpret_cast<const __nv_bfloat16*>(&u4);
    float4 out;
    __nv_bfloat16* ol = reinterpret_cast<__nv_bfloat16*>(&out);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float gate = __bfloat162float(gl[i]);
      const float up = __bfloat162float(ul[i]);
      float silu = gate / (1.0f + expf(-gate));
      if (RoundSilu) silu = __bfloat162float(__float2bfloat16(silu));
      ol[i] = __float2bfloat16(silu * up);
    }
    *reinterpret_cast<float4*>(output + row * inner + col) = out;
  }
}

// True when every 16-byte access the wide kernel makes is aligned: eight
// columns stay inside one row, and both halves of gate_up start on a multiple
// of eight elements.
__host__ __device__ __forceinline__ bool swiglu_vec8_ok(
    const void* gate_up, const void* output, int inner) {
  return (inner % 8) == 0 &&
         (reinterpret_cast<uintptr_t>(gate_up) % 16u) == 0 &&
         (reinterpret_cast<uintptr_t>(output) % 16u) == 0;
}



}  // namespace

extern "C" cudaError_t apxinf_hybrid_gelu_exact_bf16(
    const void* input, void* output, int64_t count, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  gelu_exact_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_hybrid_swiglu_bf16_rounded(
    const void* gate_up, void* output, int rows, int inner, cudaStream_t stream) {
  if (!gate_up || !output || rows <= 0 || inner <= 0) return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(rows) * inner;
  const int blocks = static_cast<int>((count + 255) / 256 > 65535 ? 65535 : (count + 255) / 256);
  if (swiglu_vec8_ok(gate_up, output, inner)) {
    const int vblocks = static_cast<int>((count / 8 + 255) / 256 > 65535
                                             ? 65535
                                             : (count / 8 + 255) / 256);
    swiglu_bf16_vec8_kernel<true><<<vblocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate_up),
        static_cast<__nv_bfloat16*>(output), rows, inner);
    return cudaGetLastError();
  }
  swiglu_bf16_kernel<true><<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate_up), static_cast<__nv_bfloat16*>(output), rows, inner);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_hybrid_swiglu_bf16_rounded_pair_interleaved(
    const void* gate_up, void* output, int rows, int inner, int pair,
    cudaStream_t stream) {
  if (!gate_up || !output || rows <= 0 || inner <= 0 || pair <= 0 ||
      inner % pair != 0) {
    return cudaErrorInvalidValue;
  }
  const int64_t count = static_cast<int64_t>(rows) * inner;
  const int blocks = static_cast<int>((count + 255) / 256 > 65535
                                          ? 65535
                                          : (count + 255) / 256);
  swiglu_bf16_pair_interleaved_kernel<true><<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate_up),
      static_cast<__nv_bfloat16*>(output), rows, inner, pair);
  return cudaGetLastError();
}


// Unrounded SwiGLU (the `RoundSilu=false` legacy form): the SiLU output feeds
// the multiply at full f32 precision. Distinct from the rounded variant above,
// which packed checkpoint path uses.
extern "C" cudaError_t apxinf_hybrid_swiglu_bf16(
    const void* gate_up, void* output, int rows, int inner, cudaStream_t stream) {
  if (!gate_up || !output || rows <= 0 || inner <= 0) return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(rows) * inner;
  const int blocks = static_cast<int>((count + 255) / 256 > 65535 ? 65535 : (count + 255) / 256);
  if (swiglu_vec8_ok(gate_up, output, inner)) {
    const int vblocks = static_cast<int>((count / 8 + 255) / 256 > 65535
                                             ? 65535
                                             : (count / 8 + 255) / 256);
    swiglu_bf16_vec8_kernel<false><<<vblocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate_up),
        static_cast<__nv_bfloat16*>(output), rows, inner);
    return cudaGetLastError();
  }
  swiglu_bf16_kernel<false><<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate_up), static_cast<__nv_bfloat16*>(output), rows, inner);
  return cudaGetLastError();
}
