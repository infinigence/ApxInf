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
}  // namespace

extern "C" cudaError_t apxinf_hybrid_sigmoid_gate_mul_bf16(
    void* attn, const void* fused, int rows, int heads, int head_dim,
    int64_t fused_width, cudaStream_t stream) {
  if (attn == nullptr || fused == nullptr || rows <= 0 || heads <= 0 ||
      head_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  // Vector BF16 route for D256 prefill. Inputs must be aligned and disjoint
  // because the vector kernel promises restrict; retain the generic fallback.
  if (rows > 1 && rows <= 65536 && heads == 16 && head_dim == 256 &&
      fused_width == 10240) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(attn);
    const uintptr_t f = reinterpret_cast<uintptr_t>(fused);
    const uint64_t a_bytes = static_cast<uint64_t>(rows) * 4096 * 2;
    const uint64_t f_bytes = static_cast<uint64_t>(rows) * 10240 * 2;
    const bool aligned = ((a | f) & 15U) == 0;
    const bool no_overflow = a <= UINTPTR_MAX - a_bytes && f <= UINTPTR_MAX - f_bytes;
    const bool disjoint = no_overflow && (a + a_bytes <= f || f + f_bytes <= a);
    if (aligned && disjoint) {
      sigmoid_gate_mul_bf16_vec8_kernel<<<rows * 2, 256, 0, stream>>>(
          static_cast<__nv_bfloat16*>(attn),
          static_cast<const __nv_bfloat16*>(fused), rows);
      return cudaGetLastError();
    }
  }
  sigmoid_gate_mul_bf16_kernel<<<rows, 256, 0, stream>>>(
      static_cast<__nv_bfloat16*>(attn),
      static_cast<const __nv_bfloat16*>(fused), heads, head_dim, fused_width);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_hybrid_expert_sigmoid_gate_mul_bf16(
    void* attn, const void* gate, int64_t count, cudaStream_t stream) {
  if (attn == nullptr || gate == nullptr || count <= 0) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  expert_sigmoid_gate_mul_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<__nv_bfloat16*>(attn),
      static_cast<const __nv_bfloat16*>(gate), count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_hybrid_concat7_cols_bf16(
    const void* s0, const void* s1, const void* s2, const void* s3,
    const void* s4, const void* s5, const void* s6, void* dst,
    int rows, int cols, int broadcast_mask, cudaStream_t stream) {
  if (s0 == nullptr || s1 == nullptr || s2 == nullptr || s3 == nullptr ||
      s4 == nullptr || s5 == nullptr || s6 == nullptr || dst == nullptr ||
      rows <= 0 || cols <= 0) {
    return cudaErrorInvalidValue;
  }
  concat7_cols_bf16_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(s0),
      static_cast<const __nv_bfloat16*>(s1),
      static_cast<const __nv_bfloat16*>(s2),
      static_cast<const __nv_bfloat16*>(s3),
      static_cast<const __nv_bfloat16*>(s4),
      static_cast<const __nv_bfloat16*>(s5),
      static_cast<const __nv_bfloat16*>(s6),
      static_cast<__nv_bfloat16*>(dst), rows, cols, broadcast_mask);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_hybrid_flow_update_f32(
    void* w, const void* endpoint, float remaining, float step, int64_t count,
    cudaStream_t stream) {
  if (w == nullptr || endpoint == nullptr || count <= 0 ||
      !(remaining > 0.0f) || !std::isfinite(step)) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  flow_update_f32_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<float*>(w), static_cast<const float*>(endpoint),
      remaining, step, count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_hybrid_suppress_logits_bf16(
    void* logits, const uint32_t* ids, int count, cudaStream_t stream) {
  if (logits == nullptr || ids == nullptr || count <= 0) {
    return cudaErrorInvalidValue;
  }
  const int blocks = (count + 63) / 64;
  suppress_logits_bf16_kernel<<<blocks, 64, 0, stream>>>(
      static_cast<__nv_bfloat16*>(logits), ids, count);
  return cudaGetLastError();
}
