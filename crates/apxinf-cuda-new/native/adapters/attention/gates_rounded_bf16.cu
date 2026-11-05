// Copyright 2026 ApxInf contributors.
// Fixed family launcher; kernel body and launch parameters are unchanged.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {
#include "../../kernels/attention/gates_rounded_bf16.cuh"
}  // namespace

namespace apxinf::attention {

cudaError_t launch_output_gate_projection(
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

cudaError_t launch_output_gate_tensor_rounded(
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

}  // namespace apxinf::attention
