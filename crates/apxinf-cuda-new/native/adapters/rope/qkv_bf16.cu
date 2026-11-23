// Copyright 2026 ApxInf contributors.
// Q/K normalization, RoPE, QKV split, and cache-write launchers. Kernel
// bodies and launch geometry are preserved from the migrated implementation.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {
#include "../../kernels/rope/qkv_bf16.cuh"
}  // namespace

extern "C" cudaError_t apxinf_rope_split_qkv_cache_bf16(
    const void* projection, const void* query_norm, const void* key_norm,
    const void* cos, const void* sin, void* query, void* key_cache,
    void* value_cache, int seq, int cache_offset, int query_heads,
    int kv_heads, int head_dim, int rotary_dim, int64_t projection_width,
    int64_t cache_width, float eps, cudaStream_t stream) {
  if (projection == nullptr || query_norm == nullptr || key_norm == nullptr ||
      cos == nullptr || sin == nullptr || query == nullptr ||
      key_cache == nullptr || value_cache == nullptr || seq <= 0 ||
      cache_offset < 0 || query_heads <= 0 || kv_heads <= 0 ||
      head_dim <= 0 || rotary_dim <= 0 || (rotary_dim & 1) != 0 ||
      rotary_dim > head_dim || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  if (seq > 1 && query_heads == 16 && kv_heads == 4 && head_dim == 256 &&
      rotary_dim == 64 && projection_width == 10240 &&
      cache_width == 1024) {
    full_attn_prepare_bf16_warp_kernel<<<dim3(seq, 6), 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projection),
        static_cast<const __nv_bfloat16*>(query_norm),
        static_cast<const __nv_bfloat16*>(key_norm),
        static_cast<const __nv_bfloat16*>(cos),
        static_cast<const __nv_bfloat16*>(sin),
        static_cast<__nv_bfloat16*>(query),
        static_cast<__nv_bfloat16*>(key_cache),
        static_cast<__nv_bfloat16*>(value_cache), cache_offset, query_heads,
        kv_heads, projection_width, cache_width, eps);
    return cudaGetLastError();
  }
  const size_t smem = static_cast<size_t>(head_dim) * sizeof(float);
  full_attn_prepare_bf16_kernel
      <<<dim3(seq, query_heads + 2 * kv_heads), 128, smem, stream>>>(
          static_cast<const __nv_bfloat16*>(projection),
          static_cast<const __nv_bfloat16*>(query_norm),
          static_cast<const __nv_bfloat16*>(key_norm),
          static_cast<const __nv_bfloat16*>(cos),
          static_cast<const __nv_bfloat16*>(sin),
          static_cast<__nv_bfloat16*>(query),
          static_cast<__nv_bfloat16*>(key_cache),
          static_cast<__nv_bfloat16*>(value_cache), cache_offset, query_heads,
          kv_heads, head_dim, rotary_dim, projection_width, cache_width, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rope_split_grouped_qkv_bf16(
    const void* projection, const void* query_norm, const void* key_norm,
    const void* cos, const void* sin, void* query, void* gate, void* key,
    void* value, int seq, int query_heads, int kv_heads, int head_dim,
    int rotary_dim, int64_t projection_width, float eps,
    cudaStream_t stream) {
  if (projection == nullptr || query_norm == nullptr || key_norm == nullptr ||
      cos == nullptr || sin == nullptr || query == nullptr || gate == nullptr ||
      key == nullptr || value == nullptr || seq <= 0 || query_heads <= 0 ||
      kv_heads <= 0 || head_dim <= 0 || rotary_dim <= 0 ||
      (rotary_dim & 1) != 0 || rotary_dim > head_dim ||
      query_heads % kv_heads != 0 || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t smem = static_cast<size_t>(head_dim) * sizeof(float);
  expert_qkv_prepare_bf16_kernel
      <<<dim3(seq, query_heads + 2 * kv_heads), 128, smem, stream>>>(
          static_cast<const __nv_bfloat16*>(projection),
          static_cast<const __nv_bfloat16*>(query_norm),
          static_cast<const __nv_bfloat16*>(key_norm),
          static_cast<const __nv_bfloat16*>(cos),
          static_cast<const __nv_bfloat16*>(sin),
          static_cast<__nv_bfloat16*>(query),
          static_cast<__nv_bfloat16*>(gate),
          static_cast<__nv_bfloat16*>(key),
          static_cast<__nv_bfloat16*>(value), query_heads, kv_heads, head_dim,
          rotary_dim, projection_width, eps);
  return cudaGetLastError();
}
