// Copyright 2026 ApxInf contributors.
//
// KV-cache append for the portable `apxinf_core::KvCache` trait. See
// `cache_ops.h` for the layout contract.

#include "cache_ops.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace apxinf::cuda_new::cache_ops {
namespace {

constexpr int kThreads = 128;

// Cache layout: [1, max_seq_len, n_kv_heads, head_dim].
// new_data layout: [append_len, n_kv_heads, head_dim].
__global__ void append_bf16_kernel(__nv_bfloat16* __restrict__ cache,
                                   const __nv_bfloat16* __restrict__ new_data,
                                   int n_kv_heads, int head_dim, int max_seq_len,
                                   int seq_len, int append_len) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  const int h = blockIdx.y;
  const int s = blockIdx.z;
  if (d >= head_dim || h >= n_kv_heads || s >= append_len) return;
  const long long src = (long long)s * n_kv_heads * head_dim +
                        (long long)h * head_dim + d;
  const long long dst = (long long)(seq_len + s) * n_kv_heads * head_dim +
                        (long long)h * head_dim + d;
  cache[dst] = new_data[src];
}

}  // namespace

int append_bf16(void* cache, const void* new_data, int n_kv_heads, int head_dim,
                int max_seq_len, int seq_len, int append_len,
                cudaStream_t stream) {
  if (cache == nullptr || new_data == nullptr || n_kv_heads <= 0 ||
      head_dim <= 0 || max_seq_len <= 0 || seq_len < 0 || append_len <= 0 ||
      seq_len + append_len > max_seq_len) {
    return -1;
  }
  const dim3 grid((head_dim + kThreads - 1) / kThreads, n_kv_heads, append_len);
  append_bf16_kernel<<<grid, kThreads, 0, stream>>>(
      static_cast<__nv_bfloat16*>(cache),
      static_cast<const __nv_bfloat16*>(new_data), n_kv_heads, head_dim,
      max_seq_len, seq_len, append_len);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

}  // namespace apxinf::cuda_new::cache_ops
