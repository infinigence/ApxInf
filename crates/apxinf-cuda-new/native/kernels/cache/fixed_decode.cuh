// Copyright 2026 ApxInf contributors.
// Fixed decode kernels, moved without arithmetic changes from the legacy port.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
namespace {
#define FIXED_DECODE_BLOCK_SIZE 256
__global__ void llmdec_kv_cache_append_f32_kernel(
    float* cache, const float* new_data,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_seq_len,
    uint32_t seq_len, uint32_t append_len)
{
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t h = blockIdx.y;
    uint32_t s = blockIdx.z;
    if (d >= head_dim || h >= n_kv_heads || s >= append_len) return;
    uint32_t src_idx = s * n_kv_heads * head_dim + h * head_dim + d;
    uint32_t dst_idx = h * max_seq_len * head_dim + (seq_len + s) * head_dim + d;
    cache[dst_idx] = new_data[src_idx];
}

__global__ void llmdec_kv_cache_append_bf16_kernel(
    __nv_bfloat16* cache, const __nv_bfloat16* new_data,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_seq_len,
    uint32_t seq_len, uint32_t append_len)
{
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t h = blockIdx.y;
    uint32_t s = blockIdx.z;
    if (d >= head_dim || h >= n_kv_heads || s >= append_len) return;
    uint32_t src_idx = s * n_kv_heads * head_dim + h * head_dim + d;
    uint32_t dst_idx = h * max_seq_len * head_dim + (seq_len + s) * head_dim + d;
    cache[dst_idx] = new_data[src_idx];
}

__global__ void llmdec_kv_cache_append_decode_f32_kernel(
    float* cache, const float* new_data,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_seq_len,
    const uint32_t* pos_ptr)
{
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t h = blockIdx.y;
    if (d >= head_dim || h >= n_kv_heads) return;

    uint32_t pos = *pos_ptr;
    uint32_t src_idx = h * head_dim + d;
    uint32_t dst_idx = h * max_seq_len * head_dim + pos * head_dim + d;
    cache[dst_idx] = new_data[src_idx];
}

__global__ void llmdec_kv_cache_append_decode_bf16_kernel(
    __nv_bfloat16* cache, const __nv_bfloat16* new_data,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_seq_len,
    const uint32_t* pos_ptr)
{
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t h = blockIdx.y;
    if (d >= head_dim || h >= n_kv_heads) return;

    uint32_t pos = *pos_ptr;
    uint32_t src_idx = h * head_dim + d;
    uint32_t dst_idx = h * max_seq_len * head_dim + pos * head_dim + d;
    cache[dst_idx] = new_data[src_idx];
}
// Gather a head-major BF16 cache prefix into token-major dense storage for
// the unchanged FlashAttention prefill path.
__global__ void kv_cache_gather_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    uint32_t tokens, uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t max_seq_len, uint32_t kv_offset)
{
    uint32_t flat = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = tokens * n_kv_heads * head_dim;
    if (flat >= total) return;
    uint32_t d = flat % head_dim;
    uint32_t h = (flat / head_dim) % n_kv_heads;
    uint32_t tok = flat / (n_kv_heads * head_dim);
    uint32_t src_pos = kv_offset + tok;
    dst[flat] = src[((size_t)h * max_seq_len + src_pos) * head_dim + d];
}

}  // namespace
