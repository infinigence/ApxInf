#pragma once

// Copyright 2026 apxinf contributors.
// Pure CUDA operators grouped by physical operation; launch policy lives under adapters/.

// Touch and rewrite a caller-owned buffer larger than L2. Autotuning launches
// this immediately before timing a candidate so activation/weight cache state
// is comparable across algorithms.
__global__ void l2_cache_evict_kernel(
    volatile uint32_t* buffer, size_t words, uint32_t seed) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (; index < words; index += stride) {
        uint32_t value = buffer[index];
        buffer[index] = value * 1664525u + 1013904223u + seed +
                        static_cast<uint32_t>(index);
    }
}

// ── KV Cache Append (no sync) ─────────────────────────────────────────────
//
// Cache layout: [n_kv_heads, max_seq_len, head_dim]
// New data layout: [append_len, n_kv_heads, head_dim]
// Copies new_data into cache starting at position seq_len.

__global__ void kv_cache_append_f32_kernel(
    float* cache, const float* new_data,
    uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t max_seq_len, uint32_t seq_len, uint32_t append_len)
{
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t h = blockIdx.y;
    uint32_t s = blockIdx.z;
    if (d >= head_dim || h >= n_kv_heads || s >= append_len) return;

    uint32_t src_idx = s * n_kv_heads * head_dim + h * head_dim + d;
    uint32_t dst_idx = h * max_seq_len * head_dim + (seq_len + s) * head_dim + d;
    cache[dst_idx] = new_data[src_idx];
}



// Append 1 row of new K/V data into the cache at position *pos_ptr.
// Cache layout: [n_kv_heads, max_seq_len, head_dim]. new_data: [n_kv_heads, head_dim].
__global__ void kv_cache_append_decode_f32_kernel(
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



// ── KV Cache Append (bf16) ────────────────────────────────────────────────

__global__ void kv_cache_append_bf16_kernel(
    __nv_bfloat16* cache, const __nv_bfloat16* new_data,
    uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t max_seq_len, uint32_t seq_len, uint32_t append_len)
{
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t h = blockIdx.y;
    uint32_t s = blockIdx.z;
    if (d >= head_dim || h >= n_kv_heads || s >= append_len) return;

    uint32_t src_idx = s * n_kv_heads * head_dim + h * head_dim + d;
    uint32_t dst_idx = h * max_seq_len * head_dim + (seq_len + s) * head_dim + d;
    cache[dst_idx] = new_data[src_idx];
}



__global__ void kv_cache_append_decode_bf16_kernel(
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




