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

// ── KV Cache Quantization / Append (E4M3 with per-row scale) ─────────────
//
// The logical row is one (KV head, token) vector. Keeping one FP32 scale per
// row preserves the full cache while halving its payload relative to BF16.

__global__ void kv_cache_quantize_bf16_e4m3_kernel(
    const __nv_bfloat16* input, __nv_fp8_e4m3* output, float* scales,
    uint32_t rows, uint32_t head_dim)
{
    __shared__ float scratch[8];
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    float maximum = 0.0f;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        maximum = fmaxf(maximum, fabsf(__bfloat162float(
            input[static_cast<int64_t>(row) * head_dim + d])));
    }
    const float scale =
        fmaxf(block_max(maximum, scratch) / 448.0f, 1.0e-12f);
    if (threadIdx.x == 0) scales[row] = scale;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const int64_t index = static_cast<int64_t>(row) * head_dim + d;
        const float value = fminf(
            448.0f, fmaxf(-448.0f, __bfloat162float(input[index]) / scale));
        output[index] = static_cast<__nv_fp8_e4m3>(value);
    }
}

__global__ void kv_cache_append_bf16_e4m3_kernel(
    __nv_fp8_e4m3* cache, float* scales, const __nv_bfloat16* new_data,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_seq_len,
    uint32_t seq_len, uint32_t append_len)
{
    __shared__ float scratch[8];
    const uint32_t head = blockIdx.x;
    const uint32_t token = blockIdx.y;
    if (head >= n_kv_heads || token >= append_len) return;
    float maximum = 0.0f;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const uint32_t source =
            token * n_kv_heads * head_dim + head * head_dim + d;
        maximum = fmaxf(
            maximum, fabsf(__bfloat162float(new_data[source])));
    }
    const float scale =
        fmaxf(block_max(maximum, scratch) / 448.0f, 1.0e-12f);
    const uint32_t position = seq_len + token;
    const int64_t row = static_cast<int64_t>(head) * max_seq_len + position;
    if (threadIdx.x == 0) scales[row] = scale;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const uint32_t source =
            token * n_kv_heads * head_dim + head * head_dim + d;
        const int64_t destination = row * head_dim + d;
        const float value = fminf(448.0f, fmaxf(
            -448.0f, __bfloat162float(new_data[source]) / scale));
        cache[destination] = static_cast<__nv_fp8_e4m3>(value);
    }
}

__global__ void kv_cache_append_decode_bf16_e4m3_kernel(
    __nv_fp8_e4m3* cache, float* scales, const __nv_bfloat16* new_data,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_seq_len,
    const uint32_t* pos_ptr)
{
    __shared__ float scratch[8];
    const uint32_t head = blockIdx.x;
    if (head >= n_kv_heads) return;
    float maximum = 0.0f;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x)
        maximum = fmaxf(maximum, fabsf(__bfloat162float(
            new_data[static_cast<int64_t>(head) * head_dim + d])));
    const float scale =
        fmaxf(block_max(maximum, scratch) / 448.0f, 1.0e-12f);
    const int64_t row =
        static_cast<int64_t>(head) * max_seq_len + *pos_ptr;
    if (threadIdx.x == 0) scales[row] = scale;
    for (uint32_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float value = fminf(448.0f, fmaxf(-448.0f,
            __bfloat162float(new_data[static_cast<int64_t>(head) * head_dim + d])
                / scale));
        cache[row * head_dim + d] = static_cast<__nv_fp8_e4m3>(value);
    }
}




