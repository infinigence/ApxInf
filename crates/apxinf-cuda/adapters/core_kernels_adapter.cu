// Copyright 2026 apxinf contributors.
// Stable C ABI and CUDA launch policy for core custom operators.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdint>

#define BLOCK_SIZE 256

#include "../kernels/custom/math.cuh"
#include "../kernels/custom/reduction.cuh"
#include "../kernels/custom/normalization.cuh"
#include "../kernels/custom/activation.cuh"
#include "../kernels/custom/attention.cuh"
#include "../kernels/custom/rope.cuh"
#include "../kernels/custom/cache.cuh"
#include "../kernels/custom/embedding.cuh"
#include "../kernels/custom/elementwise.cuh"
#include "../kernels/custom/selection.cuh"
#include "../kernels/custom/fused.cuh"

static dim3 apxinf_row_split_grid(uint32_t x_blocks, uint32_t rows)
{
    constexpr uint32_t kMaxGridY = 65535;
    uint32_t grid_z = static_cast<uint32_t>(
        (static_cast<uint64_t>(rows) + kMaxGridY - 1) / kMaxGridY);
    uint32_t grid_y = static_cast<uint32_t>(
        (static_cast<uint64_t>(rows) + grid_z - 1) / grid_z);
    return dim3(x_blocks, grid_y, grid_z);
}

extern "C" cudaError_t apxinf_rms_norm_f32(
    const void* input, const void* weight, void* output,
    uint32_t cols, uint32_t rows, float eps, void* stream)
{
    dim3 grid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, rows, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    rms_norm_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)input, (const float*)weight, (float*)output,
        cols, rows, eps);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_silu_f32(
    const void* input, void* output, uint32_t count, void* stream)
{
    dim3 grid((count + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    silu_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)input, (float*)output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_softmax_f32(
    const void* input, void* output, uint32_t cols, uint32_t rows, void* stream)
{
    dim3 grid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, rows, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    softmax_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)input, (float*)output, cols, rows);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rope_f32(
    const void* input, void* output,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float rope_theta, uint32_t pos_offset, void* stream)
{
    dim3 grid((head_dim / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, n_heads, seq_len);
    dim3 block(BLOCK_SIZE, 1, 1);
    rope_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)input, (float*)output,
        head_dim, n_heads, seq_len, rope_theta, pos_offset);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_add_f32(
    const void* a, const void* b, void* output, uint32_t count, void* stream)
{
    dim3 grid((count + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    add_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)a, (const float*)b, (float*)output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_mul_f32(
    const void* a, const void* b, void* output, uint32_t count, void* stream)
{
    dim3 grid((count + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    mul_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)a, (const float*)b, (float*)output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_embedding_f32(
    const void* table, const void* ids, void* output,
    uint32_t embed_dim, uint32_t seq_len, void* stream)
{
    dim3 grid((embed_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, seq_len, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    embedding_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)table, (const uint32_t*)ids, (float*)output,
        embed_dim, seq_len);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_causal_mask_f32(
    const void* input, void* output,
    uint32_t cols, uint32_t rows, uint32_t kv_offset, void* stream)
{
    dim3 grid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, rows, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    causal_mask_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)input, (float*)output, cols, rows, kv_offset);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rope_batched_f32(
    const void* input, void* output,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float rope_theta, uint32_t pos_offset, void* stream)
{
    dim3 grid((head_dim / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, n_heads, seq_len);
    dim3 block(BLOCK_SIZE, 1, 1);
    rope_batched_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)input, (float*)output,
        head_dim, n_heads, seq_len, rope_theta, pos_offset);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_attention_softmax_f32(
    const void* scores, void* output,
    uint32_t cols, uint32_t rows, uint32_t kv_offset, uint32_t n_heads, void* stream)
{
    dim3 grid = apxinf_row_split_grid(1, rows);
    dim3 block(BLOCK_SIZE, 1, 1);
    attention_softmax_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)scores, (float*)output, cols, rows, kv_offset, n_heads);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_kv_cache_append_f32(
    void* cache, const void* new_data,
    uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t max_seq_len, uint32_t seq_len, uint32_t append_len,
    void* stream)
{
    dim3 grid((head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, n_kv_heads, append_len);
    dim3 block(BLOCK_SIZE, 1, 1);
    kv_cache_append_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (float*)cache, (const float*)new_data,
        n_kv_heads, head_dim, max_seq_len, seq_len, append_len);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_scale_f32(
    const void* input, void* output, uint32_t count, float scale, void* stream)
{
    dim3 grid((count + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    scale_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)input, (float*)output, count, scale);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rope_decode_f32(
    const void* input, void* output,
    uint32_t head_dim, uint32_t n_heads,
    float rope_theta, const void* pos_ptr, void* stream)
{
    dim3 grid((head_dim / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, n_heads, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    rope_decode_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)input, (float*)output,
        head_dim, n_heads, rope_theta, (const uint32_t*)pos_ptr);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_attention_softmax_decode_f32(
    const void* scores, void* output,
    uint32_t cols, uint32_t n_heads, const void* pos_ptr, void* stream)
{
    dim3 grid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, n_heads, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    attention_softmax_decode_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)scores, (float*)output, cols, n_heads, (const uint32_t*)pos_ptr);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_kv_cache_append_decode_f32(
    void* cache, const void* new_data,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_seq_len,
    const void* pos_ptr, void* stream)
{
    dim3 grid((head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, n_kv_heads, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    kv_cache_append_decode_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (float*)cache, (const float*)new_data,
        n_kv_heads, head_dim, max_seq_len, (const uint32_t*)pos_ptr);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_silu_bf16(
    const void* input, void* output, uint32_t count, void* stream)
{
    dim3 grid((count + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    silu_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_silu_mul_bf16(
    const void* gate_up, void* output, uint32_t inter, void* stream)
{
    dim3 grid((inter + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    silu_mul_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)gate_up, (__nv_bfloat16*)output, inter);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_silu_mul_separate_bf16(
    const void* gate, const void* up, void* output, uint32_t count, void* stream)
{
    if (gate == nullptr || up == nullptr || output == nullptr)
        return cudaErrorInvalidConfiguration;
    dim3 grid((count + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    silu_mul_separate_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up,
        (__nv_bfloat16*)output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_silu_mul_packed_rows_exact_bf16(
    const void* gate_up, void* output, uint32_t rows, uint32_t inter,
    void* stream)
{
    if (gate_up == nullptr || output == nullptr || rows == 0 || inter == 0)
        return cudaErrorInvalidConfiguration;
    uint64_t count = static_cast<uint64_t>(rows) * inter;
    uint64_t blocks = (count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks == 0 || blocks > 0xffffffffULL)
        return cudaErrorInvalidConfiguration;
    silu_mul_packed_rows_exact_bf16_kernel<<<
        static_cast<uint32_t>(blocks), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)gate_up, (__nv_bfloat16*)output, rows, inter);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rms_norm_bf16(
    const void* input, const void* weight, void* output,
    uint32_t cols, uint32_t rows, float eps, void* stream)
{
    // One block per row. BLOCK_SIZE threads (256), strided over cols.
    dim3 grid(rows, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    size_t smem = cols * sizeof(float);
    rms_norm_bf16_kernel<<<grid, block, smem, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (const __nv_bfloat16*)weight,
        (__nv_bfloat16*)output, cols, rows, eps);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rms_norm_add_bf16(
    void* x_inout, const void* delta, const void* weight, void* output,
    uint32_t cols, uint32_t rows, float eps, void* stream)
{
    // One block per row. BLOCK_SIZE threads (256), strided over cols.
    // Shared mem: cols * sizeof(float) for x_new.
    dim3 grid(rows, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    size_t smem = cols * sizeof(float);
    rms_norm_add_bf16_kernel<false><<<grid, block, smem, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)x_inout, (const __nv_bfloat16*)delta,
        (const __nv_bfloat16*)weight, (__nv_bfloat16*)output,
        cols, rows, eps);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rms_norm_add_exact_bf16(
    void* x_inout, const void* delta, const void* weight, void* output,
    uint32_t cols, uint32_t rows, float eps, void* stream)
{
    dim3 grid(rows, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    size_t smem = cols * sizeof(float);
    rms_norm_add_bf16_kernel<true><<<grid, block, smem, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)x_inout, (const __nv_bfloat16*)delta,
        (const __nv_bfloat16*)weight, (__nv_bfloat16*)output,
        cols, rows, eps);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_softmax_bf16(
    const void* input, void* output, uint32_t cols, uint32_t rows, void* stream)
{
    dim3 grid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, rows, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    softmax_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output, cols, rows);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rope_bf16(
    const void* input, void* output,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float rope_theta, uint32_t pos_offset, void* stream)
{
    dim3 grid((head_dim / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, n_heads, seq_len);
    dim3 block(BLOCK_SIZE, 1, 1);
    rope_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output,
        head_dim, n_heads, seq_len, rope_theta, pos_offset);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_add_bf16(
    const void* a, const void* b, void* output, uint32_t count, void* stream)
{
    dim3 grid((count + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    add_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)a, (const __nv_bfloat16*)b, (__nv_bfloat16*)output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_mul_bf16(
    const void* a, const void* b, void* output, uint32_t count, void* stream)
{
    dim3 grid((count + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    mul_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)a, (const __nv_bfloat16*)b, (__nv_bfloat16*)output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_embedding_bf16(
    const void* table, const void* ids, void* output,
    uint32_t embed_dim, uint32_t seq_len, void* stream)
{
    dim3 grid((embed_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, seq_len, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    embedding_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)table, (const uint32_t*)ids,
        (__nv_bfloat16*)output, embed_dim, seq_len);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_causal_mask_bf16(
    const void* input, void* output,
    uint32_t cols, uint32_t rows, uint32_t kv_offset, void* stream)
{
    dim3 grid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, rows, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    causal_mask_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output,
        cols, rows, kv_offset);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rope_batched_bf16(
    const void* input, void* output,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float rope_theta, uint32_t pos_offset, void* stream)
{
    dim3 grid((head_dim / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, n_heads, seq_len);
    dim3 block(BLOCK_SIZE, 1, 1);
    rope_batched_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output,
        head_dim, n_heads, seq_len, rope_theta, pos_offset);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_attention_softmax_bf16(
    const void* scores, void* output,
    uint32_t cols, uint32_t rows, uint32_t kv_offset, uint32_t n_heads,
    float score_scale, void* stream)
{
    dim3 grid = apxinf_row_split_grid(1, rows);
    dim3 block(BLOCK_SIZE, 1, 1);
    attention_softmax_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)scores, (__nv_bfloat16*)output,
        cols, rows, kv_offset, n_heads, score_scale, 0);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_attention_softmax_bf16_gqa_packed(
    const void* scores, void* output,
    uint32_t cols, uint32_t rows, uint32_t kv_offset, uint32_t n_heads,
    uint32_t gqa_ratio, float score_scale, void* stream)
{
    if (gqa_ratio == 0 || n_heads % gqa_ratio != 0)
        return cudaErrorInvalidConfiguration;
    dim3 grid = apxinf_row_split_grid(1, rows);
    dim3 block(BLOCK_SIZE, 1, 1);
    attention_softmax_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)scores, (__nv_bfloat16*)output,
        cols, rows, kv_offset, n_heads, score_scale, gqa_ratio);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_attention_softmax_bf16_scale_in_place(
    void* scores_output,
    uint32_t cols, uint32_t rows, uint32_t kv_offset, uint32_t n_heads,
    uint32_t gqa_ratio, float score_scale, void* stream)
{
    if (scores_output == nullptr || gqa_ratio == 0 || n_heads % gqa_ratio != 0)
        return cudaErrorInvalidConfiguration;
    dim3 grid = apxinf_row_split_grid(1, rows);
    dim3 block(BLOCK_SIZE, 1, 1);
    attention_softmax_bf16_scale_in_place_kernel<<<
        grid, block, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)scores_output, cols, rows, kv_offset, n_heads,
        score_scale, gqa_ratio);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_attention_softmax_bf16_exp_cache(
    const void* scores, void* output,
    uint32_t cols, uint32_t rows, uint32_t kv_offset, uint32_t n_heads, void* stream)
{
    dim3 grid = apxinf_row_split_grid(1, rows);
    dim3 block(BLOCK_SIZE, 1, 1);
    size_t shared_bytes = static_cast<size_t>(cols) * sizeof(float);
    attention_softmax_bf16_exp_cache_kernel<<<grid, block, shared_bytes, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)scores, (__nv_bfloat16*)output,
        cols, rows, kv_offset, n_heads);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_attention_softmax_bf16_scaled_exp_cache(
    const void* scores, void* output,
    uint32_t cols, uint32_t rows, uint32_t kv_offset, uint32_t n_heads,
    float score_scale, void* stream)
{
    if (scores == nullptr || output == nullptr || cols == 0 || cols > 4096 ||
        rows <= n_heads || n_heads == 0 || rows % n_heads != 0 ||
        !(score_scale > 0.0f))
        return cudaErrorInvalidConfiguration;
    dim3 grid = apxinf_row_split_grid(1, rows);
    dim3 block(BLOCK_SIZE, 1, 1);
    size_t shared_bytes = static_cast<size_t>(cols) * sizeof(float);
    attention_softmax_bf16_scaled_exp_cache_kernel<<<
        grid, block, shared_bytes, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)scores, (__nv_bfloat16*)output,
        cols, rows, kv_offset, n_heads, score_scale);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_attention_softmax_bf16_global_exp_cache(
    const void* scores, void* output, void* numerators,
    uint32_t cols, uint32_t rows, uint32_t kv_offset, uint32_t n_heads,
    void* stream)
{
    dim3 grid = apxinf_row_split_grid(1, rows);
    dim3 block(BLOCK_SIZE, 1, 1);
    cudaStream_t s = (cudaStream_t)stream;
    attention_softmax_bf16_exp_global_cache_kernel<<<grid, block, 0, s>>>(
        (const __nv_bfloat16*)scores, (__nv_bfloat16*)output,
        (float*)numerators,
        cols, rows, kv_offset, n_heads);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_kv_cache_append_bf16(
    void* cache, const void* new_data,
    uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t max_seq_len, uint32_t seq_len, uint32_t append_len,
    void* stream)
{
    dim3 grid((head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, n_kv_heads, append_len);
    dim3 block(BLOCK_SIZE, 1, 1);
    kv_cache_append_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)cache, (const __nv_bfloat16*)new_data,
        n_kv_heads, head_dim, max_seq_len, seq_len, append_len);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_scale_bf16(
    const void* input, void* output, uint32_t count, float scale, void* stream)
{
    dim3 grid((count + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    scale_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output, count, scale);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rope_decode_bf16(
    const void* input, void* output,
    uint32_t head_dim, uint32_t n_heads,
    float rope_theta, const void* pos_ptr, void* stream)
{
    dim3 grid((head_dim / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, n_heads, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    rope_decode_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output,
        head_dim, n_heads, rope_theta, (const uint32_t*)pos_ptr);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_attention_softmax_decode_bf16(
    const void* scores, void* output,
    uint32_t cols, uint32_t n_heads, const void* pos_ptr, void* stream)
{
    dim3 grid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, n_heads, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    attention_softmax_decode_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)scores, (__nv_bfloat16*)output,
        cols, n_heads, (const uint32_t*)pos_ptr);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_kv_cache_append_decode_bf16(
    void* cache, const void* new_data,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_seq_len,
    const void* pos_ptr, void* stream)
{
    dim3 grid((head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, n_kv_heads, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    kv_cache_append_decode_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)cache, (const __nv_bfloat16*)new_data,
        n_kv_heads, head_dim, max_seq_len, (const uint32_t*)pos_ptr);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rope_k_write_bf16(
    const void* k_in, void* k_cache,
    uint32_t head_dim, uint32_t n_kv_heads, uint32_t max_seq_len,
    float rope_theta, const void* pos_ptr, void* stream)
{
    dim3 grid((head_dim / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, n_kv_heads, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    rope_k_write_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)k_in, (__nv_bfloat16*)k_cache,
        head_dim, n_kv_heads, max_seq_len, rope_theta, (const uint32_t*)pos_ptr);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rope_mrope_bf16(
    const void* input, void* output,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float theta, const void* pos_ids,
    uint32_t sec_h, uint32_t sec_w, void* stream)
{
    dim3 grid((head_dim / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, n_heads, seq_len);
    dim3 block(BLOCK_SIZE, 1, 1);
    rope_mrope_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output,
        head_dim, n_heads, seq_len, theta,
        (const uint32_t*)pos_ids, sec_h, sec_w);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rope_mrope_decode_bf16(
    const void* input, void* output,
    uint32_t head_dim, uint32_t n_heads,
    float theta, const void* pos_ids,
    uint32_t sec_h, uint32_t sec_w, void* stream)
{
    dim3 grid((head_dim / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, n_heads, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    rope_mrope_decode_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output,
        head_dim, n_heads, theta,
        (const uint32_t*)pos_ids, sec_h, sec_w);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_layer_norm_bf16(
    const void* input, const void* weight, const void* bias, void* output,
    uint32_t cols, uint32_t rows, float eps, void* stream)
{
    dim3 grid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, rows, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    layer_norm_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (const __nv_bfloat16*)weight,
        (const __nv_bfloat16*)bias, (__nv_bfloat16*)output,
        cols, rows, eps);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gelu_tanh_bf16(
    const void* input, void* output, uint32_t count, void* stream)
{
    dim3 grid((count + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    gelu_tanh_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_add_bias_bf16(
    const void* input, const void* bias, void* output,
    uint32_t cols, uint32_t rows, void* stream)
{
    dim3 grid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, rows, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    add_bias_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (const __nv_bfloat16*)bias,
        (__nv_bfloat16*)output, cols, rows);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rope_vision_2d_bf16(
    const void* input, void* output,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float theta, const void* pos_ids, void* stream)
{
    dim3 grid((head_dim / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, n_heads, seq_len);
    dim3 block(BLOCK_SIZE, 1, 1);
    rope_vision_2d_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output,
        head_dim, n_heads, seq_len, theta, (const uint32_t*)pos_ids);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rope_tmrope_bf16(
    const void* input, void* output,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float theta, const void* pos_ids, uint32_t sec_t, uint32_t sec_h,
    void* stream)
{
    dim3 grid((head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, n_heads, seq_len);
    dim3 block(BLOCK_SIZE, 1, 1);
    rope_tmrope_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output,
        head_dim, n_heads, seq_len, theta, (const uint32_t*)pos_ids,
        sec_t, sec_h);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rope_tmrope_kv_write_bf16(
    const void* k_in, const void* v_in, void* k_cache, void* v_cache,
    uint32_t head_dim, uint32_t n_kv_heads, uint32_t max_seq_len,
    float theta, const void* pos_ids, uint32_t sec_t, uint32_t sec_h,
    const void* cache_pos, void* stream)
{
    dim3 grid((head_dim + BLOCK_SIZE - 1) / BLOCK_SIZE, n_kv_heads, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    rope_tmrope_kv_write_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)k_in, (const __nv_bfloat16*)v_in,
        (__nv_bfloat16*)k_cache, (__nv_bfloat16*)v_cache,
        head_dim, n_kv_heads, max_seq_len, theta,
        (const uint32_t*)pos_ids, sec_t, sec_h,
        (const uint32_t*)cache_pos);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_vision_sdpa_bf16(
    const void* q, const void* k, const void* v, void* out,
    uint32_t seq_len, uint32_t n_heads, uint32_t head_dim, float scale, void* stream)
{
    if (head_dim == 0 || head_dim > 128) return cudaErrorInvalidConfiguration;
    dim3 grid(seq_len, n_heads, 1);
    dim3 block(32, 1, 1);
    size_t smem = (seq_len + 1) * sizeof(float);
    vision_sdpa_bf16_kernel<<<grid, block, smem, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
        (__nv_bfloat16*)out, seq_len, n_heads, head_dim, scale, nullptr);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_grouped_sdpa_bf16(
    const void* q, const void* k, const void* v, void* out,
    uint32_t seq_len, uint32_t n_heads, uint32_t head_dim, float scale,
    const void* group_ids, void* stream)
{
    if (head_dim == 0 || head_dim > 128 || group_ids == nullptr)
        return cudaErrorInvalidConfiguration;
    dim3 grid(seq_len, n_heads, 1);
    dim3 block(32, 1, 1);
    size_t smem = (seq_len + 1) * sizeof(float);
    vision_sdpa_bf16_kernel<<<grid, block, smem, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
        (__nv_bfloat16*)out, seq_len, n_heads, head_dim, scale,
        (const uint32_t*)group_ids);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_grouped_indexed_sdpa_bf16(
    const void* q, const void* k, const void* v, void* out,
    uint32_t seq_len, uint32_t n_heads, uint32_t head_dim, float scale,
    const void* group_ids, const void* group_offsets,
    const void* group_indices, uint32_t group_count, void* stream)
{
    if (head_dim == 0 || head_dim > 128 || group_count == 0 ||
        group_ids == nullptr || group_offsets == nullptr || group_indices == nullptr)
        return cudaErrorInvalidConfiguration;
    dim3 grid(seq_len, n_heads, 1);
    dim3 block(32, 1, 1);
    size_t smem = (seq_len + 1) * sizeof(float);
    vision_grouped_indexed_sdpa_bf16_kernel<<<grid, block, smem, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)k,
        (const __nv_bfloat16*)v, (__nv_bfloat16*)out,
        seq_len, n_heads, head_dim, scale,
        (const uint32_t*)group_ids, (const uint32_t*)group_offsets,
        (const uint32_t*)group_indices, group_count);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_pack_grouped_qkv_bf16(
    const void* q, const void* k, const void* v,
    void* packed_q, void* packed_k, void* packed_v,
    const void* group_indices, uint32_t rows, uint32_t row_elements,
    void* stream)
{
    if (q == nullptr || k == nullptr || v == nullptr || packed_q == nullptr ||
        packed_k == nullptr || packed_v == nullptr || group_indices == nullptr ||
        rows == 0 || row_elements == 0 || row_elements % 8 != 0)
        return cudaErrorInvalidConfiguration;
    uint32_t vectors_per_row = row_elements / 8;
    uint64_t vector_count = static_cast<uint64_t>(rows) * vectors_per_row;
    uint64_t blocks = (vector_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks == 0 || blocks > 0xffffffffULL) return cudaErrorInvalidConfiguration;
    vision_pack_grouped_qkv_bf16_kernel<<<
        static_cast<uint32_t>(blocks), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(
        (const uint4*)q, (const uint4*)k, (const uint4*)v,
        (uint4*)packed_q, (uint4*)packed_k, (uint4*)packed_v,
        (const uint32_t*)group_indices, rows, vectors_per_row);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_unpack_grouped_rows_bf16(
    const void* packed, void* output, const void* group_indices,
    uint32_t rows, uint32_t row_elements, void* stream)
{
    if (packed == nullptr || output == nullptr || group_indices == nullptr ||
        rows == 0 || row_elements == 0 || row_elements % 8 != 0)
        return cudaErrorInvalidConfiguration;
    uint32_t vectors_per_row = row_elements / 8;
    uint64_t vector_count = static_cast<uint64_t>(rows) * vectors_per_row;
    uint64_t blocks = (vector_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks == 0 || blocks > 0xffffffffULL) return cudaErrorInvalidConfiguration;
    vision_unpack_grouped_rows_bf16_kernel<<<
        static_cast<uint32_t>(blocks), BLOCK_SIZE, 0, (cudaStream_t)stream>>>(
        (const uint4*)packed, (uint4*)output,
        (const uint32_t*)group_indices, rows, vectors_per_row);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_flash_attn_decode_bf16(
    const void* q, const void* k_cache, const void* v_cache, void* out,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t bucket_kv_len, uint32_t max_seq_len,
    float scale, const void* pos_ptr, void* stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid(n_heads, 1, 1);
    dim3 block(SPLITK_WARPS * 32, 1, 1);
    if (head_dim == 64) {
        flash_attn_decode_bf16_splitk_kernel<64, SPLITK_WARPS><<<grid, block, 0, s>>>(
            (const __nv_bfloat16*)q, (const __nv_bfloat16*)k_cache,
            (const __nv_bfloat16*)v_cache, (__nv_bfloat16*)out,
            n_heads, n_kv_heads, bucket_kv_len, max_seq_len, scale,
            (const uint32_t*)pos_ptr);
    } else if (head_dim == 128) {
        flash_attn_decode_bf16_splitk_kernel<128, SPLITK_WARPS><<<grid, block, 0, s>>>(
            (const __nv_bfloat16*)q, (const __nv_bfloat16*)k_cache,
            (const __nv_bfloat16*)v_cache, (__nv_bfloat16*)out,
            n_heads, n_kv_heads, bucket_kv_len, max_seq_len, scale,
            (const uint32_t*)pos_ptr);
    } else {
        return cudaErrorInvalidConfiguration;
    }
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_argmax_bf16(
    const void* logits, uint32_t n, void* partials, void* out, void* stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    argmax_bf16_partials_kernel<<<APXINF_ARGMAX_PARTIAL_BLOCKS, 256, 0, s>>>(
        (const __nv_bfloat16*)logits, n, (ArgmaxPair*)partials);
    argmax_pair_final_kernel<<<1, 256, 0, s>>>(
        (const ArgmaxPair*)partials, APXINF_ARGMAX_PARTIAL_BLOCKS,
        (uint32_t*)out);
    return cudaGetLastError();
}
