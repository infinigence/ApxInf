// Copyright 2026 apxinf contributors.
// Stable C ABI and CUDA launch policy for core custom operators.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

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
#include "../kernels/custom/quantization.cuh"
#include "../kernels/custom/fused.cuh"

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
    dim3 grid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, rows, 1);
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
    const void* gate, const void* up, void* output, uint32_t count,
    void* stream) {
    const char* packed_value = std::getenv("APXINF_SILU_MUL_SEPARATE_BF16_PACKED4");
    const bool packed = packed_value != nullptr && std::strcmp(packed_value, "1") == 0 &&
        (count & 3U) == 0 &&
        (reinterpret_cast<uintptr_t>(gate) & 7U) == 0 &&
        (reinterpret_cast<uintptr_t>(up) & 7U) == 0 &&
        (reinterpret_cast<uintptr_t>(output) & 7U) == 0;
    if (packed) {
        const uint32_t quad_count = count / 4;
        int blocks = static_cast<int>((quad_count + BLOCK_SIZE - 1) / BLOCK_SIZE);
        blocks = blocks > 512 ? 512 : blocks;
        silu_mul_separate_bf16_packed4_kernel<<<blocks, BLOCK_SIZE, 0,
            (cudaStream_t)stream>>>(
            (const Bf16x4*)gate, (const Bf16x4*)up,
            (Bf16x4*)output, quad_count);
        return cudaGetLastError();
    }
    dim3 grid((count + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    silu_mul_separate_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up,
        (__nv_bfloat16*)output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_silu_mul_quant_bf16_e4m3(
    const void* gate, const void* up, void* output, int64_t count,
    float scale, void* stream) {
    if (gate == nullptr || up == nullptr || output == nullptr || count <= 0 ||
        !(scale > 0.0f)) return cudaErrorInvalidValue;
    constexpr int threads = 256;
    const int64_t pairs = (count + 1) / 2;
    int blocks = static_cast<int>((pairs + threads - 1) / threads);
    blocks = blocks > 1024 ? 1024 : blocks;
    silu_mul_quant_bf16_e4m3_kernel<<<blocks, threads, 0,
        static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(gate),
        static_cast<const __nv_bfloat16*>(up),
        static_cast<__nv_fp8_e4m3*>(output), count, 1.0f / scale);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_packed_gate_up_quant_bf16_e4m3(
    const void* gate_up, void* output, int rows, int inner,
    float scale, void* stream) {
    if (gate_up == nullptr || output == nullptr || rows <= 0 || inner <= 0 ||
        !(scale > 0.0f)) return cudaErrorInvalidValue;
    constexpr int threads = 256;
    const int64_t count = static_cast<int64_t>(rows) * inner;
    int blocks = static_cast<int>((count + threads - 1) / threads);
    blocks = blocks > 1024 ? 1024 : blocks;
    packed_gate_up_quant_bf16_e4m3_kernel<<<blocks, threads, 0,
        static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16*>(gate_up),
        static_cast<__nv_fp8_e4m3*>(output), rows, inner, 1.0f / scale);
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
    rms_norm_add_bf16_kernel<<<grid, block, smem, (cudaStream_t)stream>>>(
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
    const char* packed4_value = std::getenv("APXINF_ADD_BF16_PACKED4");
    const bool packed4_enabled =
        packed4_value == nullptr || std::strcmp(packed4_value, "0") != 0;
    if (packed4_enabled && (count & 3U) == 0 &&
        (reinterpret_cast<uintptr_t>(a) & 7U) == 0 &&
        (reinterpret_cast<uintptr_t>(b) & 7U) == 0 &&
        (reinterpret_cast<uintptr_t>(output) & 7U) == 0) {
        const uint32_t packed_count = count / 4;
        dim3 grid((packed_count + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
        dim3 block(BLOCK_SIZE, 1, 1);
        add_bf16_packed4_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
            static_cast<const Bf16x4*>(a), static_cast<const Bf16x4*>(b),
            static_cast<Bf16x4*>(output), packed_count);
        return cudaGetLastError();
    }
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
    uint32_t cols, uint32_t rows, uint32_t kv_offset, uint32_t n_heads, void* stream)
{
    dim3 grid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, rows, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    attention_softmax_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)scores, (__nv_bfloat16*)output,
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

extern "C" cudaError_t apxinf_prepare_mrope_cos_sin_f32(
    void* table, uint32_t head_dim, uint32_t seq_len, float theta,
    const void* pos_ids, uint32_t sec_h, uint32_t sec_w, void* stream)
{
    dim3 block(64, 1, 1);
    dim3 grid((head_dim / 2 + block.x - 1) / block.x, seq_len, 1);
    prepare_mrope_cos_sin_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (float2*)table, head_dim, seq_len, theta,
        (const uint32_t*)pos_ids, sec_h, sec_w);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rope_mrope_precomputed_bf16(
    const void* input, void* output, const void* table,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len, void* stream)
{
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid((head_dim / 2 + block.x - 1) / block.x, n_heads, seq_len);
    rope_mrope_precomputed_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output,
        (const float2*)table, head_dim, n_heads, seq_len);
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
    if (rows == 0 || cols == 0 || rows > INT32_MAX || cols > INT32_MAX ||
        !(eps > 0.0f)) {
        return cudaErrorInvalidConfiguration;
    }
    // One cooperative block computes each row's statistics once. The former
    // tiled implementation redundantly recomputed them for every column.
    const size_t shared_bytes = static_cast<size_t>(cols) * sizeof(float);
    // Fixed launch geometry. The former variant read a process-global
    // APXINF_GR00T_PRECISION env var to widen bf16 blocks to 512 threads,
    // which perturbed other in-process models and made the shipped output
    // diverge from the benchmarked value. Threads are now a fixed constant so
    // every caller (shipped Gr00tPolicy included) uses one deterministic path.
    const int threads = BLOCK_SIZE;
    layer_norm_bf16_kernel<<<rows, threads, shared_bytes, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (const __nv_bfloat16*)weight,
        (const __nv_bfloat16*)bias, (__nv_bfloat16*)output,
        static_cast<int>(rows), static_cast<int>(cols), eps);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_adaptive_layer_norm_bf16(
    const void* input, const void* modulation, void* output,
    uint32_t rows, uint32_t cols, float eps, void* stream)
{
    if (rows == 0 || cols == 0 || !(eps > 0.0f)) {
        return cudaErrorInvalidConfiguration;
    }
    // Fixed launch geometry. The former variant read a process-global
    // APXINF_GR00T_PRECISION env var to widen bf16 blocks to 512 threads,
    // which perturbed other in-process models and made the shipped output
    // diverge from the benchmarked value. Threads are now a fixed constant so
    // every caller (shipped Gr00tPolicy included) uses one deterministic path.
    const int threads = BLOCK_SIZE;
    adaptive_layer_norm_bf16_kernel<<<rows, threads, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (const __nv_bfloat16*)modulation,
        (__nv_bfloat16*)output, rows, cols, eps);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_adaptive_layer_norm_quant_bf16_e4m3(
    const void* input, const void* modulation, void* output, void* quantized,
    uint32_t rows, uint32_t cols, float eps, float scale, void* stream)
{
    if (rows == 0 || cols == 0 || !(eps > 0.0f) ||
        !(scale > 0.0f)) {
        return cudaErrorInvalidConfiguration;
    }
    const int threads = BLOCK_SIZE;
    adaptive_layer_norm_quant_bf16_e4m3_kernel<<<
        rows, threads, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (const __nv_bfloat16*)modulation,
        (__nv_bfloat16*)output, (__nv_fp8_e4m3*)quantized,
        rows, cols, eps, 1.0f / scale);
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

extern "C" cudaError_t apxinf_rope_vision_2d_pair_bf16(
    const void* q, const void* k, void* q_out, void* k_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float theta, const void* pos_ids, void* stream)
{
    uint64_t total = (uint64_t)seq_len * n_heads * (head_dim / 2);
    dim3 grid((uint32_t)((total + BLOCK_SIZE - 1) / BLOCK_SIZE), 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    rope_vision_2d_pair_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)k,
        (__nv_bfloat16*)q_out, (__nv_bfloat16*)k_out,
        head_dim, n_heads, seq_len, theta, (const uint32_t*)pos_ids);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qkv_split_bias_vision_rope_bf16(
    const void* qkv, const void* bias, void* q_out, void* k_out, void* v_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float theta, const void* pos_ids, void* stream)
{
    uint64_t total = (uint64_t)seq_len * n_heads * (head_dim / 2);
    dim3 grid((uint32_t)((total + BLOCK_SIZE - 1) / BLOCK_SIZE), 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    qkv_split_bias_vision_rope_bf16_kernel<false><<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)qkv, (const __nv_bfloat16*)bias,
        (__nv_bfloat16*)q_out, (__nv_bfloat16*)k_out, (__nv_bfloat16*)v_out,
        head_dim, n_heads, seq_len, theta, (const uint32_t*)pos_ids, nullptr);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_prepare_vision_rope_cos_sin_f32(
    void* table, uint32_t head_dim, uint32_t seq_len, float theta,
    const void* pos_ids, void* stream) {
    dim3 grid((head_dim / 2 + 63) / 64, seq_len, 1);
    prepare_vision_rope_cos_sin_f32_kernel<<<grid, 64, 0, (cudaStream_t)stream>>>(
        (float2*)table, head_dim, seq_len, theta, (const uint32_t*)pos_ids);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_qkv_split_bias_vision_rope_precomputed_bf16(
    const void* qkv, const void* bias, void* q_out, void* k_out, void* v_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    const void* rotation_table, void* stream) {
    uint64_t total = (uint64_t)seq_len * n_heads * (head_dim / 2);
    dim3 grid((uint32_t)((total + BLOCK_SIZE - 1) / BLOCK_SIZE), 1, 1);
    dim3 block(BLOCK_SIZE, 1, 1);
    qkv_split_bias_vision_rope_bf16_kernel<true><<<grid, block, 0,
        (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)qkv, (const __nv_bfloat16*)bias,
        (__nv_bfloat16*)q_out, (__nv_bfloat16*)k_out, (__nv_bfloat16*)v_out,
        head_dim, n_heads, seq_len, 0.0f, nullptr,
        (const float2*)rotation_table);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_vision_sdpa_bf16(
    const void* q, const void* k, const void* v, void* out,
    uint32_t seq_len, uint32_t n_heads, uint32_t head_dim, float scale, void* stream)
{
    // Only head_dim=64 (Qwen3-VL vision) is supported; the kernel uses a
    // 32-thread / 2-element-per-thread layout.
    if (head_dim != 64) return cudaErrorInvalidConfiguration;
    dim3 grid(seq_len, n_heads, 1);
    dim3 block(32, 1, 1);
    size_t smem = (seq_len + 1) * sizeof(float);
    noncausal_sdpa_bf16_kernel<<<grid, block, smem, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
        (__nv_bfloat16*)out, seq_len, seq_len, n_heads, head_dim, scale);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_noncausal_sdpa_bf16(
    const void* q, const void* k, const void* v, void* out,
    uint32_t query_len, uint32_t key_value_len,
    uint32_t n_heads, uint32_t head_dim, float scale, void* stream)
{
    // Keep dynamic shared memory within the portable 48 KiB per-block floor.
    // The kernel uses `(key_value_len + 1) * sizeof(float)` bytes.
    constexpr uint32_t kMaxPortableKeyValueLength = 12287;
    if (query_len == 0 || key_value_len == 0 ||
        key_value_len > kMaxPortableKeyValueLength || head_dim == 0 ||
        head_dim > 64 || (head_dim % 2) != 0) {
        return cudaErrorInvalidConfiguration;
    }
    dim3 grid(query_len, n_heads, 1);
    dim3 block(32, 1, 1);
    size_t smem = (key_value_len + 1) * sizeof(float);
    noncausal_sdpa_bf16_kernel<<<grid, block, smem, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)q, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
        (__nv_bfloat16*)out, query_len, key_value_len, n_heads, head_dim, scale);
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
    const void* logits, uint32_t n, void* out, void* stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    // One block of 256 threads — vocab (32k) / 256 = 128 elems/thread.
    argmax_bf16_kernel<<<1, 256, 0, s>>>(
        (const __nv_bfloat16*)logits, n, (uint32_t*)out);
    return cudaGetLastError();
}
