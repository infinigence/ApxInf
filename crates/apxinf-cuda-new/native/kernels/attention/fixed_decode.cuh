// Copyright 2026 ApxInf contributors.
// Fixed decode kernels, moved without arithmetic changes from the legacy port.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
namespace {
#define FIXED_DECODE_BLOCK_SIZE 256
__global__ void attention_softmax_f32_kernel(
    const float* scores, float* output,
    uint32_t cols, uint32_t rows, uint32_t kv_offset, uint32_t n_heads)
{
    uint32_t n_col_blocks = (cols + blockDim.x - 1) / blockDim.x;
    uint32_t row = blockIdx.x / n_col_blocks;
    uint32_t col = (blockIdx.x % n_col_blocks) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    uint32_t seq_pos = row / n_heads;
    uint32_t valid_cols = min(seq_pos + kv_offset + 1, cols);
    float max_val = -INFINITY;
    for (uint32_t c = 0; c < valid_cols; c++)
        max_val = fmaxf(max_val, scores[row * cols + c]);
    float sum_exp = 0.0f;
    for (uint32_t c = 0; c < valid_cols; c++)
        sum_exp += expf(scores[row * cols + c] - max_val);
    if (col < cols) {
        output[row * cols + col] = col < valid_cols
            ? expf(scores[row * cols + col] - max_val) / sum_exp
            : 0.0f;
    }
}

__global__ void attention_softmax_bf16_kernel(
    const __nv_bfloat16* scores, __nv_bfloat16* output,
    uint32_t cols, uint32_t rows, uint32_t kv_offset, uint32_t n_heads)
{
    uint32_t n_col_blocks = (cols + blockDim.x - 1) / blockDim.x;
    uint32_t row = blockIdx.x / n_col_blocks;
    uint32_t col = (blockIdx.x % n_col_blocks) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    uint32_t seq_pos = row / n_heads;
    uint32_t valid_cols = min(seq_pos + kv_offset + 1, cols);
    float max_val = -INFINITY;
    for (uint32_t c = 0; c < valid_cols; c++)
        max_val = fmaxf(max_val, __bfloat162float(scores[row * cols + c]));
    float sum_exp = 0.0f;
    for (uint32_t c = 0; c < valid_cols; c++)
        sum_exp += expf(__bfloat162float(scores[row * cols + c]) - max_val);
    if (col < cols) {
        float value = col < valid_cols
            ? expf(__bfloat162float(scores[row * cols + col]) - max_val) / sum_exp
            : 0.0f;
        output[row * cols + col] = __float2bfloat16(value);
    }
}

__global__ void attention_softmax_decode_f32_kernel(
    const float* scores, float* output,
    uint32_t cols, uint32_t n_heads, const uint32_t* pos_ptr)
{
    uint32_t row = blockIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_heads) return;

    uint32_t valid_cols = min(*pos_ptr + 1, cols);
    float max_val = -INFINITY;
    for (uint32_t c = 0; c < valid_cols; c++) {
        max_val = fmaxf(max_val, scores[row * cols + c]);
    }
    float sum_exp = 0.0f;
    for (uint32_t c = 0; c < valid_cols; c++) {
        sum_exp += expf(scores[row * cols + c] - max_val);
    }
    if (col < cols) {
        if (col < valid_cols) {
            output[row * cols + col] = expf(scores[row * cols + col] - max_val) / sum_exp;
        } else {
            output[row * cols + col] = 0.0f;
        }
    }
}
// ── Flash-decoding (split-K) — legacy kernels/custom/attention.cuh ───────

#ifndef LLMDEC_SPLITK_WARPS
#define LLMDEC_SPLITK_WARPS 16
#endif

template<int HEAD_DIM, int WARPS>
__global__ void flash_attn_decode_bf16_splitk_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache, __nv_bfloat16* out,
    uint32_t n_heads, uint32_t n_kv_heads,
    uint32_t bucket_kv_len, uint32_t max_seq_len,
    float scale, const uint32_t* pos_ptr)
{
    constexpr int ELEMS_PER_THREAD = HEAD_DIM / 32;
    uint32_t q_head = blockIdx.x;
    uint32_t gqa_ratio = n_heads / n_kv_heads;
    uint32_t kv_head = q_head / gqa_ratio;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    uint32_t pos = *pos_ptr;
    uint32_t valid_len = pos + 1;
    if (valid_len > bucket_kv_len) valid_len = bucket_kv_len;

    // Load Q into shared memory once; every warp reads the same Q.
    __shared__ float q_sm[HEAD_DIM];
    if (warp_id == 0) {
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++)
            q_sm[i * 32 + lane] = __bfloat162float(q[q_head * HEAD_DIM + i * 32 + lane]);
    }
    __syncthreads();
    float q_reg[ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) q_reg[i] = q_sm[i * 32 + lane];

    const __nv_bfloat16* k_base =
        k_cache + kv_head * max_seq_len * HEAD_DIM;
    const __nv_bfloat16* v_base =
        v_cache + kv_head * max_seq_len * HEAD_DIM;

    // Each warp's private online-softmax over its strided timesteps.
    float m = -INFINITY;
    float l = 0.0f;
    float acc[ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) acc[i] = 0.0f;

    for (uint32_t t = warp_id; t < valid_len; t += WARPS) {
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            float kv = __bfloat162float(k_base[t * HEAD_DIM + i * 32 + lane]);
            dot += q_reg[i] * kv;
        }
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, off);
        dot *= scale;

        float m_new = fmaxf(m, dot);
        float p = expf(dot - m_new);
        float exp_m = expf(m - m_new);
        l = l * exp_m + p;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++)
            acc[i] = acc[i] * exp_m + p * __bfloat162float(v_base[t * HEAD_DIM + i * 32 + lane]);
        m = m_new;
    }

    // Stage each warp's (m, l, acc) into shared memory and merge.
    __shared__ float warp_m[WARPS];
    __shared__ float warp_l[WARPS];
    __shared__ float warp_acc[WARPS][HEAD_DIM];
    if (lane == 0) { warp_m[warp_id] = m; warp_l[warp_id] = l; }
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++)
        warp_acc[warp_id][i * 32 + lane] = acc[i];
    __syncthreads();

    // Warp 0 merges all WARPS states into the final output.
    if (warp_id == 0) {
        float m_total = -INFINITY;
        #pragma unroll
        for (int w = 0; w < WARPS; w++) m_total = fmaxf(m_total, warp_m[w]);
        float l_total = 0.0f;
        float acc_total[ELEMS_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) acc_total[i] = 0.0f;
        #pragma unroll
        for (int w = 0; w < WARPS; w++) {
            float factor = expf(warp_m[w] - m_total);
            l_total += warp_l[w] * factor;
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_THREAD; i++)
                acc_total[i] += warp_acc[w][i * 32 + lane] * factor;
        }
        float inv_l = (l_total > 0.0f) ? (1.0f / l_total) : 0.0f;
        __nv_bfloat16* out_row = out + q_head * HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++)
            out_row[i * 32 + lane] = __float2bfloat16(acc_total[i] * inv_l);
    }
}

}  // namespace
