// Copyright 2026 ApxInf contributors.
// LLM decode fast-path primitives for the llama / qwen3-vl allocation-free
// decode graph, ported from the legacy apxinf-cuda adapters
// (adapters/core_kernels_adapter.cu) and kernels/custom/{activation,
// attention, elementwise, embedding, fused, normalization}.cuh. Symbols carry
// the apxinf_llmdec_ prefix because the legacy crate links into the same
// binary and its own launchers are already present.
//
// Arithmetic is copied verbatim. The one deliberate adaptation is the KV
// cache ADDRESSING in the three cache-touching kernels (append, rope-K-write,
// flash decode): the legacy kernels index a head-major
// [kv_heads, max_seq, head_dim] cache, but cuda-new's `CudaKvCache` — which
// prefill fills through `ops::cache_append` and reads through
// `ops::kv_cache_attention` — is token-major [1, max_seq, kv_heads, head_dim].
// The per-(head, position, element) values and their accumulation order are
// unchanged; only the linear index of each cache element differs.
//
// Kernels that do not touch the cache (decode RoPE for Q, decode mRoPE) are
// reused from native/kernels/primitives/rope.cuh — already byte-identical
// ports of the legacy kernels. Everything lives in an anonymous namespace so
// names cannot collide with identically named kernels in other translation
// units.

#include "../../include/apxinf_cuda/llm_decode.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

#define LLMDEC_BLOCK_SIZE 256

// Decode RoPE (Q path) and decode mRoPE primitives, byte-identical legacy
// ports; the cache-writing variants below are NOT taken from here.
#include "../../kernels/primitives/rope.cuh"

// ── Token-major KV cache kernels ──────────────────────────────────────────
//
// cuda-new's `CudaKvCache` is token-major [1, max_seq, kv_heads, head_dim]
// (shared with prefill `ops::cache_append` and `ops::kv_cache_attention`).
// These are the legacy decode cache kernels with only the cache element
// index rewritten for that layout; the per-element arithmetic is unchanged.

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
    uint32_t dst_idx = pos * n_kv_heads * head_dim + h * head_dim + d;
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
    uint32_t dst_idx = pos * n_kv_heads * head_dim + h * head_dim + d;
    cache[dst_idx] = new_data[src_idx];
}

// Fused single-token RoPE + K-cache write, token-major destination.
__global__ void llmdec_rope_k_write_bf16_kernel(
    const __nv_bfloat16* k_in,
    __nv_bfloat16* k_cache,
    uint32_t head_dim, uint32_t n_kv_heads, uint32_t max_seq_len,
    float rope_theta, const uint32_t* pos_ptr)
{
    uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head_idx = blockIdx.y;
    if (pair_idx >= head_dim / 2) return;

    uint32_t pos = *pos_ptr;
    float freq    = 1.0f / powf(rope_theta, 2.0f * (float)pair_idx / (float)head_dim);
    float angle   = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    uint32_t src_base = head_idx * head_dim;
    uint32_t half = head_dim / 2;
    float x0 = __bfloat162float(k_in[src_base + pair_idx]);
    float x1 = __bfloat162float(k_in[src_base + half + pair_idx]);

    uint32_t dst_base = pos * n_kv_heads * head_dim + head_idx * head_dim;
    k_cache[dst_base + pair_idx]        = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    k_cache[dst_base + half + pair_idx] = __float2bfloat16(x0 * sin_val + x1 * cos_val);
}

// ── SiLU — legacy kernels/custom/activation.cuh ──────────────────────────

__global__ void silu_f32_kernel(
    const float* input, float* output, uint32_t count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    float x = input[gid];
    output[gid] = x / (1.0f + expf(-x));
}

__global__ void silu_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output, uint32_t count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    float x = __bfloat162float(input[gid]);
    float y = x / (1.0f + expf(-x));
    output[gid] = __float2bfloat16(y);
}

// Fused SiLU + Mul (bf16) over a packed [gate | up] buffer.
__global__ void silu_mul_bf16_kernel(
    const __nv_bfloat16* gate_up, __nv_bfloat16* output, uint32_t inter)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= inter) return;
    float g = __bfloat162float(gate_up[gid]);
    float u = __bfloat162float(gate_up[gid + inter]);
    float s = g / (1.0f + expf(-g));
    output[gid] = __float2bfloat16(s * u);
}

// ── Elementwise — legacy kernels/custom/elementwise.cuh ──────────────────

__global__ void add_f32_kernel(
    const float* a, const float* b, float* output, uint32_t count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    output[gid] = a[gid] + b[gid];
}

__global__ void mul_f32_kernel(
    const float* a, const float* b, float* output, uint32_t count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    output[gid] = a[gid] * b[gid];
}

__global__ void add_bf16_kernel(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* output, uint32_t count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    float x = __bfloat162float(a[gid]) + __bfloat162float(b[gid]);
    output[gid] = __float2bfloat16(x);
}

__global__ void mul_bf16_kernel(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* output, uint32_t count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    float x = __bfloat162float(a[gid]) * __bfloat162float(b[gid]);
    output[gid] = __float2bfloat16(x);
}

// ── Embedding — legacy kernels/custom/embedding.cuh ──────────────────────

__global__ void embedding_f32_kernel(
    const float* table, const uint32_t* ids, float* output,
    uint32_t embed_dim, uint32_t seq_len)
{
    uint32_t dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t seq_idx = blockIdx.y;
    if (dim_idx >= embed_dim || seq_idx >= seq_len) return;

    uint32_t token_id = ids[seq_idx];
    output[seq_idx * embed_dim + dim_idx] = table[token_id * embed_dim + dim_idx];
}

__global__ void embedding_bf16_kernel(
    const __nv_bfloat16* table, const uint32_t* ids, __nv_bfloat16* output,
    uint32_t embed_dim, uint32_t seq_len)
{
    uint32_t dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t seq_idx = blockIdx.y;
    if (dim_idx >= embed_dim || seq_idx >= seq_len) return;

    uint32_t token_id = ids[seq_idx];
    output[seq_idx * embed_dim + dim_idx] = table[token_id * embed_dim + dim_idx];
}

// ── RMSNorm — legacy kernels/custom/normalization.cuh ────────────────────

__global__ void rms_norm_f32_kernel(
    const float* input,
    const float* weight,
    float* output,
    uint32_t cols,
    uint32_t rows,
    float eps)
{
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (col >= cols || row >= rows) return;

    uint32_t offset = row * cols;
    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < cols; i++) {
        float v = input[offset + i];
        sum_sq += v * v;
    }
    float rms = rsqrtf(sum_sq / (float)cols + eps);
    output[offset + col] = input[offset + col] * rms * weight[col];
}

// Shared-memory reduction: one block per row, strided load, warp-shuffle
// reduction.
__global__ void rms_norm_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, __nv_bfloat16* output,
    uint32_t cols, uint32_t rows, float eps)
{
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    uint32_t tid = threadIdx.x;
    uint32_t offset = row * cols;

    extern __shared__ float x_buf[];
    __shared__ float s_sum;

    float partial = 0.0f;
    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        float v = __bfloat162float(input[offset + i]);
        x_buf[i] = v;
        partial += v * v;
    }

    for (int off = 16; off > 0; off >>= 1)
        partial += __shfl_xor_sync(0xffffffff, partial, off);
    __shared__ float warp_sums[32];
    uint32_t warp_id = tid / 32;
    uint32_t lane = tid % 32;
    if (lane == 0) warp_sums[warp_id] = partial;
    __syncthreads();
    if (warp_id == 0) {
        float v = (tid < (blockDim.x + 31) / 32) ? warp_sums[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, off);
        if (lane == 0) s_sum = v;
    }
    __syncthreads();
    float rms = rsqrtf(s_sum / (float)cols + eps);

    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        float w = __bfloat162float(weight[i]);
        output[offset + i] = __float2bfloat16(x_buf[i] * rms * w);
    }
}

// ── Fused RMSNorm + residual add (bf16) — legacy kernels/custom/fused.cuh ─

__global__ void rms_norm_add_bf16_kernel(
    __nv_bfloat16* x_inout, const __nv_bfloat16* delta,
    const __nv_bfloat16* weight, __nv_bfloat16* output,
    uint32_t cols, uint32_t rows, float eps)
{
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    uint32_t tid = threadIdx.x;
    uint32_t offset = row * cols;

    extern __shared__ float smem[];
    float* x_new = smem;        // [cols]
    __shared__ float s_sum;

    float partial = 0.0f;
    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        float xv = __bfloat162float(x_inout[offset + i]);
        float dv = __bfloat162float(delta[offset + i]);
        float xn = xv + dv;
        x_new[i] = xn;
        partial += xn * xn;
    }

    for (int off = 16; off > 0; off >>= 1)
        partial += __shfl_xor_sync(0xffffffff, partial, off);
    __shared__ float warp_sums[32];
    uint32_t warp_id = tid / 32;
    uint32_t lane = tid % 32;
    if (lane == 0) warp_sums[warp_id] = partial;
    __syncthreads();
    if (warp_id == 0) {
        float v = (tid < (blockDim.x + 31) / 32) ? warp_sums[tid] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, off);
        if (lane == 0) s_sum = v;
    }
    __syncthreads();
    float rms = rsqrtf(s_sum / (float)cols + eps);

    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        float xn = x_new[i];
        float w  = __bfloat162float(weight[i]);
        x_inout[offset + i] = __float2bfloat16(xn);
        output[offset + i]  = __float2bfloat16(xn * rms * w);
    }
}

// ── Attention softmax decode (f32) — legacy kernels/custom/attention.cuh ─

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
//
// Identical online-softmax arithmetic and warp-merge order; only the K/V
// cache element addressing is token-major (see file header).

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

    // Token-major cache: row stride between consecutive positions of one
    // head is n_kv_heads * HEAD_DIM; the head offset is kv_head * HEAD_DIM.
    const uint32_t kv_stride = n_kv_heads * HEAD_DIM;
    const __nv_bfloat16* k_base = k_cache + kv_head * HEAD_DIM;
    const __nv_bfloat16* v_base = v_cache + kv_head * HEAD_DIM;

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
            float kv = __bfloat162float(k_base[t * kv_stride + i * 32 + lane]);
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
            acc[i] = acc[i] * exp_m + p * __bfloat162float(v_base[t * kv_stride + i * 32 + lane]);
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

// ── C ABI launchers — launch policy identical to the legacy
//    core_kernels_adapter.cu entries they replace ─────────────────────────

extern "C" cudaError_t apxinf_llmdec_silu_f32(
    const void* input, void* output, uint32_t count, void* stream)
{
    dim3 grid((count + LLMDEC_BLOCK_SIZE - 1) / LLMDEC_BLOCK_SIZE, 1, 1);
    dim3 block(LLMDEC_BLOCK_SIZE, 1, 1);
    silu_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)input, (float*)output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_llmdec_silu_bf16(
    const void* input, void* output, uint32_t count, void* stream)
{
    dim3 grid((count + LLMDEC_BLOCK_SIZE - 1) / LLMDEC_BLOCK_SIZE, 1, 1);
    dim3 block(LLMDEC_BLOCK_SIZE, 1, 1);
    silu_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_llmdec_silu_mul_bf16(
    const void* gate_up, void* output, uint32_t inter, void* stream)
{
    dim3 grid((inter + LLMDEC_BLOCK_SIZE - 1) / LLMDEC_BLOCK_SIZE, 1, 1);
    dim3 block(LLMDEC_BLOCK_SIZE, 1, 1);
    silu_mul_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)gate_up, (__nv_bfloat16*)output, inter);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_llmdec_add_f32(
    const void* a, const void* b, void* output, uint32_t count, void* stream)
{
    dim3 grid((count + LLMDEC_BLOCK_SIZE - 1) / LLMDEC_BLOCK_SIZE, 1, 1);
    dim3 block(LLMDEC_BLOCK_SIZE, 1, 1);
    add_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)a, (const float*)b, (float*)output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_llmdec_add_bf16(
    const void* a, const void* b, void* output, uint32_t count, void* stream)
{
    // The legacy launcher's vec8 fast path is intentionally omitted: the
    // decode graph only adds [hidden]-sized vectors, where the scalar kernel
    // is launch-bound either way and the scalar path is the arithmetic the
    // fallback (f32-style) decode compares against.
    dim3 grid((count + LLMDEC_BLOCK_SIZE - 1) / LLMDEC_BLOCK_SIZE, 1, 1);
    dim3 block(LLMDEC_BLOCK_SIZE, 1, 1);
    add_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)a, (const __nv_bfloat16*)b, (__nv_bfloat16*)output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_llmdec_mul_f32(
    const void* a, const void* b, void* output, uint32_t count, void* stream)
{
    dim3 grid((count + LLMDEC_BLOCK_SIZE - 1) / LLMDEC_BLOCK_SIZE, 1, 1);
    dim3 block(LLMDEC_BLOCK_SIZE, 1, 1);
    mul_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)a, (const float*)b, (float*)output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_llmdec_mul_bf16(
    const void* a, const void* b, void* output, uint32_t count, void* stream)
{
    dim3 grid((count + LLMDEC_BLOCK_SIZE - 1) / LLMDEC_BLOCK_SIZE, 1, 1);
    dim3 block(LLMDEC_BLOCK_SIZE, 1, 1);
    mul_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)a, (const __nv_bfloat16*)b, (__nv_bfloat16*)output, count);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_llmdec_embedding_f32(
    const void* table, const void* ids, void* output,
    uint32_t embed_dim, uint32_t seq_len, void* stream)
{
    dim3 grid((embed_dim + LLMDEC_BLOCK_SIZE - 1) / LLMDEC_BLOCK_SIZE, seq_len, 1);
    dim3 block(LLMDEC_BLOCK_SIZE, 1, 1);
    embedding_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)table, (const uint32_t*)ids, (float*)output,
        embed_dim, seq_len);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_llmdec_embedding_bf16(
    const void* table, const void* ids, void* output,
    uint32_t embed_dim, uint32_t seq_len, void* stream)
{
    dim3 grid((embed_dim + LLMDEC_BLOCK_SIZE - 1) / LLMDEC_BLOCK_SIZE, seq_len, 1);
    dim3 block(LLMDEC_BLOCK_SIZE, 1, 1);
    embedding_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)table, (const uint32_t*)ids,
        (__nv_bfloat16*)output, embed_dim, seq_len);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_llmdec_rms_norm_f32(
    const void* input, const void* weight, void* output,
    uint32_t cols, uint32_t rows, float eps, void* stream)
{
    dim3 grid((cols + LLMDEC_BLOCK_SIZE - 1) / LLMDEC_BLOCK_SIZE, rows, 1);
    dim3 block(LLMDEC_BLOCK_SIZE, 1, 1);
    rms_norm_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)input, (const float*)weight, (float*)output,
        cols, rows, eps);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_llmdec_rms_norm_bf16(
    const void* input, const void* weight, void* output,
    uint32_t cols, uint32_t rows, float eps, void* stream)
{
    // One block per row. LLMDEC_BLOCK_SIZE threads (256), strided over cols.
    dim3 grid(rows, 1, 1);
    dim3 block(LLMDEC_BLOCK_SIZE, 1, 1);
    size_t smem = cols * sizeof(float);
    rms_norm_bf16_kernel<<<grid, block, smem, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (const __nv_bfloat16*)weight,
        (__nv_bfloat16*)output, cols, rows, eps);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_llmdec_rms_norm_add_bf16(
    void* x_inout, const void* delta, const void* weight, void* output,
    uint32_t cols, uint32_t rows, float eps, void* stream)
{
    dim3 grid(rows, 1, 1);
    dim3 block(LLMDEC_BLOCK_SIZE, 1, 1);
    size_t smem = cols * sizeof(float);
    rms_norm_add_bf16_kernel<<<grid, block, smem, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)x_inout, (const __nv_bfloat16*)delta,
        (const __nv_bfloat16*)weight, (__nv_bfloat16*)output,
        cols, rows, eps);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_llmdec_attention_softmax_decode_f32(
    const void* scores, void* output,
    uint32_t cols, uint32_t n_heads, const void* pos_ptr, void* stream)
{
    dim3 grid((cols + LLMDEC_BLOCK_SIZE - 1) / LLMDEC_BLOCK_SIZE, n_heads, 1);
    dim3 block(LLMDEC_BLOCK_SIZE, 1, 1);
    attention_softmax_decode_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)scores, (float*)output, cols, n_heads, (const uint32_t*)pos_ptr);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_llmdec_flash_attn_decode_bf16(
    const void* q, const void* k_cache, const void* v_cache, void* out,
    uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
    uint32_t bucket_kv_len, uint32_t max_seq_len,
    float scale, const void* pos_ptr, void* stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid(n_heads, 1, 1);
    dim3 block(LLMDEC_SPLITK_WARPS * 32, 1, 1);
    if (head_dim == 64) {
        flash_attn_decode_bf16_splitk_kernel<64, LLMDEC_SPLITK_WARPS><<<grid, block, 0, s>>>(
            (const __nv_bfloat16*)q, (const __nv_bfloat16*)k_cache,
            (const __nv_bfloat16*)v_cache, (__nv_bfloat16*)out,
            n_heads, n_kv_heads, bucket_kv_len, max_seq_len, scale,
            (const uint32_t*)pos_ptr);
    } else if (head_dim == 128) {
        flash_attn_decode_bf16_splitk_kernel<128, LLMDEC_SPLITK_WARPS><<<grid, block, 0, s>>>(
            (const __nv_bfloat16*)q, (const __nv_bfloat16*)k_cache,
            (const __nv_bfloat16*)v_cache, (__nv_bfloat16*)out,
            n_heads, n_kv_heads, bucket_kv_len, max_seq_len, scale,
            (const uint32_t*)pos_ptr);
    } else {
        return cudaErrorInvalidConfiguration;
    }
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_llmdec_rope_decode_f32(
    const void* input, void* output,
    uint32_t head_dim, uint32_t n_heads,
    float rope_theta, const void* pos_ptr, void* stream)
{
    dim3 grid((head_dim / 2 + LLMDEC_BLOCK_SIZE - 1) / LLMDEC_BLOCK_SIZE, n_heads, 1);
    dim3 block(LLMDEC_BLOCK_SIZE, 1, 1);
    rope_decode_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const float*)input, (float*)output,
        head_dim, n_heads, rope_theta, (const uint32_t*)pos_ptr);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_llmdec_rope_decode_bf16(
    const void* input, void* output,
    uint32_t head_dim, uint32_t n_heads,
    float rope_theta, const void* pos_ptr, void* stream)
{
    dim3 grid((head_dim / 2 + LLMDEC_BLOCK_SIZE - 1) / LLMDEC_BLOCK_SIZE, n_heads, 1);
    dim3 block(LLMDEC_BLOCK_SIZE, 1, 1);
    rope_decode_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output,
        head_dim, n_heads, rope_theta, (const uint32_t*)pos_ptr);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_llmdec_rope_k_write_bf16(
    const void* k_in, void* k_cache,
    uint32_t head_dim, uint32_t n_kv_heads, uint32_t max_seq_len,
    float rope_theta, const void* pos_ptr, void* stream)
{
    dim3 grid((head_dim / 2 + LLMDEC_BLOCK_SIZE - 1) / LLMDEC_BLOCK_SIZE, n_kv_heads, 1);
    dim3 block(LLMDEC_BLOCK_SIZE, 1, 1);
    llmdec_rope_k_write_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)k_in, (__nv_bfloat16*)k_cache,
        head_dim, n_kv_heads, max_seq_len, rope_theta, (const uint32_t*)pos_ptr);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_llmdec_rope_mrope_decode_bf16(
    const void* input, void* output,
    uint32_t head_dim, uint32_t n_heads,
    float theta, const void* pos_ids,
    uint32_t sec_h, uint32_t sec_w, void* stream)
{
    dim3 grid((head_dim / 2 + LLMDEC_BLOCK_SIZE - 1) / LLMDEC_BLOCK_SIZE, n_heads, 1);
    dim3 block(LLMDEC_BLOCK_SIZE, 1, 1);
    rope_mrope_decode_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const __nv_bfloat16*)input, (__nv_bfloat16*)output,
        head_dim, n_heads, theta,
        (const uint32_t*)pos_ids, sec_h, sec_w);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_llmdec_kv_cache_append_decode_f32(
    void* cache, const void* new_data,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_seq_len,
    const void* pos_ptr, void* stream)
{
    dim3 grid((head_dim + LLMDEC_BLOCK_SIZE - 1) / LLMDEC_BLOCK_SIZE, n_kv_heads, 1);
    dim3 block(LLMDEC_BLOCK_SIZE, 1, 1);
    llmdec_kv_cache_append_decode_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (float*)cache, (const float*)new_data,
        n_kv_heads, head_dim, max_seq_len, (const uint32_t*)pos_ptr);
    return cudaGetLastError();
}

extern "C" cudaError_t apxinf_llmdec_kv_cache_append_decode_bf16(
    void* cache, const void* new_data,
    uint32_t n_kv_heads, uint32_t head_dim, uint32_t max_seq_len,
    const void* pos_ptr, void* stream)
{
    dim3 grid((head_dim + LLMDEC_BLOCK_SIZE - 1) / LLMDEC_BLOCK_SIZE, n_kv_heads, 1);
    dim3 block(LLMDEC_BLOCK_SIZE, 1, 1);
    llmdec_kv_cache_append_decode_bf16_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (__nv_bfloat16*)cache, (const __nv_bfloat16*)new_data,
        n_kv_heads, head_dim, max_seq_len, (const uint32_t*)pos_ptr);
    return cudaGetLastError();
}
