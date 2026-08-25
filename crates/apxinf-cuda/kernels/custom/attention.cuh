#pragma once

// Copyright 2026 apxinf contributors.
// Pure CUDA operators grouped by physical operation; launch policy lives under adapters/.

// ── Softmax ───────────────────────────────────────────────────────────────

__global__ void softmax_f32_kernel(
    const float* input, float* output, uint32_t cols, uint32_t rows)
{
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (col >= cols || row >= rows) return;

    uint32_t offset = row * cols;
    float max_val = input[offset];
    for (uint32_t i = 1; i < cols; i++) {
        max_val = fmaxf(max_val, input[offset + i]);
    }
    float sum_exp = 0.0f;
    for (uint32_t i = 0; i < cols; i++) {
        sum_exp += expf(input[offset + i] - max_val);
    }
    output[offset + col] = expf(input[offset + col] - max_val) / sum_exp;
}



// ── Causal Mask ───────────────────────────────────────────────────────────

__global__ void causal_mask_f32_kernel(
    const float* input, float* output,
    uint32_t cols, uint32_t rows, uint32_t kv_offset)
{
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (col >= cols || row >= rows) return;

    uint32_t idx = row * cols + col;
    if (col <= row + kv_offset) {
        output[idx] = input[idx];
    } else {
        output[idx] = -INFINITY;
    }
}



// ── Attention Softmax (fused causal mask + softmax, no sync) ──────────────
//
// Input: scores [rows, cols] where rows=seq_len*n_heads, cols=kv_len
// The causal mask is based on sequence position: row s*stride can attend to
// positions 0..s+kv_offset. The n_heads parameter tells the kernel how
// many consecutive rows share the same sequence position.

__global__ void attention_softmax_f32_kernel(
    const float* scores, float* output,
    uint32_t cols, uint32_t rows, uint32_t kv_offset, uint32_t n_heads)
{
    uint32_t row = blockIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    // Map row index to sequence position: each position has n_heads rows
    uint32_t seq_pos = row / n_heads;
    uint32_t valid_cols = min(seq_pos + kv_offset + 1, cols);

    // Find max over valid positions
    float max_val = -INFINITY;
    for (uint32_t c = 0; c < valid_cols; c++) {
        max_val = fmaxf(max_val, scores[row * cols + c]);
    }

    // Compute exp sum over valid positions
    float sum_exp = 0.0f;
    for (uint32_t c = 0; c < valid_cols; c++) {
        sum_exp += expf(scores[row * cols + c] - max_val);
    }

    // Write output
    if (col < cols) {
        if (col < valid_cols) {
            output[row * cols + col] = expf(scores[row * cols + col] - max_val) / sum_exp;
        } else {
            output[row * cols + col] = 0.0f;
        }
    }
}



// Fused causal mask + softmax for decode (rows = n_heads, seq_len=1).
// valid_cols = min(*pos_ptr + 1, cols). Padded columns (beyond pos+1) -> 0.
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



// ── Softmax (bf16) ────────────────────────────────────────────────────────

__global__ void softmax_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output, uint32_t cols, uint32_t rows)
{
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (col >= cols || row >= rows) return;

    uint32_t offset = row * cols;
    float max_val = __bfloat162float(input[offset]);
    for (uint32_t i = 1; i < cols; i++) {
        max_val = fmaxf(max_val, __bfloat162float(input[offset + i]));
    }
    float sum_exp = 0.0f;
    for (uint32_t i = 0; i < cols; i++) {
        sum_exp += expf(__bfloat162float(input[offset + i]) - max_val);
    }
    float x = __bfloat162float(input[offset + col]);
    output[offset + col] = __float2bfloat16(expf(x - max_val) / sum_exp);
}



// ── Causal Mask (bf16) — writes bf16(-INFINITY) for masked cells ──────────

__global__ void causal_mask_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    uint32_t cols, uint32_t rows, uint32_t kv_offset)
{
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (col >= cols || row >= rows) return;

    uint32_t idx = row * cols + col;
    if (col <= row + kv_offset) {
        output[idx] = input[idx];
    } else {
        output[idx] = __float2bfloat16(-INFINITY);
    }
}



// ── Attention Softmax (bf16, fused causal mask + softmax) ─────────────────

__global__ void attention_softmax_bf16_kernel(
    const __nv_bfloat16* scores, __nv_bfloat16* output,
    uint32_t cols, uint32_t rows, uint32_t kv_offset, uint32_t n_heads)
{
    uint32_t row = blockIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    uint32_t seq_pos = row / n_heads;
    uint32_t valid_cols = min(seq_pos + kv_offset + 1, cols);

    float max_val = -INFINITY;
    for (uint32_t c = 0; c < valid_cols; c++) {
        max_val = fmaxf(max_val, __bfloat162float(scores[row * cols + c]));
    }
    float sum_exp = 0.0f;
    for (uint32_t c = 0; c < valid_cols; c++) {
        sum_exp += expf(__bfloat162float(scores[row * cols + c]) - max_val);
    }
    if (col < cols) {
        if (col < valid_cols) {
            float x = __bfloat162float(scores[row * cols + col]);
            output[row * cols + col] = __float2bfloat16(expf(x - max_val) / sum_exp);
        } else {
            output[row * cols + col] = __float2bfloat16(0.0f);
        }
    }
}



__global__ void attention_softmax_decode_bf16_kernel(
    const __nv_bfloat16* scores, __nv_bfloat16* output,
    uint32_t cols, uint32_t n_heads, const uint32_t* pos_ptr)
{
    uint32_t row = blockIdx.y;
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_heads) return;

    uint32_t valid_cols = min(*pos_ptr + 1, cols);
    float max_val = -INFINITY;
    for (uint32_t c = 0; c < valid_cols; c++) {
        max_val = fmaxf(max_val, __bfloat162float(scores[row * cols + c]));
    }
    float sum_exp = 0.0f;
    for (uint32_t c = 0; c < valid_cols; c++) {
        sum_exp += expf(__bfloat162float(scores[row * cols + c]) - max_val);
    }
    if (col < cols) {
        if (col < valid_cols) {
            float x = __bfloat162float(scores[row * cols + col]);
            output[row * cols + col] = __float2bfloat16(expf(x - max_val) / sum_exp);
        } else {
            output[row * cols + col] = __float2bfloat16(0.0f);
        }
    }
}



// ── Vision SDPA (bf16) — non-causal full attention for Qwen3-VL ViT ──────
//
// Q, K, V: [seq_len, n_heads, head_dim] bf16 (contiguous, row-major)
// Output:  [seq_len, n_heads * head_dim] bf16
//
// Non-causal: every query attends to every key. One block per (head, query).
// 32 threads (= 1 warp); each thread handles strided head_dim elements. The
// current contract covers head_dim 64 and 72, so at most three values live in
// registers per thread and the dot-product reduction still uses __shfl.
//
// IMPORTANT: all 32 threads must reach every __shfl_xor_sync call (full mask
// 0xffffffff). The inner loops are therefore non-strided — every thread
// iterates every ki so the warp stays converged. (A strided `ki += 32` loop
// would deadlock when seq_len < 32 because some threads would exit early.)
//
// Shared mem: (seq_len + 1) floats for scores + max/sum scratch.

__global__ void vision_sdpa_bf16_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* out,
    uint32_t seq_len, uint32_t n_heads, uint32_t head_dim, float scale)
{
    uint32_t head = blockIdx.y;
    uint32_t qi   = blockIdx.x;
    if (qi >= seq_len) return;
    int tid = threadIdx.x;       // 0..31
    constexpr int kMaxElementsPerThread = 3;
    int dimensions[kMaxElementsPerThread];
    float query_values[kMaxElementsPerThread];
    int dimension_count = 0;

    extern __shared__ float smem[];
    float* scores = smem;        // [seq_len] + 1 scratch slot

    const __nv_bfloat16* q_row = q  + qi * n_heads * head_dim + head * head_dim;
    for (uint32_t dimension = tid; dimension < head_dim; dimension += 32u) {
        dimensions[dimension_count] = static_cast<int>(dimension);
        query_values[dimension_count] = __bfloat162float(q_row[dimension]);
        dimension_count++;
    }

    // Phase 1: scores[ki] = (Q[qi] · K[ki]) * scale. All threads iterate
    // every ki so the shfl reduction stays converged.
    for (uint32_t ki = 0; ki < seq_len; ki++) {
        const __nv_bfloat16* k_row = k + ki * n_heads * head_dim + head * head_dim;
        float dot = 0.0f;
        for (int slot = 0; slot < dimension_count; slot++) {
            dot += query_values[slot] *
                __bfloat162float(k_row[dimensions[slot]]);
        }
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
        if (tid == 0) scores[ki] = dot * scale;
    }
    __syncthreads();

    // Phase 2: softmax (max → exp → sum → normalize).
    // For seq_len > 32, the max/sum reductions are strided — but the shfl
    // only needs the threads that have data. Use mask = __activemask() to
    // avoid deadlocks when some threads drop out.
    float max_val = -INFINITY;
    for (uint32_t ki = tid; ki < seq_len; ki += 32u)
        max_val = fmaxf(max_val, scores[ki]);
    unsigned mask = __activemask();
    for (int off = 16; off > 0; off >>= 1)
        max_val = fmaxf(max_val, __shfl_xor_sync(mask, max_val, off));
    if (tid == 0) scores[seq_len] = max_val;
    __syncthreads();
    max_val = scores[seq_len];

    float sum = 0.0f;
    for (uint32_t ki = tid; ki < seq_len; ki += 32u) {
        float e = expf(scores[ki] - max_val);
        scores[ki] = e;
        sum += e;
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(mask, sum, off);
    if (tid == 0) scores[seq_len] = sum;
    __syncthreads();
    float inv_sum = 1.0f / scores[seq_len];
    for (uint32_t ki = tid; ki < seq_len; ki += 32u) scores[ki] *= inv_sum;
    __syncthreads();

    // Phase 3: out[qi, head, d] = sum_k scores[k] * V[k, head, d].
    // All threads iterate every ki; owned dimensions differ per thread.
    float accumulators[kMaxElementsPerThread] = {0.0f, 0.0f, 0.0f};
    for (uint32_t ki = 0; ki < seq_len; ki++) {
        float s = scores[ki];
        const __nv_bfloat16* v_row = v + ki * n_heads * head_dim + head * head_dim;
        for (int slot = 0; slot < dimension_count; slot++) {
            accumulators[slot] +=
                s * __bfloat162float(v_row[dimensions[slot]]);
        }
    }
    __nv_bfloat16* out_row = out + qi * n_heads * head_dim + head * head_dim;
    for (int slot = 0; slot < dimension_count; slot++) {
        out_row[dimensions[slot]] = __float2bfloat16(accumulators[slot]);
    }
}



// ── Flash Attention decode (bf16) — single-kernel online-softmax ────────
//
// Replaces the 17-kernel attention path (8 QK^T GEMMs + softmax + 8 AV
// GEMMs per layer for GQA 4:1) with one kernel per layer. One block per
// Q head; 32 threads (one warp); each thread holds HEAD_DIM/32 elements.
//
// Online softmax: streams K/V in sequence order, maintains running max +
// sum + output accumulator. Never materializes the full scores matrix in
// HBM. For decode (M=1 Q), this is optimal — one pass over K and V.
//
// Graph-capture friendly: loops over `bucket_kv_len` (static per bucket),
// reads `pos` from `pos_ptr` to compute `valid_len = pos + 1`. Positions
// >= valid_len are masked (score = -inf → exp = 0, no contribution).

template<int HEAD_DIM>
__global__ void flash_attn_decode_bf16_kernel(
    const __nv_bfloat16* q,        // [n_heads, HEAD_DIM]
    const __nv_bfloat16* k_cache,  // [n_kv_heads, max_seq_len, HEAD_DIM]
    const __nv_bfloat16* v_cache,  // [n_kv_heads, max_seq_len, HEAD_DIM]
    __nv_bfloat16* out,            // [n_heads, HEAD_DIM]
    uint32_t n_heads, uint32_t n_kv_heads,
    uint32_t bucket_kv_len, uint32_t max_seq_len,
    float scale, const uint32_t* pos_ptr)
{
    constexpr int ELEMS_PER_THREAD = HEAD_DIM / 32;
    uint32_t q_head = blockIdx.x;
    uint32_t gqa_ratio = n_heads / n_kv_heads;
    uint32_t kv_head = q_head / gqa_ratio;
    int tid = threadIdx.x;  // 0..31

    uint32_t pos = *pos_ptr;
    uint32_t valid_len = pos + 1;
    if (valid_len > bucket_kv_len) valid_len = bucket_kv_len;

    // Load Q into registers.
    float q_reg[ELEMS_PER_THREAD];
    const __nv_bfloat16* q_row = q + q_head * HEAD_DIM;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++)
        q_reg[i] = __bfloat162float(q_row[i * 32 + tid]);

    // Online softmax state.
    float m = -INFINITY;
    float l = 0.0f;
    float acc[ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) acc[i] = 0.0f;

    const __nv_bfloat16* k_base = k_cache + kv_head * max_seq_len * HEAD_DIM;
    const __nv_bfloat16* v_base = v_cache + kv_head * max_seq_len * HEAD_DIM;

    for (uint32_t t = 0; t < bucket_kv_len; t++) {
        // Dot product Q · K[t] (warp-reduced).
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++) {
            float kv = __bfloat162float(k_base[t * HEAD_DIM + i * 32 + tid]);
            dot += q_reg[i] * kv;
        }
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, off);
        dot *= scale;

        // Mask invalid positions (t >= valid_len).
        if (t >= valid_len) dot = -INFINITY;

        // Online softmax update.
        float m_new = fmaxf(m, dot);
        float p = (t < valid_len) ? expf(dot - m_new) : 0.0f;
        float exp_m = expf(m - m_new);
        l = l * exp_m + p;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; i++)
            acc[i] = acc[i] * exp_m + p * __bfloat162float(v_base[t * HEAD_DIM + i * 32 + tid]);
        m = m_new;
    }

    // Write output: out = acc / l.
    __nv_bfloat16* out_row = out + q_head * HEAD_DIM;
    float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++)
        out_row[i * 32 + tid] = __float2bfloat16(acc[i] * inv_l);
}

// ── Flash-decoding (split-K) variant ───────────────────────────────────────
//
// The single-warp variant above leaves the SM starved: one in-flight warp
// can't hide HBM/L2 load latency, so each block stalls between dependent
// K/V loads. This version keeps "one block per Q head" (so the block count
// still covers the heads) but runs SPLITK_WARPS warps per block. Each warp
// handles a strided subset of the timesteps and maintains its own online-
// softmax (m, l, acc) state; the warps then merge their states via shared
// memory. Total K/V traffic is unchanged (each timestep read once across
// the warps), but occupancy rises ~SPLITK_WARPS×, which is what Thor's
// 14-SM GPU needs to hit bandwidth.
#ifndef SPLITK_WARPS
#define SPLITK_WARPS 16
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

    const __nv_bfloat16* k_base = k_cache + kv_head * max_seq_len * HEAD_DIM;
    const __nv_bfloat16* v_base = v_cache + kv_head * max_seq_len * HEAD_DIM;

    // Each warp's private online-softmax over its strided timesteps.
    float m = -INFINITY;
    float l = 0.0f;
    float acc[ELEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) acc[i] = 0.0f;

    // Strided timestep assignment: warp w handles t where t % WARPS == w.
    // Loop to `valid_len` (read from pos_ptr) — data-dependent but fine
    // inside a captured kernel, and avoids reading/masking the padded tail
    // when the sequence is shorter than the bucket (the common case early
    // in generation). bucket_kv_len is just an upper bound now.
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





// One warp per row, matching FlashRT's Apache-2.0 FP16 softmax. The even
// path packs two values per lane; the scalar path keeps arbitrary prompt
// lengths correct without relying on half2 alignment between odd rows.
constexpr int kSoftmaxMaxCols = 1024;
constexpr int kSoftmaxIterations = kSoftmaxMaxCols / 32;

__global__ void softmax_even_f16_kernel(half* data, int rows, int cols) {
  int lane = threadIdx.x;
  int row = blockIdx.x;
  if (row >= rows) return;
  half* source = data + static_cast<int64_t>(row) * cols;
  half2* source2 = reinterpret_cast<half2*>(source);
  int cols2 = cols / 2;
  float values[kSoftmaxIterations];
  float maximum = -1.0e30f;
#pragma unroll
  for (int iteration = 0; iteration < kSoftmaxIterations / 2; ++iteration) {
    int col2 = iteration * 32 + lane;
    if (col2 < cols2) {
      half2 packed = source2[col2];
      values[2 * iteration] = __half2float(packed.x);
      values[2 * iteration + 1] = __half2float(packed.y);
      maximum = fmaxf(maximum, fmaxf(values[2 * iteration],
                                     values[2 * iteration + 1]));
    } else {
      values[2 * iteration] = -1.0e30f;
      values[2 * iteration + 1] = -1.0e30f;
    }
  }
  maximum = warp_max(maximum);
  float sum = 0.0f;
#pragma unroll
  for (int iteration = 0; iteration < kSoftmaxIterations; ++iteration) {
    values[iteration] = __expf(values[iteration] - maximum);
    sum += values[iteration];
  }
  sum = warp_sum_all(sum);
  float inverse = 1.0f / sum;
#pragma unroll
  for (int iteration = 0; iteration < kSoftmaxIterations / 2; ++iteration) {
    int col2 = iteration * 32 + lane;
    if (col2 < cols2) {
      source2[col2] = __floats2half2_rn(values[2 * iteration] * inverse,
                                        values[2 * iteration + 1] * inverse);
    }
  }
}

__global__ void softmax_scalar_f16_kernel(half* data, int rows, int cols) {
  int lane = threadIdx.x;
  int row = blockIdx.x;
  if (row >= rows) return;
  half* source = data + static_cast<int64_t>(row) * cols;
  float values[kSoftmaxIterations];
  float maximum = -1.0e30f;
#pragma unroll
  for (int iteration = 0; iteration < kSoftmaxIterations; ++iteration) {
    int col = iteration * 32 + lane;
    float value = col < cols ? __half2float(source[col]) : -1.0e30f;
    values[iteration] = value;
    maximum = fmaxf(maximum, value);
  }
  maximum = warp_max(maximum);
  float sum = 0.0f;
#pragma unroll
  for (int iteration = 0; iteration < kSoftmaxIterations; ++iteration) {
    values[iteration] = __expf(values[iteration] - maximum);
    sum += values[iteration];
  }
  sum = warp_sum_all(sum);
  float inverse = 1.0f / sum;
#pragma unroll
  for (int iteration = 0; iteration < kSoftmaxIterations; ++iteration) {
    int col = iteration * 32 + lane;
    if (col < cols) source[col] = __float2half(values[iteration] * inverse);
  }
}

// BF16 counterpart used by the Thor static-inference MQA path. One warp owns
// a row, so each score is loaded once and all reductions stay warp-local.
__global__ void softmax_scalar_bf16_kernel(
    __nv_bfloat16* data, int rows, int cols) {
  int lane = threadIdx.x;
  int row = blockIdx.x;
  if (row >= rows) return;
  __nv_bfloat16* source = data + static_cast<int64_t>(row) * cols;
  float values[kSoftmaxIterations];
  float maximum = -1.0e30f;
#pragma unroll
  for (int iteration = 0; iteration < kSoftmaxIterations; ++iteration) {
    int col = iteration * 32 + lane;
    float value =
        col < cols ? __bfloat162float(source[col]) : -1.0e30f;
    values[iteration] = value;
    maximum = fmaxf(maximum, value);
  }
  maximum = warp_max(maximum);
  float sum = 0.0f;
#pragma unroll
  for (int iteration = 0; iteration < kSoftmaxIterations; ++iteration) {
    values[iteration] = __expf(values[iteration] - maximum);
    sum += values[iteration];
  }
  sum = warp_sum_all(sum);
  float inverse = 1.0f / sum;
#pragma unroll
  for (int iteration = 0; iteration < kSoftmaxIterations; ++iteration) {
    int col = iteration * 32 + lane;
    if (col < cols) {
      source[col] = __float2bfloat16(values[iteration] * inverse);
    }
  }
}

// Batch-1 MQA flash kernel for static inference's one-KV-head Gemma experts. Scores
// remain in shared memory; only the final [suffix, heads, dim] tensor is
// written to global memory.
__global__ void mqa_flash_f16_kernel(
    const half* q, const half* prefix_k, const half* prefix_v,
    const half* suffix_k, const half* suffix_v, half* output,
    int suffix_tokens, int heads, int head_dim, int prefix_tokens) {
  extern __shared__ float shared[];
  float* scores = shared;
  float* warp_sums = scores + prefix_tokens + suffix_tokens;
  const int query = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warps = blockDim.x >> 5;
  const half* query_ptr = q + (query * heads + head) * head_dim;
  const int total_tokens = prefix_tokens + suffix_tokens;
  const float scale = rsqrtf(static_cast<float>(head_dim));

  for (int token = 0; token < total_tokens; ++token) {
    const half* key = token < prefix_tokens
        ? prefix_k + token * head_dim
        : suffix_k + (token - prefix_tokens) * head_dim;
    float dot = tid < head_dim
        ? __half2float(query_ptr[tid]) * __half2float(key[tid])
        : 0.0f;
    dot = warp_sum(dot);
    if (lane == 0) warp_sums[warp] = dot;
    __syncthreads();
    if (warp == 0) {
      float block_sum = lane < warps ? warp_sums[lane] : 0.0f;
      block_sum = warp_sum(block_sum);
      if (lane == 0) scores[token] = block_sum * scale;
    }
    __syncthreads();
  }

  if (tid == 0) {
    float maximum = -3.402823466e+38F;
    for (int token = 0; token < total_tokens; ++token)
      maximum = fmaxf(maximum, scores[token]);
    float denominator = 0.0f;
    for (int token = 0; token < total_tokens; ++token) {
      scores[token] = expf(scores[token] - maximum);
      denominator += scores[token];
    }
    float inverse = 1.0f / denominator;
    for (int token = 0; token < total_tokens; ++token)
      scores[token] *= inverse;
  }
  __syncthreads();

  if (tid < head_dim) {
    float accumulator = 0.0f;
    for (int token = 0; token < total_tokens; ++token) {
      const half* value = token < prefix_tokens
          ? prefix_v + token * head_dim
          : suffix_v + (token - prefix_tokens) * head_dim;
      accumulator += scores[token] * __half2float(value[tid]);
    }
    output[(query * heads + head) * head_dim + tid] = __float2half(accumulator);
  }
}

// Non-causal multi-head flash-style attention for SigLIP. Each block owns
// one query/head pair and retains its 256 scores in shared memory.
__global__ void mha_flash_f16_kernel(
    const half* q, const half* k, const half* v, half* output,
    int tokens_per_batch, int heads, int head_dim) {
  extern __shared__ float shared[];
  float* scores = shared;
  float* warp_sums = scores + tokens_per_batch;
  const int query = blockIdx.x;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const int batch_token_offset = batch * tokens_per_batch;
  const int global_query = batch_token_offset + query;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warps = blockDim.x >> 5;
  const half* query_ptr = q + (global_query * heads + head) * head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));

  for (int token = 0; token < tokens_per_batch; ++token) {
    const half* key = k + ((batch_token_offset + token) * heads + head) * head_dim;
    float dot = tid < head_dim
        ? __half2float(query_ptr[tid]) * __half2float(key[tid])
        : 0.0f;
    dot = warp_sum(dot);
    if (lane == 0) warp_sums[warp] = dot;
    __syncthreads();
    if (warp == 0) {
      float total = lane < warps ? warp_sums[lane] : 0.0f;
      total = warp_sum(total);
      if (lane == 0) scores[token] = total * scale;
    }
    __syncthreads();
  }
  if (tid == 0) {
    float maximum = -3.402823466e+38F;
    for (int token = 0; token < tokens_per_batch; ++token) maximum = fmaxf(maximum, scores[token]);
    float denominator = 0.0f;
    for (int token = 0; token < tokens_per_batch; ++token) {
      scores[token] = expf(scores[token] - maximum);
      denominator += scores[token];
    }
    for (int token = 0; token < tokens_per_batch; ++token) scores[token] /= denominator;
  }
  __syncthreads();
  if (tid < head_dim) {
    float accumulator = 0.0f;
    for (int token = 0; token < tokens_per_batch; ++token) {
      const half* value = v + ((batch_token_offset + token) * heads + head) * head_dim;
      accumulator += scores[token] * __half2float(value[tid]);
    }
    output[(global_query * heads + head) * head_dim + tid] = __float2half(accumulator);
  }
}


__global__ void mqa_bf16_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, __nv_bfloat16* output,
    int query_tokens, int key_tokens, int heads, int head_dim) {
  extern __shared__ float shared[];
  float* scores = shared;
  float* warp_sums = scores + key_tokens;
  const int query = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warps = blockDim.x >> 5;
  const __nv_bfloat16* query_ptr = q + (query * heads + head) * head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  for (int token = 0; token < key_tokens; ++token) {
    const __nv_bfloat16* key = k + static_cast<int64_t>(token) * head_dim;
    float dot = tid < head_dim
        ? __bfloat162float(query_ptr[tid]) * __bfloat162float(key[tid])
        : 0.0f;
    dot = warp_sum(dot);
    if (lane == 0) warp_sums[warp] = dot;
    __syncthreads();
    if (warp == 0) {
      float total = lane < warps ? warp_sums[lane] : 0.0f;
      total = warp_sum(total);
      if (lane == 0) scores[token] = total * scale;
    }
    __syncthreads();
  }
  if (tid == 0) {
    float maximum = -3.402823466e+38F;
    for (int token = 0; token < key_tokens; ++token)
      maximum = fmaxf(maximum, scores[token]);
    float denominator = 0.0f;
    for (int token = 0; token < key_tokens; ++token) {
      scores[token] = expf(scores[token] - maximum);
      denominator += scores[token];
    }
    for (int token = 0; token < key_tokens; ++token)
      scores[token] /= denominator;
  }
  __syncthreads();
  if (tid < head_dim) {
    float accumulator = 0.0f;
    for (int token = 0; token < key_tokens; ++token)
      accumulator += scores[token] *
          __bfloat162float(v[static_cast<int64_t>(token) * head_dim + tid]);
    output[(query * heads + head) * head_dim + tid] =
        __float2bfloat16(accumulator);
  }
}

__global__ void mha_bf16_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, __nv_bfloat16* output,
    int tokens_per_batch, int heads, int head_dim) {
  extern __shared__ float shared[];
  float* scores = shared;
  float* warp_sums = scores + tokens_per_batch;
  const int query = blockIdx.x;
  const int head = blockIdx.y;
  const int batch = blockIdx.z;
  const int batch_token_offset = batch * tokens_per_batch;
  const int global_query = batch_token_offset + query;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warps = blockDim.x >> 5;
  const __nv_bfloat16* query_ptr =
      q + (global_query * heads + head) * head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  for (int token = 0; token < tokens_per_batch; ++token) {
    const __nv_bfloat16* key =
        k + ((batch_token_offset + token) * heads + head) * head_dim;
    float dot = tid < head_dim
        ? __bfloat162float(query_ptr[tid]) * __bfloat162float(key[tid])
        : 0.0f;
    dot = warp_sum(dot);
    if (lane == 0) warp_sums[warp] = dot;
    __syncthreads();
    if (warp == 0) {
      float total = lane < warps ? warp_sums[lane] : 0.0f;
      total = warp_sum(total);
      if (lane == 0) scores[token] = total * scale;
    }
    __syncthreads();
  }
  if (tid == 0) {
    float maximum = -3.402823466e+38F;
    for (int token = 0; token < tokens_per_batch; ++token)
      maximum = fmaxf(maximum, scores[token]);
    float denominator = 0.0f;
    for (int token = 0; token < tokens_per_batch; ++token) {
      scores[token] = expf(scores[token] - maximum);
      denominator += scores[token];
    }
    for (int token = 0; token < tokens_per_batch; ++token)
      scores[token] /= denominator;
  }
  __syncthreads();
  if (tid < head_dim) {
    float accumulator = 0.0f;
    for (int token = 0; token < tokens_per_batch; ++token) {
      const __nv_bfloat16* value =
          v + ((batch_token_offset + token) * heads + head) * head_dim;
      accumulator += scores[token] * __bfloat162float(value[tid]);
    }
    output[(global_query * heads + head) * head_dim + tid] =
        __float2bfloat16(accumulator);
  }
}


