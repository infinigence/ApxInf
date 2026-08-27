#pragma once

// Copyright 2026 apxinf contributors.
// Pure CUDA operators grouped by physical operation; launch policy lives under adapters/.

__device__ __forceinline__ uint32_t apxinf_row_from_grid_yz()
{
    return static_cast<uint32_t>(blockIdx.y)
        + static_cast<uint32_t>(blockIdx.z) * static_cast<uint32_t>(gridDim.y);
}

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



// ── Attention Softmax (fused causal mask + one CTA per row) ────────────────
//
// Input: scores [rows, cols] where rows=seq_len*n_heads, cols=kv_len
// The causal mask is based on sequence position: row s*stride can attend to
// positions 0..s+kv_offset. The n_heads parameter tells the kernel how
// many consecutive rows share the same sequence position.

__global__ void attention_softmax_f32_kernel(
    const float* scores, float* output,
    uint32_t cols, uint32_t rows, uint32_t kv_offset, uint32_t n_heads)
{
    uint32_t row = apxinf_row_from_grid_yz();
    if (row >= rows) return;

    uint32_t seq_pos = row / n_heads;
    uint32_t valid_cols = min(seq_pos + kv_offset + 1, cols);
    uint32_t lane = threadIdx.x;
    __shared__ float reduction[2];
    if (lane == 0) {
        float max_val = -INFINITY;
        for (uint32_t c = 0; c < valid_cols; c++) {
            max_val = fmaxf(max_val, scores[row * cols + c]);
        }
        float sum_exp = 0.0f;
        for (uint32_t c = 0; c < valid_cols; c++) {
            sum_exp += expf(scores[row * cols + c] - max_val);
        }
        reduction[0] = max_val;
        reduction[1] = sum_exp;
    }
    __syncthreads();
    float max_val = reduction[0];
    float sum_exp = reduction[1];

    for (uint32_t c = lane; c < cols; c += blockDim.x) {
        output[row * cols + c] = c < valid_cols
            ? expf(scores[row * cols + c] - max_val) / sum_exp
            : 0.0f;
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

__device__ __forceinline__ float attention_scaled_bf16(
    __nv_bfloat16 score, float scale)
{
    return __bfloat162float(__float2bfloat16(__bfloat162float(score) * scale));
}

__device__ __forceinline__ uint32_t attention_sequence_position(
    uint32_t row, uint32_t rows, uint32_t n_heads, uint32_t packed_gqa_ratio)
{
    if (packed_gqa_ratio == 0) return row / n_heads;
    uint32_t kv_heads = n_heads / packed_gqa_ratio;
    uint32_t rows_per_kv_head = rows / kv_heads;
    return (row % rows_per_kv_head) / packed_gqa_ratio;
}

__global__ void attention_softmax_bf16_kernel(
    const __nv_bfloat16* scores, __nv_bfloat16* output,
    uint32_t cols, uint32_t rows, uint32_t kv_offset, uint32_t n_heads,
    float score_scale, uint32_t packed_gqa_ratio)
{
    uint32_t row = apxinf_row_from_grid_yz();
    if (row >= rows) return;

    uint32_t seq_pos = attention_sequence_position(
        row, rows, n_heads, packed_gqa_ratio);
    uint32_t valid_cols = min(seq_pos + kv_offset + 1, cols);
    uint32_t lane = threadIdx.x;
    __shared__ float max_values[256];
    float local_max = -INFINITY;
    for (uint32_t c = lane; c < valid_cols; c += blockDim.x) {
        local_max = fmaxf(
            local_max, attention_scaled_bf16(scores[row * cols + c], score_scale));
    }
    max_values[lane] = local_max;
    __syncthreads();
    for (uint32_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            max_values[lane] = fmaxf(max_values[lane], max_values[lane + offset]);
        }
        __syncthreads();
    }
    float max_val = max_values[0];
    __syncthreads();

    __shared__ float sum_shared;
    float sum_exp = 0.0f;
    for (uint32_t base = 0; base < valid_cols; base += blockDim.x) {
        uint32_t c = base + lane;
        max_values[lane] = c < valid_cols
            ? expf(attention_scaled_bf16(scores[row * cols + c], score_scale) - max_val)
            : 0.0f;
        __syncthreads();
        if (lane == 0) {
            uint32_t count = min(blockDim.x, valid_cols - base);
            for (uint32_t index = 0; index < count; index++) {
                sum_exp += max_values[index];
            }
        }
        __syncthreads();
    }
    if (lane == 0) {
        sum_shared = sum_exp;
    }
    __syncthreads();
    sum_exp = sum_shared;

    for (uint32_t c = lane; c < cols; c += blockDim.x) {
        if (c < valid_cols) {
            float x = attention_scaled_bf16(scores[row * cols + c], score_scale);
            output[row * cols + c] = __float2bfloat16(expf(x - max_val) / sum_exp);
        } else {
            output[row * cols + c] = __float2bfloat16(0.0f);
        }
    }
}

// Exact long-prefill softmax for a single-consumer score buffer. Each score
// is scaled and rounded to BF16 during the existing maximum scan, then the
// same storage is normalized in place. Later phases therefore avoid two
// repeated scale-and-round operations without changing the BF16 values used.
__global__ void attention_softmax_bf16_scale_in_place_kernel(
    __nv_bfloat16* scores_output,
    uint32_t cols, uint32_t rows, uint32_t kv_offset, uint32_t n_heads,
    float score_scale, uint32_t packed_gqa_ratio)
{
    uint32_t row = apxinf_row_from_grid_yz();
    if (row >= rows) return;

    uint32_t seq_pos = attention_sequence_position(
        row, rows, n_heads, packed_gqa_ratio);
    uint32_t valid_cols = min(seq_pos + kv_offset + 1, cols);
    uint32_t lane = threadIdx.x;
    __nv_bfloat16* row_scores = scores_output + row * cols;
    __shared__ float max_values[256];
    float local_max = -INFINITY;
    for (uint32_t c = lane; c < valid_cols; c += blockDim.x) {
        __nv_bfloat16 scaled = __float2bfloat16(
            __bfloat162float(row_scores[c]) * score_scale);
        row_scores[c] = scaled;
        local_max = fmaxf(local_max, __bfloat162float(scaled));
    }
    max_values[lane] = local_max;
    __syncthreads();
    for (uint32_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            max_values[lane] = fmaxf(max_values[lane], max_values[lane + offset]);
        }
        __syncthreads();
    }
    float max_val = max_values[0];
    __syncthreads();

    __shared__ float sum_shared;
    float sum_exp = 0.0f;
    for (uint32_t base = 0; base < valid_cols; base += blockDim.x) {
        uint32_t c = base + lane;
        max_values[lane] = c < valid_cols
            ? expf(__bfloat162float(row_scores[c]) - max_val)
            : 0.0f;
        __syncthreads();
        if (lane == 0) {
            uint32_t count = min(blockDim.x, valid_cols - base);
            for (uint32_t index = 0; index < count; index++) {
                sum_exp += max_values[index];
            }
        }
        __syncthreads();
    }
    if (lane == 0) {
        sum_shared = sum_exp;
    }
    __syncthreads();
    sum_exp = sum_shared;

    for (uint32_t c = lane; c < cols; c += blockDim.x) {
        scores_output[row * cols + c] = c < valid_cols
            ? __float2bfloat16(
                  expf(__bfloat162float(row_scores[c]) - max_val) / sum_exp)
            : __float2bfloat16(0.0f);
    }
}


// Preserve the scalar maximum and summation order while caching each FP32
// exponential once. Dynamic shared memory stores `cols` numerators.
__global__ void attention_softmax_bf16_exp_cache_kernel(
    const __nv_bfloat16* scores, __nv_bfloat16* output,
    uint32_t cols, uint32_t rows, uint32_t kv_offset, uint32_t n_heads)
{
    uint32_t row = apxinf_row_from_grid_yz();
    if (row >= rows) return;

    uint32_t seq_pos = row / n_heads;
    uint32_t valid_cols = min(seq_pos + kv_offset + 1, cols);
    uint32_t lane = threadIdx.x;
    extern __shared__ float numerators[];
    __shared__ float max_values[256];
    __shared__ float sum_shared;

    float local_max = -INFINITY;
    for (uint32_t c = lane; c < valid_cols; c += blockDim.x) {
        local_max = fmaxf(local_max, __bfloat162float(scores[row * cols + c]));
    }
    max_values[lane] = local_max;
    __syncthreads();
    for (uint32_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            max_values[lane] = fmaxf(max_values[lane], max_values[lane + offset]);
        }
        __syncthreads();
    }
    float max_val = max_values[0];

    for (uint32_t c = lane; c < valid_cols; c += blockDim.x) {
        numerators[c] = expf(__bfloat162float(scores[row * cols + c]) - max_val);
    }
    __syncthreads();

    if (lane == 0) {
        float sum_exp = 0.0f;
        for (uint32_t c = 0; c < valid_cols; c++) {
            sum_exp += numerators[c];
        }
        sum_shared = sum_exp;
    }
    __syncthreads();
    float sum_exp = sum_shared;

    for (uint32_t c = lane; c < cols; c += blockDim.x) {
        output[row * cols + c] = c < valid_cols
            ? __float2bfloat16(numerators[c] / sum_exp)
            : __float2bfloat16(0.0f);
    }
}

// Fuse the single-consumer BF16 score scale into the exact numerator-cache
// softmax. Each scale is rounded to BF16 before max/exp, matching the separate
// scale kernel, while the scalar numerator summation and final division order
// remain unchanged.
__global__ void attention_softmax_bf16_scaled_exp_cache_kernel(
    const __nv_bfloat16* scores, __nv_bfloat16* output,
    uint32_t cols, uint32_t rows, uint32_t kv_offset, uint32_t n_heads,
    float score_scale)
{
    uint32_t row = apxinf_row_from_grid_yz();
    if (row >= rows) return;

    uint32_t seq_pos = row / n_heads;
    uint32_t valid_cols = min(seq_pos + kv_offset + 1, cols);
    uint32_t lane = threadIdx.x;
    extern __shared__ float numerators[];
    __shared__ float max_values[256];
    __shared__ float sum_shared;

    float local_max = -INFINITY;
    for (uint32_t c = lane; c < valid_cols; c += blockDim.x) {
        local_max = fmaxf(
            local_max, attention_scaled_bf16(scores[row * cols + c], score_scale));
    }
    max_values[lane] = local_max;
    __syncthreads();
    for (uint32_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            max_values[lane] = fmaxf(max_values[lane], max_values[lane + offset]);
        }
        __syncthreads();
    }
    float max_val = max_values[0];

    for (uint32_t c = lane; c < valid_cols; c += blockDim.x) {
        float scaled = attention_scaled_bf16(scores[row * cols + c], score_scale);
        numerators[c] = expf(scaled - max_val);
    }
    __syncthreads();

    if (lane == 0) {
        float sum_exp = 0.0f;
        for (uint32_t c = 0; c < valid_cols; c++) {
            sum_exp += numerators[c];
        }
        sum_shared = sum_exp;
    }
    __syncthreads();
    float sum_exp = sum_shared;

    for (uint32_t c = lane; c < cols; c += blockDim.x) {
        output[row * cols + c] = c < valid_cols
            ? __float2bfloat16(numerators[c] / sum_exp)
            : __float2bfloat16(0.0f);
    }
}

// Decode-only exact numerator cache backed by global memory. This preserves
// the scalar max and summation order beyond the per-CTA shared-memory limit.
__global__ void attention_softmax_bf16_exp_global_cache_kernel(
    const __nv_bfloat16* scores, __nv_bfloat16* output, float* numerators,
    uint32_t cols, uint32_t rows, uint32_t kv_offset, uint32_t n_heads)
{
    uint32_t row = apxinf_row_from_grid_yz();
    if (row >= rows) return;

    uint32_t seq_pos = row / n_heads;
    uint32_t valid_cols = min(seq_pos + kv_offset + 1, cols);
    uint32_t lane = threadIdx.x;
    __shared__ float max_values[256];
    float local_max = -INFINITY;
    const __nv_bfloat16* row_scores = scores + row * cols;
    for (uint32_t c = lane; c < valid_cols; c += blockDim.x) {
        local_max = fmaxf(local_max, __bfloat162float(row_scores[c]));
    }
    max_values[lane] = local_max;
    __syncthreads();
    for (uint32_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            max_values[lane] = fmaxf(max_values[lane], max_values[lane + offset]);
        }
        __syncthreads();
    }
    float max_val = max_values[0];
    for (uint32_t c = lane; c < valid_cols; c += blockDim.x) {
        numerators[row * cols + c] =
            expf(__bfloat162float(row_scores[c]) - max_val);
    }
    __syncthreads();
    __shared__ float sum_shared;
    if (lane == 0) {
        float sum_exp = 0.0f;
        const float* row_numerators = numerators + row * cols;
        uint32_t c = 0;
        if (reinterpret_cast<uintptr_t>(row_numerators) % alignof(float2) == 0) {
            for (; c + 3 < valid_cols; c += 4) {
                float2 values01 = reinterpret_cast<const float2*>(row_numerators + c)[0];
                float2 values23 = reinterpret_cast<const float2*>(row_numerators + c)[1];
                sum_exp += values01.x;
                sum_exp += values01.y;
                sum_exp += values23.x;
                sum_exp += values23.y;
            }
        } else {
            for (; c + 3 < valid_cols; c += 4) {
                float value0 = row_numerators[c];
                float value1 = row_numerators[c + 1];
                float value2 = row_numerators[c + 2];
                float value3 = row_numerators[c + 3];
                sum_exp += value0;
                sum_exp += value1;
                sum_exp += value2;
                sum_exp += value3;
            }
        }
        for (; c < valid_cols; c++) {
            sum_exp += row_numerators[c];
        }
        sum_shared = sum_exp;
    }
    __syncthreads();
    for (uint32_t c = lane; c < cols; c += blockDim.x) {
        output[row * cols + c] = c < valid_cols
            ? __float2bfloat16(numerators[row * cols + c] / sum_shared)
            : __float2bfloat16(0.0f);
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
// current contract covers head dimensions through 128, so at most four values live in
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
    uint32_t seq_len, uint32_t n_heads, uint32_t head_dim, float scale,
    const uint32_t* group_ids)
{
    uint32_t head = blockIdx.y;
    uint32_t qi   = blockIdx.x;
    if (qi >= seq_len) return;
    int tid = threadIdx.x;       // 0..31
    constexpr int kMaxElementsPerThread = 4;
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
        if (group_ids != nullptr && group_ids[qi] != group_ids[ki]) {
            if (tid == 0) scores[ki] = -INFINITY;
            continue;
        }
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
    float accumulators[kMaxElementsPerThread] = {0.0f, 0.0f, 0.0f, 0.0f};
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


// Windowed vision attention over an indexed group plan. `group_offsets` and
// `group_indices` describe stable, ascending original key indices per group.
// Keeping each key's original index preserves the scalar key order, the
// original lane assignment (`key % 32`) for max/sum, and therefore the BF16
// result of the full-scan grouped kernel while avoiding all out-of-window K/V
// reads and zero-value accumulations.
__global__ void vision_grouped_indexed_sdpa_bf16_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* out,
    uint32_t seq_len, uint32_t n_heads, uint32_t head_dim, float scale,
    const uint32_t* group_ids, const uint32_t* group_offsets,
    const uint32_t* group_indices, uint32_t group_count)
{
    uint32_t head = blockIdx.y;
    uint32_t qi = blockIdx.x;
    if (qi >= seq_len) return;
    int tid = threadIdx.x;
    constexpr int kMaxElementsPerThread = 4;
    int dimensions[kMaxElementsPerThread];
    float query_values[kMaxElementsPerThread];
    int dimension_count = 0;

    uint32_t group = group_ids[qi];
    if (group >= group_count) return;
    uint32_t group_start = group_offsets[group];
    uint32_t group_end = group_offsets[group + 1];
    if (group_start >= group_end || group_end > seq_len) return;

    extern __shared__ float smem[];
    float* scores = smem;
    const __nv_bfloat16* q_row =
        q + qi * n_heads * head_dim + head * head_dim;
    for (uint32_t dimension = tid; dimension < head_dim; dimension += 32u) {
        dimensions[dimension_count] = static_cast<int>(dimension);
        query_values[dimension_count] = __bfloat162float(q_row[dimension]);
        dimension_count++;
    }

    // Scores are written at original key indices. All lanes participate in
    // every dot-product reduction exactly as in the full-scan kernel.
    for (uint32_t position = group_start; position < group_end; position++) {
        uint32_t ki = group_indices[position];
        const __nv_bfloat16* k_row =
            k + ki * n_heads * head_dim + head * head_dim;
        float dot = 0.0f;
        for (int slot = 0; slot < dimension_count; slot++) {
            dot += query_values[slot] *
                __bfloat162float(k_row[dimensions[slot]]);
        }
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, off);
        if (tid == 0) scores[ki] = dot * scale;
    }
    __syncthreads();

    // Preserve the old per-lane key partition. Omitted keys were -INFINITY
    // in max and exact zero in sum, so skipping them does not change either
    // lane accumulator or the fixed warp reduction tree.
    float max_val = -INFINITY;
    for (uint32_t position = group_start; position < group_end; position++) {
        uint32_t ki = group_indices[position];
        if ((ki & 31u) == static_cast<uint32_t>(tid))
            max_val = fmaxf(max_val, scores[ki]);
    }
    for (int off = 16; off > 0; off >>= 1)
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, off));
    if (tid == 0) scores[seq_len] = max_val;
    __syncthreads();
    max_val = scores[seq_len];

    float sum = 0.0f;
    for (uint32_t position = group_start; position < group_end; position++) {
        uint32_t ki = group_indices[position];
        if ((ki & 31u) == static_cast<uint32_t>(tid)) {
            float e = expf(scores[ki] - max_val);
            scores[ki] = e;
            sum += e;
        }
    }
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, off);
    if (tid == 0) scores[seq_len] = sum;
    __syncthreads();
    float inv_sum = 1.0f / scores[seq_len];
    for (uint32_t position = group_start; position < group_end; position++) {
        uint32_t ki = group_indices[position];
        if ((ki & 31u) == static_cast<uint32_t>(tid)) scores[ki] *= inv_sum;
    }
    __syncthreads();

    float accumulators[kMaxElementsPerThread] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t position = group_start; position < group_end; position++) {
        uint32_t ki = group_indices[position];
        float probability = scores[ki];
        const __nv_bfloat16* v_row =
            v + ki * n_heads * head_dim + head * head_dim;
        for (int slot = 0; slot < dimension_count; slot++) {
            accumulators[slot] += probability *
                __bfloat162float(v_row[dimensions[slot]]);
        }
    }
    __nv_bfloat16* out_row =
        out + qi * n_heads * head_dim + head * head_dim;
    for (int slot = 0; slot < dimension_count; slot++) {
        out_row[dimensions[slot]] = __float2bfloat16(accumulators[slot]);
    }
}


// Pack Q/K/V rows into stable group order for variable-length FA2. Each
// vector covers eight BF16 elements; the caller proves row_elements % 8 == 0
// and provides a permutation from packed row to original row.
__global__ void vision_pack_grouped_qkv_bf16_kernel(
    const uint4* q, const uint4* k, const uint4* v,
    uint4* packed_q, uint4* packed_k, uint4* packed_v,
    const uint32_t* group_indices, uint32_t rows,
    uint32_t vectors_per_row)
{
    uint64_t packed_vector =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t vector_count =
        static_cast<uint64_t>(rows) * vectors_per_row;
    if (packed_vector >= vector_count) return;
    uint32_t packed_row = static_cast<uint32_t>(packed_vector / vectors_per_row);
    uint32_t vector_in_row = static_cast<uint32_t>(packed_vector % vectors_per_row);
    uint32_t original_row = group_indices[packed_row];
    uint64_t original_vector =
        static_cast<uint64_t>(original_row) * vectors_per_row + vector_in_row;
    packed_q[packed_vector] = q[original_vector];
    packed_k[packed_vector] = k[original_vector];
    packed_v[packed_vector] = v[original_vector];
}

// Restore packed variable-length FA2 output to the model-owned token order.
// group_indices is a permutation, so stores are disjoint and need no atomics.
__global__ void vision_unpack_grouped_rows_bf16_kernel(
    const uint4* packed, uint4* output, const uint32_t* group_indices,
    uint32_t rows, uint32_t vectors_per_row)
{
    uint64_t packed_vector =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t vector_count =
        static_cast<uint64_t>(rows) * vectors_per_row;
    if (packed_vector >= vector_count) return;
    uint32_t packed_row = static_cast<uint32_t>(packed_vector / vectors_per_row);
    uint32_t vector_in_row = static_cast<uint32_t>(packed_vector % vectors_per_row);
    uint32_t original_row = group_indices[packed_row];
    uint64_t original_vector =
        static_cast<uint64_t>(original_row) * vectors_per_row + vector_in_row;
    output[original_vector] = packed[packed_vector];
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
