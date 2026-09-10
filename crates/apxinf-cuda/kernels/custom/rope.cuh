#pragma once

// Copyright 2026 apxinf contributors.
// Pure CUDA operators grouped by physical operation; launch policy lives under adapters/.

// ── RoPE ──────────────────────────────────────────────────────────────────

__global__ void rope_f32_kernel(
    const float* input, float* output,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float rope_theta, uint32_t pos_offset)
{
    uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head_idx = blockIdx.y;
    uint32_t seq_idx  = blockIdx.z;
    if (pair_idx >= head_dim / 2) return;

    uint32_t pos = seq_idx + pos_offset;
    float freq = 1.0f / powf(rope_theta, 2.0f * (float)pair_idx / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    uint32_t base = seq_idx * n_heads * head_dim + head_idx * head_dim;
    uint32_t idx0 = base + 2 * pair_idx;
    uint32_t idx1 = base + 2 * pair_idx + 1;

    float x0 = input[idx0];
    float x1 = input[idx1];
    output[idx0] = x0 * cos_val - x1 * sin_val;
    output[idx1] = x0 * sin_val + x1 * cos_val;
}



// ── RoPE Batched (half-split, no sync) ────────────────────────────────────
//
// Input/output shape: [seq_len, n_heads, head_dim]
// Half-split pairs: (i, i + head_dim/2) for i in 0..head_dim/2
// This matches the CPU RoPE convention (not interleaved).

__global__ void rope_batched_f32_kernel(
    const float* input, float* output,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float rope_theta, uint32_t pos_offset)
{
    uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head_idx = blockIdx.y;
    uint32_t seq_idx  = blockIdx.z;
    if (pair_idx >= head_dim / 2) return;

    uint32_t pos = seq_idx + pos_offset;
    float freq = 1.0f / powf(rope_theta, 2.0f * (float)pair_idx / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    uint32_t base = seq_idx * n_heads * head_dim + head_idx * head_dim;
    uint32_t half = head_dim / 2;
    uint32_t idx0 = base + pair_idx;
    uint32_t idx1 = base + half + pair_idx;

    float x0 = input[idx0];
    float x1 = input[idx1];
    output[idx0] = x0 * cos_val - x1 * sin_val;
    output[idx1] = x0 * sin_val + x1 * cos_val;
}



// RoPE for a single token (seq_len=1), pos from device ptr. Half-split pairs.
__global__ void rope_decode_f32_kernel(
    const float* input, float* output,
    uint32_t head_dim, uint32_t n_heads,
    float rope_theta, const uint32_t* pos_ptr)
{
    uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head_idx = blockIdx.y;
    if (pair_idx >= head_dim / 2) return;

    uint32_t pos = *pos_ptr;
    float freq = 1.0f / powf(rope_theta, 2.0f * (float)pair_idx / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    uint32_t base = head_idx * head_dim;
    uint32_t half = head_dim / 2;
    uint32_t idx0 = base + pair_idx;
    uint32_t idx1 = base + half + pair_idx;
    float x0 = input[idx0];
    float x1 = input[idx1];
    output[idx0] = x0 * cos_val - x1 * sin_val;
    output[idx1] = x0 * sin_val + x1 * cos_val;
}



// ── RoPE (bf16) — interleaved-pairs variant, matches rope_f32 ─────────────

__global__ void rope_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float rope_theta, uint32_t pos_offset)
{
    uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head_idx = blockIdx.y;
    uint32_t seq_idx  = blockIdx.z;
    if (pair_idx >= head_dim / 2) return;

    uint32_t pos = seq_idx + pos_offset;
    float freq = 1.0f / powf(rope_theta, 2.0f * (float)pair_idx / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    uint32_t base = seq_idx * n_heads * head_dim + head_idx * head_dim;
    uint32_t idx0 = base + 2 * pair_idx;
    uint32_t idx1 = base + 2 * pair_idx + 1;

    float x0 = __bfloat162float(input[idx0]);
    float x1 = __bfloat162float(input[idx1]);
    output[idx0] = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    output[idx1] = __float2bfloat16(x0 * sin_val + x1 * cos_val);
}



// ── RoPE Batched (bf16) — half-split pairs ────────────────────────────────

__global__ void rope_batched_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float rope_theta, uint32_t pos_offset)
{
    uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head_idx = blockIdx.y;
    uint32_t seq_idx  = blockIdx.z;
    if (pair_idx >= head_dim / 2) return;

    uint32_t pos = seq_idx + pos_offset;
    float freq = 1.0f / powf(rope_theta, 2.0f * (float)pair_idx / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    uint32_t base = seq_idx * n_heads * head_dim + head_idx * head_dim;
    uint32_t half = head_dim / 2;
    uint32_t idx0 = base + pair_idx;
    uint32_t idx1 = base + half + pair_idx;

    float x0 = __bfloat162float(input[idx0]);
    float x1 = __bfloat162float(input[idx1]);
    output[idx0] = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    output[idx1] = __float2bfloat16(x0 * sin_val + x1 * cos_val);
}



__global__ void rope_decode_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    uint32_t head_dim, uint32_t n_heads,
    float rope_theta, const uint32_t* pos_ptr)
{
    uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head_idx = blockIdx.y;
    if (pair_idx >= head_dim / 2) return;

    uint32_t pos = *pos_ptr;
    float freq = 1.0f / powf(rope_theta, 2.0f * (float)pair_idx / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    uint32_t base = head_idx * head_dim;
    uint32_t half = head_dim / 2;
    uint32_t idx0 = base + pair_idx;
    uint32_t idx1 = base + half + pair_idx;
    float x0 = __bfloat162float(input[idx0]);
    float x1 = __bfloat162float(input[idx1]);
    output[idx0] = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    output[idx1] = __float2bfloat16(x0 * sin_val + x1 * cos_val);
}



// ── mRoPE (bf16) — Qwen3-VL multimodal RoPE ───────────────────────────────
//
// Same math as rotate_half RoPE but with a per-pair axis lookup: the 64
// frequency pairs (head_dim=128 assumed by Qwen3-VL) are assigned to one
// of three position axes {T,H,W} following HF's `apply_interleaved_mrope`
// (`transformers/models/qwen3_vl/modeling_qwen3_vl.py`):
//
//   axis(p) = 1 (H)   if p % 3 == 1  and  p < sec_h * 3
//           = 2 (W)   if p % 3 == 2  and  p < sec_w * 3
//           = 0 (T)   otherwise    (defaults, includes the tail p >= max*3)
//
// This matches HF exactly for Qwen3-VL's mrope_section=[24,20,20].
// Rotation itself is GPT-J style: pair p rotates elements [p, p+head_dim/2].
// pos_ids is `[seq_len, 3]` flat u32 (t,h,w per token). For text-only calls,
// pass (i,i,i) and mRoPE degenerates to 1-D RoPE.

__device__ __forceinline__ uint32_t mrope_axis_for_pair(
    uint32_t pair_idx, uint32_t sec_h, uint32_t sec_w)
{
    uint32_t rem = pair_idx % 3;
    if (rem == 1 && pair_idx < sec_h * 3) return 1;
    if (rem == 2 && pair_idx < sec_w * 3) return 2;
    return 0;
}

__global__ void rope_mrope_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float theta, const uint32_t* pos_ids,
    uint32_t sec_h, uint32_t sec_w)
{
    uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head_idx = blockIdx.y;
    uint32_t seq_idx  = blockIdx.z;
    if (pair_idx >= head_dim / 2) return;

    uint32_t axis = mrope_axis_for_pair(pair_idx, sec_h, sec_w);
    uint32_t pos  = pos_ids[seq_idx * 3 + axis];

    float freq    = 1.0f / powf(theta, 2.0f * (float)pair_idx / (float)head_dim);
    float angle   = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    uint32_t base = seq_idx * n_heads * head_dim + head_idx * head_dim;
    uint32_t half = head_dim / 2;
    uint32_t idx0 = base + pair_idx;
    uint32_t idx1 = base + half + pair_idx;
    float x0 = __bfloat162float(input[idx0]);
    float x1 = __bfloat162float(input[idx1]);
    output[idx0] = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    output[idx1] = __float2bfloat16(x0 * sin_val + x1 * cos_val);
}

__global__ void prepare_mrope_cos_sin_f32_kernel(
    float2* table, uint32_t head_dim, uint32_t seq_len, float theta,
    const uint32_t* pos_ids, uint32_t sec_h, uint32_t sec_w)
{
    const uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t seq = blockIdx.y;
    const uint32_t half = head_dim / 2;
    if (pair >= half || seq >= seq_len) return;
    const uint32_t axis = mrope_axis_for_pair(pair, sec_h, sec_w);
    const uint32_t pos = pos_ids[seq * 3 + axis];
    const float frequency =
        1.0f / powf(theta, 2.0f * static_cast<float>(pair) /
                               static_cast<float>(head_dim));
    const float angle = static_cast<float>(pos) * frequency;
    table[static_cast<size_t>(seq) * half + pair] =
        make_float2(cosf(angle), sinf(angle));
}

__global__ void rope_mrope_precomputed_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output, const float2* table,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len)
{
    const uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t seq = blockIdx.z;
    const uint32_t half = head_dim / 2;
    if (pair >= half || head >= n_heads || seq >= seq_len) return;
    const float2 rotation = table[static_cast<size_t>(seq) * half + pair];
    const size_t base =
        (static_cast<size_t>(seq) * n_heads + head) * head_dim;
    const float first = __bfloat162float(input[base + pair]);
    const float second = __bfloat162float(input[base + half + pair]);
    output[base + pair] =
        __float2bfloat16(first * rotation.x - second * rotation.y);
    output[base + half + pair] =
        __float2bfloat16(first * rotation.y + second * rotation.x);
}




// Decode-position variant: seq_len is always 1, pos_ids is a [3] u32 buffer
// read from device memory (so the captured graph is static across replay).
__global__ void rope_mrope_decode_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    uint32_t head_dim, uint32_t n_heads,
    float theta, const uint32_t* pos_ids,
    uint32_t sec_h, uint32_t sec_w)
{
    uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head_idx = blockIdx.y;
    if (pair_idx >= head_dim / 2) return;

    uint32_t axis = mrope_axis_for_pair(pair_idx, sec_h, sec_w);
    uint32_t pos  = pos_ids[axis];

    float freq    = 1.0f / powf(theta, 2.0f * (float)pair_idx / (float)head_dim);
    float angle   = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    uint32_t base = head_idx * head_dim;
    uint32_t half = head_dim / 2;
    uint32_t idx0 = base + pair_idx;
    uint32_t idx1 = base + half + pair_idx;
    float x0 = __bfloat162float(input[idx0]);
    float x1 = __bfloat162float(input[idx1]);
    output[idx0] = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    output[idx1] = __float2bfloat16(x0 * sin_val + x1 * cos_val);
}



// ── Vision 2D-RoPE (bf16) — Qwen3-VL vision tower ────────────────────────
//
// HF's `Qwen3VLVisionRotaryEmbedding` + `rot_pos_emb` + `apply_rotary_pos_
// emb_vision`: head_dim=64, 16 freq pairs per axis, first 16 pairs use the
// h coordinate, next 16 pairs use the w coordinate. Rotation is rotate_half
// (pair p rotates elements [p, p + head_dim/2]).
//
// pos_ids is `[seq_len, 2]` flat u32 (h, w) per token. theta defaults to
// 10000.0 for the vision tower.

__global__ void prepare_vision_rope_cos_sin_f32_kernel(
    float2* table, uint32_t head_dim, uint32_t seq_len, float theta,
    const uint32_t* pos_ids)
{
    const uint32_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t seq = blockIdx.y;
    const uint32_t half = head_dim / 2;
    if (pair >= half || seq >= seq_len) return;
    const uint32_t axis = pair < half / 2 ? 0u : 1u;
    const uint32_t pair_in_axis = pair < half / 2 ? pair : pair - half / 2;
    const uint32_t pos = pos_ids[seq * 2 + axis];
    const float frequency =
        1.0f / powf(theta, 2.0f * static_cast<float>(pair_in_axis) /
                               static_cast<float>(half));
    float sine, cosine;
    sincosf(static_cast<float>(pos) * frequency, &sine, &cosine);
    table[static_cast<size_t>(seq) * half + pair] = make_float2(cosine, sine);
}

__global__ void rope_vision_2d_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float theta, const uint32_t* pos_ids)
{
    uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head_idx = blockIdx.y;
    uint32_t seq_idx  = blockIdx.z;
    uint32_t half = head_dim / 2;
    if (pair_idx >= half) return;

    // First half of pairs → h axis (pos_ids[seq, 0]); second half → w.
    uint32_t axis = (pair_idx < half / 2) ? 0u : 1u;
    uint32_t pair_in_axis = pair_idx < half / 2 ? pair_idx : pair_idx - (half / 2);
    uint32_t pos = pos_ids[seq_idx * 2 + axis];

    float freq  = 1.0f / powf(theta, 2.0f * (float)pair_in_axis / (float)half);
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    uint32_t base = seq_idx * n_heads * head_dim + head_idx * head_dim;
    uint32_t idx0 = base + pair_idx;
    uint32_t idx1 = base + half + pair_idx;
    float x0 = __bfloat162float(input[idx0]);
    float x1 = __bfloat162float(input[idx1]);
    output[idx0] = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    output[idx1] = __float2bfloat16(x0 * sin_val + x1 * cos_val);
}

// Apply the same 2D rotary embedding to Q and K in one launch. Flattening the
// sequence/head/pair dimensions avoids launching one mostly-idle 256-thread
// block for every (sequence, head) tuple when head_dim is 64.
__global__ void rope_vision_2d_pair_bf16_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k,
    __nv_bfloat16* q_out, __nv_bfloat16* k_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float theta, const uint32_t* pos_ids)
{
    uint64_t linear = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t half = head_dim / 2;
    uint64_t total = (uint64_t)seq_len * n_heads * half;
    if (linear >= total) return;

    uint32_t pair_idx = (uint32_t)(linear % half);
    uint64_t token_head = linear / half;
    uint32_t head_idx = (uint32_t)(token_head % n_heads);
    uint32_t seq_idx = (uint32_t)(token_head / n_heads);
    uint32_t axis = pair_idx < half / 2 ? 0u : 1u;
    uint32_t pair_in_axis = pair_idx < half / 2 ? pair_idx : pair_idx - half / 2;
    uint32_t pos = pos_ids[seq_idx * 2 + axis];

    float freq = 1.0f / powf(theta, 2.0f * (float)pair_in_axis / (float)half);
    float angle = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    uint64_t base = ((uint64_t)seq_idx * n_heads + head_idx) * head_dim;
    uint64_t idx0 = base + pair_idx;
    uint64_t idx1 = base + half + pair_idx;

    float q0 = __bfloat162float(q[idx0]);
    float q1 = __bfloat162float(q[idx1]);
    q_out[idx0] = __float2bfloat16(q0 * cos_val - q1 * sin_val);
    q_out[idx1] = __float2bfloat16(q0 * sin_val + q1 * cos_val);

    float k0 = __bfloat162float(k[idx0]);
    float k1 = __bfloat162float(k[idx1]);
    k_out[idx0] = __float2bfloat16(k0 * cos_val - k1 * sin_val);
    k_out[idx1] = __float2bfloat16(k0 * sin_val + k1 * cos_val);
}

// Fuse the vision QKV layout split, bias and 2D RoPE.  The explicit BF16
// rounding before rotation preserves the established two-kernel contract:
// split_qkv_bias_bf16 writes BF16 Q/K/V, then rope reads those BF16 values.
template <bool kPrecomputed>
__global__ void qkv_split_bias_vision_rope_bf16_kernel(
    const __nv_bfloat16* qkv, const __nv_bfloat16* bias,
    __nv_bfloat16* q_out, __nv_bfloat16* k_out, __nv_bfloat16* v_out,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float theta, const uint32_t* pos_ids, const float2* rotation_table)
{
    uint64_t linear = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t half = head_dim / 2;
    const uint64_t total = (uint64_t)seq_len * n_heads * half;
    if (linear >= total) return;

    const uint32_t pair_idx = (uint32_t)(linear % half);
    const uint64_t token_head = linear / half;
    const uint32_t head_idx = (uint32_t)(token_head % n_heads);
    const uint32_t seq_idx = (uint32_t)(token_head / n_heads);
    const uint32_t width = n_heads * head_dim;
    const uint64_t qkv_row = (uint64_t)seq_idx * 3 * width;
    const uint32_t head_col = head_idx * head_dim;
    const uint32_t second = pair_idx + half;
    const uint64_t out_base = ((uint64_t)seq_idx * n_heads + head_idx) * head_dim;

    // Match split_qkv_bias_bf16's intermediate output rounding exactly.
    const __nv_bfloat16 q0_bf16 = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + head_col + pair_idx]) +
        __bfloat162float(bias[head_col + pair_idx]));
    const __nv_bfloat16 q1_bf16 = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + head_col + second]) +
        __bfloat162float(bias[head_col + second]));
    const __nv_bfloat16 k0_bf16 = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + width + head_col + pair_idx]) +
        __bfloat162float(bias[width + head_col + pair_idx]));
    const __nv_bfloat16 k1_bf16 = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + width + head_col + second]) +
        __bfloat162float(bias[width + head_col + second]));

    float sine, cosine;
    if constexpr (kPrecomputed) {
        const float2 rotation =
            rotation_table[static_cast<size_t>(seq_idx) * half + pair_idx];
        cosine = rotation.x;
        sine = rotation.y;
    } else {
        const uint32_t axis = pair_idx < half / 2 ? 0u : 1u;
        const uint32_t pair_in_axis =
            pair_idx < half / 2 ? pair_idx : pair_idx - half / 2;
        const uint32_t pos = pos_ids[seq_idx * 2 + axis];
        const float frequency =
            1.0f / powf(theta, 2.0f * static_cast<float>(pair_in_axis) /
                                   static_cast<float>(half));
        sincosf(static_cast<float>(pos) * frequency, &sine, &cosine);
    }
    const float q0 = __bfloat162float(q0_bf16);
    const float q1 = __bfloat162float(q1_bf16);
    const float k0 = __bfloat162float(k0_bf16);
    const float k1 = __bfloat162float(k1_bf16);
    q_out[out_base + pair_idx] = __float2bfloat16(q0 * cosine - q1 * sine);
    q_out[out_base + second] = __float2bfloat16(q0 * sine + q1 * cosine);
    k_out[out_base + pair_idx] = __float2bfloat16(k0 * cosine - k1 * sine);
    k_out[out_base + second] = __float2bfloat16(k0 * sine + k1 * cosine);

    v_out[out_base + pair_idx] = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + 2 * width + head_col + pair_idx]) +
        __bfloat162float(bias[2 * width + head_col + pair_idx]));
    v_out[out_base + second] = __float2bfloat16(
        __bfloat162float(qkv[qkv_row + 2 * width + head_col + second]) +
        __bfloat162float(bias[2 * width + head_col + second]));
}
