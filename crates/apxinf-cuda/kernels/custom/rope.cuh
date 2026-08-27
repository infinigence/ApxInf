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

// Qwen2.5-Omni TMRoPE: the full head dimension is split into contiguous
// T/H/W sections of lengths 2*mrope_section. rotate_half is still evaluated
// across the head midpoint, so the two members of a rotated pair may use
// different modality axes exactly as in the Hugging Face reference.
__global__ void rope_tmrope_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    uint32_t head_dim, uint32_t n_heads, uint32_t seq_len,
    float theta, const uint32_t* pos_ids,
    uint32_t sec_t, uint32_t sec_h)
{
    uint32_t dimension = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head_idx = blockIdx.y;
    uint32_t seq_idx = blockIdx.z;
    if (dimension >= head_dim) return;
    uint32_t axis = dimension < 2 * sec_t ? 0u
        : (dimension < 2 * (sec_t + sec_h) ? 1u : 2u);
    uint32_t half = head_dim / 2;
    uint32_t pair = dimension % half;
    float frequency = 1.0f / powf(theta, 2.0f * (float)pair / (float)head_dim);
    float angle = (float)pos_ids[seq_idx * 3 + axis] * frequency;
    float cos_value = cosf(angle);
    float sin_value = sinf(angle);
    uint32_t base = seq_idx * n_heads * head_dim + head_idx * head_dim;
    float value = __bfloat162float(input[base + dimension]);
    float rotated = dimension < half
        ? -__bfloat162float(input[base + dimension + half])
        : __bfloat162float(input[base + dimension - half]);
    output[base + dimension] = __float2bfloat16(value * cos_value + rotated * sin_value);
}

// Decode-only Qwen2.5-Omni K TMRoPE plus K/V cache publication. K uses the
// same per-dimension arithmetic as rope_tmrope_bf16_kernel; V is copied
// unchanged into the same cache slot.
__global__ void rope_tmrope_kv_write_bf16_kernel(
    const __nv_bfloat16* k_in, const __nv_bfloat16* v_in,
    __nv_bfloat16* k_cache, __nv_bfloat16* v_cache,
    uint32_t head_dim, uint32_t n_kv_heads, uint32_t max_seq_len,
    float theta, const uint32_t* pos_ids,
    uint32_t sec_t, uint32_t sec_h, const uint32_t* cache_pos)
{
    uint32_t dimension = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head_idx = blockIdx.y;
    if (dimension >= head_dim) return;
    uint32_t axis = dimension < 2 * sec_t ? 0u
        : (dimension < 2 * (sec_t + sec_h) ? 1u : 2u);
    uint32_t half = head_dim / 2;
    uint32_t pair = dimension % half;
    float frequency = 1.0f / powf(theta, 2.0f * (float)pair / (float)head_dim);
    float angle = (float)pos_ids[axis] * frequency;
    float cos_value = cosf(angle);
    float sin_value = sinf(angle);
    uint32_t source = head_idx * head_dim;
    float value = __bfloat162float(k_in[source + dimension]);
    float rotated = dimension < half
        ? -__bfloat162float(k_in[source + dimension + half])
        : __bfloat162float(k_in[source + dimension - half]);
    uint32_t destination =
        head_idx * max_seq_len * head_dim + *cache_pos * head_dim + dimension;
    k_cache[destination] = __float2bfloat16(value * cos_value + rotated * sin_value);
    v_cache[destination] = v_in[source + dimension];
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


