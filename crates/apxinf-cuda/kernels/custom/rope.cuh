#pragma once

// Copyright 2026 apxinf contributors.
// Pure CUDA operators grouped by physical operation; launch policy lives under adapters/.

// Apply caller-provided half-split RoPE cosine/sine tensors. Each BF16
// multiply is rounded before the BF16 add, matching an unfused tensor
// expression `(x * cos) + (rotate_half(x) * sin)` rather than reassociating it
// into one FP32/FMA expression.
__global__ void rope_precomputed_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* cosine,
    const __nv_bfloat16* sine, __nv_bfloat16* output, int tokens,
    int heads, int head_dim) {
  const int half = head_dim / 2;
  const int64_t pair_count =
      static_cast<int64_t>(tokens) * heads * half;
  int64_t pair_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; pair_index < pair_count; pair_index += stride) {
    const int pair = static_cast<int>(pair_index % half);
    const int64_t token_head = pair_index / half;
    const int head = static_cast<int>(token_head % heads);
    const int token = static_cast<int>(token_head / heads);
    const int64_t input_base =
        (static_cast<int64_t>(token) * heads + head) * head_dim;
    const int64_t metadata_base = static_cast<int64_t>(token) * head_dim;
    const int64_t first = input_base + pair;
    const int64_t second = input_base + half + pair;

    const float x_first = __bfloat162float(input[first]);
    const float x_second = __bfloat162float(input[second]);
    const __nv_bfloat16 first_cos = __float2bfloat16_rn(
        x_first * __bfloat162float(cosine[metadata_base + pair]));
    const __nv_bfloat16 first_sin = __float2bfloat16_rn(
        -x_second * __bfloat162float(sine[metadata_base + pair]));
    const __nv_bfloat16 second_cos = __float2bfloat16_rn(
        x_second * __bfloat162float(cosine[metadata_base + half + pair]));
    const __nv_bfloat16 second_sin = __float2bfloat16_rn(
        x_first * __bfloat162float(sine[metadata_base + half + pair]));
    output[first] = __float2bfloat16_rn(
        __bfloat162float(first_cos) + __bfloat162float(first_sin));
    output[second] = __float2bfloat16_rn(
        __bfloat162float(second_cos) + __bfloat162float(second_sin));
  }
}


// Sinusoidal embedding from a BF16 scalar schedule. Frequency construction
// intentionally stays in FP32 and follows:
//   fraction = linspace(0, 1, dim/2)
//   period = min_period * pow(max_period / min_period, fraction)
//   angle = (1 / period * 2*pi) * time.float()
// with `[sin, cos]` concatenation and one final BF16 rounding.
__global__ void sinusoidal_time_embedding_bf16_kernel(
    const __nv_bfloat16* times, __nv_bfloat16* output, int steps,
    int dimension, float min_period, float max_period) {
  const int half = dimension / 2;
  const int64_t frequency_count = static_cast<int64_t>(steps) * half;
  int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  constexpr float kTwoPi = 6.28318530717958647692f;
  for (; index < frequency_count; index += stride) {
    const int step = static_cast<int>(index / half);
    const int frequency = static_cast<int>(index % half);
    float fraction = 0.0f;
    if (half > 1) {
      const float fraction_step =
          __fdiv_rn(1.0f, static_cast<float>(half - 1));
      const int halfway = half / 2;
      fraction = frequency < halfway
          ? __fmul_rn(fraction_step, static_cast<float>(frequency))
          : __fsub_rn(
                1.0f,
                __fmul_rn(
                    fraction_step,
                    static_cast<float>(half - frequency - 1)));
    }
    const float ratio = __fdiv_rn(max_period, min_period);
    const float period =
        __fmul_rn(min_period, powf(ratio, fraction));
    const float inverse_period = __fdiv_rn(1.0f, period);
    const float angular_scale = __fmul_rn(inverse_period, kTwoPi);
    const float angle = __fmul_rn(
        angular_scale, __bfloat162float(times[step]));
    const int64_t row = static_cast<int64_t>(step) * dimension;
    output[row + frequency] = __float2bfloat16_rn(sinf(angle));
    output[row + half + frequency] = __float2bfloat16_rn(cosf(angle));
  }
}


// Build half-split RoPE cosine/sine tables from explicit u32 positions. A
// linear factor divides inverse frequency, matching linear RoPE scaling.
__global__ void rope_tables_bf16_kernel(
    const uint32_t* positions, __nv_bfloat16* cosine,
    __nv_bfloat16* sine, int tokens, int head_dim, float theta,
    float linear_factor) {
  const int half = head_dim / 2;
  const int64_t frequency_count = static_cast<int64_t>(tokens) * half;
  int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < frequency_count; index += stride) {
    const int token = static_cast<int>(index / half);
    const int pair = static_cast<int>(index % half);
    const float exponent = __fdiv_rn(
        static_cast<float>(2 * pair), static_cast<float>(head_dim));
    const float base_frequency = powf(theta, exponent);
    const float inverse_frequency = __fdiv_rn(
        __fdiv_rn(1.0f, base_frequency), linear_factor);
    const float angle = __fmul_rn(
        static_cast<float>(positions[token]), inverse_frequency);
    const __nv_bfloat16 cos_value = __float2bfloat16_rn(cosf(angle));
    const __nv_bfloat16 sin_value = __float2bfloat16_rn(sinf(angle));
    const int64_t row = static_cast<int64_t>(token) * head_dim;
    cosine[row + pair] = cos_value;
    cosine[row + half + pair] = cos_value;
    sine[row + pair] = sin_value;
    sine[row + half + pair] = sin_value;
  }
}

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
