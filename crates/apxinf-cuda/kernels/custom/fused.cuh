#pragma once

// Copyright 2026 apxinf contributors.
// Pure CUDA operators grouped by physical operation; launch policy lives under adapters/.

// ── Fused RMSNorm + residual add (bf16) — shared-memory reduction ────────
//
// Replaces: x = x + delta; normed = rms_norm(x, weight, eps)
// In one pass: computes x_new = x + delta, writes x_new back to x_inout
// (for the next layer's residual), and writes normed = x_new * rsqrt(rms+eps)
// * weight to output.
//
// Uses shared memory so each element is read from HBM exactly once (the
// naive per-thread loop would do O(cols²) reads — 2× worse than the
// separate add + norm it replaces). One block per row; block size up to
// BLOCK_SIZE (256) threads, strided load, warp-shuffle reduction.

template <bool ROUND_BEFORE_NORM>
__global__ void rms_norm_add_bf16_kernel(
    __nv_bfloat16* x_inout, const __nv_bfloat16* delta,
    const __nv_bfloat16* weight, __nv_bfloat16* output,
    uint32_t cols, uint32_t rows, float eps)
{
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    uint32_t tid = threadIdx.x;
    uint32_t offset = row * cols;

    // Shared memory: x_new[cols] in fp32 + one slot for the reduced sum.
    extern __shared__ float smem[];
    float* x_new = smem;        // [cols]
    __shared__ float s_sum;

    // Phase 1: strided load of x+delta into shared memory; each thread
    // accumulates its partial sum_sq.
    float partial = 0.0f;
    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        float xv = __bfloat162float(x_inout[offset + i]);
        float dv = __bfloat162float(delta[offset + i]);
        float xn = xv + dv;
        if constexpr (ROUND_BEFORE_NORM) {
            xn = __bfloat162float(__float2bfloat16(xn));
        }
        x_new[i] = xn;
        partial += xn * xn;
    }

    // Phase 2: warp-shuffle reduction within each warp, then a small
    // shared-memory reduction across warps. Assumes blockDim.x <= 1024
    // (max 32 warps).
    for (int off = 16; off > 0; off >>= 1)
        partial += __shfl_xor_sync(0xffffffff, partial, off);
    // Partial now holds the warp sum for lane 0 of each warp.
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

    // Phase 3: write x_new back to x_inout and the normed output.
    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        float xn = x_new[i];
        float w  = __bfloat162float(weight[i]);
        x_inout[offset + i] = __float2bfloat16(xn);
        output[offset + i]  = __float2bfloat16(xn * rms * w);
    }
}



// ── Fused RoPE + KV cache write (bf16) for K ─────────────────────────────
//
// Applies 1-D rotate_half RoPE to K and writes the rotated values directly
// into the K cache at slot `pos`. Skips the `ws.k_rope` temp buffer that
// `rope_decode_bf16` + `kv_cache_append_decode_bf16` would use. Q still
// needs its own output buffer (the attention GEMM reads it), so Q is not
// fused. −1 kernel + 1 HBM round-trip per layer.

__global__ void rope_k_write_bf16_kernel(
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

    uint32_t dst_base = head_idx * max_seq_len * head_dim + pos * head_dim;
    k_cache[dst_base + pair_idx]        = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    k_cache[dst_base + half + pair_idx] = __float2bfloat16(x0 * sin_val + x1 * cos_val);
}




__global__ void bias_residual_rms_norm_quant_f16_e4m3_kernel(
    const half* projection, const half* bias, const half* residual,
    const half* weight, half* hidden, __nv_fp8_e4m3* normalized,
    int rows, int cols, float eps, float inverse_scale) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    int64_t index = static_cast<int64_t>(row) * cols + col;
    float value = __half2float(projection[index]) + __half2float(residual[index]);
    if (bias != nullptr) value += __half2float(bias[col]);
    half rounded = __float2half(value);
    hidden[index] = rounded;
    value = __half2float(rounded);
    square_sum += value * value;
  }
  float inverse_rms = rsqrtf(block_sum(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    int64_t index = static_cast<int64_t>(row) * cols + col;
    float value = __half2float(hidden[index]) * inverse_rms * __half2float(weight[col]);
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    normalized[index] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void bias_residual_layer_norm_quant_f16_e4m3_kernel(
    const half* projection, const half* projection_bias, const half* residual,
    const half* norm_weight, const half* norm_bias, half* hidden,
    __nv_fp8_e4m3* normalized, int rows, int cols, float eps,
    float inverse_scale) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    int64_t index = static_cast<int64_t>(row) * cols + col;
    float value = __half2float(projection[index]) + __half2float(residual[index]);
    if (projection_bias != nullptr) value += __half2float(projection_bias[col]);
    half rounded = __float2half(value);
    hidden[index] = rounded;
    sum += __half2float(rounded);
  }
  float mean = block_sum(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float centered = __half2float(hidden[static_cast<int64_t>(row) * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  float inverse_std = rsqrtf(block_sum(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    int64_t index = static_cast<int64_t>(row) * cols + col;
    float value = (__half2float(hidden[index]) - mean) * inverse_std;
    value = value * __half2float(norm_weight[col]) + __half2float(norm_bias[col]);
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    normalized[index] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void ada_gate_residual_rms_norm_quant_f16_e4m3_kernel(
    const half* projection, const half* residual, const half* gate_style,
    const half* norm_style, half* hidden, __nv_fp8_e4m3* normalized,
    int rows, int cols, float eps, float inverse_scale) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    int64_t index = static_cast<int64_t>(row) * cols + col;
    float gate = __half2float(gate_style[2 * cols + col]);
    half rounded = __float2half(
        __half2float(residual[index]) + __half2float(projection[index]) * gate);
    hidden[index] = rounded;
    float value = __half2float(rounded);
    square_sum += value * value;
  }
  float inverse_rms = rsqrtf(block_sum(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    int64_t index = static_cast<int64_t>(row) * cols + col;
    float value = __half2float(hidden[index]) * inverse_rms;
    value = value * (1.0f + __half2float(norm_style[col])) +
            __half2float(norm_style[cols + col]);
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    normalized[index] = static_cast<__nv_fp8_e4m3>(value);
  }
}

// Exact [10, 1024] action specialization. The first pass keeps
// the production 256-lane ownership and reduction tree bit-for-bit. Rounded
// hidden values are additionally cached in shared memory, and the independent
// normalize/quantize pass uses one aligned 8-byte group per thread.
__global__ void ada_gate_residual_rms_norm_quant_f16_e4m3_packed8_kernel(
    const half* projection, const half* residual, const half* gate_style,
    const half* norm_style, half* hidden, __nv_fp8_e4m3* normalized,
    float eps, float inverse_scale) {
  constexpr int cols = 1024;
  __shared__ float scratch[8];
  __shared__ half cached[cols];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float gate = __half2float(gate_style[2 * cols + col]);
    const half rounded = __float2half(
        __half2float(residual[index]) + __half2float(projection[index]) * gate);
    hidden[index] = rounded;
    cached[col] = rounded;
    const float value = __half2float(rounded);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum(square_sum, scratch) / cols + eps);

  union Half4 {
    uint2 packed;
    half values[4];
  };
  union Bytes4 {
    uint32_t packed;
    uint8_t values[4];
  };
  const int col = threadIdx.x * 4;
  Half4 h;
  Half4 scale;
  Half4 shift;
  h.packed = *reinterpret_cast<const uint2*>(cached + col);
  scale.packed = *reinterpret_cast<const uint2*>(norm_style + col);
  shift.packed = *reinterpret_cast<const uint2*>(norm_style + cols + col);
  Bytes4 output;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float value = __half2float(h.values[i]) * inverse_rms;
    value = value * (1.0f + __half2float(scale.values[i])) +
            __half2float(shift.values[i]);
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    const __nv_fp8_e4m3 quantized = static_cast<__nv_fp8_e4m3>(value);
    output.values[i] = *reinterpret_cast<const uint8_t*>(&quantized);
  }
  *reinterpret_cast<uint32_t*>(normalized + row * cols + col) = output.packed;
}


__global__ void bias_residual_f16_kernel(
    const half* projection, const half* bias, const half* residual,
    half* output, int64_t count, int cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    float value = __half2float(projection[index]) + __half2float(residual[index]);
    if (bias != nullptr) value += __half2float(bias[index % cols]);
    output[index] = __float2half(value);
  }
}

__global__ void ada_gate_residual_f16_kernel(
    const half* projection, const half* residual, const half* style,
    half* output, int rows, int cols) {
  int64_t count = static_cast<int64_t>(rows) * cols;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    int col = static_cast<int>(index % cols);
    float gate = __half2float(style[2 * cols + col]);
    float value = __half2float(residual[index]) + __half2float(projection[index]) * gate;
    output[index] = __float2half(value);
  }
}


__global__ void qkv_rope_f16_kernel(
    const half* qkv, const half* bias, half* q, half* k, half* v, int tokens, int q_heads,
    int kv_heads, int head_dim, float theta, int position_offset, int kv_output_offset) {
  const int token = blockIdx.x;
  const int projection_head = blockIdx.y;
  const int half_dim = head_dim / 2;
  const int pair = threadIdx.x;
  const int q_width = q_heads * head_dim;
  const int kv_width = kv_heads * head_dim;
  const int fused_width = q_width + 2 * kv_width;
  const int position = position_offset + token;
  if (pair >= half_dim) return;
  float frequency = powf(theta, -static_cast<float>(pair) / half_dim);
  float angle = position * frequency;
  float sine;
  float cosine;
  sincosf(angle, &sine, &cosine);

  if (projection_head < q_heads) {
    int source = token * fused_width + projection_head * head_dim;
    float first = __half2float(qkv[source + pair]);
    float second = __half2float(qkv[source + half_dim + pair]);
    if (bias != nullptr) {
      first += __half2float(bias[projection_head * head_dim + pair]);
      second += __half2float(bias[projection_head * head_dim + half_dim + pair]);
    }
    int destination = (token * q_heads + projection_head) * head_dim;
    q[destination + pair] = __float2half(first * cosine - second * sine);
    q[destination + half_dim + pair] = __float2half(second * cosine + first * sine);
  } else if (projection_head < q_heads + kv_heads) {
    int head = projection_head - q_heads;
    int source = token * fused_width + q_width + head * head_dim;
    float first = __half2float(qkv[source + pair]);
    float second = __half2float(qkv[source + half_dim + pair]);
    if (bias != nullptr) {
      first += __half2float(bias[q_width + head * head_dim + pair]);
      second += __half2float(bias[q_width + head * head_dim + half_dim + pair]);
    }
    int destination = ((kv_output_offset + token) * kv_heads + head) * head_dim;
    k[destination + pair] = __float2half(first * cosine - second * sine);
    k[destination + half_dim + pair] = __float2half(second * cosine + first * sine);
  } else {
    int head = projection_head - q_heads - kv_heads;
    int source = token * fused_width + q_width + kv_width + head * head_dim;
    int destination = ((kv_output_offset + token) * kv_heads + head) * head_dim;
    float first = __half2float(qkv[source + pair]);
    float second = __half2float(qkv[source + half_dim + pair]);
    if (bias != nullptr) {
      first += __half2float(bias[q_width + kv_width + head * head_dim + pair]);
      second += __half2float(bias[q_width + kv_width + head * head_dim + half_dim + pair]);
    }
    v[destination + pair] = __float2half(first);
    v[destination + half_dim + pair] = __float2half(second);
  }
}

__global__ void qkv_split_bias_f16_kernel(
    const half* qkv, const half* bias, half* q, half* k, half* v,
    int tokens, int projection_width) {
  const int token = blockIdx.x;
  const int fused_width = 3 * projection_width;
  for (int col = threadIdx.x; col < fused_width; col += blockDim.x) {
    float value = __half2float(qkv[token * fused_width + col]);
    if (bias != nullptr) value += __half2float(bias[col]);
    if (col < projection_width) {
      q[token * projection_width + col] = __float2half(value);
    } else if (col < 2 * projection_width) {
      k[token * projection_width + col - projection_width] = __float2half(value);
    } else {
      v[token * projection_width + col - 2 * projection_width] = __float2half(value);
    }
  }
}


__global__ void bias_residual_bf16_kernel(
    const __nv_bfloat16* projection, const __nv_bfloat16* bias,
    const __nv_bfloat16* residual, __nv_bfloat16* output,
    int64_t count, int cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    float value = __bfloat162float(projection[index]) +
                  __bfloat162float(residual[index]);
    if (bias != nullptr) value += __bfloat162float(bias[index % cols]);
    output[index] = __float2bfloat16(value);
  }
}

__global__ void bias_residual_rms_norm_bf16_kernel(
    const __nv_bfloat16* projection, const __nv_bfloat16* bias,
    const __nv_bfloat16* residual, const __nv_bfloat16* weight,
    __nv_bfloat16* hidden, __nv_bfloat16* normalized,
    int rows, int cols, float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    float value = __bfloat162float(projection[index]) +
                  __bfloat162float(residual[index]);
    if (bias != nullptr) value += __bfloat162float(bias[col]);
    const __nv_bfloat16 rounded = __float2bfloat16(value);
    hidden[index] = rounded;
    value = __bfloat162float(rounded);
    square_sum += value * value;
  }
  const float inverse_rms = rsqrtf(block_sum(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    normalized[index] = __float2bfloat16(
        __bfloat162float(hidden[index]) * inverse_rms * __bfloat162float(weight[col]));
  }
}

__global__ void bias_residual_layer_norm_bf16_kernel(
    const __nv_bfloat16* projection, const __nv_bfloat16* projection_bias,
    const __nv_bfloat16* residual, const __nv_bfloat16* norm_weight,
    const __nv_bfloat16* norm_bias, __nv_bfloat16* hidden,
    __nv_bfloat16* normalized, int rows, int cols, float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    float value = __bfloat162float(projection[index]) +
                  __bfloat162float(residual[index]);
    if (projection_bias != nullptr) value += __bfloat162float(projection_bias[col]);
    const __nv_bfloat16 rounded = __float2bfloat16(value);
    hidden[index] = rounded;
    sum += __bfloat162float(rounded);
  }
  const float mean = block_sum(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered =
        __bfloat162float(hidden[static_cast<int64_t>(row) * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  const float inverse_std = rsqrtf(block_sum(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float value =
        (__bfloat162float(hidden[index]) - mean) * inverse_std *
            __bfloat162float(norm_weight[col]) +
        __bfloat162float(norm_bias[col]);
    normalized[index] = __float2bfloat16(value);
  }
}

__global__ void ada_gate_residual_bf16_kernel(
    const __nv_bfloat16* projection, const __nv_bfloat16* residual,
    const __nv_bfloat16* style, __nv_bfloat16* output,
    int64_t count, int cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int col = static_cast<int>(index % cols);
    output[index] = __float2bfloat16(
        __bfloat162float(residual[index]) +
        __bfloat162float(projection[index]) *
            __bfloat162float(style[2 * cols + col]));
  }
}

__global__ void ada_gate_residual_rms_norm_bf16_kernel(
    const __nv_bfloat16* projection, const __nv_bfloat16* residual,
    const __nv_bfloat16* gate_style, const __nv_bfloat16* norm_style,
    __nv_bfloat16* hidden, __nv_bfloat16* normalized,
    int rows, int cols, float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const __nv_bfloat16 rounded = __float2bfloat16(
        __bfloat162float(residual[index]) +
        __bfloat162float(projection[index]) *
            __bfloat162float(gate_style[2 * cols + col]));
    hidden[index] = rounded;
    const float value = __bfloat162float(rounded);
    square_sum += value * value;
  }
  const float inverse_rms = rsqrtf(block_sum(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float value = __bfloat162float(hidden[index]) * inverse_rms;
    normalized[index] = __float2bfloat16(
        value * (1.0f + __bfloat162float(norm_style[col])) +
        __bfloat162float(norm_style[cols + col]));
  }
}

__global__ void qkv_rope_bf16_kernel(
    const __nv_bfloat16* qkv, const __nv_bfloat16* bias,
    __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v,
    int tokens, int q_heads, int kv_heads, int head_dim,
    float theta, int position_offset, int kv_output_offset) {
  const int token = blockIdx.x;
  const int projection_head = blockIdx.y;
  const int half_dim = head_dim / 2;
  const int pair = threadIdx.x;
  if (pair >= half_dim) return;
  const int q_width = q_heads * head_dim;
  const int kv_width = kv_heads * head_dim;
  const int fused_width = q_width + 2 * kv_width;
  const int position = position_offset + token;
  const float frequency = powf(theta, -static_cast<float>(pair) / half_dim);
  float sine, cosine;
  sincosf(position * frequency, &sine, &cosine);

  if (projection_head < q_heads) {
    const int source = token * fused_width + projection_head * head_dim;
    float first = __bfloat162float(qkv[source + pair]);
    float second = __bfloat162float(qkv[source + half_dim + pair]);
    if (bias != nullptr) {
      first += __bfloat162float(bias[projection_head * head_dim + pair]);
      second += __bfloat162float(bias[projection_head * head_dim + half_dim + pair]);
    }
    const int destination = (token * q_heads + projection_head) * head_dim;
    q[destination + pair] = __float2bfloat16(first * cosine - second * sine);
    q[destination + half_dim + pair] = __float2bfloat16(second * cosine + first * sine);
  } else if (projection_head < q_heads + kv_heads) {
    const int head = projection_head - q_heads;
    const int source = token * fused_width + q_width + head * head_dim;
    float first = __bfloat162float(qkv[source + pair]);
    float second = __bfloat162float(qkv[source + half_dim + pair]);
    if (bias != nullptr) {
      first += __bfloat162float(bias[q_width + head * head_dim + pair]);
      second += __bfloat162float(bias[q_width + head * head_dim + half_dim + pair]);
    }
    const int destination = ((kv_output_offset + token) * kv_heads + head) * head_dim;
    k[destination + pair] = __float2bfloat16(first * cosine - second * sine);
    k[destination + half_dim + pair] = __float2bfloat16(second * cosine + first * sine);
  } else {
    const int head = projection_head - q_heads - kv_heads;
    const int source = token * fused_width + q_width + kv_width + head * head_dim;
    const int destination = ((kv_output_offset + token) * kv_heads + head) * head_dim;
    float first = __bfloat162float(qkv[source + pair]);
    float second = __bfloat162float(qkv[source + half_dim + pair]);
    if (bias != nullptr) {
      first += __bfloat162float(bias[q_width + kv_width + head * head_dim + pair]);
      second += __bfloat162float(bias[q_width + kv_width + head * head_dim + half_dim + pair]);
    }
    v[destination + pair] = __float2bfloat16(first);
    v[destination + half_dim + pair] = __float2bfloat16(second);
  }
}

__global__ void qkv_split_bias_bf16_kernel(
    const __nv_bfloat16* qkv, const __nv_bfloat16* bias,
    __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v,
    int tokens, int projection_width) {
  const int token = blockIdx.x;
  const int fused_width = 3 * projection_width;
  for (int col = threadIdx.x; col < fused_width; col += blockDim.x) {
    float value = __bfloat162float(qkv[token * fused_width + col]);
    if (bias != nullptr) value += __bfloat162float(bias[col]);
    if (col < projection_width) {
      q[token * projection_width + col] = __float2bfloat16(value);
    } else if (col < 2 * projection_width) {
      k[token * projection_width + col - projection_width] = __float2bfloat16(value);
    } else {
      v[token * projection_width + col - 2 * projection_width] = __float2bfloat16(value);
    }
  }
}

__global__ void gqa_qkv_split_bias_bf16_kernel(
    const __nv_bfloat16* qkv, const __nv_bfloat16* bias,
    __nv_bfloat16* q, __nv_bfloat16* k, __nv_bfloat16* v,
    int tokens, int q_width, int kv_width) {
  const int token = blockIdx.x;
  const int fused_width = q_width + 2 * kv_width;
  for (int col = threadIdx.x; col < fused_width; col += blockDim.x) {
    float value = __bfloat162float(qkv[token * fused_width + col]);
    if (bias != nullptr) value += __bfloat162float(bias[col]);
    if (col < q_width) {
      q[token * q_width + col] = __float2bfloat16(value);
    } else if (col < q_width + kv_width) {
      k[token * kv_width + col - q_width] = __float2bfloat16(value);
    } else {
      v[token * kv_width + col - q_width - kv_width] = __float2bfloat16(value);
    }
  }
}
