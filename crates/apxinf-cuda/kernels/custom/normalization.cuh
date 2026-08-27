#pragma once

// Copyright 2026 apxinf contributors.
// Pure CUDA operators grouped by physical operation; launch policy lives under adapters/.

// ── RMSNorm ──────────────────────────────────────────────────────────────

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



// ── RMSNorm (bf16) ────────────────────────────────────────────────────────
//
// Shared-memory reduction: one block per row, strided load, warp-shuffle
// reduction. Each input element is read from HBM exactly once (the naive
// per-thread loop did O(cols²) reads — catastrophic on Thor's 14-SM GPU
// where a 2048-wide row launched only 8 blocks and each thread re-read the
// whole row). Mirrors rms_norm_add_bf16's reduction minus the residual.

__global__ void rms_norm_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, __nv_bfloat16* output,
    uint32_t cols, uint32_t rows, float eps)
{
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    uint32_t tid = threadIdx.x;
    uint32_t offset = row * cols;

    // Cache the row in fp32 shared memory so the normalize phase doesn't
    // re-read HBM. cols * sizeof(float) bytes (8 KB for cols=2048 — fits the
    // 48 KB per-block limit).
    extern __shared__ float x_buf[];
    __shared__ float s_sum;

    // Phase 1: strided load; each thread accumulates a partial sum_sq.
    float partial = 0.0f;
    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        float v = __bfloat162float(input[offset + i]);
        x_buf[i] = v;
        partial += v * v;
    }

    // Phase 2: warp-shuffle reduction within each warp, then across warps.
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

    // Phase 3: write the normed output from shared memory.
    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        float w = __bfloat162float(weight[i]);
        output[offset + i] = __float2bfloat16(x_buf[i] * rms * w);
    }
}



// ── LayerNorm (bf16) — Qwen3-VL vision tower ─────────────────────────────
//
// mean+variance normalization with affine transform: out = w * (x - mean) /
// sqrt(var + eps) + b. Same layout convention as rms_norm: `[rows, cols]`
// with normalization along the last axis. Vision blocks have both weight
// and bias, unlike the text stack's RMSNorm.

__global__ void layer_norm_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, __nv_bfloat16* output,
    uint32_t cols, uint32_t rows, float eps)
{
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (col >= cols || row >= rows) return;

    uint32_t offset = row * cols;
    float sum = 0.0f;
    for (uint32_t i = 0; i < cols; i++) {
        sum += __bfloat162float(input[offset + i]);
    }
    float mean = sum / (float)cols;
    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < cols; i++) {
        float d = __bfloat162float(input[offset + i]) - mean;
        sum_sq += d * d;
    }
    float inv_std = rsqrtf(sum_sq / (float)cols + eps);

    float x = __bfloat162float(input[offset + col]);
    float w = __bfloat162float(weight[col]);
    float b = __bfloat162float(bias[col]);
    output[offset + col] = __float2bfloat16(w * (x - mean) * inv_std + b);
}




__global__ void rms_norm_quant_f16_e4m3_kernel(
    const half* input, const half* weight, __nv_fp8_e4m3* output,
    int rows, int cols, float eps, float inverse_scale) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = __half2float(input[row * cols + col]);
    square_sum += value * value;
  }
  square_sum = block_sum(square_sum, scratch);
  float inverse_rms = rsqrtf(square_sum / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = __half2float(input[row * cols + col]) * inverse_rms *
                  __half2float(weight[col]);
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    output[row * cols + col] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void layer_norm_quant_f16_e4m3_kernel(
    const half* input, const half* weight, const half* bias,
    __nv_fp8_e4m3* output, int rows, int cols, float eps,
    float inverse_scale) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    sum += __half2float(input[row * cols + col]);
  float mean = block_sum(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float centered = __half2float(input[row * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  float inverse_std = rsqrtf(block_sum(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = (__half2float(input[row * cols + col]) - mean) * inverse_std;
    value = value * __half2float(weight[col]) + __half2float(bias[col]);
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    output[row * cols + col] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void ada_rms_norm_quant_f16_e4m3_kernel(
    const half* input, const half* style, __nv_fp8_e4m3* output,
    int rows, int cols, float eps, float inverse_scale) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = __half2float(input[row * cols + col]);
    square_sum += value * value;
  }
  float inverse_rms = rsqrtf(block_sum(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float normalized = __half2float(input[row * cols + col]) * inverse_rms;
    float scale = __half2float(style[col]);
    float shift = __half2float(style[cols + col]);
    float value = normalized * (1.0f + scale) + shift;
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    output[row * cols + col] = static_cast<__nv_fp8_e4m3>(value);
  }
}


__global__ void rms_norm_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, int rows, int cols, float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = __bfloat162float(input[static_cast<int64_t>(row) * cols + col]);
    square_sum += value * value;
  }
  const float inverse_rms = rsqrtf(block_sum(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    output[index] = __float2bfloat16(
        __bfloat162float(input[index]) * inverse_rms * __bfloat162float(weight[col]));
  }
}


// RMSNorm with an explicit FP32 affine offset. The raw checkpoint weight is
// kept in BF16 and promoted before `offset + weight`, so callers can preserve
// conventions such as Gemma's `(1 + weight.float())` without rounding the
// combined scale back to BF16 before the normalized multiply.
__global__ void rms_norm_affine_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* raw_weight,
    __nv_bfloat16* output, int rows, int cols, float eps,
    float weight_offset) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value =
        __bfloat162float(input[static_cast<int64_t>(row) * cols + col]);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float scale =
        weight_offset + __bfloat162float(raw_weight[col]);
    output[index] = __float2bfloat16_rn(
        __bfloat162float(input[index]) * inverse_rms * scale);
  }
}

__global__ void layer_norm_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, __nv_bfloat16* output,
    int rows, int cols, float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    sum += __bfloat162float(input[static_cast<int64_t>(row) * cols + col]);
  const float mean = block_sum(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered =
        __bfloat162float(input[static_cast<int64_t>(row) * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  const float inverse_std = rsqrtf(block_sum(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float value =
        (__bfloat162float(input[index]) - mean) * inverse_std *
            __bfloat162float(weight[col]) +
        __bfloat162float(bias[col]);
    output[index] = __float2bfloat16(value);
  }
}

__global__ void ada_rms_norm_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* style,
    __nv_bfloat16* output, int rows, int cols, float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = __bfloat162float(input[static_cast<int64_t>(row) * cols + col]);
    square_sum += value * value;
  }
  const float inverse_rms = rsqrtf(block_sum(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float normalized = __bfloat162float(input[index]) * inverse_rms;
    output[index] = __float2bfloat16(
        normalized * (1.0f + __bfloat162float(style[col])) +
        __bfloat162float(style[cols + col]));
  }
}


