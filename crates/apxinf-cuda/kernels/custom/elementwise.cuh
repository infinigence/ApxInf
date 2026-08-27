#pragma once

// Copyright 2026 apxinf contributors.
// Pure CUDA operators grouped by physical operation; launch policy lives under adapters/.

// ── Add ───────────────────────────────────────────────────────────────────

__global__ void add_f32_kernel(
    const float* a, const float* b, float* output, uint32_t count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    output[gid] = a[gid] + b[gid];
}



// ── Mul ───────────────────────────────────────────────────────────────────

__global__ void mul_f32_kernel(
    const float* a, const float* b, float* output, uint32_t count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    output[gid] = a[gid] * b[gid];
}



// ── Scale (element-wise multiply by scalar, no sync) ──────────────────────

__global__ void scale_f32_kernel(
    const float* input, float* output, uint32_t count, float scale)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    output[gid] = input[gid] * scale;
}


// ── Add (bf16) ────────────────────────────────────────────────────────────

__global__ void add_bf16_kernel(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* output, uint32_t count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    float x = __bfloat162float(a[gid]) + __bfloat162float(b[gid]);
    output[gid] = __float2bfloat16(x);
}



// ── Mul (bf16) ────────────────────────────────────────────────────────────

__global__ void mul_bf16_kernel(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* output, uint32_t count)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    float x = __bfloat162float(a[gid]) * __bfloat162float(b[gid]);
    output[gid] = __float2bfloat16(x);
}


// Multiply a matrix by the gate third of one packed `[scale, shift, gate]`
// style row. The BF16 output is deliberately materialized so a following
// residual add observes the same dtype boundary as two unfused tensor ops.
__global__ void style_gate_mul_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* style,
    __nv_bfloat16* output, int64_t count, int cols) {
  int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int col = static_cast<int>(index % cols);
    output[index] = __float2bfloat16_rn(
        __bfloat162float(input[index]) *
        __bfloat162float(style[2 * cols + col]));
  }
}



// ── Scale (bf16) ──────────────────────────────────────────────────────────

__global__ void scale_bf16_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output, uint32_t count, float scale)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    float x = __bfloat162float(input[gid]) * scale;
    output[gid] = __float2bfloat16(x);
}



// ── Add-bias (bf16) — broadcast bias vector over rows ────────────────────
//
// out[r, c] = input[r, c] + bias[c]. Used after each linear projection in
// the vision tower (qkv, o_proj, fc1, fc2, and the mergers all have bias).

__global__ void add_bias_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* bias,
    __nv_bfloat16* output, uint32_t cols, uint32_t rows)
{
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y;
    if (col >= cols || row >= rows) return;
    float x = __bfloat162float(input[row * cols + col]);
    float b = __bfloat162float(bias[col]);
    output[row * cols + col] = __float2bfloat16(x + b);
}




__global__ void bias_f16_kernel(
    const half* input, const half* bias, half* output, int64_t count, int cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    float value = __half2float(input[index]);
    if (bias != nullptr) value += __half2float(bias[index % cols]);
    output[index] = __float2half(value);
  }
}

__global__ void concat_rows_f16_kernel(
    const half* first, const half* second, half* output,
    int64_t first_count, int64_t total_count) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < total_count; index += stride) {
    output[index] = index < first_count ? first[index] : second[index - first_count];
  }
}

__global__ void euler_update_f16_kernel(
    const half* state, const half* velocity, half* output,
    int64_t count, float dt) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    output[index] = __float2half(
        __half2float(state[index]) + dt * __half2float(velocity[index]));
  }
}

__global__ void bias_position_f16_kernel(
    const half* projection, const half* bias, const half* position,
    half* output, int64_t count, int cols, int tokens_per_view) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    int col = static_cast<int>(index % cols);
    int token = static_cast<int>((index / cols) % tokens_per_view);
    float value = __half2float(projection[index]) +
                  __half2float(position[token * cols + col]);
    if (bias != nullptr) value += __half2float(bias[col]);
    output[index] = __float2half(value);
  }
}

__global__ void concat_rows_bf16_kernel(
    const __nv_bfloat16* first, const __nv_bfloat16* second,
    __nv_bfloat16* output, int64_t first_count, int64_t total_count) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < total_count; index += stride) {
    output[index] = index < first_count ? first[index] : second[index - first_count];
  }
}

__global__ void euler_update_bf16_kernel(
    const __nv_bfloat16* state, const __nv_bfloat16* velocity,
    __nv_bfloat16* output, int64_t count, float dt) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    output[index] = __float2bfloat16(
        __bfloat162float(state[index]) + dt * __bfloat162float(velocity[index]));
  }
}

__global__ void bias_position_bf16_kernel(
    const __nv_bfloat16* projection, const __nv_bfloat16* bias,
    const __nv_bfloat16* position, __nv_bfloat16* output,
    int64_t count, int cols, int tokens_per_view) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int col = static_cast<int>(index % cols);
    const int token = static_cast<int>((index / cols) % tokens_per_view);
    float value = __bfloat162float(projection[index]) +
                  __bfloat162float(position[token * cols + col]);
    if (bias != nullptr) value += __bfloat162float(bias[col]);
    output[index] = __float2bfloat16(value);
  }
}

