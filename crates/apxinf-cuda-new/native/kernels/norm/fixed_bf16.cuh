// Copyright 2026 ApxInf contributors.
// Fixed BF16 implementation for this operator family; ported without numerical changes.
#include "../custom/reduction.cuh"
#ifdef APXINF_GEMM_CUTLASS_SM87_W8A8
#include "../cutlass/ops/gemm/gemm_i8_bf16_sm80.h"
#endif

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kBlockSize = 256;

__device__ __forceinline__ float gelu_tanh(float value) {
  constexpr float kAlpha = 0.7978845608028654f;
  return 0.5f * value *
         (1.0f + tanhf(kAlpha * (value + 0.044715f * value * value * value)));
}

struct alignas(8) Bf16x4 {
  __nv_bfloat162 low;
  __nv_bfloat162 high;
};

struct alignas(8) W8Bf16x4 {
  __nv_bfloat162 low;
  __nv_bfloat162 high;
};

struct alignas(16) Bf16Pairx8 {
  Bf16x4 first;
  Bf16x4 second;
};

// ── Norm kernels (BF16-input variants) ───────────────────────────────────
// cuda-new's own norm kernels take F16 input for the E4M3-output forms; the
// This path needs BF16-input versions, so these are copied verbatim from the
// legacy normalization.cuh. `block_sum_parallel_unsafe` is cuda-new's own
// reduction helper (reduction.cuh), identical to the legacy one.

__global__ void adaptive_layer_norm_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* modulation,
    __nv_bfloat16* output, uint32_t rows, uint32_t cols, float eps) {
  __shared__ float scratch[16];
  const uint32_t row = blockIdx.x;
  if (row >= rows) return;

  float sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x)
    sum += __bfloat162float(input[(uint64_t)row * cols + col]);
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;

  float variance_sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered =
        __bfloat162float(input[(uint64_t)row * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);

  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const uint64_t index = (uint64_t)row * cols + col;
    const float normalized =
        (__bfloat162float(input[index]) - mean) * inverse_std;
    const float scale = __bfloat162float(modulation[col]);
    const float shift = __bfloat162float(modulation[cols + col]);
    output[index] = __float2bfloat16(normalized * (1.0f + scale) + shift);
  }
}

__global__ void adaptive_layer_norm_quant_bf16_e4m3_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* modulation,
    __nv_bfloat16* output, __nv_fp8_e4m3* quantized, uint32_t rows,
    uint32_t cols, float eps, float inverse_scale) {
  __shared__ float scratch[16];
  const uint32_t row = blockIdx.x;
  if (row >= rows) return;

  float sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x)
    sum += __bfloat162float(input[(uint64_t)row * cols + col]);
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;

  float variance_sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered =
        __bfloat162float(input[(uint64_t)row * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);

  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const uint64_t index = (uint64_t)row * cols + col;
    const float normalized =
        (__bfloat162float(input[index]) - mean) * inverse_std;
    const float scale = __bfloat162float(modulation[col]);
    const float shift = __bfloat162float(modulation[cols + col]);
    const __nv_bfloat16 rounded =
        __float2bfloat16(normalized * (1.0f + scale) + shift);
    output[index] = rounded;
    float value = __bfloat162float(rounded) * inverse_scale;
    value = fminf(448.0f, fmaxf(-448.0f, value));
    quantized[index] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void rms_norm_quant_bf16_e4m3_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_fp8_e4m3* output, int rows, int cols, float eps,
    float inverse_scale) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value =
        __bfloat162float(input[static_cast<int64_t>(row) * cols + col]);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    float value = __bfloat162float(input[index]) * inverse_rms *
                  __bfloat162float(weight[col]) * inverse_scale;
    value = fminf(448.0f, fmaxf(-448.0f, value));
    output[index] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void layer_norm_quant_bf16_e4m3_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, __nv_fp8_e4m3* output, int rows, int cols,
    float eps, float inverse_scale) {
  extern __shared__ float x_buf[];
  __shared__ float scratch[16];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int offset = row * cols;

  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = __bfloat162float(input[offset + col]);
    x_buf[col] = value;
    sum += value;
  }
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;

  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered = x_buf[col] - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);

  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = (x_buf[col] - mean) * inverse_std;
    value = value * __bfloat162float(weight[col]) +
            __bfloat162float(bias[col]);
    value = __bfloat162float(__float2bfloat16(value));
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    output[offset + col] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void layer_norm_quantize_rows_bf16_int8_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, __nv_bfloat16* output,
    int8_t* quantized, float* scales, int rows, int cols, float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int64_t base = static_cast<int64_t>(row) * cols;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    sum += __bfloat162float(input[base + col]);
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered = __bfloat162float(input[base + col]) - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);
  float maximum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = base + col;
    const float value =
        (__bfloat162float(input[index]) - mean) * inverse_std *
            __bfloat162float(weight[col]) +
        __bfloat162float(bias[col]);
    const __nv_bfloat16 rounded = __float2bfloat16(value);
    output[index] = rounded;
    maximum = fmaxf(maximum, fabsf(__bfloat162float(rounded)));
  }
  __syncthreads();
  const float scale =
      fmaxf(block_max_parallel_unsafe(maximum, scratch) / 127.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = scale;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = base + col;
    const float value = roundf(__bfloat162float(output[index]) / scale);
    quantized[index] =
        static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, value)));
  }
}

__global__ void adaptive_layer_norm_quantize_rows_bf16_int8_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* modulation,
    __nv_bfloat16* output, int8_t* quantized, float* scales,
    int rows, int cols, float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int64_t base = static_cast<int64_t>(row) * cols;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    sum += __bfloat162float(input[base + col]);
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered = __bfloat162float(input[base + col]) - mean;
    variance_sum += centered * centered;
  }
  // Finish reading the previous reduction before reusing scratch.
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);
  float maximum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float normalized =
        (__bfloat162float(input[base + col]) - mean) * inverse_std;
    const float scale = __bfloat162float(modulation[col]);
    const float shift = __bfloat162float(modulation[cols + col]);
    const __nv_bfloat16 rounded =
        __float2bfloat16(normalized * (1.0f + scale) + shift);
    output[base + col] = rounded;
    maximum = fmaxf(maximum, fabsf(__bfloat162float(rounded)));
  }
  // Finish reading the previous reduction before reusing scratch.
  __syncthreads();
  const float row_scale =
      fmaxf(block_max_parallel_unsafe(maximum, scratch) / 127.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = row_scale;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = roundf(__bfloat162float(output[base + col]) / row_scale);
    quantized[base + col] =
        static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, value)));
  }
}


}  // namespace

// ── Norm launchers ───────────────────────────────────────────────────────
