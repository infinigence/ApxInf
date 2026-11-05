// Copyright 2026 ApxInf contributors.
//
// Fused rowwise-quantizing epilogues for the walloss family, ported
// bit-identically from the legacy kernels. See `quant_ops.h`.
//
// Each launcher reproduces the legacy dispatch exactly: a small-row
// warp-per-row form and a vec8/vec4 wide form (which multiplies by `1/scale`
// rather than dividing, a distinct rounding), plus a full-block-per-row form
// for high row counts. The variants differ only in how the row is distributed
// across threads, so picking the wrong one is a 1-ULP fidelity loss, not a
// correctness failure -- which is exactly why they are all carried over.

#include "quant_ops.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace apxinf::cuda_new::quant_ops {
namespace {

struct alignas(16) Bf16Pack8 {
  __nv_bfloat16 values[8];
};
struct alignas(8) Bf16Pack4 {
  __nv_bfloat16 values[4];
};
struct alignas(8) Fp8Pack8 {
  __nv_fp8_e4m3 values[8];
};
struct alignas(4) Fp8Pack4 {
  __nv_fp8_e4m3 values[4];
};

__device__ __forceinline__ float warp_sum_all(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(0xffffffff, value, offset);
  return value;
}

__device__ __forceinline__ float warp_max(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
  return value;
}

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
}

// Block-wide sum; same unsafe scratch contract as the legacy helper.
__device__ __forceinline__ float block_sum_parallel_unsafe(float value,
                                                           float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  value = warp_sum(value);
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < warps ? scratch[lane] : 0.0f;
    value = warp_sum(value);
    if (lane == 0) scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}

// Block-wide maximum; same unsafe scratch contract as the legacy helper.
__device__ __forceinline__ float block_max_parallel_unsafe(float value,
                                                           float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  value = warp_max(value);
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < warps ? scratch[lane] : -INFINITY;
    value = warp_max(value);
    if (lane == 0) scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}

// ── rms_norm + rowwise E4M3 ────────────────────────────────────────────────

__global__ void rms_norm_quantize_rows_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight, __nv_fp8_e4m3* __restrict__ output,
    float* __restrict__ scales, int rows, int input_cols, int output_cols,
    float eps) {
  constexpr int kWarpsPerBlock = 8;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kWarpsPerBlock + warp;
  if (row >= rows) return;

  const int64_t input_offset = static_cast<int64_t>(row) * input_cols;
  const int64_t output_offset = static_cast<int64_t>(row) * output_cols;
  float square_sum = 0.0f;
  for (int col = lane; col < input_cols; col += 32) {
    const float value = __bfloat162float(input[input_offset + col]);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(warp_sum_all(square_sum) / static_cast<float>(input_cols) + eps);

  float maximum = 0.0f;
  for (int col = lane; col < input_cols; col += 32) {
    const float value = __bfloat162float(input[input_offset + col]) *
                        inverse_rms * __bfloat162float(weight[col]);
    maximum = fmaxf(maximum, fabsf(value));
  }
  const float scale = fmaxf(warp_max(maximum) / 448.0f, 1.0e-12f);
  if (lane == 0) scales[row] = scale;

  for (int col = lane; col < output_cols; col += 32) {
    float value = 0.0f;
    if (col < input_cols) {
      value = __bfloat162float(input[input_offset + col]) * inverse_rms *
              __bfloat162float(weight[col]) / scale;
      value = fminf(448.0f, fmaxf(-448.0f, value));
    }
    output[output_offset + col] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void rms_norm_quantize_rows_vec8_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_fp8_e4m3* output, float* scales, int rows, int input_cols,
    int output_cols, float eps) {
  constexpr int kRowsPerBlock = 8;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kRowsPerBlock + warp;
  if (row >= rows) return;

  const int input_vectors = input_cols / 8;
  const int output_vectors = output_cols / 8;
  const int64_t input_offset = static_cast<int64_t>(row) * input_cols;
  const int64_t output_offset = static_cast<int64_t>(row) * output_cols;
  float square_sum = 0.0f;
  for (int vector = lane; vector < input_vectors; vector += 32) {
    const Bf16Pack8 values =
        *reinterpret_cast<const Bf16Pack8*>(input + input_offset + vector * 8);
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      const float value = __bfloat162float(values.values[item]);
      square_sum += value * value;
    }
  }
  const float inverse_rms = rsqrtf(
      warp_sum_all(square_sum) / static_cast<float>(input_cols) + eps);

  float maximum = 0.0f;
  for (int vector = lane; vector < input_vectors; vector += 32) {
    const Bf16Pack8 values =
        *reinterpret_cast<const Bf16Pack8*>(input + input_offset + vector * 8);
    const Bf16Pack8 weights =
        *reinterpret_cast<const Bf16Pack8*>(weight + vector * 8);
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      const float value = __bfloat162float(values.values[item]) * inverse_rms *
                          __bfloat162float(weights.values[item]);
      maximum = fmaxf(maximum, fabsf(value));
    }
  }
  const float scale = fmaxf(warp_max(maximum) / 448.0f, 1.0e-12f);
  const float inverse_scale = 1.0f / scale;
  if (lane == 0) scales[row] = scale;

  for (int vector = lane; vector < output_vectors; vector += 32) {
    Fp8Pack8 quantized{};
    if (vector < input_vectors) {
      const Bf16Pack8 values = *reinterpret_cast<const Bf16Pack8*>(
          input + input_offset + vector * 8);
      const Bf16Pack8 weights =
          *reinterpret_cast<const Bf16Pack8*>(weight + vector * 8);
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        float value = __bfloat162float(values.values[item]) * inverse_rms *
                      __bfloat162float(weights.values[item]) * inverse_scale;
        value = fminf(448.0f, fmaxf(-448.0f, value));
        quantized.values[item] = static_cast<__nv_fp8_e4m3>(value);
      }
    }
    *reinterpret_cast<Fp8Pack8*>(output + output_offset + vector * 8) = quantized;
  }
}

// ── bias + residual + rms_norm + rowwise E4M3 ──────────────────────────────

__global__ void bias_residual_rms_norm_quantize_rows_kernel(
    const __nv_bfloat16* __restrict__ projection,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight, __nv_bfloat16* __restrict__ hidden,
    __nv_fp8_e4m3* __restrict__ normalized, float* __restrict__ scales,
    int rows, int cols, int output_cols, float eps) {
  constexpr int kWarpsPerBlock = 8;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kWarpsPerBlock + warp;
  if (row >= rows) return;

  const int64_t hidden_offset = static_cast<int64_t>(row) * cols;
  const int64_t output_offset = static_cast<int64_t>(row) * output_cols;
  float square_sum = 0.0f;
  for (int col = lane; col < cols; col += 32) {
    const int64_t index = hidden_offset + col;
    float value = __bfloat162float(projection[index]) +
                  __bfloat162float(residual[index]);
    if (bias != nullptr) value += __bfloat162float(bias[col]);
    const __nv_bfloat16 rounded = __float2bfloat16(value);
    hidden[index] = rounded;
    value = __bfloat162float(rounded);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(warp_sum_all(square_sum) / static_cast<float>(cols) + eps);

  float maximum = 0.0f;
  for (int col = lane; col < cols; col += 32) {
    const float value = __bfloat162float(hidden[hidden_offset + col]) *
                        inverse_rms * __bfloat162float(weight[col]);
    maximum = fmaxf(maximum, fabsf(value));
  }
  const float scale = fmaxf(warp_max(maximum) / 448.0f, 1.0e-12f);
  if (lane == 0) scales[row] = scale;

  for (int col = lane; col < output_cols; col += 32) {
    float value = 0.0f;
    if (col < cols) {
      value = __bfloat162float(hidden[hidden_offset + col]) * inverse_rms *
              __bfloat162float(weight[col]) / scale;
      value = fminf(448.0f, fmaxf(-448.0f, value));
    }
    normalized[output_offset + col] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void bias_residual_rms_norm_quantize_rows_vec8_kernel(
    const __nv_bfloat16* projection, const __nv_bfloat16* bias,
    const __nv_bfloat16* residual, const __nv_bfloat16* weight,
    __nv_bfloat16* hidden, __nv_fp8_e4m3* normalized, float* scales,
    int rows, int cols, int output_cols, float eps) {
  constexpr int kRowsPerBlock = 8;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kRowsPerBlock + warp;
  if (row >= rows) return;

  const int hidden_vectors = cols / 8;
  const int output_vectors = output_cols / 8;
  const int64_t hidden_offset = static_cast<int64_t>(row) * cols;
  const int64_t output_offset = static_cast<int64_t>(row) * output_cols;
  float square_sum = 0.0f;
  for (int vector = lane; vector < hidden_vectors; vector += 32) {
    const Bf16Pack8 projected = *reinterpret_cast<const Bf16Pack8*>(
        projection + hidden_offset + vector * 8);
    const Bf16Pack8 residual_values = *reinterpret_cast<const Bf16Pack8*>(
        residual + hidden_offset + vector * 8);
    Bf16Pack8 bias_values{};
    if (bias != nullptr) {
      bias_values = *reinterpret_cast<const Bf16Pack8*>(bias + vector * 8);
    }
    Bf16Pack8 rounded_values;
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      float value = __bfloat162float(projected.values[item]) +
                    __bfloat162float(residual_values.values[item]);
      if (bias != nullptr) {
        value += __bfloat162float(bias_values.values[item]);
      }
      const __nv_bfloat16 rounded = __float2bfloat16(value);
      rounded_values.values[item] = rounded;
      value = __bfloat162float(rounded);
      square_sum += value * value;
    }
    *reinterpret_cast<Bf16Pack8*>(hidden + hidden_offset + vector * 8) =
        rounded_values;
  }
  const float inverse_rms = rsqrtf(
      warp_sum_all(square_sum) / static_cast<float>(cols) + eps);

  float maximum = 0.0f;
  for (int vector = lane; vector < hidden_vectors; vector += 32) {
    const Bf16Pack8 hidden_values = *reinterpret_cast<const Bf16Pack8*>(
        hidden + hidden_offset + vector * 8);
    const Bf16Pack8 weights =
        *reinterpret_cast<const Bf16Pack8*>(weight + vector * 8);
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      const float value = __bfloat162float(hidden_values.values[item]) *
                          inverse_rms *
                          __bfloat162float(weights.values[item]);
      maximum = fmaxf(maximum, fabsf(value));
    }
  }
  const float scale = fmaxf(warp_max(maximum) / 448.0f, 1.0e-12f);
  const float inverse_scale = 1.0f / scale;
  if (lane == 0) scales[row] = scale;

  for (int vector = lane; vector < output_vectors; vector += 32) {
    Fp8Pack8 quantized{};
    if (vector < hidden_vectors) {
      const Bf16Pack8 hidden_values = *reinterpret_cast<const Bf16Pack8*>(
          hidden + hidden_offset + vector * 8);
      const Bf16Pack8 weights =
          *reinterpret_cast<const Bf16Pack8*>(weight + vector * 8);
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        float value = __bfloat162float(hidden_values.values[item]) *
                      inverse_rms * __bfloat162float(weights.values[item]) *
                      inverse_scale;
        value = fminf(448.0f, fmaxf(-448.0f, value));
        quantized.values[item] = static_cast<__nv_fp8_e4m3>(value);
      }
    }
    *reinterpret_cast<Fp8Pack8*>(normalized + output_offset + vector * 8) =
        quantized;
  }
}

// High-row-count companion: a full block cooperates on one row, which exposes
// enough parallelism for vision and language prefill widths.
__global__ void bias_residual_rms_norm_quantize_rows_large_vec8_kernel(
    const __nv_bfloat16* projection, const __nv_bfloat16* bias,
    const __nv_bfloat16* residual, const __nv_bfloat16* weight,
    __nv_bfloat16* hidden, __nv_fp8_e4m3* normalized, float* scales,
    int rows, int cols, int output_cols, float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  if (row >= rows) return;

  const int hidden_vectors = cols / 8;
  const int output_vectors = output_cols / 8;
  const int64_t hidden_offset = static_cast<int64_t>(row) * cols;
  const int64_t output_offset = static_cast<int64_t>(row) * output_cols;
  float square_sum = 0.0f;
  for (int vector = threadIdx.x; vector < hidden_vectors;
       vector += blockDim.x) {
    const Bf16Pack8 projected = *reinterpret_cast<const Bf16Pack8*>(
        projection + hidden_offset + vector * 8);
    const Bf16Pack8 residual_values = *reinterpret_cast<const Bf16Pack8*>(
        residual + hidden_offset + vector * 8);
    Bf16Pack8 bias_values{};
    if (bias != nullptr) {
      bias_values = *reinterpret_cast<const Bf16Pack8*>(bias + vector * 8);
    }
    Bf16Pack8 rounded_values;
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      float value = __bfloat162float(projected.values[item]) +
                    __bfloat162float(residual_values.values[item]);
      if (bias != nullptr) {
        value += __bfloat162float(bias_values.values[item]);
      }
      const __nv_bfloat16 rounded = __float2bfloat16(value);
      rounded_values.values[item] = rounded;
      value = __bfloat162float(rounded);
      square_sum += value * value;
    }
    *reinterpret_cast<Bf16Pack8*>(hidden + hidden_offset + vector * 8) =
        rounded_values;
  }
  const float inverse_rms = rsqrtf(
      block_sum_parallel_unsafe(square_sum, scratch) / static_cast<float>(cols) +
      eps);

  float maximum = 0.0f;
  for (int vector = threadIdx.x; vector < hidden_vectors;
       vector += blockDim.x) {
    const Bf16Pack8 hidden_values = *reinterpret_cast<const Bf16Pack8*>(
        hidden + hidden_offset + vector * 8);
    const Bf16Pack8 weights =
        *reinterpret_cast<const Bf16Pack8*>(weight + vector * 8);
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      const float value = __bfloat162float(hidden_values.values[item]) *
                          inverse_rms *
                          __bfloat162float(weights.values[item]);
      maximum = fmaxf(maximum, fabsf(value));
    }
  }
  // Finish reading the previous reduction before reusing scratch.
  __syncthreads();
  const float scale =
      fmaxf(block_max_parallel_unsafe(maximum, scratch) / 448.0f, 1.0e-12f);
  const float inverse_scale = 1.0f / scale;
  if (threadIdx.x == 0) scales[row] = scale;

  for (int vector = threadIdx.x; vector < output_vectors;
       vector += blockDim.x) {
    Fp8Pack8 quantized{};
    if (vector < hidden_vectors) {
      const Bf16Pack8 hidden_values = *reinterpret_cast<const Bf16Pack8*>(
          hidden + hidden_offset + vector * 8);
      const Bf16Pack8 weights =
          *reinterpret_cast<const Bf16Pack8*>(weight + vector * 8);
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        float value = __bfloat162float(hidden_values.values[item]) *
                      inverse_rms * __bfloat162float(weights.values[item]) *
                      inverse_scale;
        value = fminf(448.0f, fmaxf(-448.0f, value));
        quantized.values[item] = static_cast<__nv_fp8_e4m3>(value);
      }
    }
    *reinterpret_cast<Fp8Pack8*>(normalized + output_offset + vector * 8) =
        quantized;
  }
}

__global__ void bias_residual_rms_norm_quantize_rows_large_kernel(
    const __nv_bfloat16* projection, const __nv_bfloat16* bias,
    const __nv_bfloat16* residual, const __nv_bfloat16* weight,
    __nv_bfloat16* hidden, __nv_fp8_e4m3* normalized, float* scales,
    int rows, int cols, int output_cols, float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int64_t hidden_offset = static_cast<int64_t>(row) * cols;
  const int64_t output_offset = static_cast<int64_t>(row) * output_cols;

  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = hidden_offset + col;
    float value = __bfloat162float(projection[index]) +
                  __bfloat162float(residual[index]);
    if (bias != nullptr) value += __bfloat162float(bias[col]);
    const __nv_bfloat16 rounded = __float2bfloat16(value);
    hidden[index] = rounded;
    value = __bfloat162float(rounded);
    square_sum += value * value;
  }
  const float inverse_rms = rsqrtf(
      block_sum_parallel_unsafe(square_sum, scratch) / static_cast<float>(cols) +
      eps);

  float maximum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = __bfloat162float(hidden[hidden_offset + col]) *
                        inverse_rms * __bfloat162float(weight[col]);
    maximum = fmaxf(maximum, fabsf(value));
  }
  // Finish reading the previous reduction before reusing scratch.
  __syncthreads();
  const float scale =
      fmaxf(block_max_parallel_unsafe(maximum, scratch) / 448.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = scale;

  for (int col = threadIdx.x; col < output_cols; col += blockDim.x) {
    float value = 0.0f;
    if (col < cols) {
      value = __bfloat162float(hidden[hidden_offset + col]) * inverse_rms *
              __bfloat162float(weight[col]) / scale;
      value = fminf(448.0f, fmaxf(-448.0f, value));
    }
    normalized[output_offset + col] = static_cast<__nv_fp8_e4m3>(value);
  }
}

// ── SwiGLU + rowwise E4M3 ──────────────────────────────────────────────────

__global__ void swiglu_quantize_rows_kernel(
    const __nv_bfloat16* __restrict__ gate_up,
    const __nv_bfloat16* __restrict__ bias, __nv_fp8_e4m3* __restrict__ output,
    float* __restrict__ scales, int rows, int input_cols, int inner,
    int output_cols) {
  __shared__ float scratch[8];
  extern __shared__ float activated[];
  const int row = blockIdx.x;
  if (row >= rows) return;

  const int64_t input_offset = static_cast<int64_t>(row) * input_cols;
  const int64_t output_offset = static_cast<int64_t>(row) * output_cols;
  float maximum = 0.0f;
  for (int col = threadIdx.x; col < inner; col += blockDim.x) {
    float gate = __bfloat162float(gate_up[input_offset + col]);
    float up = __bfloat162float(gate_up[input_offset + inner + col]);
    if (bias != nullptr) {
      gate += __bfloat162float(bias[col]);
      up += __bfloat162float(bias[inner + col]);
    }
    const float value = (gate / (1.0f + expf(-gate))) * up;
    activated[col] = value;
    maximum = fmaxf(maximum, fabsf(value));
  }
  const float scale =
      fmaxf(block_max_parallel_unsafe(maximum, scratch) / 448.0f, 1.0e-12f);
  if (threadIdx.x == 0) scales[row] = scale;

  for (int col = threadIdx.x; col < output_cols; col += blockDim.x) {
    float value = 0.0f;
    if (col < inner) {
      value = activated[col] / scale;
      value = fminf(448.0f, fmaxf(-448.0f, value));
    }
    output[output_offset + col] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void swiglu_quantize_rows_vec8_kernel(
    const __nv_bfloat16* gate_up, const __nv_bfloat16* bias,
    __nv_fp8_e4m3* output, float* scales, int rows, int input_cols,
    int inner, int output_cols) {
  __shared__ float scratch[8];
  extern __shared__ float activated[];
  const int row = blockIdx.x;
  if (row >= rows) return;

  const int inner_vectors = inner / 8;
  const int output_vectors = output_cols / 8;
  const int64_t input_offset = static_cast<int64_t>(row) * input_cols;
  const int64_t output_offset = static_cast<int64_t>(row) * output_cols;
  float maximum = 0.0f;
  for (int vector = threadIdx.x; vector < inner_vectors;
       vector += blockDim.x) {
    Bf16Pack8 gates = *reinterpret_cast<const Bf16Pack8*>(
        gate_up + input_offset + vector * 8);
    Bf16Pack8 ups = *reinterpret_cast<const Bf16Pack8*>(
        gate_up + input_offset + inner + vector * 8);
    Bf16Pack8 gate_bias{};
    Bf16Pack8 up_bias{};
    if (bias != nullptr) {
      gate_bias = *reinterpret_cast<const Bf16Pack8*>(bias + vector * 8);
      up_bias = *reinterpret_cast<const Bf16Pack8*>(bias + inner + vector * 8);
    }
#pragma unroll
    for (int item = 0; item < 8; ++item) {
      float gate = __bfloat162float(gates.values[item]);
      float up = __bfloat162float(ups.values[item]);
      if (bias != nullptr) {
        gate += __bfloat162float(gate_bias.values[item]);
        up += __bfloat162float(up_bias.values[item]);
      }
      const float value = (gate / (1.0f + __expf(-gate))) * up;
      activated[vector * 8 + item] = value;
      maximum = fmaxf(maximum, fabsf(value));
    }
  }
  const float scale =
      fmaxf(block_max_parallel_unsafe(maximum, scratch) / 448.0f, 1.0e-12f);
  const float inverse_scale = 1.0f / scale;
  if (threadIdx.x == 0) scales[row] = scale;

  for (int vector = threadIdx.x; vector < output_vectors;
       vector += blockDim.x) {
    Fp8Pack8 quantized{};
    if (vector < inner_vectors) {
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        float value = activated[vector * 8 + item] * inverse_scale;
        value = fminf(448.0f, fmaxf(-448.0f, value));
        quantized.values[item] = static_cast<__nv_fp8_e4m3>(value);
      }
    }
    *reinterpret_cast<Fp8Pack8*>(output + output_offset + vector * 8) = quantized;
  }
}

__global__ void swiglu_quantize_rows_vec4_kernel(
    const __nv_bfloat16* gate_up, const __nv_bfloat16* bias,
    __nv_fp8_e4m3* output, float* scales, int rows, int input_cols,
    int inner, int output_cols) {
  __shared__ float scratch[8];
  extern __shared__ float activated[];
  const int row = blockIdx.x;
  if (row >= rows) return;

  const int inner_vectors = inner / 4;
  const int output_vectors = output_cols / 4;
  const int64_t input_offset = static_cast<int64_t>(row) * input_cols;
  const int64_t output_offset = static_cast<int64_t>(row) * output_cols;
  float maximum = 0.0f;
  for (int vector = threadIdx.x; vector < inner_vectors;
       vector += blockDim.x) {
    Bf16Pack4 gates = *reinterpret_cast<const Bf16Pack4*>(
        gate_up + input_offset + vector * 4);
    Bf16Pack4 ups = *reinterpret_cast<const Bf16Pack4*>(
        gate_up + input_offset + inner + vector * 4);
    Bf16Pack4 gate_bias{};
    Bf16Pack4 up_bias{};
    if (bias != nullptr) {
      gate_bias = *reinterpret_cast<const Bf16Pack4*>(bias + vector * 4);
      up_bias = *reinterpret_cast<const Bf16Pack4*>(bias + inner + vector * 4);
    }
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      float gate = __bfloat162float(gates.values[item]);
      float up = __bfloat162float(ups.values[item]);
      if (bias != nullptr) {
        gate += __bfloat162float(gate_bias.values[item]);
        up += __bfloat162float(up_bias.values[item]);
      }
      const float value = (gate / (1.0f + __expf(-gate))) * up;
      activated[vector * 4 + item] = value;
      maximum = fmaxf(maximum, fabsf(value));
    }
  }
  const float scale =
      fmaxf(block_max_parallel_unsafe(maximum, scratch) / 448.0f, 1.0e-12f);
  const float inverse_scale = 1.0f / scale;
  if (threadIdx.x == 0) scales[row] = scale;

  for (int vector = threadIdx.x; vector < output_vectors;
       vector += blockDim.x) {
    Fp8Pack4 quantized{};
    if (vector < inner_vectors) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        float value = activated[vector * 4 + item] * inverse_scale;
        value = fminf(448.0f, fmaxf(-448.0f, value));
        quantized.values[item] = static_cast<__nv_fp8_e4m3>(value);
      }
    }
    *reinterpret_cast<Fp8Pack4*>(output + output_offset + vector * 4) =
        quantized;
  }
}

}  // namespace

int rms_norm_quantize_rows_bf16_e4m3(const void* input, const void* weight,
                                     void* output, void* scales, int rows,
                                     int input_cols, int output_cols, float eps,
                                     cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || output == nullptr ||
      scales == nullptr || rows <= 0 || input_cols <= 0 ||
      output_cols < input_cols || !(eps >= 0.0f)) {
    return -1;
  }
  constexpr int threads = 256;
  constexpr int rows_per_block = threads / 32;
  const int blocks = (rows + rows_per_block - 1) / rows_per_block;
  if (input_cols % 8 == 0 && output_cols % 8 == 0) {
    rms_norm_quantize_rows_vec8_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_fp8_e4m3*>(output), static_cast<float*>(scales), rows,
        input_cols, output_cols, eps);
  } else {
    rms_norm_quantize_rows_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_fp8_e4m3*>(output), static_cast<float*>(scales), rows,
        input_cols, output_cols, eps);
  }
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int bias_residual_rms_norm_quantize_rows_bf16_e4m3(
    const void* projection, const void* bias, const void* residual,
    const void* weight, void* hidden, void* normalized, void* scales, int rows,
    int cols, int output_cols, float eps, cudaStream_t stream) {
  if (projection == nullptr || residual == nullptr || weight == nullptr ||
      hidden == nullptr || normalized == nullptr || scales == nullptr ||
      rows <= 0 || cols <= 0 || output_cols < cols || !(eps >= 0.0f)) {
    return -1;
  }
  constexpr int threads = 256;
  if (rows <= 64) {
    constexpr int rows_per_block = threads / 32;
    const int blocks = (rows + rows_per_block - 1) / rows_per_block;
    if (cols % 8 == 0 && output_cols % 8 == 0) {
      bias_residual_rms_norm_quantize_rows_vec8_kernel<<<blocks, threads, 0,
                                                         stream>>>(
          static_cast<const __nv_bfloat16*>(projection),
          static_cast<const __nv_bfloat16*>(bias),
          static_cast<const __nv_bfloat16*>(residual),
          static_cast<const __nv_bfloat16*>(weight),
          static_cast<__nv_bfloat16*>(hidden),
          static_cast<__nv_fp8_e4m3*>(normalized),
          static_cast<float*>(scales), rows, cols, output_cols, eps);
    } else {
      bias_residual_rms_norm_quantize_rows_kernel<<<blocks, threads, 0,
                                                    stream>>>(
          static_cast<const __nv_bfloat16*>(projection),
          static_cast<const __nv_bfloat16*>(bias),
          static_cast<const __nv_bfloat16*>(residual),
          static_cast<const __nv_bfloat16*>(weight),
          static_cast<__nv_bfloat16*>(hidden),
          static_cast<__nv_fp8_e4m3*>(normalized),
          static_cast<float*>(scales), rows, cols, output_cols, eps);
    }
  } else {
    if (cols % 8 == 0 && output_cols % 8 == 0) {
      bias_residual_rms_norm_quantize_rows_large_vec8_kernel<<<rows, threads, 0,
                                                               stream>>>(
          static_cast<const __nv_bfloat16*>(projection),
          static_cast<const __nv_bfloat16*>(bias),
          static_cast<const __nv_bfloat16*>(residual),
          static_cast<const __nv_bfloat16*>(weight),
          static_cast<__nv_bfloat16*>(hidden),
          static_cast<__nv_fp8_e4m3*>(normalized),
          static_cast<float*>(scales), rows, cols, output_cols, eps);
    } else {
      bias_residual_rms_norm_quantize_rows_large_kernel<<<rows, threads, 0,
                                                          stream>>>(
          static_cast<const __nv_bfloat16*>(projection),
          static_cast<const __nv_bfloat16*>(bias),
          static_cast<const __nv_bfloat16*>(residual),
          static_cast<const __nv_bfloat16*>(weight),
          static_cast<__nv_bfloat16*>(hidden),
          static_cast<__nv_fp8_e4m3*>(normalized),
          static_cast<float*>(scales), rows, cols, output_cols, eps);
    }
  }
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int swiglu_quantize_rows_bf16_e4m3(const void* gate_up, const void* bias,
                                   void* output, void* scales, int rows,
                                   int input_cols, int inner, int output_cols,
                                   cudaStream_t stream) {
  if (gate_up == nullptr || output == nullptr || scales == nullptr ||
      rows <= 0 || input_cols <= 0 || inner <= 0 || input_cols < 2 * inner ||
      output_cols < inner) {
    return -1;
  }
  constexpr int threads = 256;
  const size_t shared_bytes = static_cast<size_t>(inner) * sizeof(float);
  if (input_cols % 8 == 0 && inner % 8 == 0 && output_cols % 8 == 0) {
    swiglu_quantize_rows_vec8_kernel<<<rows, threads, shared_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(gate_up),
        static_cast<const __nv_bfloat16*>(bias),
        static_cast<__nv_fp8_e4m3*>(output), static_cast<float*>(scales), rows,
        input_cols, inner, output_cols);
  } else if (input_cols % 4 == 0 && inner % 4 == 0 && output_cols % 4 == 0) {
    swiglu_quantize_rows_vec4_kernel<<<rows, threads, shared_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(gate_up),
        static_cast<const __nv_bfloat16*>(bias),
        static_cast<__nv_fp8_e4m3*>(output), static_cast<float*>(scales), rows,
        input_cols, inner, output_cols);
  } else {
    swiglu_quantize_rows_kernel<<<rows, threads, shared_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(gate_up),
        static_cast<const __nv_bfloat16*>(bias),
        static_cast<__nv_fp8_e4m3*>(output), static_cast<float*>(scales), rows,
        input_cols, inner, output_cols);
  }
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

}  // namespace apxinf::cuda_new::quant_ops
