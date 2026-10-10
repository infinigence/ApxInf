// Copyright 2026 ApxInf contributors.
//
// Fused rowwise-quantizing epilogues for the walloss family, ported
// bit-identically from the legacy kernels. See `quant_ops.h`.

#include "quant_ops.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace apxinf::cuda_new::quant_ops {
namespace {

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
  constexpr int kWarpsPerBlock = 8;
  const int blocks = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  rms_norm_quantize_rows_kernel<<<blocks, kWarpsPerBlock * 32, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_fp8_e4m3*>(output), static_cast<float*>(scales), rows,
      input_cols, output_cols, eps);
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
  constexpr int kWarpsPerBlock = 8;
  const int blocks = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  bias_residual_rms_norm_quantize_rows_kernel<<<blocks, kWarpsPerBlock * 32, 0,
                                                stream>>>(
      static_cast<const __nv_bfloat16*>(projection),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(hidden),
      static_cast<__nv_fp8_e4m3*>(normalized), static_cast<float*>(scales),
      rows, cols, output_cols, eps);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int swiglu_quantize_rows_bf16_e4m3(const void* gate_up, const void* bias,
                                   void* output, void* scales, int rows,
                                   int input_cols, int inner, int output_cols,
                                   cudaStream_t stream) {
  if (gate_up == nullptr || output == nullptr || scales == nullptr ||
      rows <= 0 || input_cols <= 0 || inner <= 0 || inner * 2 != input_cols ||
      output_cols < inner) {
    return -1;
  }
  const int threads = 256;
  const size_t shared = static_cast<size_t>(inner) * sizeof(float);
  swiglu_quantize_rows_kernel<<<rows, threads, shared, stream>>>(
      static_cast<const __nv_bfloat16*>(gate_up),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_fp8_e4m3*>(output), static_cast<float*>(scales), rows,
      input_cols, inner, output_cols);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

}  // namespace apxinf::cuda_new::quant_ops
