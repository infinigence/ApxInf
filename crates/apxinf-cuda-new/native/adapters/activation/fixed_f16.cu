// Copyright 2026 ApxInf contributors.
// Fixed FP16 implementation for this operator family; ported without numerical changes.
#include "../../include/apxinf_cuda/transformer_f16.h"
#include "../../kernels/custom/reduction.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kThreads = 256;

int blocks_for(int64_t count) {
  return static_cast<int>((count + kThreads - 1) / kThreads);
}

int clamped_blocks(int64_t count) {
  int blocks = static_cast<int>((count + 255) / 256);
  return blocks > 1024 ? 1024 : blocks;
}

// ── activations (kernels/custom/activation.cuh) ─────────────────────────────

__device__ __forceinline__ half gelu_tanh_f16_one(half value)
{
    float x = __half2float(value);
    const float kBeta  = 0.7978845608028654f;
    const float kAlpha = 0.044715f;
    float inner = kBeta * (x + kAlpha * x * x * x);
    float y = 0.5f * x * (1.0f + tanhf(inner));
    return __float2half(y);
}

__global__ void bias_gelu_f16_kernel(
    const half* input, const half* bias, half* output,
    int rows, int cols) {
  const int64_t count = static_cast<int64_t>(rows) * cols;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int col = static_cast<int>(index % cols);
    output[index] = gelu_tanh_f16_one(
        __float2half(__half2float(input[index]) + __half2float(bias[col])));
  }
}

__global__ void bias_silu_f16_kernel(
    const half* input, const half* bias, half* output, int64_t count, int cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    float value = __half2float(input[index]);
    if (bias != nullptr) value += __half2float(bias[index % cols]);
    output[index] = __float2half(value / (1.0f + expf(-value)));
  }
}

__global__ void swiglu_f16_kernel(
    const half* gate_up, half* output, int rows, int inner) {
  const int64_t count = static_cast<int64_t>(rows) * inner;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int row = static_cast<int>(index / inner);
    const int col = static_cast<int>(index % inner);
    const float gate = __half2float(gate_up[static_cast<int64_t>(row) * 2 * inner + col]);
    const float up = __half2float(gate_up[static_cast<int64_t>(row) * 2 * inner + inner + col]);
    const float silu = gate / (1.0f + expf(-gate));
    output[index] = __float2half(silu * up);
  }
}

}  // namespace

extern "C" cudaError_t apxinf_transformer_f16_swiglu_f16(
    const void* gate_up, void* output, int rows, int inner,
    cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(rows) * inner;
  if (gate_up == nullptr || output == nullptr || rows <= 0 || inner <= 0)
    return cudaErrorInvalidValue;
  swiglu_f16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const half*>(gate_up), static_cast<half*>(output),
      rows, inner);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_f16_bias_gelu_f16(
    const void* input, const void* bias, void* output,
    int rows, int cols, cudaStream_t stream) {
  if (input == nullptr || bias == nullptr || output == nullptr ||
      rows <= 0 || cols <= 0) {
    return cudaErrorInvalidValue;
  }
  const int64_t count = static_cast<int64_t>(rows) * cols;
  bias_gelu_f16_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(bias),
      static_cast<half*>(output), rows, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_transformer_f16_bias_silu_f16(
    const void* input, const void* bias, void* output, int rows, int cols,
    cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  bias_silu_f16_kernel<<<clamped_blocks(count), 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(bias),
      static_cast<half*>(output), count, cols);
  return cudaGetLastError();
}
