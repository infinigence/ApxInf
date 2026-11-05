// Copyright 2026 ApxInf contributors.
// Fixed FP16 implementation for this operator family; ported without numerical changes.
#include "../custom/reduction.cuh"

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

// ── normalization (kernels/custom/normalization.cuh) ────────────────────────

__global__ void rms_norm_f16_kernel(
    const half* input, const half* weight, half* output,
    uint32_t cols, uint32_t rows, float eps)
{
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    uint32_t tid = threadIdx.x;
    uint32_t offset = row * cols;
    extern __shared__ float x_buf[];
    __shared__ float s_sum;

    float partial = 0.0f;
    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        float v = __half2float(input[offset + i]);
        x_buf[i] = v;
        partial += v * v;
    }
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
    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        float w = __half2float(weight[i]);
        output[offset + i] = __float2half(x_buf[i] * rms * w);
    }
}

__global__ void layer_norm_f16_kernel(
    const half* input, const half* weight,
    const half* bias, half* output,
    int rows, int cols, float eps) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    sum += __half2float(input[static_cast<int64_t>(row) * cols + col]);
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered =
        __half2float(input[static_cast<int64_t>(row) * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float value =
        (__half2float(input[index]) - mean) * inverse_std *
            __half2float(weight[col]) +
        __half2float(bias[col]);
    output[index] = __float2half(value);
  }
}

}  // namespace
