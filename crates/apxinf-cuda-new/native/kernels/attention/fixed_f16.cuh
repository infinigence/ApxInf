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

// ── cuBLAS FP16 GQA (adapters/cublas_adapter.cu) ────────────────────────────

constexpr int kSoftmaxMaxCols = 1024;

thread_local cublasHandle_t g_mqa_blas = nullptr;
thread_local half* g_mqa_logits = nullptr;
thread_local size_t g_mqa_logits_bytes = 0;

cublasStatus_t initialize_mqa(size_t logits_bytes) {
  if (g_mqa_blas == nullptr) {
    cublasStatus_t status = cublasCreate(&g_mqa_blas);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
  }
  if (logits_bytes > g_mqa_logits_bytes) {
    if (g_mqa_logits != nullptr) cudaFree(g_mqa_logits);
    cudaError_t cuda_status = cudaMalloc(&g_mqa_logits, logits_bytes);
    if (cuda_status != cudaSuccess) {
      g_mqa_logits = nullptr;
      g_mqa_logits_bytes = 0;
      return CUBLAS_STATUS_ALLOC_FAILED;
    }
    g_mqa_logits_bytes = logits_bytes;
  }
  return CUBLAS_STATUS_SUCCESS;
}

__global__ void gqa_softmax_f16_warp_kernel(
    half* data, int rows, int cols, int batches) {
  const int lane = threadIdx.x;
  const int row = blockIdx.x;
  const int batch = blockIdx.y;
  if (row >= rows || lane >= 32) return;
  half* source =
      data + (static_cast<int64_t>(batch) * rows + row) * cols;
  const int vectors = cols / 2;
  half2* source2 = reinterpret_cast<half2*>(source);
  float maximum = -1.0e30f;

  if ((cols & 1) == 0) {
#pragma unroll 4
    for (int vector = lane; vector < vectors; vector += 32) {
      const float2 pair = __half22float2(source2[vector]);
      maximum = fmaxf(maximum, fmaxf(pair.x, pair.y));
    }
  } else {
    for (int col = lane; col < cols; col += 32) {
      maximum = fmaxf(maximum, __half2float(source[col]));
    }
  }
  maximum = warp_max(maximum);

  float sum = 0.0f;
  if ((cols & 1) == 0) {
#pragma unroll 4
    for (int vector = lane; vector < vectors; vector += 32) {
      const float2 pair = __half22float2(source2[vector]);
      sum += __expf(pair.x - maximum) + __expf(pair.y - maximum);
    }
  } else {
    for (int col = lane; col < cols; col += 32) {
      sum += __expf(__half2float(source[col]) - maximum);
    }
  }
  sum = warp_sum_all(sum);
  const float inverse = 1.0f / sum;

  if ((cols & 1) == 0) {
#pragma unroll 4
    for (int vector = lane; vector < vectors; vector += 32) {
      const float2 pair = __half22float2(source2[vector]);
      source2[vector] = __floats2half2_rn(
          __expf(pair.x - maximum) * inverse,
          __expf(pair.y - maximum) * inverse);
    }
  } else {
    for (int col = lane; col < cols; col += 32) {
      source[col] =
          __float2half(__expf(__half2float(source[col]) - maximum) * inverse);
    }
  }
}

__global__ void gqa_softmax_f16_causal_warp_kernel(
    half* data, int rows, int cols, int batches, int offset) {
  const int lane = threadIdx.x;
  const int row = blockIdx.x;
  const int batch = blockIdx.y;
  if (row >= rows || lane >= 32) return;
  half* source = data + (static_cast<int64_t>(batch) * rows + row) * cols;
  const int valid = offset + row + 1;
  float maximum = -1.0e30f;
  for (int col = lane; col < cols; col += 32) {
    const float value = col < valid ? __half2float(source[col]) : -1.0e30f;
    maximum = fmaxf(maximum, value);
  }
  maximum = warp_max(maximum);

  float sum = 0.0f;
  for (int col = lane; col < cols; col += 32) {
    if (col < valid) {
      sum += __expf(__half2float(source[col]) - maximum);
    }
  }
  sum = warp_sum_all(sum);
  const float inverse = 1.0f / sum;
  for (int col = lane; col < cols; col += 32) {
    const float value =
        col < valid ? __expf(__half2float(source[col]) - maximum) * inverse : 0.0f;
    source[col] = __float2half(value);
  }
}

}  // namespace
