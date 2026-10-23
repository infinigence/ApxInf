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

extern "C" int apxinf_transformer_f16_cublas_gqa_f16(
    const void* q, const void* k, const void* v, void* output,
    int query_tokens, int key_tokens, int q_heads, int kv_heads,
    int head_dim, cudaStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      query_tokens <= 0 || key_tokens <= 0 || key_tokens > kSoftmaxMaxCols ||
      q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0 ||
      head_dim <= 0) {
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  }
  const int group = q_heads / kv_heads;
  const int64_t score_stride =
      static_cast<int64_t>(query_tokens) * key_tokens;
  const size_t logits_bytes =
      static_cast<size_t>(q_heads) * score_stride * sizeof(half);
  cublasStatus_t status = initialize_mqa(logits_bytes);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  status = cublasSetStream(g_mqa_blas, stream);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);

  const float attention_scale = rsqrtf(static_cast<float>(head_dim));
  const float zero = 0.0f;
  const float one = 1.0f;
  for (int kv_head = 0; kv_head < kv_heads; ++kv_head) {
    const half* query = static_cast<const half*>(q) +
                        kv_head * group * head_dim;
    const half* key = static_cast<const half*>(k) + kv_head * head_dim;
    half* scores = static_cast<half*>(g_mqa_logits) +
                   static_cast<int64_t>(kv_head) * group * score_stride;
    status = cublasGemmStridedBatchedEx(
        g_mqa_blas, CUBLAS_OP_T, CUBLAS_OP_N,
        key_tokens, query_tokens, head_dim, &attention_scale,
        key, CUDA_R_16F, kv_heads * head_dim, 0,
        query, CUDA_R_16F, q_heads * head_dim, head_dim,
        &zero, scores, CUDA_R_16F, key_tokens, score_stride,
        group, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  }

  gqa_softmax_f16_warp_kernel<<<dim3(query_tokens, q_heads), 32, 0, stream>>>(
      g_mqa_logits, query_tokens, key_tokens, q_heads);
  if (cudaPeekAtLastError() != cudaSuccess) {
    return static_cast<int>(CUBLAS_STATUS_EXECUTION_FAILED);
  }

  for (int kv_head = 0; kv_head < kv_heads; ++kv_head) {
    const half* value = static_cast<const half*>(v) + kv_head * head_dim;
    const half* scores = static_cast<const half*>(g_mqa_logits) +
                         static_cast<int64_t>(kv_head) * group * score_stride;
    half* destination = static_cast<half*>(output) +
                        kv_head * group * head_dim;
    status = cublasGemmStridedBatchedEx(
        g_mqa_blas, CUBLAS_OP_N, CUBLAS_OP_N,
        head_dim, query_tokens, key_tokens, &one,
        value, CUDA_R_16F, kv_heads * head_dim, 0,
        scores, CUDA_R_16F, key_tokens, score_stride,
        &zero, destination, CUDA_R_16F, q_heads * head_dim, head_dim,
        group, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  }
  return static_cast<int>(status);
}

extern "C" int apxinf_transformer_f16_cublas_gqa_causal_f16(
    const void* q, const void* k, const void* v, void* output,
    int query_tokens, int key_tokens, int q_heads, int kv_heads,
    int head_dim, int key_offset, cudaStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      query_tokens <= 0 || key_tokens <= 0 || key_tokens > kSoftmaxMaxCols ||
      q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0 ||
      head_dim <= 0 || key_offset < 0 || query_tokens + key_offset > key_tokens) {
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  }
  const int group = q_heads / kv_heads;
  const int64_t score_stride = static_cast<int64_t>(query_tokens) * key_tokens;
  const size_t logits_bytes = static_cast<size_t>(q_heads) * score_stride * sizeof(half);
  cublasStatus_t status = initialize_mqa(logits_bytes);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  status = cublasSetStream(g_mqa_blas, stream);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);

  const float attention_scale = rsqrtf(static_cast<float>(head_dim));
  const float zero = 0.0f;
  const float one = 1.0f;
  for (int kv_head = 0; kv_head < kv_heads; ++kv_head) {
    const half* query = static_cast<const half*>(q) + kv_head * group * head_dim;
    const half* key = static_cast<const half*>(k) + kv_head * head_dim;
    half* scores = static_cast<half*>(g_mqa_logits) +
                   static_cast<int64_t>(kv_head) * group * score_stride;
    status = cublasGemmStridedBatchedEx(
        g_mqa_blas, CUBLAS_OP_T, CUBLAS_OP_N,
        key_tokens, query_tokens, head_dim, &attention_scale,
        key, CUDA_R_16F, kv_heads * head_dim, 0,
        query, CUDA_R_16F, q_heads * head_dim, head_dim,
        &zero, scores, CUDA_R_16F, key_tokens, score_stride,
        group, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  }

  gqa_softmax_f16_causal_warp_kernel<<<dim3(query_tokens, q_heads), 32, 0, stream>>>(
      g_mqa_logits, query_tokens, key_tokens, q_heads, key_offset);
  if (cudaPeekAtLastError() != cudaSuccess) {
    return static_cast<int>(CUBLAS_STATUS_EXECUTION_FAILED);
  }

  for (int kv_head = 0; kv_head < kv_heads; ++kv_head) {
    const half* value = static_cast<const half*>(v) + kv_head * head_dim;
    half* scores = static_cast<half*>(g_mqa_logits) +
                   static_cast<int64_t>(kv_head) * group * score_stride;
    half* destination = static_cast<half*>(output) + kv_head * group * head_dim;
    status = cublasGemmStridedBatchedEx(
        g_mqa_blas, CUBLAS_OP_N, CUBLAS_OP_N,
        head_dim, query_tokens, key_tokens, &one,
        value, CUDA_R_16F, kv_heads * head_dim, 0,
        scores, CUDA_R_16F, key_tokens, score_stride,
        &zero, destination, CUDA_R_16F, q_heads * head_dim, head_dim,
        group, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  }
  return static_cast<int>(status);
}
