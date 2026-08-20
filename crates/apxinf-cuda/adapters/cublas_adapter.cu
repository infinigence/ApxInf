// Copyright 2026 apxinf contributors.
// cuBLAS MQA adapter with private logits workspace and custom softmax launch.

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

namespace {

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

#include "../kernels/custom/reduction.cuh"
#include "../kernels/custom/attention.cuh"

}  // namespace

extern "C" int apxinf_static_cublas_mqa_f16(
    const void* q, const void* k, const void* v, void* output,
    int query_tokens, int key_tokens, int heads, int head_dim,
    cudaStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      query_tokens <= 0 || key_tokens <= 0 ||
      key_tokens > kSoftmaxMaxCols || heads <= 0 || head_dim <= 0) {
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  }
  int rows = query_tokens * heads;
  size_t logits_bytes = static_cast<size_t>(rows) * key_tokens * sizeof(half);
  cublasStatus_t status = initialize_mqa(logits_bytes);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  status = cublasSetStream(g_mqa_blas, stream);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);

  float attention_scale = rsqrtf(static_cast<float>(head_dim));
  float zero = 0.0f;
  status = cublasGemmEx(
      g_mqa_blas, CUBLAS_OP_T, CUBLAS_OP_N,
      key_tokens, rows, head_dim, &attention_scale,
      k, CUDA_R_16F, head_dim,
      q, CUDA_R_16F, head_dim,
      &zero, g_mqa_logits, CUDA_R_16F, key_tokens,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  if ((key_tokens & 1) == 0) {
    softmax_even_f16_kernel<<<rows, 32, 0, stream>>>(
        g_mqa_logits, rows, key_tokens);
  } else {
    softmax_scalar_f16_kernel<<<rows, 32, 0, stream>>>(
        g_mqa_logits, rows, key_tokens);
  }
  if (cudaPeekAtLastError() != cudaSuccess) {
    return static_cast<int>(CUBLAS_STATUS_EXECUTION_FAILED);
  }

  float one = 1.0f;
  status = cublasGemmEx(
      g_mqa_blas, CUBLAS_OP_N, CUBLAS_OP_N,
      head_dim, rows, key_tokens, &one,
      v, CUDA_R_16F, head_dim,
      g_mqa_logits, CUDA_R_16F, key_tokens,
      &zero, output, CUDA_R_16F, head_dim,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  return static_cast<int>(status);
}

extern "C" int apxinf_static_cublas_mqa_bf16(
    const void* q, const void* k, const void* v, void* output,
    int query_tokens, int key_tokens, int heads, int head_dim,
    cudaStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      query_tokens <= 0 || key_tokens <= 0 ||
      key_tokens > kSoftmaxMaxCols || heads <= 0 || head_dim <= 0) {
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  }
  int rows = query_tokens * heads;
  size_t logits_bytes =
      static_cast<size_t>(rows) * key_tokens * sizeof(__nv_bfloat16);
  cublasStatus_t status = initialize_mqa(logits_bytes);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  status = cublasSetStream(g_mqa_blas, stream);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);

  float attention_scale = rsqrtf(static_cast<float>(head_dim));
  float zero = 0.0f;
  auto* logits = reinterpret_cast<__nv_bfloat16*>(g_mqa_logits);
  status = cublasGemmEx(
      g_mqa_blas, CUBLAS_OP_T, CUBLAS_OP_N,
      key_tokens, rows, head_dim, &attention_scale,
      k, CUDA_R_16BF, head_dim,
      q, CUDA_R_16BF, head_dim,
      &zero, logits, CUDA_R_16BF, key_tokens,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  softmax_scalar_bf16_kernel<<<rows, 32, 0, stream>>>(
      logits, rows, key_tokens);
  if (cudaPeekAtLastError() != cudaSuccess) {
    return static_cast<int>(CUBLAS_STATUS_EXECUTION_FAILED);
  }

  float one = 1.0f;
  status = cublasGemmEx(
      g_mqa_blas, CUBLAS_OP_N, CUBLAS_OP_N,
      head_dim, rows, key_tokens, &one,
      v, CUDA_R_16BF, head_dim,
      logits, CUDA_R_16BF, key_tokens,
      &zero, output, CUDA_R_16BF, head_dim,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  return static_cast<int>(status);
}
