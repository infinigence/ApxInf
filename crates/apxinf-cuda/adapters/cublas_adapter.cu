// Copyright 2026 apxinf contributors.
// cuBLAS MQA adapter with private logits workspace and custom softmax launch.

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

thread_local cublasHandle_t g_mqa_blas = nullptr;
thread_local half* g_mqa_logits = nullptr;
thread_local size_t g_mqa_logits_bytes = 0;
thread_local cublasHandle_t g_mha_blas = nullptr;
thread_local float* g_mha_scores = nullptr;
thread_local __nv_bfloat16* g_mha_probs = nullptr;
thread_local size_t g_mha_scores_bytes = 0;
thread_local size_t g_mha_probs_bytes = 0;

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

cublasStatus_t initialize_mha(
    size_t scores_bytes, size_t probs_bytes) {
  if (g_mha_blas == nullptr) {
    cublasStatus_t status = cublasCreate(&g_mha_blas);
    if (status != CUBLAS_STATUS_SUCCESS) return status;
  }
  if (scores_bytes > g_mha_scores_bytes) {
    if (g_mha_scores != nullptr) cudaFree(g_mha_scores);
    cudaError_t status = cudaMalloc(&g_mha_scores, scores_bytes);
    if (status != cudaSuccess) {
      g_mha_scores = nullptr;
      g_mha_scores_bytes = 0;
      return CUBLAS_STATUS_ALLOC_FAILED;
    }
    g_mha_scores_bytes = scores_bytes;
  }
  if (probs_bytes > g_mha_probs_bytes) {
    if (g_mha_probs != nullptr) cudaFree(g_mha_probs);
    cudaError_t status = cudaMalloc(&g_mha_probs, probs_bytes);
    if (status != cudaSuccess) {
      g_mha_probs = nullptr;
      g_mha_probs_bytes = 0;
      return CUBLAS_STATUS_ALLOC_FAILED;
    }
    g_mha_probs_bytes = probs_bytes;
  }
  return CUBLAS_STATUS_SUCCESS;
}

#include "../kernels/custom/reduction.cuh"
#include "../kernels/custom/attention.cuh"

}  // namespace

extern "C" cudaError_t apxinf_static_row_softmax_f32_bf16(
    const void* input, void* output, uint32_t cols, uint32_t rows,
    cudaStream_t stream);

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
  mqa_softmax_f16_block_kernel<<<rows, kMqaSoftmaxThreads, 0, stream>>>(
      g_mqa_logits, rows, key_tokens);
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

extern "C" int apxinf_static_cublas_gqa_f16(
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

extern "C" int apxinf_static_cublas_gqa_causal_f16(
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

extern "C" int apxinf_static_cublas_mha_bf16(
    const void* q, const void* k, const void* v, void* output,
    int tokens_per_batch, int batches, int heads, int head_dim,
    cudaStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
      tokens_per_batch <= 0 || batches <= 0 || heads <= 0 ||
      head_dim <= 0 || tokens_per_batch > kSoftmaxMaxCols) {
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  }

  const size_t rows = static_cast<size_t>(heads) * tokens_per_batch;
  const size_t scores_bytes =
      static_cast<size_t>(rows) * tokens_per_batch * sizeof(float);
  const size_t probs_bytes =
      static_cast<size_t>(rows) * tokens_per_batch * sizeof(__nv_bfloat16);
  cublasStatus_t status = initialize_mha(scores_bytes, probs_bytes);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  status = cublasSetStream(g_mha_blas, stream);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);

  const int row_stride = heads * head_dim;
  const int matrix_elements =
      static_cast<size_t>(tokens_per_batch) * row_stride > INT32_MAX
          ? 0
          : tokens_per_batch * row_stride;
  if (matrix_elements == 0) return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);

  const float attention_scale = rsqrtf(static_cast<float>(head_dim));
  const float zero = 0.0f;
  const float one = 1.0f;
  const long long score_stride =
      static_cast<long long>(tokens_per_batch) * tokens_per_batch;

  for (int batch = 0; batch < batches; ++batch) {
    const size_t batch_offset =
        static_cast<size_t>(batch) * tokens_per_batch * row_stride;
    const auto* q_batch = static_cast<const __nv_bfloat16*>(q) + batch_offset;
    const auto* k_batch = static_cast<const __nv_bfloat16*>(k) + batch_offset;
    const auto* v_batch = static_cast<const __nv_bfloat16*>(v) + batch_offset;
    auto* output_batch = static_cast<__nv_bfloat16*>(output) + batch_offset;

    status = cublasGemmStridedBatchedEx(
        g_mha_blas, CUBLAS_OP_T, CUBLAS_OP_N,
        tokens_per_batch, tokens_per_batch, head_dim, &attention_scale,
        k_batch, CUDA_R_16BF, row_stride, head_dim,
        q_batch, CUDA_R_16BF, row_stride, head_dim,
        &zero, g_mha_scores, CUDA_R_32F, tokens_per_batch, score_stride,
        heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);

    const cudaError_t cuda_status = apxinf_static_row_softmax_f32_bf16(
        g_mha_scores, g_mha_probs, tokens_per_batch,
        static_cast<uint32_t>(rows), stream);
    if (cuda_status != cudaSuccess) {
      return static_cast<int>(CUBLAS_STATUS_EXECUTION_FAILED);
    }

    status = cublasGemmStridedBatchedEx(
        g_mha_blas, CUBLAS_OP_N, CUBLAS_OP_N,
        head_dim, tokens_per_batch, tokens_per_batch, &one,
        v_batch, CUDA_R_16BF, row_stride, head_dim,
        g_mha_probs, CUDA_R_16BF, tokens_per_batch, score_stride,
        &zero, output_batch, CUDA_R_16BF, row_stride, head_dim,
        heads, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  }
  return static_cast<int>(CUBLAS_STATUS_SUCCESS);
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
