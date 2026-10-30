#include "../../kernels/attention/fixed_f16.cuh"
#include "internal.h"

extern "C" int apxinf_attention_cublas_gqa_f16(
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

extern "C" int apxinf_attention_cublas_gqa_causal_f16(
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

namespace apxinf::attention {

size_t cublas_f16_resource_requirements(const Spec& spec, int) {
  const uint64_t elements = static_cast<uint64_t>(spec.query_heads) *
                            static_cast<uint64_t>(spec.query_tokens) *
                            static_cast<uint64_t>(spec.key_tokens);
  if (elements > SIZE_MAX / sizeof(half)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "cuBLAS F16 Attention workspace size overflow");
  }
  return static_cast<size_t>(elements) * sizeof(half);
}

void prepare_cublas_f16(Execution& execution) {
  const size_t bytes =
      cublas_f16_resource_requirements(execution.spec,
                                       execution.configuration);
  if (bytes > execution.resource_limit) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "cuBLAS F16 Attention workspace exceeds policy");
  }
  const cublasStatus_t status = initialize_mqa(bytes);
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  "cuBLAS F16 Attention preparation failed");
  }
  execution.resource_bytes = bytes;
}

cudaError_t launch_cublas_f16(Execution& execution) {
  const auto& spec = execution.spec;
  const auto& bindings = execution.bindings;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  const int status = spec.mask == APXINF_ATTENTION_MASK_CAUSAL
                         ? apxinf_attention_cublas_gqa_causal_f16(
                               bindings.query, bindings.key, bindings.value,
                               bindings.output,
                               static_cast<int>(spec.query_tokens),
                               static_cast<int>(spec.key_tokens),
                               static_cast<int>(spec.query_heads),
                               static_cast<int>(spec.kv_heads),
                               static_cast<int>(spec.head_dim),
                               static_cast<int>(spec.query_start), stream)
                         : apxinf_attention_cublas_gqa_f16(
                               bindings.query, bindings.key, bindings.value,
                               bindings.output,
                               static_cast<int>(spec.query_tokens),
                               static_cast<int>(spec.key_tokens),
                               static_cast<int>(spec.query_heads),
                               static_cast<int>(spec.kv_heads),
                               static_cast<int>(spec.head_dim), stream);
  return status == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}

void destroy_cublas_f16(Execution&) noexcept {}

}  // namespace apxinf::attention
