#include "../internal.h"
#include "../../cublas_resources.h"
#include "../../../kernels/custom/attention_softmax.cuh"

namespace apxinf::attention {
namespace {
using apxinf::framework::check_cublas;
constexpr size_t kWorkspaceBytes = apxinf::providers::CublasStreamResources::workspace_bytes;
struct CublasState {
  std::shared_ptr<apxinf::providers::CublasStreamResources> shared;
  __nv_bfloat16* scores = nullptr;
  ~CublasState() {
    if (scores != nullptr) cudaFree(scores);
  }
};
}  // namespace

size_t cublas_resource_requirements(const Spec& spec, int) {
  const uint64_t elements = static_cast<uint64_t>(spec.batch) * spec.query_heads *
                            spec.query_tokens * spec.key_tokens;
  if (elements > (SIZE_MAX - kWorkspaceBytes) / sizeof(__nv_bfloat16))
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "Attention cuBLAS scratch size overflow");
  return kWorkspaceBytes + static_cast<size_t>(elements) * sizeof(__nv_bfloat16);
}

void prepare_cublas(Execution& execution) {
  const size_t bytes = cublas_resource_requirements(execution.spec, 0);
  if (bytes > execution.resource_limit)
    throw Failure(APXINF_STATUS_UNSUPPORTED, "Attention cuBLAS scratch exceeds policy");
  auto state = std::make_unique<CublasState>();
  state->shared = apxinf::providers::acquire_cublas_stream_resources(
      execution.device, static_cast<cudaStream_t>(execution.bindings.stream));
  check_cuda(cudaMalloc(&state->scores, bytes - kWorkspaceBytes));
  execution.resource_bytes = bytes;
  execution.provider_state = state.release();
}

cudaError_t launch_cublas(Execution& execution) {
  const auto& s = execution.spec;
  const auto& b = execution.bindings;
  auto& state = *static_cast<CublasState*>(execution.provider_state);
  const auto stream = static_cast<cudaStream_t>(b.stream);
  // Handle, stream and workspace are immutable after preparation. Serialize
  // the whole attention sequence with other executions sharing this stream.
  std::lock_guard<std::mutex> lock(state.shared->enqueue_mutex);
  const float one = 1.0F, zero = 0.0F;
  const int h = s.query_heads, d = s.head_dim, q = s.query_tokens, k = s.key_tokens;
  const auto* query = static_cast<const __nv_bfloat16*>(b.query);
  const auto* key = static_cast<const __nv_bfloat16*>(b.key);
  const auto* value = static_cast<const __nv_bfloat16*>(b.value);
  auto* output = static_cast<__nv_bfloat16*>(b.output);
  // Canonical B,T,H,D tensors are interleaved column-major D-by-T matrices.
  // Each head's scores/probabilities are contiguous column-major K-by-Q.
  for (uint32_t batch = 0; batch < s.batch; ++batch) {
    check_cublas(cublasGemmStridedBatchedEx(state.shared->handle, CUBLAS_OP_T, CUBLAS_OP_N,
        k, q, d, &b.scale, key + static_cast<int64_t>(batch) * k * h * d,
        CUDA_R_16BF, h * d, d, query + static_cast<int64_t>(batch) * q * h * d,
        CUDA_R_16BF, h * d, d, &zero,
        state.scores + static_cast<int64_t>(batch) * h * q * k,
        CUDA_R_16BF, k, static_cast<int64_t>(q) * k, h,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }
  check_cuda(kernels::launch_attention_input_softmax(state.scores,
      static_cast<const float*>(b.bias), s.batch, h, q, k,
      s.bias_broadcast_batch != 0, s.mask == APXINF_ATTENTION_MASK_CAUSAL,
      s.query_start, stream));
  for (uint32_t batch = 0; batch < s.batch; ++batch) {
    check_cublas(cublasGemmStridedBatchedEx(state.shared->handle, CUBLAS_OP_N, CUBLAS_OP_N,
        d, q, k, &one, value + static_cast<int64_t>(batch) * k * h * d,
        CUDA_R_16BF, h * d, d,
        state.scores + static_cast<int64_t>(batch) * h * q * k,
        CUDA_R_16BF, k, static_cast<int64_t>(q) * k, &zero,
        output + static_cast<int64_t>(batch) * q * h * d,
        CUDA_R_16BF, h * d, d, h, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }
  return cudaGetLastError();
}

void destroy_cublas(Execution& execution) noexcept {
  delete static_cast<CublasState*>(execution.provider_state);
  execution.provider_state = nullptr;
}
}  // namespace apxinf::attention
