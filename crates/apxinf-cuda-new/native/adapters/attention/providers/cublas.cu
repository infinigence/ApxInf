#include "../internal.h"
#include "../../../kernels/custom/attention_softmax.cuh"

namespace apxinf::attention {
namespace {
using apxinf::framework::check_cublas;
constexpr size_t kWorkspaceBytes = 4 * 1024 * 1024;
struct CublasState {
  cublasHandle_t handle = nullptr;
  void* workspace = nullptr;
  void* scores = nullptr;
  ~CublasState() {
    if (scores != nullptr) cudaFree(scores);
    if (workspace != nullptr) cudaFree(workspace);
    if (handle != nullptr) cublasDestroy(handle);
  }
};
}  // namespace

size_t cublas_resource_requirements(const Spec& spec, int) {
  const uint64_t elements = static_cast<uint64_t>(spec.batch) * spec.query_heads *
                            spec.query_tokens * spec.key_tokens;
  if (elements > (SIZE_MAX - kWorkspaceBytes) / dtype_bytes(spec.dtype))
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "Attention cuBLAS scratch size overflow");
  return kWorkspaceBytes + static_cast<size_t>(elements) * dtype_bytes(spec.dtype);
}

void prepare_cublas(Execution& execution) {
  const size_t bytes = cublas_resource_requirements(execution.spec, 0);
  if (bytes > execution.resource_limit)
    throw Failure(APXINF_STATUS_UNSUPPORTED, "Attention cuBLAS scratch exceeds policy");
  auto state = std::make_unique<CublasState>();
  check_cublas(cublasCreate(&state->handle));
  check_cublas(cublasSetMathMode(state->handle, CUBLAS_PEDANTIC_MATH));
  check_cuda(cudaMalloc(&state->workspace, kWorkspaceBytes));
  check_cublas(cublasSetStream(state->handle, static_cast<cudaStream_t>(execution.bindings.stream)));
  check_cublas(cublasSetWorkspace(state->handle, state->workspace, kWorkspaceBytes));
  check_cuda(cudaMalloc(&state->scores, bytes - kWorkspaceBytes));
  execution.resource_bytes = bytes;
  execution.provider_state = state.release();
}



// Pedantic F32 GEMMs retain full F32 products (no TF32 conversion). Prepared
// handle state is immutable and the scratch allocation precedes graph capture.
cudaError_t launch_f32_cublas(Execution& execution) {
  const auto& s = execution.spec;
  const auto& b = execution.bindings;
  auto& state = *static_cast<CublasState*>(execution.provider_state);
  const auto stream = static_cast<cudaStream_t>(b.stream);
  const float one = 1.0F, zero = 0.0F;
  const int h = s.query_heads, d = s.head_dim, q = s.query_tokens, k = s.key_tokens;
  auto* scores = static_cast<float*>(state.scores);
  for (uint32_t batch = 0; batch < s.batch; ++batch) {
    check_cublas(cublasSgemmStridedBatched(state.handle,
        CUBLAS_OP_T, CUBLAS_OP_N, k, q, d, &b.scale,
        static_cast<const float*>(b.key) + static_cast<int64_t>(batch) * k * h * d,
        h * d, d,
        static_cast<const float*>(b.query) + static_cast<int64_t>(batch) * q * h * d,
        h * d, d, &zero, scores + static_cast<int64_t>(batch) * h * q * k,
        k, static_cast<int64_t>(q) * k, h));
  }
  check_cuda(kernels::launch_attention_f32_softmax(scores, s.batch, h, q, k,
      s.mask == APXINF_ATTENTION_MASK_CAUSAL, s.query_start, stream));
  for (uint32_t batch = 0; batch < s.batch; ++batch) {
    check_cublas(cublasSgemmStridedBatched(state.handle,
        CUBLAS_OP_N, CUBLAS_OP_N, d, q, k, &one,
        static_cast<const float*>(b.value) + static_cast<int64_t>(batch) * k * h * d,
        h * d, d, scores + static_cast<int64_t>(batch) * h * q * k,
        k, static_cast<int64_t>(q) * k, &zero,
        static_cast<float*>(b.output) + static_cast<int64_t>(batch) * q * h * d,
        h * d, d, h));
  }
  return cudaGetLastError();
}

void destroy_cublas(Execution& execution) noexcept {
  delete static_cast<CublasState*>(execution.provider_state);
  execution.provider_state = nullptr;
}
}  // namespace apxinf::attention
