#include "../internal.h"
#include "../../../kernels/attention/wide_bf16.cuh"
#include <cublas_v2.h>

namespace apxinf::attention {
namespace {
constexpr size_t kBlasWorkspace = 4 * 1024 * 1024;
void check_blas(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw Failure(APXINF_STATUS_PROVIDER_ERROR, "cuBLAS wide Attention failed: " + std::to_string(status));
}
struct WideState {
  cublasHandle_t handle = nullptr;
  float* probabilities = nullptr;
  float* value = nullptr;
  float* output = nullptr;
  void* workspace = nullptr;
  ~WideState() {
    if (handle) cublasDestroy(handle);
    if (probabilities) cudaFree(probabilities);
    if (value) cudaFree(value);
    if (output) cudaFree(output);
    if (workspace) cudaFree(workspace);
  }
};
}
size_t wide_bf16_resource_requirements(const Spec& spec, int) {
  // The registry bounds both token axes to 4096 and head_dim to 384.
  return size_t(spec.batch) * size_t(spec.query_tokens * spec.key_tokens +
                (spec.query_tokens + spec.key_tokens) * spec.head_dim) * sizeof(float) + kBlasWorkspace;
}
void prepare_wide_bf16(Execution& execution) {
  const size_t bytes = wide_bf16_resource_requirements(execution.spec, execution.configuration);
  if (bytes > execution.resource_limit)
    throw Failure(APXINF_STATUS_UNSUPPORTED, "wide BF16 Attention workspace exceeds policy");
  auto state = std::make_unique<WideState>();
  const auto& s = execution.spec;
  check_cuda(cudaMalloc(&state->probabilities, size_t(s.batch * s.query_tokens * s.key_tokens) * sizeof(float)));
  check_cuda(cudaMalloc(&state->value, size_t(s.batch * s.key_tokens * s.head_dim) * sizeof(float)));
  check_cuda(cudaMalloc(&state->output, size_t(s.batch * s.query_tokens * s.head_dim) * sizeof(float)));
  check_cuda(cudaMalloc(&state->workspace, kBlasWorkspace));
  check_blas(cublasCreate(&state->handle));
  check_blas(cublasSetStream(state->handle, static_cast<cudaStream_t>(execution.bindings.stream)));
  check_blas(cublasSetMathMode(state->handle, CUBLAS_PEDANTIC_MATH));
  check_blas(cublasSetWorkspace(state->handle, state->workspace, kBlasWorkspace));
  execution.resource_bytes = bytes;
  execution.provider_state = state.release();
}
cudaError_t launch_wide_bf16(Execution& execution) {
  auto& state = *static_cast<WideState*>(execution.provider_state);
  const auto& b = execution.bindings;
  const int m = execution.spec.query_tokens, n = execution.spec.key_tokens, d = execution.spec.head_dim;
  const int batch = execution.spec.batch;
  const auto stream = static_cast<cudaStream_t>(b.stream);
  const float one = 1.f, zero = 0.f;
  check_blas(cublasGemmStridedBatchedEx(state.handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, d,
      &b.scale, b.key, CUDA_R_16BF, d, int64_t(n) * d,
      b.query, CUDA_R_16BF, d, int64_t(m) * d, &zero,
      state.probabilities, CUDA_R_32F, n, int64_t(n) * m, batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  if (execution.configuration == 0)
    kernels::wide_softmax_serial<<<(batch * m + 127) / 128, 128, 0, stream>>>(state.probabilities, batch * m, n);
  else
    kernels::wide_softmax_parallel<<<batch * m, 256, 0, stream>>>(state.probabilities, n);
  kernels::wide_bf16_to_float<<<(batch * n * d + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(b.value), state.value, batch * n * d);
  check_cuda(cudaGetLastError());
  // Preserve FP32 probabilities and FP32 V accumulation, with no TF32 reduction.
  check_blas(cublasGemmStridedBatchedEx(state.handle, CUBLAS_OP_N, CUBLAS_OP_N, d, m, n,
      &one, state.value, CUDA_R_32F, d, int64_t(n) * d,
      state.probabilities, CUDA_R_32F, n, int64_t(n) * m, &zero,
      state.output, CUDA_R_32F, d, int64_t(m) * d, batch, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
  kernels::wide_float_to_bf16<<<(batch * m * d + 255) / 256, 256, 0, stream>>>(
      state.output, static_cast<__nv_bfloat16*>(b.output), batch * m * d);
  return cudaGetLastError();
}
void destroy_wide_bf16(Execution& execution) noexcept {
  delete static_cast<WideState*>(execution.provider_state);
  execution.provider_state = nullptr;
}
}  // namespace apxinf::attention
