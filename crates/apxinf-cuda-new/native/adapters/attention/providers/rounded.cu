#include "../internal.h"
#include "../../../kernels/attention/rounded.cuh"
#include <cublas_v2.h>
#include <algorithm>
#include <limits>

namespace apxinf::attention {
namespace {
constexpr size_t kBlasWorkspace = 4*1024*1024;
void check_blas(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw Failure(APXINF_STATUS_PROVIDER_ERROR, "rounded attention cuBLAS status " + std::to_string(status));
}
size_t product(std::initializer_list<int64_t> dimensions) {
  size_t size=sizeof(__nv_bfloat16);
  for (auto d: dimensions) {
    if (d<=0 || static_cast<uint64_t>(d)>SIZE_MAX/size)
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "rounded attention size overflow");
    size *= static_cast<size_t>(d);
  }
  return size;
}
struct State {
  cublasHandle_t handle=nullptr;
  void* scores=nullptr;
  void* output=nullptr;
  void* workspace=nullptr;
  ~State() {
    if (handle) cublasDestroy(handle);
    if (scores) cudaFree(scores);
    if (output) cudaFree(output);
    if (workspace) cudaFree(workspace);
  }
};

template<class T>
cudaError_t enqueue(Execution& execution, cudaDataType_t dtype) {
  auto& state=*static_cast<State*>(execution.provider_state);
  const auto& s=execution.spec;
  const auto& b=execution.bindings;
  const auto stream=static_cast<cudaStream_t>(b.stream);
  const int q=static_cast<int>(s.query_tokens), k=static_cast<int>(s.key_tokens);
  const int h=static_cast<int>(s.query_heads), d=static_cast<int>(s.head_dim);
  const int64_t score_stride=static_cast<int64_t>(q)*k;
  const int64_t output_stride=static_cast<int64_t>(q)*d;
  const float zero=0.0f, one=1.0f;
  for (int64_t batch=0; batch<s.batch; ++batch) {
    // Transposed column-major interpretation of BSHD input: K^T Q gives
    // contiguous [heads, queries, keys] materialized low-precision logits.
    check_blas(cublasGemmStridedBatchedEx(state.handle, CUBLAS_OP_T, CUBLAS_OP_N,
        k, q, d, &b.scale,
        static_cast<const T*>(b.key)+batch*k*h*d, dtype, h*d, d,
        static_cast<const T*>(b.query)+batch*q*h*d, dtype, h*d, d,
        &zero, static_cast<T*>(state.scores)+batch*h*score_stride, dtype, k,
        score_stride, h, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }
  if (k<=2048) {
    const int rows=static_cast<int>(s.batch*h*q);
    rounded_kernels::bias_softmax_warp<T><<<(static_cast<unsigned>(rows)+3u)/4,dim3(32,4),0,stream>>>(
        static_cast<T*>(state.scores),static_cast<const T*>(b.bias),k,rows);
  } else {
    rounded_kernels::bias_softmax<T><<<static_cast<unsigned>(s.batch*h*q),256,0,stream>>>(
        static_cast<T*>(state.scores), static_cast<const T*>(b.bias), k);
  }
  auto status=cudaGetLastError();
  if (status!=cudaSuccess) return status;
  for (int64_t batch=0; batch<s.batch; ++batch) {
    check_blas(cublasGemmStridedBatchedEx(state.handle, CUBLAS_OP_N, CUBLAS_OP_N,
        d, q, k, &one,
        static_cast<const T*>(b.value)+batch*k*h*d, dtype, h*d, d,
        static_cast<T*>(state.scores)+batch*h*score_stride, dtype, k, score_stride,
        &zero, static_cast<T*>(state.output)+batch*h*output_stride, dtype, d,
        output_stride, h, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }
  const int64_t count=s.batch*h*output_stride;
  const int blocks=static_cast<int>(std::min<int64_t>(4096,(count+255)/256));
  rounded_kernels::unpack_heads<T><<<blocks,256,0,stream>>>(static_cast<const T*>(state.output),
      static_cast<T*>(b.output),count,q,h,d);
  return cudaGetLastError();
}
} // namespace

size_t rounded_resource_requirements(const Spec& s, int) {
  const size_t scores=product({s.batch,s.query_heads,s.query_tokens,s.key_tokens});
  const size_t output=product({s.batch,s.query_heads,s.query_tokens,s.head_dim});
  if (scores>SIZE_MAX-kBlasWorkspace || output>SIZE_MAX-kBlasWorkspace-scores)
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "rounded attention workspace overflow");
  return scores+output+kBlasWorkspace;
}
void prepare_rounded(Execution& execution) {
  const auto& s=execution.spec;
  const size_t bytes=rounded_resource_requirements(s,execution.configuration);
  if (bytes>execution.resource_limit)
    throw Failure(APXINF_STATUS_UNSUPPORTED, "rounded attention workspace exceeds policy");
  auto state=std::make_unique<State>();
  check_cuda(cudaMalloc(&state->scores,product({s.batch,s.query_heads,s.query_tokens,s.key_tokens})));
  check_cuda(cudaMalloc(&state->output,product({s.batch,s.query_heads,s.query_tokens,s.head_dim})));
  check_cuda(cudaMalloc(&state->workspace,kBlasWorkspace));
  check_blas(cublasCreate(&state->handle));
  check_blas(cublasSetStream(state->handle,static_cast<cudaStream_t>(execution.bindings.stream)));
  check_blas(cublasSetWorkspace(state->handle,state->workspace,kBlasWorkspace));
  check_blas(cublasSetMathMode(state->handle,CUBLAS_PEDANTIC_MATH));
  check_blas(cublasSetAtomicsMode(state->handle,CUBLAS_ATOMICS_NOT_ALLOWED));
  execution.resource_bytes=bytes;
  execution.provider_state=state.release();
}
cudaError_t launch_rounded(Execution& execution) {
  return execution.spec.dtype==APXINF_DTYPE_BF16
      ? enqueue<__nv_bfloat16>(execution,CUDA_R_16BF)
      : enqueue<__half>(execution,CUDA_R_16F);
}
void destroy_rounded(Execution& execution) noexcept {
  delete static_cast<State*>(execution.provider_state);
  execution.provider_state=nullptr;
}
} // namespace apxinf::attention
