#include "vendor.h"
#include "../../../kernels/custom/gemm.cuh"

namespace apxinf::gemm {
namespace {

struct CublasState {
  vendor::CommonResources common;
  cublasHandle_t handle = nullptr;
  void* workspace = nullptr;
  size_t workspace_bytes = 0;

  ~CublasState() {
    release_resources();
    if (handle != nullptr) cublasDestroy(handle);
  }

  void release_resources() noexcept {
    if (workspace != nullptr) cudaFree(workspace);
    workspace = nullptr;
    common.release();
  }
};

CublasState& provider(Execution& state) {
  return *static_cast<CublasState*>(state.provider_state);
}

}  // namespace

size_t cublas_resource_requirements(const Spec& spec) {
  return vendor::common_resource_requirements(spec, false) +
         4 * 1024 * 1024;
}

void prepare_cublas(Execution& state) {
  auto resources = std::make_unique<CublasState>();
  vendor::allocate_common_resources(state.spec, resources->common, false);
  check_cublas(cublasCreate(&resources->handle));
  check_cublas(cublasSetMathMode(resources->handle, CUBLAS_PEDANTIC_MATH));
  resources->workspace_bytes = 4 * 1024 * 1024;
  check_cuda(cudaMalloc(&resources->workspace, resources->workspace_bytes));
  state.resource_bytes =
      resources->common.resource_bytes + resources->workspace_bytes;
  state.provider_state = resources.release();
}

void destroy_cublas(Execution& state) noexcept {
  delete static_cast<CublasState*>(state.provider_state);
  state.provider_state = nullptr;
}

cudaError_t launch_cublas(Execution& state) {
  const auto& bindings = state.bindings;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  const auto& spec = state.spec;
  auto& resources = provider(state);
  const void* activation = bindings.a;
  const void* weight = bindings.b;
  if (resources.common.unpack_a != nullptr) {
    check_cuda(apxinf::cuda::custom::unpack_gemm(
        activation, resources.common.unpack_a,
        resources.common.projection_dtype, spec.a_dtype,
        spec.m, spec.k, APXINF_GEMM_LAYOUT_KN, stream));
    check_cuda(apxinf::cuda::custom::unpack_gemm(
        weight, resources.common.unpack_b,
        resources.common.projection_dtype, spec.b_dtype, spec.k, spec.n,
        APXINF_GEMM_LAYOUT_KN, stream));
    activation = resources.common.unpack_a;
    weight = resources.common.unpack_b;
  }

  void* projection = resources.common.projection != nullptr
                         ? resources.common.projection
                         : bindings.output;
  const float alpha =
      has_row_channel_scales(spec) ? 1.0F : bindings.alpha;
  const float beta = 0.0F;
  const int32_t integer_alpha = 1;
  const int32_t integer_beta = 0;
  const void* alpha_pointer = spec.a_dtype == APXINF_DTYPE_I8
                                  ? static_cast<const void*>(&integer_alpha)
                                  : static_cast<const void*>(&alpha);
  const void* beta_pointer = spec.a_dtype == APXINF_DTYPE_I8
                                 ? static_cast<const void*>(&integer_beta)
                                 : static_cast<const void*>(&beta);
  const cudaDataType_t data_type =
      resources.common.projection_dtype == 5
          ? CUDA_R_32I
          : resources.common.projection_dtype == APXINF_DTYPE_F32
                ? CUDA_R_32F
                : resources.common.projection_dtype == APXINF_DTYPE_F16
                      ? CUDA_R_16F
                      : CUDA_R_16BF;
  check_cublas(cublasSetStream(resources.handle, stream));
  check_cublas(cublasSetWorkspace(resources.handle, resources.workspace,
                                  resources.workspace_bytes));
  check_cublas(cublasGemmEx(
      resources.handle, CUBLAS_OP_N, CUBLAS_OP_N, spec.n, spec.m, spec.k,
      alpha_pointer, weight,
      spec.a_dtype == APXINF_DTYPE_I8 ? CUDA_R_8I : data_type,
      spec.n, activation,
      spec.a_dtype == APXINF_DTYPE_I8 ? CUDA_R_8I : data_type, spec.k,
      beta_pointer, projection, data_type, spec.n,
      spec.a_dtype == APXINF_DTYPE_I8 ? CUBLAS_COMPUTE_32I
                                      : CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT));
  return vendor::launch_postprocess(spec, resources.common, bindings,
                                    projection);
}

}  // namespace apxinf::gemm
