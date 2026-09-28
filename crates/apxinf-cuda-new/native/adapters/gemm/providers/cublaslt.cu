#include "vendor.h"
#include "../../../kernels/custom/gemm.cuh"

namespace apxinf::gemm {
namespace {

struct CublasLtState {
  vendor::CommonResources common;
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr;
  cublasLtMatrixLayout_t b_layout = nullptr;
  cublasLtMatrixLayout_t output_layout = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  bool has_algorithm = false;
  bool native_fp4 = false;
  void* workspace = nullptr;
  size_t workspace_bytes = 0;

  ~CublasLtState() {
    release_resources();
    if (a_layout != nullptr) cublasLtMatrixLayoutDestroy(a_layout);
    if (b_layout != nullptr) cublasLtMatrixLayoutDestroy(b_layout);
    if (output_layout != nullptr) cublasLtMatrixLayoutDestroy(output_layout);
    if (operation != nullptr) cublasLtMatmulDescDestroy(operation);
    if (handle != nullptr) cublasLtDestroy(handle);
  }

  void release_resources() noexcept {
    if (workspace != nullptr) cudaFree(workspace);
    workspace = nullptr;
    common.release();
  }
};

CublasLtState& provider(Execution& state) {
  return *static_cast<CublasLtState*>(state.provider_state);
}

const CublasLtState& provider(const Execution& state) {
  return *static_cast<const CublasLtState*>(state.provider_state);
}

cudaDataType_t projection_cuda_dtype(const vendor::CommonResources& common) {
  if (common.projection_dtype == APXINF_DTYPE_I32) return CUDA_R_32I;
  if (common.projection_dtype == APXINF_DTYPE_F32) return CUDA_R_32F;
  if (common.projection_dtype == APXINF_DTYPE_F16) return CUDA_R_16F;
  return CUDA_R_16BF;
}

size_t common_requirement(const Spec& spec, bool native_fp8) {
  return vendor::common_resource_requirements(spec, native_fp8);
}

void prepare_cublaslt_impl(Execution& state, bool native_fp8,
                          bool native_fp4 = false) {
  auto resources = std::make_unique<CublasLtState>();
  const auto& spec = state.spec;
  resources->native_fp4 = native_fp4;
  resources->common.projection_dtype =
      native_fp4 ? APXINF_DTYPE_BF16 : vendor::common_projection_dtype(spec);
  const size_t fixed_bytes = native_fp4 ? 0 : common_requirement(spec, native_fp8);
  const size_t available_workspace = state.resource_limit - fixed_bytes;
  const cudaDataType_t projection_type =
      projection_cuda_dtype(resources->common);
  cudaDataType_t input_type =
      spec.a_dtype == APXINF_DTYPE_I8
          ? CUDA_R_8I
          : native_fp8 ? CUDA_R_8F_E4M3 : projection_type;
  const cublasComputeType_t compute_type =
      spec.a_dtype == APXINF_DTYPE_I8 ? CUBLAS_COMPUTE_32I
                                      : CUBLAS_COMPUTE_32F;
  const cudaDataType_t scale_type =
      spec.a_dtype == APXINF_DTYPE_I8 ? CUDA_R_32I : CUDA_R_32F;
  const cublasOperation_t transpose = native_fp4 ? CUBLAS_OP_T : CUBLAS_OP_N;
#if CUDART_VERSION >= 12080
  if (native_fp4) input_type = CUDA_R_4F_E2M1;
#else
  if (native_fp4) {
    throw Failure(APXINF_STATUS_UNSUPPORTED, "NVFP4 requires CUDA 12.8 or newer");
  }
#endif

  check_cublas(cublasLtCreate(&resources->handle));
  check_cublas(cublasLtMatmulDescCreate(&resources->operation, compute_type,
                                        scale_type));
  check_cublas(cublasLtMatmulDescSetAttribute(
      resources->operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose,
      sizeof(transpose)));
#if CUDART_VERSION >= 12080
  if (native_fp4) {
    const cublasLtMatmulMatrixScale_t scale_mode =
        CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    check_cublas(cublasLtMatmulDescSetAttribute(
        resources->operation, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
        &scale_mode, sizeof(scale_mode)));
    check_cublas(cublasLtMatmulDescSetAttribute(
        resources->operation, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
        &scale_mode, sizeof(scale_mode)));
    check_cublas(cublasLtMatmulDescSetAttribute(
        resources->operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        &state.bindings.b_block_scales, sizeof(state.bindings.b_block_scales)));
    check_cublas(cublasLtMatmulDescSetAttribute(
        resources->operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        &state.bindings.a_block_scales, sizeof(state.bindings.a_block_scales)));
  }
#endif
  check_cublas(cublasLtMatrixLayoutCreate(
      &resources->a_layout, input_type,
      transpose == CUBLAS_OP_T ? spec.k : spec.n,
      transpose == CUBLAS_OP_T ? spec.n : spec.k,
      transpose == CUBLAS_OP_T ? spec.k : spec.n));
  check_cublas(cublasLtMatrixLayoutCreate(
      &resources->b_layout, input_type, spec.k, spec.m, spec.k));
  check_cublas(cublasLtMatrixLayoutCreate(&resources->output_layout,
                                          projection_type, spec.n, spec.m,
                                          spec.n));

  cublasLtMatmulPreference_t preference = nullptr;
  check_cublas(cublasLtMatmulPreferenceCreate(&preference));
  const size_t workspace_limit = available_workspace;
  cublasStatus_t status = cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_limit, sizeof(workspace_limit));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulPreferenceDestroy(preference);
    check_cublas(status);
  }
  cublasLtMatmulHeuristicResult_t candidates[8]{};
  int candidate_count = 0;
  status = cublasLtMatmulAlgoGetHeuristic(
      resources->handle, resources->operation, resources->a_layout,
      resources->b_layout, resources->output_layout,
      resources->output_layout, preference, 8, candidates, &candidate_count);
  cublasLtMatmulPreferenceDestroy(preference);
  check_cublas(status);
  if (state.configuration >= candidate_count ||
      candidates[state.configuration].state != CUBLAS_STATUS_SUCCESS) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "cuBLASLt heuristic is unavailable");
  }
  resources->algorithm = candidates[state.configuration].algo;
  resources->has_algorithm = true;
  resources->workspace_bytes = candidates[state.configuration].workspaceSize;

  if (resources->workspace_bytes > available_workspace) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "cuBLASLt workspace policy exceeded");
  }

  // Descriptor and heuristic queries do not allocate provider device
  // scratch. Allocate common buffers only after the full requirement is
  // known to fit the caller's policy.
  if (!native_fp4) {
    vendor::allocate_common_resources(spec, resources->common, native_fp8);
  }
  if (resources->workspace_bytes != 0) {
    check_cuda(cudaMalloc(&resources->workspace, resources->workspace_bytes));
  }
  state.resource_bytes =
      resources->common.resource_bytes + resources->workspace_bytes;
  state.provider_state = resources.release();
}

}  // namespace

size_t cublaslt_resource_requirements(const Spec& spec) {
  return common_requirement(spec, false);
}

size_t cublaslt_native_fp8_resource_requirements(const Spec& spec) {
  return common_requirement(spec, true);
}

void prepare_cublaslt(Execution& state) {
  prepare_cublaslt_impl(state, false);
}

void prepare_cublaslt_native_fp8(Execution& state) {
  prepare_cublaslt_impl(state, true);
}

size_t cublaslt_native_fp4_resource_requirements(const Spec&) {
  return 0;
}

void prepare_cublaslt_native_fp4(Execution& state) {
  prepare_cublaslt_impl(state, false, true);
}

void destroy_cublaslt(Execution& state) noexcept {
  delete static_cast<CublasLtState*>(state.provider_state);
  state.provider_state = nullptr;
}

cudaError_t launch_cublaslt(Execution& state) {
  const auto& bindings = state.bindings;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  const auto& spec = state.spec;
  auto& resources = provider(state);
#if CUDART_VERSION >= 12080
  if (resources.native_fp4) {
    check_cublas(cublasLtMatmulDescSetAttribute(
        resources.operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        &bindings.b_block_scales, sizeof(bindings.b_block_scales)));
    check_cublas(cublasLtMatmulDescSetAttribute(
        resources.operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        &bindings.a_block_scales, sizeof(bindings.a_block_scales)));
  }
#endif
  const void* activation = bindings.a;
  const void* weight = bindings.b;
  if (resources.common.unpack_a != nullptr) {
    check_cuda(apxinf::cuda_new::custom::unpack_gemm(
        activation, resources.common.unpack_a,
        resources.common.projection_dtype, spec.a_dtype,
        spec.m, spec.k, APXINF_GEMM_LAYOUT_KN, stream));
    check_cuda(apxinf::cuda_new::custom::unpack_gemm(
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
  check_cublas(cublasLtMatmul(
      resources.handle, resources.operation, alpha_pointer, weight,
      resources.a_layout, activation, resources.b_layout, beta_pointer,
      projection, resources.output_layout, projection,
      resources.output_layout, &resources.algorithm, resources.workspace,
      resources.workspace_bytes, stream));
  return vendor::launch_postprocess(spec, resources.common, bindings,
                                    projection);
}

}  // namespace apxinf::gemm
