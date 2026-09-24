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
  cublasLtMatrixLayout_t c_layout = nullptr;
  cublasLtMatrixLayout_t d_layout = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  bool has_algorithm = false;
  bool direct_fp8_gelu = false;
  void* c_buffer = nullptr;
  float* output_scale = nullptr;
  void* workspace = nullptr;
  void* zero_bias = nullptr;
  size_t workspace_bytes = 0;

  ~CublasLtState() {
    release_resources();
    if (a_layout != nullptr) cublasLtMatrixLayoutDestroy(a_layout);
    if (b_layout != nullptr) cublasLtMatrixLayoutDestroy(b_layout);
    if (c_layout != nullptr) cublasLtMatrixLayoutDestroy(c_layout);
    if (d_layout != nullptr) cublasLtMatrixLayoutDestroy(d_layout);
    if (operation != nullptr) cublasLtMatmulDescDestroy(operation);
    if (handle != nullptr) cublasLtDestroy(handle);
  }

  void release_resources() noexcept {
    if (workspace != nullptr) cudaFree(workspace);
    if (zero_bias != nullptr) cudaFree(zero_bias);
    if (c_buffer != nullptr) cudaFree(c_buffer);
    if (output_scale != nullptr) cudaFree(output_scale);
    workspace = nullptr;
    zero_bias = nullptr;
    c_buffer = nullptr;
    output_scale = nullptr;
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
  const bool needs_zero_bias =
      native_fp8 &&
      spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS_RESIDUAL &&
      spec.bias_alignment == 0;
  return vendor::common_resource_requirements(spec, native_fp8) +
         (needs_zero_bias
              ? static_cast<size_t>(spec.n) * dtype_bytes(APXINF_DTYPE_F16)
              : 0);
}

void prepare_cublaslt_impl(Execution& state, bool native_fp8,
                           bool direct_fp8_gelu) {
  auto resources = std::make_unique<CublasLtState>();
  const auto& spec = state.spec;
  resources->common.projection_dtype =
      vendor::common_projection_dtype(spec);
  const size_t c_buffer_bytes =
      direct_fp8_gelu ? static_cast<size_t>(spec.m * spec.n) * sizeof(half) : 0;
  const size_t fixed_bytes = direct_fp8_gelu
                                 ? c_buffer_bytes + sizeof(float)
                                 : common_requirement(spec, native_fp8);
  const size_t available_workspace = state.resource_limit - fixed_bytes;
  const cudaDataType_t projection_type =
      direct_fp8_gelu ? CUDA_R_16F
                      : projection_cuda_dtype(resources->common);
  const cudaDataType_t output_type =
      direct_fp8_gelu ? CUDA_R_8F_E4M3 : projection_type;
  const cudaDataType_t input_type =
      spec.a_dtype == APXINF_DTYPE_I8
          ? CUDA_R_8I
          : native_fp8 ? CUDA_R_8F_E4M3 : projection_type;
  const cublasComputeType_t compute_type =
      spec.a_dtype == APXINF_DTYPE_I8 ? CUBLAS_COMPUTE_32I
                                      : CUBLAS_COMPUTE_32F;
  const cudaDataType_t scale_type =
      spec.a_dtype == APXINF_DTYPE_I8 ? CUDA_R_32I : CUDA_R_32F;
  const cublasOperation_t transpose = CUBLAS_OP_N;

  check_cublas(cublasLtCreate(&resources->handle));
  check_cublas(cublasLtMatmulDescCreate(&resources->operation, compute_type,
                                        scale_type));
  check_cublas(cublasLtMatmulDescSetAttribute(
      resources->operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose,
      sizeof(transpose)));
  const bool native_bias_residual =
      native_fp8 &&
      spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS_RESIDUAL &&
      spec.quantization == APXINF_GEMM_QUANT_FP8_UNIT_SCALE &&
      spec.output_scale_is_unit != 0;
  const bool native_bias =
      native_fp8 &&
      spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS &&
      spec.quantization == APXINF_GEMM_QUANT_FP8_UNIT_SCALE &&
      spec.output_scale_is_unit != 0;
  if (native_bias_residual && state.bindings.bias == nullptr) {
    check_cuda(cudaMalloc(&resources->zero_bias,
                          static_cast<size_t>(spec.n) *
                              dtype_bytes(APXINF_DTYPE_F16)));
    check_cuda(cudaMemset(resources->zero_bias, 0,
                          static_cast<size_t>(spec.n) *
                              dtype_bytes(APXINF_DTYPE_F16)));
  }
  if (direct_fp8_gelu) {
    const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;
    check_cublas(cublasLtMatmulDescSetAttribute(
        resources->operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
        sizeof(epilogue)));
    const cudaDataType_t bias_type = CUDA_R_16F;
    check_cublas(cublasLtMatmulDescSetAttribute(
        resources->operation, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
        &bias_type, sizeof(bias_type)));
    check_cublas(cublasLtMatmulDescSetAttribute(
        resources->operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
        &state.bindings.bias, sizeof(state.bindings.bias)));
  } else if (native_bias || native_bias_residual) {
    const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
    check_cublas(cublasLtMatmulDescSetAttribute(
        resources->operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
        sizeof(epilogue)));
    const cudaDataType_t bias_type = projection_type;
    check_cublas(cublasLtMatmulDescSetAttribute(
        resources->operation, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
        &bias_type, sizeof(bias_type)));
    const void* bias = state.bindings.bias != nullptr
                           ? state.bindings.bias
                           : resources->zero_bias;
    check_cublas(cublasLtMatmulDescSetAttribute(
        resources->operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
        &bias, sizeof(bias)));
  }
  check_cublas(cublasLtMatrixLayoutCreate(
      &resources->a_layout, input_type,
      transpose == CUBLAS_OP_T ? spec.k : spec.n,
      transpose == CUBLAS_OP_T ? spec.n : spec.k,
      transpose == CUBLAS_OP_T ? spec.k : spec.n));
  check_cublas(cublasLtMatrixLayoutCreate(
      &resources->b_layout, input_type, spec.k, spec.m, spec.k));
  check_cublas(cublasLtMatrixLayoutCreate(&resources->c_layout,
                                          projection_type, spec.n, spec.m,
                                          spec.n));
  check_cublas(cublasLtMatrixLayoutCreate(&resources->d_layout,
                                          output_type, spec.n, spec.m,
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
      resources->b_layout, resources->c_layout,
      resources->d_layout, preference, 8, candidates, &candidate_count);
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
  if (direct_fp8_gelu) {
    check_cuda(cudaMalloc(&resources->c_buffer, c_buffer_bytes));
    check_cuda(cudaMalloc(&resources->output_scale, sizeof(float)));
    const float inverse_output_scale = 1.0F / state.bindings.output_scale;
    check_cuda(cudaMemcpy(resources->output_scale, &inverse_output_scale,
                          sizeof(float), cudaMemcpyHostToDevice));
    check_cublas(cublasLtMatmulDescSetAttribute(
        resources->operation, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
        &resources->output_scale, sizeof(resources->output_scale)));
    resources->direct_fp8_gelu = true;
    resources->common.resource_bytes = fixed_bytes;
  } else {
    vendor::allocate_common_resources(spec, resources->common, native_fp8);
  }
  if (resources->workspace_bytes != 0) {
    check_cuda(cudaMalloc(&resources->workspace, resources->workspace_bytes));
  }
  state.resource_bytes =
      resources->common.resource_bytes + resources->workspace_bytes +
      (resources->zero_bias != nullptr
           ? static_cast<size_t>(spec.n) * dtype_bytes(APXINF_DTYPE_F16)
           : 0);
  state.provider_state = resources.release();
}

}  // namespace

size_t cublaslt_resource_requirements(const Spec& spec) {
  return common_requirement(spec, false);
}

size_t cublaslt_native_fp8_resource_requirements(const Spec& spec) {
  return common_requirement(spec, true);
}

size_t cublaslt_native_fp8_gelu_resource_requirements(const Spec& spec) {
  return static_cast<size_t>(spec.m * spec.n) * sizeof(half) + sizeof(float);
}

void prepare_cublaslt(Execution& state) {
  prepare_cublaslt_impl(state, false, false);
}

void prepare_cublaslt_native_fp8(Execution& state) {
  prepare_cublaslt_impl(state, true, false);
}

void prepare_cublaslt_native_fp8_gelu(Execution& state) {
  prepare_cublaslt_impl(state, true, true);
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
  void* projection = resources.direct_fp8_gelu
                         ? resources.c_buffer
                         : resources.common.projection != nullptr
                               ? resources.common.projection
                               : bindings.output;
  void* output = resources.direct_fp8_gelu ? bindings.output : projection;
  const float alpha =
      has_row_channel_scales(spec) ? 1.0F : bindings.alpha;
  const bool native_bias_residual =
      spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS_RESIDUAL &&
      resources.common.projection == nullptr;
  const float beta = native_bias_residual ? 1.0F : 0.0F;
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
      native_bias_residual ? bindings.residual : projection,
      resources.c_layout, output,
      resources.d_layout, &resources.algorithm, resources.workspace,
      resources.workspace_bytes, stream));
  if (resources.direct_fp8_gelu) return cudaSuccess;
  return vendor::launch_postprocess(spec, resources.common, bindings,
                                    projection);
}

}  // namespace apxinf::gemm
