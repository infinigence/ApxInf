#include "vendor.h"
#include "../../../kernels/custom/gemm.cuh"

#ifdef APXINF_GEMM_CUTLASS
#include "../../../kernels/cutlass/ops/gemm/gemm_e4m3_sm100.h"
#endif

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
  bool native_fp4 = false;
  bool direct_fp8_gelu = false;
  bool split_geglu = false;
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
                           bool direct_fp8_gelu, bool split_geglu = false,
                           bool native_fp4 = false) {
  auto resources = std::make_unique<CublasLtState>();
  const auto& spec = state.spec;
  resources->native_fp4 = native_fp4;
  const int64_t projection_columns = split_geglu ? spec.n / 2 : spec.n;
  const int64_t projection_stride = spec.n;
  const int heuristic_rank =
      split_geglu ? state.configuration / 3 : state.configuration;
  resources->common.projection_dtype =
      native_fp4 ? APXINF_DTYPE_BF16 : vendor::common_projection_dtype(spec);
  const size_t c_buffer_bytes =
      (direct_fp8_gelu || split_geglu)
          ? static_cast<size_t>(spec.m * projection_stride) * sizeof(half)
          : 0;
  const size_t fixed_bytes = native_fp4 ? 0 : direct_fp8_gelu
                                 ? c_buffer_bytes + sizeof(float)
                                 : split_geglu ? c_buffer_bytes
                                               : common_requirement(spec, native_fp8);
  const size_t available_workspace = state.resource_limit - fixed_bytes;
  const cudaDataType_t projection_type =
      (direct_fp8_gelu || split_geglu)
          ? CUDA_R_16F
          : projection_cuda_dtype(resources->common);
  const cudaDataType_t output_type =
      direct_fp8_gelu ? CUDA_R_8F_E4M3 : projection_type;
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
      transpose == CUBLAS_OP_T ? spec.k : projection_columns,
      transpose == CUBLAS_OP_T ? projection_columns : spec.k,
      transpose == CUBLAS_OP_T ? spec.k : spec.n));
  check_cublas(cublasLtMatrixLayoutCreate(
      &resources->b_layout, input_type, spec.k, spec.m, spec.k));
  check_cublas(cublasLtMatrixLayoutCreate(
      &resources->c_layout, projection_type, projection_columns, spec.m,
      projection_stride));
  check_cublas(cublasLtMatrixLayoutCreate(
      &resources->d_layout, output_type, projection_columns, spec.m,
      projection_stride));

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
  if (heuristic_rank >= candidate_count ||
      candidates[heuristic_rank].state != CUBLAS_STATUS_SUCCESS) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "cuBLASLt heuristic is unavailable");
  }
  resources->algorithm = candidates[heuristic_rank].algo;
  resources->has_algorithm = true;
  resources->workspace_bytes = candidates[heuristic_rank].workspaceSize;

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
  } else if (split_geglu) {
    check_cuda(cudaMalloc(&resources->c_buffer, c_buffer_bytes));
    resources->split_geglu = true;
    resources->common.resource_bytes = fixed_bytes;
  } else if (!native_fp4) {
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

size_t cublaslt_native_fp4_resource_requirements(const Spec&) {
  return 0;
}

void prepare_cublaslt_native_fp4(Execution& state) {
  prepare_cublaslt_impl(state, false, false, false, true);
}

size_t cublaslt_split_geglu_resource_requirements(const Spec& spec) {
  return static_cast<size_t>(spec.m * spec.n) * sizeof(half);
}

void prepare_cublaslt_split_geglu(Execution& state) {
  if (!(state.bindings.alpha > 0.0F)) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "split CUTLASS GeGLU requires positive alpha");
  }
  prepare_cublaslt_impl(state, true, false, true);
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
  void* projection = (resources.direct_fp8_gelu || resources.split_geglu)
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
  if (resources.direct_fp8_gelu || resources.split_geglu) return cudaSuccess;
  return vendor::launch_postprocess(spec, resources.common, bindings,
                                    projection);
}

cudaError_t launch_cublaslt_split_geglu(Execution& state) {
#ifdef APXINF_GEMM_CUTLASS
  check_cuda(launch_cublaslt(state));
  const auto& spec = state.spec;
  const auto& bindings = state.bindings;
  const int status = apxinf::cuda_new::cutlass_ops::fp8_gemm_geglu_e4m3(
      bindings.a, bindings.b, provider(state).c_buffer, bindings.output,
      spec.m, spec.n / 2, spec.k, spec.n, bindings.alpha, bindings.output_scale,
      state.configuration % 3, static_cast<cudaStream_t>(bindings.stream));
  if (status != 0) {
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  "split CUTLASS GeGLU launch status " + std::to_string(status));
  }
  return cudaSuccess;
#else
  return cudaErrorNotSupported;
#endif
}

}  // namespace apxinf::gemm
