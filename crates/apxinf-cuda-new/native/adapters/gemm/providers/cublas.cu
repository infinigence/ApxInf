#include "vendor.h"
#include "../../../kernels/custom/gemm.cuh"
#include "../../../kernels/elementwise/broadcast_bf16_f32.cuh"

namespace apxinf::gemm {
namespace {

struct CublasState {
  vendor::CommonResources common;
  cublasHandle_t handle = nullptr;
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
  void* dequant_a = nullptr;
  void* dequant_b = nullptr;

  ~CublasState() {
    release_resources();
    if (handle != nullptr) cublasDestroy(handle);
  }

  void release_resources() noexcept {
    if (workspace != nullptr) cudaFree(workspace);
    if (dequant_a != nullptr) cudaFree(dequant_a);
    if (dequant_b != nullptr) cudaFree(dequant_b);
    workspace = nullptr;
    dequant_a = nullptr;
    dequant_b = nullptr;
    common.release();
  }
};

CublasState& provider(Execution& state) {
  return *static_cast<CublasState*>(state.provider_state);
}

bool needs_safe_dequantization(const Spec& spec) {
  return spec.quantization == APXINF_GEMM_QUANT_W8A8_ROW_CHANNEL &&
         spec.b_layout == APXINF_GEMM_LAYOUT_KN;
}

bool uses_checkpoint_bias_numerics(const Spec& spec) {
  return spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS &&
         spec.quantization == APXINF_GEMM_QUANT_NONE &&
         spec.b_layout == APXINF_GEMM_LAYOUT_NK &&
         spec.a_dtype == APXINF_DTYPE_BF16 &&
         spec.b_dtype == APXINF_DTYPE_BF16 &&
         spec.output_dtype == APXINF_DTYPE_BF16 &&
         spec.accumulation_dtype == APXINF_DTYPE_F32;
}

bool uses_legacy_cublas_numerics(const Spec& spec) {
  return uses_checkpoint_bias_numerics(spec) ||
         spec.numerics == APXINF_GEMM_NUMERICS_LEGACY_CUBLAS;
}

cudaDataType_t input_cuda_dtype(uint32_t dtype) {
  switch (dtype) {
    case APXINF_DTYPE_F32: return CUDA_R_32F;
    case APXINF_DTYPE_F16: return CUDA_R_16F;
    case APXINF_DTYPE_BF16: return CUDA_R_16BF;
    case APXINF_DTYPE_E4M3: return CUDA_R_8F_E4M3;
    case APXINF_DTYPE_I8: return CUDA_R_8I;
    default:
      throw Failure(APXINF_STATUS_UNSUPPORTED,
                    "cuBLAS GEMM input dtype is unsupported");
  }
}

Spec cublas_compute_spec(const Spec& spec) {
  Spec compute = spec;
  if (needs_safe_dequantization(spec)) {
    // Some devices accept an INT8 cuBLAS plan but reject this row/channel
    // scaled W8A8 combination at enqueue. BF16 is the portable cuBLAS
    // baseline; apply the L3 scales while converting both operands.
    compute.a_dtype = APXINF_DTYPE_BF16;
    compute.b_dtype = APXINF_DTYPE_BF16;
    compute.accumulation_dtype = APXINF_DTYPE_F32;
    compute.quantization = APXINF_GEMM_QUANT_NONE;
  }
  return compute;
}

size_t dequantization_resource_requirements(const Spec& spec) {
  if (!needs_safe_dequantization(spec)) return 0;
  return static_cast<size_t>(spec.m * spec.k + spec.k * spec.n) *
         dtype_bytes(APXINF_DTYPE_BF16);
}

}  // namespace

size_t cublas_resource_requirements(const Spec& spec) {
  return dequantization_resource_requirements(spec) +
         vendor::common_resource_requirements(cublas_compute_spec(spec), false) +
         4 * 1024 * 1024;
}

void prepare_cublas(Execution& state) {
  auto resources = std::make_unique<CublasState>();
  const Spec compute_spec = cublas_compute_spec(state.spec);
  vendor::allocate_common_resources(compute_spec, resources->common, false);
  if (needs_safe_dequantization(state.spec)) {
    const size_t a_bytes = static_cast<size_t>(state.spec.m * state.spec.k) *
                           dtype_bytes(APXINF_DTYPE_BF16);
    const size_t b_bytes = static_cast<size_t>(state.spec.k * state.spec.n) *
                           dtype_bytes(APXINF_DTYPE_BF16);
    check_cuda(cudaMalloc(&resources->dequant_a, a_bytes));
    check_cuda(cudaMalloc(&resources->dequant_b, b_bytes));
  }
  check_cublas(cublasCreate(&resources->handle));
  check_cublas(cublasSetMathMode(
      resources->handle, uses_legacy_cublas_numerics(state.spec)
                             ? CUBLAS_DEFAULT_MATH
                             : CUBLAS_PEDANTIC_MATH));
  resources->workspace_bytes = 4 * 1024 * 1024;
  check_cuda(cudaMalloc(&resources->workspace, resources->workspace_bytes));
  state.resource_bytes =
      resources->common.resource_bytes + resources->workspace_bytes +
      dequantization_resource_requirements(state.spec);
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
  const Spec compute_spec = cublas_compute_spec(spec);
  auto& resources = provider(state);
  const void* activation = bindings.a;
  const void* weight = bindings.b;
  if (needs_safe_dequantization(spec)) {
    check_cuda(apxinf::cuda_new::custom::dequantize_i8_gemm(
        activation, resources.dequant_a, bindings.a_scales, spec.m, spec.k,
        true, stream));
    check_cuda(apxinf::cuda_new::custom::dequantize_i8_gemm(
        weight, resources.dequant_b, bindings.b_scales, spec.k, spec.n, false,
        stream));
    activation = resources.dequant_a;
    weight = resources.dequant_b;
  } else if (resources.common.unpack_a != nullptr) {
    check_cuda(apxinf::cuda_new::custom::unpack_gemm(
        activation, resources.common.unpack_a,
        resources.common.projection_dtype, compute_spec.a_dtype, compute_spec.m,
        compute_spec.k, APXINF_GEMM_LAYOUT_KN, stream));
    check_cuda(apxinf::cuda_new::custom::unpack_gemm(
        weight, resources.common.unpack_b,
        resources.common.projection_dtype, compute_spec.b_dtype, compute_spec.k,
        compute_spec.n, APXINF_GEMM_LAYOUT_KN, stream));
    activation = resources.common.unpack_a;
    weight = resources.common.unpack_b;
  }

  void* projection = resources.common.projection != nullptr
                         ? resources.common.projection
                         : bindings.output;
  const bool checkpoint_bias = uses_checkpoint_bias_numerics(compute_spec);
  if (checkpoint_bias) {
    const int64_t count = compute_spec.m * compute_spec.n;
    const int blocks = static_cast<int>(
        std::min<int64_t>((count + 255) / 256, 4096));
    broadcast_bf16_f32_rows_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(bindings.bias),
        static_cast<float*>(projection), count,
        static_cast<int>(compute_spec.n));
    check_cuda(cudaGetLastError());
  }
  const float alpha =
      has_row_channel_scales(compute_spec) ? 1.0F : bindings.alpha;
  const float beta = checkpoint_bias ? 1.0F : 0.0F;
  const int32_t integer_alpha = 1;
  const int32_t integer_beta = 0;
  const void* alpha_pointer = compute_spec.a_dtype == APXINF_DTYPE_I8
                                  ? static_cast<const void*>(&integer_alpha)
                                  : static_cast<const void*>(&alpha);
  const void* beta_pointer = compute_spec.a_dtype == APXINF_DTYPE_I8
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
  const cudaDataType_t input_type =
      resources.common.unpack_a != nullptr
          ? data_type
          : input_cuda_dtype(compute_spec.a_dtype);
  check_cublas(cublasSetStream(resources.handle, stream));
  check_cublas(cublasSetWorkspace(resources.handle, resources.workspace,
                                  resources.workspace_bytes));
  const bool output_major_weight =
      compute_spec.b_layout == APXINF_GEMM_LAYOUT_NK;
  check_cublas(cublasGemmEx(
      resources.handle,
      output_major_weight ? CUBLAS_OP_T : CUBLAS_OP_N,
      CUBLAS_OP_N, compute_spec.n, compute_spec.m, compute_spec.k,
      alpha_pointer, weight,
      compute_spec.a_dtype == APXINF_DTYPE_I8 ? CUDA_R_8I : input_type,
      output_major_weight ? compute_spec.k : compute_spec.n, activation,
      compute_spec.a_dtype == APXINF_DTYPE_I8 ? CUDA_R_8I : input_type,
      compute_spec.k, beta_pointer, projection, data_type, compute_spec.n,
      compute_spec.a_dtype == APXINF_DTYPE_I8 ? CUBLAS_COMPUTE_32I
                                              : CUBLAS_COMPUTE_32F,
      output_major_weight &&
              compute_spec.quantization == APXINF_GEMM_QUANT_W8A8_ROW_CHANNEL
          ? CUBLAS_GEMM_DEFAULT_TENSOR_OP
          : CUBLAS_GEMM_DEFAULT));
  auto postprocess_bindings = bindings;
  if (checkpoint_bias) postprocess_bindings.bias = nullptr;
  return vendor::launch_postprocess(compute_spec, resources.common,
                                    postprocess_bindings, projection);
}

}  // namespace apxinf::gemm
