#include "vendor.h"
#include "../../cublas_resources.h"
#include "../../../kernels/custom/gemm.cuh"

namespace apxinf::gemm {
namespace {

struct CublasState {
  vendor::CommonResources common;
  std::shared_ptr<providers::CublasStreamResources> shared;
  void* dequant_a = nullptr;
  void* dequant_b = nullptr;

  ~CublasState() {
    release_resources();
  }

  void release_resources() noexcept {
    if (dequant_a != nullptr) cudaFree(dequant_a);
    if (dequant_b != nullptr) cudaFree(dequant_b);
    dequant_a = nullptr;
    dequant_b = nullptr;
    common.release();
  }
};

CublasState& provider(Execution& state) {
  return *static_cast<CublasState*>(state.provider_state);
}

bool needs_safe_dequantization(const Spec& spec) {
  return spec.quantization == APXINF_GEMM_QUANT_W8A8_ROW_CHANNEL;
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

uint32_t cublas_projection_dtype(const Spec& spec) {
  // W8A8's safe dequantization fallback retains its existing BF16 projection;
  // converting its internal compute Spec to None must not widen its epilogue.
  return needs_safe_dequantization(spec) ? APXINF_DTYPE_BF16
                                        : vendor::common_projection_dtype(spec);
}

}  // namespace

size_t cublas_resource_requirements(const Spec& spec) {
  return dequantization_resource_requirements(spec) +
         vendor::common_resource_requirements(
             cublas_compute_spec(spec), false, cublas_projection_dtype(spec)) +
         providers::CublasStreamResources::workspace_bytes;
}

void prepare_cublas(Execution& state) {
  auto resources = std::make_unique<CublasState>();
  const Spec compute_spec = cublas_compute_spec(state.spec);
  vendor::allocate_common_resources(
      compute_spec, resources->common, false, cublas_projection_dtype(state.spec));
  if (needs_safe_dequantization(state.spec)) {
    const size_t a_bytes = static_cast<size_t>(state.spec.m * state.spec.k) *
                           dtype_bytes(APXINF_DTYPE_BF16);
    const size_t b_bytes = static_cast<size_t>(state.spec.k * state.spec.n) *
                           dtype_bytes(APXINF_DTYPE_BF16);
    check_cuda(cudaMalloc(&resources->dequant_a, a_bytes));
    check_cuda(cudaMalloc(&resources->dequant_b, b_bytes));
  }
  resources->shared = providers::acquire_cublas_stream_resources(
      state.device, static_cast<cudaStream_t>(state.bindings.stream));
  state.resource_bytes =
      resources->common.resource_bytes + resources->shared->workspace_bytes +
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
  const float alpha =
      has_row_channel_scales(compute_spec) ? 1.0F : bindings.alpha;
  const float beta = 0.0F;
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
      compute_spec.a_dtype == APXINF_DTYPE_I8 ? CUDA_R_8I
      : compute_spec.quantization != APXINF_GEMM_QUANT_NONE ? data_type
      : compute_spec.a_dtype == APXINF_DTYPE_F16 ? CUDA_R_16F
      : compute_spec.a_dtype == APXINF_DTYPE_BF16 ? CUDA_R_16BF
                                                : CUDA_R_32F;
  // A cuBLAS call can enqueue multiple kernels using the shared workspace.
  // Serialize the complete host call; stream ordering then serializes its GPU
  // work with every other execution using these resources.
  std::lock_guard<std::mutex> lock(resources.shared->enqueue_mutex);
  check_cublas(cublasGemmEx(
      resources.shared->handle, CUBLAS_OP_N, CUBLAS_OP_N, compute_spec.n,
      compute_spec.m, compute_spec.k,
      alpha_pointer, weight,
      input_type,
      compute_spec.n, activation,
      input_type,
      compute_spec.k, beta_pointer, projection, data_type, compute_spec.n,
      compute_spec.a_dtype == APXINF_DTYPE_I8 ? CUBLAS_COMPUTE_32I
                                              : CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT));
  return vendor::launch_postprocess(compute_spec, resources.common, bindings,
                                    projection);
}

}  // namespace apxinf::gemm

// Internal resource regression entry point, following the framework contract
// test helpers. It is not a model-visible operator or provider-selection API.
extern "C" int apxinf_gemm_test_cublas_stream_resources(
    int device, void* first_stream, void* second_stream, char* error,
    size_t capacity) {
  try {
    using apxinf::providers::acquire_cublas_stream_resources;
    if (first_stream == second_stream) throw std::runtime_error("test needs distinct streams");
    std::weak_ptr<apxinf::providers::CublasStreamResources> first_lifetime;
    std::weak_ptr<apxinf::providers::CublasStreamResources> second_lifetime;
    {
      auto first = acquire_cublas_stream_resources(device, static_cast<cudaStream_t>(first_stream));
      auto second = acquire_cublas_stream_resources(device, static_cast<cudaStream_t>(second_stream));
      first_lifetime = first;
      second_lifetime = second;
      if (first->handle == second->handle || first->workspace == second->workspace)
        throw std::runtime_error("different streams share cuBLAS state");
      std::vector<std::shared_ptr<apxinf::providers::CublasStreamResources>> retained;
      for (int i = 0; i < 64; ++i) {
        auto again = acquire_cublas_stream_resources(device, static_cast<cudaStream_t>(first_stream));
        if (again.get() != first.get())
          throw std::runtime_error("one stream allocated another cuBLAS pool");
        retained.push_back(std::move(again));
      }
      first.reset();
      if (first_lifetime.expired()) throw std::runtime_error("live executions lost shared resources");
      second.reset();
      if (!second_lifetime.expired()) throw std::runtime_error("weak cache kept unused resources alive");
    }
    if (!first_lifetime.expired()) throw std::runtime_error("last execution failed to release resources");
    return 1;
  } catch (const std::exception& failure) {
    if (error != nullptr && capacity != 0) {
      const size_t bytes = std::min(capacity - 1, std::strlen(failure.what()));
      std::memcpy(error, failure.what(), bytes);
      error[bytes] = '\0';
    }
    return 0;
  }
}
