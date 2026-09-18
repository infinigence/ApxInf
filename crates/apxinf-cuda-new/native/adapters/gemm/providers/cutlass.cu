#include "../internal.h"
#include "../../../kernels/custom/gemm.cuh"

#ifdef APXINF_GEMM_CUTLASS
#include "../../../kernels/cutlass/ops/gemm/gemm_bf16_sm100.h"
#include "../../../kernels/cutlass/ops/gemm/gemm_e4m3_sm100.h"
#endif

#ifdef APXINF_GEMM_CUTLASS_SM80_W8A8
namespace apxinf::cuda::cutlass_ops {
cudaError_t w8a8_gemm_bf16(
    const void* activation, const void* weight_output_major,
    const void* row_scales, const void* column_scales, void* output, int m,
    int n, int k, cudaStream_t stream);
}
#endif

namespace apxinf::gemm {
namespace {

struct CutlassGegluState {
  void* packed_weight = nullptr;
  size_t packed_weight_bytes = 0;
  const void* packed_weight_source = nullptr;
  uint64_t packed_weight_version = 0;
  uint64_t prepack_count = 0;
  bool packed_weight_ready = false;

  ~CutlassGegluState() { release_resources(); }

  void release_resources() noexcept {
    if (packed_weight != nullptr) cudaFree(packed_weight);
    packed_weight = nullptr;
    packed_weight_source = nullptr;
    packed_weight_version = 0;
    packed_weight_ready = false;
  }
};

struct CutlassW8a8State {
  void* packed_weight = nullptr;
  size_t packed_weight_bytes = 0;
  void* projection = nullptr;
  size_t projection_bytes = 0;
  const void* packed_weight_source = nullptr;
  uint64_t packed_weight_version = 0;
  bool packed_weight_ready = false;

  ~CutlassW8a8State() {
    if (packed_weight != nullptr) cudaFree(packed_weight);
    if (projection != nullptr) cudaFree(projection);
  }
};

CutlassGegluState& provider(Execution& state) {
  return *static_cast<CutlassGegluState*>(state.provider_state);
}

CutlassW8a8State& w8a8_provider(Execution& state) {
  return *static_cast<CutlassW8a8State*>(state.provider_state);
}

__global__ void transpose_w8a8_weight_kn_to_nk(
    const int8_t* source, int8_t* destination, int64_t k, int64_t n) {
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < k * n; index += int64_t(gridDim.x) * blockDim.x) {
    const int64_t source_k = index / n;
    const int64_t source_n = index % n;
    destination[source_n * k + source_k] = source[index];
  }
}

void pack_w8a8_weight(Execution& state,
                       const apxinf_gemm_bindings_t& bindings) {
  auto& resources = w8a8_provider(state);
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  const auto& spec = state.spec;
  const int blocks = static_cast<int>(
      std::min<int64_t>((spec.k * spec.n + 255) / 256, 4096));
  transpose_w8a8_weight_kn_to_nk<<<blocks, 256, 0, stream>>>(
      static_cast<const int8_t*>(bindings.b),
      static_cast<int8_t*>(resources.packed_weight), spec.k, spec.n);
  check_cuda(cudaGetLastError());
}

void pack_geglu_weight(Execution& state,
                       const apxinf_gemm_bindings_t& bindings) {
  auto& resources = provider(state);
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  const auto& spec = state.spec;
  const int blocks = static_cast<int>(
      std::min<int64_t>((spec.k * spec.n + 255) / 256, 4096));
  apxinf::cuda::custom::pack_gate_up<<<blocks, 256, 0, stream>>>(
      bindings.b, resources.packed_weight, spec.b_dtype, spec.k, spec.n);
  check_cuda(cudaGetLastError());
  ++resources.prepack_count;
}

void check_cutlass_status(int status) {
  if (status != 0) {
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  "CUTLASS launch status " + std::to_string(status));
  }
}

}  // namespace

size_t cutlass_fp8_resource_requirements(const Spec&) { return 0; }

size_t cutlass_geglu_resource_requirements(const Spec& spec) {
  return static_cast<size_t>(spec.k * spec.n) * dtype_bytes(spec.b_dtype);
}

size_t cutlass_w8a8_resource_requirements(const Spec& spec) {
  size_t bytes = static_cast<size_t>(spec.k * spec.n) * dtype_bytes(spec.b_dtype);
  if (spec.semantic != APXINF_GEMM_SEMANTIC_GEMM) {
    bytes += static_cast<size_t>(spec.m * spec.n) * dtype_bytes(APXINF_DTYPE_BF16);
  }
  return bytes;
}

void prepare_cutlass_fp8_gemm(Execution&) {}

void prepare_cutlass_geglu(Execution& state) {
  auto owned_resources = std::make_unique<CutlassGegluState>();
  owned_resources->packed_weight_bytes =
      state.spec.k * state.spec.n * dtype_bytes(state.spec.b_dtype);
  check_cuda(cudaMalloc(&owned_resources->packed_weight,
                        owned_resources->packed_weight_bytes));
  state.resource_bytes = owned_resources->packed_weight_bytes;
  state.provider_state = owned_resources.release();
  const auto& bindings = state.bindings;
  if (bindings.b_is_immutable == 0) return;
  auto& resources = provider(state);
  if (resources.packed_weight_ready) {
    if (resources.packed_weight_source != bindings.b ||
        resources.packed_weight_version != bindings.b_version) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "prepared CUTLASS weight identity changed");
    }
    return;
  }
  pack_geglu_weight(state, bindings);
  resources.packed_weight_source = bindings.b;
  resources.packed_weight_version = bindings.b_version;
  resources.packed_weight_ready = true;
}

void prepare_cutlass_w8a8(Execution& state) {
  auto resources = std::make_unique<CutlassW8a8State>();
  resources->packed_weight_bytes = cutlass_w8a8_resource_requirements(state.spec);
  check_cuda(cudaMalloc(&resources->packed_weight,
                        resources->packed_weight_bytes));
  if (state.spec.semantic != APXINF_GEMM_SEMANTIC_GEMM) {
    resources->projection_bytes =
        static_cast<size_t>(state.spec.m * state.spec.n) *
        dtype_bytes(APXINF_DTYPE_BF16);
    check_cuda(cudaMalloc(&resources->projection, resources->projection_bytes));
  }
  state.resource_bytes =
      resources->packed_weight_bytes + resources->projection_bytes;
  state.provider_state = resources.release();
  const auto& bindings = state.bindings;
  if (bindings.b_is_immutable == 0) return;
  pack_w8a8_weight(state, bindings);
  auto& prepared = w8a8_provider(state);
  prepared.packed_weight_source = bindings.b;
  prepared.packed_weight_version = bindings.b_version;
  prepared.packed_weight_ready = true;
}

void destroy_cutlass(Execution& state) noexcept {
  delete static_cast<CutlassGegluState*>(state.provider_state);
  state.provider_state = nullptr;
}

void destroy_cutlass_w8a8(Execution& state) noexcept {
  delete static_cast<CutlassW8a8State*>(state.provider_state);
  state.provider_state = nullptr;
}

cudaError_t launch_cutlass_w8a8(Execution& state) {
#ifdef APXINF_GEMM_CUTLASS_SM80_W8A8
  const auto& bindings = state.bindings;
  auto& resources = w8a8_provider(state);
  if (bindings.b_is_immutable != 0) {
    if (!resources.packed_weight_ready ||
        resources.packed_weight_source != bindings.b ||
        resources.packed_weight_version != bindings.b_version) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "CUTLASS W8A8 immutable weight was not prepared for these bindings");
    }
  } else {
    pack_w8a8_weight(state, bindings);
  }
  const auto& spec = state.spec;
  void* projection = resources.projection != nullptr ? resources.projection
                                                     : bindings.output;
  const auto status = apxinf::cuda::cutlass_ops::w8a8_gemm_bf16(
      bindings.a, resources.packed_weight, bindings.a_scales,
      bindings.b_scales, projection, static_cast<int>(spec.m),
      static_cast<int>(spec.n), static_cast<int>(spec.k),
      static_cast<cudaStream_t>(bindings.stream));
  if (status != cudaSuccess || resources.projection == nullptr) return status;
  const int64_t output_width = is_gated_semantic(spec) ? spec.n / 2 : spec.n;
  const int blocks = static_cast<int>(
      std::min<int64_t>((spec.m * output_width + 255) / 256, 4096));
  apxinf::cuda::custom::finish<<<
      blocks, 256, 0, static_cast<cudaStream_t>(bindings.stream)>>>(
      projection, APXINF_DTYPE_BF16, bindings.output, spec.output_dtype,
      bindings.bias, APXINF_DTYPE_BF16, bindings.residual, nullptr, nullptr,
      spec.m, spec.n, static_cast<int>(spec.semantic), 0, bindings.alpha,
      bindings.output_scale);
  return cudaGetLastError();
#else
  (void)state;
  return cudaErrorNotSupported;
#endif
}

cudaError_t launch_cutlass_fp8_gemm(Execution& state) {
  const auto& bindings = state.bindings;
#ifdef APXINF_GEMM_CUTLASS
  const auto& spec = state.spec;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  using namespace apxinf::cuda::cutlass_ops;
  check_cutlass_status(fp8_gemm_f16(
      bindings.a, bindings.b, bindings.output, spec.m, spec.n, spec.k,
      bindings.alpha, state.configuration, stream));
  return cudaSuccess;
#else
  (void)state;
  (void)bindings;
  return cudaErrorNotSupported;
#endif
}

cudaError_t launch_cutlass_fp8_bf16_gemm(Execution& state) {
  const auto& bindings = state.bindings;
#ifdef APXINF_GEMM_CUTLASS
  const auto& spec = state.spec;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  using namespace apxinf::cuda::cutlass_ops;
  check_cutlass_status(fp8_gemm_bf16(
      bindings.a, bindings.b, bindings.output, spec.m, spec.n, spec.k,
      bindings.alpha, state.configuration, stream));
  return cudaSuccess;
#else
  (void)state;
  (void)bindings;
  return cudaErrorNotSupported;
#endif
}

cudaError_t launch_cutlass_fp8_geglu(Execution& state) {
  const auto& bindings = state.bindings;
#ifdef APXINF_GEMM_CUTLASS
  const auto& spec = state.spec;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  auto& resources = provider(state);
  if (bindings.b_is_immutable != 0) {
    if (!resources.packed_weight_ready ||
        resources.packed_weight_source != bindings.b ||
        resources.packed_weight_version != bindings.b_version) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "CUTLASS immutable weight was not prepared for these bindings");
    }
  } else {
    pack_geglu_weight(state, bindings);
  }
  const void* weight = resources.packed_weight;
  check_cutlass_status(
      apxinf::cuda::cutlass_ops::fp8_dual_geglu_detail::production_dual_geglu(
          bindings.a, weight, bindings.output, spec.m, spec.n / 2, spec.k,
          spec.n, bindings.alpha, bindings.output_scale, stream));
  return cudaSuccess;
#else
  (void)state;
  (void)bindings;
  return cudaErrorNotSupported;
#endif
}

cudaError_t launch_cutlass_bf16_geglu(Execution& state) {
  const auto& bindings = state.bindings;
#ifdef APXINF_GEMM_CUTLASS
  const auto& spec = state.spec;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  auto& resources = provider(state);
  if (bindings.b_is_immutable != 0) {
    if (!resources.packed_weight_ready ||
        resources.packed_weight_source != bindings.b ||
        resources.packed_weight_version != bindings.b_version) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "CUTLASS immutable weight was not prepared for these bindings");
    }
  } else {
    pack_geglu_weight(state, bindings);
  }
  const void* weight = resources.packed_weight;
  check_cutlass_status(apxinf::cuda::cutlass_ops::bf16_dual_geglu_detail::
                           production_dual_geglu_bf16(
                               bindings.a, weight, bindings.output, spec.m,
                               spec.n / 2, spec.k, spec.n, stream));
  return cudaSuccess;
#else
  (void)state;
  (void)bindings;
  return cudaErrorNotSupported;
#endif
}

uint64_t cutlass_weight_prepack_count(const Execution& state) {
  if (state.provider_state == nullptr) return 0;
  if (state.implementation != nullptr &&
      std::strcmp(state.implementation->name, "cutlass-w8a8-sm80") == 0) {
    return 0;
  }
  return static_cast<const CutlassGegluState*>(state.provider_state)
      ->prepack_count;
}

}  // namespace apxinf::gemm
