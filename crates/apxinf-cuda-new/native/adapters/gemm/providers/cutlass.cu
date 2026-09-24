#include "../internal.h"
#include "../../../kernels/custom/gemm.cuh"

#ifdef APXINF_GEMM_CUTLASS
#include "../../../kernels/cutlass/ops/gemm/gemm_bf16_sm100.h"
#include "../../../kernels/cutlass/ops/gemm/gemm_e4m3_sm100.h"
#endif
#ifdef APXINF_GEMM_CUTLASS_SM89
#include "../../../kernels/cutlass/ops/gemm/gemm_bf16_sm89.h"
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

CutlassGegluState& provider(Execution& state) {
  return *static_cast<CutlassGegluState*>(state.provider_state);
}

void pack_geglu_weight(Execution& state,
                       const apxinf_gemm_bindings_t& bindings) {
  auto& resources = provider(state);
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  const auto& spec = state.spec;
  const int blocks = static_cast<int>(
      std::min<int64_t>((spec.k * spec.n + 255) / 256, 4096));
  apxinf::cuda_new::custom::pack_gate_up<<<blocks, 256, 0, stream>>>(
      bindings.b, resources.packed_weight, spec.b_dtype, spec.k, spec.n);
  check_cuda(cudaGetLastError());
  ++resources.prepack_count;
}

#ifdef APXINF_GEMM_CUTLASS_SM89
__global__ void pack_adjacent_bf16(const __nv_bfloat16* source,
                                   __nv_bfloat16* destination,
                                   int64_t rows,
                                   int64_t columns) {
  const int64_t half = columns / 2;
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < rows * columns;
       index += int64_t(gridDim.x) * blockDim.x) {
    const int64_t row = index / columns;
    const int64_t column = index % columns;
    const int64_t pair = column < half ? column : column - half;
    const int64_t lane = column < half ? 0 : 1;
    destination[row * columns + 2 * pair + lane] = source[index];
  }
}

void pack_geglu_weight_sm89(Execution& state,
                             const apxinf_gemm_bindings_t& bindings) {
  auto& resources = provider(state);
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  const auto& spec = state.spec;
  const int blocks = static_cast<int>(
      std::min<int64_t>((spec.k * spec.n + 255) / 256, 4096));
  pack_adjacent_bf16<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(bindings.b),
      static_cast<__nv_bfloat16*>(resources.packed_weight), spec.k, spec.n);
  check_cuda(cudaGetLastError());
  ++resources.prepack_count;
}
#endif

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

void prepare_cutlass_geglu_sm89(Execution& state) {
#ifdef APXINF_GEMM_CUTLASS_SM89
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
  pack_geglu_weight_sm89(state, bindings);
  resources.packed_weight_source = bindings.b;
  resources.packed_weight_version = bindings.b_version;
  resources.packed_weight_ready = true;
#else
  (void)state;
#endif
}

void destroy_cutlass(Execution& state) noexcept {
  delete static_cast<CutlassGegluState*>(state.provider_state);
  state.provider_state = nullptr;
}

cudaError_t launch_cutlass_fp8_gemm(Execution& state) {
  const auto& bindings = state.bindings;
#ifdef APXINF_GEMM_CUTLASS
  const auto& spec = state.spec;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  using namespace apxinf::cuda_new::cutlass_ops;
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
      apxinf::cuda_new::cutlass_ops::fp8_dual_geglu_detail::production_dual_geglu(
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
  check_cutlass_status(apxinf::cuda_new::cutlass_ops::bf16_dual_geglu_detail::
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

cudaError_t launch_cutlass_bf16_geglu_sm89(Execution& state) {
  const auto& bindings = state.bindings;
#ifdef APXINF_GEMM_CUTLASS_SM89
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
    pack_geglu_weight_sm89(state, bindings);
  }
  check_cutlass_status(
      apxinf::cuda_new::cutlass_ops::bf16_sm89_detail::interleaved_geglu(
          bindings.a, resources.packed_weight, bindings.output, spec.m,
          spec.n / 2, spec.k, spec.n, 0, stream));
  return cudaSuccess;
#else
  (void)state;
  (void)bindings;
  return cudaErrorNotSupported;
#endif
}

uint64_t cutlass_weight_prepack_count(const Execution& state) {
  if (state.provider_state == nullptr) return 0;
  return static_cast<const CutlassGegluState*>(state.provider_state)
      ->prepack_count;
}

}  // namespace apxinf::gemm
