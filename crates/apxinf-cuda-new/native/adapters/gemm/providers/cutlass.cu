#include "../internal.h"
#include "../../../kernels/custom/gemm.cuh"

#ifdef APXINF_GEMM_CUTLASS
#include "../../../kernels/cutlass/ops/gemm/gemm_bf16_sm100.h"
#include "../../../kernels/cutlass/ops/gemm/gemm_e4m3_sm100.h"
#include "../../../kernels/cutlass/ops/gemm/gemm_nvfp4_sm100.h"
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

struct CutlassNvFp4State {
  void* runner = nullptr;
  void* workspace = nullptr;
  void* projection = nullptr;
  void* packed_b = nullptr;
  void* packed_a_scales = nullptr;
  void* packed_b_scales = nullptr;
  const void* packed_weight_source = nullptr;
  const void* packed_weight_scale_source = nullptr;
  uint64_t packed_weight_version = 0;
  bool packed_weight_ready = false;

  ~CutlassNvFp4State() {
#ifdef APXINF_GEMM_CUTLASS
    apxinf::cuda::cutlass_ops::nvfp4_destroy(runner);
#endif
    if (workspace != nullptr) cudaFree(workspace);
    if (projection != nullptr) cudaFree(projection);
    if (packed_b != nullptr) cudaFree(packed_b);
    if (packed_a_scales != nullptr) cudaFree(packed_a_scales);
    if (packed_b_scales != nullptr) cudaFree(packed_b_scales);
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

void pack_nvfp4_weight(CutlassNvFp4State& resources, const Spec& spec,
                       const apxinf_gemm_bindings_t& bindings) {
#ifdef APXINF_GEMM_CUTLASS
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  check_cuda(apxinf::cuda::cutlass_ops::nvfp4_pack_b(
      bindings.b, resources.packed_b, static_cast<int>(spec.n),
      static_cast<int>(spec.k), stream));
  check_cuda(apxinf::cuda::cutlass_ops::nvfp4_pack_scales(
      bindings.b_scales, resources.packed_b_scales,
      static_cast<int>(spec.n), static_cast<int>(spec.k), stream));
#else
  (void)resources;
  (void)spec;
  (void)bindings;
#endif
}

cudaError_t finish_nvfp4(const Execution& state,
                         const CutlassNvFp4State& resources) {
  if (resources.projection == nullptr) return cudaSuccess;
  const auto& spec = state.spec;
  const auto& bindings = state.bindings;
  const int64_t width =
      spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_GEGLU ? spec.n / 2 : spec.n;
  const int blocks = static_cast<int>(
      std::min<int64_t>((spec.m * width + 255) / 256, 4096));
  apxinf::cuda::custom::finish<<<
      blocks, 256, 0, static_cast<cudaStream_t>(bindings.stream)>>>(
      resources.projection, APXINF_DTYPE_F16, bindings.output,
      APXINF_DTYPE_F16, bindings.bias, APXINF_DTYPE_F16, nullptr, nullptr,
      spec.m, spec.n, static_cast<int>(spec.semantic), 0, bindings.alpha,
      bindings.output_scale);
  return cudaGetLastError();
}

}  // namespace

size_t cutlass_fp8_resource_requirements(const Spec&) { return 0; }

size_t cutlass_geglu_resource_requirements(const Spec& spec) {
  return static_cast<size_t>(spec.k * spec.n) * dtype_bytes(spec.b_dtype);
}

size_t cutlass_nvfp4_resource_requirements(const Spec& spec) {
#ifdef APXINF_GEMM_CUTLASS
  size_t workspace = 0;
  for (int configuration = 0;
       configuration < apxinf::cuda::cutlass_ops::nvfp4_num_configurations();
       ++configuration) {
    workspace = std::max(
        workspace, apxinf::cuda::cutlass_ops::nvfp4_workspace_size(
                       configuration, static_cast<int>(spec.m),
                       static_cast<int>(spec.n), static_cast<int>(spec.k)));
  }
  return workspace + apxinf::cuda::cutlass_ops::nvfp4_packed_b_size(
                         static_cast<int>(spec.n), static_cast<int>(spec.k)) +
         apxinf::cuda::cutlass_ops::nvfp4_scale_workspace_size(
             static_cast<int>(spec.m), static_cast<int>(spec.k)) +
         apxinf::cuda::cutlass_ops::nvfp4_scale_workspace_size(
             static_cast<int>(spec.n), static_cast<int>(spec.k)) +
         (spec.semantic == APXINF_GEMM_SEMANTIC_GEMM
              ? 0
              : static_cast<size_t>(spec.m * spec.n) * sizeof(half));
#else
  (void)spec;
  return 0;
#endif
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

void prepare_cutlass_nvfp4_gemm(Execution& state) {
#ifdef APXINF_GEMM_CUTLASS
  auto resources = std::make_unique<CutlassNvFp4State>();
  const auto& spec = state.spec;
  const auto& bindings = state.bindings;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  const size_t workspace_bytes =
      apxinf::cuda::cutlass_ops::nvfp4_workspace_size(
          state.configuration, static_cast<int>(spec.m),
          static_cast<int>(spec.n), static_cast<int>(spec.k));
  const size_t packed_b_bytes =
      apxinf::cuda::cutlass_ops::nvfp4_packed_b_size(
          static_cast<int>(spec.n), static_cast<int>(spec.k));
  const size_t packed_a_scale_bytes =
      apxinf::cuda::cutlass_ops::nvfp4_scale_workspace_size(
          static_cast<int>(spec.m), static_cast<int>(spec.k));
  const size_t packed_b_scale_bytes =
      apxinf::cuda::cutlass_ops::nvfp4_scale_workspace_size(
          static_cast<int>(spec.n), static_cast<int>(spec.k));
  const size_t projection_bytes =
      spec.semantic == APXINF_GEMM_SEMANTIC_GEMM
          ? 0
          : static_cast<size_t>(spec.m * spec.n) * sizeof(half);
  const size_t resource_bytes = workspace_bytes + packed_b_bytes +
                                packed_a_scale_bytes + packed_b_scale_bytes +
                                projection_bytes;
  if (workspace_bytes == static_cast<size_t>(-1) ||
      resource_bytes > state.resource_limit) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "NVFP4 CUTLASS workspace exceeds policy");
  }
  if (workspace_bytes != 0) {
    check_cuda(cudaMalloc(&resources->workspace, workspace_bytes));
  }
  check_cuda(cudaMalloc(&resources->packed_b, packed_b_bytes));
  check_cuda(cudaMalloc(&resources->packed_a_scales, packed_a_scale_bytes));
  check_cuda(cudaMalloc(&resources->packed_b_scales, packed_b_scale_bytes));
  if (projection_bytes != 0) {
    check_cuda(cudaMalloc(&resources->projection, projection_bytes));
  }
  check_cuda(cudaMemsetAsync(resources->packed_a_scales, 0,
                             packed_a_scale_bytes, stream));
  check_cuda(cudaMemsetAsync(resources->packed_b_scales, 0,
                             packed_b_scale_bytes, stream));
  if (bindings.b_is_immutable != 0) {
    pack_nvfp4_weight(*resources, spec, bindings);
    resources->packed_weight_source = bindings.b;
    resources->packed_weight_scale_source = bindings.b_scales;
    resources->packed_weight_version = bindings.b_version;
    resources->packed_weight_ready = true;
  }
  void* gemm_output =
      resources->projection != nullptr ? resources->projection : bindings.output;
  const float gemm_alpha =
      resources->projection != nullptr
          ? 1.0F
          : bindings.alpha / bindings.output_scale;
  const int status = apxinf::cuda::cutlass_ops::nvfp4_create(
      state.configuration, bindings.a, resources->packed_a_scales,
      resources->packed_b, resources->packed_b_scales, gemm_output,
      static_cast<int>(spec.m), static_cast<int>(spec.n),
      static_cast<int>(spec.k), gemm_alpha,
      resources->workspace, stream, &resources->runner);
  check_cutlass_status(status);
  state.resource_bytes = resource_bytes;
  state.provider_state = resources.release();
#else
  (void)state;
  throw Failure(APXINF_STATUS_UNSUPPORTED, "CUTLASS was not built");
#endif
}

void destroy_cutlass(Execution& state) noexcept {
  delete static_cast<CutlassGegluState*>(state.provider_state);
  state.provider_state = nullptr;
}

void destroy_cutlass_nvfp4(Execution& state) noexcept {
  delete static_cast<CutlassNvFp4State*>(state.provider_state);
  state.provider_state = nullptr;
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

cudaError_t launch_cutlass_nvfp4_gemm(Execution& state) {
#ifdef APXINF_GEMM_CUTLASS
  auto* resources = static_cast<CutlassNvFp4State*>(state.provider_state);
  if (resources == nullptr || resources->runner == nullptr) {
    return cudaErrorInvalidResourceHandle;
  }
  const auto& bindings = state.bindings;
  const auto& spec = state.spec;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  check_cuda(apxinf::cuda::cutlass_ops::nvfp4_pack_scales(
      bindings.a_scales, resources->packed_a_scales,
      static_cast<int>(spec.m), static_cast<int>(spec.k), stream));
  if (bindings.b_is_immutable != 0) {
    if (!resources->packed_weight_ready ||
        resources->packed_weight_source != bindings.b ||
        resources->packed_weight_scale_source != bindings.b_scales ||
        resources->packed_weight_version != bindings.b_version) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "prepared NVFP4 weight identity changed");
    }
  } else {
    pack_nvfp4_weight(*resources, spec, bindings);
  }
  const int status =
      apxinf::cuda::cutlass_ops::nvfp4_run(resources->runner, stream);
  check_cutlass_status(status);
  return finish_nvfp4(state, *resources);
#else
  (void)state;
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
  return static_cast<const CutlassGegluState*>(state.provider_state)
      ->prepack_count;
}

}  // namespace apxinf::gemm
