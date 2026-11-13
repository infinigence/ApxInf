#include "../internal.h"

#if defined(APXINF_ATTENTION_CUTLASS)

#include "../../../kernels/cutlass/ops/attention/fmha_persistent_sm100.cu"

namespace apxinf::attention {
namespace {

using CutlassState = apxinf::cuda_new::cutlass_persistent_ops::FmhaState;

int prepare_provider(Execution& execution, CutlassState& state) {
  const auto& spec = execution.spec;
  const auto& bindings = execution.bindings;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);

  return apxinf::cuda_new::cutlass_persistent_ops::prepare_mha_bf16(
      bindings.query, bindings.key, bindings.value, bindings.output,
      static_cast<int>(spec.batch), static_cast<int>(spec.query_tokens),
      static_cast<int>(spec.key_tokens), static_cast<int>(spec.query_heads),
      static_cast<int>(spec.kv_heads), static_cast<int>(spec.head_dim),
      bindings.scale, stream, state, execution.resource_limit);
}

int launch_provider(Execution& execution, CutlassState& state) {
  const auto& spec = execution.spec;
  const auto& bindings = execution.bindings;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);

  return apxinf::cuda_new::cutlass_persistent_ops::mha_bf16(
      bindings.query, bindings.key, bindings.value, bindings.output,
      static_cast<int>(spec.batch), static_cast<int>(spec.query_tokens),
      static_cast<int>(spec.key_tokens), static_cast<int>(spec.query_heads),
      static_cast<int>(spec.kv_heads), static_cast<int>(spec.head_dim),
      bindings.scale, stream, state, execution.resource_limit);
}

}  // namespace

size_t cutlass_persistent_resource_requirements(const Spec&, int configuration) {
  if (configuration != 0) {
    throw Failure(APXINF_STATUS_UNSUPPORTED, "Unknown persistent FMHA configuration");
  }
  // This scheduler has no workspace and inference does not request LSE.
  return 0;
}

void prepare_cutlass_persistent(Execution& execution) {
  auto state = std::make_unique<CutlassState>();
  const int status = prepare_provider(execution, *state);
  if (status == -1 || status == -5) {
    apxinf::cuda_new::cutlass_persistent_ops::destroy_mha(*state);
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  status == -5 ? "CUTLASS FMHA workspace exceeds policy"
                               : "CUTLASS FMHA cannot implement this Spec");
  }
  if (status != 0) {
    apxinf::cuda_new::cutlass_persistent_ops::destroy_mha(*state);
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  "CUTLASS FMHA preparation failed with status " +
                      std::to_string(status));
  }
  execution.resource_bytes = state->lse_size + state->workspace_size;
  execution.provider_state = state.release();
}

cudaError_t launch_cutlass_persistent(Execution& execution) {
  auto* state = static_cast<CutlassState*>(execution.provider_state);
  return launch_provider(execution, *state) == 0 ? cudaSuccess
                                                 : cudaErrorUnknown;
}

void destroy_cutlass_persistent(Execution& execution) noexcept {
  auto* state = static_cast<CutlassState*>(execution.provider_state);
  if (state == nullptr) return;
  apxinf::cuda_new::cutlass_persistent_ops::destroy_mha(*state);
  delete state;
  execution.provider_state = nullptr;
}

}  // namespace apxinf::attention

#endif
