#include "../internal.h"

#if defined(APXINF_ATTENTION_CUTLASS)

#include "../../../kernels/cutlass/ops/attention/fmha_sm100.cu"

namespace apxinf::attention {
namespace {

using CutlassState = apxinf::cuda::cutlass_ops::FmhaState;

int prepare_provider(Execution& execution, CutlassState& state) {
  const auto& spec = execution.spec;
  const auto& bindings = execution.bindings;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  if (spec.dtype == APXINF_DTYPE_F16) {
    return apxinf::cuda::cutlass_ops::prepare_mha_f16(
        bindings.query, bindings.key, bindings.value, bindings.output,
        static_cast<int>(spec.batch), static_cast<int>(spec.query_tokens),
        static_cast<int>(spec.key_tokens), static_cast<int>(spec.query_heads),
        static_cast<int>(spec.kv_heads), static_cast<int>(spec.head_dim),
        bindings.scale, stream, state, execution.resource_limit);
  }
  return apxinf::cuda::cutlass_ops::prepare_mha_bf16(
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
  if (spec.dtype == APXINF_DTYPE_F16) {
    return apxinf::cuda::cutlass_ops::mha_f16(
        bindings.query, bindings.key, bindings.value, bindings.output,
        static_cast<int>(spec.batch), static_cast<int>(spec.query_tokens),
        static_cast<int>(spec.key_tokens), static_cast<int>(spec.query_heads),
        static_cast<int>(spec.kv_heads), static_cast<int>(spec.head_dim),
        bindings.scale, stream, state, execution.resource_limit);
  }
  return apxinf::cuda::cutlass_ops::mha_bf16(
      bindings.query, bindings.key, bindings.value, bindings.output,
      static_cast<int>(spec.batch), static_cast<int>(spec.query_tokens),
      static_cast<int>(spec.key_tokens), static_cast<int>(spec.query_heads),
      static_cast<int>(spec.kv_heads), static_cast<int>(spec.head_dim),
      bindings.scale, stream, state, execution.resource_limit);
}

}  // namespace

size_t cutlass_resource_requirements(const Spec& spec) {
  const uint64_t rounded_query =
      (static_cast<uint64_t>(spec.query_tokens) + 127) / 128 * 128;
  const uint64_t elements = static_cast<uint64_t>(spec.batch) *
                            spec.query_heads * rounded_query;
  if (elements > SIZE_MAX / sizeof(float)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "CUTLASS FMHA workspace size overflow");
  }
  return static_cast<size_t>(elements) * sizeof(float);
}

void prepare_cutlass(Execution& execution) {
  auto state = std::make_unique<CutlassState>();
  const int status = prepare_provider(execution, *state);
  if (status == -1 || status == -5) {
    apxinf::cuda::cutlass_ops::destroy_mha(*state);
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  status == -5 ? "CUTLASS FMHA workspace exceeds policy"
                               : "CUTLASS FMHA cannot implement this Spec");
  }
  if (status != 0) {
    apxinf::cuda::cutlass_ops::destroy_mha(*state);
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  "CUTLASS FMHA preparation failed with status " +
                      std::to_string(status));
  }
  execution.resource_bytes = state->lse_size + state->workspace_size;
  execution.provider_state = state.release();
}

cudaError_t launch_cutlass(Execution& execution) {
  auto* state = static_cast<CutlassState*>(execution.provider_state);
  return launch_provider(execution, *state) == 0 ? cudaSuccess
                                                 : cudaErrorUnknown;
}

void destroy_cutlass(Execution& execution) noexcept {
  auto* state = static_cast<CutlassState*>(execution.provider_state);
  if (state == nullptr) return;
  apxinf::cuda::cutlass_ops::destroy_mha(*state);
  delete state;
  execution.provider_state = nullptr;
}

}  // namespace apxinf::attention

#endif
