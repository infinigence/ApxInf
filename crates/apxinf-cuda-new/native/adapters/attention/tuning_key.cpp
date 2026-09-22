#include "internal.h"

#include <algorithm>

#ifndef APXINF_ATTENTION_BUILD_ID
#define APXINF_ATTENTION_BUILD_ID "attention-development"
#endif

namespace apxinf::attention {
namespace {

int cuda_compat_version(int version) {
  return (version / 1000) * 100 + ((version % 1000) / 10);
}

uint32_t alignment_class(uint32_t alignment, uint32_t maximum) {
  return alignment == 0 ? 0 : std::min(alignment, maximum);
}

}  // namespace

TuningKeys tuning_keys(const Spec& spec,
                       const apxinf_attention_policy_t& policy, int device) {
  int runtime_version = 0;
  int driver_version = 0;
  check_cuda(cudaRuntimeGetVersion(&runtime_version));
  check_cuda(cudaDriverGetVersion(&driver_version));
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device));

  // One exact key: the namespace prefix captures the candidate/toolchain/GPU
  // identity and the suffix captures the normalized operation and policy.
  // There is no compatible fallback key; a miss runs the full autotuner.
  std::ostringstream key;
  key << "attention-recipe-v4|ns|" << APXINF_ATTENTION_BUILD_ID << '|'
      << "toolkit=" << CUDART_VERSION << '|'
      << "cc=" << properties.major * 10 + properties.minor
      << "|sms=" << properties.multiProcessorCount
      << "|cuda=" << cuda_compat_version(runtime_version)
      << "|driver=" << cuda_compat_version(driver_version)
      << "|op|" << spec.version << '|' << spec.semantic << '|'
      << spec.dtype << '|' << spec.output_dtype << '|' << spec.mask << '|'
      << spec.batch << '|' << spec.query_tokens << '|' << spec.key_tokens << '|'
      << spec.key_capacity << '|' << spec.query_heads << '|' << spec.kv_heads
      << '|' << spec.head_dim << '|' << spec.query_start << '|'
      << spec.segments << '|' << spec.max_segment_tokens << '|'
      << spec.scale_is_default << '|'
      << alignment_class(spec.q_alignment, 16) << '|'
      << alignment_class(spec.k_alignment, 16) << '|'
      << alignment_class(spec.v_alignment, 16) << '|'
      << alignment_class(spec.output_alignment, 16) << '|'
      << alignment_class(spec.offsets_alignment, 4) << '|'
      << policy.workspace_limit << '|' << policy.graph_safe << '|'
      << policy.deterministic;
  return {key.str()};
}

}  // namespace apxinf::attention
