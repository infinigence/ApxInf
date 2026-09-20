#pragma once

#include "../../framework/autotune.h"
#include "../../framework/registry.h"
#include "../../framework/runtime_internal.h"
#include "../../include/apxinf_cuda/attention.h"
#include "apxinf_cuda_arches.h"

#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace apxinf::attention {

using apxinf::framework::Failure;
using apxinf::framework::Recipe;
using apxinf::framework::abi_boundary;
using apxinf::framework::check_cuda;

struct Spec : apxinf_attention_spec_t {};

inline size_t dtype_bytes(uint32_t dtype) {
  if (dtype == APXINF_DTYPE_F32) return 4;
  if (dtype == APXINF_DTYPE_E4M3 || dtype == APXINF_DTYPE_I8) return 1;
  return 2;
}

struct Execution;
using PrepareExecutionFn = void (*)(Execution&);
using EnqueueFn = cudaError_t (*)(Execution&);
using DestroyExecutionFn = void (*)(Execution&) noexcept;

struct AlignmentRequirements {
  uint32_t query = 1;
  uint32_t key = 1;
  uint32_t value = 1;
  uint32_t output = 1;
};

struct Implementation {
  uint32_t provider_id;
  uint32_t implementation_id;
  uint32_t implementation_version;
  const char* name;
  uint64_t required_device_features;
  bool graph_safe;
  bool deterministic;
  bool fallback;
  bool (*supports)(const Spec&);
  AlignmentRequirements (*alignment_requirements)(const Spec&);
  size_t (*resource_requirements)(const Spec&);
  void (*enumerate_configs)(const Spec&, std::vector<int>&);
  PrepareExecutionFn prepare;
  EnqueueFn enqueue;
  DestroyExecutionFn destroy;
};

using ImplementationRegistry = apxinf::framework::Registry<Implementation>;

struct TuningKeys {
  std::string key;
};

struct Execution {
  Spec spec{};
  apxinf_attention_bindings_t bindings{};
  int configuration = 0;
  int device = 0;
  const Implementation* implementation = nullptr;
  void* provider_state = nullptr;
  size_t resource_bytes = 0;
  size_t resource_limit = 0;
  std::string summary;

  ~Execution();
};

const ImplementationRegistry& registry(uint32_t semantic);
bool supports_device(const Implementation& implementation, int device,
                     std::string* reason = nullptr);
bool supports_alignment(const Implementation& implementation, const Spec& spec);
std::unique_ptr<Execution> prepare(const Implementation& implementation,
                                   int configuration, const Spec& spec,
                                   const apxinf_attention_policy_t& policy,
                                   const apxinf_attention_bindings_t& bindings,
                                   int device);
Recipe tune(const Spec& spec, const apxinf_attention_policy_t& policy,
            const apxinf_attention_bindings_t& bindings, int device,
            std::string& report);
TuningKeys tuning_keys(const Spec& spec,
                       const apxinf_attention_policy_t& policy, int device);

size_t custom_resource_requirements(const Spec& spec);
void prepare_custom(Execution& execution);
cudaError_t launch_custom(Execution& execution);
void destroy_custom(Execution& execution) noexcept;

#if defined(APXINF_ATTENTION_FA2)
size_t fa2_resource_requirements(const Spec& spec);
void prepare_fa2(Execution& execution);
cudaError_t launch_fa2(Execution& execution);
void destroy_fa2(Execution& execution) noexcept;
#endif

#if defined(APXINF_ATTENTION_CUTLASS)
size_t cutlass_resource_requirements(const Spec& spec);
void prepare_cutlass(Execution& execution);
cudaError_t launch_cutlass(Execution& execution);
void destroy_cutlass(Execution& execution) noexcept;
#endif

}  // namespace apxinf::attention

struct apxinf_attention_execution {
  std::unique_ptr<apxinf::attention::Execution> state;
};
