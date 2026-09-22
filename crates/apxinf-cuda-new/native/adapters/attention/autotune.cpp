#include "internal.h"

namespace apxinf::attention {
namespace {

struct Allocation {
  void* pointer = nullptr;
  explicit Allocation(size_t bytes) { check_cuda(cudaMalloc(&pointer, bytes)); }
  ~Allocation() {
    if (pointer != nullptr) cudaFree(pointer);
  }
};

class AttentionTuningProblem {
 public:
  using Implementation = apxinf::attention::Implementation;
  using Execution = apxinf::attention::Execution;

  AttentionTuningProblem(const Spec& spec,
                         const apxinf_attention_policy_t& policy,
                         const apxinf_attention_bindings_t& bindings,
                         int device)
      : spec_(spec),
        policy_(policy),
        device_(device),
        output_(static_cast<size_t>(spec.batch * spec.query_tokens *
                                    spec.query_heads * spec.head_dim) *
                dtype_bytes(spec.output_dtype)),
        bindings_(bindings) {
    bindings_.output = output_.pointer;
  }

  const ImplementationRegistry& registry() const {
    return apxinf::attention::registry(spec_.semantic);
  }

  bool supports(const Implementation& implementation,
                std::string& reason) const {
    if (!supports_device(implementation, device_)) {
      reason = "device";
      return false;
    }
    if (!implementation.supports(spec_)) {
      reason = "contract";
      return false;
    }
    if (!supports_alignment(implementation, spec_)) {
      reason = "alignment";
      return false;
    }
    if (policy_.graph_safe && !implementation.graph_safe) {
      reason = "graph-safe";
      return false;
    }
    if (policy_.deterministic && !implementation.deterministic) {
      reason = "determinism";
      return false;
    }
    return true;
  }

  void configurations(const Implementation& implementation,
                      std::vector<int>& values) const {
    implementation.enumerate_configs(spec_, values);
  }

  std::unique_ptr<Execution> prepare(const Implementation& implementation,
                                     int configuration) const {
    return apxinf::attention::prepare(implementation, configuration, spec_,
                                      policy_, bindings_, device_);
  }

  cudaError_t enqueue(Execution& execution) const {
    return execution.implementation->enqueue(execution);
  }

  cudaStream_t stream() const {
    return static_cast<cudaStream_t>(bindings_.stream);
  }

  bool graph_safe() const { return policy_.graph_safe != 0; }

 private:
  const Spec& spec_;
  const apxinf_attention_policy_t& policy_;
  int device_;
  Allocation output_;
  apxinf_attention_bindings_t bindings_;
};

}  // namespace

Recipe tune(const Spec& spec, const apxinf_attention_policy_t& policy,
            const apxinf_attention_bindings_t& bindings, int device,
            std::string& report) {
  AttentionTuningProblem problem(spec, policy, bindings, device);
  return apxinf::framework::autotune(problem, report);
}

}  // namespace apxinf::attention
