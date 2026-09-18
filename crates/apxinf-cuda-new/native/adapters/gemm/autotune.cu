#include "internal.h"

#include <limits>

namespace apxinf::gemm {
namespace {

struct Events {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  Events() {
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));
  }
  ~Events() {
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
  }
};

struct Allocation {
  void* pointer = nullptr;

  explicit Allocation(size_t bytes) { check_cuda(cudaMalloc(&pointer, bytes)); }
  ~Allocation() {
    if (pointer != nullptr) cudaFree(pointer);
  }
};

struct CapturedExecution {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  cudaStream_t stream = nullptr;

  explicit CapturedExecution(Execution& candidate)
      : stream(static_cast<cudaStream_t>(candidate.bindings.stream)) {
    check_cuda(cudaStreamBeginCapture(stream,
                                      cudaStreamCaptureModeThreadLocal));
    try {
      check_cuda(candidate.implementation->enqueue(candidate));
    } catch (...) {
      cudaStreamEndCapture(stream, &graph);
      if (graph != nullptr) cudaGraphDestroy(graph);
      graph = nullptr;
      throw;
    }
    check_cuda(cudaStreamEndCapture(stream, &graph));
    const auto status =
        cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0);
    if (status != cudaSuccess) {
      cudaGraphDestroy(graph);
      graph = nullptr;
      check_cuda(status);
    }
  }

  ~CapturedExecution() {
    if (executable != nullptr) cudaGraphExecDestroy(executable);
    if (graph != nullptr) cudaGraphDestroy(graph);
  }
};

}  // namespace

Recipe tune(
    const Spec& spec, const apxinf_gemm_policy_t& policy,
    const apxinf_gemm_bindings_t& execution_bindings, int device,
    std::string& report, const Recipe* preferred) {
  const size_t count = static_cast<size_t>(
      spec.m * (is_gated_semantic(spec) ? spec.n / 2 : spec.n));
  Allocation output(count * dtype_bytes(spec.output_dtype));
  auto bindings = execution_bindings;
  bindings.output = output.pointer;

  std::unique_ptr<Execution> winner;
  float best = std::numeric_limits<float>::infinity();
  Events events;
  int checked = 0;
  int rejected = 0;
  std::vector<std::string> diagnostics;
  const auto& implementations = registry(spec.semantic);
  struct Candidate {
    const Implementation* implementation;
    int configuration;
  };
  std::vector<Candidate> candidates;
  for (const auto& implementation : implementations) {
    if (!supports_device(implementation, device)) {
      diagnostics.push_back(std::string(implementation.name) +
                            "=skip(device)");
      continue;
    }
    if (!implementation.supports(spec)) {
      diagnostics.push_back(std::string(implementation.name) +
                            "=skip(contract)");
      continue;
    }
    if (!supports_alignment(implementation, spec)) {
      diagnostics.push_back(std::string(implementation.name) +
                            "=skip(alignment)");
      continue;
    }
    if (policy.graph_safe && !implementation.graph_safe) {
      diagnostics.push_back(std::string(implementation.name) +
                            "=skip(graph-safe)");
      continue;
    }
    if (policy.deterministic && !implementation.deterministic) {
      diagnostics.push_back(std::string(implementation.name) +
                            "=skip(determinism)");
      continue;
    }
    std::vector<int> configurations;
    implementation.enumerate_configs(spec, configurations);
    for (int configuration : configurations) {
      candidates.push_back({&implementation, configuration});
    }
  }
  bool preferred_first = false;
  if (preferred != nullptr) {
    const auto match = [&](const Candidate& candidate) {
      return candidate.implementation->provider_id == preferred->provider_id &&
             candidate.implementation->implementation_id ==
                 preferred->implementation_id &&
             candidate.implementation->implementation_version ==
                 preferred->implementation_version &&
             candidate.configuration == preferred->configuration;
    };
    const auto found = std::find_if(candidates.begin(), candidates.end(), match);
    if (found != candidates.end()) {
      std::rotate(candidates.begin(), found, std::next(found));
      preferred_first = true;
    }
  }
  for (const auto& selected : candidates) {
    const auto& implementation = *selected.implementation;
    const int configuration = selected.configuration;
    try {
        auto candidate = prepare(implementation, configuration, spec, policy,
                                 bindings, device);
        const std::string label = std::string(implementation.name) + "#" +
                                  std::to_string(configuration);
        for (int iteration = 0; iteration < 3; ++iteration) {
          check_cuda(implementation.enqueue(*candidate));
        }
        check_cuda(cudaEventRecord(events.start,
                                   static_cast<cudaStream_t>(bindings.stream)));
        for (int iteration = 0; iteration < 10; ++iteration) {
          check_cuda(implementation.enqueue(*candidate));
        }
        check_cuda(cudaEventRecord(events.stop,
                                   static_cast<cudaStream_t>(bindings.stream)));
        check_cuda(cudaEventSynchronize(events.stop));
        float milliseconds = 0.0F;
        check_cuda(
            cudaEventElapsedTime(&milliseconds, events.start, events.stop));
        milliseconds /= 10.0F;
        ++checked;
        diagnostics.push_back(label + "=timed(" +
                              std::to_string(milliseconds) + "ms)");
        if (milliseconds < best) {
          best = milliseconds;
          winner = std::move(candidate);
        }
    } catch (const Failure& failure) {
      ++rejected;
      diagnostics.push_back(std::string(implementation.name) + "#" +
                            std::to_string(configuration) + "=reject(" +
                            failure.what() + ")");
      cudaGetLastError();
    }
  }
  if (winner == nullptr) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "no candidate satisfies the GEMM Spec and Policy");
  }
  if (policy.graph_safe) {
    // Graph support is a capability requirement, not a separate tuning
    // objective. Numeric replay validation belongs to the candidate tests.
    CapturedExecution graph(*winner);
    diagnostics.push_back("winner-graph=capture-pass");
  }
  report = "tuned preferred=" + std::to_string(preferred_first) +
           " checked=" + std::to_string(checked) +
           " rejected=" + std::to_string(rejected) +
           " ms=" + std::to_string(best) + " candidates=[";
  for (size_t index = 0; index < diagnostics.size(); ++index) {
    if (index != 0) report += ",";
    report += diagnostics[index];
  }
  report += "]";
  // Tuning executions are bound to a private output and must never be rebound
  // for serving. Only the address-independent Recipe crosses this boundary;
  // the caller creates a fresh Execution with the real bindings.
  return {winner->implementation->provider_id,
          winner->implementation->implementation_id,
          winner->implementation->implementation_version,
          winner->configuration};
}

}  // namespace apxinf::gemm
