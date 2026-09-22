// Operator-independent CUDA autotuning loop.
//
// A downstream adapter instantiates autotune<Problem> and must provide
// Problem::Implementation, Problem::Execution, and these methods:
//   registry(), supports(implementation, reason),
//   configurations(implementation, values), prepare(implementation, config),
//   enqueue(execution), stream(), and graph_safe().
// Implementation exposes name and recipe identity fields; Execution exposes
// implementation and configuration. The adapter must benchmark with isolated
// output storage so tuning cannot modify caller-visible results. Cache lookup,
// recipe persistence, and fallback policy remain in the operator adapter.
#pragma once

#include "runtime_internal.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace apxinf::framework {

struct EventPair {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  EventPair() {
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));
  }
  ~EventPair() {
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
  }
};

template <class Problem>
class CapturedCandidate {
 public:
  using Execution = typename Problem::Execution;

  CapturedCandidate(Problem& problem, Execution& execution)
      : problem_(problem), stream_(problem.stream()) {
    check_cuda(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
    try {
      check_cuda(problem_.enqueue(execution));
    } catch (...) {
      cudaStreamEndCapture(stream_, &graph_);
      if (graph_ != nullptr) cudaGraphDestroy(graph_);
      graph_ = nullptr;
      throw;
    }
    check_cuda(cudaStreamEndCapture(stream_, &graph_));
    const auto status =
        cudaGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0);
    if (status != cudaSuccess) {
      cudaGraphDestroy(graph_);
      graph_ = nullptr;
      check_cuda(status);
    }
  }

  ~CapturedCandidate() {
    if (executable_ != nullptr) cudaGraphExecDestroy(executable_);
    if (graph_ != nullptr) cudaGraphDestroy(graph_);
  }

 private:
  Problem& problem_;
  cudaStream_t stream_ = nullptr;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
};

// Operator-independent tactic loop. Problem supplies only operator-specific
// candidate filtering, configuration enumeration, preparation and launch.
template <class Problem>
Recipe autotune(Problem& problem, std::string& report) {
  using Implementation = typename Problem::Implementation;
  using Execution = typename Problem::Execution;
  struct Choice {
    const Implementation* implementation;
    int configuration;
  };

  std::vector<std::string> diagnostics;
  std::vector<Choice> choices;
  for (const auto& implementation : problem.registry()) {
    std::string reason;
    if (!problem.supports(implementation, reason)) {
      diagnostics.push_back(std::string(implementation.name) + "=skip(" +
                            reason + ")");
      continue;
    }
    std::vector<int> configurations;
    problem.configurations(implementation, configurations);
    for (int configuration : configurations) {
      choices.push_back({&implementation, configuration});
    }
  }

  std::unique_ptr<Execution> winner;
  float best = std::numeric_limits<float>::infinity();
  EventPair events;
  int checked = 0;
  int rejected = 0;
  for (const auto& choice : choices) {
    const auto& implementation = *choice.implementation;
    try {
      auto candidate = problem.prepare(implementation, choice.configuration);
      const std::string label = std::string(implementation.name) + "#" +
                                std::to_string(choice.configuration);
      for (int iteration = 0; iteration < 3; ++iteration) {
        check_cuda(problem.enqueue(*candidate));
      }
      check_cuda(cudaEventRecord(events.start, problem.stream()));
      for (int iteration = 0; iteration < 10; ++iteration) {
        check_cuda(problem.enqueue(*candidate));
      }
      check_cuda(cudaEventRecord(events.stop, problem.stream()));
      check_cuda(cudaEventSynchronize(events.stop));
      float milliseconds = 0.0F;
      check_cuda(cudaEventElapsedTime(&milliseconds, events.start, events.stop));
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
                            std::to_string(choice.configuration) + "=reject(" +
                            failure.what() + ")");
      cudaGetLastError();
    }
  }
  if (winner == nullptr) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "no candidate satisfies the operator Spec and Policy");
  }
  if (problem.graph_safe()) {
    CapturedCandidate<Problem> graph(problem, *winner);
    diagnostics.push_back("winner-graph=capture-pass");
  }

  report = "tuned checked=" + std::to_string(checked) +
           " rejected=" + std::to_string(rejected) +
           " ms=" + std::to_string(best) + " candidates=[";
  for (size_t index = 0; index < diagnostics.size(); ++index) {
    if (index != 0) report += ',';
    report += diagnostics[index];
  }
  report += ']';
  return {winner->implementation->provider_id,
          winner->implementation->implementation_id,
          winner->implementation->implementation_version,
          winner->configuration};
}

}  // namespace apxinf::framework
