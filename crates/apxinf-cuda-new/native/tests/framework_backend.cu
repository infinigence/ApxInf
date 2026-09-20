#include "../framework/autotune.h"
#include "../framework/registry.h"
#include "../framework/runtime_internal.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using apxinf::framework::Failure;
using apxinf::framework::Recipe;
using apxinf::framework::Registry;

void set_error(char* output, size_t capacity, const std::string& message) {
  if (output == nullptr || capacity == 0) return;
  std::snprintf(output, capacity, "%s", message.c_str());
}

template <class Function>
int contract_boundary(char* error, size_t capacity, Function&& function) {
  try {
    function();
    set_error(error, capacity, "");
    return 1;
  } catch (const std::exception& failure) {
    set_error(error, capacity, failure.what());
  } catch (...) {
    set_error(error, capacity, "unknown C++ exception");
  }
  return 0;
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

struct RegistryCandidate {
  uint32_t provider_id;
  uint32_t implementation_id;
  uint32_t implementation_version;
  const char* name;
};

void check_registry_contract() {
  const RegistryCandidate expected{17, 23, 42, "exact"};
  const RegistryCandidate neighbor{17, 23, 43, "neighbor"};
  const Registry<RegistryCandidate> registry{expected, neighbor};

  const auto* match = registry.find(17, 23, 42);
  require(match != nullptr && std::strcmp(match->name, "exact") == 0,
          "registry missed the exact provider/implementation/version triple");
  require(registry.find(18, 23, 42) == nullptr,
          "registry matched a different provider_id");
  require(registry.find(17, 24, 42) == nullptr,
          "registry matched a different implementation_id");
  require(registry.find(17, 23, 41) == nullptr,
          "registry matched a different implementation_version");

  size_t count = 0;
  for (const auto& candidate : registry) {
    (void)candidate;
    ++count;
  }
  require(count == 2, "registry iteration did not expose every candidate");
}

std::vector<std::filesystem::path> regular_files(
    const std::filesystem::path& root) {
  std::vector<std::filesystem::path> files;
  if (!std::filesystem::exists(root)) return files;
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(root)) {
    if (entry.is_regular_file()) files.push_back(entry.path());
  }
  return files;
}

void check_recipe_db_contract(const std::string& directory) {
  constexpr const char* key = "framework-contract/exact-key";
  constexpr const char* wrong_key = "framework-contract/wrong-key";
  constexpr const char* first = "recipe-contract-value-v1";
  constexpr const char* updated = "recipe-contract-value-v2-updated";

  apxinf::framework::write_recipe(directory, key, first);
  require(apxinf::framework::read_recipe(directory, key) == first,
          "recipe DB failed exact-key round-trip");
  require(apxinf::framework::read_recipe(directory, wrong_key).empty(),
          "recipe DB returned a value for a different key");

  apxinf::framework::write_recipe(directory, key, updated);
  require(apxinf::framework::read_recipe(directory, key) == updated,
          "recipe DB did not return the updated value for the same key");

  const auto files = regular_files(directory);
  require(!files.empty(), "recipe DB write created no persistent file");
  bool truncated = false;
  for (const auto& file : files) {
    std::error_code error;
    const auto size = std::filesystem::file_size(file, error);
    if (error || size == 0) continue;
    std::filesystem::resize_file(file, 1, error);
    require(!error, "failed to truncate recipe DB file " + file.string() +
                        ": " + error.message());
    truncated = true;
  }
  require(truncated, "recipe DB had no non-empty file to truncate");
  require(apxinf::framework::read_recipe(directory, key).empty(),
          "recipe DB accepted corrupted/truncated persistent content");
}

__global__ void timed_kernel(unsigned long long cycles,
                             unsigned long long* result) {
  const unsigned long long start = clock64();
  while (clock64() - start < cycles) {
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) *result = clock64() - start;
}

struct AutotuneCandidate {
  uint32_t provider_id;
  uint32_t implementation_id;
  uint32_t implementation_version;
  const char* name;
};

struct AutotuneExecution {
  const AutotuneCandidate* implementation;
  int configuration;
};

class ContractProblem {
 public:
  using Implementation = AutotuneCandidate;
  using Execution = AutotuneExecution;

  ContractProblem(bool reverse, bool graph_safe)
      : registry_({{100, 1, 1, "unsupported"},
                   {200, 2, 7, "contract-valid"}}),
        reverse_(reverse),
        graph_safe_(graph_safe) {
    const cudaError_t stream_status =
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (stream_status != cudaSuccess) {
      throw std::runtime_error(std::string("cudaStreamCreate failed: ") +
                               cudaGetErrorString(stream_status));
    }
    const cudaError_t allocation_status = cudaMalloc(&result_, sizeof(*result_));
    if (allocation_status != cudaSuccess) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
      throw std::runtime_error(std::string("cudaMalloc failed: ") +
                               cudaGetErrorString(allocation_status));
    }
  }

  ~ContractProblem() {
    if (result_ != nullptr) cudaFree(result_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
  }

  const Registry<Implementation>& registry() const { return registry_; }

  bool supports(const Implementation& implementation, std::string& reason) {
    if (implementation.provider_id == 100) {
      ++unsupported_support_checks;
      reason = "contract-test-unsupported";
      return false;
    }
    ++supported_support_checks;
    return true;
  }

  void configurations(const Implementation& implementation,
                      std::vector<int>& values) {
    if (implementation.provider_id == 100) {
      ++unsupported_configuration_calls;
      values.push_back(99);
      return;
    }
    ++supported_configuration_calls;
    if (reverse_) {
      values.insert(values.end(), {40, 30, 20, 10});
    } else {
      values.insert(values.end(), {10, 20, 30, 40});
    }
  }

  std::unique_ptr<Execution> prepare(const Implementation& implementation,
                                     int configuration) {
    if (implementation.provider_id == 100) ++unsupported_prepare_calls;
    ++prepare_calls[configuration];
    if (configuration == 10) {
      throw Failure(APXINF_STATUS_INTERNAL_ERROR,
                    "intentional prepare rejection");
    }
    return std::make_unique<Execution>(
        Execution{&implementation, configuration});
  }

  cudaError_t enqueue(Execution& execution) {
    ++enqueue_calls[execution.configuration];
    if (execution.configuration == 20) return cudaErrorInvalidValue;

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const cudaError_t query_status =
        cudaStreamIsCapturing(stream_, &capture_status);
    if (query_status != cudaSuccess) return query_status;
    if (capture_status != cudaStreamCaptureStatusNone) {
      ++capture_calls[execution.configuration];
    }

    const unsigned long long cycles =
        execution.configuration == 30 ? 20000000ULL : 10000ULL;
    timed_kernel<<<1, 1, 0, stream_>>>(cycles, result_);
    return cudaGetLastError();
  }

  cudaStream_t stream() const { return stream_; }
  bool graph_safe() const { return graph_safe_; }

  int unsupported_support_checks = 0;
  int supported_support_checks = 0;
  int unsupported_configuration_calls = 0;
  int supported_configuration_calls = 0;
  int unsupported_prepare_calls = 0;
  int prepare_calls[64]{};
  int enqueue_calls[64]{};
  int capture_calls[64]{};

 private:
  Registry<Implementation> registry_;
  bool reverse_;
  bool graph_safe_;
  cudaStream_t stream_ = nullptr;
  unsigned long long* result_ = nullptr;
};

void validate_autotune_run(bool reverse, bool graph_safe) {
  ContractProblem problem(reverse, graph_safe);
  std::string report;
  const Recipe winner = apxinf::framework::autotune(problem, report);

  std::ostringstream context;
  context << " (order=" << (reverse ? "reverse" : "forward")
          << ", graph_safe=" << graph_safe << ", report=" << report << ')';
  const std::string suffix = context.str();
  require(winner.provider_id == 200 && winner.implementation_id == 2 &&
              winner.implementation_version == 7 &&
              winner.configuration == 40,
          "autotune did not choose the valid fast CUDA candidate" + suffix);
  require(problem.unsupported_support_checks > 0,
          "autotune never evaluated unsupported candidate" + suffix);
  require(problem.unsupported_configuration_calls == 0 &&
              problem.unsupported_prepare_calls == 0,
          "autotune did not skip unsupported candidate" + suffix);
  require(problem.supported_configuration_calls > 0,
          "autotune did not enumerate supported configurations" + suffix);
  require(problem.prepare_calls[10] > 0 && problem.prepare_calls[20] > 0 &&
              problem.prepare_calls[30] > 0 && problem.prepare_calls[40] > 0,
          "autotune did not visit every legal configuration" + suffix);
  require(problem.enqueue_calls[20] > 0,
          "autotune did not exercise/reject enqueue failure" + suffix);
  require(problem.enqueue_calls[30] > 0 && problem.enqueue_calls[40] > 0,
          "autotune did not time both valid CUDA configurations" + suffix);
  if (graph_safe) {
    require(problem.capture_calls[40] > 0,
            "graph-safe winner was not successfully CUDA Graph captured" +
                suffix);
  }
}

void check_autotune_contract(bool graph_safe) {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  require(count_status == cudaSuccess,
          std::string("cudaGetDeviceCount failed: ") +
              cudaGetErrorString(count_status));
  require(device_count > 0, "autotune contract test requires a CUDA GPU");
  const cudaError_t device_status = cudaSetDevice(0);
  require(device_status == cudaSuccess,
          std::string("cudaSetDevice(0) failed: ") +
              cudaGetErrorString(device_status));

  validate_autotune_run(false, graph_safe);
  validate_autotune_run(true, graph_safe);
}

}  // namespace

extern "C" int apxinf_framework_test_registry_contract(char* error,
                                                         size_t capacity) {
  return contract_boundary(error, capacity, check_registry_contract);
}

extern "C" int apxinf_framework_test_recipe_db_contract(
    const char* directory, char* error, size_t capacity) {
  return contract_boundary(error, capacity, [&] {
    require(directory != nullptr && directory[0] != '\0',
            "recipe DB test requires a non-empty directory");
    check_recipe_db_contract(directory);
  });
}

extern "C" int apxinf_framework_test_autotune_contract(int graph_safe,
                                                         char* error,
                                                         size_t capacity) {
  return contract_boundary(error, capacity,
                           [&] { check_autotune_contract(graph_safe != 0); });
}
