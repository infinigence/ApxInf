#include "internal.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <iomanip>

namespace apxinf::gemm {
namespace {

struct Allocation {
  void* pointer = nullptr;

  explicit Allocation(size_t bytes) { check_cuda(cudaMalloc(&pointer, bytes)); }
  ~Allocation() {
    if (pointer != nullptr) cudaFree(pointer);
  }
};

std::vector<float> read_output(void* pointer, size_t count, int dtype,
                               cudaStream_t stream) {
  std::vector<unsigned char> bytes(count * dtype_bytes(dtype));
  check_cuda(cudaMemcpyAsync(bytes.data(), pointer, bytes.size(),
                             cudaMemcpyDeviceToHost, stream));
  check_cuda(cudaStreamSynchronize(stream));
  std::vector<float> result(count);
  for (size_t index = 0; index < count; ++index) {
    if (dtype == APXINF_DTYPE_F32) {
      std::memcpy(&result[index], bytes.data() + 4 * index, 4);
    } else if (dtype == APXINF_DTYPE_F16) {
      half value;
      std::memcpy(&value, bytes.data() + 2 * index, 2);
      result[index] = __half2float(value);
    } else if (dtype == APXINF_DTYPE_BF16) {
      __nv_bfloat16 value;
      std::memcpy(&value, bytes.data() + 2 * index, 2);
      result[index] = __bfloat162float(value);
    } else {
      __nv_fp8_e4m3 value;
      std::memcpy(&value, bytes.data() + index, 1);
      result[index] = static_cast<float>(value);
    }
  }
  return result;
}

void poison(const apxinf_gemm_bindings_t& bindings, size_t bytes) {
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  check_cuda(cudaMemsetAsync(bindings.output, 0xff, bytes, stream));
  check_cuda(cudaStreamSynchronize(stream));
}

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

  void launch() { check_cuda(cudaGraphLaunch(executable, stream)); }
};

}  // namespace

AccuracyMetrics compare_reference(const std::vector<float>& expected,
                                  const std::vector<float>& actual) {
  constexpr double kMaximumScaledElementError = 0.08;
  constexpr double kMaximumRelativeL2 = 0.05;
  constexpr double kMinimumCosine = 0.9999;
  AccuracyMetrics metrics;
  if (expected.size() != actual.size()) return metrics;
  double squared_error = 0.0;
  double expected_squared = 0.0;
  double actual_squared = 0.0;
  double dot = 0.0;
  for (size_t index = 0; index < expected.size(); ++index) {
    if (!std::isfinite(expected[index]) || !std::isfinite(actual[index])) {
      return metrics;
    }
    const double difference =
        static_cast<double>(actual[index]) - expected[index];
    const double absolute = std::abs(difference);
    metrics.max_absolute_error =
        std::max(metrics.max_absolute_error, absolute);
    metrics.max_scaled_element_error = std::max(
        metrics.max_scaled_element_error,
        absolute / std::max(1.0, std::abs(static_cast<double>(expected[index]))));
    squared_error += difference * difference;
    expected_squared +=
        static_cast<double>(expected[index]) * expected[index];
    actual_squared += static_cast<double>(actual[index]) * actual[index];
    dot += static_cast<double>(expected[index]) * actual[index];
  }
  metrics.finite = true;
  metrics.relative_l2 =
      std::sqrt(squared_error / std::max(expected_squared, 1e-20));
  if (expected_squared <= 1e-20 && actual_squared <= 1e-20) {
    metrics.cosine = 1.0;
  } else if (expected_squared <= 1e-20 || actual_squared <= 1e-20) {
    metrics.cosine = 0.0;
  } else {
    metrics.cosine = dot / std::sqrt(expected_squared * actual_squared);
  }
  metrics.valid = metrics.max_scaled_element_error <=
                      kMaximumScaledElementError &&
                  metrics.relative_l2 <= kMaximumRelativeL2 &&
                  metrics.cosine >= kMinimumCosine;
  return metrics;
}

std::string format_accuracy(const AccuracyMetrics& metrics) {
  std::ostringstream output;
  output << std::setprecision(8) << "finite=" << metrics.finite
         << " max_absolute=" << metrics.max_absolute_error
         << " max_element=" << metrics.max_scaled_element_error
         << " rel_l2=" << metrics.relative_l2
         << " cosine=" << metrics.cosine;
  return output.str();
}

void validate_candidates(const Spec& spec,
                         const apxinf_gemm_policy_t& policy,
                         const apxinf_gemm_bindings_t& execution_bindings,
                         int device, const float* expected,
                         size_t expected_len) {
  const size_t count = static_cast<size_t>(
      spec.m * (spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_GEGLU
                    ? spec.n / 2
                    : spec.n));
  if (expected == nullptr || expected_len != count) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid candidate validation output");
  }
  const std::vector<float> reference(expected, expected + expected_len);
  Allocation output(count * dtype_bytes(spec.output_dtype));
  auto bindings = execution_bindings;
  bindings.output = output.pointer;
  const size_t output_bytes = count * dtype_bytes(spec.output_dtype);
  size_t implementations_checked = 0;

  for (const auto& implementation : registry(spec.semantic)) {
    if (!supports_device(implementation, device) ||
        !implementation.supports(spec) ||
        !supports_alignment(implementation, spec) ||
        (policy.graph_safe && !implementation.graph_safe) ||
        (policy.deterministic && !implementation.deterministic)) {
      continue;
    }
    bool implementation_checked = false;
    std::vector<int> configurations;
    implementation.enumerate_configs(spec, configurations);
    for (int configuration : configurations) {
      std::unique_ptr<Execution> candidate;
      std::vector<float> actual;
      try {
        candidate = prepare(implementation, configuration, spec, policy,
                            bindings, device);
        poison(bindings, output_bytes);
        check_cuda(implementation.enqueue(*candidate));
        actual = read_output(output.pointer, count, spec.output_dtype,
                             static_cast<cudaStream_t>(bindings.stream));
      } catch (const Failure&) {
        cudaGetLastError();
        continue;
      }

      const std::string label = std::string(implementation.name) + "#" +
                                std::to_string(configuration);
      const auto eager_accuracy = compare_reference(reference, actual);
      if (!eager_accuracy.valid) {
        throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                      label + " failed eager precision: " +
                          format_accuracy(eager_accuracy));
      }

      if (policy.graph_safe) {
        CapturedExecution graph(*candidate);
        poison(bindings, output_bytes);
        graph.launch();
        const auto graph_actual = read_output(
            output.pointer, count, spec.output_dtype,
            static_cast<cudaStream_t>(bindings.stream));
        const auto graph_accuracy = compare_reference(reference, graph_actual);
        if (!graph_accuracy.valid) {
          throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                        label + " failed graph replay precision: " +
                            format_accuracy(graph_accuracy));
        }
      }
      implementation_checked = true;
    }
    if (!implementation_checked) {
      throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                    std::string(implementation.name) +
                        " has no executable configuration for candidate validation");
    }
    ++implementations_checked;
  }

  if (implementations_checked == 0) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "no registered candidate applies to the validation Spec");
  }
}

}  // namespace apxinf::gemm
