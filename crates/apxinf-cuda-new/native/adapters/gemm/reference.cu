#include "internal.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <iomanip>

namespace apxinf::gemm {
namespace {

std::vector<float> read_device_values(const void* pointer, size_t count,
                                      uint32_t dtype, cudaStream_t stream) {
  std::vector<unsigned char> bytes(count * dtype_bytes(dtype));
  check_cuda(cudaMemcpyAsync(bytes.data(), pointer, bytes.size(),
                             cudaMemcpyDeviceToHost, stream));
  check_cuda(cudaStreamSynchronize(stream));
  std::vector<float> values(count);
  for (size_t index = 0; index < count; ++index) {
    if (dtype == APXINF_DTYPE_F32) {
      std::memcpy(&values[index], bytes.data() + 4 * index, 4);
    } else if (dtype == APXINF_DTYPE_F16) {
      half value;
      std::memcpy(&value, bytes.data() + 2 * index, 2);
      values[index] = __half2float(value);
    } else if (dtype == APXINF_DTYPE_BF16) {
      __nv_bfloat16 value;
      std::memcpy(&value, bytes.data() + 2 * index, 2);
      values[index] = __bfloat162float(value);
    } else if (dtype == APXINF_DTYPE_E4M3) {
      __nv_fp8_e4m3 value;
      std::memcpy(&value, bytes.data() + index, 1);
      values[index] = static_cast<float>(value);
    } else if (dtype == APXINF_DTYPE_I8) {
      int8_t value;
      std::memcpy(&value, bytes.data() + index, 1);
      values[index] = static_cast<float>(value);
    } else {
      int32_t value;
      std::memcpy(&value, bytes.data() + 4 * index, 4);
      values[index] = static_cast<float>(value);
    }
  }
  return values;
}

uint32_t bias_dtype(const Spec& spec) {
  if (has_row_channel_scales(spec)) return spec.output_dtype;
  if (spec.a_dtype == APXINF_DTYPE_E4M3) {
    return spec.output_dtype == APXINF_DTYPE_F32 ? APXINF_DTYPE_F32
                                                 : APXINF_DTYPE_F16;
  }
  return spec.a_dtype;
}

float gelu(float value) {
  return 0.5F * value *
         (1.0F + std::tanh(0.7978845608028654F *
                           (value + 0.044715F * value * value * value)));
}

std::vector<float> project(const Spec& spec, const std::vector<float>& a,
                           const std::vector<float>& b) {
  std::vector<float> projection(static_cast<size_t>(spec.m * spec.n));
  for (int64_t row = 0; row < spec.m; ++row) {
    for (int64_t column = 0; column < spec.n; ++column) {
      float accumulator = 0.0F;
      for (int64_t inner = 0; inner < spec.k; ++inner) {
        accumulator += a[static_cast<size_t>(row * spec.k + inner)] *
                       b[static_cast<size_t>(inner * spec.n + column)];
      }
      projection[static_cast<size_t>(row * spec.n + column)] = accumulator;
    }
  }
  return projection;
}

std::vector<float> reference_gemm(const std::vector<float>& projection,
                                  float alpha, float output_scale) {
  std::vector<float> output(projection.size());
  for (size_t index = 0; index < output.size(); ++index) {
    output[index] = alpha * projection[index] / output_scale;
  }
  return output;
}

std::vector<float> reference_gemm_bias(
    const Spec& spec, const std::vector<float>& projection,
    const std::vector<float>& bias, bool apply_gelu, float alpha,
    float output_scale) {
  std::vector<float> output(projection.size());
  for (int64_t row = 0; row < spec.m; ++row) {
    for (int64_t column = 0; column < spec.n; ++column) {
      const size_t index = static_cast<size_t>(row * spec.n + column);
      float value = alpha * projection[index] + bias[column];
      if (apply_gelu) value = gelu(value);
      output[index] = value / output_scale;
    }
  }
  return output;
}

std::vector<float> reference_gemm_geglu(
    const Spec& spec, const std::vector<float>& projection, float alpha,
    float output_scale) {
  const int64_t width = spec.n / 2;
  std::vector<float> output(static_cast<size_t>(spec.m * width));
  for (int64_t row = 0; row < spec.m; ++row) {
    for (int64_t column = 0; column < width; ++column) {
      const float gate =
          alpha * projection[static_cast<size_t>(row * spec.n + column)];
      const float up =
          alpha * projection[static_cast<size_t>(row * spec.n +
                                                 column + width)];
      output[static_cast<size_t>(row * width + column)] =
          gelu(gate) * up / output_scale;
    }
  }
  return output;
}

}  // namespace

ReferenceOutput cpu_reference(
    const Spec& spec, const apxinf_gemm_tuning_bindings_t& bindings) {
  if (bindings.reference_kind == APXINF_GEMM_REFERENCE_TORCH_OUTPUT) {
    ReferenceOutput result;
    result.kind = "torch";
    result.values.assign(bindings.expected_output,
                         bindings.expected_output +
                             bindings.expected_output_len);
    return result;
  }
  const auto stream = static_cast<cudaStream_t>(bindings.execution.stream);
  std::vector<float> a;
  std::vector<float> b;
  std::vector<float> bias;
  a = read_device_values(bindings.execution.a,
                         static_cast<size_t>(spec.m * spec.k), spec.a_dtype,
                         stream);
  b = read_device_values(bindings.execution.b,
                         static_cast<size_t>(spec.k * spec.n), spec.b_dtype,
                         stream);
  if (has_row_channel_scales(spec)) {
    const auto a_scales = read_device_values(
        bindings.execution.a_scales, static_cast<size_t>(spec.m),
        APXINF_DTYPE_F32, stream);
    const auto b_scales = read_device_values(
        bindings.execution.b_scales, static_cast<size_t>(spec.n),
        APXINF_DTYPE_F32, stream);
    for (int64_t row = 0; row < spec.m; ++row) {
      for (int64_t inner = 0; inner < spec.k; ++inner) {
        a[static_cast<size_t>(row * spec.k + inner)] *= a_scales[row];
      }
    }
    for (int64_t inner = 0; inner < spec.k; ++inner) {
      for (int64_t column = 0; column < spec.n; ++column) {
        b[static_cast<size_t>(inner * spec.n + column)] *= b_scales[column];
      }
    }
  }
  if (spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS ||
      spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS_GELU) {
    bias = read_device_values(bindings.execution.bias,
                              static_cast<size_t>(spec.n), bias_dtype(spec),
                              stream);
  }

  const auto projection = project(spec, a, b);
  ReferenceOutput result;
  result.kind = "dequantized-input";
  // The reference must reproduce the scales the candidate will actually be
  // launched with, which now live in the execution bindings.
  const float alpha = bindings.execution.alpha;
  const float output_scale = bindings.execution.output_scale;
  if (spec.semantic == APXINF_GEMM_SEMANTIC_GEMM) {
    result.values = reference_gemm(projection, alpha, output_scale);
  } else if (spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS) {
    result.values =
        reference_gemm_bias(spec, projection, bias, false, alpha, output_scale);
  } else if (spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS_GELU) {
    result.values =
        reference_gemm_bias(spec, projection, bias, true, alpha, output_scale);
  } else {
    result.values =
        reference_gemm_geglu(spec, projection, alpha, output_scale);
  }
  return result;
}

AccuracyMetrics compare_reference(const std::vector<float>& expected,
                                  const std::vector<float>& actual) {
  // One immutable acceptance contract applies to every provider and candidate.
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
  output << std::setprecision(6) << "finite=" << (metrics.finite ? 1 : 0)
         << ",max_abs=" << metrics.max_absolute_error
         << ",max_element=" << metrics.max_scaled_element_error
         << ",rel_l2=" << metrics.relative_l2
         << ",cosine=" << metrics.cosine;
  return output.str();
}

}  // namespace apxinf::gemm
