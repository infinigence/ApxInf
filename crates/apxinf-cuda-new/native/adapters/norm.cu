#include "../include/apxinf_cuda/norm.h"
#include "../framework/runtime_internal.h"
#include "../kernels/custom/norm.cuh"

#include <climits>
#include <cmath>
#include <cstdint>
#include <cuda_fp8.h>

namespace {

using apxinf::framework::Failure;
namespace kernels = apxinf::norm::kernels;

constexpr int kThreads = 256;
constexpr int kMaxElementwiseBlocks = 4096;

bool valid_alignment(uint32_t alignment) {
  return alignment <= 256 && alignment != 0 &&
         (alignment & (alignment - 1)) == 0;
}

bool writes_hidden(uint32_t semantic) {
  return semantic != APXINF_NORM_SEMANTIC_RMS &&
         semantic != APXINF_NORM_SEMANTIC_LAYER &&
         semantic != APXINF_NORM_SEMANTIC_ADAPTIVE_RMS;
}

bool writes_normalized(uint32_t semantic) {
  return semantic != APXINF_NORM_SEMANTIC_BIAS_RESIDUAL &&
         semantic != APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL &&
         semantic != APXINF_NORM_SEMANTIC_BIAS_THEN_RESIDUAL;
}

bool reads_residual(uint32_t semantic) { return writes_hidden(semantic); }

bool reads_weight(uint32_t semantic) {
  return semantic == APXINF_NORM_SEMANTIC_RMS ||
         semantic == APXINF_NORM_SEMANTIC_LAYER ||
         semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_RMS ||
         semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_LAYER;
}

bool reads_norm_bias(uint32_t semantic) {
  return semantic == APXINF_NORM_SEMANTIC_LAYER ||
         semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_LAYER;
}

bool reads_norm_style(uint32_t semantic) {
  return semantic == APXINF_NORM_SEMANTIC_ADAPTIVE_RMS ||
         semantic == APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL_RMS;
}

bool reads_gate_style(uint32_t semantic) {
  return semantic == APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL ||
         semantic == APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL_RMS;
}

bool may_have_bias(uint32_t semantic) {
  return semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL ||
         semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_RMS ||
         semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_LAYER ||
         semantic == APXINF_NORM_SEMANTIC_BIAS_THEN_RESIDUAL;
}

void validate_spec(const apxinf_norm_spec_t& spec) {
  const bool quantized_output =
      spec.dtype == APXINF_DTYPE_F16 &&
      spec.output_dtype == APXINF_DTYPE_E4M3 &&
      writes_normalized(spec.semantic);
  if (spec.version != APXINF_NORM_SPEC_VERSION ||
      spec.semantic > APXINF_NORM_SEMANTIC_BIAS_THEN_RESIDUAL ||
      (spec.dtype != APXINF_DTYPE_F16 && spec.dtype != APXINF_DTYPE_BF16) ||
      (spec.output_dtype != spec.dtype && !quantized_output) ||
      spec.rows <= 0 || spec.cols <= 0 ||
      spec.rows > INT32_MAX || spec.cols > INT32_MAX || spec.has_bias > 1 ||
      spec.output_scale_is_unit > 1 ||
      !valid_alignment(spec.input_alignment) ||
      !valid_alignment(spec.weight_alignment) ||
      !valid_alignment(spec.bias_alignment) ||
      !valid_alignment(spec.residual_alignment) ||
      !valid_alignment(spec.style_alignment) ||
      !valid_alignment(spec.hidden_alignment) ||
      !valid_alignment(spec.normalized_alignment)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid Norm Spec");
  }
  if (spec.semantic == APXINF_NORM_SEMANTIC_BIAS_THEN_RESIDUAL &&
      spec.dtype != APXINF_DTYPE_BF16) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "BiasThenResidual requires BF16 input and output");
  }
  if (spec.has_bias != 0 && !may_have_bias(spec.semantic)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Norm semantic does not take a bias");
  }
}

void validate_bindings(const apxinf_norm_spec_t& spec,
                       const apxinf_norm_bindings_t& bindings) {
  const uint32_t semantic = spec.semantic;
  const bool missing =
      bindings.input == nullptr ||
      (writes_hidden(semantic) && bindings.hidden == nullptr) ||
      (writes_normalized(semantic) && bindings.normalized == nullptr) ||
      (reads_residual(semantic) && bindings.residual == nullptr) ||
      (reads_weight(semantic) && bindings.weight == nullptr) ||
      (reads_norm_bias(semantic) && bindings.norm_bias == nullptr) ||
      (reads_norm_style(semantic) && bindings.norm_style == nullptr) ||
      (reads_gate_style(semantic) && bindings.gate_style == nullptr);
  if (missing) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "missing Norm binding for semantic");
  }
  if ((spec.has_bias != 0) != (bindings.bias != nullptr)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Norm bias binding disagrees with Spec.has_bias");
  }
  const bool scale_matches_spec =
      (bindings.output_scale == 1.0f) ==
      (spec.output_scale_is_unit != 0);
  const bool valid_output_scale =
      std::isfinite(bindings.output_scale) && bindings.output_scale > 0.0f &&
      scale_matches_spec &&
      (spec.output_dtype == APXINF_DTYPE_E4M3 ||
       bindings.output_scale == 1.0f);
  if (!valid_output_scale || !(bindings.eps > 0.0f) ||
      !std::isfinite(bindings.eps)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid Norm scalar");
  }
}

int elementwise_blocks(int64_t count) {
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  if (blocks < 1) return 1;
  return static_cast<int>(blocks < kMaxElementwiseBlocks
                              ? blocks
                              : kMaxElementwiseBlocks);
}

cudaError_t launch_quantized_f16(const apxinf_norm_spec_t& spec,
                                 const apxinf_norm_bindings_t& bindings) {
  const auto* input = static_cast<const half*>(bindings.input);
  const auto* bias = static_cast<const half*>(bindings.bias);
  const auto* residual = static_cast<const half*>(bindings.residual);
  const auto* weight = static_cast<const half*>(bindings.weight);
  const auto* norm_bias = static_cast<const half*>(bindings.norm_bias);
  const auto* norm_style = static_cast<const half*>(bindings.norm_style);
  const auto* gate_style = static_cast<const half*>(bindings.gate_style);
  auto* hidden = static_cast<half*>(bindings.hidden);
  auto* normalized = static_cast<__nv_fp8_e4m3*>(bindings.normalized);
  const int rows = static_cast<int>(spec.rows);
  const int cols = static_cast<int>(spec.cols);
  const float inverse_scale = 1.0f / bindings.output_scale;
  auto stream = static_cast<cudaStream_t>(bindings.stream);

  switch (spec.semantic) {
    case APXINF_NORM_SEMANTIC_RMS:
      kernels::rms_norm_quant_f16_e4m3<<<rows, kThreads, 0, stream>>>(
          input, weight, normalized, rows, cols, bindings.eps, inverse_scale);
      break;
    case APXINF_NORM_SEMANTIC_LAYER:
      kernels::layer_norm_quant_f16_e4m3<<<rows, kThreads, 0, stream>>>(
          input, weight, norm_bias, normalized, rows, cols, bindings.eps,
          inverse_scale);
      break;
    case APXINF_NORM_SEMANTIC_ADAPTIVE_RMS:
      kernels::ada_rms_norm_quant_f16_e4m3<<<rows, kThreads, 0, stream>>>(
          input, norm_style, normalized, rows, cols, bindings.eps,
          inverse_scale);
      break;
    case APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_RMS:
      kernels::bias_residual_rms_norm_quant_f16_e4m3
          <<<rows, kThreads, 0, stream>>>(
              input, bias, residual, weight, hidden, normalized, rows, cols,
              bindings.eps, inverse_scale);
      break;
    case APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_LAYER:
      kernels::bias_residual_layer_norm_quant_f16_e4m3
          <<<rows, kThreads, 0, stream>>>(
              input, bias, residual, weight, norm_bias, hidden, normalized,
              rows, cols, bindings.eps, inverse_scale);
      break;
    case APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL_RMS:
      if (rows == 10 && cols == 1024) {
        kernels::ada_gate_residual_rms_norm_quant_f16_e4m3_10x1024
            <<<rows, kThreads, 0, stream>>>(
                input, residual, gate_style, norm_style, hidden, normalized,
                bindings.eps, inverse_scale);
      } else {
        kernels::ada_gate_residual_rms_norm_quant_f16_e4m3
            <<<rows, kThreads, 0, stream>>>(
                input, residual, gate_style, norm_style, hidden, normalized,
                rows, cols, bindings.eps, inverse_scale);
      }
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

template <class T>
cudaError_t launch(const apxinf_norm_spec_t& spec,
                   const apxinf_norm_bindings_t& bindings) {
  const auto* input = static_cast<const T*>(bindings.input);
  const auto* bias = static_cast<const T*>(bindings.bias);
  const auto* residual = static_cast<const T*>(bindings.residual);
  const auto* weight = static_cast<const T*>(bindings.weight);
  const auto* norm_bias = static_cast<const T*>(bindings.norm_bias);
  const auto* norm_style = static_cast<const T*>(bindings.norm_style);
  const auto* gate_style = static_cast<const T*>(bindings.gate_style);
  auto* hidden = static_cast<T*>(bindings.hidden);
  auto* normalized = static_cast<T*>(bindings.normalized);
  const int rows = static_cast<int>(spec.rows);
  const int cols = static_cast<int>(spec.cols);
  const int64_t count = spec.rows * spec.cols;
  auto stream = static_cast<cudaStream_t>(bindings.stream);

  switch (spec.semantic) {
    case APXINF_NORM_SEMANTIC_RMS:
      kernels::rms_norm<T><<<rows, kThreads, 0, stream>>>(
          input, weight, normalized, rows, cols, bindings.eps);
      break;
    case APXINF_NORM_SEMANTIC_LAYER:
      kernels::layer_norm<T><<<rows, kThreads, 0, stream>>>(
          input, weight, norm_bias, normalized, rows, cols, bindings.eps);
      break;
    case APXINF_NORM_SEMANTIC_ADAPTIVE_RMS:
      kernels::ada_rms_norm<T><<<rows, kThreads, 0, stream>>>(
          input, norm_style, normalized, rows, cols, bindings.eps);
      break;
    case APXINF_NORM_SEMANTIC_BIAS_RESIDUAL:
      kernels::bias_residual<T>
          <<<elementwise_blocks(count), kThreads, 0, stream>>>(
              input, bias, residual, hidden, count, cols);
      break;
    case APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_RMS:
      kernels::bias_residual_rms_norm<T><<<rows, kThreads, 0, stream>>>(
          input, bias, residual, weight, hidden, normalized, rows, cols,
          bindings.eps);
      break;
    case APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_LAYER:
      kernels::bias_residual_layer_norm<T><<<rows, kThreads, 0, stream>>>(
          input, bias, residual, weight, norm_bias, hidden, normalized, rows,
          cols, bindings.eps);
      break;
    case APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL:
      kernels::ada_gate_residual<T>
          <<<elementwise_blocks(count), kThreads, 0, stream>>>(
              input, residual, gate_style, hidden, count, cols);
      break;
    case APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL_RMS:
      kernels::ada_gate_residual_rms_norm<T><<<rows, kThreads, 0, stream>>>(
          input, residual, gate_style, norm_style, hidden, normalized, rows,
          cols, bindings.eps);
      break;
    case APXINF_NORM_SEMANTIC_BIAS_THEN_RESIDUAL:
      kernels::bias_then_residual<T>
          <<<elementwise_blocks(count), kThreads, 0, stream>>>(
              input, bias, residual, hidden, count, cols);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace

extern "C" apxinf_status_t apxinf_norm_launch(
    apxinf_runtime_t runtime, const apxinf_norm_spec_t* spec,
    const apxinf_norm_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || bindings == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null Norm argument");
    }
    validate_spec(*spec);
    validate_bindings(*spec, *bindings);
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    apxinf::framework::check_cuda(
        spec->output_dtype == APXINF_DTYPE_E4M3
            ? launch_quantized_f16(*spec, *bindings)
            : (spec->dtype == APXINF_DTYPE_BF16
                   ? launch<__nv_bfloat16>(*spec, *bindings)
                   : launch<__half>(*spec, *bindings)));
  });
}
