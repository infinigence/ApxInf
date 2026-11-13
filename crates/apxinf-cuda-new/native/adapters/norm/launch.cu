#include "../../include/apxinf_cuda/norm.h"
#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/norm.cuh"
#include "../../kernels/custom/norm_head.h"
#include "../../kernels/custom/norm_weight_offset.h"
#include "../../kernels/custom/quant_ops.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <cuda_fp8.h>
#include <type_traits>

extern "C" cudaError_t apxinf_norm_rms_norm_f16(
    const void* input, const void* weight, void* output, int rows, int cols,
    float eps, cudaStream_t stream);
extern "C" cudaError_t apxinf_norm_layer_norm_f16(
    const void* input, const void* weight, const void* bias, void* output,
    int rows, int cols, float eps, cudaStream_t stream);
extern "C" cudaError_t apxinf_norm_bias_then_residual_bf16(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream);
extern "C" cudaError_t apxinf_norm_bias_residual_bf16_packed4(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream);
extern "C" cudaError_t apxinf_norm_bias_then_residual_bf16_packed4(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream);
extern "C" cudaError_t apxinf_norm_bias_residual_layer_norm_bf16_cached_1024(
    const void* projection, const void* projection_bias, const void* residual,
    const void* norm_weight, const void* norm_bias, void* hidden,
    void* normalized, int rows, int cols, float eps, cudaStream_t stream);
extern "C" cudaError_t apxinf_norm_bias_then_residual_layer_norm_bf16_cached_1536(
    const void* projection, const void* projection_bias, const void* residual,
    const void* norm_weight, const void* norm_bias, void* hidden,
    void* normalized, int rows, int cols, float eps, cudaStream_t stream);
extern "C" cudaError_t
apxinf_norm_bias_then_residual_adaptive_layer_norm_bf16_cached_1536(
    const void* projection, const void* projection_bias, const void* residual,
    const void* modulation, void* hidden, void* normalized, int rows, int cols,
    float eps, cudaStream_t stream);
extern "C" cudaError_t apxinf_norm_adaptive_layer_norm_bf16(
    const void* input, const void* modulation, void* output, uint32_t rows,
    uint32_t cols, float eps, cudaStream_t stream);
extern "C" cudaError_t apxinf_norm_adaptive_layer_norm_quant_bf16_e4m3(
    const void* input, const void* modulation, void* output, void* quantized,
    uint32_t rows, uint32_t cols, float eps, float scale,
    cudaStream_t stream);
extern "C" cudaError_t apxinf_norm_rms_norm_quant_bf16_e4m3(
    const void* input, const void* weight, void* output, int rows, int cols,
    float eps, float scale, cudaStream_t stream);
extern "C" cudaError_t apxinf_norm_layer_norm_quant_bf16_e4m3(
    const void* input, const void* weight, const void* bias, void* output,
    int rows, int cols, float eps, float scale, cudaStream_t stream);
extern "C" cudaError_t apxinf_norm_bias_residual_layer_norm_quant_bf16_e4m3(
    const void* projection, const void* projection_bias,
    const void* residual, const void* norm_weight, const void* norm_bias,
    void* hidden, void* normalized, int rows, int cols, float eps,
    float scale, cudaStream_t stream);
extern "C" cudaError_t apxinf_norm_fixed_decode_rms_norm_f32(
    const void* input, const void* weight, void* output, uint32_t cols,
    uint32_t rows, float eps, void* stream);
extern "C" cudaError_t apxinf_norm_fixed_decode_rms_norm_bf16(
    const void* input, const void* weight, void* output, uint32_t cols,
    uint32_t rows, float eps, void* stream);
extern "C" cudaError_t apxinf_norm_fixed_decode_rms_norm_add_bf16(
    void* x_inout, const void* delta, const void* weight, void* output,
    uint32_t cols, uint32_t rows, float eps, void* stream);
extern "C" cudaError_t
apxinf_norm_adaptive_layer_norm_quantize_rows_bf16_int8(
    const void* input, const void* modulation, void* output, void* quantized,
    void* scales, int rows, int cols, float eps, cudaStream_t stream);
extern "C" cudaError_t
apxinf_norm_layer_norm_quantize_rows_bf16_int8(
    const void* input, const void* weight, const void* bias, void* output,
    void* quantized, void* scales, int rows, int cols, float eps,
    cudaStream_t stream);
extern "C" cudaError_t apxinf_norm_rms_norm_plus1_bf16(
    const void* input, const void* weight, void* output, int rows, int cols,
    float eps, cudaStream_t stream);
extern "C" cudaError_t apxinf_norm_add_rms_norm_plus1_bf16(
    const void* a, const void* b, const void* weight, void* sum_out,
    void* output, int rows, int cols, float eps, cudaStream_t stream);

namespace {

using apxinf::framework::Failure;
namespace kernels = apxinf::norm::kernels;

constexpr int kThreads = 256;
constexpr int kMaxElementwiseBlocks = 4096;

bool valid_alignment(uint32_t alignment) {
  return alignment <= 256 && alignment != 0 &&
         (alignment & (alignment - 1)) == 0;
}

bool channel_update(const apxinf_norm_spec_t& spec) {
  return spec.operation == APXINF_NORM_OPERATION_BIAS_NORM_SILU ||
         spec.operation == APXINF_NORM_OPERATION_FUSED_ADD_NORM_SILU;
}

bool writes_hidden(const apxinf_norm_spec_t& spec) {
  return spec.operation == APXINF_NORM_OPERATION_FUSED_ADD_NORM ||
         spec.operation == APXINF_NORM_OPERATION_RESIDUAL_ADD || channel_update(spec);
}

bool writes_normalized(const apxinf_norm_spec_t& spec) {
  return spec.operation != APXINF_NORM_OPERATION_RESIDUAL_ADD;
}

bool reads_residual(const apxinf_norm_spec_t& spec) {
  return writes_hidden(spec) && spec.operation != APXINF_NORM_OPERATION_BIAS_NORM_SILU;
}

bool reads_weight(const apxinf_norm_spec_t& spec) {
  return spec.affine == APXINF_NORM_AFFINE_WEIGHT;
}

bool reads_norm_bias(const apxinf_norm_spec_t& spec) {
  return spec.statistic == APXINF_NORM_STATISTIC_LAYER &&
         spec.affine == APXINF_NORM_AFFINE_WEIGHT;
}

bool reads_norm_style(const apxinf_norm_spec_t& spec) {
  return writes_normalized(spec) &&
         spec.affine == APXINF_NORM_AFFINE_ADAPTIVE;
}

bool reads_gate_style(const apxinf_norm_spec_t& spec) {
  return spec.gate == APXINF_NORM_GATE_STYLE;
}

bool may_have_bias(const apxinf_norm_spec_t& spec) {
  return writes_hidden(spec) && spec.gate == APXINF_NORM_GATE_NONE;
}

bool is_contract(const apxinf_norm_spec_t& spec, uint32_t operation,
                 uint32_t statistic, uint32_t affine, uint32_t gate) {
  return spec.operation == operation && spec.statistic == statistic &&
         spec.affine == affine && spec.gate == gate;
}

bool valid_contract(const apxinf_norm_spec_t& spec) {
  return
      (channel_update(spec) && spec.statistic == APXINF_NORM_STATISTIC_CHANNEL_L2 &&
       spec.affine == APXINF_NORM_AFFINE_WEIGHT && spec.gate == APXINF_NORM_GATE_NONE) ||
      is_contract(spec, APXINF_NORM_OPERATION_NORM_SILU,
                  APXINF_NORM_STATISTIC_CHANNEL_L2, APXINF_NORM_AFFINE_WEIGHT,
                  APXINF_NORM_GATE_NONE) ||
      is_contract(spec, APXINF_NORM_OPERATION_NORM,
                  APXINF_NORM_STATISTIC_CHANNEL_L2, APXINF_NORM_AFFINE_WEIGHT,
                  APXINF_NORM_GATE_NONE) ||
      is_contract(spec, APXINF_NORM_OPERATION_NORM,
                  APXINF_NORM_STATISTIC_RMS, APXINF_NORM_AFFINE_WEIGHT,
                  APXINF_NORM_GATE_NONE) ||
      is_contract(spec, APXINF_NORM_OPERATION_NORM,
                  APXINF_NORM_STATISTIC_LAYER, APXINF_NORM_AFFINE_WEIGHT,
                  APXINF_NORM_GATE_NONE) ||
      is_contract(spec, APXINF_NORM_OPERATION_NORM,
                  APXINF_NORM_STATISTIC_RMS, APXINF_NORM_AFFINE_ADAPTIVE,
                  APXINF_NORM_GATE_NONE) ||
      is_contract(spec, APXINF_NORM_OPERATION_NORM,
                  APXINF_NORM_STATISTIC_LAYER, APXINF_NORM_AFFINE_ADAPTIVE,
                  APXINF_NORM_GATE_NONE) ||
      is_contract(spec, APXINF_NORM_OPERATION_RESIDUAL_ADD,
                  APXINF_NORM_STATISTIC_NONE, APXINF_NORM_AFFINE_NONE,
                  APXINF_NORM_GATE_NONE) ||
      is_contract(spec, APXINF_NORM_OPERATION_RESIDUAL_ADD,
                  APXINF_NORM_STATISTIC_NONE, APXINF_NORM_AFFINE_NONE,
                  APXINF_NORM_GATE_STYLE) ||
      is_contract(spec, APXINF_NORM_OPERATION_FUSED_ADD_NORM,
                  APXINF_NORM_STATISTIC_RMS, APXINF_NORM_AFFINE_WEIGHT,
                  APXINF_NORM_GATE_NONE) ||
      is_contract(spec, APXINF_NORM_OPERATION_FUSED_ADD_NORM,
                  APXINF_NORM_STATISTIC_LAYER, APXINF_NORM_AFFINE_WEIGHT,
                  APXINF_NORM_GATE_NONE) ||
      is_contract(spec, APXINF_NORM_OPERATION_FUSED_ADD_NORM,
                  APXINF_NORM_STATISTIC_RMS, APXINF_NORM_AFFINE_ADAPTIVE,
                  APXINF_NORM_GATE_STYLE) ||
      is_contract(spec, APXINF_NORM_OPERATION_FUSED_ADD_NORM,
                  APXINF_NORM_STATISTIC_LAYER,
                  APXINF_NORM_AFFINE_ADAPTIVE, APXINF_NORM_GATE_NONE) ||
      is_contract(spec, APXINF_NORM_OPERATION_FUSED_ADD_NORM,
                  APXINF_NORM_STATISTIC_LAYER, APXINF_NORM_AFFINE_ADAPTIVE,
                  APXINF_NORM_GATE_STYLE);
}

void validate_spec(const apxinf_norm_spec_t& spec) {
  const bool grouped = spec.style_group_rows != 0;
  const bool grouped_norm = is_contract(spec, APXINF_NORM_OPERATION_NORM,
      APXINF_NORM_STATISTIC_LAYER, APXINF_NORM_AFFINE_ADAPTIVE, APXINF_NORM_GATE_NONE);
  const bool grouped_gate = is_contract(spec, APXINF_NORM_OPERATION_RESIDUAL_ADD,
      APXINF_NORM_STATISTIC_NONE, APXINF_NORM_AFFINE_NONE, APXINF_NORM_GATE_STYLE) &&
      spec.style_parts == 3 && spec.shift_first == 0;
  const bool grouped_fused = is_contract(spec, APXINF_NORM_OPERATION_FUSED_ADD_NORM,
      APXINF_NORM_STATISTIC_LAYER, APXINF_NORM_AFFINE_ADAPTIVE, APXINF_NORM_GATE_STYLE);
  if (grouped) {
    if (spec.style_group_rows < 1 || spec.rows <= 0 || spec.cols <= 0 ||
        spec.rows % spec.style_group_rows != 0 ||
        (spec.style_parts != 2 && spec.style_parts != 3) || spec.shift_first > 1 ||
        spec.numerics != APXINF_NORM_NUMERICS_STAGED_BF16 ||
        spec.dtype != APXINF_DTYPE_BF16 || spec.weight_offset != 0 ||
        spec.quantization != APXINF_NORM_QUANTIZATION_NONE ||
        spec.bias_order != APXINF_NORM_BIAS_ORDER_SINGLE_ROUND ||
        spec.has_bias != 0 || !(grouped_norm || grouped_gate || grouped_fused)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid grouped adaptive Norm contract");
    }
  } else if (grouped_fused || spec.style_parts != 0 || spec.shift_first != 0 ||
             spec.numerics == APXINF_NORM_NUMERICS_STAGED_BF16) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "staged BF16 requires grouped adaptive Norm");
  }
  if (spec.statistic == APXINF_NORM_STATISTIC_CHANNEL_L2 &&
      !(spec.dtype == APXINF_DTYPE_BF16 &&
        (spec.operation == APXINF_NORM_OPERATION_NORM ||
         spec.operation == APXINF_NORM_OPERATION_NORM_SILU || channel_update(spec)) &&
        spec.affine == APXINF_NORM_AFFINE_WEIGHT &&
        spec.gate == APXINF_NORM_GATE_NONE &&
        spec.quantization == APXINF_NORM_QUANTIZATION_NONE &&
        spec.numerics == APXINF_NORM_NUMERICS_FLEXIBLE &&
        spec.weight_offset == 0)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "ChannelL2 requires plain unquantized BF16 weighted normalization");
  }
  if (channel_update(spec) && spec.bias_order != APXINF_NORM_BIAS_ORDER_THEN_RESIDUAL)
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "channel update requires staged bias/residual rounding");
  const bool stable_decode_plain =
      is_contract(spec, APXINF_NORM_OPERATION_NORM,
                  APXINF_NORM_STATISTIC_RMS, APXINF_NORM_AFFINE_WEIGHT,
                  APXINF_NORM_GATE_NONE);
  const bool stable_decode_residual =
      is_contract(spec, APXINF_NORM_OPERATION_FUSED_ADD_NORM,
                  APXINF_NORM_STATISTIC_RMS, APXINF_NORM_AFFINE_WEIGHT,
                  APXINF_NORM_GATE_NONE) &&
      spec.dtype == APXINF_DTYPE_BF16 && spec.has_bias == 0;
  const bool stable_decode =
      spec.numerics == APXINF_NORM_NUMERICS_STABLE_DECODE;
  const bool stable_planning =
      spec.numerics == APXINF_NORM_NUMERICS_STABLE_PLANNING;
  const bool weight_offset_plain =
      spec.weight_offset != 0 && spec.dtype == APXINF_DTYPE_BF16 &&
      spec.quantization == APXINF_NORM_QUANTIZATION_NONE &&
      spec.has_bias == 0 &&
      spec.bias_order == APXINF_NORM_BIAS_ORDER_SINGLE_ROUND &&
      (spec.numerics == APXINF_NORM_NUMERICS_FLEXIBLE || stable_planning) &&
      is_contract(spec, APXINF_NORM_OPERATION_NORM,
                  APXINF_NORM_STATISTIC_RMS, APXINF_NORM_AFFINE_WEIGHT,
                  APXINF_NORM_GATE_NONE);
  const bool weight_offset_fused =
      spec.weight_offset != 0 && spec.dtype == APXINF_DTYPE_BF16 &&
      spec.quantization == APXINF_NORM_QUANTIZATION_NONE &&
      spec.has_bias == 0 &&
      spec.bias_order == APXINF_NORM_BIAS_ORDER_SINGLE_ROUND &&
      stable_planning &&
      is_contract(spec, APXINF_NORM_OPERATION_FUSED_ADD_NORM,
                  APXINF_NORM_STATISTIC_RMS, APXINF_NORM_AFFINE_WEIGHT,
                  APXINF_NORM_GATE_NONE);
  const bool static_bf16_contract =
      spec.dtype == APXINF_DTYPE_BF16 &&
      (is_contract(spec, APXINF_NORM_OPERATION_NORM,
                   APXINF_NORM_STATISTIC_RMS, APXINF_NORM_AFFINE_WEIGHT,
                   APXINF_NORM_GATE_NONE) ||
       is_contract(spec, APXINF_NORM_OPERATION_NORM,
                   APXINF_NORM_STATISTIC_LAYER,
                   APXINF_NORM_AFFINE_WEIGHT, APXINF_NORM_GATE_NONE) ||
       is_contract(spec, APXINF_NORM_OPERATION_FUSED_ADD_NORM,
                   APXINF_NORM_STATISTIC_LAYER,
                   APXINF_NORM_AFFINE_WEIGHT, APXINF_NORM_GATE_NONE) ||
       is_contract(spec, APXINF_NORM_OPERATION_NORM,
                   APXINF_NORM_STATISTIC_LAYER,
                   APXINF_NORM_AFFINE_ADAPTIVE, APXINF_NORM_GATE_NONE));
  const bool dual_output_contract =
      spec.dtype == APXINF_DTYPE_BF16 &&
      is_contract(spec, APXINF_NORM_OPERATION_NORM,
                  APXINF_NORM_STATISTIC_LAYER,
                  APXINF_NORM_AFFINE_ADAPTIVE, APXINF_NORM_GATE_NONE);
  const bool static_e4m3 =
      spec.quantization == APXINF_NORM_QUANTIZATION_STATIC_E4M3 &&
      (spec.dtype == APXINF_DTYPE_F16 || static_bf16_contract) &&
      spec.output_dtype == APXINF_DTYPE_E4M3 && writes_normalized(spec) &&
      spec.quantized_cols == spec.cols &&
      (spec.has_native_output == 0 || dual_output_contract);
  const bool rowwise_e4m3 =
      spec.quantization == APXINF_NORM_QUANTIZATION_ROWWISE_E4M3 &&
      spec.dtype == APXINF_DTYPE_BF16 &&
      spec.output_dtype == APXINF_DTYPE_E4M3 &&
      spec.has_native_output == 0 && spec.output_scale_is_unit != 0 &&
      spec.quantized_cols >= spec.cols &&
      (is_contract(spec, APXINF_NORM_OPERATION_NORM,
                   APXINF_NORM_STATISTIC_RMS,
                   APXINF_NORM_AFFINE_WEIGHT, APXINF_NORM_GATE_NONE) ||
       is_contract(spec, APXINF_NORM_OPERATION_FUSED_ADD_NORM,
                   APXINF_NORM_STATISTIC_RMS,
                   APXINF_NORM_AFFINE_WEIGHT, APXINF_NORM_GATE_NONE));
  const bool rowwise_i8 =
      spec.quantization == APXINF_NORM_QUANTIZATION_ROWWISE_I8 &&
      spec.dtype == APXINF_DTYPE_BF16 &&
      spec.output_dtype == APXINF_DTYPE_I8 &&
      spec.has_native_output != 0 && spec.output_scale_is_unit != 0 &&
      spec.quantized_cols == spec.cols &&
      (is_contract(spec, APXINF_NORM_OPERATION_NORM,
                   APXINF_NORM_STATISTIC_LAYER,
                   APXINF_NORM_AFFINE_WEIGHT, APXINF_NORM_GATE_NONE) ||
       is_contract(spec, APXINF_NORM_OPERATION_NORM,
                   APXINF_NORM_STATISTIC_LAYER,
                   APXINF_NORM_AFFINE_ADAPTIVE, APXINF_NORM_GATE_NONE));
  const bool unquantized =
      spec.quantization == APXINF_NORM_QUANTIZATION_NONE &&
      spec.output_dtype == spec.dtype && spec.quantized_cols == 0 &&
      spec.output_scale_is_unit != 0 &&
      spec.has_native_output == static_cast<uint32_t>(writes_normalized(spec));
  if (spec.version != APXINF_NORM_SPEC_VERSION ||
      !valid_contract(spec) ||
      spec.quantization > APXINF_NORM_QUANTIZATION_ROWWISE_I8 ||
      spec.numerics > APXINF_NORM_NUMERICS_STAGED_BF16 ||
      spec.weight_offset > 1 ||
      (spec.weight_offset != 0 &&
       !weight_offset_plain && !weight_offset_fused) ||
      (spec.weight_offset == 0 && stable_planning) ||
      (stable_decode &&
       (!stable_decode_plain && !stable_decode_residual ||
        spec.quantization != APXINF_NORM_QUANTIZATION_NONE)) ||
      (spec.dtype == APXINF_DTYPE_F32 &&
       (!stable_decode || !stable_decode_plain ||
        spec.quantization != APXINF_NORM_QUANTIZATION_NONE)) ||
      (spec.dtype != APXINF_DTYPE_F32 && spec.dtype != APXINF_DTYPE_F16 &&
       spec.dtype != APXINF_DTYPE_BF16) ||
      (!unquantized && !static_e4m3 && !rowwise_e4m3 && !rowwise_i8) ||
      spec.has_native_output > 1 ||
      spec.rows <= 0 || spec.cols <= 0 ||
      spec.rows > INT32_MAX || spec.cols > INT32_MAX ||
      spec.quantized_cols > INT32_MAX || spec.has_bias > 1 ||
      spec.bias_order > APXINF_NORM_BIAS_ORDER_THEN_RESIDUAL ||
      spec.output_scale_is_unit > 1 ||
      !valid_alignment(spec.input_alignment) ||
      !valid_alignment(spec.weight_alignment) ||
      !valid_alignment(spec.bias_alignment) ||
      !valid_alignment(spec.residual_alignment) ||
      !valid_alignment(spec.style_alignment) ||
      !valid_alignment(spec.hidden_alignment) ||
      !valid_alignment(spec.normalized_alignment) ||
      !valid_alignment(spec.quantized_alignment) ||
      !valid_alignment(spec.quantized_scales_alignment)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid Norm Spec");
  }
  if (spec.bias_order == APXINF_NORM_BIAS_ORDER_THEN_RESIDUAL &&
      spec.dtype != APXINF_DTYPE_BF16) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "BiasThenResidual order requires BF16 input and output");
  }
  if (spec.statistic == APXINF_NORM_STATISTIC_LAYER &&
      spec.affine == APXINF_NORM_AFFINE_ADAPTIVE &&
      spec.dtype != APXINF_DTYPE_BF16) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "adaptive LayerNorm requires BF16 input and output");
  }
  if (spec.bias_order == APXINF_NORM_BIAS_ORDER_THEN_RESIDUAL &&
      (spec.operation == APXINF_NORM_OPERATION_NORM ||
       spec.operation == APXINF_NORM_OPERATION_NORM_SILU ||
       spec.gate != APXINF_NORM_GATE_NONE ||
       spec.statistic == APXINF_NORM_STATISTIC_RMS)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "BiasThenResidual requires an ungated residual, LayerNorm, or channel update");
  }
  if (spec.has_bias != 0 && !may_have_bias(spec)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Norm contract does not take a bias");
  }
}

void validate_bindings(const apxinf_norm_spec_t& spec,
                       const apxinf_norm_bindings_t& bindings) {
  const bool quantized =
      spec.quantization != APXINF_NORM_QUANTIZATION_NONE;
  const bool rowwise =
      spec.quantization == APXINF_NORM_QUANTIZATION_ROWWISE_E4M3 ||
      spec.quantization == APXINF_NORM_QUANTIZATION_ROWWISE_I8;
  const bool missing =
      bindings.input == nullptr ||
      (writes_hidden(spec) && bindings.hidden == nullptr) ||
      ((spec.has_native_output != 0) != (bindings.normalized != nullptr)) ||
      (quantized != (bindings.quantized != nullptr)) ||
      (rowwise != (bindings.quantized_scales != nullptr)) ||
      (reads_residual(spec) && bindings.residual == nullptr) ||
      (reads_weight(spec) && bindings.weight == nullptr) ||
      (reads_norm_bias(spec) && bindings.norm_bias == nullptr) ||
      (reads_norm_style(spec) && bindings.norm_style == nullptr) ||
      (reads_gate_style(spec) && bindings.gate_style == nullptr);
  if (missing) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "missing Norm binding for contract");
  }
  if ((spec.has_bias != 0) != (bindings.bias != nullptr)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Norm bias binding disagrees with Spec.has_bias");
  }
  if (spec.numerics == APXINF_NORM_NUMERICS_STABLE_DECODE &&
      spec.operation == APXINF_NORM_OPERATION_FUSED_ADD_NORM &&
      bindings.hidden != bindings.residual) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "stable decode residual RMSNorm must update residual in place");
  }
  if (channel_update(spec)) {
    if ((bindings.residual != nullptr) != reads_residual(spec))
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "channel residual binding disagrees with operation");
    const size_t bytes = size_t(spec.rows) * size_t(spec.cols) * 2;
    const size_t vector_bytes = size_t(spec.cols) * 2;
    auto overlaps=[](const void* a,size_t an,const void* b,size_t bn){
      const uintptr_t x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
      return x<y ? y-x<an : x-y<bn;
    };
    if(overlaps(bindings.hidden,bytes,bindings.normalized,bytes))
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"channel outputs must be disjoint");
    for(const void* out:{bindings.hidden,bindings.normalized}) {
      for(const void* input:{bindings.input,bindings.residual})
        if(input && input!=out && overlaps(out,bytes,input,bytes))
          throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"channel output partially overlaps activation");
      for(const void* vector:{bindings.weight,bindings.bias})
        if(vector && overlaps(out,bytes,vector,vector_bytes))
          throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"channel output overlaps weight or bias");
    }
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
  auto* normalized = static_cast<__nv_fp8_e4m3*>(bindings.quantized);
  const int rows = static_cast<int>(spec.rows);
  const int cols = static_cast<int>(spec.cols);
  const float inverse_scale = 1.0f / bindings.output_scale;
  auto stream = static_cast<cudaStream_t>(bindings.stream);

  if (is_contract(spec, APXINF_NORM_OPERATION_NORM,
                  APXINF_NORM_STATISTIC_RMS, APXINF_NORM_AFFINE_WEIGHT,
                  APXINF_NORM_GATE_NONE)) {
    kernels::rms_norm_quant_f16_e4m3<<<rows, kThreads, 0, stream>>>(
        input, weight, normalized, rows, cols, bindings.eps, inverse_scale);
  } else if (is_contract(spec, APXINF_NORM_OPERATION_NORM,
                         APXINF_NORM_STATISTIC_LAYER,
                         APXINF_NORM_AFFINE_WEIGHT,
                         APXINF_NORM_GATE_NONE)) {
    kernels::layer_norm_quant_f16_e4m3<<<rows, kThreads, 0, stream>>>(
        input, weight, norm_bias, normalized, rows, cols, bindings.eps,
        inverse_scale);
  } else if (is_contract(spec, APXINF_NORM_OPERATION_NORM,
                         APXINF_NORM_STATISTIC_RMS,
                         APXINF_NORM_AFFINE_ADAPTIVE,
                         APXINF_NORM_GATE_NONE)) {
    kernels::ada_rms_norm_quant_f16_e4m3<<<rows, kThreads, 0, stream>>>(
        input, norm_style, normalized, rows, cols, bindings.eps,
        inverse_scale);
  } else if (is_contract(spec, APXINF_NORM_OPERATION_FUSED_ADD_NORM,
                         APXINF_NORM_STATISTIC_RMS,
                         APXINF_NORM_AFFINE_WEIGHT,
                         APXINF_NORM_GATE_NONE)) {
    kernels::bias_residual_rms_norm_quant_f16_e4m3
        <<<rows, kThreads, 0, stream>>>(
            input, bias, residual, weight, hidden, normalized, rows, cols,
            bindings.eps, inverse_scale);
  } else if (is_contract(spec, APXINF_NORM_OPERATION_FUSED_ADD_NORM,
                         APXINF_NORM_STATISTIC_LAYER,
                         APXINF_NORM_AFFINE_WEIGHT,
                         APXINF_NORM_GATE_NONE)) {
    kernels::bias_residual_layer_norm_quant_f16_e4m3
        <<<rows, kThreads, 0, stream>>>(
            input, bias, residual, weight, norm_bias, hidden, normalized, rows,
            cols, bindings.eps, inverse_scale);
  } else if (is_contract(spec, APXINF_NORM_OPERATION_FUSED_ADD_NORM,
                         APXINF_NORM_STATISTIC_RMS,
                         APXINF_NORM_AFFINE_ADAPTIVE,
                         APXINF_NORM_GATE_STYLE)) {
    // The 10x1024 specialization performs uint2 loads on norm_style and a
    // uint32 store on normalized; gate on the actual pointer alignment.
    if (rows == 10 && cols == 1024 &&
        reinterpret_cast<uintptr_t>(norm_style) % 8 == 0 &&
        reinterpret_cast<uintptr_t>(normalized) % 4 == 0) {
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
  } else {
    return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

cudaError_t launch_quantized_bf16(const apxinf_norm_spec_t& spec,
                                  const apxinf_norm_bindings_t& bindings) {
  auto stream = static_cast<cudaStream_t>(bindings.stream);
  const int rows = static_cast<int>(spec.rows);
  const int cols = static_cast<int>(spec.cols);
  const int quantized_cols = static_cast<int>(spec.quantized_cols);
  if (spec.quantization == APXINF_NORM_QUANTIZATION_ROWWISE_E4M3 &&
      is_contract(spec, APXINF_NORM_OPERATION_NORM,
                  APXINF_NORM_STATISTIC_RMS, APXINF_NORM_AFFINE_WEIGHT,
                  APXINF_NORM_GATE_NONE)) {
    return static_cast<cudaError_t>(
        apxinf::cuda_new::quant_ops::rms_norm_quantize_rows_bf16_e4m3(
            bindings.input, bindings.weight, bindings.quantized,
            bindings.quantized_scales, rows, cols, quantized_cols,
            bindings.eps, stream));
  }
  if (spec.quantization == APXINF_NORM_QUANTIZATION_ROWWISE_E4M3 &&
      is_contract(spec, APXINF_NORM_OPERATION_FUSED_ADD_NORM,
                  APXINF_NORM_STATISTIC_RMS, APXINF_NORM_AFFINE_WEIGHT,
                  APXINF_NORM_GATE_NONE)) {
    return static_cast<cudaError_t>(apxinf::cuda_new::quant_ops::
        bias_residual_rms_norm_quantize_rows_bf16_e4m3(
            bindings.input, bindings.bias, bindings.residual, bindings.weight,
            bindings.hidden, bindings.quantized, bindings.quantized_scales,
            rows, cols, quantized_cols, bindings.eps, stream));
  }
  if (spec.quantization == APXINF_NORM_QUANTIZATION_ROWWISE_I8 &&
      is_contract(spec, APXINF_NORM_OPERATION_NORM,
                  APXINF_NORM_STATISTIC_LAYER,
                  APXINF_NORM_AFFINE_ADAPTIVE, APXINF_NORM_GATE_NONE)) {
    return apxinf_norm_adaptive_layer_norm_quantize_rows_bf16_int8(
        bindings.input, bindings.norm_style, bindings.normalized,
        bindings.quantized, bindings.quantized_scales, rows, cols,
        bindings.eps, stream);
  }
  if (spec.quantization == APXINF_NORM_QUANTIZATION_ROWWISE_I8 &&
      is_contract(spec, APXINF_NORM_OPERATION_NORM,
                  APXINF_NORM_STATISTIC_LAYER,
                  APXINF_NORM_AFFINE_WEIGHT, APXINF_NORM_GATE_NONE)) {
    return apxinf_norm_layer_norm_quantize_rows_bf16_int8(
        bindings.input, bindings.weight, bindings.norm_bias,
        bindings.normalized, bindings.quantized,
        bindings.quantized_scales, rows, cols, bindings.eps, stream);
  }
  if (spec.quantization != APXINF_NORM_QUANTIZATION_STATIC_E4M3) {
    return cudaErrorInvalidValue;
  }
  if (is_contract(spec, APXINF_NORM_OPERATION_NORM,
                  APXINF_NORM_STATISTIC_RMS, APXINF_NORM_AFFINE_WEIGHT,
                  APXINF_NORM_GATE_NONE)) {
    return apxinf_norm_rms_norm_quant_bf16_e4m3(
        bindings.input, bindings.weight, bindings.quantized, rows, cols,
        bindings.eps, bindings.output_scale, stream);
  }
  if (is_contract(spec, APXINF_NORM_OPERATION_NORM,
                  APXINF_NORM_STATISTIC_LAYER,
                  APXINF_NORM_AFFINE_WEIGHT, APXINF_NORM_GATE_NONE)) {
    return apxinf_norm_layer_norm_quant_bf16_e4m3(
        bindings.input, bindings.weight, bindings.norm_bias,
        bindings.quantized, rows, cols, bindings.eps,
        bindings.output_scale, stream);
  }
  if (is_contract(spec, APXINF_NORM_OPERATION_FUSED_ADD_NORM,
                  APXINF_NORM_STATISTIC_LAYER,
                  APXINF_NORM_AFFINE_WEIGHT, APXINF_NORM_GATE_NONE)) {
    return apxinf_norm_bias_residual_layer_norm_quant_bf16_e4m3(
        bindings.input, bindings.bias, bindings.residual, bindings.weight,
        bindings.norm_bias, bindings.hidden, bindings.quantized, rows, cols,
        bindings.eps, bindings.output_scale, stream);
  }
  if (is_contract(spec, APXINF_NORM_OPERATION_NORM,
                  APXINF_NORM_STATISTIC_LAYER,
                  APXINF_NORM_AFFINE_ADAPTIVE, APXINF_NORM_GATE_NONE)) {
    return apxinf_norm_adaptive_layer_norm_quant_bf16_e4m3(
        bindings.input, bindings.norm_style, bindings.normalized,
        bindings.quantized, static_cast<uint32_t>(rows),
        static_cast<uint32_t>(cols), bindings.eps, bindings.output_scale,
        stream);
  }
  return cudaErrorInvalidValue;
}

template <class T, bool Silu>
void launch_channel_l2(const T* input, const T* weight, T* normalized,
                       int rows, int cols, float eps, cudaStream_t stream) {
    if constexpr (Silu && std::is_same_v<T, __nv_bfloat16>) {
      if (rows >= 1024 && (cols == 96 || cols == 192) &&
          reinterpret_cast<uintptr_t>(input) % 16 == 0 &&
          reinterpret_cast<uintptr_t>(weight) % 16 == 0 &&
          reinterpret_cast<uintptr_t>(normalized) % 16 == 0) {
        if (cols == 96) {
          kernels::channel_l2_silu_packed_bf16<96><<<(rows - 1) / 64 + 1, 256, 0, stream>>>(
              input, weight, normalized, rows, eps);
        } else {
          kernels::channel_l2_silu_packed_bf16<192><<<(rows - 1) / 64 + 1, 256, 0, stream>>>(
              input, weight, normalized, rows, eps);
        }
        return;
      }
    }
    const int blocks = (rows - 1) / 8 + 1;
    if (rows >= 1024 && cols == 96) {
      kernels::channel_l2_norm_warp<T, 96, Silu><<<blocks, 256, 0, stream>>>(
          input, weight, normalized, rows, eps);
    } else if (rows >= 1024 && cols == 192) {
      kernels::channel_l2_norm_warp<T, 192, Silu><<<blocks, 256, 0, stream>>>(
          input, weight, normalized, rows, eps);
    } else if (rows >= 1024 && cols == 384) {
      kernels::channel_l2_norm_warp<T, 384, Silu><<<blocks, 256, 0, stream>>>(
          input, weight, normalized, rows, eps);
    } else {
      kernels::channel_l2_norm<T, Silu><<<rows, kThreads, 0, stream>>>(
          input, weight, normalized, rows, cols, eps);
    }
}

template <class T>
cudaError_t launch(const apxinf_norm_spec_t& spec,
                   const apxinf_norm_bindings_t& bindings);

cudaError_t launch_fixed_f16(const apxinf_norm_spec_t& spec,
                             const apxinf_norm_bindings_t& bindings) {
  auto stream = static_cast<cudaStream_t>(bindings.stream);
  const int rows = static_cast<int>(spec.rows);
  const int cols = static_cast<int>(spec.cols);
  if (is_contract(spec, APXINF_NORM_OPERATION_NORM,
                  APXINF_NORM_STATISTIC_RMS,
                  APXINF_NORM_AFFINE_WEIGHT, APXINF_NORM_GATE_NONE)) {
    return apxinf_norm_rms_norm_f16(
        bindings.input, bindings.weight, bindings.normalized, rows, cols,
        bindings.eps, stream);
  }
  if (is_contract(spec, APXINF_NORM_OPERATION_NORM,
                  APXINF_NORM_STATISTIC_LAYER,
                  APXINF_NORM_AFFINE_WEIGHT, APXINF_NORM_GATE_NONE)) {
    return apxinf_norm_layer_norm_f16(
        bindings.input, bindings.weight, bindings.norm_bias,
        bindings.normalized, rows, cols, bindings.eps, stream);
  }
  return launch<__half>(spec, bindings);
}

cudaError_t launch_stable_decode(const apxinf_norm_spec_t& spec,
                                 const apxinf_norm_bindings_t& bindings) {
  auto stream = static_cast<cudaStream_t>(bindings.stream);
  const auto rows = static_cast<uint32_t>(spec.rows);
  const auto cols = static_cast<uint32_t>(spec.cols);
  if (is_contract(spec, APXINF_NORM_OPERATION_NORM,
                  APXINF_NORM_STATISTIC_RMS, APXINF_NORM_AFFINE_WEIGHT,
                  APXINF_NORM_GATE_NONE)) {
    if (spec.dtype == APXINF_DTYPE_F32) {
      return apxinf_norm_fixed_decode_rms_norm_f32(
          bindings.input, bindings.weight, bindings.normalized, cols, rows,
          bindings.eps, stream);
    }
    if (spec.dtype == APXINF_DTYPE_BF16) {
      return apxinf_norm_fixed_decode_rms_norm_bf16(
          bindings.input, bindings.weight, bindings.normalized, cols, rows,
          bindings.eps, stream);
    }
  }
  if (is_contract(spec, APXINF_NORM_OPERATION_FUSED_ADD_NORM,
                  APXINF_NORM_STATISTIC_RMS, APXINF_NORM_AFFINE_WEIGHT,
                  APXINF_NORM_GATE_NONE) &&
      spec.dtype == APXINF_DTYPE_BF16 && bindings.bias == nullptr &&
      bindings.hidden == bindings.residual) {
    return apxinf_norm_fixed_decode_rms_norm_add_bf16(
        bindings.hidden, bindings.input, bindings.weight,
        bindings.normalized, cols, rows, bindings.eps, stream);
  }
  return cudaErrorInvalidValue;
}

cudaError_t launch_weight_offset(const apxinf_norm_spec_t& spec,
                                 const apxinf_norm_bindings_t& bindings) {
  auto stream = static_cast<cudaStream_t>(bindings.stream);
  const int rows = static_cast<int>(spec.rows);
  const int cols = static_cast<int>(spec.cols);
  if (spec.operation == APXINF_NORM_OPERATION_NORM) {
    if (spec.numerics == APXINF_NORM_NUMERICS_STABLE_PLANNING) {
      return apxinf_norm_rms_norm_plus1_bf16(
          bindings.input, bindings.weight, bindings.normalized, rows, cols,
          bindings.eps, stream);
    }
    const int status = bindings.input == bindings.normalized
                           ? apxinf::cuda_new::norm::head_rms_norm(
                                 bindings.normalized, bindings.weight, rows,
                                 cols, bindings.eps, stream)
                           : apxinf::cuda_new::norm::rms_norm_weight_offset(
                                 bindings.input, bindings.weight,
                                 bindings.normalized, rows, cols, bindings.eps,
                                 stream);
    return status == 0 ? cudaSuccess : cudaErrorUnknown;
  }
  if (spec.operation == APXINF_NORM_OPERATION_FUSED_ADD_NORM &&
      spec.numerics == APXINF_NORM_NUMERICS_STABLE_PLANNING) {
    return apxinf_norm_add_rms_norm_plus1_bf16(
        bindings.input, bindings.residual, bindings.weight, bindings.hidden,
        bindings.normalized, rows, cols, bindings.eps, stream);
  }
  return cudaErrorInvalidValue;
}

cudaError_t launch_fixed_bf16(const apxinf_runtime& runtime,
                              const apxinf_norm_spec_t& spec,
                              const apxinf_norm_bindings_t& bindings) {
  auto stream = static_cast<cudaStream_t>(bindings.stream);
  const int rows = static_cast<int>(spec.rows);
  const int cols = static_cast<int>(spec.cols);
  const int sm = runtime.compute_major * 10 + runtime.compute_minor;
  if (spec.style_group_rows != 0 && spec.operation == APXINF_NORM_OPERATION_FUSED_ADD_NORM) {
    if (cols == 2048 && spec.input_alignment >= 16 && spec.residual_alignment >= 16 &&
        spec.style_alignment >= 16 && spec.hidden_alignment >= 16 && spec.normalized_alignment >= 16) {
      kernels::grouped_gate_residual_layer_norm_warp_bf16<<<(uint64_t(rows) + 3) / 4, 128, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(bindings.input), static_cast<const __nv_bfloat16*>(bindings.residual),
          static_cast<const __nv_bfloat16*>(bindings.gate_style), static_cast<__nv_bfloat16*>(bindings.hidden),
          static_cast<const __nv_bfloat16*>(bindings.norm_style), static_cast<__nv_bfloat16*>(bindings.normalized),
          static_cast<int>(spec.style_group_rows), spec.style_parts, spec.shift_first != 0, bindings.eps, rows);
      return cudaGetLastError();
    }
    if (cols == 2048 && spec.input_alignment >= 4 && spec.residual_alignment >= 4 &&
        spec.style_alignment >= 4 && spec.hidden_alignment >= 4 && spec.normalized_alignment >= 4) {
      kernels::grouped_gate_residual_layer_norm_packed_bf16<<<rows, 128, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(bindings.input), static_cast<const __nv_bfloat16*>(bindings.residual),
          static_cast<const __nv_bfloat16*>(bindings.gate_style), static_cast<__nv_bfloat16*>(bindings.hidden),
          static_cast<const __nv_bfloat16*>(bindings.norm_style), static_cast<__nv_bfloat16*>(bindings.normalized),
          static_cast<int>(spec.style_group_rows), spec.style_parts, spec.shift_first != 0, bindings.eps);
      return cudaGetLastError();
    }
    // Generic fallback retains the same staged operators and caller-owned hidden storage.
    auto gate_spec = spec;
    gate_spec.operation = APXINF_NORM_OPERATION_RESIDUAL_ADD;
    const auto status = launch_fixed_bf16(runtime, gate_spec, bindings);
    if (status != cudaSuccess) return status;
    auto norm_spec = spec;
    norm_spec.operation = APXINF_NORM_OPERATION_NORM;
    auto norm_bindings = bindings;
    norm_bindings.input = bindings.hidden;
    return launch_fixed_bf16(runtime, norm_spec, norm_bindings);
  }
  if (spec.style_group_rows != 0 && spec.operation == APXINF_NORM_OPERATION_RESIDUAL_ADD) {
    const int64_t count = spec.rows * spec.cols;
    if (cols % 8 == 0 && count / 8 <= UINT32_MAX && spec.input_alignment >= 16 && spec.residual_alignment >= 16 &&
        spec.style_alignment >= 16 && spec.hidden_alignment >= 16) {
      kernels::grouped_gate_residual_bf16_vec8<<<elementwise_blocks(count / 8), kThreads, 0, stream>>>(
          static_cast<const uint4*>(bindings.input), static_cast<const uint4*>(bindings.residual),
          static_cast<const uint4*>(bindings.gate_style), static_cast<uint4*>(bindings.hidden),
          static_cast<uint32_t>(count / 8), cols / 8, static_cast<uint32_t>(spec.style_group_rows));
    } else {
      kernels::grouped_gate_residual_bf16<<<elementwise_blocks(count), kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(bindings.input),
          static_cast<const __nv_bfloat16*>(bindings.residual),
          static_cast<const __nv_bfloat16*>(bindings.gate_style),
          static_cast<__nv_bfloat16*>(bindings.hidden), count, cols,
          static_cast<int>(spec.style_group_rows));
    }
    return cudaGetLastError();
  }
  if (spec.style_group_rows != 0) {
    if (cols == 2048) {
    kernels::grouped_adaptive_layer_norm_cached_bf16<<<rows, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(bindings.input),
        static_cast<const __nv_bfloat16*>(bindings.norm_style),
        static_cast<__nv_bfloat16*>(bindings.normalized), rows, cols,
        static_cast<int>(spec.style_group_rows), spec.style_parts,
        spec.shift_first != 0, bindings.eps);
    } else {
    kernels::grouped_adaptive_layer_norm_bf16<<<rows, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(bindings.input),
        static_cast<const __nv_bfloat16*>(bindings.norm_style),
        static_cast<__nv_bfloat16*>(bindings.normalized), rows, cols,
        static_cast<int>(spec.style_group_rows), spec.style_parts,
        spec.shift_first != 0, bindings.eps);
    }
    return cudaGetLastError();
  }
  const bool bias_then_residual =
      spec.bias_order == APXINF_NORM_BIAS_ORDER_THEN_RESIDUAL;
  const bool packed4 = (cols & 3) == 0 && spec.input_alignment >= 8 &&
      spec.bias_alignment >= 8 && spec.residual_alignment >= 8 &&
      spec.hidden_alignment >= 8;

  if (is_contract(spec, APXINF_NORM_OPERATION_NORM,
                  APXINF_NORM_STATISTIC_LAYER,
                  APXINF_NORM_AFFINE_ADAPTIVE,
                  APXINF_NORM_GATE_NONE)) {
    return apxinf_norm_adaptive_layer_norm_bf16(
        bindings.input, bindings.norm_style, bindings.normalized,
        static_cast<uint32_t>(rows), static_cast<uint32_t>(cols),
        bindings.eps, stream);
  }

  if (is_contract(spec, APXINF_NORM_OPERATION_RESIDUAL_ADD,
                  APXINF_NORM_STATISTIC_NONE, APXINF_NORM_AFFINE_NONE,
                  APXINF_NORM_GATE_NONE)) {
    if (bias_then_residual && packed4) {
      return apxinf_norm_bias_then_residual_bf16_packed4(
          bindings.input, bindings.bias, bindings.residual, bindings.hidden,
          rows, cols, stream);
    }
    if (bias_then_residual) {
      return apxinf_norm_bias_then_residual_bf16(
          bindings.input, bindings.bias, bindings.residual, bindings.hidden,
          rows, cols, stream);
    }
    if (bindings.bias != nullptr && packed4) {
      return apxinf_norm_bias_residual_bf16_packed4(
          bindings.input, bindings.bias, bindings.residual, bindings.hidden,
          rows, cols, stream);
    }
    return launch<__nv_bfloat16>(spec, bindings);
  }

  if (is_contract(spec, APXINF_NORM_OPERATION_FUSED_ADD_NORM,
                  APXINF_NORM_STATISTIC_LAYER,
                  APXINF_NORM_AFFINE_WEIGHT, APXINF_NORM_GATE_NONE)) {
    if (!bias_then_residual && sm == 87 && rows == 512 && cols == 1024) {
      return apxinf_norm_bias_residual_layer_norm_bf16_cached_1024(
          bindings.input, bindings.bias, bindings.residual, bindings.weight,
          bindings.norm_bias, bindings.hidden, bindings.normalized, rows, cols,
          bindings.eps, stream);
    }
    if (bias_then_residual && (sm == 87 || sm == 110) && rows == 41 &&
        cols == 1536 && bindings.bias != nullptr) {
      return apxinf_norm_bias_then_residual_layer_norm_bf16_cached_1536(
          bindings.input, bindings.bias, bindings.residual, bindings.weight,
          bindings.norm_bias, bindings.hidden, bindings.normalized, rows, cols,
          bindings.eps, stream);
    }
    if (!bias_then_residual) {
      return launch<__nv_bfloat16>(spec, bindings);
    }
    cudaError_t error = apxinf_norm_bias_then_residual_bf16(
        bindings.input, bindings.bias, bindings.residual, bindings.hidden,
        rows, cols, stream);
    if (error != cudaSuccess) return error;
    kernels::layer_norm<__nv_bfloat16><<<rows, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(bindings.hidden),
        static_cast<const __nv_bfloat16*>(bindings.weight),
        static_cast<const __nv_bfloat16*>(bindings.norm_bias),
        static_cast<__nv_bfloat16*>(bindings.normalized), rows, cols,
        bindings.eps);
    return cudaGetLastError();
  }

  if (is_contract(spec, APXINF_NORM_OPERATION_FUSED_ADD_NORM,
                  APXINF_NORM_STATISTIC_LAYER,
                  APXINF_NORM_AFFINE_ADAPTIVE,
                  APXINF_NORM_GATE_NONE)) {
    if (bias_then_residual && (sm == 87 || sm == 110) && rows == 41 &&
        cols == 1536 && bindings.bias != nullptr) {
      return apxinf_norm_bias_then_residual_adaptive_layer_norm_bf16_cached_1536(
          bindings.input, bindings.bias, bindings.residual,
          bindings.norm_style, bindings.hidden, bindings.normalized, rows,
          cols, bindings.eps, stream);
    }
    cudaError_t error;
    if (bias_then_residual) {
      error = apxinf_norm_bias_then_residual_bf16(
          bindings.input, bindings.bias, bindings.residual, bindings.hidden,
          rows, cols, stream);
    } else {
      const int64_t count = spec.rows * spec.cols;
      kernels::bias_residual<__nv_bfloat16>
          <<<elementwise_blocks(count), kThreads, 0, stream>>>(
              static_cast<const __nv_bfloat16*>(bindings.input),
              static_cast<const __nv_bfloat16*>(bindings.bias),
              static_cast<const __nv_bfloat16*>(bindings.residual),
              static_cast<__nv_bfloat16*>(bindings.hidden), count, cols);
      error = cudaGetLastError();
    }
    if (error != cudaSuccess) return error;
    return apxinf_norm_adaptive_layer_norm_bf16(
        bindings.hidden, bindings.norm_style, bindings.normalized,
        static_cast<uint32_t>(rows), static_cast<uint32_t>(cols),
        bindings.eps, stream);
  }
  return launch<__nv_bfloat16>(spec, bindings);
}

cudaError_t launch_channel_update(const apxinf_norm_spec_t& spec,
                                  const apxinf_norm_bindings_t& bindings) {
  using B=__nv_bfloat16;
  const auto* input=static_cast<const B*>(bindings.input);
  const auto* bias=static_cast<const B*>(bindings.bias);
  const auto* residual=static_cast<const B*>(bindings.residual);
  const auto* weight=static_cast<const B*>(bindings.weight);
  auto* hidden=static_cast<B*>(bindings.hidden);
  auto* normalized=static_cast<B*>(bindings.normalized);
  const int rows=int(spec.rows),cols=int(spec.cols);
  const int64_t count=spec.rows*spec.cols;
  auto stream=static_cast<cudaStream_t>(bindings.stream);
  auto aligned=[](const void* p,uintptr_t a){return !p || reinterpret_cast<uintptr_t>(p)%a==0;};
  bool packed=true;
  for(const void* p:{bindings.input,bindings.bias,bindings.residual,bindings.weight,
                    static_cast<const void*>(bindings.hidden),static_cast<const void*>(bindings.normalized)})
    packed=packed&&aligned(p,16);
  if(rows>=1024 && packed && (cols==96 || cols==192)) {
    if(cols==96) {
      if(residual)kernels::channel_l2_silu_update_packed_bf16<96,true><<<(rows-1)/64+1,256,0,stream>>>(input,bias,residual,hidden,weight,normalized,rows,bindings.eps);
      else kernels::channel_l2_silu_update_packed_bf16<96,false><<<(rows-1)/64+1,256,0,stream>>>(input,bias,residual,hidden,weight,normalized,rows,bindings.eps);
    } else {
      if(residual)kernels::channel_l2_silu_update_packed_bf16<192,true><<<(rows-1)/64+1,256,0,stream>>>(input,bias,residual,hidden,weight,normalized,rows,bindings.eps);
      else kernels::channel_l2_silu_update_packed_bf16<192,false><<<(rows-1)/64+1,256,0,stream>>>(input,bias,residual,hidden,weight,normalized,rows,bindings.eps);
    }
    return cudaGetLastError();
  }
  cudaError_t error;
  if(residual) {
    if(cols%4==0 && aligned(input,8) && aligned(bias,8) && aligned(residual,8) && aligned(hidden,8))
      error=apxinf_norm_bias_then_residual_bf16_packed4(input,bias,residual,hidden,rows,cols,stream);
    else error=apxinf_norm_bias_then_residual_bf16(input,bias,residual,hidden,rows,cols,stream);
  } else {
    if(cols%8==0 && aligned(input,16) && aligned(bias,16) && aligned(hidden,16))
      kernels::channel_bias_packed_bf16<<<elementwise_blocks(count/8),256,0,stream>>>(
        reinterpret_cast<const kernels::ChannelL2Pack8*>(input),reinterpret_cast<const kernels::ChannelL2Pack8*>(bias),
        reinterpret_cast<kernels::ChannelL2Pack8*>(hidden),count/8,cols/8);
    else kernels::channel_bias_bf16<<<elementwise_blocks(count),256,0,stream>>>(input,bias,hidden,count,cols);
    error=cudaGetLastError();
  }
  if(error!=cudaSuccess)return error;
  launch_channel_l2<B,true>(hidden,weight,normalized,rows,cols,bindings.eps,stream);
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

  if (spec.statistic == APXINF_NORM_STATISTIC_CHANNEL_L2) {
    if (channel_update(spec)) return launch_channel_update(spec, bindings);
    if (spec.operation == APXINF_NORM_OPERATION_NORM_SILU)
      launch_channel_l2<T, true>(input, weight, normalized, rows, cols, bindings.eps, stream);
    else
      launch_channel_l2<T, false>(input, weight, normalized, rows, cols, bindings.eps, stream);
  } else if (is_contract(spec, APXINF_NORM_OPERATION_NORM,
                  APXINF_NORM_STATISTIC_RMS,
                  APXINF_NORM_AFFINE_WEIGHT, APXINF_NORM_GATE_NONE)) {
    if (cols == 128) {
      if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        if (reinterpret_cast<uintptr_t>(input) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(weight) % 16 == 0 &&
            reinterpret_cast<uintptr_t>(normalized) % 16 == 0) {
          kernels::rms_norm_128_packed<8><<<(rows - 1) / 64 + 1, 256, 0, stream>>>(
              input, weight, normalized, rows, bindings.eps);
          return cudaGetLastError();
        }
      }
      kernels::rms_norm_128<T><<<(rows - 1) / 8 + 1, 256, 0, stream>>>(
          input, weight, normalized, rows, bindings.eps);
    } else {
      kernels::rms_norm<T><<<rows, kThreads, 0, stream>>>(
          input, weight, normalized, rows, cols, bindings.eps);
    }
  } else if (is_contract(spec, APXINF_NORM_OPERATION_NORM,
                         APXINF_NORM_STATISTIC_LAYER,
                         APXINF_NORM_AFFINE_WEIGHT,
                         APXINF_NORM_GATE_NONE)) {
    kernels::layer_norm<T><<<rows, kThreads, 0, stream>>>(
        input, weight, norm_bias, normalized, rows, cols, bindings.eps);
  } else if (is_contract(spec, APXINF_NORM_OPERATION_NORM,
                         APXINF_NORM_STATISTIC_RMS,
                         APXINF_NORM_AFFINE_ADAPTIVE,
                         APXINF_NORM_GATE_NONE)) {
    kernels::ada_rms_norm<T><<<rows, kThreads, 0, stream>>>(
        input, norm_style, normalized, rows, cols, bindings.eps);
  } else if (is_contract(spec, APXINF_NORM_OPERATION_RESIDUAL_ADD,
                         APXINF_NORM_STATISTIC_NONE,
                         APXINF_NORM_AFFINE_NONE,
                         APXINF_NORM_GATE_NONE)) {
    kernels::bias_residual<T>
        <<<elementwise_blocks(count), kThreads, 0, stream>>>(
            input, bias, residual, hidden, count, cols);
  } else if (is_contract(spec, APXINF_NORM_OPERATION_FUSED_ADD_NORM,
                         APXINF_NORM_STATISTIC_RMS,
                         APXINF_NORM_AFFINE_WEIGHT,
                         APXINF_NORM_GATE_NONE)) {
    kernels::bias_residual_rms_norm<T><<<rows, kThreads, 0, stream>>>(
        input, bias, residual, weight, hidden, normalized, rows, cols,
        bindings.eps);
  } else if (is_contract(spec, APXINF_NORM_OPERATION_FUSED_ADD_NORM,
                         APXINF_NORM_STATISTIC_LAYER,
                         APXINF_NORM_AFFINE_WEIGHT,
                         APXINF_NORM_GATE_NONE)) {
    kernels::bias_residual_layer_norm<T><<<rows, kThreads, 0, stream>>>(
        input, bias, residual, weight, norm_bias, hidden, normalized, rows,
        cols, bindings.eps);
  } else if (is_contract(spec, APXINF_NORM_OPERATION_RESIDUAL_ADD,
                         APXINF_NORM_STATISTIC_NONE,
                         APXINF_NORM_AFFINE_NONE,
                         APXINF_NORM_GATE_STYLE)) {
    kernels::ada_gate_residual<T>
        <<<elementwise_blocks(count), kThreads, 0, stream>>>(
            input, residual, gate_style, hidden, count, cols);
  } else if (is_contract(spec, APXINF_NORM_OPERATION_FUSED_ADD_NORM,
                         APXINF_NORM_STATISTIC_RMS,
                         APXINF_NORM_AFFINE_ADAPTIVE,
                         APXINF_NORM_GATE_STYLE)) {
    kernels::ada_gate_residual_rms_norm<T><<<rows, kThreads, 0, stream>>>(
        input, residual, gate_style, norm_style, hidden, normalized, rows,
        cols, bindings.eps);
  } else {
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
        spec->weight_offset != 0
            ? launch_weight_offset(*spec, *bindings)
            : spec->quantization != APXINF_NORM_QUANTIZATION_NONE
            ? (spec->dtype == APXINF_DTYPE_BF16
                   ? launch_quantized_bf16(*spec, *bindings)
                   : launch_quantized_f16(*spec, *bindings))
            : (spec->numerics == APXINF_NORM_NUMERICS_STABLE_DECODE
                   ? launch_stable_decode(*spec, *bindings)
                   : (spec->dtype == APXINF_DTYPE_BF16
                          ? launch_fixed_bf16(*runtime, *spec, *bindings)
                          : launch_fixed_f16(*spec, *bindings))));
  });
}
