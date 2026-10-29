#pragma once

#include <stdint.h>

/* Shares the dtype enum and status codes with the gemm family. */
#include "gemm_types.h"

/* Normalization family.
 *
 * Every call is described by independent operation, statistic, affine, gate,
 * and quantization fields.  This avoids multiplying one L3 mathematical API
 * into an enum value for every supported combination.
 *
 * Which bindings each operation reads is derived from those fields:
 *
 *   RMS                    input, weight                     -> normalized
 *   LAYER                  input, weight, norm_bias          -> normalized
 *   ADAPTIVE_RMS           input, norm_style                 -> normalized
 *   BIAS_RESIDUAL          input, bias, residual             -> hidden
 *   BIAS_RESIDUAL_RMS      input, bias, residual, weight     -> hidden, normalized
 *   BIAS_RESIDUAL_LAYER    input, bias, residual, weight,
 *                          norm_bias                         -> hidden, normalized
 *   ADA_GATE_RESIDUAL      input, residual, gate_style       -> hidden
 *   ADA_GATE_RESIDUAL_RMS  input, residual, gate_style,
 *                          norm_style                        -> hidden, normalized
 *   BIAS_RESIDUAL_ADAPTIVE_LAYER
 *                          input, bias, residual, norm_style -> hidden, normalized
 *
 * The bias_order field selects whether input+bias is rounded to BF16 before
 * adding residual. The two-step order is BF16-only.
 *
 * The two adaptive-conditioning vectors are distinct and are not
 * interchangeable:
 *
 *   norm_style  [2 * cols]  scale in [0, cols), shift in [cols, 2 * cols)
 *   gate_style  [3 * cols]  the gate is the third segment,
 *                           [2 * cols, 3 * cols)
 *
 * ADA_GATE_RESIDUAL_RMS reads the gate from the *current* layer's gate_style
 * and normalizes with the *next* layer's norm_style: the pi05 action expert
 * folds the next layer's normalization into the current layer's residual.
 */
typedef enum {
  APXINF_NORM_OPERATION_NORM = 0,
  APXINF_NORM_OPERATION_FUSED_ADD_NORM = 1,
  APXINF_NORM_OPERATION_RESIDUAL_ADD = 2,
} apxinf_norm_operation_t;

typedef enum {
  APXINF_NORM_STATISTIC_NONE = 0,
  APXINF_NORM_STATISTIC_RMS = 1,
  APXINF_NORM_STATISTIC_LAYER = 2,
} apxinf_norm_statistic_t;

typedef enum {
  APXINF_NORM_AFFINE_NONE = 0,
  APXINF_NORM_AFFINE_WEIGHT = 1,
  APXINF_NORM_AFFINE_ADAPTIVE = 2,
} apxinf_norm_affine_t;

typedef enum {
  APXINF_NORM_GATE_NONE = 0,
  APXINF_NORM_GATE_STYLE = 1,
} apxinf_norm_gate_t;

typedef enum {
  APXINF_NORM_QUANTIZATION_NONE = 0,
  APXINF_NORM_QUANTIZATION_STATIC_E4M3 = 1,
  APXINF_NORM_QUANTIZATION_ROWWISE_E4M3 = 2,
  APXINF_NORM_QUANTIZATION_ROWWISE_I8 = 3,
} apxinf_norm_quantization_t;

typedef enum {
  APXINF_NORM_NUMERICS_FLEXIBLE = 0,
  APXINF_NORM_NUMERICS_STABLE_DECODE = 1,
  APXINF_NORM_NUMERICS_STABLE_PLANNING = 2,
} apxinf_norm_numerics_t;

typedef enum {
  APXINF_NORM_BIAS_ORDER_SINGLE_ROUND = 0,
  APXINF_NORM_BIAS_ORDER_THEN_RESIDUAL = 1,
} apxinf_norm_bias_order_t;

#define APXINF_NORM_SPEC_VERSION 7u

typedef struct {
  uint32_t version;
  uint32_t operation;
  uint32_t statistic;
  uint32_t affine;
  uint32_t gate;
  uint32_t quantization;
  uint32_t numerics;
  uint32_t dtype;
  /* E4M3 output is the fused-quantization form and consumes output_scale.
     BF16/F16 output leaves the scale at one. */
  uint32_t output_dtype;
  /* Whether the native-dtype normalized result is retained in addition to a
     quantized epilogue. Always one for an unquantized operation. */
  uint32_t has_native_output;
  /* A null bias is a different kernel path, so it belongs to the identity. */
  uint32_t has_bias;
  uint32_t bias_order;
  /* The checkpoint stores a zero-centered affine multiplier and the effective
     RMSNorm weight is `1 + weight`. */
  uint32_t weight_offset;
  /* Largest guaranteed power-of-two byte alignment, capped at 256. */
  uint32_t input_alignment;
  uint32_t weight_alignment;
  uint32_t bias_alignment;
  uint32_t residual_alignment;
  uint32_t style_alignment;
  uint32_t hidden_alignment;
  uint32_t normalized_alignment;
  uint32_t quantized_alignment;
  uint32_t quantized_scales_alignment;
  int64_t rows;
  int64_t cols;
  /* Quantized row width. Zero when quantization is disabled. */
  int64_t quantized_cols;
  /* Structural scale predicate, not the scale value.  Keeping the value in the
     bindings is what lets one tuned Recipe serve every layer that differs only
     by its calibration scale. */
  uint32_t output_scale_is_unit;
} apxinf_norm_spec_t;

typedef struct {
  const void* input;
  const void* bias;
  const void* residual;
  const void* weight;
  const void* norm_bias;
  /* [2 * cols] adaptive scale and shift. */
  const void* norm_style;
  /* [3 * cols]; only the third segment, the gate, is read. */
  const void* gate_style;
  /* Pre-normalization result.  Null for the pure normalization operation. */
  void* hidden;
  /* Normalized result.  Null for the residual-add operation. */
  void* normalized;
  /* Optional quantization epilogue output. */
  void* quantized;
  /* Optional F32 `[rows]` dynamic quantization scales. */
  void* quantized_scales;
  apxinf_cuda_stream_t stream;
  /* Numeric per-call data: changes the result, never the fastest candidate. */
  float eps;
  float output_scale;
} apxinf_norm_bindings_t;
