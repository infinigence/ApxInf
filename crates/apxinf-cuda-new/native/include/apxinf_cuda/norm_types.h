#pragma once

#include <stdint.h>

/* Shares the dtype enum and status codes with the gemm family. */
#include "gemm_types.h"

/* Normalization family.
 *
 * Every semantic here is a row-wise operation over a [rows, cols] row-major
 * activation: an optional bias/residual/gate combine, followed by an optional
 * normalization, followed by an optional output quantization.  The family is
 * grouped into one adapter because the bindings shape is shared; the semantic
 * selects which of the bindings participate.
 *
 * Which bindings each semantic reads:
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
 *   BIAS_THEN_RESIDUAL     input, bias, residual             -> hidden
 *
 * BIAS_THEN_RESIDUAL is BF16-only.  Unlike BIAS_RESIDUAL, it rounds
 * input+bias to BF16 before adding residual, preserving the legacy two-step
 * numerical contract in a single launch.
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
  APXINF_NORM_SEMANTIC_RMS = 0,
  APXINF_NORM_SEMANTIC_LAYER = 1,
  APXINF_NORM_SEMANTIC_ADAPTIVE_RMS = 2,
  APXINF_NORM_SEMANTIC_BIAS_RESIDUAL = 3,
  APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_RMS = 4,
  APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_LAYER = 5,
  APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL = 6,
  APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL_RMS = 7,
  APXINF_NORM_SEMANTIC_BIAS_THEN_RESIDUAL = 8,
} apxinf_norm_semantic_t;

#define APXINF_NORM_SPEC_VERSION 1u

typedef struct {
  uint32_t version;
  uint32_t semantic;
  uint32_t dtype;
  /* E4M3 output is the fused-quantization form and consumes output_scale.
     BF16/F16 output leaves the scale at one. */
  uint32_t output_dtype;
  /* A null bias is a different kernel path, so it belongs to the identity. */
  uint32_t has_bias;
  /* Largest guaranteed power-of-two byte alignment, capped at 256. */
  uint32_t input_alignment;
  uint32_t weight_alignment;
  uint32_t bias_alignment;
  uint32_t residual_alignment;
  uint32_t style_alignment;
  uint32_t hidden_alignment;
  uint32_t normalized_alignment;
  int64_t rows;
  int64_t cols;
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
  /* Pre-normalization result.  Null for the pure-normalization semantics. */
  void* hidden;
  /* Normalized result.  Null for the combine-only semantics. */
  void* normalized;
  apxinf_cuda_stream_t stream;
  /* Numeric per-call data: changes the result, never the fastest candidate. */
  float eps;
  float output_scale;
} apxinf_norm_bindings_t;
