#pragma once

#include <stdint.h>

/* Shares the dtype enum and status codes with the gemm family. */
#include "gemm_types.h"

/* Normalization family.
 *
 * Every operation here is a row-wise pipeline over a [rows, cols] row-major
 * activation:
 *
 *   combine   optional bias or gate applied to `input`, plus `residual`
 *   norm      optional RMS or LayerNorm reduction over the combined row
 *   affine    optional rescale of the normalized row
 *   quant     optional output quantization
 *
 * The Spec describes those stages with independent fields rather than one
 * fused semantic code, so the adapter dispatches on the stage combination.
 * `semantic` only says which outputs are written:
 *
 *   NORM          combine skipped; writes `normalized` only
 *   FUSED_ADD_NORM  combine runs; writes `hidden` and `normalized`
 *   RESIDUAL_ADD  norm skipped; writes `hidden` only
 *
 * Which bindings participate is fully determined by the fields:
 *
 *   input         always
 *   residual      semantic != NORM
 *   bias          optional, and only when semantic != NORM && !has_gate
 *   gate_style    has_gate
 *   weight        affine_kind == WEIGHT or WEIGHT_OFFSET
 *   norm_bias     norm_kind == LAYER
 *   norm_style    affine_kind == ADA_SCALE_SHIFT
 *   hidden        semantic != NORM
 *   normalized    semantic != RESIDUAL_ADD
 *
 * The two adaptive-conditioning vectors are distinct and are not
 * interchangeable:
 *
 *   norm_style  [2 * cols]  scale in [0, cols), shift in [cols, 2 * cols)
 *   gate_style  [3 * cols]  the gate is the third segment,
 *                           [2 * cols, 3 * cols)
 *
 * The gated FUSED_ADD_NORM form reads the gate from the *current* layer's
 * gate_style and normalizes with the *next* layer's norm_style: the pi05
 * action expert folds the next layer's normalization into the current layer's
 * residual.
 */
typedef enum {
  /* Pure normalization: no residual combine, single output. */
  APXINF_NORM_SEMANTIC_NORM = 0,
  /* Residual combine followed by normalization: hidden + normalized. */
  APXINF_NORM_SEMANTIC_FUSED_ADD_NORM = 1,
  /* Residual combine only: hidden, no reduction. */
  APXINF_NORM_SEMANTIC_RESIDUAL_ADD = 2,
} apxinf_norm_semantic_t;

typedef enum {
  APXINF_NORM_KIND_NONE = 0,
  APXINF_NORM_KIND_RMS = 1,
  APXINF_NORM_KIND_LAYER = 2,
} apxinf_norm_kind_t;

typedef enum {
  APXINF_NORM_AFFINE_NONE = 0,
  /* normalized * weight[col] */
  APXINF_NORM_AFFINE_WEIGHT = 1,
  /* normalized * (1 + weight[col]).  Not a flag on WEIGHT: it is served by a
     kernel of its own in kernels/custom/norm_offset.cu, whose block size
     follows `cols` and whose reduction tree differs from the norm.cuh family,
     so the two are not bit-comparable beyond the `1 +`.  That kernel covers
     exactly one combination -- semantic NORM, KIND_RMS, BF16 dtype,
     QUANT_NONE -- and the adapter reports UNSUPPORTED for the rest rather
     than falling back to WEIGHT. */
  APXINF_NORM_AFFINE_WEIGHT_OFFSET = 2,
  /* normalized * (1 + norm_style[col]) + norm_style[cols + col] */
  APXINF_NORM_AFFINE_ADA_SCALE_SHIFT = 3,
} apxinf_norm_affine_t;

typedef enum {
  /* One rounding of input + bias + residual. */
  APXINF_NORM_BIAS_ORDER_SINGLE_ROUND = 0,
  /* input + bias rounds to the storage dtype before residual is added and
     rounded again.  BF16-only, and distinct from SINGLE_ROUND at rounding
     boundaries: it preserves the legacy two-kernel contract in one launch. */
  APXINF_NORM_BIAS_ORDER_ROUND_AFTER_BIAS = 1,
} apxinf_norm_bias_order_t;

typedef enum {
  APXINF_NORM_QUANT_NONE = 0,
  /* Fused E4M3 output; consumes output_scale.  F16 input only. */
  APXINF_NORM_QUANT_E4M3 = 1,
} apxinf_norm_quant_t;

#define APXINF_NORM_SPEC_VERSION 2u

typedef struct {
  uint32_t version;
  /* apxinf_norm_semantic_t: which outputs are written. */
  uint32_t semantic;
  /* apxinf_norm_kind_t: which reduction, if any, runs over the row. */
  uint32_t norm_kind;
  /* apxinf_norm_affine_t: how the normalized row is rescaled. */
  uint32_t affine_kind;
  /* Whether the combine multiplies the projection by gate_style's gate. */
  uint32_t has_gate;
  /* apxinf_norm_bias_order_t: rounding order of the combine. */
  uint32_t bias_order;
  /* apxinf_norm_quant_t: fused output quantization. */
  uint32_t quant_kind;
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
  /* Pre-normalization result.  Null when semantic is NORM. */
  void* hidden;
  /* Normalized result.  Null when semantic is RESIDUAL_ADD. */
  void* normalized;
  apxinf_cuda_stream_t stream;
  /* Numeric per-call data: changes the result, never the fastest candidate. */
  float eps;
  float output_scale;
} apxinf_norm_bindings_t;
