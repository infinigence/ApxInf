#pragma once

#include <stdint.h>

/* Shares the dtype enum and status codes with the gemm family. */
#include "gemm_types.h"

/* Element-wise family: one output element per input element (or per pair of
 * input elements for GeGLU), no cross-row reduction.
 *
 * Which bindings each semantic reads:
 *
 *   GEGLU             input [rows, 2 * cols]            -> output [rows, cols]
 *   BIAS_ACTIVATION   input [rows, cols], bias [cols]   -> output [rows, cols]
 *   EULER_UPDATE      input [rows, cols] (state),
 *                     secondary [rows, cols] (velocity) -> output [rows, cols]
 *   SWIGLU            input [rows, 2 * cols]            -> output [rows, cols]
 *   ADD_INTO          input [rows, cols]                 -> output [rows, cols]
 *   SWIGLU_ROWWISE_E4M3
 *                     input [rows, input_cols], optional bias [input_cols]
 *                     -> output [rows, cols], scales [rows]
 *
 * `cols` is always the *output* width, so GeGLU's input is twice as wide.
 */
typedef enum {
  APXINF_POINTWISE_SEMANTIC_GEGLU = 0,
  APXINF_POINTWISE_SEMANTIC_BIAS_ACTIVATION = 1,
  APXINF_POINTWISE_SEMANTIC_EULER_UPDATE = 2,
  APXINF_POINTWISE_SEMANTIC_SWIGLU = 3,
  APXINF_POINTWISE_SEMANTIC_ADD_INTO = 4,
  APXINF_POINTWISE_SEMANTIC_SWIGLU_ROWWISE_E4M3 = 5,
  APXINF_POINTWISE_SEMANTIC_ADD = 6,
  APXINF_POINTWISE_SEMANTIC_SUBTRACT = 7,
  APXINF_POINTWISE_SEMANTIC_MULTIPLY = 8,
  APXINF_POINTWISE_SEMANTIC_BROADCAST_ADD = 9,
  APXINF_POINTWISE_SEMANTIC_BROADCAST_MULTIPLY = 10,
  APXINF_POINTWISE_SEMANTIC_ROW_MULTIPLY = 11,
  APXINF_POINTWISE_SEMANTIC_SCALE = 12,
  APXINF_POINTWISE_SEMANTIC_SCALAR_ADD = 13,
  APXINF_POINTWISE_SEMANTIC_SCALAR_DIVIDE = 14,
  APXINF_POINTWISE_SEMANTIC_EXP = 15,
  APXINF_POINTWISE_SEMANTIC_SIN = 16,
  APXINF_POINTWISE_SEMANTIC_COS = 17,
  /* secondary [1,cols] = scale; bias [cols] = shift. Round 1+scale,
   * multiplication, then addition separately to dtype. */
  APXINF_POINTWISE_SEMANTIC_STAGED_MODULATION = 18,
  /* secondary [rows,cols] = residual; bias [cols] = gate. Round the
   * multiplication before adding residual and rounding again. */
  APXINF_POINTWISE_SEMANTIC_STAGED_GATED_RESIDUAL = 19,
} apxinf_pointwise_semantic_t;

/* Stable pointwise activation ABI encoding. */
typedef enum {
  APXINF_POINTWISE_ACTIVATION_NONE = 0,
  APXINF_POINTWISE_ACTIVATION_GELU = 1,
  APXINF_POINTWISE_ACTIVATION_SILU = 2,
  APXINF_POINTWISE_ACTIVATION_GELU_EXACT = 3,
  APXINF_POINTWISE_ACTIVATION_RELU = 4,
  APXINF_POINTWISE_ACTIVATION_GELU_ROUNDED = 5,
  APXINF_POINTWISE_ACTIVATION_SWISH_ROUNDED = 6,
} apxinf_pointwise_activation_t;

#define APXINF_POINTWISE_SPEC_VERSION 3u

typedef struct {
  uint32_t version;
  uint32_t semantic;
  uint32_t dtype;
  uint32_t output_dtype;
  /* Only meaningful for BIAS_ACTIVATION. */
  uint32_t activation;
  /* A null bias is a different kernel path, so it belongs to the identity. */
  uint32_t has_bias;
  /* Largest guaranteed power-of-two byte alignment, capped at 256. */
  uint32_t input_alignment;
  uint32_t secondary_alignment;
  uint32_t bias_alignment;
  uint32_t output_alignment;
  int64_t rows;
  /* Output width.  GEGLU reads a [rows, 2 * cols] input. */
  int64_t cols;
  /* Physical input width and logical unpadded output width. */
  int64_t input_cols;
  int64_t logical_cols;
  /* Structural scale predicate, not the scale value. */
  uint32_t output_scale_is_unit;
} apxinf_pointwise_spec_t;

typedef struct {
  const void* input;
  /* Binary operand: full shape, [1,cols] for BROADCAST_*, [rows,1] for ROW_*. */
  const void* secondary;
  /* Per-column bias, or required shift/gate for staged affine semantics. */
  const void* bias;
  void* output;
  /* SWIGLU_ROWWISE_E4M3 writes one F32 scale per row. */
  void* scales;
  apxinf_cuda_stream_t stream;
  /* EULER_UPDATE step size or SCALE/SCALAR_ADD scalar. */
  float dt;
  float output_scale;
} apxinf_pointwise_bindings_t;
