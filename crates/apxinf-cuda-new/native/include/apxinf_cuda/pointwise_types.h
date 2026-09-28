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
 *
 * `cols` is always the *output* width, so GeGLU's input is twice as wide.
 */
typedef enum {
  APXINF_POINTWISE_SEMANTIC_GEGLU = 0,
  APXINF_POINTWISE_SEMANTIC_BIAS_ACTIVATION = 1,
  APXINF_POINTWISE_SEMANTIC_EULER_UPDATE = 2,
} apxinf_pointwise_semantic_t;

/* Stable pointwise activation ABI encoding. */
typedef enum {
  APXINF_POINTWISE_ACTIVATION_NONE = 0,
  APXINF_POINTWISE_ACTIVATION_GELU = 1,
  APXINF_POINTWISE_ACTIVATION_SILU = 2,
} apxinf_pointwise_activation_t;

#define APXINF_POINTWISE_SPEC_VERSION 1u

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
  /* Structural scale predicate, not the scale value. */
  uint32_t output_scale_is_unit;
} apxinf_pointwise_spec_t;

typedef struct {
  const void* input;
  /* EULER_UPDATE reads the velocity here; unused elsewhere. */
  const void* secondary;
  const void* bias;
  void* output;
  apxinf_cuda_stream_t stream;
  /* EULER_UPDATE step size; unused elsewhere. */
  float dt;
  float output_scale;
} apxinf_pointwise_bindings_t;
