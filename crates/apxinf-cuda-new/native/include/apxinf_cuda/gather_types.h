#pragma once

#include <stdint.h>

/* Shares the dtype enum and status codes with the gemm family. */
#include "gemm_types.h"

/* Gather / layout family: operations whose output index does not map one to
 * one onto a single input index.
 *
 *   EMBEDDING_LOOKUP  table [vocab_size, cols], ids [rows] (u32)
 *                     -> output [rows, cols], scaled by sqrt(cols).
 *                     Out-of-range ids produce zeros.
 *
 *   BIAS_POSITION     input [rows, cols], position [tokens_per_view, cols],
 *                     optional bias [cols] -> output [rows, cols].
 *                     The position vector repeats across views.
 *
 *   RGB_TO_PATCHES    images (u8) -> output
 *                     [views * patches_per_view, 3 * patch_size^2],
 *                     mapping [0, 255] to [-1, 1].
 *                     `rows` and `cols` must agree with that shape.
 */
typedef enum {
  APXINF_GATHER_SEMANTIC_EMBEDDING_LOOKUP = 0,
  APXINF_GATHER_SEMANTIC_BIAS_POSITION = 1,
  APXINF_GATHER_SEMANTIC_RGB_TO_PATCHES = 2,
} apxinf_gather_semantic_t;

#define APXINF_GATHER_SPEC_VERSION 1u

typedef struct {
  uint32_t version;
  uint32_t semantic;
  uint32_t dtype;
  /* A null bias is a different kernel path, so it belongs to the identity. */
  uint32_t has_bias;
  /* EMBEDDING_LOOKUP only. */
  uint32_t vocab_size;
  /* BIAS_POSITION only. */
  uint32_t tokens_per_view;
  /* RGB_TO_PATCHES only. */
  uint32_t views;
  uint32_t image_size;
  uint32_t patch_size;
  /* 1 when the source images are NHWC, 0 for NCHW. */
  uint32_t nhwc;
  /* Largest guaranteed power-of-two byte alignment, capped at 256. */
  uint32_t input_alignment;
  uint32_t bias_alignment;
  uint32_t output_alignment;
  int64_t rows;
  int64_t cols;
} apxinf_gather_spec_t;

typedef struct {
  /* Embedding table, activation, or packed u8 images depending on semantic. */
  const void* input;
  /* EMBEDDING_LOOKUP token ids. */
  const uint32_t* ids;
  const void* bias;
  /* BIAS_POSITION learned position embedding. */
  const void* position;
  void* output;
  apxinf_cuda_stream_t stream;
} apxinf_gather_bindings_t;
