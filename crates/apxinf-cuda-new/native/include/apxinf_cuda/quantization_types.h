#pragma once

#include <stdint.h>

#include "gemm_types.h"

/* Quantization and representation-conversion family.
 *
 * FIXED_E4M3       input [rows, input_cols] -> E4M3 [rows, output_cols]
 *                  (input_cols == output_cols), using bindings.scale.
 * ROWWISE_E4M3     BF16 input [rows, input_cols] -> E4M3
 *                  [rows, output_cols] plus F32 scales [rows].  Columns in
 *                  [input_cols, output_cols) are exactly zero.
 * CAST_F16_BF16    F16 input -> BF16 output, same shape.
 * SLICE_BF16       leading columns of a contiguous BF16 matrix.
 * ROWWISE_I8       BF16 input -> I8 output plus F32 scales [rows].
 */
typedef enum {
  APXINF_QUANTIZATION_SEMANTIC_FIXED_E4M3 = 0,
  APXINF_QUANTIZATION_SEMANTIC_ROWWISE_E4M3 = 1,
  APXINF_QUANTIZATION_SEMANTIC_CAST_F16_BF16 = 2,
  APXINF_QUANTIZATION_SEMANTIC_SLICE_BF16 = 3,
  APXINF_QUANTIZATION_SEMANTIC_ROWWISE_I8 = 4,
} apxinf_quantization_semantic_t;

#define APXINF_QUANTIZATION_SPEC_VERSION 1u

typedef struct {
  uint32_t version;
  uint32_t semantic;
  uint32_t input_dtype;
  uint32_t output_dtype;
  uint32_t scale_dtype;
  uint32_t input_alignment;
  uint32_t output_alignment;
  uint32_t scales_alignment;
  int64_t rows;
  int64_t input_cols;
  int64_t output_cols;
} apxinf_quantization_spec_t;

typedef struct {
  const void* input;
  void* output;
  /* Required only by the two ROWWISE semantics. */
  float* scales;
  apxinf_cuda_stream_t stream;
  /* Required only by FIXED_E4M3. Numeric values are call bindings rather
     than recipe identity because they do not affect candidate legality. */
  float scale;
} apxinf_quantization_bindings_t;
