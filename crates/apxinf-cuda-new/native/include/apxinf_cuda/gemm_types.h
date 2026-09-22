#pragma once

#include <stdint.h>

#include "status.h"
#include "tuning_types.h"
#include "types.h"

typedef enum {
  APXINF_DTYPE_F32 = 0,
  APXINF_DTYPE_F16 = 1,
  APXINF_DTYPE_BF16 = 2,
  APXINF_DTYPE_E4M3 = 3,
  APXINF_DTYPE_I8 = 4,
  APXINF_DTYPE_I32 = 5,
} apxinf_dtype_t;

typedef enum {
  APXINF_GEMM_LAYOUT_KN = 0,
  APXINF_GEMM_LAYOUT_NK = 1,
  APXINF_GEMM_LAYOUT_GATE_UP_INTERLEAVED_256 = 2,
} apxinf_gemm_layout_t;

typedef enum {
  APXINF_GEMM_QUANT_NONE = 0,
  APXINF_GEMM_QUANT_FP8_UNIT_SCALE = 1,
  APXINF_GEMM_QUANT_FP8_ROW_CHANNEL = 2,
  APXINF_GEMM_QUANT_W8A8_ROW_CHANNEL = 3,
} apxinf_gemm_quantization_t;

typedef enum {
  APXINF_GEMM_SEMANTIC_GEMM = 0,
  APXINF_GEMM_SEMANTIC_GEMM_BIAS_GELU = 1,
  APXINF_GEMM_SEMANTIC_GEMM_GEGLU = 2,
  APXINF_GEMM_SEMANTIC_GEMM_BIAS = 3,
} apxinf_gemm_semantic_t;

typedef struct {
  uint32_t version;
  uint32_t semantic;
  uint32_t a_dtype;
  uint32_t b_dtype;
  uint32_t accumulation_dtype;
  uint32_t output_dtype;
  uint32_t quantization;
  /* Changes whether candidate preparation may hoist weight preprocessing. */
  uint32_t b_is_immutable;
  /* Largest guaranteed power-of-two byte alignment, capped at 256. */
  uint32_t a_alignment;
  uint32_t b_alignment;
  uint32_t bias_alignment;
  uint32_t a_scales_alignment;
  uint32_t b_scales_alignment;
  uint32_t output_alignment;
  int64_t m;
  int64_t n;
  int64_t k;
  /* Structural scale predicates, not scale values.  Some candidates only
     implement the unit case, so "is this scale exactly one" belongs to the
     operation identity while the scale itself is a per-call binding.  Keeping
     the value out of the Spec is what lets one tuned Recipe serve every layer
     that differs only by its calibration scale. */
  uint32_t alpha_is_unit;
  uint32_t output_scale_is_unit;
} apxinf_gemm_spec_t;

typedef apxinf_tuning_policy_t apxinf_gemm_policy_t;

typedef struct {
  const void* a;
  const void* b;
  /* A prepared-weight identity is (b allocation address, b_version). */
  uint64_t b_version;
  uint32_t b_is_immutable;
  const void* bias;
  const float* a_scales;
  const float* b_scales;
  void* output;
  apxinf_cuda_stream_t stream;
  /* Numeric scales are per-call data: they change the result but never which
     candidate is fastest, so they are bound at execution time. */
  float alpha;
  float output_scale;
} apxinf_gemm_bindings_t;
