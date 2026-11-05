#pragma once

#include <stdint.h>

#include "gemm_types.h"

typedef enum {
  APXINF_ATTENTION_MASK_NONE = 0,
  APXINF_ATTENTION_MASK_CAUSAL = 1,
} apxinf_attention_mask_t;

typedef enum {
  APXINF_ATTENTION_SEMANTIC_DENSE = 0,
  APXINF_ATTENTION_SEMANTIC_KV_CACHE = 1,
  APXINF_ATTENTION_SEMANTIC_SEGMENTED = 2,
} apxinf_attention_semantic_t;

typedef enum {
  APXINF_ATTENTION_QKV_LAYOUT_SEPARATE = 0,
  APXINF_ATTENTION_QKV_LAYOUT_PACKED_TOKEN = 1,
} apxinf_attention_qkv_layout_t;

typedef enum {
  APXINF_ATTENTION_NUMERICS_FLEXIBLE = 0,
  // Preserve the reduction/rounding behavior validated before Attention
  // candidate consolidation. Candidates must opt in after model-level
  // qualification; this is a numerical contract, not a provider request.
  APXINF_ATTENTION_NUMERICS_STABLE = 1,
} apxinf_attention_numerics_t;

typedef struct {
  uint32_t version;
  uint32_t semantic;
  uint32_t dtype;
  uint32_t output_dtype;
  uint32_t mask;
  uint32_t qkv_layout;
  uint32_t q_alignment;
  uint32_t k_alignment;
  uint32_t v_alignment;
  uint32_t output_alignment;
  uint32_t offsets_alignment;
  int64_t batch;
  int64_t query_tokens;
  int64_t key_tokens;
  int64_t key_capacity;
  int64_t query_heads;
  int64_t kv_heads;
  int64_t head_dim;
  int64_t query_start;
  int64_t segments;
  int64_t max_segment_tokens;
  uint64_t offsets_hash;
  uint32_t scale_is_default;
  uint32_t dynamic_decode;
  uint32_t numerics;
} apxinf_attention_spec_t;

typedef apxinf_tuning_policy_t apxinf_attention_policy_t;

typedef struct {
  const void* query;
  const void* key;
  const void* value;
  const void* offsets;
  const uint32_t* decode_meta;
  void* output;
  apxinf_cuda_stream_t stream;
  float scale;
  float output_scale;
} apxinf_attention_bindings_t;

#define APXINF_ATTENTION_GATE_SPEC_VERSION 1u

typedef enum {
  APXINF_ATTENTION_GATE_SEMANTIC_SPLIT_QUERY_GATE = 0,
  APXINF_ATTENTION_GATE_SEMANTIC_APPLY_OUTPUT_GATE = 1,
} apxinf_attention_gate_semantic_t;

typedef enum {
  APXINF_ATTENTION_GATE_SOURCE_TENSOR = 0,
  APXINF_ATTENTION_GATE_SOURCE_PROJECTION = 1,
} apxinf_attention_gate_source_t;

typedef enum {
  APXINF_ATTENTION_GATE_NUMERICS_FLEXIBLE = 0,
  APXINF_ATTENTION_GATE_NUMERICS_ROUNDED_SIGMOID = 1,
} apxinf_attention_gate_numerics_t;

typedef struct {
  uint32_t version;
  uint32_t semantic;
  uint32_t dtype;
  uint32_t source;
  uint32_t numerics;
  uint32_t input_alignment;
  uint32_t output_alignment;
  uint32_t auxiliary_alignment;
  int64_t tokens;
  int64_t heads;
  int64_t head_dim;
  int64_t elements;
  int64_t source_row_width;
} apxinf_attention_gate_spec_t;

typedef struct {
  const void* input;
  void* output;
  void* auxiliary;
  apxinf_cuda_stream_t stream;
} apxinf_attention_gate_bindings_t;
