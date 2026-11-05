#pragma once

#include <stdint.h>

#include "gemm_types.h"

/* Gated DeltaNet family.
 *
 * One versioned Spec describes the mathematical operation and tensor
 * geometry. Per-call pointers and numeric scalars live in Bindings. Native
 * dispatch preserves the existing kernels and rounding paths; dtype, rank,
 * layout and quantization no longer multiply the family ABI.
 */
typedef enum {
  APXINF_GDN_OPERATION_RECURRENT_STEP = 0,
  APXINF_GDN_OPERATION_GATED_NORM = 1,
  APXINF_GDN_OPERATION_CAUSAL_CONV = 2,
  APXINF_GDN_OPERATION_DECAY_AND_BETA = 3,
  APXINF_GDN_OPERATION_L2_NORMALIZE = 4,
  APXINF_GDN_OPERATION_CHUNK_SCAN = 5,
  APXINF_GDN_OPERATION_PREFILL = 6,
  APXINF_GDN_OPERATION_PREPARE_PREFILL = 7,
  APXINF_GDN_OPERATION_CONV_PREPARE = 8,
} apxinf_gdn_operation_t;

typedef enum {
  APXINF_GDN_LAYOUT_DENSE = 0,
  APXINF_GDN_LAYOUT_STRIDED_GATE = 1,
  APXINF_GDN_LAYOUT_SPLIT_QKV = 2,
  APXINF_GDN_LAYOUT_INTERLEAVED_QKV = 3,
} apxinf_gdn_layout_t;

typedef enum {
  APXINF_GDN_QUANTIZATION_NONE = 0,
  APXINF_GDN_QUANTIZATION_STATIC_E4M3 = 1,
} apxinf_gdn_quantization_t;

#define APXINF_GDN_SPEC_VERSION 1u

typedef struct {
  uint32_t version;
  uint32_t operation;
  uint32_t input_dtype;
  uint32_t output_dtype;
  uint32_t rank;
  uint32_t layout;
  uint32_t quantization;
  uint32_t has_window;
  int64_t tokens;
  int64_t heads;
  int64_t head_dim;
  int64_t key_heads;
  int64_t value_heads;
  int64_t key_dim;
  int64_t value_dim;
  int64_t channels;
  int64_t kernel_width;
  int64_t row_width;
  int64_t num_sequences;
  int64_t chunk_size;
  int64_t num_chunks;
  int64_t q_row_stride;
  int64_t k_row_stride;
  int64_t v_row_stride;
  int64_t gate_row_stride;
  int64_t gate_column_offset;
} apxinf_gdn_spec_t;

typedef struct {
  const void* input;
  const void* gate;
  const void* weight;
  const void* q;
  const void* k;
  const void* v;
  void* decay;
  void* beta;
  const void* a;
  const void* b;
  const void* a_log;
  const void* dt_bias;
  const void* gate_log;
  const void* cu_seqlens;
  void* output;
  void* state;
  void* window;
  void* q_output;
  void* k_output;
  void* v_output;
  void* alpha;
  void* quantized;
  void* workspace;
  apxinf_cuda_stream_t stream;
  float epsilon;
  float scale;
  float input_scale;
} apxinf_gdn_bindings_t;
