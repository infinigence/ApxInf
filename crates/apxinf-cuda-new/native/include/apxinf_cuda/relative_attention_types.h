#pragma once

#include <stdint.h>

#include "gemm_types.h"

/* Relative-position self-attention used by the SoVITS encoder family.
 *
 * Q/K/V are [batch, length, heads, head_dim] contiguous.  The relative key
 * and value tables are shared across heads and shaped [2 * window + 1,
 * head_dim]; positions farther than `window` contribute nothing.
 *
 *   score[t, j] = scale * q[t] . k[j]
 *               + scale * q[t] . emb_k[j - t + window]   (when in range)
 *   out[t]      = sum_j softmax(score[t, :])[j] *
 *                 (v[j] + emb_v[j - t + window])          (when in range)
 */
#define APXINF_RELATIVE_ATTENTION_SPEC_VERSION 1u

typedef struct {
  uint32_t version;
  uint32_t dtype;
  uint32_t heads_share_relative;
  uint32_t reserved;
  int64_t batch;
  int64_t length;
  int64_t heads;
  int64_t head_dim;
  int64_t window;
  float scale;
} apxinf_relative_attention_spec_t;

typedef struct {
  const void* query;
  const void* key;
  const void* value;
  const void* relative_key;
  const void* relative_value;
  void* output;
  apxinf_cuda_stream_t stream;
} apxinf_relative_attention_bindings_t;
