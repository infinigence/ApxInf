#pragma once

#include <stdint.h>

/* Shares the dtype enum and status codes with the gemm family. */
#include "gemm_types.h"

/* Packed-QKV split family.
 *
 *   SPLIT_QKV_ROPE  qkv [tokens, q_heads*head_dim + 2*kv_heads*head_dim]
 *                   -> q [tokens, q_heads, head_dim]
 *                      k, v [*, kv_heads, head_dim] written at
 *                      kv_output_offset rows
 *                   Q and K are rotated; V is copied.
 *
 *   SPLIT_QKV_BIAS  qkv [tokens, 3 * head_dim * q_heads]
 *                   -> q, k, v [tokens, q_heads, head_dim], no rotation.
 *                   The vision tower's MHA layout: kv_heads == q_heads.
 *
 * `kv_output_offset` is what lets one kernel serve both prefill (offset 0,
 * fresh K/V buffers) and decode (offset = prefix length, appending into a KV
 * cache).  It is per-call data and lives in the bindings, so one prepared
 * execution is not pinned to a single decode step.
 */
typedef enum {
  APXINF_ROPE_SEMANTIC_SPLIT_QKV_ROPE = 0,
  APXINF_ROPE_SEMANTIC_SPLIT_QKV_BIAS = 1,
  /* Single-token BF16 decode. Q is rotated into q, K is rotated directly
     into the token-major cache, and V is appended to its cache. The current
     position is read from bindings.position at enqueue/replay time. */
  APXINF_ROPE_SEMANTIC_DECODE_QKV_CACHE = 2,
} apxinf_rope_semantic_t;

#define APXINF_ROPE_SPEC_VERSION 2u

typedef struct {
  uint32_t version;
  uint32_t semantic;
  uint32_t dtype;
  /* A null bias is a different kernel path, so it belongs to the identity. */
  uint32_t has_bias;
  uint32_t q_heads;
  uint32_t kv_heads;
  uint32_t head_dim;
  /* Largest guaranteed power-of-two byte alignment, capped at 256. */
  uint32_t qkv_alignment;
  uint32_t bias_alignment;
  uint32_t q_alignment;
  uint32_t kv_alignment;
  uint32_t position_alignment;
  int64_t tokens;
  int64_t cache_capacity;
} apxinf_rope_spec_t;

typedef struct {
  const void* qkv;
  const void* bias;
  void* q;
  void* k;
  void* v;
  /* DECODE_QKV_CACHE-only K/V source rows. */
  const void* key_input;
  const void* value_input;
  /* Stable device u32 read at enqueue/replay time. */
  const uint32_t* position;
  apxinf_cuda_stream_t stream;
  /* RoPE base; unused by SPLIT_QKV_BIAS. */
  float theta;
  /* Absolute position of token 0; unused by SPLIT_QKV_BIAS. */
  int32_t position_offset;
  /* Row offset into the K/V destination; 0 for fresh buffers. */
  int32_t kv_output_offset;
} apxinf_rope_bindings_t;
