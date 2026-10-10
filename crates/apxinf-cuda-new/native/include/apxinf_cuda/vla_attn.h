#pragma once

#include "types.h"
#include "status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* VLA attention primitives (walloss family): fused QKV split + mRoPE with
   KV-cache write, fused vision QKV split + 2D RoPE, and segmented dense
   MHA. */

apxinf_status_t apxinf_vla_gqa_qkv_mrope_cache_bf16(
    const void* qkv, const void* bias, const void* position_ids, void* q,
    void* k_cache, void* v_cache, int32_t tokens, int32_t q_heads,
    int32_t kv_heads, int32_t head_dim, float theta, int32_t section_h,
    int32_t section_w, int32_t cache_offset, apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_vla_vision_qkv_rope_bf16(
    const void* qkv, const void* bias, const void* position_ids, void* q,
    void* k, void* v, int32_t tokens, int32_t heads, int32_t head_dim,
    float theta, apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_vla_segmented_mha_bf16(
    const void* q, const void* k, const void* v, const void* offsets,
    void* output, int32_t segments, int32_t max_tokens, int32_t heads,
    int32_t head_dim, apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
