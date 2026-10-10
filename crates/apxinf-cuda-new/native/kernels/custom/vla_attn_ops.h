// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

// VLA attention primitives ported bit-identically from the legacy walloss
// kernels: fused QKV split + mRoPE with KV-cache write, fused vision QKV
// split + 2D RoPE, and segmented (per-image) dense MHA.
namespace apxinf::cuda_new::vla_attn_ops {

// Split a packed `[tokens, (q_heads + 2*kv_heads) * head_dim]` projection,
// add the optional bias, rotate Q/K by mRoPE (three position ids per token,
// sections split the rotary pairs across temporal/height/width axes), and
// write K/V at `cache_offset` in `[cache_tokens, kv_heads, head_dim]` caches.
int gqa_qkv_mrope_cache_bf16(const void* qkv, const void* bias,
                             const uint32_t* position_ids, void* q,
                             void* k_cache, void* v_cache, int tokens,
                             int q_heads, int kv_heads, int head_dim,
                             float theta, int section_h, int section_w,
                             int cache_offset, cudaStream_t stream);

// Split a packed `[tokens, 3 * heads * head_dim]` vision projection, add the
// optional bias, and rotate Q/K by 2D RoPE (two position ids per token; the
// first half of each head's pairs uses axis 0, the second half axis 1).
// The biased value rounds to BF16 before rotation, matching the legacy
// kernel exactly.
int vision_qkv_rope_bf16(const void* qkv, const void* bias,
                         const uint32_t* position_ids, void* q, void* k,
                         void* v, int tokens, int heads, int head_dim,
                         float theta, cudaStream_t stream);

// Dense non-causal MHA over `[tokens, heads, head_dim]` where attention is
// confined to segments delimited by `offsets` (`segments + 1` u32 entries).
// F32 scores, serial softmax on thread 0 — bit-identical to the legacy
// fallback kernel.
int segmented_mha_bf16(const void* q, const void* k, const void* v,
                       const void* offsets, void* output, int segments,
                       int max_tokens, int heads, int head_dim,
                       cudaStream_t stream);

}  // namespace apxinf::cuda_new::vla_attn_ops
