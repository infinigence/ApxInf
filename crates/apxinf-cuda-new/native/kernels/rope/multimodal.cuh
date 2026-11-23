// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace apxinf::rope::multimodal {

int gqa_qkv_mrope_cache_bf16(const void* qkv, const void* bias,
                             const uint32_t* position_ids, void* q,
                             void* k_cache, void* v_cache, int tokens,
                             int q_heads, int kv_heads, int head_dim,
                             float theta, int section_h, int section_w,
                             int cache_offset, cudaStream_t stream);

int vision_qkv_rope_bf16(const void* qkv, const void* bias,
                         const uint32_t* position_ids, void* q, void* k,
                         void* v, int tokens, int heads, int head_dim,
                         float theta, cudaStream_t stream);

}  // namespace apxinf::rope::multimodal
