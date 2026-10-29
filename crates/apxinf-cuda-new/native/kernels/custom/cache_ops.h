// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

// KV-cache append for the portable `apxinf_core::KvCache` trait.
//
// Storage is sequence-major within one batch: a flat
// `[1, max_seq_len, n_kv_heads, head_dim]` buffer, the exact view cuda-new's
// `kv_cache_attention` consumes. A model that owns a `CudaKVCache` never
// inspects the layout directly — it only appends through this kernel and reads
// through the Attention operator — so the storage order is an implementation
// detail, chosen here to avoid a relayout on every attention call.
namespace apxinf::cuda_new::cache_ops {

// Append `append_len` rows of `new_data [append_len, n_kv_heads, head_dim]`
// into a flat `[1, max_seq_len, n_kv_heads, head_dim]` cache starting at
// sequence position `seq_len`.
int append_bf16(void* cache, const void* new_data, int n_kv_heads, int head_dim,
                int max_seq_len, int seq_len, int append_len,
                cudaStream_t stream);

}  // namespace apxinf::cuda_new::cache_ops
