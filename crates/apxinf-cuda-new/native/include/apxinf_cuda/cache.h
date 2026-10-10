#pragma once

#include "types.h"
#include "status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Append `append_len` rows of BF16 K/V data (or V data) into a portable
   KV cache laid out `[n_kv_heads, max_seq_len, head_dim]`, starting at row
   `seq_len`. `new_data` is `[append_len, n_kv_heads, head_dim]`. */
apxinf_status_t apxinf_cache_append_bf16(const void* new_data, void* cache,
                                         int32_t n_kv_heads, int32_t head_dim,
                                         int32_t max_seq_len, int32_t seq_len,
                                         int32_t append_len,
                                         apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
