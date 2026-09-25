#pragma once

#include "types.h"
#include "status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Full-attention primitives. See kernels/custom/attn_ops.h for the scope
   limits on mRoPE and the rotary pairing convention. */

apxinf_status_t apxinf_attn_partial_rope(void* data, const void* positions,
                                         int64_t tokens, int64_t heads,
                                         int64_t head_dim, int64_t rotary_dim,
                                         float theta,
                                         apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_attn_head_rms_norm(void* data, const void* weight,
                                          int64_t rows, int64_t head_dim,
                                          float epsilon,
                                          apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_attn_split_query_and_gate(
    const void* fused, void* query, void* gate, int64_t tokens, int64_t heads,
    int64_t head_dim, apxinf_cuda_stream_t stream);

/* data *= sigmoid(gate). Not silu: see native/kernels/custom/attn_ops.h. */
apxinf_status_t apxinf_attn_apply_output_gate(void* data, const void* gate,
                                              int64_t count,
                                              apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
