#pragma once

#include "rope_types.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_rope_partial(
    void* data, const void* positions, int64_t tokens, int64_t heads,
    int64_t head_dim, int64_t rotary_dim, float theta,
    apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_rope_launch(
    apxinf_runtime_t runtime, const apxinf_rope_spec_t* spec,
    const apxinf_rope_bindings_t* bindings);

#ifdef __cplusplus
}
#endif

/* Fixed adjacent-pair rotation, ABI version 1. Coefficients are FP64
 * [tokens, head_dim / 2, 2]. Input BF16; output dtype 0=F32 or 2=BF16. */
#ifdef __cplusplus
extern "C"
#endif
apxinf_status_t apxinf_pairwise_rope_launch(
    apxinf_runtime_t runtime, uint32_t version, const void* input,
    const void* coefficients, void* output, int64_t tokens, int64_t heads,
    int64_t head_dim, uint32_t output_dtype, apxinf_cuda_stream_t stream);
