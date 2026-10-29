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
