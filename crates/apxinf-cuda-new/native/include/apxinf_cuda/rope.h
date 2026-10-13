#pragma once

#include "rope_types.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_rope_launch(
    apxinf_runtime_t runtime, const apxinf_rope_spec_t* spec,
    const apxinf_rope_bindings_t* bindings);

/* The family's non-Spec member: rotates one [tokens, heads, head_dim] tensor
   in place, where every Spec semantic fuses a QKV split and writes three
   outputs. Kernel in kernels/custom/rope_partial.cu, which also records the
   two scope limits -- the mRoPE collapse is text-only, and the rotate-half
   pairing convention is part of the contract. */
apxinf_status_t apxinf_rope_partial(void* data, const void* positions,
                                         int64_t tokens, int64_t heads,
                                         int64_t head_dim, int64_t rotary_dim,
                                         float theta,
                                         apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
