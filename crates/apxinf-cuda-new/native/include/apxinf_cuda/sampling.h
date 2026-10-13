#pragma once

#include "types.h"
#include "status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Greedy token selection: the degenerate sampler, with no state, no workspace
   and no RNG, so it is a flat entry point rather than a prepare/enqueue pair
   (doc/cuda-new-design.md section 5). The stateful samplers in this adapter
   use the apxinf_sampling_* prefix and are declared in src/ffi/raw/sampling.rs. */
apxinf_status_t apxinf_sampling_argmax(const void* logits, void* index,
                                         int64_t count,
                                         apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
