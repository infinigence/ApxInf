#pragma once

#include "status.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_sampling_argmax(
    const void* logits, void* index, int64_t count,
    apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
