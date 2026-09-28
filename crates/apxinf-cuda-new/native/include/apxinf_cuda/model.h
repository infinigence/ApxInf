#pragma once

#include "types.h"
#include "status.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_model_embedding_gather(const void* table,
                                              const void* ids, void* output,
                                              int64_t tokens, int64_t hidden,
                                              int64_t vocab,
                                              apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_model_argmax_bf16(const void* logits, void* index,
                                         int64_t count,
                                         apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
