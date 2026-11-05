#pragma once
#include "gemm_types.h"
#define APXINF_INDEXED_MEAN_SPEC_VERSION 1u
typedef struct { uint32_t version,reserved; int64_t input_elements,rows,cols; } apxinf_indexed_mean_spec_t;
typedef struct { const float* input; const uint32_t* indices; float* output; apxinf_cuda_stream_t stream; } apxinf_indexed_mean_bindings_t;
#ifdef __cplusplus
extern "C" {
#endif
apxinf_status_t apxinf_indexed_mean_prepare(apxinf_runtime_t,const apxinf_indexed_mean_spec_t*,const apxinf_indexed_mean_bindings_t*,void**);
apxinf_status_t apxinf_indexed_mean_enqueue(void*);
void apxinf_indexed_mean_destroy(void*);
#ifdef __cplusplus
}
#endif
