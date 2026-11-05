#pragma once
#include "gemm_types.h"
#define APXINF_RECURRENCE_SPEC_VERSION 1u
// Model-neutral mixed-precision linear recurrence. Each FP64 multiply and add
// rounds independently; state and previous values remain FP64 between calls.
typedef struct { uint32_t version, reset; int64_t count; double a,b,c,d; } apxinf_recurrence_spec_t;
typedef struct {
  const float* input; float* output; double* state; double* previous;
  apxinf_cuda_stream_t stream;
} apxinf_recurrence_bindings_t;
#ifdef __cplusplus
extern "C" {
#endif
apxinf_status_t apxinf_recurrence_prepare(apxinf_runtime_t,
    const apxinf_recurrence_spec_t*, const apxinf_recurrence_bindings_t*, void**);
apxinf_status_t apxinf_recurrence_enqueue(void*);
void apxinf_recurrence_destroy(void*);
#ifdef __cplusplus
}
#endif
