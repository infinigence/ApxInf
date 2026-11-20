#pragma once

#include "conv1d_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct apxinf_conv1d_execution* apxinf_conv1d_execution_t;
apxinf_status_t apxinf_conv1d_prepare(
    apxinf_runtime_t runtime, const apxinf_conv1d_spec_t* spec,
    const apxinf_conv1d_bindings_t* bindings, uint64_t workspace_limit, uint32_t ordered_accumulation,
    apxinf_conv1d_execution_t* execution);
apxinf_status_t apxinf_conv1d_enqueue(apxinf_conv1d_execution_t execution);
void apxinf_conv1d_destroy(apxinf_conv1d_execution_t execution);
const char* apxinf_conv1d_summary(apxinf_conv1d_execution_t execution);

/* Direct baseline retained for independent provider validation. */
apxinf_status_t apxinf_conv1d_launch(
    apxinf_runtime_t runtime, const apxinf_conv1d_spec_t* spec,
    const apxinf_conv1d_bindings_t* bindings);

#ifdef __cplusplus
}
#endif
