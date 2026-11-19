#pragma once

#include "conv2d_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct apxinf_conv2d_execution* apxinf_conv2d_execution_t;
apxinf_status_t apxinf_conv2d_prepare(
    apxinf_runtime_t runtime, const apxinf_conv2d_spec_t* spec,
    const apxinf_conv2d_bindings_t* bindings,
    apxinf_conv2d_execution_t* execution);
apxinf_status_t apxinf_conv2d_enqueue(apxinf_conv2d_execution_t execution);
void apxinf_conv2d_destroy(apxinf_conv2d_execution_t execution);
const char* apxinf_conv2d_summary(apxinf_conv2d_execution_t execution);

/* Direct baseline launch without a prepared execution. */
apxinf_status_t apxinf_conv2d_launch(
    apxinf_runtime_t runtime, const apxinf_conv2d_spec_t* spec,
    const apxinf_conv2d_bindings_t* bindings);

#ifdef __cplusplus
}
#endif
