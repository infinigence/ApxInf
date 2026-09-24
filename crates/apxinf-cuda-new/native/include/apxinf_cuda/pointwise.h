#pragma once

#include "pointwise_types.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_pointwise_launch(
    apxinf_runtime_t runtime, const apxinf_pointwise_spec_t* spec,
    const apxinf_pointwise_bindings_t* bindings);

#ifdef __cplusplus
}
#endif
