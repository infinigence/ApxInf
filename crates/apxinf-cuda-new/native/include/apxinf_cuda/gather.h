#pragma once

#include "gather_types.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_gather_launch(
    apxinf_runtime_t runtime, const apxinf_gather_spec_t* spec,
    const apxinf_gather_bindings_t* bindings);

#ifdef __cplusplus
}
#endif
