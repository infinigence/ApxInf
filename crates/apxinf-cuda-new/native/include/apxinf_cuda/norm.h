#pragma once

#include "norm_types.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_norm_launch(
    apxinf_runtime_t runtime, const apxinf_norm_spec_t* spec,
    const apxinf_norm_bindings_t* bindings);

#ifdef __cplusplus
}
#endif
