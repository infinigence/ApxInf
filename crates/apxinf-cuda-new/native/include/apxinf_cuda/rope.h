#pragma once

#include "rope_types.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_rope_launch(
    apxinf_runtime_t runtime, const apxinf_rope_spec_t* spec,
    const apxinf_rope_bindings_t* bindings);

#ifdef __cplusplus
}
#endif
