#pragma once

#include "relative_attention_types.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_relative_attention_launch(
    apxinf_runtime_t runtime, const apxinf_relative_attention_spec_t* spec,
    const apxinf_relative_attention_bindings_t* bindings);

#ifdef __cplusplus
}
#endif
