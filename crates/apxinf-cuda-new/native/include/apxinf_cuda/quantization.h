#pragma once

#include "quantization_types.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_quantization_launch(
    apxinf_runtime_t runtime, const apxinf_quantization_spec_t* spec,
    const apxinf_quantization_bindings_t* bindings);

#ifdef __cplusplus
}
#endif
