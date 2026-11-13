#pragma once
#include "volume_types.h"
#ifdef __cplusplus
extern "C" {
#endif
apxinf_status_t apxinf_unfold3d_prepare(apxinf_runtime_t runtime,
    const apxinf_unfold3d_spec_t* spec, const apxinf_unfold3d_bindings_t* bindings,
    void** execution);
apxinf_status_t apxinf_unfold3d_enqueue(void* execution);
void apxinf_unfold3d_destroy(void* execution);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif
apxinf_status_t apxinf_conv3d_prepare(apxinf_runtime_t runtime,
    const apxinf_conv3d_spec_t* spec, const apxinf_conv3d_policy_t* policy,
    const apxinf_conv3d_bindings_t* bindings, void** execution);
apxinf_status_t apxinf_conv3d_enqueue(void* execution);
void apxinf_conv3d_destroy(void* execution);
const char* apxinf_conv3d_summary(void* execution);
#ifdef __cplusplus
}
#endif
