#pragma once

#include "gemm_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct apxinf_gemm_execution* apxinf_gemm_execution_t;

apxinf_status_t apxinf_gemm_prepare(
    apxinf_runtime_t runtime, const apxinf_gemm_spec_t* spec,
    const apxinf_gemm_policy_t* policy,
    const apxinf_gemm_tuning_bindings_t* tuning_bindings,
    apxinf_gemm_execution_t* execution);
apxinf_status_t apxinf_gemm_enqueue(apxinf_gemm_execution_t execution);
void apxinf_gemm_destroy(apxinf_gemm_execution_t execution);
const char* apxinf_gemm_summary(apxinf_gemm_execution_t execution);
uint64_t apxinf_gemm_execution_weight_prepack_count(
    apxinf_gemm_execution_t execution);

#ifdef __cplusplus
}
#endif
