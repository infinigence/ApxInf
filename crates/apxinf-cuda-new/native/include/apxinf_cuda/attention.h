#pragma once

#include "attention_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct apxinf_attention_execution* apxinf_attention_execution_t;

apxinf_status_t apxinf_attention_prepare(
    apxinf_runtime_t runtime, const apxinf_attention_spec_t* spec,
    const apxinf_attention_policy_t* policy,
    const apxinf_attention_bindings_t* bindings,
    apxinf_attention_execution_t* execution);
apxinf_status_t apxinf_attention_enqueue(apxinf_attention_execution_t execution);
void apxinf_attention_destroy(apxinf_attention_execution_t execution);
const char* apxinf_attention_summary(apxinf_attention_execution_t execution);

apxinf_status_t apxinf_attention_test_validate_candidates(
    apxinf_runtime_t runtime, const apxinf_attention_spec_t* spec,
    const apxinf_attention_policy_t* policy,
    const apxinf_attention_bindings_t* bindings,
    const float* expected_output, uint64_t expected_output_len);

#ifdef __cplusplus
}
#endif
