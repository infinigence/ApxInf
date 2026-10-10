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

// Allocation-free Qwen3.8 decode fast path. The caller owns one reusable
// workspace and may enqueue consecutive attention layers on the same stream.
int64_t apxinf_decode_attention_workspace_bytes(void);
apxinf_status_t apxinf_decode_attention(
    apxinf_runtime_t runtime, const void* query, const void* key_cache,
    const void* value_cache, void* output, void* workspace,
    int64_t workspace_bytes, int64_t key_tokens, float scale,
    apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
