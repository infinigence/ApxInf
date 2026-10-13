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

// Qwen3.5's `attn_output_gate: true` pair: the strided split out of q_proj's
// doubled head slots, and the sigmoid the attention output passes through
// before o_proj. One implementation each and nothing to measure, so they take
// flat parameters rather than joining prepare/enqueue.
// Kernels in kernels/custom/attention_gates.cu.
apxinf_status_t apxinf_attention_split_query_and_gate(
    const void* fused, void* query, void* gate, int64_t tokens, int64_t heads,
    int64_t head_dim, apxinf_cuda_stream_t stream);

// data *= sigmoid(gate). Not silu: see kernels/custom/attention_gates.h.
apxinf_status_t apxinf_attention_apply_output_gate(void* data, const void* gate,
                                              int64_t count,
                                              apxinf_cuda_stream_t stream);
apxinf_status_t apxinf_decode_attention(
    apxinf_runtime_t runtime, const void* query, const void* key_cache,
    const void* value_cache, void* output, void* workspace,
    int64_t workspace_bytes, int64_t key_tokens, float scale,
    apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
