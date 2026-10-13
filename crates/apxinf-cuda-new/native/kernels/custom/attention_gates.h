// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

namespace apxinf::cuda_new::attention_gates {

// Qwen3.5's `attn_output_gate: true` pair. Both are attention-block layout and
// gating steps rather than members of any other family: the split is a strided
// copy out of q_proj's doubled head slots, and the gate is the sigmoid the
// attention output passes through before o_proj.
//
// They are not part of the tunable attention family's prepare/enqueue: one
// implementation each, nothing to measure, so they take flat parameters.

// Split q_proj's `[tokens, heads, 2 * head_dim]` output into the query and its
// gate. `attn_output_gate: true` in this architecture, so q_proj is twice as
// wide as the query it produces.
int split_query_and_gate(const void* fused, void* query, void* gate,
                         int tokens, int heads, int head_dim,
                         cudaStream_t stream);

// data *= sigmoid(gate), elementwise -- not silu, despite `config.json`
// saying `output_gate_type: "swish"`. Nothing in the reference implementation
// reads that key; Qwen3_5Attention.forward does
// `attn_output * torch.sigmoid(gate)` (modeling_qwen3_5.py:818). The GDN
// output gate is the silu one; it lives in gdn_ops.h.
int apply_output_gate(void* data, const void* gate, long long count,
                      cudaStream_t stream);

}  // namespace apxinf::cuda_new::attention_gates
