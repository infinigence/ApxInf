// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

namespace apxinf::cuda::attn_ops {

// Partial rotary embedding over `[tokens, heads, head_dim]` BF16, in place.
//
// Only the first `rotary_dim` elements of each head rotate
// (partial_rotary_factor 0.25 over head_dim 256 leaves 192 untouched).
// `positions` is `[tokens]` int32.
//
// This is the text-only form: with a single position id per token, the
// config's mrope_section [11, 11, 10] split rotates every section by the same
// angle, so it collapses to plain partial RoPE. It stops being correct once
// image or video tokens carry distinct temporal/height/width positions.
int partial_rope(void* data, const void* positions, int tokens, int heads,
                 int head_dim, int rotary_dim, float theta,
                 cudaStream_t stream);

// Per-head RMSNorm in place over `[rows, head_dim]`, for q_norm and k_norm.
int head_rms_norm(void* data, const void* weight, int rows, int head_dim,
                  float epsilon, cudaStream_t stream);

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

}  // namespace apxinf::cuda::attn_ops
