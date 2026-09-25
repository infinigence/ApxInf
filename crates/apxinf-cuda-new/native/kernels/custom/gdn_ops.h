// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

namespace apxinf::cuda::gdn_ops {

// Gated DeltaNet single-token recurrent step.
//
// `state` is [v_heads, v_dim, k_dim] f32 and is updated in place. `q` and `k`
// are [k_heads, k_dim] BF16; `v` is [v_heads, v_dim] BF16. Value head h reads
// k-head `h / (v_heads / k_heads)`, the GQA-style sharing this architecture
// uses (48 value heads over 16 key heads).
//
// `decay` holds log decay per value head (<= 0) and `beta` the delta-rule
// gate; both f32, produced by gdn_decay_and_beta.
int gdn_recurrent_step(void* state, const void* q, const void* k,
                       const void* v, const void* decay, const void* beta,
                       void* output, int v_heads, int k_heads, int v_dim,
                       int k_dim, cudaStream_t stream);

// Per-head RMSNorm followed by the swish output gate.
//
// Qwen3.5 sets `output_gate_type: swish`, so the gate is silu(z); a
// sigmoid-gated implementation would be silently wrong here.
int gdn_gated_norm(const void* input, const void* gate, const void* weight,
                   void* output, int heads, int head_dim, float epsilon,
                   cudaStream_t stream);

// Causal depthwise conv1d advanced by one token, then SiLU.
//
// `window` is [channels, kernel_width] f32 recurrent state holding the last
// `kernel_width` inputs per channel.
int gdn_causal_conv_step(void* window, const void* input, const void* weight,
                         void* output, int channels, int kernel_width,
                         cudaStream_t stream);

// Causal depthwise conv1d over a whole prompt, then SiLU.
//
// `input` and `output` are [tokens, channels] BF16 (time-major, which is why
// this cannot reuse Dao-AILab/causal-conv1d's time-vectorized parallelization
// -- see the kernel comment). `weight` is [channels, kernel_width] BF16.
//
// `window` may be null. When it is not, the last `kernel_width` inputs are
// written back in the [channels, kernel_width] f32 layout the single-token
// step expects, so decode can continue straight after a prompt.
int gdn_causal_conv_forward(const void* input, const void* weight,
                            void* output, void* window, int tokens,
                            int channels, int kernel_width,
                            cudaStream_t stream);

// Convert one prompt's GDN projection into what the FlashInfer prefill
// kernel expects, in a single pass.
//
// `fused` is [tokens, row_width] BF16 with q, k and v along each row. Writes
// q/k L2-normalized and narrowed to FP16, v narrowed, and `alpha = exp(g)`.
// q is left unscaled -- that kernel takes the scale as an argument.
// Widen an FP16 buffer to BF16. The FlashInfer scan emits FP16 and the rest
// of this model is BF16.
int gdn_widen_f16_to_bf16(const void* input, void* output, long long count,
                          cudaStream_t stream);

int gdn_prepare_flashinfer(const void* fused, void* q_out, void* k_out,
                           void* v_out, const void* g, void* alpha, int tokens,
                           int row_width, int k_heads, int v_heads, int dim,
                           float epsilon, cudaStream_t stream);

// L2-normalize each head in place; the delta rule needs unit-norm q and k.
int gdn_l2_normalize_heads(void* data, int heads, int head_dim, float epsilon,
                           cudaStream_t stream);

// decay = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b).
int gdn_decay_and_beta(const void* a, const void* b, const void* a_log,
                       const void* dt_bias, void* decay, void* beta, int heads,
                       cudaStream_t stream);

// Sequence-axis decay and beta. `a` and `b` are [tokens, heads] BF16; `a_log`
// and `dt_bias` stay per-head [heads]; `decay` and `beta` are [tokens, heads]
// f32.
int gdn_decay_and_beta_seq(const void* a, const void* b, const void* a_log,
                           const void* dt_bias, void* decay, void* beta,
                           int tokens, int heads, cudaStream_t stream);

// Sequence-axis gated norm. Tensors are [tokens, heads, head_dim] BF16;
// `weight` stays [head_dim].
int gdn_gated_norm_seq(const void* input, const void* gate, const void* weight,
                       void* output, int tokens, int heads, int head_dim,
                       float epsilon, cudaStream_t stream);


// Gated DeltaNet chunked scan (parallel prefill). Faithful port of
// torch_chunk_gated_delta_rule (forward-substitution export path). One block
// per value head; chunk loop sequential inside. q/k are L2-normalized in fp32
// and q scaled by k_dim**-0.5 inside the kernel. State is carried in the PORT
// layout [v_heads, v_dim, k_dim] f32 so prefill leaves exactly the state the
// single-token recurrent step expects.
//
//   q,k  : [seq_padded, k_heads, k_dim]  bf16 (post-conv, pre-l2norm)
//   v    : [seq_padded, v_heads, k_dim]  bf16
//   g,beta: [seq_padded, v_heads]        f32  (log decay, delta gate)
//   out  : [seq_padded, v_heads, k_dim]  bf16 (core_attn_out)
//   state: [v_heads, v_dim, k_dim]       f32  in/out//
// The three row strides give the distance in elements between consecutive
// tokens of q, k and v. Contiguous [seq, heads, dim] inputs pass
// heads*dim; a caller holding q, k and v interleaved in one projection row
// passes the full row width instead and points each at its own offset, which
// is how prefill feeds the fused conv output without copying it apart.
int gdn_chunk_scan(const void* q, const void* k, const void* v, const void* g,
                   const void* beta, void* out, void* state, int seq_padded,
                   int v_heads, int k_heads, int chunk_size, int k_dim,
                   int num_chunks, int q_row_stride, int k_row_stride,
                   int v_row_stride, cudaStream_t stream);

}  // namespace apxinf::cuda::gdn_ops
