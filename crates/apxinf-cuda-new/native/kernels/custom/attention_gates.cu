// Copyright 2026 ApxInf contributors.

#include "attention_gates.h"

#include <cuda_bf16.h>

namespace apxinf::cuda_new::attention {
namespace {
// q_proj emits [tokens, 2 * heads * head_dim]: the query in the first half of
// each head's slot and its gate in the second (`attn_output_gate: true`).
// See apply_output_gate_kernel for what is done with that gate.
__global__ void split_query_and_gate_kernel(
    const __nv_bfloat16* __restrict__ fused, __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ gate, int tokens, int heads, int head_dim) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long total = (long long)tokens * heads * head_dim;
  if (index >= total) return;
  const int element = static_cast<int>(index % head_dim);
  const long long head_index = index / head_dim;
  const int head = static_cast<int>(head_index % heads);
  const int token = static_cast<int>(head_index / heads);

  const long long slot =
      ((long long)token * heads + head) * 2 * head_dim + element;
  query[index] = fused[slot];
  gate[index] = fused[slot + head_dim];
}

// data *= sigmoid(gate), elementwise.
//
// `config.json` says `output_gate_type: "swish"`, which reads as silu. It is
// not what the model does. `grep -rn output_gate` over the reference
// implementation finds nothing -- transformers never reads that key -- and
// Qwen3_5Attention.forward (modeling_qwen3_5.py:818) is literally
//
//     attn_output = attn_output * torch.sigmoid(gate)
//
// with no extra factor of `gate`. Where the two official artifacts disagree,
// the code that produced the checkpoint's behaviour wins. Do not "fix" this
// back to silu on the strength of the config key.
//
// The GDN output gate *is* silu (Qwen3_5RMSNormGated sets activation="silu");
// that one lives in gdn_ops.cu and is correct as written.
__global__ void apply_output_gate_kernel(__nv_bfloat16* __restrict__ data,
                                         const __nv_bfloat16* __restrict__ gate,
                                         long long count) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (index >= count) return;
  const float z = __bfloat162float(gate[index]);
  data[index] =
      __float2bfloat16(__bfloat162float(data[index]) / (1.0f + __expf(-z)));
}

}  // namespace

int split_query_and_gate(const void* fused, void* query, void* gate,
                         int tokens, int heads, int head_dim,
                         cudaStream_t stream) {
  if (tokens <= 0 || heads <= 0 || head_dim <= 0) return -1;
  const long long total = (long long)tokens * heads * head_dim;
  const int threads = 256;
  const long long blocks = (total + threads - 1) / threads;
  split_query_and_gate_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(fused),
      static_cast<__nv_bfloat16*>(query), static_cast<__nv_bfloat16*>(gate),
      tokens, heads, head_dim);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int apply_output_gate(void* data, const void* gate, long long count,
                      cudaStream_t stream) {
  if (count <= 0) return -1;
  const int threads = 256;
  const long long blocks = (count + threads - 1) / threads;
  apply_output_gate_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      static_cast<__nv_bfloat16*>(data),
      static_cast<const __nv_bfloat16*>(gate), count);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

}  // namespace apxinf::cuda_new::attention
