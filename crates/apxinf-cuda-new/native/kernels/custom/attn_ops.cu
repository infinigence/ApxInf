// Copyright 2026 ApxInf contributors.
//
// Full-attention primitives for Qwen3.5: partial rotary embedding, per-head
// query/key normalization, and the sigmoid output gate.
//
// Two scope notes.
//
// mRoPE: the config specifies mrope_section [11, 11, 10] over the 32 rotary
// pairs, splitting them across temporal, height and width position ids. For
// text-only input all three ids are the token position, so every section
// rotates by the same angle and the result is identical to plain partial
// RoPE. That equivalence is exactly why this kernel is correct for text and
// exactly why it stops being correct once image or video tokens appear -- at
// that point the sections carry different positions and must be applied
// separately.
//
// Pairing convention: this implements the rotate-half form, where pair i is
// (x[i], x[i + rotary_dim/2]). The alternative interleaved form pairs
// (x[2i], x[2i+1]). Both run and both produce plausible logits, so this
// choice must be confirmed against a reference engine before any accuracy
// claim.

#include "attn_ops.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace apxinf::cuda::attn_ops {
namespace {

// One block per (token, head). Only the first `rotary_dim` elements rotate;
// partial_rotary_factor 0.25 over head_dim 256 leaves 192 untouched.
__global__ void partial_rope_kernel(__nv_bfloat16* __restrict__ data,
                                    const int32_t* __restrict__ positions,
                                    int tokens, int heads, int head_dim,
                                    int rotary_dim, float theta) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  if (token >= tokens || head >= heads) return;

  const int half = rotary_dim / 2;
  const long long base =
      ((long long)token * heads + head) * head_dim;
  const float position = static_cast<float>(positions[token]);

  for (int index = threadIdx.x; index < half; index += blockDim.x) {
    const float inverse_frequency =
        __powf(theta, -2.0f * static_cast<float>(index) /
                          static_cast<float>(rotary_dim));
    const float angle = position * inverse_frequency;
    float sine;
    float cosine;
    __sincosf(angle, &sine, &cosine);

    const float low = __bfloat162float(data[base + index]);
    const float high = __bfloat162float(data[base + half + index]);
    data[base + index] = __float2bfloat16(low * cosine - high * sine);
    data[base + half + index] = __float2bfloat16(high * cosine + low * sine);
  }
}

// Per-head RMSNorm without a gate, for q_norm and k_norm.
__global__ void head_rms_norm_kernel(__nv_bfloat16* __restrict__ data,
                                     const __nv_bfloat16* __restrict__ weight,
                                     int rows, int head_dim, float epsilon) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const long long base = (long long)row * head_dim;

  float sum = 0.0f;
  for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
    const float value = __bfloat162float(data[base + index]);
    sum += value * value;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
  }
  __shared__ float partial[32];
  const int warp = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) partial[warp] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    const int warps = (blockDim.x + 31) / 32;
    float total = 0.0f;
    for (int index = 0; index < warps; ++index) total += partial[index];
    partial[0] = total;
  }
  __syncthreads();

  const float scale = rsqrtf(partial[0] / static_cast<float>(head_dim) + epsilon);
  for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
    data[base + index] = __float2bfloat16(
        __bfloat162float(data[base + index]) * scale *
        (1.0f + __bfloat162float(weight[index])));
  }
}

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

int partial_rope(void* data, const void* positions, int tokens, int heads,
                 int head_dim, int rotary_dim, float theta,
                 cudaStream_t stream) {
  if (tokens <= 0 || heads <= 0 || head_dim <= 0) return -1;
  if (rotary_dim <= 0 || rotary_dim > head_dim || rotary_dim % 2 != 0) return -2;
  const int half = rotary_dim / 2;
  const int threads = half >= 128 ? 128 : ((half + 31) / 32) * 32;
  partial_rope_kernel<<<dim3(tokens, heads), threads, 0, stream>>>(
      static_cast<__nv_bfloat16*>(data),
      static_cast<const int32_t*>(positions), tokens, heads, head_dim,
      rotary_dim, theta);
  return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

int head_rms_norm(void* data, const void* weight, int rows, int head_dim,
                  float epsilon, cudaStream_t stream) {
  if (rows <= 0 || head_dim <= 0) return -1;
  const int threads = head_dim >= 256 ? 256 : ((head_dim + 31) / 32) * 32;
  head_rms_norm_kernel<<<rows, threads, 0, stream>>>(
      static_cast<__nv_bfloat16*>(data),
      static_cast<const __nv_bfloat16*>(weight), rows, head_dim, epsilon);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

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

}  // namespace apxinf::cuda::attn_ops
