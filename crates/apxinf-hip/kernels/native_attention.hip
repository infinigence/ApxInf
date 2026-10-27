// Dense non-causal attention for `apxinf_hip::native`.
//
// Every pi05 attention is non-causal (cuda-new's `AttentionMask::None`):
// vision attends within each view, the prefix bidirectionally over itself, and
// the action expert's queries over the whole prefix plus every action row.
// One kernel covers all three:
//
//   q   [batch, q_tokens, q_heads, head_dim]   contiguous
//   k,v batch b, token j, KV head g at  b * kv_batch_stride + j * kv_token_stride + g * head_dim
//   out [batch, q_tokens, q_heads, head_dim]
//
// with query head h reading KV head h / (q_heads / kv_heads). The strides let
// the action expert read a prefix cache that holds more rows than it attends.
//
// One block per (query token, query head, batch); a single pass with an online
// softmax, so no score matrix is materialized. Scores, probabilities and the PV
// sum stay F32 and the output is rounded once. This is the correctness
// kernel; the tuned kernels are milestone C.

#include "common.h"

namespace {

using namespace apxinf_hip;
using bf16 = uint16_t;

// Dynamic shared memory: q and acc (head_dim floats each), then the tile's
// probabilities and a reduction buffer (blockDim floats each).
__global__ void attention_kernel(const bf16* q, const bf16* k, const bf16* v, bf16* out,
                                 int q_tokens, int q_heads, int kv_heads, int head_dim,
                                 int kv_tokens, int64_t kv_batch_stride, int64_t kv_token_stride,
                                 float scale) {
  extern __shared__ float smem[];
  float* q_s = smem;
  float* acc = q_s + head_dim;
  float* p_s = acc + head_dim;
  float* red = p_s + blockDim.x;

  const int i = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int tile = blockDim.x;
  const int64_t kv_base = static_cast<int64_t>(b) * kv_batch_stride +
                          static_cast<int64_t>(h / (q_heads / kv_heads)) * head_dim;
  const int64_t q_base = ((static_cast<int64_t>(b) * q_tokens + i) * q_heads + h) * head_dim;

  for (int d = tid; d < head_dim; d += tile) {
    q_s[d] = bf16_to_f32(q[q_base + d]);
    acc[d] = 0.0f;
  }
  __syncthreads();

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  for (int t0 = 0; t0 < kv_tokens; t0 += tile) {
    const int j = t0 + tid;
    float score = -INFINITY;
    if (j < kv_tokens) {
      const bf16* key = k + kv_base + j * kv_token_stride;
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) dot += q_s[d] * bf16_to_f32(key[d]);
      score = dot * scale;
    }

    red[tid] = score;
    __syncthreads();
    for (int stride = tile / 2; stride > 0; stride >>= 1) {
      if (tid < stride) red[tid] = fmaxf(red[tid], red[tid + stride]);
      __syncthreads();
    }
    // At least one key in every tile is in range, so the maximum is finite.
    const float tile_max = fmaxf(running_max, red[0]);
    __syncthreads();

    const float p = j < kv_tokens ? expf(score - tile_max) : 0.0f;
    p_s[tid] = p;
    red[tid] = p;
    __syncthreads();
    for (int stride = tile / 2; stride > 0; stride >>= 1) {
      if (tid < stride) red[tid] += red[tid + stride];
      __syncthreads();
    }
    // exp(-inf) is 0 on the first tile, which zeroes the empty accumulator.
    const float correction = expf(running_max - tile_max);
    running_sum = running_sum * correction + red[0];

    const int in_tile = min(tile, kv_tokens - t0);
    for (int d = tid; d < head_dim; d += tile) {
      float a = acc[d] * correction;
      for (int t = 0; t < in_tile; ++t) {
        a += p_s[t] * bf16_to_f32(v[kv_base + (t0 + t) * kv_token_stride + d]);
      }
      acc[d] = a;
    }
    running_max = tile_max;
    __syncthreads();
  }

  const float inverse_sum = 1.0f / running_sum;
  for (int d = tid; d < head_dim; d += tile) {
    out[q_base + d] = f32_to_bf16(acc[d] * inverse_sum);
  }
}

}  // namespace

extern "C" int apxinf_hip_native_attention(const void* q, const void* k, const void* v, void* out,
                                           int batch, int q_tokens, int q_heads, int kv_heads,
                                           int head_dim, int kv_tokens, int64_t kv_batch_stride,
                                           int64_t kv_token_stride, float scale, void* stream) {
  if (batch <= 0 || q_tokens <= 0 || q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0 ||
      head_dim <= 0 || kv_tokens <= 0 || kv_token_stride < static_cast<int64_t>(kv_heads) * head_dim ||
      batch > 65535 || q_heads > 65535) {
    return kInvalidArgument;
  }
  const size_t shared =
      (2 * static_cast<size_t>(head_dim) + 2 * static_cast<size_t>(kAttentionBlock)) * sizeof(float);
  attention_kernel<<<dim3(q_tokens, q_heads, batch), kAttentionBlock, shared, as_stream(stream)>>>(
      static_cast<const bf16*>(q), static_cast<const bf16*>(k), static_cast<const bf16*>(v),
      static_cast<bf16*>(out), q_tokens, q_heads, kv_heads, head_dim, kv_tokens, kv_batch_stride,
      kv_token_stride, scale);
  return last_launch_status();
}
