// Copyright 2026 ApxInf contributors.
//
// The rope family's in-place partial rotary embedding.
//
// Moved verbatim out of kernels/custom/attn_ops.cu, which was named after a
// call site rather than a mathematical family (doc/cuda-new-design.md section
// 1.3). The kernel, the host launch and the two scope notes below are that
// code unchanged apart from the enclosing namespace.
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

#include "rope_partial.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace apxinf::cuda_new::rope_partial {
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

}  // namespace apxinf::cuda_new::rope_partial
