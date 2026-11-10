// Qwen3-VL RoPE variants for `apxinf_hip::native`: the text stack's mRoPE and
// (as they land) the vision tower's 2D rotation.
//
// Ports of the cuda-new kernels GR00T runs on CUDA
// (`apxinf-cuda-new/native/kernels/custom/rope.cuh`); the arithmetic, its
// order and the rounding points are theirs. BF16 storage, F32 math, one
// rounding per stored value.
//
// Unlike cuda-new's in-place `partial_multimodal_rope`, these take an explicit
// output so a caller can keep the input (the native layer's "outputs are
// passed in" rule). Passing the same pointer for both is still correct: a
// thread reads both halves of its own pair before storing either.

#include "common.h"

namespace {

using namespace apxinf_hip;
using bf16 = uint16_t;

__device__ __forceinline__ float ld(const bf16* p, int64_t i) { return bf16_to_f32(p[i]); }

// ── Text mRoPE ──────────────────────────────────────────────────────

// Which position axis frequency pair `pair` reads: T (0), H (1) or W (2).
// The axis cycles with `pair % 3` while `pair` stays inside that section's
// span, so the three sections interleave rather than occupy contiguous
// ranges -- this is Qwen3-VL's `mrope_interleaved` layout.
__device__ __forceinline__ int multimodal_axis(int pair, int section_h, int section_w) {
  const int remainder = pair % 3;
  if (remainder == 1 && pair < section_h * 3) return 1;
  if (remainder == 2 && pair < section_w * 3) return 2;
  return 0;
}

// Rotate the half-split pair (p, half + p) of every head by
// `position[axis(p)] * theta^(-2p/rotary_dim)`. `positions` is [tokens, 3],
// laid out as (t, h, w) per token. Columns at or past `rotary_dim` are copied
// through unrotated, so a partial rotation still fills the whole output.
// Grid (tokens, heads); threads loop over the head's pairs.
__global__ void apply_mrope_kernel(const bf16* source, bf16* destination,
                                   const uint32_t* positions, int tokens, int heads, int head_dim,
                                   int rotary_dim, int section_h, int section_w, float theta) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  if (token >= tokens || head >= heads) return;
  const int half = rotary_dim / 2;
  const int64_t base = (static_cast<int64_t>(token) * heads + head) * head_dim;
  for (int pair = threadIdx.x; pair < half; pair += blockDim.x) {
    const int axis = multimodal_axis(pair, section_h, section_w);
    const float position = static_cast<float>(positions[token * 3 + axis]);
    const float frequency = powf(theta, -2.0f * static_cast<float>(pair) / rotary_dim);
    float sine, cosine;
    sincosf(position * frequency, &sine, &cosine);
    const float low = ld(source, base + pair);
    const float high = ld(source, base + half + pair);
    destination[base + pair] = f32_to_bf16(low * cosine - high * sine);
    destination[base + half + pair] = f32_to_bf16(high * cosine + low * sine);
  }
  // Untouched tail of a partial rotation; a no-op when rotary_dim == head_dim.
  for (int column = rotary_dim + threadIdx.x; column < head_dim; column += blockDim.x) {
    destination[base + column] = source[base + column];
  }
}

// ── Vision 2D (axial) RoPE ──────────────────────────────────────────

// Qwen3-VL vision tower rotation. `positions` is [tokens, 2], laid out as
// (h, w) per token: the first quarter of the half-split pairs rotate by the
// height position and the rest by the width one, each at frequency
// `theta^(-2 * pair_within_axis / half)`.
//
// Composed rather than fused with the QKV split: `split_qkv_bias` already
// rounds the biased projection to BF16, which is exactly the rounding
// cuda-new's fused `split_qkv_axial_rope` performs before it rotates, so the
// two-call sequence and the fused kernel round identically.
// Grid (tokens, heads); threads loop over the head's pairs.
__global__ void apply_vision_2d_kernel(const bf16* source, bf16* destination,
                                       const uint32_t* positions, int tokens, int heads,
                                       int head_dim, float theta) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  if (token >= tokens || head >= heads) return;
  const int half = head_dim / 2;
  const int quarter = half / 2;
  const int64_t base = (static_cast<int64_t>(token) * heads + head) * head_dim;
  for (int pair = threadIdx.x; pair < half; pair += blockDim.x) {
    const int axis = pair < quarter ? 0 : 1;
    const int within = pair < quarter ? pair : pair - quarter;
    const float position = static_cast<float>(positions[token * 2 + axis]);
    const float frequency = powf(theta, -2.0f * static_cast<float>(within) / half);
    float sine, cosine;
    sincosf(position * frequency, &sine, &cosine);
    const float low = ld(source, base + pair);
    const float high = ld(source, base + half + pair);
    destination[base + pair] = f32_to_bf16(low * cosine - high * sine);
    destination[base + half + pair] = f32_to_bf16(low * sine + high * cosine);
  }
}

const bf16* in(const void* p) { return static_cast<const bf16*>(p); }
bf16* out(void* p) { return static_cast<bf16*>(p); }

}  // namespace

extern "C" int apxinf_hip_native_apply_mrope(const void* input, void* output,
                                             const void* positions, int tokens, int heads,
                                             int head_dim, int rotary_dim, int section_h,
                                             int section_w, float theta, void* stream) {
  if (tokens <= 0 || heads <= 0 || head_dim <= 0 || rotary_dim <= 0 || rotary_dim > head_dim ||
      rotary_dim % 2 != 0 || section_h < 0 || section_w < 0) {
    return kInvalidArgument;
  }
  const int half = rotary_dim / 2;
  const int block = half >= kAttentionBlock ? kAttentionBlock : ((half + 31) / 32) * 32;
  apply_mrope_kernel<<<dim3(tokens, heads), block, 0, as_stream(stream)>>>(
      in(input), out(output), static_cast<const uint32_t*>(positions), tokens, heads, head_dim,
      rotary_dim, section_h, section_w, theta);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_apply_vision_2d(const void* input, void* output,
                                                 const void* positions, int tokens, int heads,
                                                 int head_dim, float theta, void* stream) {
  // head_dim % 4 keeps the half-split pairs evenly divisible between the two
  // axes, which the per-axis frequency indexing assumes.
  if (tokens <= 0 || heads <= 0 || head_dim <= 0 || head_dim % 4 != 0) {
    return kInvalidArgument;
  }
  const int half = head_dim / 2;
  const int block = half >= kAttentionBlock ? kAttentionBlock : ((half + 31) / 32) * 32;
  apply_vision_2d_kernel<<<dim3(tokens, heads), block, 0, as_stream(stream)>>>(
      in(input), out(output), static_cast<const uint32_t*>(positions), tokens, heads, head_dim,
      theta);
  return last_launch_status();
}
