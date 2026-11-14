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
// The tower's QKV path now runs the fused `split_qkv_axial_rope_kernel`
// (elementwise.hip); this rotation-only kernel is that fusion's acceptance
// reference: composed with `split_qkv_bias`, it rounds the biased projection
// to BF16 the same way, so the two agree bit for bit.
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

// Per-head RMSNorm over flat [tokens * heads, head_dim] Q and K, followed
// directly by the text stack's mRoPE -- cuda-new's fused
// `rms_norm_apply_mrope_qk`, one kernel instead of rms_norm + apply_mrope.
//
// The normalization is `rms_norm_kernel`'s arithmetic (apxinf_hip.hip) under
// the same launch geometry -- one kReduceBlock block per row, the same
// strided square-sum and the same reduction tree -- and its result is
// rounded to BF16 into shared memory before the rotation: the rounding the
// separate rms_norm store performed, which apply_mrope then read back. The
// rotation expressions are `apply_mrope_kernel`'s, pinned with explicit FMAs
// to the forms that kernel compiles to; leaving the contraction to the
// compiler drifted 1 ulp against the reference between kernels before (see
// split_qkv_axial_rope_kernel in elementwise.hip).
//
// Grid (tokens, q_heads + kv_heads): the block normalizes its head first,
// then threads below `half` rotate one pair each.
__global__ void rms_norm_mrope_qk_kernel(const bf16* q, const bf16* q_weight, bf16* q_out,
                                         const bf16* k, const bf16* k_weight, bf16* k_out,
                                         const uint32_t* positions, int q_heads, int kv_heads,
                                         int head_dim, float eps, float theta, int section_h,
                                         int section_w) {
  __shared__ float partial[kReduceBlock];
  extern __shared__ bf16 rounded[];  // this head, normalized
  const int token = blockIdx.x;
  const bool is_query = blockIdx.y < q_heads;
  const int heads = is_query ? q_heads : kv_heads;
  const int head = is_query ? blockIdx.y : blockIdx.y - q_heads;
  const int64_t base = (static_cast<int64_t>(token) * heads + head) * head_dim;
  const bf16* input = is_query ? q : k;
  const bf16* weight = is_query ? q_weight : k_weight;
  bf16* output = is_query ? q_out : k_out;

  float square_sum = 0.0f;
  for (int c = threadIdx.x; c < head_dim; c += blockDim.x) {
    const float v = ld(input, base + c);
    square_sum += v * v;
  }
  const float inverse_rms =
      rsqrtf(block_sum(square_sum, partial) / static_cast<float>(head_dim) + eps);
  for (int c = threadIdx.x; c < head_dim; c += blockDim.x) {
    rounded[c] = f32_to_bf16(ld(input, base + c) * inverse_rms * ld(weight, c));
  }
  __syncthreads();

  const int half = head_dim / 2;
  const int pair = threadIdx.x;
  if (pair >= half) return;
  const int axis = multimodal_axis(pair, section_h, section_w);
  const float position = static_cast<float>(positions[token * 3 + axis]);
  const float frequency = powf(theta, -2.0f * static_cast<float>(pair) / head_dim);
  float sine, cosine;
  sincosf(position * frequency, &sine, &cosine);
  const float low = ld(rounded, pair);
  const float high = ld(rounded, half + pair);
  // Pinned to apply_mrope_kernel's compiled contraction: the sine product
  // rounds on its own and the cosine product fuses into the add/sub, in both
  // stores.
  output[base + pair] = f32_to_bf16(__fmaf_rn(low, cosine, -(high * sine)));
  output[base + half + pair] = f32_to_bf16(__fmaf_rn(high, cosine, low * sine));
}

// Vectorized variant for head_dim == 32 * EPL (EPL in {2, 4, 8}: head_dim
// 64 / 128 / 256). One warp per (token, head) instead of a 256-thread block:
// lane l owns elements [EPL*l, EPL*l + EPL), so a lane touches EPL/2 uint32
// words of x, weight and y (16-byte-aligned rows: head_dim * 2 B is a
// multiple of 16 for every EPL). The half-split rope pair (i, i + half) sits
// in lanes l and l ^ 16 -- half = 16 * EPL -- so the partner value arrives
// with one __shfl_xor instead of the shared-memory round trip, and the sum of
// squares reduces with warp shuffles instead of the block-wide tree + two
// barriers. The 256-thread block on a 128-wide head was mostly idle lanes
// waiting on barriers; the arithmetic per element (including the pinned FMA
// contraction forms above and the one-rounding normalized value) is unchanged.
// The reduction partitioning differs from the block tree, so the inverse_rms
// can differ by 1 ulp -- the norm.hip vec kernels set that precedent.
template <int EPL>
__global__ void rms_norm_mrope_qk_vec_kernel(const bf16* q, const bf16* q_weight, bf16* q_out,
                                             const bf16* k, const bf16* k_weight, bf16* k_out,
                                             const uint32_t* positions, int q_heads, int kv_heads,
                                             int head_dim, float eps, float theta, int section_h,
                                             int section_w) {
  const int token = blockIdx.x;
  const bool is_query = blockIdx.y < q_heads;
  const int heads = is_query ? q_heads : kv_heads;
  const int head = is_query ? blockIdx.y : blockIdx.y - q_heads;
  const int64_t base = (static_cast<int64_t>(token) * heads + head) * head_dim;
  const bf16* input = is_query ? q : k;
  const bf16* weight = is_query ? q_weight : k_weight;
  bf16* output = is_query ? q_out : k_out;
  const int lane = threadIdx.x;

  // Load the lane's EPL elements and its weight values into registers.
  const uint32_t* xw = reinterpret_cast<const uint32_t*>(input + base);
  const uint32_t* ww = reinterpret_cast<const uint32_t*>(weight);
  uint32_t xq[EPL / 2], wq[EPL / 2];
#pragma unroll
  for (int w = 0; w < EPL / 2; ++w) xq[w] = xw[lane * (EPL / 2) + w];
#pragma unroll
  for (int w = 0; w < EPL / 2; ++w) wq[w] = ww[lane * (EPL / 2) + w];

  float square_sum = 0.0f;
#pragma unroll
  for (int j = 0; j < EPL; ++j) {
    const float v = bf16_to_f32(static_cast<uint16_t>(xq[j >> 1] >> ((j & 1) * 16)));
    square_sum += v * v;
  }
  // Butterfly (xor) reduction: every lane ends with the warp total, so all
  // lanes see the same inverse_rms.
  float warp_total = square_sum;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) warp_total += __shfl_xor_sync(0xffffffffffffffffull, warp_total, off);
  const float inverse_rms =
      rsqrtf(warp_total / static_cast<float>(head_dim) + eps);

  // normalized = (x * inverse_rms) * weight, rounded once to BF16; the
  // rotation consumes the rounded value, as the shared-memory original does.
  float val[EPL];
#pragma unroll
  for (int j = 0; j < EPL; ++j) {
    const float x = bf16_to_f32(static_cast<uint16_t>(xq[j >> 1] >> ((j & 1) * 16)));
    const float wv = bf16_to_f32(static_cast<uint16_t>(wq[j >> 1] >> ((j & 1) * 16)));
    val[j] = bf16_to_f32(f32_to_bf16(x * inverse_rms * wv));
  }

  // Element i = EPL*lane + j; pair = i mod half, partner element = i xor half
  // (lane ^ 16, same slot). Lanes below 16 hold the low halves, lanes above
  // the high halves, and both stores keep the original contraction forms.
  const int half = head_dim / 2;
  const bool low_half = lane < 16;
  uint32_t oq[EPL / 2] = {};
#pragma unroll
  for (int j = 0; j < EPL; ++j) {
    const int pair = (EPL * lane + j) & (half - 1);
    const float partner = __shfl_xor_sync(0xffffffffffffffffull, val[j], 16);
    const int axis = multimodal_axis(pair, section_h, section_w);
    const float position = static_cast<float>(positions[token * 3 + axis]);
    const float frequency = powf(theta, -2.0f * static_cast<float>(pair) / head_dim);
    float sine, cosine;
    sincosf(position * frequency, &sine, &cosine);
    const float rotated = low_half
        ? __fmaf_rn(val[j], cosine, -(partner * sine))
        : __fmaf_rn(val[j], cosine, partner * sine);
    oq[j >> 1] |= static_cast<uint32_t>(f32_to_bf16(rotated)) << ((j & 1) * 16);
  }
  uint32_t* ow = reinterpret_cast<uint32_t*>(output + base);
#pragma unroll
  for (int w = 0; w < EPL / 2; ++w) ow[lane * (EPL / 2) + w] = oq[w];
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

extern "C" int apxinf_hip_native_rms_norm_apply_mrope_qk(
    const void* q, const void* q_weight, void* q_out, const void* k, const void* k_weight,
    void* k_out, const void* positions, int tokens, int q_heads, int kv_heads, int head_dim,
    float eps, float theta, int section_h, int section_w, void* stream) {
  // The same bounds apply_mrope enforces: an even head_dim for the half-split
  // pairs, and the two spatial sections together within half the pairs.
  if (tokens <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0 ||
      section_h < 0 || section_w < 0) {
    return kInvalidArgument;
  }
  const size_t shared_bytes = static_cast<size_t>(head_dim) * sizeof(bf16);
  // Warp-per-head path for the shapes it fits; the 256-thread block kernel
  // stays as the fallback (small or odd head_dims, misaligned tensors).
  const bool aligned = [](std::initializer_list<const void*> ps) {
    for (const void* p : ps) {
      if (p == nullptr || (reinterpret_cast<uintptr_t>(p) & 0xf) != 0) return false;
    }
    return true;
  }({q, q_weight, q_out, k, k_weight, k_out});
  if (aligned) {
    switch (head_dim) {
      case 64:
        rms_norm_mrope_qk_vec_kernel<2><<<dim3(tokens, q_heads + kv_heads), 32, 0,
                                         as_stream(stream)>>>(
            in(q), in(q_weight), out(q_out), in(k), in(k_weight), out(k_out),
            static_cast<const uint32_t*>(positions), q_heads, kv_heads, head_dim, eps, theta,
            section_h, section_w);
        return last_launch_status();
      case 128:
        rms_norm_mrope_qk_vec_kernel<4><<<dim3(tokens, q_heads + kv_heads), 32, 0,
                                         as_stream(stream)>>>(
            in(q), in(q_weight), out(q_out), in(k), in(k_weight), out(k_out),
            static_cast<const uint32_t*>(positions), q_heads, kv_heads, head_dim, eps, theta,
            section_h, section_w);
        return last_launch_status();
      case 256:
        rms_norm_mrope_qk_vec_kernel<8><<<dim3(tokens, q_heads + kv_heads), 32, 0,
                                         as_stream(stream)>>>(
            in(q), in(q_weight), out(q_out), in(k), in(k_weight), out(k_out),
            static_cast<const uint32_t*>(positions), q_heads, kv_heads, head_dim, eps, theta,
            section_h, section_w);
        return last_launch_status();
      default:
        break;
    }
  }
  rms_norm_mrope_qk_kernel<<<dim3(tokens, q_heads + kv_heads), kReduceBlock, shared_bytes,
                             as_stream(stream)>>>(
      in(q), in(q_weight), out(q_out), in(k), in(k_weight), out(k_out),
      static_cast<const uint32_t*>(positions), q_heads, kv_heads, head_dim, eps, theta, section_h,
      section_w);
  return last_launch_status();
}
