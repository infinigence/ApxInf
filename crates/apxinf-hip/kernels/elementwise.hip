// Pointwise, gather and RoPE operators for `apxinf_hip::native`: BF16 storage,
// F32 math, one rounding per stored value.
//
// Ports of the cuda-new kernels pi05 runs on CUDA
// (`apxinf-cuda-new/native/kernels/custom/{pointwise,gather,rope}.cuh`); the
// arithmetic, its order and the rounding points are theirs. All launches are
// grid-stride over the output, so any size is covered.

#include "common.h"

namespace {

using namespace apxinf_hip;
using bf16 = uint16_t;

__device__ __forceinline__ float ld(const bf16* p, int64_t i) { return bf16_to_f32(p[i]); }

// ── Pointwise ───────────────────────────────────────────────────────

constexpr int kActivationNone = 0;
constexpr int kActivationGeluTanh = 1;
constexpr int kActivationSilu = 2;
constexpr int kActivationRelu = 3;
constexpr int kActivationGeluErf = 4;
// Keep this at the highest code: the entry points validate against it.
constexpr int kActivationMax = kActivationGeluErf;

__device__ __forceinline__ float gelu_tanh(float v) {
  constexpr float kAlpha = 0.7978845608028654f;  // sqrt(2 / pi)
  return 0.5f * v * (1.0f + tanhf(kAlpha * (v + 0.044715f * v * v * v)));
}

// Torch's default `nn.GELU()` -- the exact form, not the tanh approximation.
// The two differ by up to 3e-4 absolute, which is above BF16 resolution where
// GELU itself is small, so a model that asks for one must not get the other.
__device__ __forceinline__ float gelu_erf(float v) {
  constexpr float kInvSqrt2 = 0.70710678118654752f;
  return 0.5f * v * (1.0f + erff(v * kInvSqrt2));
}

__device__ __forceinline__ float activate(float v, int activation) {
  if (activation == kActivationGeluTanh) return gelu_tanh(v);
  if (activation == kActivationSilu) return v / (1.0f + expf(-v));
  if (activation == kActivationRelu) return fmaxf(v, 0.0f);
  if (activation == kActivationGeluErf) return gelu_erf(v);
  return v;
}

// y = activation(x + bias), the bias per column and optional.
__global__ void bias_activation_kernel(const bf16* x, const bf16* bias, bf16* y, int64_t count,
                                       int cols, int activation) {
  for (int64_t i = global_index(); i < count; i += grid_stride()) {
    float v = ld(x, i);
    if (bias != nullptr) v += ld(bias, i % cols);
    y[i] = f32_to_bf16(activate(v, activation));
  }
}

// Vectorized form of `bias_activation_kernel`: one uint4 (8 bf16) per thread,
// the bias as a register-loaded uint4, and the per-element int64 `i % cols`
// hoisted to one modulo at entry plus an incremental wrap -- element i is
// 8*v, and with cols % 8 == 0, i % cols is 8*(v % (cols/8)). The per-element
// F32 math and its order are unchanged, so the output is bit-identical to the
// scalar kernel (verified against it on gfx1151 across all activations,
// shapes and the no-bias form; devlocal probe `bias_act_probe`).
// The scalar kernel stays as the fallback for `cols % 8 != 0` or pointers
// that are not 16-byte aligned.
template <bool HAS_BIAS>
__global__ void bias_activation_vec_kernel(const bf16* x, const bf16* bias, bf16* y,
                                           int64_t vcount, int cols8, int activation) {
  const int64_t step = grid_stride();
  const int64_t v0 = global_index();
  const uint4* xv = reinterpret_cast<const uint4*>(x);
  const uint4* bv = reinterpret_cast<const uint4*>(bias);
  uint4* yv = reinterpret_cast<uint4*>(y);
  int64_t c8 = HAS_BIAS ? (v0 % cols8) : 0;
  const int64_t step8 = HAS_BIAS ? (step % cols8) : 0;
  for (int64_t v = v0; v < vcount; v += step) {
    const uint4 xq = xv[v];
    uint4 bq;
    if constexpr (HAS_BIAS) {
      bq = bv[c8];
    } else {
      bq = uint4{0u, 0u, 0u, 0u};
    }
    const uint32_t w[4] = {xq.x, xq.y, xq.z, xq.w};
    const uint32_t bw[4] = {bq.x, bq.y, bq.z, bq.w};
    uint32_t r[4] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int sh = (j & 1) * 16;
      float v = bf16_to_f32(static_cast<uint16_t>(w[j >> 1] >> sh));
      if constexpr (HAS_BIAS) v += bf16_to_f32(static_cast<uint16_t>(bw[j >> 1] >> sh));
      r[j >> 1] |= static_cast<uint32_t>(f32_to_bf16(activate(v, activation))) << sh;
    }
    yv[v] = uint4{r[0], r[1], r[2], r[3]};
    if constexpr (HAS_BIAS) {
      c8 += step8;
      if (c8 >= cols8) c8 -= cols8;
    }
  }
}

// gate_up is [rows, 2 * inner], gate half first: y = gelu_tanh(gate) * up.
__global__ void geglu_kernel(const bf16* gate_up, bf16* y, int64_t rows, int inner) {
  const int64_t count = rows * inner;
  for (int64_t i = global_index(); i < count; i += grid_stride()) {
    const int64_t row = i / inner;
    const int64_t col = i % inner;
    const bf16* r = gate_up + row * 2 * inner;
    y[i] = f32_to_bf16(gelu_tanh(ld(r, col)) * ld(r, inner + col));
  }
}

// y = silu(gate) * up over separate operands -- GR00T's SwiGLU MLP, where the
// two projections are distinct tensors rather than the packed `gate_up` that
// `geglu` splits.
__global__ void silu_mul_kernel(const bf16* gate, const bf16* up, bf16* y, int64_t count) {
  for (int64_t i = global_index(); i < count; i += grid_stride()) {
    const float g = ld(gate, i);
    y[i] = f32_to_bf16(g / (1.0f + expf(-g)) * ld(up, i));
  }
}

// Vectorized silu_mul: uint4, eight elements per thread over a flat index.
// The per-element F32 expression (and its evaluation order) is unchanged, so
// the output is bit-identical to the scalar kernel; that kernel stays as the
// fallback for `count % 8 != 0` or tensors that are not 16-byte aligned.
__global__ void silu_mul_vec_kernel(const bf16* gate, const bf16* up, bf16* y, int64_t vcount) {
  const int64_t step = grid_stride();
  const uint4* gv = reinterpret_cast<const uint4*>(gate);
  const uint4* uv = reinterpret_cast<const uint4*>(up);
  uint4* yv = reinterpret_cast<uint4*>(y);
  for (int64_t v = global_index(); v < vcount; v += step) {
    const uint4 gq = gv[v];
    const uint4 uq = uv[v];
    const uint32_t gw[4] = {gq.x, gq.y, gq.z, gq.w};
    const uint32_t uw[4] = {uq.x, uq.y, uq.z, uq.w};
    uint32_t r[4] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int sh = (j & 1) * 16;
      const float g = bf16_to_f32(static_cast<uint16_t>(gw[j >> 1] >> sh));
      const float u = bf16_to_f32(static_cast<uint16_t>(uw[j >> 1] >> sh));
      r[j >> 1] |= static_cast<uint32_t>(f32_to_bf16(g / (1.0f + expf(-g)) * u)) << sh;
    }
    yv[v] = uint4{r[0], r[1], r[2], r[3]};
  }
}

// One explicit-Euler flow step: y = state + dt * velocity.
__global__ void euler_update_kernel(const bf16* state, const bf16* velocity, bf16* y,
                                    int64_t count, float dt) {
  for (int64_t i = global_index(); i < count; i += grid_stride()) {
    y[i] = f32_to_bf16(ld(state, i) + dt * ld(velocity, i));
  }
}

// ── Gather ──────────────────────────────────────────────────────────

// Token-embedding rows scaled by sqrt(width), Gemma's embedding normalizer,
// which the backbone folds into the lookup. Out-of-range ids produce zeros
// rather than reading past the table (the host range-checks them as well).
// `scale_by_sqrt_width` is pi05's Gemma convention; GR00T's Qwen3-VL table is
// consumed unscaled (cuda-new's `EmbeddingScale::None`), and the two differ
// numerically, so the caller states which it wants.
__global__ void embedding_kernel(const bf16* table, const uint32_t* ids, bf16* y, int tokens,
                                 int width, uint32_t vocab, int scale_by_sqrt_width) {
  const int64_t count = static_cast<int64_t>(tokens) * width;
  const float normalizer = scale_by_sqrt_width ? sqrtf(static_cast<float>(width)) : 1.0f;
  for (int64_t i = global_index(); i < count; i += grid_stride()) {
    const int64_t token = i / width;
    const int64_t col = i % width;
    const uint32_t id = ids[token];
    y[i] = id < vocab ? f32_to_bf16(ld(table, static_cast<int64_t>(id) * width + col) * normalizer)
                      : f32_to_bf16(0.0f);
  }
}

// Vision patch-embedding epilogue: y = projection + position (+ bias), with
// `position` [tokens_per_view, cols] repeating across views.
__global__ void position_kernel(const bf16* projection, const bf16* bias, const bf16* position,
                                bf16* y, int64_t count, int cols, int tokens_per_view) {
  for (int64_t i = global_index(); i < count; i += grid_stride()) {
    const int64_t col = i % cols;
    const int64_t token = (i / cols) % tokens_per_view;
    float v = ld(projection, i) + ld(position, token * cols + col);
    if (bias != nullptr) v += ld(bias, col);
    y[i] = f32_to_bf16(v);
  }
}

// RGB u8 images to flattened patches [views * patches_per_view, 3 * p * p],
// channel-major inside a patch, mapping [0, 255] to [-1, 1].
__global__ void rgb_to_patches_kernel(const uint8_t* images, bf16* patches, int views,
                                      int image_size, int patch_size, int nhwc) {
  const int per_side = image_size / patch_size;
  const int per_view = per_side * per_side;
  const int area = patch_size * patch_size;
  const int width = 3 * area;
  const int64_t count = static_cast<int64_t>(views) * per_view * width;
  for (int64_t o = global_index(); o < count; o += grid_stride()) {
    const int element = static_cast<int>(o % width);
    const int patch = static_cast<int>(o / width);
    const int view = patch / per_view;
    const int in_view = patch - view * per_view;
    const int py = in_view / per_side;
    const int px = in_view - py * per_side;
    const int channel = element / area;
    const int pixel = element - channel * area;
    const int y = py * patch_size + pixel / patch_size;
    const int x = px * patch_size + pixel % patch_size;
    const int64_t source =
        nhwc ? ((static_cast<int64_t>(view) * image_size + y) * image_size + x) * 3 + channel
             : ((static_cast<int64_t>(view) * 3 + channel) * image_size + y) * image_size + x;
    patches[o] = f32_to_bf16((static_cast<float>(images[source]) / 255.0f) * 2.0f - 1.0f);
  }
}

// ── QKV split and RoPE ──────────────────────────────────────────────

// Split a packed [tokens, (q_heads + 2 kv_heads) * head_dim] projection, add
// the optional bias, rotate Q and K (half-split pairs, frequency
// theta^(-pair / half), position = position_offset + token) and copy V.
// K and V land at row `kv_row_offset + token` of [rows, kv_heads, head_dim]
// buffers, so the same kernel fills a fresh prefix cache or appends to one.
// Grid (tokens, q_heads + 2 kv_heads); threads loop over the head's pairs.
__global__ void split_qkv_rope_kernel(const bf16* qkv, const bf16* bias, bf16* q, bf16* k,
                                      bf16* v, int q_heads, int kv_heads, int head_dim,
                                      float theta, int position_offset, int kv_row_offset) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int half = head_dim / 2;
  const int64_t q_width = static_cast<int64_t>(q_heads) * head_dim;
  const int64_t kv_width = static_cast<int64_t>(kv_heads) * head_dim;
  const int64_t row = static_cast<int64_t>(token) * (q_width + 2 * kv_width);
  const int position = position_offset + token;
  int64_t column;  // offset of this head inside the packed projection
  bf16* destination;
  bool rotate = true;
  if (head < q_heads) {
    column = static_cast<int64_t>(head) * head_dim;
    destination = q + (static_cast<int64_t>(token) * q_heads + head) * head_dim;
  } else if (head < q_heads + kv_heads) {
    const int h = head - q_heads;
    column = q_width + static_cast<int64_t>(h) * head_dim;
    destination = k + (static_cast<int64_t>(kv_row_offset + token) * kv_heads + h) * head_dim;
  } else {
    const int h = head - q_heads - kv_heads;
    column = q_width + kv_width + static_cast<int64_t>(h) * head_dim;
    destination = v + (static_cast<int64_t>(kv_row_offset + token) * kv_heads + h) * head_dim;
    rotate = false;
  }
  for (int pair = threadIdx.x; pair < half; pair += blockDim.x) {
    float first = ld(qkv, row + column + pair);
    float second = ld(qkv, row + column + half + pair);
    if (bias != nullptr) {
      first += ld(bias, column + pair);
      second += ld(bias, column + half + pair);
    }
    if (rotate) {
      const float frequency = powf(theta, -static_cast<float>(pair) / half);
      float sine, cosine;
      sincosf(position * frequency, &sine, &cosine);
      destination[pair] = f32_to_bf16(first * cosine - second * sine);
      destination[half + pair] = f32_to_bf16(second * cosine + first * sine);
    } else {
      destination[pair] = f32_to_bf16(first);
      destination[half + pair] = f32_to_bf16(second);
    }
  }
}

// Split a packed [tokens, 3 * width] projection with an optional bias and no
// rotation: the vision tower's MHA layout, where Q, K and V share one width.
__global__ void split_qkv_bias_kernel(const bf16* qkv, const bf16* bias, bf16* q, bf16* k, bf16* v,
                                      int width) {
  const int64_t token = blockIdx.x;
  const int fused = 3 * width;
  for (int col = threadIdx.x; col < fused; col += blockDim.x) {
    float value = ld(qkv, token * fused + col);
    if (bias != nullptr) value += ld(bias, col);
    const int64_t base = token * width;
    if (col < width) {
      q[base + col] = f32_to_bf16(value);
    } else if (col < 2 * width) {
      k[base + col - width] = f32_to_bf16(value);
    } else {
      v[base + col - 2 * width] = f32_to_bf16(value);
    }
  }
}

// Split a packed [tokens, 3 * width] projection with the optional bias and
// (h, w) axial rotation, all in one pass: cuda-new's fused vision
// `split_qkv_axial_rope`, for the MHA layout `split_qkv_bias_kernel` serves.
// `positions` is [tokens, 2] u32 as (h, w); the first quarter of each head's
// half-split pairs rotate by the height position, the rest by the width one,
// each at frequency `theta^(-2 * pair_within_axis / half)` -- the indexing
// `apply_vision_2d_kernel` (rope.hip) uses, so the two agree bit for bit when
// this is compared against split + rotate: the biased projection is rounded
// to BF16 before rotating here, which is the rounding the split performs when
// it writes q, k and v out.
//
// The block builds the per-token sine/cosine tables in shared memory once
// (4 * quarter floats, dynamic because quarter follows head_dim), then covers
// the head/pair work of Q and K and the columns of V. Grid (tokens).
// `VEC_V` swaps the V copy loop to uint4 eight-column steps; the per-element
// F32 math (qkv + bias, one rounding) is unchanged, so it is bit-identical to
// the scalar V loop. Gated on `projection_width % 8 == 0` + 16-byte alignment.
template <bool VEC_V>
__global__ void split_qkv_axial_rope_kernel(const bf16* qkv, const bf16* bias,
                                            const uint32_t* positions, bf16* q, bf16* k,
                                            bf16* v, int heads, int head_dim, float theta) {
  const int token = blockIdx.x;
  const int half = head_dim / 2;
  const int quarter = half / 2;
  const int projection_width = heads * head_dim;
  const int fused_width = 3 * projection_width;
  const int pairs_per_projection = heads * half;

  extern __shared__ float rope[];
  float* sine = rope;
  float* cosine = rope + 2 * quarter;
  for (int slot = threadIdx.x; slot < 2 * quarter; slot += blockDim.x) {
    const int axis = slot / quarter;
    const int pair = slot - axis * quarter;
    const float position = static_cast<float>(positions[token * 2 + axis]);
    const float frequency = powf(theta, -2.0f * static_cast<float>(pair) / half);
    sincosf(position * frequency, &sine[slot], &cosine[slot]);
  }
  __syncthreads();

  for (int work = threadIdx.x; work < 2 * pairs_per_projection; work += blockDim.x) {
    const bool is_key = work >= pairs_per_projection;
    const int local = is_key ? work - pairs_per_projection : work;
    const int head = local / half;
    const int pair = local - head * half;
    const int projection_offset = is_key ? projection_width : 0;
    const int64_t source = static_cast<int64_t>(token) * fused_width +
                           projection_offset + head * head_dim;
    const int bias_base = projection_offset + head * head_dim;
    float first = ld(qkv, source + pair);
    float second = ld(qkv, source + half + pair);
    if (bias != nullptr) {
      first += ld(bias, bias_base + pair);
      second += ld(bias, bias_base + half + pair);
    }
    first = round_bf16(first);
    second = round_bf16(second);
    const int axis = pair < quarter ? 0 : 1;
    const int slot = axis * quarter + (pair < quarter ? pair : pair - quarter);
    bf16* destination =
        (is_key ? k : q) + (static_cast<int64_t>(token) * heads + head) * head_dim;
    // The rotations are pinned with explicit FMAs rather than left to the
    // compiler's contraction choice: `a * c - b * s` contracts two ways (round
    // either product, fuse the other), the choice differed between this
    // translation unit and rope.hip's, and that showed up as 1-ulp drift on
    // 3 of 524288 elements against the split-then-rotate reference. The forms
    // here are the ones rope.hip's compiled kernels use, which is what keeps
    // this bit-identical to them: the first store rounds b * s and fuses
    // a * c; the second rounds a * s and fuses b * c.
    destination[pair] = f32_to_bf16(__fmaf_rn(first, cosine[slot], -(second * sine[slot])));
    destination[half + pair] = f32_to_bf16(__fmaf_rn(second, cosine[slot], first * sine[slot]));
  }

  if constexpr (VEC_V) {
    for (int v0 = threadIdx.x * 8; v0 < projection_width; v0 += blockDim.x * 8) {
      const int64_t source =
          static_cast<int64_t>(token) * fused_width + 2 * projection_width + v0;
      const uint4 sq = *reinterpret_cast<const uint4*>(qkv + source);
      uint4 bq = uint4{0u, 0u, 0u, 0u};
      if (bias != nullptr) {
        bq = *reinterpret_cast<const uint4*>(bias + 2 * projection_width + v0);
      }
      const uint32_t sw[4] = {sq.x, sq.y, sq.z, sq.w};
      const uint32_t bw[4] = {bq.x, bq.y, bq.z, bq.w};
      uint32_t r[4] = {0u, 0u, 0u, 0u};
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int sh = (j & 1) * 16;
        float value = bf16_to_f32(static_cast<uint16_t>(sw[j >> 1] >> sh));
        if (bias != nullptr) {
          value += bf16_to_f32(static_cast<uint16_t>(bw[j >> 1] >> sh));
        }
        r[j >> 1] |= static_cast<uint32_t>(f32_to_bf16(value)) << sh;
      }
      *reinterpret_cast<uint4*>(v + static_cast<int64_t>(token) * projection_width + v0) =
          uint4{r[0], r[1], r[2], r[3]};
    }
  } else {
    for (int column = threadIdx.x; column < projection_width; column += blockDim.x) {
      const int64_t source =
          static_cast<int64_t>(token) * fused_width + 2 * projection_width + column;
      float value = ld(qkv, source);
      if (bias != nullptr) {
        value += ld(bias, 2 * projection_width + column);
      }
      v[static_cast<int64_t>(token) * projection_width + column] = f32_to_bf16(value);
    }
  }
}

// ── Row and column movement ─────────────────────────────────────────

// Gather whole rows of `input` named by u32 `indices`.
__global__ void gather_rows_kernel(const bf16* input, const uint32_t* indices, bf16* y, int rows,
                                   int cols) {
  const int64_t count = static_cast<int64_t>(rows) * cols;
  for (int64_t i = global_index(); i < count; i += grid_stride()) {
    const int64_t row = i / cols;
    const int64_t col = i % cols;
    y[i] = input[static_cast<int64_t>(indices[row]) * cols + col];
  }
}

// Scatter whole rows of `source` into the rows of `destination` named by u32
// `indices`, replacing or accumulating. The indices GR00T scatters by are a
// permutation of distinct rows, so accumulation needs no atomics.
__global__ void scatter_rows_kernel(bf16* destination, const uint32_t* indices, const bf16* source,
                                    int rows, int cols, int accumulate) {
  const int64_t count = static_cast<int64_t>(rows) * cols;
  for (int64_t i = global_index(); i < count; i += grid_stride()) {
    const int64_t row = i / cols;
    const int64_t col = i % cols;
    const int64_t target = static_cast<int64_t>(indices[row]) * cols + col;
    destination[target] =
        accumulate ? f32_to_bf16(ld(destination, target) + ld(source, i)) : source[i];
  }
}

// Copy `src` into the column window of `dst` starting at `dst_col`. Column
// concatenation is one of these per source, the way `copy_rows` composes into
// row concatenation.
__global__ void copy_columns_kernel(const bf16* src, bf16* dst, int rows, int src_cols,
                                    int dst_cols, int dst_col) {
  const int64_t count = static_cast<int64_t>(rows) * src_cols;
  for (int64_t i = global_index(); i < count; i += grid_stride()) {
    const int64_t row = i / src_cols;
    const int64_t col = i % src_cols;
    dst[row * dst_cols + dst_col + col] = src[i];
  }
}

const bf16* in(const void* p) { return static_cast<const bf16*>(p); }
bf16* out(void* p) { return static_cast<bf16*>(p); }

unsigned elementwise_grid(int64_t count) { return grid_for(count, kElementwiseBlock); }

}  // namespace

extern "C" int apxinf_hip_native_bias_activation(const void* x, const void* bias, void* y,
                                                 int64_t rows, int cols, int activation,
                                                 void* stream) {
  if (rows <= 0 || cols <= 0 || activation < kActivationNone || activation > kActivationMax) {
    return kInvalidArgument;
  }
  const int64_t count = rows * cols;
  hipStream_t s = as_stream(stream);
  // Vector path: count is a multiple of 8 whenever cols is, and tensor
  // allocations are 16-byte aligned; the checks cost nothing and keep the
  // scalar kernel correct for any view the fast path cannot take.
  const bool vec = (cols % 8 == 0) && (reinterpret_cast<uintptr_t>(x) & 15) == 0 &&
                   (reinterpret_cast<uintptr_t>(y) & 15) == 0 &&
                   (bias == nullptr || (reinterpret_cast<uintptr_t>(bias) & 15) == 0);
  if (vec) {
    const int64_t vcount = count >> 3;
    const int cols8 = cols >> 3;
    if (bias != nullptr) {
      bias_activation_vec_kernel<true><<<elementwise_grid(vcount), kElementwiseBlock, 0, s>>>(
          in(x), in(bias), out(y), vcount, cols8, activation);
    } else {
      bias_activation_vec_kernel<false><<<elementwise_grid(vcount), kElementwiseBlock, 0, s>>>(
          in(x), in(bias), out(y), vcount, cols8, activation);
    }
  } else {
    bias_activation_kernel<<<elementwise_grid(count), kElementwiseBlock, 0, s>>>(
        in(x), in(bias), out(y), count, cols, activation);
  }
  return last_launch_status();
}

extern "C" int apxinf_hip_native_geglu(const void* gate_up, void* y, int64_t rows, int inner,
                                       void* stream) {
  if (rows <= 0 || inner <= 0) return kInvalidArgument;
  geglu_kernel<<<elementwise_grid(rows * inner), kElementwiseBlock, 0, as_stream(stream)>>>(
      in(gate_up), out(y), rows, inner);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_gather_rows(const void* input, const void* indices, void* y,
                                            int rows, int cols, void* stream) {
  if (rows <= 0 || cols <= 0) return kInvalidArgument;
  gather_rows_kernel<<<elementwise_grid(static_cast<int64_t>(rows) * cols), kElementwiseBlock, 0,
                       as_stream(stream)>>>(in(input), static_cast<const uint32_t*>(indices),
                                            out(y), rows, cols);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_scatter_rows(void* destination, const void* indices,
                                              const void* source, int rows, int cols,
                                              int accumulate, void* stream) {
  if (rows <= 0 || cols <= 0) return kInvalidArgument;
  scatter_rows_kernel<<<elementwise_grid(static_cast<int64_t>(rows) * cols), kElementwiseBlock, 0,
                        as_stream(stream)>>>(out(destination),
                                             static_cast<const uint32_t*>(indices), in(source),
                                             rows, cols, accumulate);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_copy_columns(const void* src, void* dst, int rows, int src_cols,
                                              int dst_cols, int dst_col, void* stream) {
  if (rows <= 0 || src_cols <= 0 || dst_cols <= 0 || dst_col < 0 || dst_col + src_cols > dst_cols) {
    return kInvalidArgument;
  }
  copy_columns_kernel<<<elementwise_grid(static_cast<int64_t>(rows) * src_cols), kElementwiseBlock,
                        0, as_stream(stream)>>>(in(src), out(dst), rows, src_cols, dst_cols,
                                                dst_col);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_silu_mul(const void* gate, const void* up, void* y,
                                          int64_t count, void* stream) {
  if (count <= 0) return kInvalidArgument;
  const bool aligned = [](std::initializer_list<const void*> ps) {
    for (const void* p : ps) {
      if (p == nullptr || (reinterpret_cast<uintptr_t>(p) & 0xf) != 0) return false;
    }
    return true;
  }({gate, up, y});
  if (count % 8 == 0 && aligned) {
    silu_mul_vec_kernel<<<elementwise_grid(count / 8), kElementwiseBlock, 0, as_stream(stream)>>>(
        in(gate), in(up), out(y), count / 8);
    return last_launch_status();
  }
  silu_mul_kernel<<<elementwise_grid(count), kElementwiseBlock, 0, as_stream(stream)>>>(
      in(gate), in(up), out(y), count);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_euler_update(const void* state, const void* velocity, void* y,
                                              int64_t count, float dt, void* stream) {
  if (count <= 0) return kInvalidArgument;
  euler_update_kernel<<<elementwise_grid(count), kElementwiseBlock, 0, as_stream(stream)>>>(
      in(state), in(velocity), out(y), count, dt);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_embedding(const void* table, const void* ids, void* y, int tokens,
                                           int width, uint32_t vocab, int scale_by_sqrt_width,
                                           void* stream) {
  if (tokens <= 0 || width <= 0 || vocab == 0) return kInvalidArgument;
  const int64_t count = static_cast<int64_t>(tokens) * width;
  embedding_kernel<<<elementwise_grid(count), kElementwiseBlock, 0, as_stream(stream)>>>(
      in(table), static_cast<const uint32_t*>(ids), out(y), tokens, width, vocab,
      scale_by_sqrt_width);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_add_position(const void* projection, const void* bias,
                                              const void* position, void* y, int64_t rows,
                                              int cols, int tokens_per_view, void* stream) {
  if (rows <= 0 || cols <= 0 || tokens_per_view <= 0 || rows % tokens_per_view != 0) {
    return kInvalidArgument;
  }
  const int64_t count = rows * cols;
  position_kernel<<<elementwise_grid(count), kElementwiseBlock, 0, as_stream(stream)>>>(
      in(projection), in(bias), in(position), out(y), count, cols, tokens_per_view);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_rgb_to_patches(const void* images, void* patches, int views,
                                                int image_size, int patch_size, int nhwc,
                                                void* stream) {
  if (views <= 0 || patch_size <= 0 || image_size <= 0 || image_size % patch_size != 0) {
    return kInvalidArgument;
  }
  const int64_t count = static_cast<int64_t>(views) * image_size * image_size * 3;
  rgb_to_patches_kernel<<<elementwise_grid(count), kElementwiseBlock, 0, as_stream(stream)>>>(
      static_cast<const uint8_t*>(images), out(patches), views, image_size, patch_size, nhwc);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_split_qkv_rope(const void* qkv, const void* bias, void* q, void* k,
                                                void* v, int tokens, int q_heads, int kv_heads,
                                                int head_dim, float theta, int position_offset,
                                                int kv_row_offset, void* stream) {
  if (tokens <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0 ||
      position_offset < 0 || kv_row_offset < 0) {
    return kInvalidArgument;
  }
  const int half = head_dim / 2;
  const int block = half >= kAttentionBlock ? kAttentionBlock : ((half + 31) / 32) * 32;
  split_qkv_rope_kernel<<<dim3(tokens, q_heads + 2 * kv_heads), block, 0, as_stream(stream)>>>(
      in(qkv), in(bias), out(q), out(k), out(v), q_heads, kv_heads, head_dim, theta,
      position_offset, kv_row_offset);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_split_qkv_bias(const void* qkv, const void* bias, void* q,
                                                void* k, void* v, int tokens, int width,
                                                void* stream) {
  if (tokens <= 0 || width <= 0) return kInvalidArgument;
  split_qkv_bias_kernel<<<tokens, kElementwiseBlock, 0, as_stream(stream)>>>(
      in(qkv), in(bias), out(q), out(k), out(v), width);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_split_qkv_axial_rope(
    const void* qkv, const void* bias, const void* positions, void* q, void* k, void* v,
    int tokens, int heads, int head_dim, float theta, void* stream) {
  // head_dim % 4 keeps the half-split pairs evenly divisible between the two
  // axes, which the per-axis frequency indexing assumes (as apply_vision_2d).
  if (tokens <= 0 || heads <= 0 || head_dim <= 0 || head_dim % 4 != 0) {
    return kInvalidArgument;
  }
  const int quarter = head_dim / 4;
  const unsigned shared_bytes = 4 * static_cast<unsigned>(quarter) * sizeof(float);
  const int projection_width = heads * head_dim;
  const bool aligned = [](std::initializer_list<const void*> ps) {
    for (const void* p : ps) {
      if (p == nullptr || (reinterpret_cast<uintptr_t>(p) & 0xf) != 0) return false;
    }
    return true;
  }({qkv, bias, q, k, v});
  const bool use_vec = projection_width % 8 == 0 && aligned;
  if (use_vec) {
    split_qkv_axial_rope_kernel<true><<<tokens, kElementwiseBlock, shared_bytes,
                                        as_stream(stream)>>>(
        in(qkv), in(bias), static_cast<const uint32_t*>(positions), out(q), out(k), out(v), heads,
        head_dim, theta);
  } else {
    split_qkv_axial_rope_kernel<false><<<tokens, kElementwiseBlock, shared_bytes,
                                         as_stream(stream)>>>(
        in(qkv), in(bias), static_cast<const uint32_t*>(positions), out(q), out(k), out(v), heads,
        head_dim, theta);
  }
  return last_launch_status();
}
