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
  bias_activation_kernel<<<elementwise_grid(count), kElementwiseBlock, 0, as_stream(stream)>>>(
      in(x), in(bias), out(y), count, cols, activation);
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
