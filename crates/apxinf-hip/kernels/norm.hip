// Normalization operators for `apxinf_hip::native`: BF16 storage, F32 math.
//
// Each kernel is a port of the cuda-new kernel of the same name
// (`apxinf-cuda-new/native/kernels/custom/norm.cuh`), including where it
// rounds. The fused residual kernels round the residual sum to BF16, store it
// as `hidden`, and normalize the stored value rather than the F32 sum — the
// observable behaviour of two separate BF16 kernels, and what pi05 was
// validated against on CUDA.
//
// One block of kReduceBlock threads per row. Only the reduction order differs
// from CUDA (a shared-memory tree instead of warp shuffles), so results agree
// to within the rounding of an F32 sum, not bit for bit.

#include "common.h"

namespace {

using namespace apxinf_hip;
using bf16 = uint16_t;

__device__ __forceinline__ float ld(const bf16* p, int64_t i) { return bf16_to_f32(p[i]); }

__global__ void layer_norm_kernel(const bf16* x, const bf16* weight, const bf16* bias, bf16* y,
                                  int cols, float eps) {
  __shared__ float scratch[kReduceBlock];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) sum += ld(x, base + c);
  const float mean = block_sum(sum, scratch) / cols;
  float centered_sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float centered = ld(x, base + c) - mean;
    centered_sum += centered * centered;
  }
  const float inverse_std = rsqrtf(block_sum(centered_sum, scratch) / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    y[base + c] =
        f32_to_bf16((ld(x, base + c) - mean) * inverse_std * ld(weight, c) + ld(bias, c));
  }
}

// GR00T's DiT conditioning: LayerNorm statistics with the per-call
// `[scale; shift]` modulation `ada_rms_norm` uses, i.e.
// `(x - mean) * rsqrt(var + eps) * (1 + scale) + shift`. The `1 +` offset and
// the single rounding at the store are cuda-new's
// (`native/kernels/norm/fixed_bf16.cuh: adaptive_layer_norm_bf16_kernel`).
__global__ void ada_layer_norm_kernel(const bf16* x, const bf16* style, bf16* y, int cols,
                                      float eps) {
  __shared__ float scratch[kReduceBlock];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) sum += ld(x, base + c);
  const float mean = block_sum(sum, scratch) / cols;
  float centered_sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float centered = ld(x, base + c) - mean;
    centered_sum += centered * centered;
  }
  const float inverse_std = rsqrtf(block_sum(centered_sum, scratch) / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float normalized = (ld(x, base + c) - mean) * inverse_std;
    y[base + c] = f32_to_bf16(normalized * (1.0f + ld(style, c)) + ld(style, cols + c));
  }
}

// `style` is [2 * cols]: scale in [0, cols), shift in [cols, 2 * cols).
__global__ void ada_rms_norm_kernel(const bf16* x, const bf16* style, bf16* y, int cols,
                                    float eps) {
  __shared__ float scratch[kReduceBlock];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float square_sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float v = ld(x, base + c);
    square_sum += v * v;
  }
  const float inverse_rms = rsqrtf(block_sum(square_sum, scratch) / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float normalized = ld(x, base + c) * inverse_rms;
    y[base + c] = f32_to_bf16(normalized * (1.0f + ld(style, c)) + ld(style, cols + c));
  }
}

// hidden = projection + residual (+ bias), rounded once. Elementwise.
__global__ void bias_residual_kernel(const bf16* projection, const bf16* bias, const bf16* residual,
                                     bf16* hidden, int64_t count, int cols) {
  for (int64_t i = global_index(); i < count; i += grid_stride()) {
    float v = ld(projection, i) + ld(residual, i);
    if (bias != nullptr) v += ld(bias, i % cols);
    hidden[i] = f32_to_bf16(v);
  }
}

// Vectorized bias_residual: uint4, eight elements per thread over a flat
// index. The bias lookup's per-element int64 `i % cols` is hoisted to one
// modulo at entry plus an incremental wrap -- element i is 8*v, and with
// cols % 8 == 0, i % cols is 8*(v % (cols/8)) -- following
// bias_activation_vec_kernel. The per-element F32 math and its order are
// unchanged, so the output is bit-identical to the scalar kernel. Gated on
// count % 8 == 0, cols % 8 == 0 and 16-byte alignment; the scalar kernel
// stays as the fallback.
template <bool HAS_BIAS>
__global__ void bias_residual_vec_kernel(const bf16* projection, const bf16* bias,
                                         const bf16* residual, bf16* hidden, int64_t vcount,
                                         int cols8) {
  const int64_t step = grid_stride();
  const int64_t v0 = global_index();
  const uint4* pv = reinterpret_cast<const uint4*>(projection);
  const uint4* rv = reinterpret_cast<const uint4*>(residual);
  const uint4* bv = reinterpret_cast<const uint4*>(bias);
  uint4* hv = reinterpret_cast<uint4*>(hidden);
  int64_t c8 = HAS_BIAS ? (v0 % cols8) : 0;
  const int64_t step8 = HAS_BIAS ? (step % cols8) : 0;
  for (int64_t v = v0; v < vcount; v += step) {
    const uint4 pq = pv[v];
    const uint4 rq = rv[v];
    uint4 bq = uint4{0u, 0u, 0u, 0u};
    if constexpr (HAS_BIAS) bq = bv[c8];
    const uint32_t pw[4] = {pq.x, pq.y, pq.z, pq.w};
    const uint32_t rw[4] = {rq.x, rq.y, rq.z, rq.w};
    const uint32_t bw[4] = {bq.x, bq.y, bq.z, bq.w};
    uint32_t r[4] = {0u, 0u, 0u, 0u};
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int sh = (j & 1) * 16;
      float v = bf16_to_f32(static_cast<uint16_t>(pw[j >> 1] >> sh));
      v += bf16_to_f32(static_cast<uint16_t>(rw[j >> 1] >> sh));
      if constexpr (HAS_BIAS) v += bf16_to_f32(static_cast<uint16_t>(bw[j >> 1] >> sh));
      r[j >> 1] |= static_cast<uint32_t>(f32_to_bf16(v)) << sh;
    }
    hv[v] = uint4{r[0], r[1], r[2], r[3]};
    if constexpr (HAS_BIAS) {
      c8 += step8;
      if (c8 >= cols8) c8 -= cols8;
    }
  }
}

// hidden = round(projection + residual (+ bias)); normalized = rms(hidden) * weight.
__global__ void bias_residual_rms_norm_kernel(const bf16* projection, const bf16* bias,
                                              const bf16* residual, const bf16* weight,
                                              bf16* hidden, bf16* normalized, int cols,
                                              float eps) {
  __shared__ float scratch[kReduceBlock];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float square_sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = ld(projection, base + c) + ld(residual, base + c);
    if (bias != nullptr) v += ld(bias, c);
    const bf16 rounded = f32_to_bf16(v);
    hidden[base + c] = rounded;
    const float stored = bf16_to_f32(rounded);
    square_sum += stored * stored;
  }
  const float inverse_rms = rsqrtf(block_sum(square_sum, scratch) / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    normalized[base + c] = f32_to_bf16(ld(hidden, base + c) * inverse_rms * ld(weight, c));
  }
}

// hidden = round(projection + residual (+ bias)); normalized = layer_norm(hidden).
__global__ void bias_residual_layer_norm_kernel(const bf16* projection, const bf16* bias,
                                                const bf16* residual, const bf16* norm_weight,
                                                const bf16* norm_bias, bf16* hidden,
                                                bf16* normalized, int cols, float eps) {
  __shared__ float scratch[kReduceBlock];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = ld(projection, base + c) + ld(residual, base + c);
    if (bias != nullptr) v += ld(bias, c);
    const bf16 rounded = f32_to_bf16(v);
    hidden[base + c] = rounded;
    sum += bf16_to_f32(rounded);
  }
  const float mean = block_sum(sum, scratch) / cols;
  float centered_sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float centered = ld(hidden, base + c) - mean;
    centered_sum += centered * centered;
  }
  const float inverse_std = rsqrtf(block_sum(centered_sum, scratch) / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    normalized[base + c] = f32_to_bf16((ld(hidden, base + c) - mean) * inverse_std *
                                           ld(norm_weight, c) +
                                       ld(norm_bias, c));
  }
}

// hidden = round(residual + projection * gate), the gate being the third
// segment of this layer's [3 * cols] `gate_style`; normalized is the adaptive
// RMS norm of hidden under the next layer's [2 * cols] `norm_style`.
// hidden = round(projection + residual (+ bias)); normalized is the adaptive
// LayerNorm of hidden under `style`.
//
// The pair this replaces spans a DiT block boundary: a block ends by writing
// `hidden`, and the next begins by normalizing it under its own modulation.
// Fusing them removes a launch and a round trip of `hidden` through memory --
// at 41 rows of 1536 the launch is the larger half, since 126 KB is half a
// microsecond of traffic against about 1.8 us of dispatch.
//
// `hidden` is still written, and written *before* it is read back for the
// statistics, because the separate path rounds it to BF16 there and the
// normalization consumes the rounded value. Computing the mean from the
// unrounded sum would be more accurate and would not match.
__global__ void bias_residual_ada_layer_norm_kernel(const bf16* projection, const bf16* bias,
                                                    const bf16* residual, const bf16* style,
                                                    bf16* hidden, bf16* normalized, int cols,
                                                    float eps) {
  __shared__ float scratch[kReduceBlock];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = ld(projection, base + c) + ld(residual, base + c);
    if (bias != nullptr) v += ld(bias, c);
    const bf16 rounded = f32_to_bf16(v);
    hidden[base + c] = rounded;
    sum += bf16_to_f32(rounded);
  }
  const float mean = block_sum(sum, scratch) / cols;
  float centered_sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float centered = ld(hidden, base + c) - mean;
    centered_sum += centered * centered;
  }
  const float inverse_std = rsqrtf(block_sum(centered_sum, scratch) / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float v = (ld(hidden, base + c) - mean) * inverse_std;
    normalized[base + c] =
        f32_to_bf16(v * (1.0f + ld(style, c)) + ld(style, cols + c));
  }
}

__global__ void ada_gate_residual_rms_norm_kernel(const bf16* projection, const bf16* residual,
                                                  const bf16* gate_style, const bf16* norm_style,
                                                  bf16* hidden, bf16* normalized, int cols,
                                                  float eps) {
  __shared__ float scratch[kReduceBlock];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float square_sum = 0.0f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const bf16 rounded =
        f32_to_bf16(ld(residual, base + c) + ld(projection, base + c) * ld(gate_style, 2 * cols + c));
    hidden[base + c] = rounded;
    const float stored = bf16_to_f32(rounded);
    square_sum += stored * stored;
  }
  const float inverse_rms = rsqrtf(block_sum(square_sum, scratch) / cols + eps);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    const float v = ld(hidden, base + c) * inverse_rms;
    normalized[base + c] =
        f32_to_bf16(v * (1.0f + ld(norm_style, c)) + ld(norm_style, cols + c));
  }
}

// ---------------------------------------------------------------------------
// Vectorized paths (2026-11-12). The scalar kernels above are instruction-
// bound on RDNA3: 2-byte loads and two or three passes over each row (probe
// `devlocal/gr00t-hip-port/perf/norm_register_probe.cpp`: 1.4-2.9x per kernel
// from this change). These variants load uint4 (8 bf16) with a contiguous
// 8-element-per-thread mapping, cache the row in registers, and scan it once.
// The block_sum tree is unchanged, but each partial now covers 8 contiguous
// elements instead of a strided set, so the F32 sums round slightly
// differently (measured 1-56 elements per million at 1 BF16 ulp). Gated on
// `cols % 8 == 0`, `cols <= 8192` (4 slices x 2048) and 16-byte alignment;
// everything else takes the scalar kernel above.
// ---------------------------------------------------------------------------

union V8 {
  uint4 raw;
  bf16 e[8];
};

__device__ __forceinline__ uint4 ld8(const bf16* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ void st8(bf16* p, uint4 v) { *reinterpret_cast<uint4*>(p) = v; }

// ceil(cols / 2048) when the vectorized path applies (cols a multiple of 8,
// at most 8192), else 0 -- the caller takes the scalar kernel.
int vec_slices(int cols) {
  if (cols % 8 != 0 || cols <= 0 || cols > 8192) return 0;
  return (cols + 2047) / 2048;
}

bool aligned16(std::initializer_list<const void*> ps) {
  for (const void* p : ps)
    if (p != nullptr && (reinterpret_cast<uintptr_t>(p) & 15u) != 0) return false;
  return true;
}

template <int S>
__global__ void layer_norm_vec(const bf16* x, const bf16* weight, const bf16* bias, bf16* y,
                               int cols, float eps) {
  __shared__ float scratch[kReduceBlock];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float cache[S][8];
  float sum = 0.0f;
#pragma unroll
  for (int s = 0; s < S; ++s) {
    const int v0 = threadIdx.x * 8 + s * 2048;
    if (v0 < cols) {
      V8 raw;
      raw.raw = ld8(x + base + v0);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float v = bf16_to_f32(raw.e[j]);
        cache[s][j] = v;
        sum += v;
      }
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) cache[s][j] = 0.0f;
    }
  }
  const float mean = block_sum(sum, scratch) / cols;
  float centered_sum = 0.0f;
#pragma unroll
  for (int s = 0; s < S; ++s) {
    if (threadIdx.x * 8 + s * 2048 < cols) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float centered = cache[s][j] - mean;
        centered_sum += centered * centered;
      }
    }
  }
  const float inverse_std = rsqrtf(block_sum(centered_sum, scratch) / cols + eps);
#pragma unroll
  for (int s = 0; s < S; ++s) {
    const int v0 = threadIdx.x * 8 + s * 2048;
    if (v0 < cols) {
      V8 w, b, o;
      w.raw = ld8(weight + v0);
      if (bias != nullptr) b.raw = ld8(bias + v0);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        o.e[j] = f32_to_bf16((cache[s][j] - mean) * inverse_std * bf16_to_f32(w.e[j]) +
                             (bias != nullptr ? bf16_to_f32(b.e[j]) : 0.0f));
      st8(y + base + v0, o.raw);
    }
  }
}

template <int S>
__global__ void ada_layer_norm_vec(const bf16* x, const bf16* style, bf16* y, int cols,
                                   float eps) {
  __shared__ float scratch[kReduceBlock];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float cache[S][8];
  float sum = 0.0f;
#pragma unroll
  for (int s = 0; s < S; ++s) {
    const int v0 = threadIdx.x * 8 + s * 2048;
    if (v0 < cols) {
      V8 raw;
      raw.raw = ld8(x + base + v0);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float v = bf16_to_f32(raw.e[j]);
        cache[s][j] = v;
        sum += v;
      }
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) cache[s][j] = 0.0f;
    }
  }
  const float mean = block_sum(sum, scratch) / cols;
  float centered_sum = 0.0f;
#pragma unroll
  for (int s = 0; s < S; ++s) {
    if (threadIdx.x * 8 + s * 2048 < cols) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float centered = cache[s][j] - mean;
        centered_sum += centered * centered;
      }
    }
  }
  const float inverse_std = rsqrtf(block_sum(centered_sum, scratch) / cols + eps);
#pragma unroll
  for (int s = 0; s < S; ++s) {
    const int v0 = threadIdx.x * 8 + s * 2048;
    if (v0 < cols) {
      V8 sc, sh, o;
      sc.raw = ld8(style + v0);
      sh.raw = ld8(style + cols + v0);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float normalized = (cache[s][j] - mean) * inverse_std;
        o.e[j] = f32_to_bf16(normalized * (1.0f + bf16_to_f32(sc.e[j])) + bf16_to_f32(sh.e[j]));
      }
      st8(y + base + v0, o.raw);
    }
  }
}

template <int S>
__global__ void ada_rms_norm_vec(const bf16* x, const bf16* style, bf16* y, int cols,
                                 float eps) {
  __shared__ float scratch[kReduceBlock];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float cache[S][8];
  float square_sum = 0.0f;
#pragma unroll
  for (int s = 0; s < S; ++s) {
    const int v0 = threadIdx.x * 8 + s * 2048;
    if (v0 < cols) {
      V8 raw;
      raw.raw = ld8(x + base + v0);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float v = bf16_to_f32(raw.e[j]);
        cache[s][j] = v;
        square_sum += v * v;
      }
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) cache[s][j] = 0.0f;
    }
  }
  const float inverse_rms = rsqrtf(block_sum(square_sum, scratch) / cols + eps);
#pragma unroll
  for (int s = 0; s < S; ++s) {
    const int v0 = threadIdx.x * 8 + s * 2048;
    if (v0 < cols) {
      V8 sc, sh, o;
      sc.raw = ld8(style + v0);
      sh.raw = ld8(style + cols + v0);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float normalized = cache[s][j] * inverse_rms;
        o.e[j] = f32_to_bf16(normalized * (1.0f + bf16_to_f32(sc.e[j])) + bf16_to_f32(sh.e[j]));
      }
      st8(y + base + v0, o.raw);
    }
  }
}

// Shared pass 1 of the fused residual family: hidden = round(projection +
// residual (+ bias)), stored to `hidden` and cached (as the rounded value the
// norm consumes) in `cache`.
template <int S>
__device__ __forceinline__ void fused_store_hidden(const bf16* projection, const bf16* bias,
                                                   const bf16* residual, bf16* hidden,
                                                   int cols, float (&cache)[S][8]) {
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
#pragma unroll
  for (int s = 0; s < S; ++s) {
    const int v0 = threadIdx.x * 8 + s * 2048;
    if (v0 < cols) {
      V8 p, r, b, h;
      p.raw = ld8(projection + base + v0);
      r.raw = ld8(residual + base + v0);
      if (bias != nullptr) b.raw = ld8(bias + v0);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float v = bf16_to_f32(p.e[j]) + bf16_to_f32(r.e[j]);
        if (bias != nullptr) v += bf16_to_f32(b.e[j]);
        h.e[j] = f32_to_bf16(v);
        cache[s][j] = bf16_to_f32(h.e[j]);
      }
      st8(hidden + base + v0, h.raw);
    }
  }
}

template <int S>
__global__ void bias_residual_rms_norm_vec(const bf16* projection, const bf16* bias,
                                           const bf16* residual, const bf16* weight,
                                           bf16* hidden, bf16* normalized, int cols, float eps) {
  __shared__ float scratch[kReduceBlock];
  float cache[S][8];
  fused_store_hidden<S>(projection, bias, residual, hidden, cols, cache);
  float square_sum = 0.0f;
#pragma unroll
  for (int s = 0; s < S; ++s) {
    if (threadIdx.x * 8 + s * 2048 < cols) {
#pragma unroll
      for (int j = 0; j < 8; ++j) square_sum += cache[s][j] * cache[s][j];
    }
  }
  const float inverse_rms = rsqrtf(block_sum(square_sum, scratch) / cols + eps);
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
#pragma unroll
  for (int s = 0; s < S; ++s) {
    const int v0 = threadIdx.x * 8 + s * 2048;
    if (v0 < cols) {
      V8 w, o;
      w.raw = ld8(weight + v0);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        o.e[j] = f32_to_bf16(cache[s][j] * inverse_rms * bf16_to_f32(w.e[j]));
      st8(normalized + base + v0, o.raw);
    }
  }
}

template <int S>
__global__ void bias_residual_layer_norm_vec(const bf16* projection, const bf16* bias,
                                             const bf16* residual, const bf16* norm_weight,
                                             const bf16* norm_bias, bf16* hidden, bf16* normalized,
                                             int cols, float eps) {
  __shared__ float scratch[kReduceBlock];
  float cache[S][8];
  fused_store_hidden<S>(projection, bias, residual, hidden, cols, cache);
  float sum = 0.0f;
#pragma unroll
  for (int s = 0; s < S; ++s) {
    if (threadIdx.x * 8 + s * 2048 < cols) {
#pragma unroll
      for (int j = 0; j < 8; ++j) sum += cache[s][j];
    }
  }
  const float mean = block_sum(sum, scratch) / cols;
  float centered_sum = 0.0f;
#pragma unroll
  for (int s = 0; s < S; ++s) {
    if (threadIdx.x * 8 + s * 2048 < cols) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float centered = cache[s][j] - mean;
        centered_sum += centered * centered;
      }
    }
  }
  const float inverse_std = rsqrtf(block_sum(centered_sum, scratch) / cols + eps);
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
#pragma unroll
  for (int s = 0; s < S; ++s) {
    const int v0 = threadIdx.x * 8 + s * 2048;
    if (v0 < cols) {
      V8 w, b, o;
      w.raw = ld8(norm_weight + v0);
      b.raw = ld8(norm_bias + v0);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        o.e[j] = f32_to_bf16((cache[s][j] - mean) * inverse_std * bf16_to_f32(w.e[j]) +
                             bf16_to_f32(b.e[j]));
      st8(normalized + base + v0, o.raw);
    }
  }
}

template <int S>
__global__ void bias_residual_ada_layer_norm_vec(const bf16* projection, const bf16* bias,
                                                 const bf16* residual, const bf16* style,
                                                 bf16* hidden, bf16* normalized, int cols,
                                                 float eps) {
  __shared__ float scratch[kReduceBlock];
  float cache[S][8];
  fused_store_hidden<S>(projection, bias, residual, hidden, cols, cache);
  float sum = 0.0f;
#pragma unroll
  for (int s = 0; s < S; ++s) {
    if (threadIdx.x * 8 + s * 2048 < cols) {
#pragma unroll
      for (int j = 0; j < 8; ++j) sum += cache[s][j];
    }
  }
  const float mean = block_sum(sum, scratch) / cols;
  float centered_sum = 0.0f;
#pragma unroll
  for (int s = 0; s < S; ++s) {
    if (threadIdx.x * 8 + s * 2048 < cols) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float centered = cache[s][j] - mean;
        centered_sum += centered * centered;
      }
    }
  }
  const float inverse_std = rsqrtf(block_sum(centered_sum, scratch) / cols + eps);
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
#pragma unroll
  for (int s = 0; s < S; ++s) {
    const int v0 = threadIdx.x * 8 + s * 2048;
    if (v0 < cols) {
      V8 sc, sh, o;
      sc.raw = ld8(style + v0);
      sh.raw = ld8(style + cols + v0);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float v = (cache[s][j] - mean) * inverse_std;
        o.e[j] = f32_to_bf16(v * (1.0f + bf16_to_f32(sc.e[j])) + bf16_to_f32(sh.e[j]));
      }
      st8(normalized + base + v0, o.raw);
    }
  }
}

template <int S>
__global__ void ada_gate_residual_rms_norm_vec(const bf16* projection, const bf16* residual,
                                               const bf16* gate_style, const bf16* norm_style,
                                               bf16* hidden, bf16* normalized, int cols,
                                               float eps) {
  __shared__ float scratch[kReduceBlock];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float cache[S][8];
  float square_sum = 0.0f;
#pragma unroll
  for (int s = 0; s < S; ++s) {
    const int v0 = threadIdx.x * 8 + s * 2048;
    if (v0 < cols) {
      V8 p, r, g, h;
      p.raw = ld8(projection + base + v0);
      r.raw = ld8(residual + base + v0);
      g.raw = ld8(gate_style + 2 * cols + v0);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const bf16 rounded = f32_to_bf16(
            bf16_to_f32(r.e[j]) + bf16_to_f32(p.e[j]) * bf16_to_f32(g.e[j]));
        h.e[j] = rounded;
        const float stored = bf16_to_f32(rounded);
        cache[s][j] = stored;
        square_sum += stored * stored;
      }
      st8(hidden + base + v0, h.raw);
    }
  }
  const float inverse_rms = rsqrtf(block_sum(square_sum, scratch) / cols + eps);
#pragma unroll
  for (int s = 0; s < S; ++s) {
    const int v0 = threadIdx.x * 8 + s * 2048;
    if (v0 < cols) {
      V8 sc, sh, o;
      sc.raw = ld8(norm_style + v0);
      sh.raw = ld8(norm_style + cols + v0);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float v = cache[s][j] * inverse_rms;
        o.e[j] = f32_to_bf16(v * (1.0f + bf16_to_f32(sc.e[j])) + bf16_to_f32(sh.e[j]));
      }
      st8(normalized + base + v0, o.raw);
    }
  }
}

const bf16* in(const void* p) { return static_cast<const bf16*>(p); }
bf16* out(void* p) { return static_cast<bf16*>(p); }

bool valid_rows(int64_t rows, int cols) { return rows > 0 && cols > 0 && rows <= INT32_MAX; }

}  // namespace

extern "C" int apxinf_hip_native_layer_norm(const void* x, const void* weight, const void* bias,
                                            void* y, int64_t rows, int cols, float eps,
                                            void* stream) {
  if (!valid_rows(rows, cols)) return kInvalidArgument;
  if (vec_slices(cols) > 0 && aligned16({x, weight, bias, y})) {
    switch (vec_slices(cols)) {
      case 1:
        layer_norm_vec<1><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(x), in(weight), in(bias), out(y), cols, eps);
        return last_launch_status();
      case 2:
        layer_norm_vec<2><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(x), in(weight), in(bias), out(y), cols, eps);
        return last_launch_status();
      case 3:
        layer_norm_vec<3><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(x), in(weight), in(bias), out(y), cols, eps);
        return last_launch_status();
      case 4:
        layer_norm_vec<4><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(x), in(weight), in(bias), out(y), cols, eps);
        return last_launch_status();
      default: break;
    }
  }
  layer_norm_kernel<<<static_cast<unsigned>(rows), kReduceBlock, 0, as_stream(stream)>>>(
      in(x), in(weight), in(bias), out(y), cols, eps);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_ada_layer_norm(const void* x, const void* style, void* y,
                                               int64_t rows, int cols, float eps, void* stream) {
  if (rows <= 0 || cols <= 0) return kInvalidArgument;
  if (vec_slices(cols) > 0 && aligned16({x, style, y})) {
    switch (vec_slices(cols)) {
      case 1:
        ada_layer_norm_vec<1><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(x), in(style), out(y), cols, eps);
        return last_launch_status();
      case 2:
        ada_layer_norm_vec<2><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(x), in(style), out(y), cols, eps);
        return last_launch_status();
      case 3:
        ada_layer_norm_vec<3><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(x), in(style), out(y), cols, eps);
        return last_launch_status();
      case 4:
        ada_layer_norm_vec<4><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(x), in(style), out(y), cols, eps);
        return last_launch_status();
      default: break;
    }
  }
  ada_layer_norm_kernel<<<static_cast<unsigned>(rows), kReduceBlock, 0, as_stream(stream)>>>(
      in(x), in(style), out(y), cols, eps);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_ada_rms_norm(const void* x, const void* style, void* y,
                                              int64_t rows, int cols, float eps, void* stream) {
  if (!valid_rows(rows, cols)) return kInvalidArgument;
  if (vec_slices(cols) > 0 && aligned16({x, style, y})) {
    switch (vec_slices(cols)) {
      case 1:
        ada_rms_norm_vec<1><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(x), in(style), out(y), cols, eps);
        return last_launch_status();
      case 2:
        ada_rms_norm_vec<2><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(x), in(style), out(y), cols, eps);
        return last_launch_status();
      case 3:
        ada_rms_norm_vec<3><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(x), in(style), out(y), cols, eps);
        return last_launch_status();
      case 4:
        ada_rms_norm_vec<4><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(x), in(style), out(y), cols, eps);
        return last_launch_status();
      default: break;
    }
  }
  ada_rms_norm_kernel<<<static_cast<unsigned>(rows), kReduceBlock, 0, as_stream(stream)>>>(
      in(x), in(style), out(y), cols, eps);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_bias_residual(const void* projection, const void* bias,
                                               const void* residual, void* hidden, int64_t rows,
                                               int cols, void* stream) {
  if (!valid_rows(rows, cols)) return kInvalidArgument;
  const int64_t count = rows * cols;
  if (cols % 8 == 0 && count % 8 == 0 && aligned16({projection, bias, residual, hidden})) {
    if (bias != nullptr) {
      bias_residual_vec_kernel<true><<<grid_for(count / 8, kElementwiseBlock), kElementwiseBlock,
                                      0, as_stream(stream)>>>(in(projection), in(bias),
                                                              in(residual), out(hidden), count / 8,
                                                              cols / 8);
    } else {
      bias_residual_vec_kernel<false><<<grid_for(count / 8, kElementwiseBlock), kElementwiseBlock,
                                       0, as_stream(stream)>>>(in(projection), in(bias),
                                                               in(residual), out(hidden), count / 8,
                                                               cols / 8);
    }
    return last_launch_status();
  }
  bias_residual_kernel<<<grid_for(count, kElementwiseBlock), kElementwiseBlock, 0,
                         as_stream(stream)>>>(in(projection), in(bias), in(residual), out(hidden),
                                              count, cols);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_bias_residual_rms_norm(const void* projection, const void* bias,
                                                        const void* residual, const void* weight,
                                                        void* hidden, void* normalized,
                                                        int64_t rows, int cols, float eps,
                                                        void* stream) {
  if (!valid_rows(rows, cols)) return kInvalidArgument;
  if (vec_slices(cols) > 0 && aligned16({projection, bias, residual, weight, hidden, normalized})) {
    switch (vec_slices(cols)) {
      case 1:
        bias_residual_rms_norm_vec<1><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(projection), in(bias), in(residual), in(weight), out(hidden), out(normalized), cols, eps);
        return last_launch_status();
      case 2:
        bias_residual_rms_norm_vec<2><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(projection), in(bias), in(residual), in(weight), out(hidden), out(normalized), cols, eps);
        return last_launch_status();
      case 3:
        bias_residual_rms_norm_vec<3><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(projection), in(bias), in(residual), in(weight), out(hidden), out(normalized), cols, eps);
        return last_launch_status();
      case 4:
        bias_residual_rms_norm_vec<4><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(projection), in(bias), in(residual), in(weight), out(hidden), out(normalized), cols, eps);
        return last_launch_status();
      default: break;
    }
  }
  bias_residual_rms_norm_kernel<<<static_cast<unsigned>(rows), kReduceBlock, 0,
                                  as_stream(stream)>>>(in(projection), in(bias), in(residual),
                                                       in(weight), out(hidden), out(normalized),
                                                       cols, eps);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_bias_residual_layer_norm(
    const void* projection, const void* bias, const void* residual, const void* norm_weight,
    const void* norm_bias, void* hidden, void* normalized, int64_t rows, int cols, float eps,
    void* stream) {
  if (!valid_rows(rows, cols)) return kInvalidArgument;
  if (vec_slices(cols) > 0 && aligned16({projection, bias, residual, norm_weight, norm_bias, hidden, normalized})) {
    switch (vec_slices(cols)) {
      case 1:
        bias_residual_layer_norm_vec<1><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(projection), in(bias), in(residual), in(norm_weight), in(norm_bias), out(hidden), out(normalized), cols, eps);
        return last_launch_status();
      case 2:
        bias_residual_layer_norm_vec<2><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(projection), in(bias), in(residual), in(norm_weight), in(norm_bias), out(hidden), out(normalized), cols, eps);
        return last_launch_status();
      case 3:
        bias_residual_layer_norm_vec<3><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(projection), in(bias), in(residual), in(norm_weight), in(norm_bias), out(hidden), out(normalized), cols, eps);
        return last_launch_status();
      case 4:
        bias_residual_layer_norm_vec<4><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(projection), in(bias), in(residual), in(norm_weight), in(norm_bias), out(hidden), out(normalized), cols, eps);
        return last_launch_status();
      default: break;
    }
  }
  bias_residual_layer_norm_kernel<<<static_cast<unsigned>(rows), kReduceBlock, 0,
                                    as_stream(stream)>>>(
      in(projection), in(bias), in(residual), in(norm_weight), in(norm_bias), out(hidden),
      out(normalized), cols, eps);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_bias_residual_ada_layer_norm(
    const void* projection, const void* bias, const void* residual, const void* style,
    void* hidden, void* normalized, int64_t rows, int cols, float eps, void* stream) {
  if (rows <= 0 || cols <= 0) return kInvalidArgument;
  if (vec_slices(cols) > 0 && aligned16({projection, bias, residual, style, hidden, normalized})) {
    switch (vec_slices(cols)) {
      case 1:
        bias_residual_ada_layer_norm_vec<1><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(projection), in(bias), in(residual), in(style), out(hidden), out(normalized), cols, eps);
        return last_launch_status();
      case 2:
        bias_residual_ada_layer_norm_vec<2><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(projection), in(bias), in(residual), in(style), out(hidden), out(normalized), cols, eps);
        return last_launch_status();
      case 3:
        bias_residual_ada_layer_norm_vec<3><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(projection), in(bias), in(residual), in(style), out(hidden), out(normalized), cols, eps);
        return last_launch_status();
      case 4:
        bias_residual_ada_layer_norm_vec<4><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(projection), in(bias), in(residual), in(style), out(hidden), out(normalized), cols, eps);
        return last_launch_status();
      default: break;
    }
  }
  bias_residual_ada_layer_norm_kernel<<<static_cast<unsigned>(rows), kReduceBlock, 0,
                                        as_stream(stream)>>>(
      in(projection), in(bias), in(residual), in(style), out(hidden), out(normalized), cols,
      eps);
  return last_launch_status();
}

extern "C" int apxinf_hip_native_ada_gate_residual_rms_norm(
    const void* projection, const void* residual, const void* gate_style, const void* norm_style,
    void* hidden, void* normalized, int64_t rows, int cols, float eps, void* stream) {
  if (!valid_rows(rows, cols)) return kInvalidArgument;
  if (vec_slices(cols) > 0 && aligned16({projection, residual, gate_style, norm_style, hidden, normalized})) {
    switch (vec_slices(cols)) {
      case 1:
        ada_gate_residual_rms_norm_vec<1><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(projection), in(residual), in(gate_style), in(norm_style), out(hidden), out(normalized), cols, eps);
        return last_launch_status();
      case 2:
        ada_gate_residual_rms_norm_vec<2><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(projection), in(residual), in(gate_style), in(norm_style), out(hidden), out(normalized), cols, eps);
        return last_launch_status();
      case 3:
        ada_gate_residual_rms_norm_vec<3><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(projection), in(residual), in(gate_style), in(norm_style), out(hidden), out(normalized), cols, eps);
        return last_launch_status();
      case 4:
        ada_gate_residual_rms_norm_vec<4><<<(unsigned)rows, kReduceBlock, 0, as_stream(stream)>>>(in(projection), in(residual), in(gate_style), in(norm_style), out(hidden), out(normalized), cols, eps);
        return last_launch_status();
      default: break;
    }
  }
  ada_gate_residual_rms_norm_kernel<<<static_cast<unsigned>(rows), kReduceBlock, 0,
                                      as_stream(stream)>>>(
      in(projection), in(residual), in(gate_style), in(norm_style), out(hidden), out(normalized),
      cols, eps);
  return last_launch_status();
}
