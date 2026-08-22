#pragma once
// Qwen3.5-specific bf16 device kernels (C1 eager path). Launchers live in
// adapters/custom_kernels.cu; safe Rust wrappers in kernels/qwen35.rs.

#include <cuda_fp16.h>
#include <cstdint>

__device__ __forceinline__ float q35_bf16(const __nv_bfloat16* p, size_t i) {
  return __bfloat162float(p[i]);
}

__device__ __forceinline__ __nv_bfloat16 q35_b16(float v) {
  return __float2bfloat16(v);
}

// ── AWQ W4A16 → bf16, transposed layout [in, out] for cuBLAS row-major ──
__global__ void qwen_dequant_w4a16_bf16_kernel(
    const int32_t* packed,          // [out, in/8]
    const __nv_bfloat16* scale,     // [out, groups]
    const int32_t* zp,              // [ceil(out/8), groups]
    __nv_bfloat16* out,             // [in, out] (transposed)
    int out_dim, int in_dim) {
  int groups = in_dim / 32;
  int64_t total = (int64_t)out_dim * in_dim;
  for (int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       e < total;
       e += (int64_t)gridDim.x * blockDim.x) {
    int o = (int)(e / in_dim);
    int i = (int)(e % in_dim);
    int g = i / 32;
    int j8 = i % 8;
    int zpv = ((zp[(o / 8) * groups + g] >> (4 * (o % 8))) & 0xF) - 8;
    int w4 = ((packed[o * (in_dim / 8) + i / 8] >> (4 * j8)) & 0xF) - 8;
    float s = __bfloat162float(scale[o * groups + g]);
    out[(int64_t)i * out_dim + o] = __float2bfloat16(s * ((float)w4 - (float)zpv));
  }
}

// ── elementwise ─────────────────────────────────────────────────────────
__global__ void qwen_mul_bf16_kernel(
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    __nv_bfloat16* out, int64_t n) {
  int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n) return;
  out[e] = q35_b16(q35_bf16(a, e) * q35_bf16(b, e));
}

__global__ void qwen_silu_bf16_kernel(
    const __nv_bfloat16* a, __nv_bfloat16* out, int64_t n) {
  int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n) return;
  float x = q35_bf16(a, e);
  out[e] = q35_b16(x / (1.0f + expf(-x)));
}

__global__ void qwen_sigmoid_mul_bf16_kernel(
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    __nv_bfloat16* out, int64_t n) {
  int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n) return;
  float g = q35_bf16(a, e);
  float sig = 1.0f / (1.0f + expf(-g));
  out[e] = q35_b16(sig * q35_bf16(b, e));
}

// dst += src
__global__ void qwen_accum_bf16_kernel(
    __nv_bfloat16* dst, const __nv_bfloat16* src, int64_t n) {
  int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n) return;
  dst[e] = q35_b16(q35_bf16(dst, e) + q35_bf16(src, e));
}

// ── RMSNorm with precomputed (1+w) weights ──────────────────────────────
__global__ void qwen_rms_norm_bf16_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* w,
    __nv_bfloat16* out, int rows, int cols, float eps) {
  extern __shared__ float red[];
  int row = blockIdx.x;
  if (row >= rows) return;
  int tid = threadIdx.x;
  float acc = 0.0f;
  for (int j = tid; j < cols; j += blockDim.x) {
    float v = q35_bf16(x, (int64_t)row * cols + j);
    acc += v * v;
  }
  red[tid] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  float rstd = rsqrtf(red[0] / (float)cols + eps);
  for (int j = tid; j < cols; j += blockDim.x) {
    float v = q35_bf16(x, (int64_t)row * cols + j);
    float wv = q35_bf16(w, j);
    out[(int64_t)row * cols + j] = q35_b16(v * rstd * wv);
  }
}

// ── split qg [L, heads, 2*hd] into q / gate [L*heads, hd] ───────────────
__global__ void qwen_qg_split_bf16_kernel(
    const __nv_bfloat16* qg, __nv_bfloat16* q, __nv_bfloat16* gate,
    int64_t total_elems, int heads, int hd) {
  int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= total_elems) return;
  int t = (int)(e / ((int64_t)heads * hd));
  int rem = (int)(e % ((int64_t)heads * hd));
  int h = rem / hd;
  int j = rem % hd;
  int64_t base = ((int64_t)t * heads + h) * (2 * hd);
  q[e] = qg[base + j];
  gate[e] = qg[base + hd + j];
}

// ── partial RoPE (first rotary dims, interleaved split-half) in-place ───
__global__ void qwen_partial_rope_bf16_kernel(
    __nv_bfloat16* x, const __nv_bfloat16* cos, const __nv_bfloat16* sin,
    int64_t pairs, int heads, int hd, int half) {
  // one thread per (row, i) pair; row = e / half, i = e % half
  int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= pairs) return;
  int row = (int)(e / half);
  int i = (int)(e % half);
  int t = row / heads;                 // token index from flattened rows
  float c = q35_bf16(cos, (int64_t)t * half + i);
  float s = q35_bf16(sin, (int64_t)t * half + i);
  float a = q35_bf16(x, (int64_t)row * hd + i);
  float b = q35_bf16(x, (int64_t)row * hd + half + i);
  x[(int64_t)row * hd + i] = q35_b16(a * c - b * s);
  x[(int64_t)row * hd + half + i] = q35_b16(a * s + b * c);
}

// ── depthwise causal conv kernel=4 + SiLU ───────────────────────────────
__global__ void qwen_conv_silu_bf16_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* w,
    __nv_bfloat16* out, int L, int conv_dim) {
  int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = (int64_t)L * conv_dim;
  if (e >= total) return;
  int t = (int)(e / conv_dim);
  int c = (int)(e % conv_dim);
  float acc = 0.0f;
  for (int j = 0; j < 4; j++) {
    if (t >= j) {
      acc += q35_bf16(w, (int64_t)c * 4 + j) *
             q35_bf16(x, (int64_t)(t - j) * conv_dim + c);
    }
  }
  out[e] = q35_b16(acc / (1.0f + expf(-acc)));
}

// ── GatedDeltaNet recurrence (fp32 state) ───────────────────────────────
// grid = (nv); block = vd threads. Thread j owns state column j for head h;
// all reads/writes of column j are thread-local, so no cross-thread sync.
__global__ void qwen_delta_recurrence_bf16_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const __nv_bfloat16* beta, const __nv_bfloat16* g,
    float* state,                    // [nv, kd, vd]
    __nv_bfloat16* out,              // [L, nv, vd]
    int L, int nv, int kd, int vd) {
  int h = blockIdx.x;
  int j = threadIdx.x;
  if (h >= nv || j >= vd) return;
  float* S = state + (int64_t)h * kd * vd;
  // zero this thread's state columns: collectively resets the whole
  // per-layer state matrix before the recurrence.
  for (int kk = 0; kk < kd; kk++) {
    S[(int64_t)kk * vd + j] = 0.0f;
  }
  for (int t = 0; t < L; t++) {
    float decay = expf(q35_bf16(g, (int64_t)t * nv + h));
    for (int kk = 0; kk < kd; kk++) {
      S[(int64_t)kk * vd + j] *= decay;
    }
    float kv = 0.0f;
    for (int kk = 0; kk < kd; kk++) {
      kv += S[(int64_t)kk * vd + j] * q35_bf16(k, ((int64_t)t * nv + h) * kd + kk);
    }
    float delta = (q35_bf16(v, ((int64_t)t * nv + h) * vd + j) - kv) *
                  q35_bf16(beta, (int64_t)t * nv + h);
    for (int kk = 0; kk < kd; kk++) {
      S[(int64_t)kk * vd + j] += q35_bf16(k, ((int64_t)t * nv + h) * kd + kk) * delta;
    }
    float o = 0.0f;
    for (int kk = 0; kk < kd; kk++) {
      o += S[(int64_t)kk * vd + j] * q35_bf16(q, ((int64_t)t * nv + h) * kd + kk);
    }
    out[((int64_t)t * nv + h) * vd + j] = q35_b16(o);
  }
}

// ── Full attention for short prefill (L <= 128) ─────────────────────────
// grid = (L); block = heads threads. Each thread computes one (t, h) output.
__global__ void qwen_attention_bf16_kernel(
    const __nv_bfloat16* q,     // [L, heads, hd]
    const __nv_bfloat16* k,     // [L, kvheads, hd]
    const __nv_bfloat16* v,     // [L, kvheads, hd]
    const __nv_bfloat16* gate,  // [L, heads, hd]
    __nv_bfloat16* out,         // [L, heads, hd]
    int L, int heads, int kvheads, int hd) {
  float scores[128];
  int t = blockIdx.x;
  int h = threadIdx.x;
  if (t >= L || h >= heads) return;
  int kh = h / (heads / kvheads);
  float scale = rsqrtf((float)hd);
  float maxv = -1e30f;
  for (int s = 0; s <= t; s++) {
    float dot = 0.0f;
    for (int d = 0; d < hd; d++) {
      dot += q35_bf16(q, ((int64_t)t * heads + h) * hd + d) *
             q35_bf16(k, ((int64_t)s * kvheads + kh) * hd + d);
    }
    dot *= scale;
    scores[s] = dot;
    if (dot > maxv) maxv = dot;
  }
  float sum = 0.0f;
  for (int s = 0; s <= t; s++) {
    scores[s] = expf(scores[s] - maxv);
    sum += scores[s];
  }
  float inv_sum = 1.0f / sum;
  for (int d = 0; d < hd; d++) {
    float acc = 0.0f;
    for (int s = 0; s <= t; s++) {
      acc += scores[s] * q35_bf16(v, ((int64_t)s * kvheads + kh) * hd + d);
    }
    float g = q35_bf16(gate, ((int64_t)t * heads + h) * hd + d);
    float sig = 1.0f / (1.0f + expf(-g));
    out[((int64_t)t * heads + h) * hd + d] = q35_b16(acc * inv_sum * sig);
  }
}

// ── expand conv output into per-head q/k (GQA 3x) and v ─────────────────
__global__ void qwen_conv_split_bf16_kernel(
    const __nv_bfloat16* conv, __nv_bfloat16* q, __nv_bfloat16* k,
    __nv_bfloat16* v, int L, int nk, int nv, int kd, int vd, int conv_dim) {
  int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t qk = (int64_t)L * nv * kd;
  int64_t vv = (int64_t)L * nv * vd;
  int reps = nv / nk;
  if (e < qk) {
    int t = (int)(e / ((int64_t)nv * kd));
    int rem = (int)(e % ((int64_t)nv * kd));
    int h = rem / kd;
    int j = rem % kd;
    int r = h / reps;
    q[e] = conv[(int64_t)t * conv_dim + (int64_t)r * kd + j];
    k[e] = conv[(int64_t)t * conv_dim + (int64_t)nk * kd + (int64_t)r * kd + j];
  } else if (e < qk + vv) {
    int64_t f = e - qk;
    int t = (int)(f / ((int64_t)nv * vd));
    int rem = (int)(f % ((int64_t)nv * vd));
    int h = rem / vd;
    int j = rem % vd;
    v[f] = conv[(int64_t)t * conv_dim + 2 * (int64_t)nk * kd + (int64_t)h * vd + j];
  }
}

// ── L2 normalization (no mean) then scalar scale: x/sqrt(sum+eps)*scale ──
__global__ void qwen_l2norm_bf16_kernel(
    const __nv_bfloat16* x, __nv_bfloat16* out,
    int rows, int cols, float eps, float scale) {
  extern __shared__ float red[];
  int row = blockIdx.x;
  if (row >= rows) return;
  int tid = threadIdx.x;
  float acc = 0.0f;
  for (int j = tid; j < cols; j += blockDim.x) {
    float v = q35_bf16(x, (int64_t)row * cols + j);
    acc += v * v;
  }
  red[tid] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  float rstd = rsqrtf(red[0] + eps) * scale;
  for (int j = tid; j < cols; j += blockDim.x) {
    float v = q35_bf16(x, (int64_t)row * cols + j);
    out[(int64_t)row * cols + j] = q35_b16(v * rstd);
  }
}
