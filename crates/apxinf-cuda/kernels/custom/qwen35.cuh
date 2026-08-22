#pragma once
// Qwen3.5-specific bf16 device kernels (C1 eager path). Launchers live in
// adapters/custom_kernels.cu; safe Rust wrappers in kernels/qwen35.rs.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define DR_KDCHUNK 4
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
  // Process two channels per thread (bf16x2), keeping explicit causal taps.
  int64_t total = (int64_t)L * (conv_dim / 2);
  for (int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       e < total;
       e += (int64_t)gridDim.x * blockDim.x) {
    int t = (int)(e / (conv_dim / 2));
    int cp = (int)(e % (conv_dim / 2));
    int c0 = 2 * cp;
    float acc0 = 0.0f, acc1 = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      if (t >= j) {
        __nv_bfloat162 xv = *reinterpret_cast<const __nv_bfloat162*>(
            &x[(int64_t)(t - j) * conv_dim + c0]);
        float x0 = __bfloat162float(__low2bfloat16(xv));
        float x1 = __bfloat162float(__high2bfloat16(xv));
        float w0 = __bfloat162float(w[(int64_t)c0 * 4 + j]);
        float w1 = __bfloat162float(w[(int64_t)(c0 + 1) * 4 + j]);
        acc0 += w0 * x0;
        acc1 += w1 * x1;
      }
    }
    out[(int64_t)t * conv_dim + c0] = q35_b16(acc0 / (1.0f + expf(-acc0)));
    out[(int64_t)t * conv_dim + c0 + 1] = q35_b16(acc1 / (1.0f + expf(-acc1)));
  }
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
  // grid = (nv), block = (vd, DR_KDCHUNK). Each thread owns kd-slice
  // [kk0,kk1) of state column j; the kd reductions are split across the
  // y dimension and combined through shared memory per time step.
  int h = blockIdx.x;
  int j = threadIdx.x;
  int y = threadIdx.y;
  if (h >= nv || j >= vd) return;

  const int chunk = kd / DR_KDCHUNK;
  const int kk0 = y * chunk;
  const int kk1 = kk0 + chunk;
  float* S = state + (int64_t)h * kd * vd;

  // zero this thread's state slice (collectively resets the whole matrix)
  for (int kk = kk0; kk < kk1; kk++) {
    S[(int64_t)kk * vd + j] = 0.0f;
  }

  __shared__ float kv_part[DR_KDCHUNK][128];
  __shared__ float o_part[DR_KDCHUNK][128];
  __shared__ float delta_s[128];

  for (int t = 0; t < L; t++) {
    float decay = expf(q35_bf16(g, (int64_t)t * nv + h));
    const __nv_bfloat16* krow = k + ((int64_t)t * nv + h) * kd;
    const __nv_bfloat16* qrow = q + ((int64_t)t * nv + h) * kd;

    for (int kk = kk0; kk < kk1; kk++) {
      S[(int64_t)kk * vd + j] *= decay;
    }

    float pkv = 0.0f;
    for (int kk = kk0; kk < kk1; kk++) {
      pkv += S[(int64_t)kk * vd + j] * q35_bf16(krow, kk);
    }
    kv_part[y][j] = pkv;
    __syncthreads();

    if (y == 0) {
      float kv = kv_part[0][j];
      #pragma unroll
      for (int yy = 1; yy < DR_KDCHUNK; yy++) kv += kv_part[yy][j];
      float delta =
          (q35_bf16(v, ((int64_t)t * nv + h) * vd + j) - kv) *
          q35_bf16(beta, (int64_t)t * nv + h);
      delta_s[j] = delta;
    }
    __syncthreads();

    float dlt = delta_s[j];
    for (int kk = kk0; kk < kk1; kk++) {
      S[(int64_t)kk * vd + j] += q35_bf16(krow, kk) * dlt;
    }

    float po = 0.0f;
    for (int kk = kk0; kk < kk1; kk++) {
      po += S[(int64_t)kk * vd + j] * q35_bf16(qrow, kk);
    }
    o_part[y][j] = po;
    __syncthreads();

    if (y == 0) {
      float o = o_part[0][j];
      #pragma unroll
      for (int yy = 1; yy < DR_KDCHUNK; yy++) o += o_part[yy][j];
      out[((int64_t)t * nv + h) * vd + j] = q35_b16(o);
    }
    __syncthreads();
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
  // One block per (t, h); blockDim.x == hd. Causal, GQA-grouped, online
  // softmax with warp-shuffle reductions (no per-s global atomics).
  int bid = blockIdx.x;
  int t = bid / heads;
  int h = bid % heads;
  if (t >= L || h >= heads) return;
  int d = threadIdx.x;
  if (d >= hd) return;
  int kh = h / (heads / kvheads);

  __shared__ float warp_part[8];
  __shared__ float total_s[1];

  float qv = q35_bf16(q, ((int64_t)t * heads + h) * hd + d);
  float gv = q35_bf16(gate, ((int64_t)t * heads + h) * hd + d);
  float acc = 0.0f;
  float m = -1e30f;
  float l = 0.0f;
  float scale = rsqrtf((float)hd);

  for (int s = 0; s <= t; s++) {
    float part = qv * q35_bf16(k, ((int64_t)s * kvheads + kh) * hd + d);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      part += __shfl_xor_sync(0xffffffffu, part, off);
    }
    if ((d & 31) == 0) warp_part[d >> 5] = part;
    __syncthreads();
    if (d < 8) {
      float wp = warp_part[d];
      #pragma unroll
      for (int off = 4; off > 0; off >>= 1) {
        wp += __shfl_xor_sync(0x000000ffu, wp, off);
      }
      if (d == 0) total_s[0] = wp;
    }
    __syncthreads();
    float dot = total_s[0] * scale;

    float m_new = fmaxf(m, dot);
    float corr = expf(m - m_new);
    float p = expf(dot - m_new);
    l = l * corr + p;
    acc = acc * corr + p * q35_bf16(v, ((int64_t)s * kvheads + kh) * hd + d);
    m = m_new;
  }

  acc = acc / l;
  float sig = 1.0f / (1.0f + expf(-gv));
  out[((int64_t)t * heads + h) * hd + d] = q35_b16(acc * sig);
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

// ── INT4 W4A16 native GEMM (dequant fused into the GEMM) ────────────────
// C[M,N] = A[M,K] @ Wt[N,K]^T, where Wt is packed int4 with per-group scale
// and packed zero points (AWQ). All product dims here are multiples of 64,
// 32 (group) and 8 (nibble); guards are kept for safety.
// Tile: BM=32, BN=64, BK=64. A and the dequantized W tile live in shared.

#define QW4_BM 32
#define QW4_BN 64
#define QW4_BK 64
#define QW4_MT 4
#define QW4_NT 2

__global__ void qwen_gemm_w4a16_bf16_kernel(
    const __nv_bfloat16* a,     // [M, K]
    const int32_t* packed,      // [N, K/8]
    const __nv_bfloat16* scale, // [N, K/32]
    const int32_t* zp,          // [N/8, K/32]
    __nv_bfloat16* c,           // [M, N]
    int M, int N, int K) {
  const int groups = K / 32;
  const int packed_cols = K / 8;
  __shared__ __nv_bfloat162 A_s[QW4_BM][QW4_BK / 2];
  __shared__ __nv_bfloat162 W_s[QW4_BN][QW4_BK / 2];

  const int n0 = blockIdx.x * QW4_BN;
  const int m0 = blockIdx.y * QW4_BM;
  const int tid = threadIdx.x;
  const int tx = tid & 31;            // 32 lanes across BN
  const int ty = tid >> 5;            // 8 lanes across BM
  const int ml0 = ty * QW4_MT;        // 4 consecutive rows
  const int nl0 = tx * QW4_NT;        // 2 consecutive cols

  float acc[QW4_MT][QW4_NT];
  #pragma unroll
  for (int r = 0; r < QW4_MT; r++)
    #pragma unroll
    for (int cv = 0; cv < QW4_NT; cv++) acc[r][cv] = 0.0f;

  for (int k0 = 0; k0 < K; k0 += QW4_BK) {
    // Cooperative bf16x2 A tile load: A_s[32][32] = 1024 vec2, 4 / thread
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      int e = tid + i * 256;
      int mm = e >> 5;
      int kk2 = e & 31;
      int mg = m0 + mm;
      int kg = k0 + 2 * kk2;
      if (mg < M && kg + 1 < K) {
        A_s[mm][kk2] = *reinterpret_cast<const __nv_bfloat162*>(&a[(int64_t)mg * K + kg]);
      } else {
        A_s[mm][kk2] = __floats2bfloat162_rn(0.0f, 0.0f);
      }
    }
    // Cooperative dequant W tile: W_s[64][32] = 2048 vec2, 8 / thread
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      int e = tid + i * 256;
      int nn = e >> 5;
      int kk2 = e & 31;
      int ng = n0 + nn;
      int kg = k0 + 2 * kk2;
      if (ng < N && kg + 1 < K) {
        int g = kg / 32;
        float sc = __bfloat162float(scale[(int64_t)ng * groups + g]);
        int zpv = ((zp[((int64_t)ng / 8) * groups + g] >> (4 * (ng & 7))) & 0xF) - 8;
        int32_t pw0 = packed[(int64_t)ng * packed_cols + kg / 8];
        int32_t pw1 = packed[(int64_t)ng * packed_cols + (kg + 1) / 8];
        int w0 = ((pw0 >> (4 * (kg & 7))) & 0xF) - 8;
        int w1 = ((pw1 >> (4 * ((kg + 1) & 7))) & 0xF) - 8;
        W_s[nn][kk2] = __floats2bfloat162_rn(
            sc * (float)(w0 - zpv), sc * (float)(w1 - zpv));
      } else {
        W_s[nn][kk2] = __floats2bfloat162_rn(0.0f, 0.0f);
      }
    }
    __syncthreads();

    // 4x2 register micro-tile over the K chunk (bf16x2 -> 2 FMAs each)
    #pragma unroll
    for (int kk2 = 0; kk2 < QW4_BK / 2; kk2++) {
      float2 av[QW4_MT], wv[QW4_NT];
      #pragma unroll
      for (int r = 0; r < QW4_MT; r++) {
        __nv_bfloat162 v = A_s[ml0 + r][kk2];
        av[r].x = __bfloat162float(__low2bfloat16(v));
        av[r].y = __bfloat162float(__high2bfloat16(v));
      }
      #pragma unroll
      for (int cv = 0; cv < QW4_NT; cv++) {
        __nv_bfloat162 v = W_s[nl0 + cv][kk2];
        wv[cv].x = __bfloat162float(__low2bfloat16(v));
        wv[cv].y = __bfloat162float(__high2bfloat16(v));
      }
      #pragma unroll
      for (int r = 0; r < QW4_MT; r++)
        #pragma unroll
        for (int cv = 0; cv < QW4_NT; cv++)
          acc[r][cv] += av[r].x * wv[cv].x + av[r].y * wv[cv].y;
    }
    __syncthreads();
  }

  #pragma unroll
  for (int r = 0; r < QW4_MT; r++) {
    int mg = m0 + ml0 + r;
    #pragma unroll
    for (int cv = 0; cv < QW4_NT; cv++) {
      int ng = n0 + nl0 + cv;
      if (mg < M && ng < N) {
        c[(int64_t)mg * N + ng] = __float2bfloat16(acc[r][cv]);
      }
    }
  }
}

// ── beta/g gate computation (replaces the host round-trip) ──────────────
__global__ void qwen_beta_g_bf16_kernel(
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    const float* a_log, const float* dt_bias,
    __nv_bfloat16* beta, __nv_bfloat16* g,
    int total, int nv) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= total) return;
  int h = e % nv;
  float act = q35_bf16(a, e) + dt_bias[h];
  float sp = act > 20.0f ? act : __logf(1.0f + __expf(act));
  g[e] = q35_b16(-__expf(a_log[h]) * sp);
  beta[e] = q35_b16(1.0f / (1.0f + __expf(-q35_bf16(b, e))));
}
