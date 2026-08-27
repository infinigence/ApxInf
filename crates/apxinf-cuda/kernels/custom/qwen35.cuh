#pragma once
// Qwen3.5-specific bf16 device kernels (C1 eager path). Launchers live in
// adapters/custom_kernels.cu; safe Rust wrappers in kernels/qwen35.rs.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define DR_KDCHUNK 4
#include <cstdint>

__device__ __forceinline__ float q35_bf16(const __half* p, size_t i) {
  return __half2float(p[i]);
}

__device__ __forceinline__ __half q35_b16(float v) {
  return __float2half(v);
}

// ── AWQ W4A16 → bf16, transposed layout [in, out] for cuBLAS row-major ──
__global__ void qwen_dequant_w4a16_bf16_kernel(
    const int32_t* packed,          // [out, in/8]
    const __half* scale,     // [out, groups]
    const int32_t* zp,              // [ceil(out/8), groups]
    __half* out,             // [in, out] (transposed)
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
    float s = __half2float(scale[o * groups + g]);
    out[(int64_t)i * out_dim + o] = __float2half(s * ((float)w4 - (float)zpv));
  }
}

// ── elementwise ─────────────────────────────────────────────────────────
__global__ void qwen_mul_bf16_kernel(
    const __half* a, const __half* b,
    __half* out, int64_t n) {
  for (int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       e < n;
       e += (int64_t)gridDim.x * blockDim.x) {
    out[e] = q35_b16(q35_bf16(a, e) * q35_bf16(b, e));
  }
}

__global__ void qwen_silu_bf16_kernel(
    const __half* a, __half* out, int64_t n) {
  for (int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       e < n;
       e += (int64_t)gridDim.x * blockDim.x) {
    float x = q35_bf16(a, e);
    out[e] = q35_b16(x / (1.0f + expf(-x)));
  }
}

__global__ void qwen_sigmoid_mul_bf16_kernel(
    const __half* a, const __half* b,
    __half* out, int64_t n) {
  for (int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       e < n;
       e += (int64_t)gridDim.x * blockDim.x) {
    float g = q35_bf16(a, e);
    float sig = 1.0f / (1.0f + expf(-g));
    out[e] = q35_b16(sig * q35_bf16(b, e));
  }
}

// dst += src
__global__ void qwen_accum_bf16_kernel(
    __half* dst, const __half* src, int64_t n) {
  for (int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       e < n;
       e += (int64_t)gridDim.x * blockDim.x) {
    dst[e] = q35_b16(q35_bf16(dst, e) + q35_bf16(src, e));
  }
}

// ── RMSNorm with precomputed (1+w) weights ──────────────────────────────
__global__ void qwen_rms_norm_bf16_kernel(
    const __half* x, const __half* w,
    __half* out, int rows, int cols, float eps) {
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
    const __half* qg, __half* q, __half* gate,
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
    __half* x, const __half* cos, const __half* sin,
    int64_t pairs, int heads, int hd, int half, const int* pos0_ptr) {
  // one thread per (row, i) pair; row = e / half, i = e % half
  int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= pairs) return;
  int row = (int)(e / half);
  int i = (int)(e % half);
  int t = *pos0_ptr + row / heads;     // absolute token index
  float c = q35_bf16(cos, (int64_t)t * half + i);
  float s = q35_bf16(sin, (int64_t)t * half + i);
  float a = q35_bf16(x, (int64_t)row * hd + i);
  float b = q35_bf16(x, (int64_t)row * hd + half + i);
  x[(int64_t)row * hd + i] = q35_b16(a * c - b * s);
  x[(int64_t)row * hd + half + i] = q35_b16(a * s + b * c);
}

// ── depthwise causal conv kernel=4 + SiLU ───────────────────────────────
__global__ void qwen_conv_silu_bf16_kernel(
    const __half* x, const __half* w,
    __half* out, int L, int conv_dim) {
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
        __half2 xv = *reinterpret_cast<const __half2*>(
            &x[(int64_t)(t - j) * conv_dim + c0]);
        float x0 = __half2float(__low2half(xv));
        float x1 = __half2float(__high2half(xv));
        float w0 = __half2float(w[(int64_t)c0 * 4 + (3 - j)]);
        float w1 = __half2float(w[(int64_t)(c0 + 1) * 4 + (3 - j)]);
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
    const __half* q, const __half* k, const __half* v,
    const __half* beta, const __half* g,
    float* state,                    // [nv, kd, vd]
    __half* out,              // [L, nv, vd]
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
    const __half* krow = k + ((int64_t)t * nv + h) * kd;
    const __half* qrow = q + ((int64_t)t * nv + h) * kd;

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
    const __half* q,     // [L, heads, hd]
    const __half* k,     // [L, kvheads, hd]
    const __half* v,     // [L, kvheads, hd]
    const __half* gate,  // [L, heads, hd]
    __half* out,         // [L, heads, hd]
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
    float p = __expf(dot - m_new);
    if (m_new > m) {
      float corr = __expf(m - m_new);
      l = fmaf(l, corr, p);
      acc = fmaf(acc, corr, p * q35_bf16(v, ((int64_t)s * kvheads + kh) * hd + d));
      m = m_new;
    } else {
      l = fmaf(l, 1.0f, p);
      acc = fmaf(acc, 1.0f, p * q35_bf16(v, ((int64_t)s * kvheads + kh) * hd + d));
    }
  }

  acc = acc / l;
  float sig = 1.0f / (1.0f + expf(-gv));
  out[((int64_t)t * heads + h) * hd + d] = q35_b16(acc * sig);
}

// ── expand conv output into per-head q/k (GQA 3x) and v ─────────────────
__global__ void qwen_conv_split_bf16_kernel(
    const __half* conv, __half* q, __half* k,
    __half* v, int L, int nk, int nv, int kd, int vd, int conv_dim) {
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
    const __half* x, __half* out,
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
    const __half* a,     // [M, K]
    const int32_t* packed,      // [N, K/8]
    const __half* scale, // [N, K/32]
    const int32_t* zp,          // [N/8, K/32]
    __half* c,           // [M, N]
    int M, int N, int K) {
  const int groups = K / 32;
  const int packed_cols = K / 8;
  __shared__ __half2 A_s[QW4_BM][QW4_BK / 2];
  __shared__ __half2 W_s[QW4_BN][QW4_BK / 2];

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
        A_s[mm][kk2] = *reinterpret_cast<const __half2*>(&a[(int64_t)mg * K + kg]);
      } else {
        A_s[mm][kk2] = __floats2half2_rn(0.0f, 0.0f);
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
        float sc = __half2float(scale[(int64_t)ng * groups + g]);
        int zpv = ((zp[((int64_t)ng / 8) * groups + g] >> (4 * (ng & 7))) & 0xF) - 8;
        int32_t pw0 = packed[(int64_t)ng * packed_cols + kg / 8];
        int32_t pw1 = packed[(int64_t)ng * packed_cols + (kg + 1) / 8];
        int w0 = ((pw0 >> (4 * (kg & 7))) & 0xF) - 8;
        int w1 = ((pw1 >> (4 * ((kg + 1) & 7))) & 0xF) - 8;
        W_s[nn][kk2] = __floats2half2_rn(
            sc * (float)(w0 - zpv), sc * (float)(w1 - zpv));
      } else {
        W_s[nn][kk2] = __floats2half2_rn(0.0f, 0.0f);
      }
    }
    __syncthreads();

    // 4x2 register micro-tile over the K chunk (bf16x2 -> 2 FMAs each)
    #pragma unroll
    for (int kk2 = 0; kk2 < QW4_BK / 2; kk2++) {
      float2 av[QW4_MT], wv[QW4_NT];
      #pragma unroll
      for (int r = 0; r < QW4_MT; r++) {
        __half2 v = A_s[ml0 + r][kk2];
        av[r].x = __half2float(__low2half(v));
        av[r].y = __half2float(__high2half(v));
      }
      #pragma unroll
      for (int cv = 0; cv < QW4_NT; cv++) {
        __half2 v = W_s[nl0 + cv][kk2];
        wv[cv].x = __half2float(__low2half(v));
        wv[cv].y = __half2float(__high2half(v));
      }
      #pragma unroll
      for (int r = 0; r < QW4_MT; r++)
        #pragma unroll
        for (int cv = 0; cv < QW4_NT; cv++) {
          acc[r][cv] = fmaf(av[r].x, wv[cv].x, acc[r][cv]);
          acc[r][cv] = fmaf(av[r].y, wv[cv].y, acc[r][cv]);
        }
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
        c[(int64_t)mg * N + ng] = __float2half(acc[r][cv]);
      }
    }
  }
}

#define FW4_BM 32
#define FW4_BN 64
#define FW4_BK 64
#define FW4_MT 4
#define FW4_NT 2

// Dense fp16 GEMM: C[M,N] = A[M,K] @ B[K,N]^T, device `b` stored transposed
// as [K,N] (same layout as upload_transposed). Deterministic per-output
// accumulation order (K chunks of 64, pairwise fp16 FMAs) identical to the
// w4a16 kernel, so m=1 (decode) and m=l (prefill) produce bit-identical rows.
__global__ void qwen_gemm_f16_kernel(
    const __half* a, const __half* b, __half* c, int M, int N, int K) {
  __shared__ __half2 A_s[FW4_BM][FW4_BK / 2];
  __shared__ __half2 B_s[FW4_BN][FW4_BK / 2];

  const int n0 = blockIdx.x * FW4_BN;
  const int m0 = blockIdx.y * FW4_BM;
  const int tid = threadIdx.x;
  const int tx = tid & 31;
  const int ty = tid >> 5;
  const int ml0 = ty * FW4_MT;
  const int nl0 = tx * FW4_NT;

  float acc[FW4_MT][FW4_NT];
  #pragma unroll
  for (int r = 0; r < FW4_MT; r++)
    #pragma unroll
    for (int cv = 0; cv < FW4_NT; cv++) acc[r][cv] = 0.0f;

  for (int k0 = 0; k0 < K; k0 += FW4_BK) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
      int e = tid + i * 256;
      int mm = e >> 5;
      int kk2 = e & 31;
      int mg = m0 + mm;
      int kg = k0 + 2 * kk2;
      if (mg < M && kg + 1 < K) {
        A_s[mm][kk2] = *reinterpret_cast<const __half2*>(&a[(int64_t)mg * K + kg]);
      } else {
        A_s[mm][kk2] = __floats2half2_rn(0.0f, 0.0f);
      }
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      int e = tid + i * 256;
      int nn = e >> 5;
      int kk2 = e & 31;
      int ng = n0 + nn;
      int kg = k0 + 2 * kk2;
      float f1 = 0.0f, f2 = 0.0f;
      if (ng < N && kg < K) f1 = __half2float(b[(int64_t)kg * N + ng]);
      if (ng < N && kg + 1 < K) f2 = __half2float(b[(int64_t)(kg + 1) * N + ng]);
      B_s[nn][kk2] = __floats2half2_rn(f1, f2);
    }
    __syncthreads();

    #pragma unroll
    for (int kk2 = 0; kk2 < FW4_BK / 2; kk2++) {
      float2 av[FW4_MT], bv[FW4_NT];
      #pragma unroll
      for (int r = 0; r < FW4_MT; r++) {
        __half2 v = A_s[ml0 + r][kk2];
        av[r].x = __half2float(__low2half(v));
        av[r].y = __half2float(__high2half(v));
      }
      #pragma unroll
      for (int cv = 0; cv < FW4_NT; cv++) {
        __half2 v = B_s[nl0 + cv][kk2];
        bv[cv].x = __half2float(__low2half(v));
        bv[cv].y = __half2float(__high2half(v));
      }
      #pragma unroll
      for (int r = 0; r < FW4_MT; r++)
        #pragma unroll
        for (int cv = 0; cv < FW4_NT; cv++)
          acc[r][cv] += av[r].x * bv[cv].x + av[r].y * bv[cv].y;
    }
    __syncthreads();
  }

  #pragma unroll
  for (int r = 0; r < FW4_MT; r++) {
    int mg = m0 + ml0 + r;
    #pragma unroll
    for (int cv = 0; cv < FW4_NT; cv++) {
      int ng = n0 + nl0 + cv;
      if (mg < M && ng < N) {
        c[(int64_t)mg * N + ng] = __float2half(acc[r][cv]);
      }
    }
  }
}

// ── beta/g gate computation (replaces the host round-trip) ──────────────
__global__ void qwen_beta_g_bf16_kernel(
    const __half* a, const __half* b,
    const float* a_log, const float* dt_bias,
    __half* beta, __half* g,
    int total, int nv) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= total) return;
  int h = e % nv;
  float act = q35_bf16(a, e) + dt_bias[h];
  float sp = act > 20.0f ? act : __logf(1.0f + __expf(act));
  g[e] = q35_b16(-__expf(a_log[h]) * sp);
  beta[e] = q35_b16(1.0f / (1.0f + __expf(-q35_bf16(b, e))));
}

// ── device-to-device bf16 copy (for cache population) ───────────────────
__global__ void qwen_copy_bf16_kernel(
    const __half* src, __half* dst, int64_t n) {
  int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n) return;
  dst[e] = src[e];
}

// ── single-token attention over a KV cache (online softmax) ─────────────
__global__ void qwen_attention_decode_bf16_kernel(
    const __half* q,      // [heads, hd]
    const __half* kcache, // [seq, kvheads, hd]
    const __half* vcache, // [seq, kvheads, hd]
    const __half* gate,   // [heads, hd]
    __half* out,          // [heads, hd]
    const int* seq_ptr, int heads, int kvheads, int hd) {
  int h = blockIdx.x;
  int d = threadIdx.x;
  if (h >= heads || d >= hd) return;
  int kh = h / (heads / kvheads);
  int seq = *seq_ptr + 1;

  __shared__ float warp_part[8];
  __shared__ float total_s[1];

  float qv = q35_bf16(q, (int64_t)h * hd + d);
  float gv = q35_bf16(gate, (int64_t)h * hd + d);
  float acc = 0.0f;
  float m = -1e30f;
  float l = 0.0f;
  float scale = rsqrtf((float)hd);

  for (int s = 0; s < seq; s++) {
    float part = qv * q35_bf16(kcache, ((int64_t)s * kvheads + kh) * hd + d);
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
    float p = __expf(dot - m_new);
    if (m_new > m) {
      float corr = __expf(m - m_new);
      l = fmaf(l, corr, p);
      acc = fmaf(acc, corr, p * q35_bf16(vcache, ((int64_t)s * kvheads + kh) * hd + d));
      m = m_new;
    } else {
      l = fmaf(l, 1.0f, p);
      acc = fmaf(acc, 1.0f, p * q35_bf16(vcache, ((int64_t)s * kvheads + kh) * hd + d));
    }
  }

  acc = acc / l;
  float sig = 1.0f / (1.0f + expf(-gv));
  out[(int64_t)h * hd + d] = q35_b16(acc * sig);
}

// ── single-token causal conv + history shift ────────────────────────────
// hist layout: [3, conv_dim], hist[0] = newest tap (t-1), hist[2] = t-3.
__global__ void qwen_conv_step_silu_bf16_kernel(
    const __half* cur, __half* hist, const __half* w,
    __half* out, int conv_dim) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= conv_dim) return;
  float acc = q35_bf16(w, (int64_t)c * 4 + 3) * q35_bf16(cur, c);
  #pragma unroll
  for (int j = 1; j < 4; j++) {
    acc += q35_bf16(w, (int64_t)c * 4 + (3 - j)) * q35_bf16(hist, (int64_t)(j - 1) * conv_dim + c);
  }
  out[c] = q35_b16(acc / (1.0f + expf(-acc)));
  // shift history (each thread owns channel c; source reads precede writes)
  hist[(int64_t)2 * conv_dim + c] = hist[(int64_t)1 * conv_dim + c];
  hist[(int64_t)1 * conv_dim + c] = hist[c];
  hist[c] = cur[c];
}

// ── single-token GatedDeltaNet step over cached state (kd split) ────────
__global__ void qwen_delta_step_bf16_kernel(
    const __half* q, const __half* k, const __half* v,
    const __half* beta, const __half* g,
    float* state, __half* out, int nv, int kd, int vd) {
  int h = blockIdx.x;
  int j = threadIdx.x;
  int y = threadIdx.y;
  if (h >= nv || j >= vd) return;
  const int chunk = kd / DR_KDCHUNK;
  const int kk0 = y * chunk;
  const int kk1 = kk0 + chunk;
  float* S = state + (int64_t)h * kd * vd;

  __shared__ float kv_part[DR_KDCHUNK][128];
  __shared__ float o_part[DR_KDCHUNK][128];
  __shared__ float delta_s[128];

  const __half* krow = k + (int64_t)h * kd;
  const __half* qrow = q + (int64_t)h * kd;

  float decay = expf(q35_bf16(g, h));
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
    float delta = (q35_bf16(v, (int64_t)h * vd + j) - kv) *
                  q35_bf16(beta, h);
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
    out[(int64_t)h * vd + j] = q35_b16(o);
  }
}


// ── copy bf16 runs into a cache slot addressed by a device position ──────
__global__ void qwen_copy_at_bf16_kernel(
    const __half* src, __half* dst_base,
    const int* pos_ptr, int stride_elems, int n) {
  int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n) return;
  int64_t off = (int64_t)(*pos_ptr) * stride_elems;
  dst_base[off + e] = src[e];
}

// ── M=1 Q4 GEMM (decode specialist): one output row, one thread per column
// A[K] is cooperatively staged in dynamic shared once; each thread walks the
// packed weight row reusing the per-32-group scale/zp. No per-token W tile.
__global__ void qwen_gemm_w4a16_m1_bf16_kernel(
    const __half* a,     // [K]
    const int32_t* packed,      // [N, K/8]
    const __half* scale, // [N, K/32]
    const int32_t* zp,          // [N/8, K/32]
    __half* c,           // [N]
    int N, int K) {
  // M=1 decode GEMM: packed tile staged coalesced in shared; scale/zp read
  // per group like the batched kernel. One output column per thread.
  const int BN = 128;
  const int BK = 128;
  __shared__ int32_t ps[BN][BK / 8 + 1];
  __shared__ __half A_s[BK];

  int n0 = blockIdx.x * BN;
  int tid = threadIdx.x;
  const int groups = K / 32;
  const int pcols = K / 8;
  float acc = 0.0f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    if (tid < BK) A_s[tid] = a[k0 + tid];
    #pragma unroll
    for (int it = 0; it < BK / 8; it++) {
      int e = it * BN + tid;
      int nn = e >> 4;
      int j = e & 15;
      int ng = n0 + nn;
      ps[nn][j] = (ng < N) ? packed[(int64_t)ng * pcols + (k0 >> 3) + j] : 0;
    }
    __syncthreads();

    int n = n0 + tid;
    if (n >= N) {
      __syncthreads();
      continue;
    }
    #pragma unroll
    for (int kk2 = 0; kk2 < BK / 2; kk2++) {
      int kk = 2 * kk2;
      int g = (k0 >> 5) + (kk >> 5);
      float s = __half2float(scale[(int64_t)n * groups + g]);
      int zpv = ((zp[((int64_t)n / 8) * groups + g] >> (4 * (n & 7))) & 0xF) - 8;
      int w0 = ((ps[tid][kk >> 3] >> (4 * (kk & 7))) & 0xF) - 8;
      int w1 = ((ps[tid][(kk + 1) >> 3] >> (4 * ((kk + 1) & 7))) & 0xF) - 8;
      float wv0 = __half2float(__float2half_rn(s * (float)(w0 - zpv)));
      float wv1 = __half2float(__float2half_rn(s * (float)(w1 - zpv)));
      acc = fmaf(__half2float(A_s[kk]), wv0, acc);
      acc = fmaf(__half2float(A_s[kk + 1]), wv1, acc);
    }
    __syncthreads();
  }

  int n = n0 + tid;
  if (n < N) c[n] = __float2half(acc);
}

// ── v2 dequant: 32x32 tiled transpose, coalesced writes ─────────────────
// Block (32,8): x = i offset, y*4+r = o offset. Values staged transposed in
// shared memory so both the packed read and the [in,out] write are coalesced.
__global__ void qwen_dequant_w4a16_bf16_kernel_v2(
    const int32_t* packed, const __half* scale, const int32_t* zp,
    __half* out, int out_dim, int in_dim) {
  __shared__ float tile[32][33]; // [i_off][o_off], padded (no bank conflicts)
  int o0 = blockIdx.x * 32;
  int i0 = blockIdx.y * 32;
  int x = threadIdx.x;
  int y = threadIdx.y;
  int groups = in_dim / 32;
  int i = i0 + x;
  bool iok = i < in_dim;
  #pragma unroll
  for (int r = 0; r < 4; r++) {
    int o = o0 + y * 4 + r;
    float val = 0.0f;
    if (o < out_dim && iok) {
      int g = i >> 5;
      int j8 = i & 7;
      int zpv = ((zp[(o >> 3) * groups + g] >> (4 * (o & 7))) & 0xF) - 8;
      int w4 = ((packed[(int64_t)o * (in_dim >> 3) + (i >> 3)] >> (4 * j8)) & 0xF) - 8;
      float sv = __half2float(scale[(int64_t)o * groups + g]);
      val = sv * ((float)w4 - (float)zpv);
    }
    tile[x][y * 4 + r] = val;
  }
  __syncthreads();
  #pragma unroll
  for (int r = 0; r < 4; r++) {
    int iw = i0 + y * 4 + r;
    int ow = o0 + x;
    if (iw < in_dim && ow < out_dim) {
      out[(int64_t)iw * out_dim + ow] = __float2half(tile[y * 4 + r][x]);
    }
  }
}

// ── GatedDeltaNet recurrence v2: warp-per-(head, column), register state ─
// Requires kd % 32 == 0 && kd <= 512. grid = (nv, ceil(vd/8)), block = 256.
// Each warp owns one state column j of head h; its kd slice (4 floats per
// lane at kd=128) lives in registers for the whole sequence, and the two
// kd reductions use warp shuffles (no __syncthreads in the time loop).
__global__ void qwen_delta_recurrence_bf16_kernel_v2(
    const __half* q, const __half* k, const __half* v,
    const __half* beta, const __half* g,
    float* state, __half* out,
    int L, int nv, int kd, int vd) {
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int h = blockIdx.x;
  int j = blockIdx.y * 8 + warp;
  if (h >= nv || j >= vd) return;
  int nchunk = kd >> 5;
  float* S = state + (int64_t)h * kd * vd;
  float s[16];
  float kreg[16];
  for (int c = 0; c < nchunk; c++) s[c] = 0.0f;

  for (int t = 0; t < L; t++) {
    int64_t th = (int64_t)t * nv + h;
    float decay = expf(__half2float(g[th]));
    float bt = __half2float(beta[th]);
    float vt = __half2float(v[th * vd + j]);
    float pkv = 0.0f;
    for (int c = 0; c < nchunk; c++) {
      int kk = lane + (c << 5);
      float kv = __half2float(k[th * kd + kk]);
      kreg[c] = kv;
      s[c] *= decay;
      pkv += s[c] * kv;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) pkv += __shfl_xor_sync(0xffffffffu, pkv, off);
    float delta = (vt - pkv) * bt;
    float po = 0.0f;
    for (int c = 0; c < nchunk; c++) {
      int kk = lane + (c << 5);
      s[c] += kreg[c] * delta;
      po += s[c] * __half2float(q[th * kd + kk]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) po += __shfl_xor_sync(0xffffffffu, po, off);
    if (lane == 0) out[th * vd + j] = __float2half(po);
  }
  for (int c = 0; c < nchunk; c++) {
    int kk = lane + (c << 5);
    S[(int64_t)kk * vd + j] = s[c];
  }
}

// v3: fully-register-resident recurrence. NCHUNK = kd/32 must be a compile-time
// constant so that the per-warp state s[] stays in registers (v2 indexed s[]
// with a runtime bound and spilled to local memory).
template <int NCHUNK>
__global__ void qwen_delta_recurrence_bf16_kernel_v3(
    const __half* __restrict__ q, const __half* __restrict__ k,
    const __half* __restrict__ v, const __half* __restrict__ beta,
    const __half* __restrict__ g,
    float* __restrict__ state, __half* __restrict__ out,
    int L, int nv, int vd) {
  constexpr int KD = 32 * NCHUNK;
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int h = blockIdx.x;
  int j = blockIdx.y * 8 + warp;
  if (h >= nv || j >= vd) return;
  float s[NCHUNK];
  #pragma unroll
  for (int c = 0; c < NCHUNK; c++) s[c] = 0.0f;

  const __half* kp = k + h * KD;
  const __half* qp = q + h * KD;
  const __half* vp = v + h * vd + j;
  const __half* gp = g + h;
  const __half* bp = beta + h;
  const long long sk = (long long)nv * KD;
  const long long sv = (long long)nv * vd;
  const long long nvvd = (long long)nv * vd;

  for (int t = 0; t < L; t++) {
    float decay = expf(__half2float(*gp));
    float bt = __half2float(*bp);
    float vt = __half2float(*vp);
    float kreg[NCHUNK];
    float pkv = 0.0f;
    #pragma unroll
    for (int c = 0; c < NCHUNK; c++) {
      float kv = __half2float(kp[c * 32 + lane]);
      kreg[c] = kv;
      s[c] = s[c] * decay;
      pkv += s[c] * kv;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) pkv += __shfl_xor_sync(0xffffffffu, pkv, off);
    float delta = (vt - pkv) * bt;
    float po = 0.0f;
    #pragma unroll
    for (int c = 0; c < NCHUNK; c++) {
      s[c] += kreg[c] * delta;
      po += s[c] * __half2float(qp[c * 32 + lane]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) po += __shfl_xor_sync(0xffffffffu, po, off);
    if (lane == 0) out[(long long)t * nvvd + h * vd + j] = __float2half(po);
    kp += sk; qp += sk; vp += sv; gp += nv; bp += nv;
  }
  float* S = state + (long long)h * KD * vd;
  #pragma unroll
  for (int c = 0; c < NCHUNK; c++) S[(long long)(c * 32 + lane) * vd + j] = s[c];
}

// ── embedding gather: bf16 table -> f16 rows (no scaling) ───────────────
__global__ void qwen_embed_gather_f16_kernel(
    const __nv_bfloat16* table, const uint32_t* ids, __half* out,
    int64_t count, int hidden) {
  for (int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       e < count; e += (int64_t)gridDim.x * blockDim.x) {
    int t = (int)(e / hidden);
    int c = (int)(e % hidden);
    uint32_t id = ids[t];
    out[e] = __float2half(__bfloat162float(table[(int64_t)id * hidden + c]));
  }
}
