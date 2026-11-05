#pragma once

// Three independent 16x16 products. Each of three two-warp groups owns one
// product and four cells per thread; the fourth group stays out of this helper
// but participates in the caller's CTA barrier. No barrier is inside.
__device__ __forceinline__ void gdn_mul16_tf32_offdiag3(
    const float* a0, const float* b0, float* o0,
    const float* a1, const float* b1, float* o1,
    const float* a2, const float* b2, float* o2,
    int as, int bs, int os) {
  const int group = threadIdx.x / 64;
  if (group < 3) {
    const float* a = group == 0 ? a0 : (group == 1 ? a1 : a2);
    const float* b = group == 0 ? b0 : (group == 1 ? b1 : b2);
    float* out = group == 0 ? o0 : (group == 1 ? o1 : o2);
    const int lane = threadIdx.x % 64;
    const int row0 = lane / 16;
    const int col = lane % 16;
    float value[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k = 0; k < 16; ++k) {
      const float bv = gdn_tf32_truncate(b[k * bs + col]);
#pragma unroll
      for (int s = 0; s < 4; ++s)
        value[s] = fmaf(gdn_tf32_truncate(a[(row0 + 4 * s) * as + k]),
                        bv, value[s]);
    }
#pragma unroll
    for (int s = 0; s < 4; ++s)
      out[(row0 + 4 * s) * os + col] = value[s];
  }
}

// Fixed 64-token, 128-key raw attention plus blocked inverse. Shared storage
// is reused only after CTA barriers; raw products retain increasing-d FP32
// accumulation and the inverse retains gdn_block_inverse64_kernel's order.
// Strictly upper tiles produce zero without evaluating unused dot products.

template <typename QK>
__global__ __launch_bounds__(256, 4) void gdn_raw_inverse_f1_t(
    const QK* q, const QK* k, const float* beta, const float* g_cum,
    float* a_out, __nv_bfloat16* t_out, int seq_pad) {
  constexpr int C = 64;
  constexpr int D = 128;
  // Float keeps the established bank-padded layout. BF16 uses 130 elements
  // per row, so successive row starts shift by one 32-bit shared bank.
  constexpr int STRIDE = D + (sizeof(QK) == sizeof(float) ? 1 : 2);
  constexpr int TI = 4;
  constexpr int TJ = 4;
  const int head = blockIdx.y;
  const int chunk = blockIdx.x;
  const int64_t token_base = static_cast<int64_t>(head) * seq_pad + chunk * C;
  const int64_t matrix_base =
      (static_cast<int64_t>(head) * gridDim.x + chunk) * C * C;

  // Float Q/K use 66,048 bytes, BF16 Q/K 33,280 bytes. The later FP32
  // raw/inv/scratch region needs 35,840 bytes and reuses this allocation.
  extern __shared__ float smem[];
  QK* k_tile = reinterpret_cast<QK*>(smem);
  QK* q_tile = k_tile + C * STRIDE;
  for (int idx = threadIdx.x; idx < C * D; idx += blockDim.x) {
    const int t = idx / D;
    const int d = idx - t * D;
    const int64_t src = (token_base + t) * D + d;
    k_tile[t * STRIDE + d] = k[src];
    q_tile[t * STRIDE + d] = q[src];
  }
  __syncthreads();

  // Map the 136 lower/diagonal 4x4 tiles to the first 136 threads. Thus four
  // warps are fully active, the fifth is partially active, and the last
  // three skip the dot loop. The 120 strictly upper tiles only write +0.
  // Every live cell retains the original d=0..127 accumulation order.
  const int tile = threadIdx.x;
  const bool dot_tile = tile < 136;
  int tile_row = 0, tile_col = 0;
  if (dot_tile) {
    int remaining = tile;
    for (int row = 0; row < 16; ++row) {
      const int count = row + 1;
      if (remaining < count) {
        tile_row = row;
        tile_col = remaining;
        break;
      }
      remaining -= count;
    }
  } else {
    int remaining = tile - 136;
    for (int row = 0; row < 15; ++row) {
      const int count = 15 - row;
      if (remaining < count) {
        tile_row = row;
        tile_col = row + 1 + remaining;
        break;
      }
      remaining -= count;
    }
  }
  const int i0 = tile_row * TI;
  const int j0 = tile_col * TJ;
  float a1[TI][TJ];
  float a2[TI][TJ];
#pragma unroll
  for (int ii = 0; ii < TI; ++ii) {
#pragma unroll
    for (int jj = 0; jj < TJ; ++jj) {
      a1[ii][jj] = 0.0f;
      a2[ii][jj] = 0.0f;
    }
  }
  if (dot_tile) {
  for (int d = 0; d < D; ++d) {
    float kv[TI];
    float qv[TI];
    float kj[TJ];
#pragma unroll
    for (int ii = 0; ii < TI; ++ii) {
      kv[ii] = gdn_qk_widen(k_tile[(i0 + ii) * STRIDE + d]);
      qv[ii] = gdn_qk_widen(q_tile[(i0 + ii) * STRIDE + d]);
    }
#pragma unroll
    for (int jj = 0; jj < TJ; ++jj)
      kj[jj] = gdn_qk_widen(k_tile[(j0 + jj) * STRIDE + d]);
#pragma unroll
    for (int ii = 0; ii < TI; ++ii) {
#pragma unroll
      for (int jj = 0; jj < TJ; ++jj) {
        a1[ii][jj] += kv[ii] * kj[jj];
        a2[ii][jj] += qv[ii] * kj[jj];
      }
    }
  }
#pragma unroll
  for (int ii = 0; ii < TI; ++ii) {
    const int i = i0 + ii;
    const float beta_i = beta[token_base + i];
    const float g_i = g_cum[token_base + i];
#pragma unroll
    for (int jj = 0; jj < TJ; ++jj) {
      const int j = j0 + jj;
      const int cell = i * C + j;
      const float decay = gdn_exp2_approx(g_i - g_cum[token_base + j]);
      a1[ii][jj] = (j < i)
          ? -__fmul_rn(__fmul_rn(a1[ii][jj], decay), beta_i) : 0.0f;
      t_out[matrix_base + cell] =
          __float2bfloat16((j <= i) ? a2[ii][jj] * decay : 0.0f);
    }
  }
  } else {
#pragma unroll
    for (int ii = 0; ii < TI; ++ii) {
#pragma unroll
      for (int jj = 0; jj < TJ; ++jj) {
        const int cell = (i0 + ii) * C + (j0 + jj);
        t_out[matrix_base + cell] = __float2bfloat16(0.0f);
      }
    }
  }
  __syncthreads();  // Every Q/K read and T write is complete before alias.

  float* raw = smem;
  float* inv = raw + C * C;
  float* t0 = inv + C * C;
  float* t1 = t0 + 256;
  float* t2 = t1 + 256;
#pragma unroll
  for (int ii = 0; ii < TI; ++ii) {
#pragma unroll
    for (int jj = 0; jj < TJ; ++jj) {
      const int cell = (i0 + ii) * C + (j0 + jj);
      raw[cell] = a1[ii][jj];
    }
  }
  __syncthreads();
  for (int cell = threadIdx.x; cell < C * C; cell += blockDim.x)
    inv[cell] = raw[cell];
  __syncthreads();

  // Below mirrors gdn_block_inverse64_kernel exactly, after its global A1
  // load. The same thread ownership, order, TF32 truncation and BF16 write
  // are intentional. Pointers replace its static shared arrays only.
  const int group = threadIdx.x / 64;
  const int j = threadIdx.x % 64;
  const int start = group * 16;
  for (int row = 2; row < 16; ++row) {
    if (j < row) {
      float products[16];
      for (int kk = 0; kk < 16; ++kk)
        products[kk] = kk < row
            ? __fmul_rn(raw[(start + row) * 64 + start + kk],
                        inv[(start + kk) * 64 + start + j]) : 0.f;
      for (int offset = 8; offset > 0; offset >>= 1)
        for (int kk = 0; kk < offset; ++kk)
          products[kk] = __fadd_rn(products[kk], products[kk + offset]);
      inv[(start + row) * 64 + start + j] =
          __fadd_rn(raw[(start + row) * 64 + start + j], products[0]);
    }
    // BF16 Q/K: each 16x16 diagonal group is warp-local. All eight complete
    // warps execute the same synchronization branch. Keep the final CTA
    // boundary before other warps write identity; retain old float behavior.
    if (row == 15 || sizeof(QK) == sizeof(float)) __syncthreads();
    else __syncwarp();
  }
  for (int cell = threadIdx.x; cell < 64; cell += 256)
    inv[cell * 64 + cell] = 1.f;
  __syncthreads();
  if constexpr (sizeof(QK) == sizeof(__nv_bfloat16)) {
    gdn_mul16_tf32_offdiag3(
        inv + 16 * 64 + 16, raw + 16 * 64, t0,
        inv + 32 * 64 + 32, raw + 32 * 64 + 16, t1,
        inv + 48 * 64 + 48, raw + 48 * 64 + 32, t2,
        64, 64, 16);
    __syncthreads();
    gdn_mul16_tf32_offdiag3(
        t0, inv, inv + 16 * 64,
        t1, inv + 16 * 64 + 16, inv + 32 * 64 + 16,
        t2, inv + 32 * 64 + 32, inv + 48 * 64 + 32,
        16, 64, 64);
    __syncthreads();
  } else {
    gdn_mul16_tf32(inv + 16 * 64 + 16, 64, raw + 16 * 64, 64, t0, 16);
    gdn_mul16_tf32(t0, 16, inv, 64, inv + 16 * 64, 64);
    gdn_mul16_tf32(inv + 32 * 64 + 32, 64, raw + 32 * 64 + 16, 64, t0, 16);
    gdn_mul16_tf32(t0, 16, inv + 16 * 64 + 16, 64, inv + 32 * 64 + 16, 64);
    gdn_mul16_tf32(inv + 48 * 64 + 48, 64, raw + 48 * 64 + 32, 64, t0, 16);
    gdn_mul16_tf32(t0, 16, inv + 32 * 64 + 32, 64, inv + 48 * 64 + 32, 64);
  }
  gdn_mul16_tf32(raw + 32 * 64, 64, inv, 64, t0, 16);
  gdn_mul16_tf32(raw + 32 * 64 + 16, 64, inv + 16 * 64, 64, t1, 16);
  t0[threadIdx.x] += t1[threadIdx.x];
  __syncthreads();
  gdn_mul16_tf32(inv + 32 * 64 + 32, 64, t0, 16, inv + 32 * 64, 64);
  gdn_mul16_tf32(raw + 48 * 64 + 16, 64, inv + 16 * 64 + 16, 64, t0, 16);
  gdn_mul16_tf32(raw + 48 * 64 + 32, 64, inv + 32 * 64 + 16, 64, t1, 16);
  t0[threadIdx.x] += t1[threadIdx.x];
  __syncthreads();
  gdn_mul16_tf32(inv + 48 * 64 + 48, 64, t0, 16, inv + 48 * 64 + 16, 64);
  gdn_mul16_tf32(raw + 48 * 64, 64, inv, 64, t0, 16);
  gdn_mul16_tf32(raw + 48 * 64 + 16, 64, inv + 16 * 64, 64, t1, 16);
  gdn_mul16_tf32(raw + 48 * 64 + 32, 64, inv + 32 * 64, 64, t2, 16);
  t0[threadIdx.x] = t0[threadIdx.x] + t1[threadIdx.x] + t2[threadIdx.x];
  __syncthreads();
  gdn_mul16_tf32(inv + 48 * 64 + 48, 64, t0, 16, inv + 48 * 64, 64);
  for (int cell = threadIdx.x; cell < C * C; cell += blockDim.x)
    a_out[matrix_base + cell] =
        __bfloat162float(__float2bfloat16(inv[cell]));
}
