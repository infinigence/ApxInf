// Tensor-core form of the GDN raw-attention term.
//
// Third and last of the GDN prefill kernels that are GEMM-shaped and run in
// scalar fp32 on the CUDA cores. It forms K.K^T and Q.K^T over one chunk:
// [C,K] @ [K,C] twice, 72.2 ms per scene.
//
// Unlike the chunk-state scan and the chunk GEMM, neither operand here is
// already on the BF16 grid -- q and k come out of gdn_qk_prep normalised -- so
// both are split, and the product runs as the three terms that matter:
//
//   (qh + ql)(kh + kl) = qh.kh + qh.kl + ql.kh + ql.kl
//
// dropping only ql.kl, which is about 2^-16 relative. Every kept partial
// product is BF16 times BF16 into fp32 and therefore exact.
//
// The two outputs are not equally forgiving. `t_out` is rounded to BF16 on the
// way out, so anything below 2^-8 is invisible there. `a_out` stays fp32 and
// feeds the triangular solve, which is why the third term is kept rather than
// settling for the two-term product.
#pragma once

template <bool SPLIT>
__global__ __launch_bounds__(256) void gdn_attn_raw_wmma_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ beta, const float* __restrict__ g_cum,
    float* __restrict__ a_out, float* __restrict__ t_out, int seq_pad) {
  using namespace nvcuda;
  constexpr int C = 64;    // chunk_size
  constexpr int K = 128;   // head_k_dim
  constexpr int T = 16;

  const int head = blockIdx.y;
  const int chunk = blockIdx.x;
  const int tid = static_cast<int>(threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x);
  const int warp = tid >> 5;

  const int64_t token_base =
      static_cast<int64_t>(head) * seq_pad + static_cast<int64_t>(chunk) * C;
  const int64_t matrix_base =
      (static_cast<int64_t>(head) * gridDim.x + chunk) * C * C;

  extern __shared__ char gdn_attn_wmma_smem[];
  __nv_bfloat16* s_q_hi = reinterpret_cast<__nv_bfloat16*>(gdn_attn_wmma_smem);  // [C][K]
  __nv_bfloat16* s_q_lo = s_q_hi + C * K;
  __nv_bfloat16* s_k_hi = s_q_lo + C * K;
  __nv_bfloat16* s_k_lo = s_k_hi + C * K;
  float* s_a = reinterpret_cast<float*>(s_k_lo + C * K);                        // [C][C]
  float* s_t = s_a + C * C;                                                     // [C][C]

  for (int idx = tid; idx < C * K; idx += nthreads) {
    const int r = idx / K, d = idx - r * K;
    const int64_t src = (token_base + r) * K + d;
    gdn_split_bf16(q[src], &s_q_hi[idx], &s_q_lo[idx]);
    gdn_split_bf16(k[src], &s_k_hi[idx], &s_k_lo[idx]);
  }
  __syncthreads();

  // Output is [C,C] = 4x4 tiles; eight warps, two tiles each.
  for (int tile = warp; tile < (C / T) * (C / T); tile += nthreads >> 5) {
    const int trow = tile >> 2, tcol = tile & 3;
    wmma::fragment<wmma::accumulator, T, T, T, float> acc_a, acc_t;
    wmma::fill_fragment(acc_a, 0.0f);
    wmma::fill_fragment(acc_t, 0.0f);
    wmma::fragment<wmma::matrix_a, T, T, T, __nv_bfloat16, wmma::row_major> fa;
    // k is [j][d] and the product wants [d][j], so it is read column-major.
    wmma::fragment<wmma::matrix_b, T, T, T, __nv_bfloat16, wmma::col_major> fb;
    for (int kk = 0; kk < K / T; ++kk) {
      // term 1: hi . hi
      wmma::load_matrix_sync(fb, s_k_hi + (tcol * T) * K + kk * T, K);
      wmma::load_matrix_sync(fa, s_k_hi + (trow * T) * K + kk * T, K);
      wmma::mma_sync(acc_a, fa, fb, acc_a);
      wmma::load_matrix_sync(fa, s_q_hi + (trow * T) * K + kk * T, K);
      wmma::mma_sync(acc_t, fa, fb, acc_t);
      if (SPLIT) {
        // term 2: lo . hi
        wmma::load_matrix_sync(fa, s_k_lo + (trow * T) * K + kk * T, K);
        wmma::mma_sync(acc_a, fa, fb, acc_a);
        wmma::load_matrix_sync(fa, s_q_lo + (trow * T) * K + kk * T, K);
        wmma::mma_sync(acc_t, fa, fb, acc_t);
        // term 3: hi . lo
        wmma::load_matrix_sync(fb, s_k_lo + (tcol * T) * K + kk * T, K);
        wmma::load_matrix_sync(fa, s_k_hi + (trow * T) * K + kk * T, K);
        wmma::mma_sync(acc_a, fa, fb, acc_a);
        wmma::load_matrix_sync(fa, s_q_hi + (trow * T) * K + kk * T, K);
        wmma::mma_sync(acc_t, fa, fb, acc_t);
      }
    }
    wmma::store_matrix_sync(s_a + (trow * T) * C + tcol * T, acc_a, C,
                            wmma::mem_row_major);
    wmma::store_matrix_sync(s_t + (trow * T) * C + tcol * T, acc_t, C,
                            wmma::mem_row_major);
  }
  __syncthreads();

  for (int cell = tid; cell < C * C; cell += nthreads) {
    const int i = cell / C, j = cell - i * C;
    const float decay =
        gdn_exp2_approx(g_cum[token_base + i] - g_cum[token_base + j]);
    a_out[matrix_base + cell] =
        (j < i) ? -__fmul_rn(__fmul_rn(s_a[cell], decay), beta[token_base + i])
                : 0.0f;
    t_out[matrix_base + cell] =
        (j <= i) ? __bfloat162float(__float2bfloat16(s_t[cell] * decay)) : 0.0f;
  }
}
