// Tensor-core form of the GDN chunk-state scan, with the fp32 operand carried
// as two BF16 terms.
//
// The scalar kernel beside this one is the largest single kernel in a VQA
// scene -- 452.3 ms, 38% of prefill -- spending it on four GEMM-shaped inner
// products in scalar fp32 on the CUDA cores, where this device delivers 5.43
// TFLOP/s against 164-195 on its tensor cores.
//
// In all four products the right-hand operand is already on the BF16 grid: the
// carried state is rounded to BF16 by the scalar kernel itself, and v_new is
// rounded before both the intra term and the state update. Rounding the left
// operand to BF16 as well was measured against an fp64 reference of the scan
// and costs 3.5x -- relative L1 1.418808e-3 to 4.954247e-3 -- which is more
// than the term already carries and more than this workload should pay.
//
// So the left operand is split instead: x = hi + lo with both BF16, and each
// product runs as two tensor-core passes against the same right fragment.
// Every partial product is then exact -- BF16 times BF16 into fp32 -- and the
// only new error is the residual x - (hi + lo), about 2^-16 relative, below
// the 2^-8 the right operand already contributes.
//
// APXINF_GDN_CHUNK_STATE_WMMA=1 selects the split form, =lossy the single-pass
// one, unset the scalar kernel. All three on one binary, all three measurable
// by tests::operators::gdn_chunk_state_scan_error_against_fp64_oracle.
#pragma once

// <mma.h> is pulled in by the adapter at global scope; this header is included
// inside the translation unit's anonymous namespace and must not open one.

__device__ __forceinline__ void gdn_split_bf16(float x, __nv_bfloat16* hi,
                                               __nv_bfloat16* lo) {
  const __nv_bfloat16 h = __float2bfloat16(x);
  *hi = h;
  *lo = __float2bfloat16(x - __bfloat162float(h));
}

// Three BF16 terms rather than two. Two carry the operand to about 2^-16
// relative, which is below what a BF16 right operand contributes and so is
// free in the chunk-state scan; it is not free where the product is otherwise
// nearly exact, and the third term takes the residual to roughly 2^-24, the
// resolution fp32 itself has.
__device__ __forceinline__ void gdn_split3_bf16(float x, __nv_bfloat16* hi,
                                                __nv_bfloat16* mid,
                                                __nv_bfloat16* lo) {
  const __nv_bfloat16 h = __float2bfloat16(x);
  const float r1 = x - __bfloat162float(h);
  const __nv_bfloat16 m = __float2bfloat16(r1);
  *hi = h;
  *mid = m;
  *lo = __float2bfloat16(r1 - __bfloat162float(m));
}

// One block per value head, 32 warps, shapes fixed; the launcher checks them.
// SPLIT selects one pass (left operand rounded to BF16) or two (hi + lo).
template <bool SPLIT>
__global__ __launch_bounds__(1024) void gdn_chunk_state_wmma_kernel(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ g_cum, const float* __restrict__ t_in,
    const float* __restrict__ vt_in, const float* __restrict__ kcd_in,
    float* state, __nv_bfloat16* out, int seq, int seq_pad, int total_chunks,
    int out_row_width, float scale) {
  using namespace nvcuda;
  constexpr int K = 128;   // head_k_dim
  constexpr int V = 128;   // head_v_dim
  constexpr int C = 64;    // chunk_size
  constexpr int T = 16;    // WMMA tile
  constexpr int PASSES = SPLIT ? 2 : 1;

  const int head = blockIdx.x;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int tid = static_cast<int>(threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x);
  const int trow = warp >> 3;
  const int tcol = warp & 7;

  extern __shared__ char gdn_wmma_smem[];
  // One 64 KB region holds the left operands, restaged between the two halves
  // of the chunk: kcd and q before the inter term, t and k afterwards. Their
  // lives do not overlap, and 224 KB of separate buffers would not fit.
  __nv_bfloat16* s_state = reinterpret_cast<__nv_bfloat16*>(gdn_wmma_smem);  // [K][V]
  __nv_bfloat16* s_lhs = s_state + K * V;                                    // 64 KB
  __nv_bfloat16* s_vr = s_lhs + 32 * 1024;                                   // [C][V]
  float* s_vnew = reinterpret_cast<float*>(s_vr + C * V);                    // [C][V]
  float* s_inter = s_vnew + C * V;                                           // [C][V], then the output staging
  // Phase 1 layout inside s_lhs.
  __nv_bfloat16* s_kcd_hi = s_lhs;                 // [C][K]
  __nv_bfloat16* s_kcd_lo = s_kcd_hi + C * K;
  __nv_bfloat16* s_q_hi = s_kcd_lo + C * K;
  __nv_bfloat16* s_q_lo = s_q_hi + C * K;
  // Phase 2 layout, over the same bytes.
  __nv_bfloat16* s_t_hi = s_lhs;                   // [C][C]
  __nv_bfloat16* s_t_lo = s_t_hi + C * C;
  __nv_bfloat16* s_kt_hi = s_t_lo + C * C;         // [C][K]
  __nv_bfloat16* s_kt_lo = s_kt_hi + C * K;

  const int64_t head_token_base = static_cast<int64_t>(head) * seq_pad;
  float* state_head = state + static_cast<int64_t>(head) * K * V;

  for (int c = 0; c < total_chunks; ++c) {
    const int64_t token_base = head_token_base + static_cast<int64_t>(c) * C;
    const int64_t vt_base = (static_cast<int64_t>(head) * total_chunks + c) * C * V;
    const int64_t kcd_base = (static_cast<int64_t>(head) * total_chunks + c) * C * K;
    const int64_t a_base = (static_cast<int64_t>(head) * total_chunks + c) * C * C;

    for (int i = tid; i < K * V; i += nthreads) {
      s_state[i] = __float2bfloat16(state_head[i]);
    }
    for (int i = tid; i < C * K; i += nthreads) {
      const int r = i / K, m = i - r * K;
      gdn_split_bf16(kcd_in[kcd_base + i], &s_kcd_hi[i], &s_kcd_lo[i]);
      gdn_split_bf16(q[(token_base + r) * K + m], &s_q_hi[i], &s_q_lo[i]);
    }
    __syncthreads();

    // vp and ai: [C,V] = [C,K] @ [K,V], 4 row tiles x 8 column tiles = 32,
    // one per warp.
    {
      wmma::fragment<wmma::accumulator, T, T, T, float> acc_vp, acc_ai;
      wmma::fill_fragment(acc_vp, 0.0f);
      wmma::fill_fragment(acc_ai, 0.0f);
      wmma::fragment<wmma::matrix_b, T, T, T, __nv_bfloat16, wmma::row_major> fb;
      wmma::fragment<wmma::matrix_a, T, T, T, __nv_bfloat16, wmma::row_major> fa;
      for (int kk = 0; kk < K / T; ++kk) {
        wmma::load_matrix_sync(fb, s_state + (kk * T) * V + tcol * T, V);
        #pragma unroll
        for (int pass = 0; pass < PASSES; ++pass) {
          const __nv_bfloat16* kcd_p = pass ? s_kcd_lo : s_kcd_hi;
          const __nv_bfloat16* q_p = pass ? s_q_lo : s_q_hi;
          wmma::load_matrix_sync(fa, kcd_p + (trow * T) * K + kk * T, K);
          wmma::mma_sync(acc_vp, fa, fb, acc_vp);
          wmma::load_matrix_sync(fa, q_p + (trow * T) * K + kk * T, K);
          wmma::mma_sync(acc_ai, fa, fb, acc_ai);
        }
      }
      wmma::store_matrix_sync(s_vnew + (trow * T) * V + tcol * T, acc_vp, V,
                              wmma::mem_row_major);
      wmma::store_matrix_sync(s_inter + (trow * T) * V + tcol * T, acc_ai, V,
                              wmma::mem_row_major);
    }
    __syncthreads();

    for (int cell = tid; cell < C * V; cell += nthreads) {
      const int i = cell / V;
      s_vnew[cell] = vt_in[vt_base + cell] - s_vnew[cell];
      s_inter[cell] *= gdn_exp2_approx(g_cum[token_base + i]);
      s_vr[cell] = __float2bfloat16(s_vnew[cell]);
    }
    __syncthreads();

    for (int i = tid; i < C * C; i += nthreads) {
      gdn_split_bf16(t_in[a_base + i], &s_t_hi[i], &s_t_lo[i]);
    }
    for (int i = tid; i < C * K; i += nthreads) {
      const int r = i / K, m = i - r * K;
      gdn_split_bf16(k[(token_base + r) * K + m], &s_kt_hi[i], &s_kt_lo[i]);
    }
    __syncthreads();

    // intra: [C,V] = [C,C] @ [C,V], accumulated on top of the inter term so
    // the two never meet in memory.
    {
      wmma::fragment<wmma::accumulator, T, T, T, float> acc;
      wmma::load_matrix_sync(acc, s_inter + (trow * T) * V + tcol * T, V,
                             wmma::mem_row_major);
      wmma::fragment<wmma::matrix_a, T, T, T, __nv_bfloat16, wmma::row_major> fa;
      wmma::fragment<wmma::matrix_b, T, T, T, __nv_bfloat16, wmma::row_major> fb;
      for (int kk = 0; kk < C / T; ++kk) {
        wmma::load_matrix_sync(fb, s_vr + (kk * T) * V + tcol * T, V);
        #pragma unroll
        for (int pass = 0; pass < PASSES; ++pass) {
          const __nv_bfloat16* t_p = pass ? s_t_lo : s_t_hi;
          wmma::load_matrix_sync(fa, t_p + (trow * T) * C + kk * T, C);
          wmma::mma_sync(acc, fa, fb, acc);
        }
      }
      #pragma unroll
      for (int e = 0; e < acc.num_elements; ++e) acc.x[e] *= scale;
      wmma::store_matrix_sync(s_inter + (trow * T) * V + tcol * T, acc, V,
                              wmma::mem_row_major);
    }
    __syncthreads();

    for (int cell = tid; cell < C * V; cell += nthreads) {
      const int i = cell / V, j = cell - i * V;
      const int token = c * C + i;
      if (token < seq) {
        out[static_cast<int64_t>(token) * out_row_width + head * V + j] =
            __float2bfloat16(s_inter[cell]);
      }
    }

    // State update. v_round is rebuilt with the chunk's trailing decay, then
    // state = state * decay + k^T @ v_round, in two halves of 64 rows because
    // one half of the fp32 result is exactly the space s_vnew occupies.
    const float g_last = g_cum[token_base + C - 1];
    const float decay = gdn_exp2_approx(g_last);
    __syncthreads();
    for (int cell = tid; cell < C * V; cell += nthreads) {
      const int i = cell / V;
      s_vr[cell] = __float2bfloat16(
          s_vnew[cell] * gdn_exp2_approx(g_last - g_cum[token_base + i]));
    }
    __syncthreads();
    for (int half = 0; half < 2; ++half) {
      const int mbase = half * (K / 2);
      {
        wmma::fragment<wmma::accumulator, T, T, T, float> acc;
        wmma::fill_fragment(acc, 0.0f);
        // s_kt is [i][m]; matrix_a read column-major over it is [m][i], the
        // transpose this product wants.
        wmma::fragment<wmma::matrix_a, T, T, T, __nv_bfloat16, wmma::col_major> fa;
        wmma::fragment<wmma::matrix_b, T, T, T, __nv_bfloat16, wmma::row_major> fb;
        for (int kk = 0; kk < C / T; ++kk) {
          wmma::load_matrix_sync(fb, s_vr + (kk * T) * V + tcol * T, V);
          #pragma unroll
          for (int pass = 0; pass < PASSES; ++pass) {
            const __nv_bfloat16* kt_p = pass ? s_kt_lo : s_kt_hi;
            wmma::load_matrix_sync(fa, kt_p + (kk * T) * K + mbase + trow * T, K);
            wmma::mma_sync(acc, fa, fb, acc);
          }
        }
        wmma::store_matrix_sync(s_vnew + (trow * T) * V + tcol * T, acc, V,
                                wmma::mem_row_major);
      }
      __syncthreads();
      for (int cell = tid; cell < (K / 2) * V; cell += nthreads) {
        const int m = mbase + cell / V, j = cell % V;
        float* slot = state_head + static_cast<int64_t>(m) * V + j;
        *slot = *slot * decay + s_vnew[cell];
      }
      __syncthreads();
    }
  }
}
