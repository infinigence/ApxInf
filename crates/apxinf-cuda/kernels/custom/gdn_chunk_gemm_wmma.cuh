// Tensor-core form of the GDN chunk GEMM.
//
// Same shape of opportunity as the chunk-state scan, and the same argument.
// Both products are [C,C] @ [C,V] with the right operand already on the BF16
// grid -- vb_tile is bf16(v * beta) and kb_tile is bf16(bf16(k * beta) *
// exp2(g)), both rounded by the scalar kernel before use -- and both outputs
// are rounded to BF16 on the way out. Only `a` is fp32, so only `a` is split
// into hi + lo and run as two passes; every partial product is then exact and
// the residual is about 2^-16 relative.
//
// The grid here is (chunks, heads), 1568 blocks at the shipped shape, so this
// one is not short of parallelism the way the scan was; 256 threads and eight
// warps, one output column block each.
#pragma once

template <bool SPLIT>
__global__ __launch_bounds__(256) void gdn_chunk_gemm_wmma_kernel(
    const float* __restrict__ a, const float* __restrict__ v,
    const float* __restrict__ k, const float* __restrict__ beta,
    const float* __restrict__ g_cum, float* __restrict__ vt_out,
    float* __restrict__ kcd_out, int seq_pad) {
  using namespace nvcuda;
  constexpr int C = 64;    // chunk_size
  constexpr int V = 128;   // head_v_dim == head_k_dim
  constexpr int T = 16;
  constexpr int PASSES = SPLIT ? 2 : 1;

  const int head = blockIdx.y;
  const int chunk = blockIdx.x;
  const int tid = static_cast<int>(threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x);
  const int warp = tid >> 5;

  const int64_t token_base =
      static_cast<int64_t>(head) * seq_pad + static_cast<int64_t>(chunk) * C;
  const int64_t a_base = (static_cast<int64_t>(head) * gridDim.x + chunk) * C * C;
  const int64_t vt_base = (static_cast<int64_t>(head) * gridDim.x + chunk) * C * V;
  const int64_t kcd_base = vt_base;

  extern __shared__ char gdn_gemm_wmma_smem[];
  __nv_bfloat16* s_vb = reinterpret_cast<__nv_bfloat16*>(gdn_gemm_wmma_smem);  // [C][V]
  __nv_bfloat16* s_kb = s_vb + C * V;                                          // [C][V]
  __nv_bfloat16* s_a_hi = s_kb + C * V;                                        // [C][C]
  __nv_bfloat16* s_a_lo = s_a_hi + C * C;                                      // [C][C]
  // The fragment element layout is implementation-defined, so the results are
  // staged here before the BF16 rounding and the strided write. [C][C] of
  // BF16 is a quarter of what [C][V] of float needs, so this is its own.
  float* s_stage = reinterpret_cast<float*>(s_a_lo + C * C);                   // [C][V]

  for (int idx = tid; idx < C * V; idx += nthreads) {
    const int m = idx / V, j = idx - m * V;
    const float b = beta[token_base + m];
    s_vb[idx] = __float2bfloat16(v[(token_base + m) * V + j] * b);
    const float kb0 = __bfloat162float(
        __float2bfloat16(k[(token_base + m) * V + j] * b));
    s_kb[idx] = __float2bfloat16(kb0 * gdn_exp2_approx(g_cum[token_base + m]));
  }
  for (int idx = tid; idx < C * C; idx += nthreads) {
    gdn_split_bf16(a[a_base + idx], &s_a_hi[idx], &s_a_lo[idx]);
  }
  __syncthreads();

  // Eight warps, one output column block each, all four row blocks.
  const int tcol = warp;
  wmma::fragment<wmma::accumulator, T, T, T, float> acc_vt[C / T], acc_kcd[C / T];
#pragma unroll
  for (int r = 0; r < C / T; ++r) {
    wmma::fill_fragment(acc_vt[r], 0.0f);
    wmma::fill_fragment(acc_kcd[r], 0.0f);
  }
  wmma::fragment<wmma::matrix_a, T, T, T, __nv_bfloat16, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, T, T, T, __nv_bfloat16, wmma::row_major> fvb, fkb;
  for (int kk = 0; kk < C / T; ++kk) {
    wmma::load_matrix_sync(fvb, s_vb + (kk * T) * V + tcol * T, V);
    wmma::load_matrix_sync(fkb, s_kb + (kk * T) * V + tcol * T, V);
#pragma unroll
    for (int pass = 0; pass < PASSES; ++pass) {
      const __nv_bfloat16* ap = pass ? s_a_lo : s_a_hi;
#pragma unroll
      for (int r = 0; r < C / T; ++r) {
        wmma::load_matrix_sync(fa, ap + (r * T) * C + kk * T, C);
        wmma::mma_sync(acc_vt[r], fa, fvb, acc_vt[r]);
        wmma::mma_sync(acc_kcd[r], fa, fkb, acc_kcd[r]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < C / T; ++r) {
    wmma::store_matrix_sync(s_stage + (r * T) * V + tcol * T, acc_vt[r], V,
                            wmma::mem_row_major);
  }
  __syncthreads();
  for (int idx = tid; idx < C * V; idx += nthreads) {
    vt_out[vt_base + idx] = __bfloat162float(__float2bfloat16(s_stage[idx]));
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < C / T; ++r) {
    wmma::store_matrix_sync(s_stage + (r * T) * V + tcol * T, acc_kcd[r], V,
                            wmma::mem_row_major);
  }
  __syncthreads();
  for (int idx = tid; idx < C * V; idx += nthreads) {
    kcd_out[kcd_base + idx] = __bfloat162float(__float2bfloat16(s_stage[idx]));
  }
}
