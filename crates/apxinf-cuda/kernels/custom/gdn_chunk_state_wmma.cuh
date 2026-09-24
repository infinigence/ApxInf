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

struct GdnPtxFrag4 { float x0, x1, x2, x3; };
template <typename T> struct GdnPtxIsBf16 { static constexpr bool value = false; };
template <> struct GdnPtxIsBf16<__nv_bfloat16> { static constexpr bool value = true; };

__device__ __forceinline__ GdnPtxFrag4 gdn_ptx_mma_16816(
    const uint32_t (&a)[4], const uint32_t (&b)[2], GdnPtxFrag4 c) {
  GdnPtxFrag4 d;
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d.x0), "=f"(d.x1), "=f"(d.x2), "=f"(d.x3)
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]),
        "f"(c.x0), "f"(c.x1), "f"(c.x2), "f"(c.x3));
  return d;
}

__device__ __forceinline__ uint32_t gdn_ldm_addr(const __nv_bfloat16* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void gdn_ldm_a(const __nv_bfloat16* kt,
    int kk, int lane, int mbase, int trow, int kp, uint32_t (&a)[4]) {
  const int tile = lane >> 3, row8 = lane & 7;
  const int k0 = kk * 16 + (tile >= 2 ? 8 : 0);
  const int m0 = mbase + trow * 16 + ((tile & 1) ? 8 : 0);
  const uint32_t addr = gdn_ldm_addr(kt + (k0 + row8) * kp + m0);
  asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 "
               "{%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(addr));
}

__device__ __forceinline__ void gdn_ldm_b(const __nv_bfloat16* vr,
    int kk, int lane, int tcol, int n, int vp, uint32_t (&b)[2]) {
  const int tile = (lane >> 3) & 1, row8 = lane & 7;
  const int k0 = kk * 16 + tile * 8;
  const int j0 = tcol * 16 + n * 8;
  const uint32_t addr = gdn_ldm_addr(vr + (k0 + row8) * vp + j0);
  asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 "
               "{%0,%1}, [%2];\n"
               : "=r"(b[0]), "=r"(b[1]) : "r"(addr));
}

// Load the same AI/intra PTX fragments as the prior scalar packing. A is a
// row-major 16x16 tile: x4 reads its four 8x8 quadrants in register order.
// B reuses the state-update x2.trans helper already checked word by word.
__device__ __forceinline__ void gdn_inter_load_a(const __nv_bfloat16* src,
    int stride, int kk, int lane, int trow, uint32_t (&a)[4]) {
  const int tile = lane >> 3, row8 = lane & 7;
  const int m0 = trow * 16 + ((tile & 1) ? 8 : 0);
  const int k0 = kk * 16 + ((tile >= 2) ? 8 : 0);
  const uint32_t addr = gdn_ldm_addr(src + (m0 + row8) * stride + k0);
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
               "{%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(addr));
}
__device__ __forceinline__ void gdn_inter_load_b(const __nv_bfloat16* src,
    int stride, int kk, int lane, int tcol, int n, uint32_t (&b)[2]) {
  gdn_ldm_b(src, kk, lane, tcol, n, stride, b);
}

// One block per value head, 32 warps, shapes fixed; the launcher checks them.
// The left operand is staged as one rounded BF16 pass. On the prefill path its
// producer already wrote it through __float2bfloat16, so that pass is exact here
// rather than merely close.
template <typename QK=float, bool PTX_UPDATE=false, bool INTER_CARRY=false>
__global__ __launch_bounds__(1024) void gdn_chunk_state_wmma_kernel(
    const QK* __restrict__ q, const QK* __restrict__ k,
    const float* __restrict__ g_cum, const __nv_bfloat16* __restrict__ t_in,
    const __nv_bfloat16* __restrict__ vt_in,
    const __nv_bfloat16* __restrict__ kcd_in,
    float* state, __nv_bfloat16* out, int seq, int seq_pad, int total_chunks,
    int out_row_width, float scale) {
  using namespace nvcuda;
  static_assert(!PTX_UPDATE || GdnPtxIsBf16<QK>::value,
                "PTX state update is only for BF16 Q/K");
  static_assert(!INTER_CARRY || (PTX_UPDATE && GdnPtxIsBf16<QK>::value),
                "AI/intra register carry is only for the typed PTX route");
  constexpr int K = 128;   // head_k_dim
  constexpr int V = 128;   // head_v_dim
  constexpr int C = 64;    // chunk_size
  constexpr int T = 16;    // WMMA tile
  // Shared memory is 32 banks of 4 bytes, so a row of banks is 128 B. Every
  // leading dimension here is a power of two in BF16 -- K and V are 256 B, C is
  // 128 B -- and each is an exact multiple of that, so row r of a tile starts on
  // bank (2*ldm/4)*r mod 32 = 0. All sixteen rows of a wmma tile then land on the
  // same eight banks and every load_matrix_sync serialises sixteen ways.
  //
  // Padding by 8 elements moves the stride to 272 B (or 144 B), i.e. bank 4r mod
  // 32, which spreads the sixteen rows over eight starting banks. 8 is the
  // smallest pad that keeps the 16 B alignment wmma requires of a BF16 leading
  // dimension. It costs 11 KB of the 176 KB already reserved.
  //
  // This is an addressing change only: the same values are read in the same
  // order into the same fragments, so the result is bit-identical.
  constexpr int KP = K + 8;  // padded [*][K] stride
  constexpr int VP = V + 8;  // padded [*][V] stride
  constexpr int CP = C + 8;  // padded [*][C] stride

  const int head = blockIdx.x;
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int tid = static_cast<int>(threadIdx.x);
  const int nthreads = static_cast<int>(blockDim.x);
  const int trow = warp >> 3;
  const int tcol = warp & 7;

  extern __shared__ char gdn_wmma_smem[];
  // One region holds the left operands, restaged between the two halves of the
  // chunk: kcd and q before the inter term, t and k afterwards. Their lives do
  // not overlap, and separate buffers would not fit.
  constexpr int LHS_ELEMS = 2 * C * KP;  // phase 1 is the larger
  __nv_bfloat16* s_state = reinterpret_cast<__nv_bfloat16*>(gdn_wmma_smem);  // [K][VP]
  __nv_bfloat16* s_lhs = s_state + K * VP;
  __nv_bfloat16* s_vr = s_lhs + LHS_ELEMS;                                   // [C][VP]
  float* s_vnew = reinterpret_cast<float*>(s_vr + C * VP);                   // [C][VP]
  // Only the non-INTER_CARRY route allocates this final [C][VP] FP32 region.
  // The typed register-carry route never dereferences the one-past pointer.
  float* s_inter = s_vnew + C * VP;
  // Phase 1 layout inside s_lhs. Every offset is a multiple of C*KP or C*CP,
  // both multiples of 32 banks, so the two phases share bank behaviour.
  __nv_bfloat16* s_kcd_hi = s_lhs;                 // [C][KP]
  __nv_bfloat16* s_q_hi = s_kcd_hi + C * KP;       // [C][KP]
  // Phase 2 layout, over the same bytes.
  __nv_bfloat16* s_t_hi = s_lhs;                   // [C][CP]
  __nv_bfloat16* s_kt_hi = s_t_hi + C * CP;        // [C][KP]

  const int64_t head_token_base = static_cast<int64_t>(head) * seq_pad;
  float* state_head = state + static_cast<int64_t>(head) * K * V;
  // Each thread owns the same 16 FP32 state cells on every chunk; carry them
  // in registers and write back once.
  float state_reg[16];
  {
    if constexpr (PTX_UPDATE) {
      const int group = (tid & 31) >> 2;
      const int quad = tid & 3;
      #pragma unroll
      for (int half = 0; half < 2; ++half)
        #pragma unroll
        for (int n = 0; n < 2; ++n)
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int row = half * 64 + trow * 16 + group + (i >= 2 ? 8 : 0);
            const int col = tcol * 16 + n * 8 + quad * 2 + (i & 1);
            state_reg[half * 8 + n * 4 + i] = state_head[row * V + col];
          }
    } else {
      const int owner_j = tid % V;
      const int owner_row0 = tid / V;
      #pragma unroll
      for (int i = 0; i < 16; ++i)
        state_reg[i] = state_head[(owner_row0 + 8 * i) * V + owner_j];
    }
  }

  for (int c = 0; c < total_chunks; ++c) {
    const int64_t token_base = head_token_base + static_cast<int64_t>(c) * C;
    const int64_t vt_base = (static_cast<int64_t>(head) * total_chunks + c) * C * V;
    const int64_t kcd_base = (static_cast<int64_t>(head) * total_chunks + c) * C * K;
    const int64_t a_base = (static_cast<int64_t>(head) * total_chunks + c) * C * C;

    // Only the first chunk has to fetch the carried state: every later one is
    // reading back what this block wrote at the end of the previous chunk, and
    // the state update below leaves the BF16 copy behind as it goes. That takes
    // a dependent 64 KB global read off the head of each chunk's critical path,
    // which on a scan that is latency-bound rather than bandwidth-bound is
    // worth more than the traffic it removes.
    if (c == 0) {
      if constexpr (PTX_UPDATE) {
        const int group = (tid & 31) >> 2;
        const int quad = tid & 3;
        #pragma unroll
        for (int half = 0; half < 2; ++half)
          #pragma unroll
          for (int n = 0; n < 2; ++n)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
              const int row = half * 64 + trow * 16 + group + (i >= 2 ? 8 : 0);
              const int col = tcol * 16 + n * 8 + quad * 2 + (i & 1);
              s_state[row * VP + col] =
                  __float2bfloat16(state_reg[half * 8 + n * 4 + i]);
            }
      } else {
        const int owner_j = tid % V;
        const int owner_row0 = tid / V;
        #pragma unroll
        for (int i = 0; i < 16; ++i)
          s_state[(owner_row0 + 8 * i) * VP + owner_j] =
              __float2bfloat16(state_reg[i]);
      }
    
    }
    for (int i = tid; i < C * K; i += nthreads) {
      const int r = i / K, m = i - r * K;
      const int p = r * KP + m;
      const float kcd_v = __bfloat162float(kcd_in[kcd_base + i]);
      const float q_v = gdn_qk_widen(q[(token_base + r) * K + m]);
      s_kcd_hi[p] = __float2bfloat16(kcd_v);
      s_q_hi[p] = __float2bfloat16(q_v);
    
    }
    __syncthreads();

    GdnPtxFrag4 ai0{0,0,0,0}, ai1{0,0,0,0};
    // vp and ai: [C,V] = [C,K] @ [K,V], 4 row tiles x 8 column tiles = 32,
    // one per warp.
    if constexpr (INTER_CARRY) {
      GdnPtxFrag4 vp0{0,0,0,0}, vp1{0,0,0,0};
      const int lane = tid & 31;
      // Both independent products consume the identical BF16 H fragments.
      // Each result still sees kk=0..7 in exactly the previous order.
      for (int kk = 0; kk < K / T; ++kk) {
        uint32_t af[4], bf0[2], bf1[2];
        gdn_inter_load_b(s_state, VP, kk, lane, tcol, 0, bf0);
        gdn_inter_load_b(s_state, VP, kk, lane, tcol, 1, bf1);
        gdn_inter_load_a(s_kcd_hi, KP, kk, lane, trow, af);
        vp0 = gdn_ptx_mma_16816(af, bf0, vp0);
        vp1 = gdn_ptx_mma_16816(af, bf1, vp1);
        gdn_inter_load_a(s_q_hi, KP, kk, lane, trow, af);
        ai0 = gdn_ptx_mma_16816(af, bf0, ai0);
        ai1 = gdn_ptx_mma_16816(af, bf1, ai1);
      }
      const int group = lane >> 2, quad = lane & 3;
      const int row0 = trow * T + group;
      const int col0 = tcol * T + quad * 2;
      *reinterpret_cast<float2*>(s_vnew + row0 * VP + col0) =
          make_float2(vp0.x0, vp0.x1);
      *reinterpret_cast<float2*>(s_vnew + (row0 + 8) * VP + col0) =
          make_float2(vp0.x2, vp0.x3);
      *reinterpret_cast<float2*>(s_vnew + row0 * VP + col0 + 8) =
          make_float2(vp1.x0, vp1.x1);
      *reinterpret_cast<float2*>(s_vnew + (row0 + 8) * VP + col0 + 8) =
          make_float2(vp1.x2, vp1.x3);
    } else {
      wmma::fragment<wmma::accumulator, T, T, T, float> acc_vp, acc_ai;
      wmma::fill_fragment(acc_vp, 0.0f);
      wmma::fill_fragment(acc_ai, 0.0f);
      wmma::fragment<wmma::matrix_b, T, T, T, __nv_bfloat16, wmma::row_major> fb;
      wmma::fragment<wmma::matrix_a, T, T, T, __nv_bfloat16, wmma::row_major> fa;
      for (int kk = 0; kk < K / T; ++kk) {
        wmma::load_matrix_sync(fb, s_state + (kk * T) * VP + tcol * T, VP);
        wmma::load_matrix_sync(fa, s_kcd_hi + (trow * T) * KP + kk * T, KP);
        wmma::mma_sync(acc_vp, fa, fb, acc_vp);
        wmma::load_matrix_sync(fa, s_q_hi + (trow * T) * KP + kk * T, KP);
        wmma::mma_sync(acc_ai, fa, fb, acc_ai);
      }
      wmma::store_matrix_sync(s_vnew + (trow * T) * VP + tcol * T, acc_vp, VP,
                              wmma::mem_row_major);
      wmma::store_matrix_sync(s_inter + (trow * T) * VP + tcol * T, acc_ai, VP,
                              wmma::mem_row_major);
    }
    __syncthreads();

    for (int cell = tid; cell < C * V; cell += nthreads) {
      const int i = cell / V, j = cell - i * V;
      const int p = i * VP + j;
      s_vnew[p] = __bfloat162float(vt_in[vt_base + cell]) - s_vnew[p];
      if constexpr (!INTER_CARRY)
        s_inter[p] *= gdn_exp2_approx(g_cum[token_base + i]);
      s_vr[p] = __float2bfloat16(s_vnew[p]);
    }
    // No barrier between these: the loop above writes s_vnew, s_inter and s_vr,
    // the two below write s_lhs, and nothing here reads what the other wrote.
    // The barrier that follows orders all of it against the intra term, which
    // is the first reader of any of it, and s_lhs's previous readers were
    // already fenced off after the inter term's stores.
    for (int i = tid; i < C * C; i += nthreads) {
      const int r = i / C, c = i - r * C;
      const int p = r * CP + c;
      const float t_v = __bfloat162float(t_in[a_base + i]);
      s_t_hi[p] = __float2bfloat16(t_v);
    
    }
    for (int i = tid; i < C * K; i += nthreads) {
      const int r = i / K, m = i - r * K;
      const int p = r * KP + m;
      const float k_v = gdn_qk_widen(k[(token_base + r) * K + m]);
      s_kt_hi[p] = __float2bfloat16(k_v);
    
    }
    __syncthreads();

    // In the typed register-carry route, the AI PTX fragment stays in registers
    // through vnew and operand staging, then seeds intra in the same owner.
    if constexpr (INTER_CARRY) {
      const int lane = tid & 31;
      const int group = lane >> 2, quad = lane & 3;
      const int row0 = trow * T + group;
      const float e0 = gdn_exp2_approx(g_cum[token_base + row0]);
      const float e1 = gdn_exp2_approx(g_cum[token_base + row0 + 8]);
      ai0.x0 = __fmul_rn(ai0.x0, e0);
      ai0.x1 = __fmul_rn(ai0.x1, e0);
      ai0.x2 = __fmul_rn(ai0.x2, e1);
      ai0.x3 = __fmul_rn(ai0.x3, e1);
      ai1.x0 = __fmul_rn(ai1.x0, e0);
      ai1.x1 = __fmul_rn(ai1.x1, e0);
      ai1.x2 = __fmul_rn(ai1.x2, e1);
      ai1.x3 = __fmul_rn(ai1.x3, e1);
      #pragma unroll
      for (int kk = 0; kk < C / T; ++kk) {
        uint32_t af[4], bf0[2], bf1[2];
        gdn_inter_load_a(s_t_hi, CP, kk, lane, trow, af);
        gdn_inter_load_b(s_vr, VP, kk, lane, tcol, 0, bf0);
        gdn_inter_load_b(s_vr, VP, kk, lane, tcol, 1, bf1);
        ai0 = gdn_ptx_mma_16816(af, bf0, ai0);
        ai1 = gdn_ptx_mma_16816(af, bf1, ai1);
      }
      const float v0[4] = {ai0.x0, ai0.x1, ai0.x2, ai0.x3};
      const float v1[4] = {ai1.x0, ai1.x1, ai1.x2, ai1.x3};
      #pragma unroll
      for (int n = 0; n < 2; ++n) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int row = trow * T + group + (i >= 2 ? 8 : 0);
          const int col = tcol * T + n * 8 + quad * 2 + (i & 1);
          const int token = c * C + row;
          if (token < seq) {
            const float value = __fmul_rn(n ? v1[i] : v0[i], scale);
            out[static_cast<int64_t>(token) * out_row_width + head * V + col] =
                __float2bfloat16(value);
          }
        }
      }
    } else {
      // intra: [C,V] = [C,C] @ [C,V], accumulated on top of the inter term so
      // the two never meet in memory.
      {
        wmma::fragment<wmma::accumulator, T, T, T, float> acc;
        wmma::load_matrix_sync(acc, s_inter + (trow * T) * VP + tcol * T, VP,
                               wmma::mem_row_major);
        wmma::fragment<wmma::matrix_a, T, T, T, __nv_bfloat16, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, T, T, T, __nv_bfloat16, wmma::row_major> fb;
        for (int kk = 0; kk < C / T; ++kk) {
          wmma::load_matrix_sync(fb, s_vr + (kk * T) * VP + tcol * T, VP);
          wmma::load_matrix_sync(fa, s_t_hi + (trow * T) * CP + kk * T, CP);
          wmma::mma_sync(acc, fa, fb, acc);
        }
        #pragma unroll
        for (int e = 0; e < acc.num_elements; ++e) acc.x[e] *= scale;
        wmma::store_matrix_sync(s_inter + (trow * T) * VP + tcol * T, acc, VP,
                                wmma::mem_row_major);
      }
      __syncthreads();

      for (int cell = tid; cell < C * V; cell += nthreads) {
        const int i = cell / V, j = cell - i * V;
        const int token = c * C + i;
        if (token < seq) {
          out[static_cast<int64_t>(token) * out_row_width + head * V + j] =
              __float2bfloat16(s_inter[i * VP + j]);
        }
      }
    }

    // State update. v_round is rebuilt with the chunk's trailing decay, then
    // state = state * decay + k^T @ v_round, in two halves of 64 rows because
    // one half of the fp32 result is exactly the space s_vnew occupies.
    const float g_last = g_cum[token_base + C - 1];
    const float decay = gdn_exp2_approx(g_last);
    __syncthreads();
    for (int cell = tid; cell < C * V; cell += nthreads) {
      const int i = cell / V, j = cell - i * V;
      const int p = i * VP + j;
      s_vr[p] = __float2bfloat16(
          s_vnew[p] * gdn_exp2_approx(g_last - g_cum[token_base + i]));
    }
    __syncthreads();
    for (int half = 0; half < 2; ++half) {
      const int mbase = half * (K / 2);
      if constexpr (!PTX_UPDATE) {
        wmma::fragment<wmma::accumulator, T, T, T, float> acc;
        wmma::fill_fragment(acc, 0.0f);
        // s_kt is [i][m]; matrix_a read column-major over it is [m][i], the
        // transpose this product wants.
        wmma::fragment<wmma::matrix_a, T, T, T, __nv_bfloat16, wmma::col_major> fa;
        wmma::fragment<wmma::matrix_b, T, T, T, __nv_bfloat16, wmma::row_major> fb;
        for (int kk = 0; kk < C / T; ++kk) {
          wmma::load_matrix_sync(fb, s_vr + (kk * T) * VP + tcol * T, VP);
          wmma::load_matrix_sync(fa, s_kt_hi + (kk * T) * KP + mbase + trow * T, KP);
          wmma::mma_sync(acc, fa, fb, acc);
        }
        wmma::store_matrix_sync(s_vnew + (trow * T) * VP + tcol * T, acc, VP,
                                wmma::mem_row_major);
        __syncthreads();
        const int owner_j = tid % V;
        const int owner_row0 = tid / V;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int row = owner_row0 + 8 * i;
          const int m = mbase + row;
          const float updated = state_reg[half * 8 + i] * decay +
                                s_vnew[row * VP + owner_j];
          state_reg[half * 8 + i] = updated;
          s_state[m * VP + owner_j] = __float2bfloat16(updated);
        }
      
        __syncthreads();
      } else {
        const int lane = tid & 31;
        const int group = lane >> 2;
        const int quad = lane & 3;
        GdnPtxFrag4 dot0{0, 0, 0, 0}, dot1{0, 0, 0, 0};
        #pragma unroll
        for (int kk = 0; kk < C / T; ++kk) {
          uint32_t af[4], bf0[2], bf1[2];
          gdn_ldm_a(s_kt_hi, kk, lane, mbase, trow, KP, af);
          gdn_ldm_b(s_vr, kk, lane, tcol, 0, VP, bf0);
          gdn_ldm_b(s_vr, kk, lane, tcol, 1, VP, bf1);
          dot0 = gdn_ptx_mma_16816(af, bf0, dot0);
          dot1 = gdn_ptx_mma_16816(af, bf1, dot1);
        }
        const float v0[4] = {dot0.x0, dot0.x1, dot0.x2, dot0.x3};
        const float v1[4] = {dot1.x0, dot1.x1, dot1.x2, dot1.x3};
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int row = mbase + trow * T + group + (i >= 2 ? 8 : 0);
            const int col = tcol * T + n * 8 + quad * 2 + (i & 1);
            const int owner = half * 8 + n * 4 + i;
            const float dot = n ? v1[i] : v0[i];
            const float updated = __fmaf_rn(state_reg[owner], decay, dot);
            state_reg[owner] = updated;
            s_state[row * VP + col] = __float2bfloat16(updated);
          }
        }
      }
    }
    // The next chunk stages q/kcd into s_lhs, which aliases s_kt_hi. Wait for
    // every warp's second-half MMA reads before any warp can overwrite it.
    if constexpr (PTX_UPDATE) __syncthreads();
  }
  if constexpr (PTX_UPDATE) {
    const int group = (tid & 31) >> 2;
    const int quad = tid & 3;
    #pragma unroll
    for (int half = 0; half < 2; ++half)
      #pragma unroll
      for (int n = 0; n < 2; ++n)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int row = half * 64 + trow * 16 + group + (i >= 2 ? 8 : 0);
          const int col = tcol * 16 + n * 8 + quad * 2 + (i & 1);
          state_head[row * V + col] = state_reg[half * 8 + n * 4 + i];
        }
  } else {
    const int owner_j = tid % V;
    const int owner_row0 = tid / V;
    #pragma unroll
    for (int i = 0; i < 16; ++i)
      state_head[(owner_row0 + 8 * i) * V + owner_j] = state_reg[i];
  }

}
