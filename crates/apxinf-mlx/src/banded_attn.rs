//! Banded relative-bias attention, attention-residual depth mixing, NAX full
//! prefill attention, and exact 2-pass multi-row decode/verify attention on Metal.
//!
//! Ported from EngineTailor's Inkling-2 MLX kernels (Apache-2.0) and
//! Apple MLX 0.32.4's `sdpa_vector` / `steel_attention_nax` (MIT).

use crate::{loaded_library_path, Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::{Error, Result};
use std::cell::{Cell, OnceCell, RefCell};
use std::collections::HashMap;
use std::io::Read;
use std::sync::OnceLock;

const ATTN_RES_TG: usize = 256;
const ATTN_RES_REGS_MAX_ROWS: usize = 8;
const FLASH_MIN_QUERIES: usize = 32;
const NAX_BQ: usize = 64;
const NAX_BK: usize = 32;
const MASK_BUDGET: usize = 1 << 29;
const FULL_MASK_CAP: usize = 1 << 31;
const FULL_ALIGN: usize = 64;

const STEEL_ATTN_NAX_PREFIX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/steel_attn_nax_header.metal"));

const NAX_KERNEL_FN: &str = r#"
namespace inkling2_nax {

using namespace mlx::steel;

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return metal::max(x, y); }
};
struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return x + y; }
};
struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return x * y; }
};
struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) { return fast::exp2(x - y); }
};

template <typename T, int BQ, int BK, int BD, int BV, int WM, int WN, int HQ, int HKV, int R>
METAL_FUNC void banded_attention(
    const device T* Q, const device T* K, const device T* V,
    const device T* rel, const device T* sinks, device T* O,
    const constant int64_t* q_strides, const constant int64_t* k_strides,
    const constant int64_t* v_strides, const constant int64_t* rel_strides,
    const constant int64_t* sink_strides, const int qL, const int kL,
    const float scale, uint simd_lane_id, uint simd_group_id, uint3 tid) {
  using AccumType = float;
  (void)simd_lane_id;
  ulong3 tidl{tid.x, tid.y, tid.z};
  const int b = int(tid.z);
  const int h = int(tid.y);
  const int qL_off = kL - qL;
  const int NQ_aligned = qL / BQ;
  const int qL_rem = qL - NQ_aligned * BQ;
  const int NK = (kL + BK - 1) / BK;
  const int NK_aligned = kL / BK;
  const int kL_rem = kL - NK_aligned * BK;
  const bool align_Q = qL_rem == 0;
  const bool align_K = kL_rem == 0;
  const int q_ld = int(q_strides[2]);
  const int k_ld = int(k_strides[2]);
  const int v_ld = int(v_strides[2]);
  constexpr int o_ld = HQ * BV;

  Q += tidl.z * q_strides[0] + tidl.y * q_strides[1] + tidl.x * BQ * q_strides[2];
  const ulong kv_head_idx = ulong(h / (HQ / HKV));
  K += tidl.z * k_strides[0] + kv_head_idx * k_strides[1];
  V += tidl.z * v_strides[0] + kv_head_idx * v_strides[1];
  O += tidl.z * ulong(qL) * o_ld + tidl.y * BV + tidl.x * BQ * ulong(o_ld);

  const metal::uniform<float> scale2 =
      make_uniform(scale) * make_uniform(1.44269504089f);

  constexpr short kU = 16;
  constexpr int kNWarps = WM * WN;
  static_assert(BQ >= (kNWarps * kU) && BQ % (kNWarps * kU) == 0, "BQ");
  constexpr int TQ = BQ / (kNWarps * kU);
  constexpr int TD = BD / kU;
  constexpr int TV = BV / kU;
  constexpr short TK = BK / kU;
  static_assert(TQ == 1, "Check TQ");
  using otile_t = NAXTile<AccumType, TQ, TV>;
  otile_t Otile;
  Otile.clear();

  const short tm = kU * TQ * simd_group_id;
  Q += tm * q_ld;

  const short2 simd_coord = otile_t::NAXFrag_t::get_coord();
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;

  constexpr short kRowsPT = otile_t::kRowsPerThread;
  metal::vec<AccumType, kRowsPT> max_score;
  metal::vec<AccumType, kRowsPT> sum_score{0};

  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    int r = int(tid.x) * BQ + tm + i * otile_t::kFragRowsJump + sm;
    r = r < qL ? r : qL - 1;
    const AccumType s = static_cast<AccumType>(
        sinks[b * sink_strides[0] + r * sink_strides[1] + h * sink_strides[2]]);
    max_score[i] = M_LOG2E_F * s;
    sum_score[i] = 1;
  }

  int kb_lim;
  int kb_min_causal;
  {
    int q_max = (int(tid.x) + 1) * BQ + qL_off;
    kb_lim = min(NK, (q_max + BK - 1) / BK);
    int q_min = max(0, int(tid.x) * BQ + qL_off);
    kb_min_causal = q_min / BK;
  }
  const int band_first_col = int(tid.x) * BQ + qL_off - R + 1;

  const bool is_last_q = int(tid.x) == NQ_aligned;
  const short lim_rows_q = qL_rem - tm;
  const short lim_rows_k = kL_rem;

  for (int kb = 0; kb < kb_lim; kb++) {
    const int is_last_k = (kb == NK_aligned);

    using stile_t = NAXTile<AccumType, TQ, TK>;
    stile_t Stile;
    Stile.clear();

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik += 2) {
#pragma clang loop unroll_count(4)
        for (short id = 0; id < TD; id++) {
          NAXTile<T, 1, 1> Qtile;
          NAXTile<T, 2, 1> Ktile;
          const int Q_load_off = iq * kU * q_ld + id * kU;
          const int K_load_off = ik * kU * k_ld + id * kU;
          if (!align_Q && is_last_q) {
            Qtile.load_rows(Q + Q_load_off, q_ld, lim_rows_q - iq * kU);
          } else {
            Qtile.load(Q + Q_load_off, q_ld);
          }
          if (!align_K && is_last_k) {
            Ktile.load_rows(K + K_load_off, k_ld, lim_rows_k - ik * kU);
          } else {
            Ktile.load(K + K_load_off, k_ld);
          }
          stile_t::NAXFrag_t::mma(
              Stile.frag_at(iq, ik), Stile.frag_at(iq, ik + 1),
              Qtile.frag_at(0, 0), metal::false_type{},
              Ktile.frag_at(0, 0), Ktile.frag_at(1, 0), metal::true_type{});
        }
      }
    }

    STEEL_PRAGMA_UNROLL
    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {
      Stile.elems()[ii] *= float(scale2);
    }

    if (!align_K && is_last_k) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const short col_pos = ik * kU + sn;
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = ((col_pos + jj) < kL_rem) ? fg[loc] : neg_inf;
            }
          }
        }
      }
    }

    if (kb >= kb_min_causal) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;
      const int base_row = int(tid.x) * BQ + qL_off + tm;
      const int base_col = kb * BK;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const auto r = base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
              const auto c = base_col + ik * kU + jj + sn;
              const auto loc = ii * stile_t::kFragThrCols + jj;
              fg[loc] = (r < c) ? neg_inf : fg[loc];
            }
          }
        }
      }
    }

    if (kb * BK + BK - 1 >= band_first_col) {
      const int base_row = int(tid.x) * BQ + tm;
      const int base_col = kb * BK;
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; iq++) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          thread auto& fg = Stile.frag_at(iq, ik);
          STEEL_PRAGMA_UNROLL
          for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
            STEEL_PRAGMA_UNROLL
            for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
              const int r = base_row + iq * kU + ii * stile_t::kFragRowsJump + sm;
              const int c = base_col + ik * kU + jj + sn;
              const int dist = qL_off + r - c;
              const auto loc = ii * stile_t::kFragThrCols + jj;
              if (dist >= 0 && dist < R && r < qL && c < kL) {
                fg[loc] += M_LOG2E_F * AccumType(rel[
                    b * rel_strides[0] + r * rel_strides[1] +
                    h * rel_strides[2] + dist * rel_strides[3]]);
              }
            }
          }
        }
      }
    }

    metal::vec<AccumType, kRowsPT> new_max;
    metal::vec<AccumType, kRowsPT> factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }
    Stile.template row_reduce<MaxOp>(new_max);
    Stile.template row_bin_op<ExpSubOp>(new_max);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i];
    }
    Stile.template row_reduce<SumOp>(sum_score);
    Otile.template row_bin_op<MulOp>(factor);

    simdgroup_barrier(mem_flags::mem_none);

    STEEL_PRAGMA_UNROLL
    for (short iq = 0; iq < TQ; iq++) {
      STEEL_PRAGMA_UNROLL
      for (short id = 0; id < TV; id += 2) {
        if constexpr (BV == 128) {
          if (id == 4) {
            threadgroup_barrier(mem_flags::mem_none);
          }
        }
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          NAXTile<T, 1, 2> Vtile;
          const int V_load_off = ik * kU * v_ld + id * kU;
          if (!align_K && is_last_k) {
            Vtile.load_rows(V + V_load_off, v_ld, lim_rows_k - ik * kU);
          } else {
            Vtile.load(V + V_load_off, v_ld);
          }
          otile_t::NAXFrag_t::mma(
              Otile.frag_at(iq, id), Otile.frag_at(iq, id + 1),
              Stile.frag_at(iq, ik), metal::false_type{},
              Vtile.frag_at(0, 0), Vtile.frag_at(0, 1), metal::false_type{});
        }
      }
    }

    K += BK * k_ld;
    V += BK * v_ld;
  }

  threadgroup_barrier(mem_flags::mem_none);

  metal::vec<AccumType, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.f / sum_score[i];
  }
  Otile.template row_bin_op<MulOp>(rcp);

  O += tm * o_ld;
  if (!align_Q && is_last_q) {
    if (lim_rows_q <= 0)
      return;
    Otile.store_rows(O, o_ld, lim_rows_q);
  } else {
    Otile.store(O, o_ld);
  }
}

}  // namespace inkling2_nax
"#;

const NAX_BODY: &str = r#"
    inkling2_nax::banded_attention<T, BQ, BK, 128, 128, 4, 1, HQ, HKV, R>(
        q, k, v, rel, sink, out, q_strides, k_strides, v_strides,
        rel_strides, sink_strides, dims[0], dims[1], scale[0],
        thread_index_in_simdgroup, simdgroup_index_in_threadgroup,
        threadgroup_position_in_grid);
"#;

const FLASH_SOURCE: &str = r#"
    constexpr int BQ = 32;
    constexpr int BK = 16;
    constexpr int NT = 128;
    constexpr float LOG2E = 1.4426950408889634f;
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = tid / 32;
    const uint lane = tid % 32;
    const int qt = threadgroup_position_in_grid.x;
    const int h = threadgroup_position_in_grid.y;
    const int b = threadgroup_position_in_grid.z;
    const int L = dims[0];
    const int S = dims[1];
    const int lp = dims[2 + b];
    const int hk = h / (HQ / HKV);
    const int q0 = qt * BQ;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    const int row = q0 + int(sg) * 8 + fm;
    const bool row_ok = row < L;
    const int a_row = S - L + row;
    const float qscale = scale[0] * LOG2E;

    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    threadgroup half Ks[D * LDK];
    threadgroup half Vs[BK * LDV];

    simdgroup_half8x8 Qf[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        float2 qv = float2(0.0f);
        if (row_ok) {
            const device T* qp = q + b * q_strides[0] + h * q_strides[1]
                + row * q_strides[2] + (db * 8 + fn) * q_strides[3];
            qv = float2(float(qp[0]), float(qp[q_strides[3]]));
        }
        Qf[db].thread_elements()[0] = half(qv.x);
        Qf[db].thread_elements()[1] = half(qv.y);
    }
    simdgroup_float8x8 Of[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        Of[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float m = row_ok ? float(sink[(b * L + row) * HQ + h]) * LOG2E : 0.0f;
    float l = 1.0f;
    const device T* relp = rel + ((size_t)(b * L + (row_ok ? row : 0)) * HQ + h) * R;

    const int a_first = S - L + q0;
    const int a_last = S - L + min(q0 + BQ, L) - 1;
    int lo = WINDOW > 0 ? max(0, a_first - WINDOW + 1) : 0;
    lo = max(lo, lp);
    const int k_begin = (lo / BK) * BK;
    const device T* kb_ptr = k + b * k_strides[0] + hk * k_strides[1];
    const device T* vb_ptr = v + b * v_strides[0] + hk * v_strides[1];
    const bool vec_kv = k_strides[3] == 1 && v_strides[3] == 1;

    for (int k0 = k_begin; k0 <= a_last; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (vec_kv) {
            for (int c = tid; c < BK * D / 8; c += NT) {
                const int r = c / (D / 8);
                const int col = (c % (D / 8)) * 8;
                const int key = k0 + r;
                uint4 kw = uint4(0);
                uint4 vw = uint4(0);
                if (key < S) {
                    kw = *(const device uint4*)(kb_ptr + key * k_strides[2] + col);
                    vw = *(const device uint4*)(vb_ptr + key * v_strides[2] + col);
                }
                threadgroup half* kd = Ks + col * LDK + r;
                threadgroup half* vd = Vs + r * LDV + col;
                for (int j = 0; j < 4; ++j) {
                    kd[(2 * j) * LDK] = half(as_type<float>(kw[j] << 16));
                    kd[(2 * j + 1) * LDK] = half(as_type<float>(kw[j] & 0xFFFF0000u));
                    vd[2 * j] = half(as_type<float>(vw[j] << 16));
                    vd[2 * j + 1] = half(as_type<float>(vw[j] & 0xFFFF0000u));
                }
            }
        } else {
            for (int c = tid; c < BK * D; c += NT) {
                const int r = c / D;
                const int col = c % D;
                const int key = k0 + r;
                float kv = 0.0f;
                float vv = 0.0f;
                if (key < S) {
                    kv = float(kb_ptr[key * k_strides[2] + col * k_strides[3]]);
                    vv = float(vb_ptr[key * v_strides[2] + col * v_strides[3]]);
                }
                Ks[col * LDK + r] = half(kv);
                Vs[r * LDV + col] = half(vv);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + int(sg) * 8 >= L) {
            continue;
        }

        simdgroup_float8x8 Sf[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            Sf[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 Kt;
                simdgroup_load(Kt, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(Sf[kb], Qf[db], Kt, Sf[kb]);
            }
        }

        float rmax = -INFINITY;
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const int key = k0 + kb * 8 + fn + e;
                const int d = a_row - key;
                const bool vis = row_ok && d >= 0 && key >= lp
                    && (WINDOW == 0 || d < WINDOW);
                float s = Sf[kb].thread_elements()[e] * qscale;
                if (vis) {
                    if (d < R) {
                        s += float(relp[d]) * LOG2E;
                    }
                } else {
                    s = -INFINITY;
                }
                Sf[kb].thread_elements()[e] = s;
                rmax = max(rmax, s);
            }
        }
        rmax = max(rmax, simd_shuffle_xor(rmax, 1));
        rmax = max(rmax, simd_shuffle_xor(rmax, 8));
        const float m_new = max(m, rmax);
        const float factor = fast::exp2(m - m_new);
        float rsum = 0.0f;
        simdgroup_half8x8 Pf[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float p = fast::exp2(Sf[kb].thread_elements()[e] - m_new);
                Pf[kb].thread_elements()[e] = half(p);
                rsum += p;
            }
        }
        rsum += simd_shuffle_xor(rsum, 1);
        rsum += simd_shuffle_xor(rsum, 8);
        l = l * factor + rsum;
        m = m_new;
        for (int db = 0; db < D / 8; ++db) {
            Of[db].thread_elements()[0] *= factor;
            Of[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_half8x8 Vm;
                simdgroup_load(Vm, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(Of[db], Pf[kb], Vm, Of[db]);
            }
        }
    }

    if (row_ok) {
        const float inv = 1.0f / l;
        device T* op = out + ((size_t)(b * L + row) * HQ + h) * D;
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = T(Of[db].thread_elements()[0] * inv);
            op[db * 8 + fn + 1] = T(Of[db].thread_elements()[1] * inv);
        }
    }
"#;

const XATTN_PASS1: &str = r#"
    constexpr int BD = 32;
    constexpr int qk_per_thread = D / BD;
    constexpr int v_per_thread = D / BD;
    constexpr int TK = 4;
    typedef float U;
    thread U qv[qk_per_thread];
    thread U o[v_per_thread] = {0};

    const int kv_head_idx = threadgroup_position_in_grid.x;
    const int r = threadgroup_position_in_grid.y;
    const int block_idx = threadgroup_position_in_grid.z;
    const int g = thread_position_in_threadgroup.y;
    const uint simd_lid = thread_index_in_simdgroup;
    const int h = GQA * kv_head_idx + g;
    const int s_r = dims[0] + r + 1;
    const int lo = WINDOW > 0 ? max(0, s_r - WINDOW) : 0;
    const int N = s_r - lo;
    const int o_offset = h * L + r;

    const device T* queries = q + o_offset * D + simd_lid * qk_per_thread;
    const device T* keys = k + kv_head_idx * k_strides[1] + (lo + block_idx) * k_strides[2]
        + simd_lid * qk_per_thread;
    const device T* values = v + kv_head_idx * v_strides[1] + (lo + block_idx) * v_strides[2]
        + simd_lid * v_per_thread;
    const device T* relp = rel + (r * HQ + h) * R;
    device T* outp = partials + o_offset * BLOCKS * D + block_idx * D + simd_lid * v_per_thread;

    for (int i = 0; i < qk_per_thread; i++) {
      qv[i] = static_cast<U>(scale[0]) * queries[i];
    }
    U max_score = Limits<U>::finite_min;
    U sum_exp_score = 0;
    if (block_idx == 0) {
      max_score = static_cast<U>(sinks[r * HQ + h]);
      sum_exp_score = 1;
    }
    const int stride_k_tok = BLOCKS * int(k_strides[2]);
    const int stride_v_tok = BLOCKS * int(v_strides[2]);

    int i = block_idx;
    for (; i + (TK - 1) * BLOCKS < N; i += BLOCKS * TK) {
      T k_reg[TK][qk_per_thread];
      T v_reg[TK][v_per_thread];
      for (int t = 0; t < TK; t++) {
        const device T* kp = keys + t * stride_k_tok;
        for (int j = 0; j < qk_per_thread; j++) {
          k_reg[t][j] = kp[j];
        }
      }
      for (int t = 0; t < TK; t++) {
        const device T* vp = values + t * stride_v_tok;
        for (int j = 0; j < v_per_thread; j++) {
          v_reg[t][j] = vp[j];
        }
      }
      for (int t = 0; t < TK; t++) {
        int ii = i + t * BLOCKS;
        int dd = N - 1 - ii;
        T fm = dd < R ? relp[dd] : T(0);
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
          score += qv[j] * static_cast<U>(k_reg[t][j]);
        }
        score = simd_sum(score);
        score += static_cast<U>(fm);
        U new_max = max(max_score, score);
        U factor = fast::exp(max_score - new_max);
        U exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;
        for (int j = 0; j < v_per_thread; j++) {
          o[j] = o[j] * factor + exp_score * static_cast<U>(v_reg[t][j]);
        }
      }
      keys += TK * stride_k_tok;
      values += TK * stride_v_tok;
    }
    for (; i < N; i += BLOCKS) {
      int dd = N - 1 - i;
      T fm = dd < R ? relp[dd] : T(0);
      U score = 0;
      for (int j = 0; j < qk_per_thread; j++) {
        score += qv[j] * keys[j];
      }
      score = simd_sum(score);
      score += fm;
      U new_max = max(max_score, score);
      U factor = fast::exp(max_score - new_max);
      U exp_score = fast::exp(score - new_max);
      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor + exp_score * values[j];
      }
      keys += stride_k_tok;
      values += stride_v_tok;
    }
    if (simd_lid == 0) {
      sums[o_offset * BLOCKS + block_idx] = sum_exp_score;
      maxs[o_offset * BLOCKS + block_idx] = max_score;
    }
    for (int j = 0; j < v_per_thread; j++) {
      outp[j] = static_cast<T>(o[j]);
    }
"#;

const XATTN_PASS2: &str = r#"
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int elem_per_thread = D / BD;
    typedef float U;
    thread U o[elem_per_thread] = {0};
    threadgroup U outputs[BN * BD];
    const int q_offset = threadgroup_position_in_grid.x;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const device T* pp = partials + q_offset * BLOCKS * D + simd_gid * D + simd_lid * elem_per_thread;
    const device float* sp = sums + q_offset * BLOCKS;
    const device float* mp = maxs + q_offset * BLOCKS;
    device T* op = out + q_offset * D + simd_gid * elem_per_thread;

    U sum_exp_score = 0.0;
    U max_score = Limits<U>::finite_min;
    for (int b = 0; b < BLOCKS / BN; ++b) {
      max_score = max(max_score, mp[simd_lid + BN * b]);
    }
    max_score = simd_max(max_score);
    for (int b = 0; b < BLOCKS / BN; ++b) {
      U factor = fast::exp(mp[simd_lid + BN * b] - max_score);
      sum_exp_score += factor * sp[simd_lid + BN * b];
    }
    sum_exp_score = simd_sum(sum_exp_score);
    for (int b = 0; b < BLOCKS / BN; ++b) {
      U factor = fast::exp(mp[simd_gid] - max_score);
      for (int i = 0; i < elem_per_thread; i++) {
        o[i] += factor * static_cast<U>(pp[i]);
      }
      mp += BN;
      sp += BN;
      pp += BN * D;
    }
    for (int i = 0; i < elem_per_thread; i++) {
      outputs[simd_lid * BD + simd_gid] = o[i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
      o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (simd_lid == 0) {
      for (int i = 0; i < elem_per_thread; i++) {
        op[i] = static_cast<T>(o[i]);
      }
    }
"#;

const XATTN_XR_PROBE: &str = r#"
    uint gid = thread_position_in_grid.x;
    uint hsh = gid * 2654435761u + 0x9e3779b9u;
    hsh ^= hsh >> 15; hsh *= 2246822519u; hsh ^= hsh >> 13;
    float x = (float(hsh & 0xffffu) / 65536.0f - 0.5f) * metal::exp2(float(int((hsh >> 16) % 40u)) - 20.0f);
    float s0 = simd_sum(x);
    float b = x;
    b = b + simd_shuffle_xor(b, 1);
    b = b + simd_shuffle_xor(b, 2);
    b = b + simd_shuffle_xor(b, 4);
    b = b + simd_shuffle_xor(b, 8);
    b = b + simd_shuffle_xor(b, 16);
    out[gid] = (as_type<uint>(s0) == as_type<uint>(b)) ? 0 : 1;
"#;

fn xattn_pass1_rows_source() -> String {
    let row_old = "    const int r = threadgroup_position_in_grid.y;\n";
    let head_old = "    const int g = thread_position_in_threadgroup.y;\n";
    debug_assert_eq!(XATTN_PASS1.matches(row_old).count(), 1);
    debug_assert_eq!(XATTN_PASS1.matches(head_old).count(), 1);
    XATTN_PASS1
        .replace(row_old, "    const int r = thread_position_in_threadgroup.y / GQA;\n")
        .replace(head_old, "    const int g = thread_position_in_threadgroup.y % GQA;\n")
}

fn xattn_pass1_rows_xr_source() -> String {
    let base = xattn_pass1_rows_source();
    let old_tk = "    constexpr int TK = 4;\n";
    let old_loop = r#"      for (int t = 0; t < TK; t++) {
        int ii = i + t * BLOCKS;
        int dd = N - 1 - ii;
        T fm = dd < R ? relp[dd] : T(0);
        U score = 0;
        for (int j = 0; j < qk_per_thread; j++) {
          score += qv[j] * static_cast<U>(k_reg[t][j]);
        }
        score = simd_sum(score);
        score += static_cast<U>(fm);
"#;
    let mut reduce = String::new();
    let mut cur: Vec<String> = (0..8).map(|t| format!("sc[{t}]")).collect();
    for lev in 0..3 {
        let m = 1usize << lev;
        let h = cur.len() / 2;
        reduce.push_str(&format!("      const bool xb{lev} = (simd_lid & {m}u) != 0;\n"));
        let mut next = Vec::with_capacity(h);
        for u in 0..h {
            reduce.push_str(&format!(
                "      const U x{lev}_{u} = (xb{lev} ? {} : {}) + simd_shuffle_xor(xb{lev} ? {} : {}, {m});\n",
                cur[u + h], cur[u], cur[u], cur[u + h]
            ));
            next.push(format!("x{lev}_{u}"));
        }
        cur = next;
    }
    reduce.push_str(&format!("      U xr = {};\n", cur[0]));
    for lev in 3..5 {
        reduce.push_str(&format!("      xr = xr + simd_shuffle_xor(xr, {});\n", 1usize << lev));
    }
    for t in 0..8usize {
        // 3-bit reversal of t
        let lane = ((t & 1) << 2) | (t & 2) | ((t >> 2) & 1);
        reduce.push_str(&format!("      sc[{t}] = simd_shuffle(xr, {lane});\n"));
    }
    let replacement = format!(
        "      U sc[TK];\n      for (int t = 0; t < TK; t++) {{\n        U score = 0;\n        for (int j = 0; j < qk_per_thread; j++) {{\n          score += qv[j] * static_cast<U>(k_reg[t][j]);\n        }}\n        sc[t] = score;\n      }}\n{reduce}      for (int t = 0; t < TK; t++) {{\n        int ii = i + t * BLOCKS;\n        int dd = N - 1 - ii;\n        T fm = dd < R ? relp[dd] : T(0);\n        U score = sc[t];\n        score += static_cast<U>(fm);\n"
    );
    debug_assert_eq!(base.matches(old_tk).count(), 1);
    debug_assert_eq!(base.matches(old_loop).count(), 1);
    base.replace(old_tk, "    constexpr int TK = 8;\n")
        .replace(old_loop, &replacement)
}

fn xattn_pass2_gate_source() -> String {
    let op_decl = "    device T* op = out + q_offset * D + simd_gid * elem_per_thread;\n";
    let old_write = r#"    if (simd_lid == 0) {
      for (int i = 0; i < elem_per_thread; i++) {
        op[i] = static_cast<T>(o[i]);
      }
    }
"#;
    let new_write = r#"    if (simd_lid == 0) {
      const int hh = q_offset / L;
      const int rr = q_offset - hh * L;
      for (int i = 0; i < elem_per_thread; i++) {
        const int c = hh * D + simd_gid * elem_per_thread + i;
        const float ov = static_cast<float>(static_cast<T>(o[i]));
        const float gv = static_cast<float>(gate[rr * gate_strides[1] + c * gate_strides[2]]);
        const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(gv)));
        const float sg = (gv < 0) ? y : 1.0f - y;
        out[rr * (HQ * D) + c] = static_cast<GT>(ov * sg);
      }
    }
"#;
    debug_assert_eq!(XATTN_PASS2.matches(op_decl).count(), 1);
    debug_assert_eq!(XATTN_PASS2.matches(old_write).count(), 1);
    XATTN_PASS2
        .replace(op_decl, "")
        .replace(old_write, new_write)
}

fn build_attn_res_source(blocks: usize, regs: bool) -> (Vec<String>, String) {
    let mut srcs: Vec<String> = (0..blocks).map(|i| format!("b{i}")).collect();
    srcs.push("p".into());
    let ns = srcs.len();
    let loads = srcs
        .iter()
        .enumerate()
        .map(|(i, name)| format!("        float s{i} = {name}[base + h];"))
        .collect::<Vec<_>>()
        .join("\n");
    let dots = (0..ns)
        .map(|i| format!("        acc[{i} * Q + qi] += s{i} * qv;"))
        .collect::<Vec<_>>()
        .join("\n");
    let mix = (0..ns)
        .map(|i| format!("w[{i} * Q + qi] * s{i}"))
        .collect::<Vec<_>>()
        .join(" + ");

    let phase1 = if regs {
        let ld = srcs
            .iter()
            .enumerate()
            .map(|(i, name)| format!("        sv[{i}][j] = {name}[base + h];"))
            .collect::<Vec<_>>()
            .join("\n");
        let sreg = (0..ns)
            .map(|i| format!("        float s{i} = sv[{i}][j];"))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "    constexpr uint HI = H / TG;\n\
    float sv[NS][HI];\n\
    float qq[Q][HI];\n\
    for (uint j = 0; j < HI; ++j) {{\n\
        uint h = tid + j * TG;\n\
{ld}\n\
        for (uint qi = 0; qi < Q; ++qi) qq[qi][j] = query[qi * H + h];\n\
    }}\n\
    for (uint j = 0; j < HI; ++j) {{\n\
{sreg}\n\
        for (uint qi = 0; qi < Q; ++qi) {{\n\
            float qv = qq[qi][j];\n\
{dots}\n\
        }}\n\
    }}"
        )
    } else {
        format!(
            "    for (uint h = tid; h < H; h += TG) {{\n\
{loads}\n\
        for (uint qi = 0; qi < Q; ++qi) {{\n\
            float qv = query[qi * H + h];\n\
{dots}\n\
        }}\n\
    }}"
        )
    };

    let phase2 = if regs {
        let sreg = (0..ns)
            .map(|i| format!("        float s{i} = sv[{i}][j];"))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "    for (uint j = 0; j < HI; ++j) {{\n\
        uint h = tid + j * TG;\n\
{sreg}\n\
        for (uint qi = 0; qi < Q; ++qi) {{\n\
            out[((size_t)row * Q + qi) * H + h] = {mix};\n\
        }}\n\
    }}"
        )
    } else {
        format!(
            "    for (uint h = tid; h < H; h += TG) {{\n\
{loads}\n\
        for (uint qi = 0; qi < Q; ++qi) {{\n\
            out[((size_t)row * Q + qi) * H + h] = {mix};\n\
        }}\n\
    }}"
        )
    };

    let source = format!(
        "    constexpr uint NS = {ns};\n\
    uint row = threadgroup_position_in_grid.x;\n\
    uint tid = thread_position_in_threadgroup.x;\n\
    uint lane = thread_index_in_simdgroup;\n\
    uint sg = simdgroup_index_in_threadgroup;\n\
    size_t base = (size_t)row * H;\n\
    float acc[NS * Q];\n\
    for (uint i = 0; i < NS * Q; ++i) acc[i] = 0.0f;\n\
{phase1}\n\
    threadgroup float red[NS * Q][TG / 32];\n\
    for (uint i = 0; i < NS * Q; ++i) {{\n\
        float v = simd_sum(acc[i]);\n\
        if (lane == 0) red[i][sg] = v;\n\
    }}\n\
    threadgroup_barrier(mem_flags::mem_threadgroup);\n\
    float w[NS * Q];\n\
    for (uint i = 0; i < NS * Q; ++i) {{\n\
        float t = 0.0f;\n\
        for (uint j = 0; j < TG / 32; ++j) t += red[i][j];\n\
        w[i] = t * scale[0];\n\
    }}\n\
    for (uint qi = 0; qi < Q; ++qi) {{\n\
        float m = -INFINITY;\n\
        for (uint n = 0; n < NS; ++n) m = max(m, w[n * Q + qi]);\n\
        float z = 0.0f;\n\
        for (uint n = 0; n < NS; ++n) {{\n\
            w[n * Q + qi] = metal::exp(w[n * Q + qi] - m);\n\
            z += w[n * Q + qi];\n\
        }}\n\
        for (uint n = 0; n < NS; ++n) w[n * Q + qi] /= z;\n\
    }}\n\
{phase2}\n"
    );
    (srcs, source)
}

fn stock_metallib_has_nax() -> bool {
    static HAS_NAX: OnceLock<bool> = OnceLock::new();
    *HAS_NAX.get_or_init(|| {
        let Ok(dylib) = loaded_library_path() else {
            return true;
        };
        let Some(metallib) = dylib.parent().map(|p| p.join("mlx.metallib")) else {
            return true;
        };
        let Ok(mut file) = std::fs::File::open(metallib) else {
            return true;
        };
        let needle = b"affine_qmm_t_nax";
        let overlap = needle.len() - 1;
        let mut buf = vec![0u8; 4 << 20];
        let mut tail = Vec::with_capacity(overlap);
        loop {
            let Ok(n) = file.read(&mut buf) else {
                return true;
            };
            if n == 0 {
                return false;
            }
            let mut hay = Vec::with_capacity(tail.len() + n);
            hay.extend_from_slice(&tail);
            hay.extend_from_slice(&buf[..n]);
            if hay.windows(needle.len()).any(|w| w == needle) {
                return true;
            }
            tail.clear();
            let keep = hay.len().min(overlap);
            tail.extend_from_slice(&hay[hay.len() - keep..]);
        }
    })
}

/// True when the Metal device and loaded MLX metallib both support M5 NAX kernels.
pub fn is_nax_available(stream: &Stream) -> bool {
    stream
        .metal_caps()
        .is_some_and(|caps| caps.supports_nax() && stock_metallib_has_nax())
}

fn full_query_block(b: usize, hq: usize, length: usize, nk: usize) -> usize {
    let per_q = (b * hq * (nk + 1)).max(1);
    let max_q = (FULL_MASK_CAP / per_q).max(16);
    if max_q >= length {
        return length;
    }
    let n = length.div_ceil(max_q);
    let qb = length.div_ceil(n).div_ceil(FULL_ALIGN) * FULL_ALIGN;
    qb.min(max_q)
}

/// Cached Metal kernels for banded/NAX/verify attention and attention-residual mixing.
pub struct BandedAttnKernels {
    stream: Stream,
    attn_res: RefCell<HashMap<(usize, usize, bool), MetalKernel>>,
    scale_scalars: RefCell<HashMap<u32, Array>>,
    rev_indices: RefCell<HashMap<usize, Array>>,
    flash: OnceCell<MetalKernel>,
    nax: OnceCell<MetalKernel>,
    xattn_p1: OnceCell<MetalKernel>,
    xattn_p1_rows: OnceCell<MetalKernel>,
    xattn_p1_rows_xr: OnceCell<MetalKernel>,
    xattn_p2: OnceCell<MetalKernel>,
    xattn_p2_gate: OnceCell<MetalKernel>,
    xr_ok: Cell<Option<bool>>,
}

impl BandedAttnKernels {
    pub fn new(stream: Stream) -> Self {
        Self {
            stream,
            attn_res: RefCell::new(HashMap::new()),
            scale_scalars: RefCell::new(HashMap::new()),
            rev_indices: RefCell::new(HashMap::new()),
            flash: OnceCell::new(),
            nax: OnceCell::new(),
            xattn_p1: OnceCell::new(),
            xattn_p1_rows: OnceCell::new(),
            xattn_p1_rows_xr: OnceCell::new(),
            xattn_p2: OnceCell::new(),
            xattn_p2_gate: OnceCell::new(),
            xr_ok: Cell::new(None),
        }
    }

    fn cached_scale(&self, scale: f32) -> Result<Array> {
        let key = scale.to_bits();
        let mut cache = self.scale_scalars.borrow_mut();
        if let Some(arr) = cache.get(&key) {
            return Ok(arr.clone());
        }
        let arr = Array::from_f32(&self.stream, &[1], &[scale])?;
        self.stream.eval(std::slice::from_ref(&arr))?;
        cache.insert(key, arr.clone());
        Ok(arr)
    }

    fn cached_rev(&self, n: usize) -> Result<Array> {
        let mut cache = self.rev_indices.borrow_mut();
        if let Some(arr) = cache.get(&n) {
            return Ok(arr.clone());
        }
        let arr = Array::arange(&self.stream, (n as f32) - 1.0, -1.0, -1.0, MlxDType::I32)?;
        self.stream.eval(std::slice::from_ref(&arr))?;
        cache.insert(n, arr.clone());
        Ok(arr)
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Probes once whether this Metal GPU's `simd_sum` matches the xor 1,2,4,8,16
    /// butterfly bit-for-bit (precondition for the TK=8 XR pass-1 verify kernel).
    pub fn xattn_xr_supported(&self) -> Result<bool> {
        if let Some(ok) = self.xr_ok.get() {
            return Ok(ok);
        }
        let n = 32 * 8192usize;
        let probe = MetalKernel::new(
            &self.stream,
            "et_inkling2_xattn_xr_probe",
            &["dummy"],
            &["out"],
            XATTN_XR_PROBE,
            "",
        )?;
        let dummy = Array::zeros(&self.stream, &[1], MlxDType::U32)?;
        let bad = unsafe {
            probe.call(
                &[dummy],
                &[(&[n][..], MlxDType::U32)],
                [n, 1, 1],
                [256, 1, 1],
                &[],
            )?
        }
        .remove(0);
        let total = bad.sum(&[0], false)?.to_u32_scalar()?;
        let ok = total == 0;
        self.xr_ok.set(Some(ok));
        Ok(ok)
    }

    /// Fused attention-residual depth mixing: mixes `blocks` + `partial_n` (`[B, L, H]`)
    /// with `query` (`[Q, H]`) and `scale`, returning `[B, L, Q, H]` F32.
    pub fn attn_res_mix(
        &self,
        blocks: &[Array],
        partial_n: &Array,
        query: &Array,
        scale: f32,
        regs: bool,
    ) -> Result<Array> {
        let shape = partial_n.shape();
        if shape.len() != 3 || query.shape().len() != 2 || query.shape()[1] != shape[2] {
            return Err(Error::Contract(
                "attn_res_mix expects [B, L, H] partial_n and [Q, H] query",
            ));
        }
        let (b, length, hidden) = (shape[0], shape[1], shape[2]);
        let queries = query.shape()[0];
        let rows = b * length;
        let use_regs = regs && length <= ATTN_RES_REGS_MAX_ROWS && hidden % ATTN_RES_TG == 0;

        let key = (blocks.len(), queries, use_regs);
        let mut cache = self.attn_res.borrow_mut();
        if !cache.contains_key(&key) {
            let (srcs, source) = build_attn_res_source(blocks.len(), use_regs);
            let mut input_names: Vec<&str> = srcs.iter().map(String::as_str).collect();
            input_names.push("query");
            input_names.push("scale");
            let name = format!(
                "omlx_inkling2_attn_res_{}{}_{}",
                if use_regs { "regs_" } else { "" },
                blocks.len(),
                queries
            );
            let kernel = MetalKernel::new(
                &self.stream,
                &name,
                &input_names,
                &["out"],
                &source,
                "",
            )?;
            cache.insert(key, kernel);
        }
        let kernel = cache.get(&key).unwrap();
        let mut inputs = Vec::with_capacity(blocks.len() + 3);
        for blk in blocks {
            inputs.push(if blk.dtype() == MlxDType::F32 {
                blk.clone()
            } else {
                blk.cast(MlxDType::F32)?
            });
        }
        inputs.push(if partial_n.dtype() == MlxDType::F32 {
            partial_n.clone()
        } else {
            partial_n.cast(MlxDType::F32)?
        });
        inputs.push(if query.dtype() == MlxDType::F32 {
            query.clone()
        } else {
            query.cast(MlxDType::F32)?
        });
        inputs.push(self.cached_scale(scale)?);

        let out_shape = [b, length, queries, hidden];
        let mut outs = unsafe {
            kernel.call(
                &inputs,
                &[(&out_shape[..], MlxDType::F32)],
                [rows * ATTN_RES_TG, 1, 1],
                [ATTN_RES_TG, 1, 1],
                &[
                    ("H", TemplateArg::Int(hidden as i32)),
                    ("Q", TemplateArg::Int(queries as i32)),
                    ("TG", TemplateArg::Int(ATTN_RES_TG as i32)),
                ],
            )?
        };
        Ok(outs.remove(0))
    }

    /// Single-token decode attention (`B == 1, L == 1`) using MLX fast SDPA with sinks.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_attention(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        rel: &Array,
        sink: &Array,
        scale: f32,
        window: usize,
    ) -> Result<Array> {
        let s = k.shape()[2];
        let lo = if window > 0 { s.saturating_sub(window) } else { 0 };
        let (k_view, v_view) = if lo > 0 {
            (k.slice_axis(2, lo, s)?, v.slice_axis(2, lo, s)?)
        } else {
            (k.clone(), v.clone())
        };
        let sv = s - lo;
        let r = rel.shape()[3];
        let n = r.min(sv);
        let hq = q.shape()[1];
        // rel: [1, 1, Hq, R] -> rel[0, 0, :, :n][:, ::-1]
        let rev = self.cached_rev(n)?;
        let tail = rel
            .slice(&[0, 0, 0, 0], &[1, 1, hq, n])?
            .reshape(&[hq, n])?
            .take(&rev, 1)?;
        let tail = if n < sv {
            let pad = Array::zeros(&self.stream, &[hq, sv - n], tail.dtype())?;
            Array::concat(&[&pad, &tail], 1)?
        } else {
            tail
        };
        let mask = tail.reshape(&[1, hq, 1, sv])?.cast(q.dtype())?;
        let sinks = sink.reshape(&[hq])?.cast(q.dtype())?;
        q.sdpa_with_sinks(&k_view, &v_view, scale, false, Some(&mask), Some(&sinks))
    }

    /// NAX tensor-unit full-attention prefill (`window == 0, D == 128, BF16`).
    pub fn nax_full_attention(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        rel: &Array,
        sink: &Array,
        scale: f32,
    ) -> Result<Array> {
        let (b, hq, length, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
        let (hkv, s) = (k.shape()[1], k.shape()[2]);
        let r = rel.shape()[3];
        if d != 128 {
            return Err(Error::Contract("nax_full_attention requires head_dim == 128"));
        }
        let kernel = self.nax.get_or_init(|| {
            let header = format!("{STEEL_ATTN_NAX_PREFIX}\n{NAX_KERNEL_FN}");
            MetalKernel::new_strided(
                &self.stream,
                "enginetailor_inkling2_full_nax",
                &["q", "k", "v", "rel", "sink", "dims", "scale"],
                &["out"],
                NAX_BODY,
                &header,
            )
            .expect("nax_full_attention kernel creation failed")
        });
        let dims = Array::from_i32(&self.stream, &[2], &[length as i32, s as i32])?;
        let scale_a = Array::from_f32(&self.stream, &[1], &[scale])?;
        let out_shape = [b, length, hq, d];
        let out = unsafe {
            kernel.call(
                &[
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    rel.clone(),
                    sink.clone(),
                    dims,
                    scale_a,
                ],
                &[(&out_shape[..], q.dtype())],
                [length.div_ceil(NAX_BQ) * 128, hq, b],
                [128, 1, 1],
                &[
                    ("T", TemplateArg::DType(q.dtype())),
                    ("BQ", TemplateArg::Int(NAX_BQ as i32)),
                    ("BK", TemplateArg::Int(NAX_BK as i32)),
                    ("HQ", TemplateArg::Int(hq as i32)),
                    ("HKV", TemplateArg::Int(hkv as i32)),
                    ("R", TemplateArg::Int(r as i32)),
                ],
            )?
        }
        .remove(0);
        out.transpose(&[0, 2, 1, 3])
    }

    /// Flash banded attention for BF16 prefill blocks (sliding or non-NAX full).
    #[allow(clippy::too_many_arguments)]
    pub fn flash_banded_attention(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        rel: &Array,
        sink: &Array,
        scale: f32,
        window: usize,
    ) -> Result<Array> {
        let (b, hq, length, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
        let (hkv, s) = (k.shape()[1], k.shape()[2]);
        let r = rel.shape()[3];
        let kernel = self.flash.get_or_init(|| {
            MetalKernel::new_strided(
                &self.stream,
                "omlx_inkling2_flash_banded",
                &["q", "k", "v", "rel", "sink", "dims", "scale"],
                &["out"],
                FLASH_SOURCE,
                "",
            )
            .expect("flash_banded_attention kernel creation failed")
        });
        let mut dims_vec = vec![length as i32, s as i32];
        dims_vec.resize(2 + b, 0);
        let dims = Array::from_i32(&self.stream, &[2 + b], &dims_vec)?;
        let scale_a = Array::from_f32(&self.stream, &[1], &[scale])?;
        let out_shape = [b, length, hq, d];
        let out = unsafe {
            kernel.call(
                &[
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    rel.contiguous()?,
                    sink.cast(MlxDType::F32)?.contiguous()?,
                    dims,
                    scale_a,
                ],
                &[(&out_shape[..], q.dtype())],
                [length.div_ceil(32) * 128, hq, b],
                [128, 1, 1],
                &[
                    ("T", TemplateArg::DType(q.dtype())),
                    ("D", TemplateArg::Int(d as i32)),
                    ("HQ", TemplateArg::Int(hq as i32)),
                    ("HKV", TemplateArg::Int(hkv as i32)),
                    ("R", TemplateArg::Int(r as i32)),
                    ("WINDOW", TemplateArg::Int(window as i32)),
                ],
            )?
        }
        .remove(0);
        out.transpose(&[0, 2, 1, 3])
    }

    /// General additive-mask SDPA fallback for short multi-token prefills (`L < 32`).
    #[allow(clippy::too_many_arguments)]
    pub fn masked_attention(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        rel: &Array,
        sink: &Array,
        scale: f32,
        window: usize,
    ) -> Result<Array> {
        let (b, hq, length, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
        let (hkv, s) = (k.shape()[1], k.shape()[2]);
        let r = rel.shape()[3];
        let lo = if window > 0 {
            s.saturating_sub(length + window - 1)
        } else {
            0
        };
        let zero = Array::zeros(&self.stream, &[b, hkv, 1, d], k.dtype())?;
        let k_tail = if lo > 0 { k.slice_axis(2, lo, s)? } else { k.clone() };
        let v_tail = if lo > 0 { v.slice_axis(2, lo, s)? } else { v.clone() };
        let kz = Array::concat(&[&zero, &k_tail], 2)?;
        let vz = Array::concat(&[&zero, &v_tail], 2)?;
        let nk = s - lo;
        let qb = if window == 0 {
            full_query_block(b, hq, length, nk)
        } else {
            (MASK_BUDGET / (b * hq * (nk + 1)).max(1))
                .min(length)
                .max(16)
        };
        let neg = Array::scalar(&self.stream, f32::NEG_INFINITY, q.dtype())?;
        let zero_s = Array::scalar(&self.stream, 0.0, q.dtype())?;
        let zero_i = Array::from_i32(&self.stream, &[1], &[0])?;
        let r_i = Array::from_i32(&self.stream, &[1], &[r as i32])?;
        let r_m1_i = Array::from_i32(&self.stream, &[1], &[(r - 1) as i32])?;
        let win_i = Array::from_i32(&self.stream, &[1], &[window as i32])?;

        let mut outs = Vec::new();
        let mut i0 = 0;
        while i0 < length {
            let i1 = (i0 + qb).min(length);
            let nq = i1 - i0;
            let a0 = s - length + i0;
            let hi = s - length + i1;
            let f = lo.max(a0.saturating_sub(r - 1));
            let near = hi - f;
            let q_idx = Array::arange(&self.stream, 0.0, nq as f32, 1.0, MlxDType::I32)?
                .reshape(&[nq, 1])?;
            let k_idx = Array::arange(&self.stream, 0.0, near as f32, 1.0, MlxDType::I32)?
                .reshape(&[1, near])?;
            let off_i = Array::from_i32(&self.stream, &[1, 1], &[(a0 - f) as i32])?;
            let dist = off_i.add(&q_idx)?.sub(&k_idx)?;
            let relq = rel.slice_axis(1, i0, i1)?.transpose(&[0, 2, 1, 3])?;
            let gidx = dist
                .clip(&zero_i, &r_m1_i)?
                .reshape(&[1, 1, nq, near])?
                .broadcast_to(&[b, hq, nq, near])?;
            let bias = relq.take_along_axis(&gidx, 3)?;
            let mut vis = dist.greater_equal(&zero_i)?;
            if window > 0 {
                vis = vis.logical_and(&dist.less(&win_i)?)?;
            }
            let in_band = dist.less(&r_i)?.reshape(&[1, 1, nq, near])?;
            let near_vals = in_band.where_select(&bias.cast(q.dtype())?, &zero_s)?;
            let near_mask = vis
                .reshape(&[1, 1, nq, near])?
                .where_select(&near_vals, &neg)?;

            let sink_part = sink
                .slice_axis(1, i0, i1)?
                .transpose(&[0, 2, 1])?
                .reshape(&[b, hq, nq, 1])?
                .cast(q.dtype())?;
            let mut parts: Vec<Array> = vec![sink_part];
            if f > lo {
                let fill = if window > 0 { &neg } else { &zero_s };
                parts.push(fill.broadcast_to(&[b, hq, nq, f - lo])?);
            }
            parts.push(near_mask);
            if hi < s {
                parts.push(neg.broadcast_to(&[b, hq, nq, s - hi])?);
            }
            let refs: Vec<&Array> = parts.iter().collect();
            let mask = Array::concat(&refs, 3)?;
            let q_sub = q.slice_axis(2, i0, i1)?;
            let out = q_sub.sdpa_with_sinks(&kz, &vz, scale, false, Some(&mask), None)?;
            if qb < length {
                out.eval()?;
            }
            outs.push(out);
            i0 = i1;
        }
        if outs.len() == 1 {
            Ok(outs.remove(0))
        } else {
            let refs: Vec<&Array> = outs.iter().collect();
            Array::concat(&refs, 2)
        }
    }

    /// Top-level banded sink attention dispatcher (decode -> NAX full -> flash -> masked).
    #[allow(clippy::too_many_arguments)]
    pub fn banded_sink_attention(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        rel: &Array,
        sink: &Array,
        scale: f32,
        window: usize,
    ) -> Result<Array> {
        let (b, length, d) = (q.shape()[0], q.shape()[2], q.shape()[3]);
        if b == 1 && length == 1 {
            return self.decode_attention(q, k, v, rel, sink, scale, window);
        }
        let nax = is_nax_available(&self.stream);
        if window == 0 && q.dtype() == MlxDType::BF16 && d == 128 && length >= FLASH_MIN_QUERIES && nax {
            return self.nax_full_attention(q, k, v, rel, sink, scale);
        }
        if q.dtype() == MlxDType::BF16 && length >= FLASH_MIN_QUERIES && (window > 0 || !nax) {
            return self.flash_banded_attention(q, k, v, rel, sink, scale, window);
        }
        self.masked_attention(q, k, v, rel, sink, scale, window)
    }

    /// Exact multi-row decode attention for MTP verify, bit-identical to running
    /// `decode_attention` independently at key counts `base + 1 ..= base + L`.
    /// When `gate` (`[1, L, Hq * D]`) is provided, fuses the sigmoid output gate
    /// into pass 2 and returns `[1, L, Hq * D]` directly.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_attention_exact(
        &self,
        q: &Array,
        keys: &Array,
        values: &Array,
        rel: &Array,
        sink: &Array,
        scale: f32,
        window: usize,
        base: usize,
        blocks: usize,
        rows_in_tg: bool,
        gate: Option<&Array>,
        xr: bool,
    ) -> Result<Array> {
        let (_, hq, length, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
        let hkv = keys.shape()[1];
        let r = rel.shape()[3];
        let gqa = hq / hkv;
        let use_rows_in_tg = rows_in_tg && 32 * gqa * length <= 1024;
        let use_xr = xr && use_rows_in_tg && window == 0 && self.xattn_xr_supported()?;

        let p1 = if use_xr {
            self.xattn_p1_rows_xr.get_or_init(|| {
                let src = xattn_pass1_rows_xr_source();
                MetalKernel::new_strided(
                    &self.stream,
                    "et_inkling2_xattn_pass1_rows_xr",
                    &["q", "k", "v", "rel", "sinks", "scale", "dims"],
                    &["partials", "sums", "maxs"],
                    &src,
                    "",
                )
                .expect("xattn p1_rows_xr kernel creation failed")
            })
        } else if use_rows_in_tg {
            self.xattn_p1_rows.get_or_init(|| {
                let src = xattn_pass1_rows_source();
                MetalKernel::new_strided(
                    &self.stream,
                    "et_inkling2_xattn_pass1_rows",
                    &["q", "k", "v", "rel", "sinks", "scale", "dims"],
                    &["partials", "sums", "maxs"],
                    &src,
                    "",
                )
                .expect("xattn p1_rows kernel creation failed")
            })
        } else {
            self.xattn_p1.get_or_init(|| {
                MetalKernel::new_strided(
                    &self.stream,
                    "et_inkling2_xattn_pass1",
                    &["q", "k", "v", "rel", "sinks", "scale", "dims"],
                    &["partials", "sums", "maxs"],
                    XATTN_PASS1,
                    "",
                )
                .expect("xattn p1 kernel creation failed")
            })
        };

        let scale_a = self.cached_scale(scale)?;
        let dims_a = Array::from_i32(&self.stream, &[1], &[base as i32])?;
        let p_shape = [hq * length, blocks, d];
        let sm_shape = [hq * length, blocks];
        let mut p1_outs = unsafe {
            p1.call(
                &[
                    q.contiguous()?,
                    keys.clone(),
                    values.clone(),
                    rel.contiguous()?,
                    sink.contiguous()?,
                    scale_a,
                    dims_a,
                ],
                &[
                    (&p_shape[..], q.dtype()),
                    (&sm_shape[..], MlxDType::F32),
                    (&sm_shape[..], MlxDType::F32),
                ],
                [32 * hkv, gqa * length, blocks],
                [32, if use_rows_in_tg { gqa * length } else { gqa }, 1],
                &[
                    ("T", TemplateArg::DType(q.dtype())),
                    ("D", TemplateArg::Int(d as i32)),
                    ("HQ", TemplateArg::Int(hq as i32)),
                    ("GQA", TemplateArg::Int(gqa as i32)),
                    ("L", TemplateArg::Int(length as i32)),
                    ("R", TemplateArg::Int(r as i32)),
                    ("WINDOW", TemplateArg::Int(window as i32)),
                    ("BLOCKS", TemplateArg::Int(blocks as i32)),
                ],
            )?
        };
        let maxs = p1_outs.pop().unwrap();
        let sums = p1_outs.pop().unwrap();
        let partials = p1_outs.pop().unwrap();

        if let Some(gate) = gate {
            let p2g = self.xattn_p2_gate.get_or_init(|| {
                let src = xattn_pass2_gate_source();
                MetalKernel::new_strided(
                    &self.stream,
                    "et_inkling2_xattn_pass2_gate",
                    &["partials", "sums", "maxs", "gate"],
                    &["out"],
                    &src,
                    "",
                )
                .expect("xattn p2_gate kernel creation failed")
            });
            let out_shape = [1, length, hq * d];
            let mut outs = unsafe {
                p2g.call(
                    &[partials, sums, maxs, gate.clone()],
                    &[(&out_shape[..], gate.dtype())],
                    [hq * length * 1024, 1, 1],
                    [1024, 1, 1],
                    &[
                        ("T", TemplateArg::DType(q.dtype())),
                        ("GT", TemplateArg::DType(gate.dtype())),
                        ("D", TemplateArg::Int(d as i32)),
                        ("HQ", TemplateArg::Int(hq as i32)),
                        ("L", TemplateArg::Int(length as i32)),
                        ("BLOCKS", TemplateArg::Int(blocks as i32)),
                    ],
                )?
            };
            return Ok(outs.remove(0));
        }

        let p2 = self.xattn_p2.get_or_init(|| {
            MetalKernel::new(
                &self.stream,
                "et_inkling2_xattn_pass2",
                &["partials", "sums", "maxs"],
                &["out"],
                XATTN_PASS2,
                "",
            )
            .expect("xattn p2 kernel creation failed")
        });
        let out_shape = [hq * length, d];
        let out = unsafe {
            p2.call(
                &[partials, sums, maxs],
                &[(&out_shape[..], q.dtype())],
                [hq * length * 1024, 1, 1],
                [1024, 1, 1],
                &[
                    ("T", TemplateArg::DType(q.dtype())),
                    ("D", TemplateArg::Int(d as i32)),
                    ("BLOCKS", TemplateArg::Int(blocks as i32)),
                ],
            )?
        }
        .remove(0);
        out.reshape(&[1, hq, length, d])
    }
}
