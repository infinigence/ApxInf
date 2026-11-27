//! Exact decode-bit-equal multi-row GEMVs, permuted loop-controller GEMV_T,
//! wide GEMV, parallel first-index argmax, and certified q8 g64 LM-head screen.
//!
//! Ported from EngineTailor's Inkling-2 MLX kernels (Apache-2.0) and
//! Apple MLX 0.32.4's `gemv.h` (MIT).

use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::Result;
use std::cell::OnceCell;

const LOOPMIX_U: usize = 8;
const LOOPMIX_TN: usize = 1;
const LOOPMIX_PERM_U: usize = 6;
const LOOPMIX_PERM_SGS: usize = 2;
const WIDE_GEMV_SGS: usize = 4;
const WIDE_GEMV_RPS: usize = 1;
const ARGMAX_TG: usize = 256;
const ARGMAX_CH: usize = 2048;
const SCREEN_TM: usize = 2;
const SCREEN_BM: usize = 8;
const SCREEN_CAP: usize = 256;

const MULTIROW_GEMV_SOURCE: &str = r#"
    constexpr int threadsM = BM * SM;
    constexpr int threadsN = BN * SN;
    constexpr int blockM = threadsM * TM;
    constexpr int blockN = threadsN * TN;
    constexpr int tgp_mem_size = BN > 1 ? BN * (blockM + TM) : 1;
    constexpr bool needs_tgp_reduction = BN > 1;
    threadgroup float tgp_memory[R][tgp_mem_size];

    const int in_vec_size = IN;
    const int out_vec_size = OUT;
    const int matrix_ld = IN;
    uint3 tid = threadgroup_position_in_grid;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    thread float result[R][TM];
    for (int r = 0; r < R; r++) for (int tm = 0; tm < TM; tm++) result[r][tm] = 0;
    thread T inter[TN];
    thread float v_coeff[R][TN];

    const int thrM = SN != 32 ? simd_lid / SN : 0;
    const int thrN = SN != 32 ? simd_lid % SN : int(simd_lid);
    const int sgN = BN != 1 ? (simd_gid % BN) : 0;
    const int simdM = BN != 1 ? SM * (simd_gid / BN) : int(SM * simd_gid);
    const int simdN = BN != 1 ? SN * (simd_gid % BN) : 0;
    int bm = (simdM + thrM) * TM;
    int bn = (simdN + thrN) * TN;
    int out_row = tid.x * blockM + bm;
    if (out_row >= out_vec_size) return;
    out_row = out_row + TM <= out_vec_size ? out_row : out_vec_size - TM;
    const device T* matp = mat + size_t(out_row) * matrix_ld;

    const int n_iter = in_vec_size / blockN;
    const int last_iter = blockN * n_iter;
    const int leftover = in_vec_size - last_iter;

    for (int i = 0; i < n_iter; ++i) {
      for (int r = 0; r < R; r++) {
        for (int tn = 0; tn < TN; tn++) {
          v_coeff[r][tn] = static_cast<float>(vec[r * in_vec_size + bn + tn]);
        }
      }
      int mat_offset = 0;
      for (int tm = 0; tm < TM; tm++) {
        for (int tn = 0; tn < TN; tn++) {
          inter[tn] = matp[mat_offset + bn + tn];
        }
        for (int r = 0; r < R; r++) {
          for (int tn = 0; tn < TN; tn++) {
            result[r][tm] += inter[tn] * v_coeff[r][tn];
          }
        }
        mat_offset += matrix_ld;
      }
      bn += blockN;
    }
    if (leftover > 0) {
      for (int r = 0; r < R; r++) {
        for (int tn = 0; tn < TN; tn++) {
          v_coeff[r][tn] = bn + tn < in_vec_size
              ? static_cast<float>(vec[r * in_vec_size + bn + tn]) : 0.0f;
        }
      }
      for (int tm = 0; tm < TM; tm++) {
        for (int tn = 0; tn < TN; tn++) {
          inter[tn] = bn + tn < in_vec_size ? matp[tm * matrix_ld + bn + tn] : T(0);
        }
        for (int r = 0; r < R; r++) {
          for (int tn = 0; tn < TN; tn++) {
            result[r][tm] += inter[tn] * v_coeff[r][tn];
          }
        }
      }
    }
    for (int r = 0; r < R; r++) {
      for (int tm = 0; tm < TM; tm++) {
        for (ushort sn = (SN / 2); sn >= 1; sn >>= 1) {
          result[r][tm] += simd_shuffle_down(result[r][tm], sn);
        }
      }
    }
    if (needs_tgp_reduction) {
      if (thrN == 0) {
        for (int r = 0; r < R; r++) {
          for (int tm = 0; tm < TM; tm++) {
            tgp_memory[r][sgN * (blockM + TM) + bm + tm] = result[r][tm];
          }
        }
        threadgroup_barrier(mem_flags::mem_none);
        if (sgN == 0) {
          for (int sgn = 1; sgn < BN; sgn++) {
            for (int r = 0; r < R; r++) {
              for (int tm = 0; tm < TM; tm++) {
                result[r][tm] += tgp_memory[r][sgn * (blockM + TM) + bm + tm];
              }
            }
          }
        }
      }
    }
    if (simdN == 0 && thrN == 0) {
      for (int r = 0; r < R; r++) {
        for (int tm = 0; tm < TM; tm++) {
          out[r * out_vec_size + out_row + tm] = static_cast<T>(result[r][tm]);
        }
      }
    }
"#;

const LOOPMIX_GEMV_T_SOURCE: &str = r#"
    constexpr int SM = 8;
    constexpr int SN = 4;
    constexpr int TM = 4;
    constexpr int blockM = SM * TM;
    constexpr int n_iter = K / blockM;
    const uint simd_lid = thread_index_in_simdgroup;
    const int thrM = int(simd_lid) / SN;
    const int thrN = int(simd_lid) % SN;
    const int sg = int(threadgroup_position_in_grid.x) * SGS + int(simdgroup_index_in_threadgroup);
    const int row = int(threadgroup_position_in_grid.y);
    const int b = row / R;
    const int out_col = sg * (SN * TN) + thrN * TN;
    const device bfloat* matp = mat + size_t(b) * K * N + out_col;
    const device float* vecp = vec + size_t(row) * K;
    device float* outp = out + size_t(row) * N + out_col;

    thread float result[TN];
    for (int tn = 0; tn < TN; tn++) {
      result[tn] = 0;
    }
    int bm = thrM * TM;
    for (int i0 = 0; i0 < n_iter; i0 += U) {
      thread bfloat w[U][TM][TN];
      thread float vc[U][TM];
      for (int u = 0; u < U; u++) {
        for (int tm = 0; tm < TM; tm++) {
          const device bfloat* src = matp + size_t(bm + u * blockM + tm) * N;
          for (int tn = 0; tn < TN; tn++) {
            w[u][tm][tn] = src[tn];
          }
          vc[u][tm] = vecp[bm + u * blockM + tm];
        }
      }
      for (int u = 0; u < U; u++) {
        for (int tm = 0; tm < TM; tm++) {
          thread float inter[TN];
          for (int tn = 0; tn < TN; tn++) {
            inter[tn] = static_cast<float>(w[u][tm][tn]);
          }
          for (int tn = 0; tn < TN; tn++) {
            result[tn] += vc[u][tm] * inter[tn];
          }
        }
      }
      bm += U * blockM;
    }
    for (int tn = 0; tn < TN; tn++) {
      for (ushort sm = (SM / 2); sm >= 1; sm >>= 1) {
        result[tn] += simd_shuffle_down(result[tn], SN * sm);
      }
    }
    if (thrM == 0) {
      for (int tn = 0; tn < TN; tn++) {
        outp[tn] = result[tn];
      }
    }
"#;

const LOOPMIX_PAIR_HEAD: &str = r#"
    constexpr int SM = 8;
    constexpr int SN = 4;
    constexpr int TM = 4;
    constexpr int blockM = SM * TM;
    constexpr int n_iter = K / blockM;
    threadgroup float partner[SGS * SN * TN];
    const uint simd_lid = thread_index_in_simdgroup;
    const int thrM = int(simd_lid) / SN;
    const int thrN = int(simd_lid) % SN;
    const int sgi = int(simdgroup_index_in_threadgroup);
    const int half_ = sgi / SGS;
    const int lsg = sgi - half_ * SGS;
    const int sg = int(threadgroup_position_in_grid.x) * SGS + lsg;
    const int prow = int(threadgroup_position_in_grid.y);
    const int lcol = lsg * (SN * TN) + thrN * TN;
"#;

const LOOPMIX_PAIR_SWAP: &str = r#"
    if (half_ == 1 && thrM == 0) {
      for (int tn = 0; tn < TN; tn++) {
        partner[lcol + tn] = result[tn];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"#;

const LOOPMIX_ACC_PERM: &str = r#"
    thread float result[TN];
    for (int tn = 0; tn < TN; tn++) {
      result[tn] = 0;
    }
    constexpr int NJ = n_iter / 2;
    for (int j0 = 0; j0 < NJ; j0 += U) {
      thread uint4 wv[U];
      thread float vc[U][8];
      for (int u = 0; u < U; u++) {
        wv[u] = pp[(j0 + u) * 32];
        for (int v = 0; v < 8; v++) {
          vc[u][v] = vecp[thrM * TM + (2 * (j0 + u) + v / 4) * blockM + (v % 4)];
        }
      }
      for (int u = 0; u < U; u++) {
        for (int v = 0; v < 8; v++) {
          const uint wd = wv[u][v / 2];
          thread float inter[TN];
          inter[0] = (v % 2 == 0) ? as_type<float>(wd << 16) : as_type<float>(wd & 0xFFFF0000u);
          for (int tn = 0; tn < TN; tn++) {
            result[tn] += vc[u][v] * inter[tn];
          }
        }
      }
    }
    for (int tn = 0; tn < TN; tn++) {
      for (ushort sm = (SM / 2); sm >= 1; sm >>= 1) {
        result[tn] += simd_shuffle_down(result[tn], SN * sm);
      }
    }
"#;

const WIDE_GEMV_SOURCE: &str = r#"
    constexpr int KL = 32;
    constexpr int UNR = 8;
    constexpr int n_v4 = K / 4;
    constexpr int n_main = n_v4 - n_v4 % (KL * UNR);
    const int k_lane = int(thread_index_in_simdgroup);
    const int row0 = (int(threadgroup_position_in_grid.x) * SGS
                      + int(simdgroup_index_in_threadgroup)) * RPS;
    const device vec<T, 4>* x4 = (const device vec<T, 4>*)xv;
    const device vec<T, 4>* w4[RPS];
    for (int r = 0; r < RPS; r++) {
      const int row = min(row0 + r, N - 1);
      w4[r] = (const device vec<T, 4>*)(mat + size_t(row) * K);
    }
    float result[RPS][V];
    for (int r = 0; r < RPS; r++) {
      for (int v = 0; v < V; v++) {
        result[r][v] = 0;
      }
    }
    for (int base = 0; base < n_main; base += KL * UNR) {
      vec<float, 4> wf[RPS][UNR];
      for (int r = 0; r < RPS; r++) {
        for (int i = 0; i < UNR; i++) {
          wf[r][i] = vec<float, 4>(w4[r][base + i * KL + k_lane]);
        }
      }
      for (int v = 0; v < V; v++) {
        float acc[RPS];
        for (int r = 0; r < RPS; r++) {
          acc[r] = 0;
        }
        for (int i = 0; i < UNR; i++) {
          const vec<float, 4> xf = vec<float, 4>(x4[v * n_v4 + base + i * KL + k_lane]);
          for (int r = 0; r < RPS; r++) {
            acc[r] += dot(wf[r][i], xf);
          }
        }
        for (int r = 0; r < RPS; r++) {
          result[r][v] += acc[r];
        }
      }
    }
    for (int idx = n_main + k_lane; idx < n_v4; idx += KL) {
      vec<float, 4> wf[RPS];
      for (int r = 0; r < RPS; r++) {
        wf[r] = vec<float, 4>(w4[r][idx]);
      }
      for (int v = 0; v < V; v++) {
        const vec<float, 4> xf = vec<float, 4>(x4[v * n_v4 + idx]);
        for (int r = 0; r < RPS; r++) {
          result[r][v] += dot(wf[r], xf);
        }
      }
    }
    for (int r = 0; r < RPS; r++) {
      for (int v = 0; v < V; v++) {
        for (ushort off = KL / 2; off >= 1; off >>= 1) {
          result[r][v] += simd_shuffle_down(result[r][v], off);
        }
      }
    }
    if (k_lane == 0) {
      for (int r = 0; r < RPS; r++) {
        if (row0 + r < N) {
          for (int v = 0; v < V; v++) {
            out[size_t(v) * N + row0 + r] = static_cast<T>(result[r][v]);
          }
        }
      }
    }
"#;

const ARGMAX_BETTER: &str = r#"
static inline bool et_argmax_better(float v, uint i, float bv, uint bi) {
  if (metal::isnan(bv)) return metal::isnan(v) && i < bi;
  if (metal::isnan(v)) return true;
  return v > bv || (v == bv && i < bi);
}
"#;

const ARGMAX_COMBINE: &str = r#"
    for (ushort off = 16; off >= 1; off >>= 1) {
      float ov = simd_shuffle_down(bv, off);
      uint oi = simd_shuffle_down(bi, off);
      if (et_argmax_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
    }
"#;

const Q8_SCREEN_SOURCE: &str = r#"
    constexpr int G = H / 64;
    constexpr int NIT = H / 256;
    const uint tg = threadgroup_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int base = (int(tg) * BM + int(sg)) * TM;
    threadgroup float tmax[BM][R];
    float acc[R][TM];
    float bnd[R][TM];
    float l1[R];
    for (int j = 0; j < R; j++) { l1[j] = 0; for (int m = 0; m < TM; m++) { acc[j][m] = 0; bnd[j][m] = 0; } }
    for (int it = 0; it < NIT; it++) {
      const int k0 = it * 256 + int(lane) * 8;
      const int g = k0 / 64;
      float xv[R][8];
      float xs[R];
      float xa[R];
      for (int j = 0; j < R; j++) {
        xs[j] = 0; xa[j] = 0;
        for (int t = 0; t < 8; t++) {
          float v = static_cast<float>(x[j * H + k0 + t]);
          xv[j][t] = v; xs[j] += v; xa[j] += metal::abs(v);
        }
        l1[j] += xa[j];
      }
      for (int m = 0; m < TM; m++) {
        const int row = min(base + m, V - 1);
        const uint2 q = *((const device uint2*)(wq + size_t(row) * (H / 4) + k0 / 4));
        const float s = static_cast<float>(sc[row * G + g]);
        const float b = static_cast<float>(bi[row * G + g]);
        const float e = static_cast<float>(err[row * G + g]);
        float qf[8];
        for (int t = 0; t < 4; t++) { qf[t] = float((q.x >> (8 * t)) & 0xffu); qf[4 + t] = float((q.y >> (8 * t)) & 0xffu); }
        for (int j = 0; j < R; j++) {
          float d = 0;
          for (int t = 0; t < 8; t++) d += qf[t] * xv[j][t];
          acc[j][m] += s * d + b * xs[j];
          bnd[j][m] += e * xa[j];
        }
      }
    }
    float lmax[R];
    for (int j = 0; j < R; j++) {
      const float tl1 = simd_sum(l1[j]);
      lmax[j] = -INFINITY;
      for (int m = 0; m < TM; m++) {
        const float a = simd_sum(acc[j][m]);
        const float B = simd_sum(bnd[j][m]) * (1.0f + 0x1p-14f) + ca[0] * tl1;
        const int row = base + m;
        const bool ok = row < VU;
        const float up = ok ? a + B : -INFINITY;
        const float lo = ok ? a - B : -INFINITY;
        if (lane == 0 && row < V) { upper[j * V + row] = up; }
        lmax[j] = metal::max(lmax[j], lo);
      }
    }
    if (lane == 0) { for (int j = 0; j < R; j++) tmax[sg][j] = lmax[j]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane < R) {
      float mx_ = -INFINITY;
      for (int s2 = 0; s2 < BM; s2++) mx_ = metal::max(mx_, tmax[s2][lane]);
      tgmax[lane * NTG + tg] = mx_;
    }
"#;

const SCREEN_LSTAR_SOURCE: &str = r#"
    const uint j = threadgroup_position_in_grid.y;
    const uint t = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup float sv[32];
    float m = -INFINITY;
    for (uint i = t; i < NTG; i += TG) m = metal::max(m, tgmax[j * NTG + i]);
    m = simd_max(m);
    if (lane == 0) sv[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
      m = lane < TG / 32 ? sv[lane] : -INFINITY;
      m = simd_max(m);
      if (lane == 0) {
        float th = NAN;
        if (metal::isfinite(m)) {
          uint u = as_type<uint>(m);
          uint hi = u >> 16;
          if ((u & 0xffffu) != 0u && (u >> 31) != 0u) hi += 1u;
          th = ptab[hi & 0xffffu];
        }
        thr[j] = th;
      }
    }
"#;

const SCREEN_SELECT_SOURCE: &str = r#"
    const uint r = thread_position_in_grid.x;
    if (r >= V) return;
    bool keep = false;
    for (int j = 0; j < R; j++) keep = keep || (upper[j * V + r] > thr[j]);
    if (keep) {
      uint slot = atomic_fetch_add_explicit(&count[0], 1u, memory_order_relaxed);
      if (slot < CAP) atomic_store_explicit(&cand[slot], r, memory_order_relaxed);
    }
"#;

fn gemv_params(n_out: usize, k: usize) -> (usize, usize, usize, usize, usize, usize) {
    let (mut tm, tn, mut sm, mut sn, mut bn) = (4usize, 4usize, 1usize, 32usize, 1usize);
    let mut bm = if n_out >= 4096 { 8 } else { 4 };
    if k <= 64 {
        bm = 1;
        sm = 8;
        sn = 4;
    } else if k >= 16 * n_out {
        bm = 1;
        bn = 8;
    }
    if n_out < tm {
        tm = 1;
    }
    (bm, bn, sm, sn, tm, tn)
}

fn loopmix_acc_slice() -> &'static str {
    let start = LOOPMIX_GEMV_T_SOURCE.find("    thread float result[TN];").unwrap();
    let end = LOOPMIX_GEMV_T_SOURCE.find("    if (thrM == 0) {").unwrap();
    &LOOPMIX_GEMV_T_SOURCE[start..end]
}

fn loopmix_down_pair_source() -> String {
    format!(
        "{LOOPMIX_PAIR_HEAD}\
    const int p = prow / R;\n\
    const int r_ = prow - p * R;\n\
    const int b = 2 * p + half_;\n\
    const int out_col = sg * (SN * TN) + thrN * TN;\n\
    const device bfloat* matp = mat + size_t(b) * K * N + out_col;\n\
    const device float* vecp = vec + size_t(b * R + r_) * K;\n\
{}{LOOPMIX_PAIR_SWAP}\
    if (half_ == 0 && thrM == 0) {{\n\
      device float* outp = out + size_t(prow) * N + out_col;\n\
      for (int tn = 0; tn < TN; tn++) {{\n\
        float s = 0.0f;\n\
        s = s + result[tn];\n\
        s = s + partner[lcol + tn];\n\
        outp[tn] = static_cast<float>(static_cast<bfloat>(s));\n\
      }}\n\
    }}\n",
        loopmix_acc_slice()
    )
}

fn loopmix_down_perm_source() -> String {
    let base = loopmix_down_pair_source();
    let old_matp = "const device bfloat* matp = mat + size_t(b) * K * N + out_col;\n";
    let new_matp = "    const device uint4* pp = (const device uint4*)(mat) + (size_t(b) * (N / 4) + sg) * (K / 64) * 32 + simd_lid;\n";
    debug_assert!(base.contains(old_matp));
    base.replace(old_matp, new_matp)
        .replace(loopmix_acc_slice(), LOOPMIX_ACC_PERM)
}

fn loopmix_gates_mix_source() -> String {
    format!(
        "{LOOPMIX_PAIR_HEAD}\
    const int b = prow / R;\n\
    const int jcol = sg * (SN * TN) + thrN * TN;\n\
    const int out_col = half_ * HH + jcol;\n\
    const device bfloat* matp = mat + size_t(b) * K * N + out_col;\n\
    const device float* vecp = vec + size_t(prow) * K;\n\
{}{LOOPMIX_PAIR_SWAP}\
    if (half_ == 0 && thrM == 0) {{\n\
#pragma clang fp contract(off)\n\
      const int L = int(x_shape[2]);\n\
      const int ri = prow - b * R;\n\
      const int bi = ri / L;\n\
      const int li = ri - bi * L;\n\
      for (int tn = 0; tn < TN; tn++) {{\n\
        const int ji = jcol + tn;\n\
        const float gain = metal::precise::tanh(result[tn]);\n\
        const float forget = metal::precise::tanh(partner[lcol + tn]);\n\
        const float xv = x[b * int(x_strides[0]) + bi * int(x_strides[1]) + li * int(x_strides[2]) + ji * int(x_strides[3])];\n\
        const float mv = msg[b * int(msg_strides[0]) + bi * int(msg_strides[1]) + li * int(msg_strides[2]) + ji * int(msg_strides[3])];\n\
        const float a = 1.0f - forget;\n\
        const float bx = a * xv;\n\
        const float c = gain * mv;\n\
        out[size_t(prow) * HH + ji] = bx + c;\n\
      }}\n\
    }}\n",
        loopmix_acc_slice()
    )
}

fn loopmix_launch(k: usize, n_cols: usize) -> (usize, usize, usize, usize) {
    let mut u = LOOPMIX_U;
    while (k / 32) % u != 0 {
        u /= 2;
    }
    let tn = LOOPMIX_TN;
    let sgs_total = n_cols / (4 * tn);
    let sgs = if sgs_total % 4 == 0 {
        4
    } else if sgs_total % 2 == 0 {
        2
    } else {
        1
    };
    (u, tn, sgs, sgs_total)
}

fn gather_gemv_source() -> String {
    MULTIROW_GEMV_SOURCE
        .replace(
            "    if (out_row >= out_vec_size) return;\n    out_row = out_row + TM <= out_vec_size ? out_row : out_vec_size - TM;\n    const device T* matp = mat + size_t(out_row) * matrix_ld;\n",
            "    const uint n_cand = min(count[0], uint(out_vec_size));\n    if (out_row >= int(n_cand)) return;\n    const device T* rowp[TM];\n    for (int tm = 0; tm < TM; tm++) {\n      const int slot = min(out_row + tm, int(n_cand) - 1);\n      rowp[tm] = mat + size_t(cand[slot]) * matrix_ld;\n    }\n",
        )
        .replace(
            "      int mat_offset = 0;\n      for (int tm = 0; tm < TM; tm++) {\n        for (int tn = 0; tn < TN; tn++) {\n          inter[tn] = matp[mat_offset + bn + tn];\n        }",
            "      for (int tm = 0; tm < TM; tm++) {\n        for (int tn = 0; tn < TN; tn++) {\n          inter[tn] = rowp[tm][bn + tn];\n        }",
        )
        .replace(
            "        mat_offset += matrix_ld;\n      }\n      bn += blockN;",
            "      }\n      bn += blockN;",
        )
        .replace(
            "          inter[tn] = bn + tn < in_vec_size ? matp[tm * matrix_ld + bn + tn] : T(0);",
            "          inter[tn] = bn + tn < in_vec_size ? rowp[tm][bn + tn] : T(0);",
        )
}

fn argmax_p1_source(value_expr: &str) -> String {
    format!(
        r#"
    const uint c = threadgroup_position_in_grid.x;
    const uint r = threadgroup_position_in_grid.y;
    const uint t = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup float sv[32];
    threadgroup uint si[32];
    float bv = -INFINITY;
    uint bi = 0;
    const uint base = c * CH;
    for (uint k = 0; k < CH / TG; k++) {{
      const uint i = base + k * TG + t;
      if (i < N) {{
        float v = {value_expr};
        if (et_argmax_better(v, i, bv, bi)) {{ bv = v; bi = i; }}
      }}
    }}
{ARGMAX_COMBINE}
    if (lane == 0) {{ sv[sg] = bv; si[sg] = bi; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {{
      bv = lane < TG / 32 ? sv[lane] : -INFINITY;
      bi = lane < TG / 32 ? si[lane] : 0;
{ARGMAX_COMBINE}
      if (lane == 0) {{ pv[r * NC + c] = bv; pi[r * NC + c] = bi; }}
    }}
"#
    )
}

fn argmax_p2_source() -> String {
    format!(
        r#"
    const uint r = threadgroup_position_in_grid.y;
    const uint t = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup float sv[32];
    threadgroup uint si[32];
    float bv = -INFINITY;
    uint bi = 0;
    for (uint c = t; c < NC; c += TG) {{
      float v = pv[r * NC + c];
      uint i = pi[r * NC + c];
      if (et_argmax_better(v, i, bv, bi)) {{ bv = v; bi = i; }}
    }}
{ARGMAX_COMBINE}
    if (lane == 0) {{ sv[sg] = bv; si[sg] = bi; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {{
      bv = lane < TG / 32 ? sv[lane] : -INFINITY;
      bi = lane < TG / 32 ? si[lane] : 0;
{ARGMAX_COMBINE}
      if (lane == 0) {{ out[r] = bi; }}
    }}
"#
    )
}

fn screen_final_source() -> String {
    format!(
        r#"
    const uint j = threadgroup_position_in_grid.y;
    const uint t = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    threadgroup float sv[32];
    threadgroup uint si[32];
    const uint n = count[0];
    bool cert = n >= 1u && n <= CAP;
    for (int jj = 0; jj < R; jj++) cert = cert && !metal::isnan(thr[jj]);
    float bv = -INFINITY;
    uint bi = 0xffffffffu;
    if (cert) {{
      for (uint s = t; s < n; s += TG) {{
        const uint i = cand[s];
        const float v = ttab[as_type<ushort>(rech[j * CAP + s])] + bias[i];
        if (et_argmax_better(v, i, bv, bi)) {{ bv = v; bi = i; }}
      }}
    }}
{ARGMAX_COMBINE}
    if (lane == 0) {{ sv[sg] = bv; si[sg] = bi; }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {{
      bv = lane < TG / 32 ? sv[lane] : -INFINITY;
      bi = lane < TG / 32 ? si[lane] : 0xffffffffu;
{ARGMAX_COMBINE}
      if (lane == 0) {{ out[j] = (cert && bi != 0xffffffffu) ? int(bi) : -1; }}
    }}
"#
    )
}

/// Precomputed certified q8 g64 screen tables for a BF16 LM head.
pub struct VerifyHeadScreen {
    pub wq: Array,
    pub sc: Array,
    pub bi: Array,
    pub err: Array,
    pub ca: Array,
    pub ttab: Array,
    pub ptab: Array,
    pub bias: Array,
    pub unpadded: usize,
}

/// Cached Metal kernels for exact multi-row GEMV, loop-mix GEMV_T, argmax, and LM-head screen.
pub struct ExactGemvKernels {
    stream: Stream,
    multirow: OnceCell<MetalKernel>,
    multirow_w16: OnceCell<MetalKernel>,
    wide: OnceCell<MetalKernel>,
    loopmix_gemv_t: OnceCell<MetalKernel>,
    loopmix_down: OnceCell<MetalKernel>,
    loopmix_down_perm: OnceCell<MetalKernel>,
    loopmix_gates: OnceCell<MetalKernel>,
    argmax_p1_table: OnceCell<MetalKernel>,
    argmax_p1_plain: OnceCell<MetalKernel>,
    argmax_p2: OnceCell<MetalKernel>,
    screen_q8: OnceCell<MetalKernel>,
    screen_lstar: OnceCell<MetalKernel>,
    screen_select: OnceCell<MetalKernel>,
    screen_gather: OnceCell<MetalKernel>,
    screen_final: OnceCell<MetalKernel>,
}

impl ExactGemvKernels {
    pub fn new(stream: Stream) -> Self {
        Self {
            stream,
            multirow: OnceCell::new(),
            multirow_w16: OnceCell::new(),
            wide: OnceCell::new(),
            loopmix_gemv_t: OnceCell::new(),
            loopmix_down: OnceCell::new(),
            loopmix_down_perm: OnceCell::new(),
            loopmix_gates: OnceCell::new(),
            argmax_p1_table: OnceCell::new(),
            argmax_p1_plain: OnceCell::new(),
            argmax_p2: OnceCell::new(),
            screen_q8: OnceCell::new(),
            screen_lstar: OnceCell::new(),
            screen_select: OnceCell::new(),
            screen_gather: OnceCell::new(),
            screen_final: OnceCell::new(),
        }
    }

    /// `x @ w.T` for `x` `[R, K]` and `w` `[N, K]`, bit-equal row-by-row to 1-row MLX GEMV.
    pub fn multirow_gemv(&self, w: &Array, x: &Array, widen: bool) -> Result<Array> {
        let (n_out, k) = (w.shape()[0], w.shape()[1]);
        let rows = x.shape()[0];
        let (bm, bn, sm, sn, tm, tn) = gemv_params(n_out, k);
        let per_tgp = bm * sm * tm;
        let threads = 32 * bn * bm;
        let n_tgp = n_out.div_ceil(per_tgp);
        let xin = if widen && x.dtype() != w.dtype() && w.dtype() == MlxDType::F32 {
            x.clone()
        } else if x.dtype() == w.dtype() {
            x.clone()
        } else {
            x.cast(w.dtype())?
        };
        let kernel = self.multirow.get_or_init(|| {
            MetalKernel::new(
                &self.stream,
                "et_inkling2_multirow_gemv",
                &["mat", "vec"],
                &["out"],
                MULTIROW_GEMV_SOURCE,
                "",
            )
            .expect("multirow_gemv kernel creation failed")
        });
        let out_shape = [rows, n_out];
        let mut outs = unsafe {
            kernel.call(
                &[w.clone(), xin],
                &[(&out_shape[..], w.dtype())],
                [n_tgp * threads, 1, 1],
                [threads, 1, 1],
                &[
                    ("T", TemplateArg::DType(w.dtype())),
                    ("R", TemplateArg::Int(rows as i32)),
                    ("IN", TemplateArg::Int(k as i32)),
                    ("OUT", TemplateArg::Int(n_out as i32)),
                    ("BM", TemplateArg::Int(bm as i32)),
                    ("BN", TemplateArg::Int(bn as i32)),
                    ("SM", TemplateArg::Int(sm as i32)),
                    ("SN", TemplateArg::Int(sn as i32)),
                    ("TM", TemplateArg::Int(tm as i32)),
                    ("TN", TemplateArg::Int(tn as i32)),
                ],
            )?
        };
        Ok(outs.remove(0))
    }

    /// F32 router `multirow_gemv` reading from its exact BF16 weight copy `w16` with `TM=1`.
    pub fn multirow_gemv_router16(&self, w16: &Array, x: &Array) -> Result<Array> {
        let (n_out, k) = (w16.shape()[0], w16.shape()[1]);
        let rows = x.shape()[0];
        let (bm, bn, sm, sn, _, tn) = gemv_params(n_out, k);
        let tm = 1usize;
        let per_tgp = bm * sm * tm;
        let threads = 32 * bn * bm;
        let n_tgp = n_out.div_ceil(per_tgp);
        let kernel = self.multirow_w16.get_or_init(|| {
            let src = MULTIROW_GEMV_SOURCE.replace(
                "    const device T* matp = mat + size_t(out_row) * matrix_ld;\n",
                "    const device bfloat* matp = mat + size_t(out_row) * matrix_ld;\n",
            );
            MetalKernel::new(
                &self.stream,
                "et_inkling2_multirow_gemv_w16",
                &["mat", "vec"],
                &["out"],
                &src,
                "",
            )
            .expect("multirow_gemv_w16 kernel creation failed")
        });
        let out_shape = [rows, n_out];
        let mut outs = unsafe {
            kernel.call(
                &[w16.clone(), x.clone()],
                &[(&out_shape[..], MlxDType::F32)],
                [n_tgp * threads, 1, 1],
                [threads, 1, 1],
                &[
                    ("T", TemplateArg::DType(MlxDType::F32)),
                    ("R", TemplateArg::Int(rows as i32)),
                    ("IN", TemplateArg::Int(k as i32)),
                    ("OUT", TemplateArg::Int(n_out as i32)),
                    ("BM", TemplateArg::Int(bm as i32)),
                    ("BN", TemplateArg::Int(bn as i32)),
                    ("SM", TemplateArg::Int(sm as i32)),
                    ("SN", TemplateArg::Int(sn as i32)),
                    ("TM", TemplateArg::Int(tm as i32)),
                    ("TN", TemplateArg::Int(tn as i32)),
                ],
            )?
        };
        Ok(outs.remove(0))
    }

    /// Exact `gemv_wide` for looped-pass BF16 `in_proj` (`w: [N, K]`, `x: [V, K]`).
    pub fn wide_gemv(&self, w: &Array, x: &Array) -> Result<Array> {
        let (n, k) = (w.shape()[0], w.shape()[1]);
        let v = x.shape()[0];
        let sgs = WIDE_GEMV_SGS;
        let rps = WIDE_GEMV_RPS;
        let per_tg = sgs * rps;
        let n_tg = n.div_ceil(per_tg);
        let kernel = self.wide.get_or_init(|| {
            MetalKernel::new(
                &self.stream,
                "et_inkling2_wide_gemv",
                &["mat", "xv"],
                &["out"],
                WIDE_GEMV_SOURCE,
                "",
            )
            .expect("wide_gemv kernel creation failed")
        });
        let out_shape = [v, n];
        let mut outs = unsafe {
            kernel.call(
                &[w.clone(), x.clone()],
                &[(&out_shape[..], w.dtype())],
                [n_tg * 32 * sgs, 1, 1],
                [32 * sgs, 1, 1],
                &[
                    ("T", TemplateArg::DType(w.dtype())),
                    ("V", TemplateArg::Int(v as i32)),
                    ("K", TemplateArg::Int(k as i32)),
                    ("N", TemplateArg::Int(n as i32)),
                    ("SGS", TemplateArg::Int(sgs as i32)),
                    ("RPS", TemplateArg::Int(rps as i32)),
                ],
            )?
        };
        Ok(outs.remove(0))
    }

    /// Permutes `[B, K, N]` BF16 weight into `[B, N/4, K/64, 32, 4]` U32 for `loopmix_down_pair_perm`.
    pub fn loopmix_perm_weight(w16: &Array) -> Result<Array> {
        let (b, k, n) = (w16.shape()[0], w16.shape()[1], w16.shape()[2]);
        let ni = k / 32;
        w16.reshape(&[b, ni / 2, 2, 8, 4, n / 4, 4])?
            .transpose(&[0, 5, 1, 3, 6, 2, 4])?
            .contiguous()?
            .reshape(&[b, n / 4, ni / 2, 32, 8])?
            .view(MlxDType::U32)
    }

    pub fn loopmix_perm_supported(w16: &Array) -> bool {
        if w16.shape().len() != 3 {
            return false;
        }
        let (k, n_cols) = (w16.shape()[1], w16.shape()[2]);
        k % 64 == 0
            && (k / 64) % LOOPMIX_PERM_U == 0
            && n_cols % 4 == 0
            && (n_cols / 4) % LOOPMIX_PERM_SGS == 0
    }

    /// Batched `x @ w16` for `[B, R, K]` F32 and `[B, K, N]` BF16.
    pub fn loopmix_gemv_t(&self, x: &Array, w16: &Array) -> Result<Array> {
        let (b, rows, k) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let n = w16.shape()[2];
        let (u, tn, sgs, sgs_total) = loopmix_launch(k, n);
        let kernel = self.loopmix_gemv_t.get_or_init(|| {
            MetalKernel::new(
                &self.stream,
                "et_inkling2_loopmix_gemv_t",
                &["mat", "vec"],
                &["out"],
                LOOPMIX_GEMV_T_SOURCE,
                "",
            )
            .expect("loopmix_gemv_t kernel creation failed")
        });
        let out_shape = [b, rows, n];
        let mut outs = unsafe {
            kernel.call(
                &[w16.clone(), x.clone()],
                &[(&out_shape[..], MlxDType::F32)],
                [sgs_total * 32, b * rows, 1],
                [32 * sgs, 1, 1],
                &[
                    ("R", TemplateArg::Int(rows as i32)),
                    ("K", TemplateArg::Int(k as i32)),
                    ("N", TemplateArg::Int(n as i32)),
                    ("U", TemplateArg::Int(u as i32)),
                    ("TN", TemplateArg::Int(tn as i32)),
                    ("SGS", TemplateArg::Int(sgs as i32)),
                ],
            )?
        };
        Ok(outs.remove(0))
    }

    /// Fused `loopmix_down_pair` using pre-permuted `perm = loopmix_perm_weight(w16)`.
    pub fn loopmix_down_pair_perm(&self, z: &Array, w16: &Array, perm: &Array) -> Result<Array> {
        let (b2, rows, k) = (z.shape()[0], z.shape()[1], z.shape()[2]);
        let n_cols = w16.shape()[2];
        let sgs_total = n_cols / 4;
        let sgs = LOOPMIX_PERM_SGS;
        let kernel = self.loopmix_down_perm.get_or_init(|| {
            let src = loopmix_down_perm_source();
            MetalKernel::new(
                &self.stream,
                "et_inkling2_loopmix_down_perm",
                &["mat", "vec"],
                &["out"],
                &src,
                "",
            )
            .expect("loopmix_down_perm kernel creation failed")
        });
        let out_shape = [b2 / 2, rows, n_cols];
        let mut outs = unsafe {
            kernel.call(
                &[perm.clone(), z.clone()],
                &[(&out_shape[..], MlxDType::F32)],
                [sgs_total * 64, (b2 / 2) * rows, 1],
                [64 * sgs, 1, 1],
                &[
                    ("R", TemplateArg::Int(rows as i32)),
                    ("K", TemplateArg::Int(k as i32)),
                    ("N", TemplateArg::Int(n_cols as i32)),
                    ("U", TemplateArg::Int(LOOPMIX_PERM_U as i32)),
                    ("TN", TemplateArg::Int(1)),
                    ("SGS", TemplateArg::Int(sgs as i32)),
                ],
            )?
        };
        Ok(outs.remove(0))
    }

    /// Fused `loopmix_down_pair` reading unpermuted BF16 `w16`.
    pub fn loopmix_down_pair(&self, z: &Array, w16: &Array) -> Result<Array> {
        let (b2, rows, k) = (z.shape()[0], z.shape()[1], z.shape()[2]);
        let n_cols = w16.shape()[2];
        let (u, tn, sgs, sgs_total) = loopmix_launch(k, n_cols);
        let kernel = self.loopmix_down.get_or_init(|| {
            let src = loopmix_down_pair_source();
            MetalKernel::new(
                &self.stream,
                "et_inkling2_loopmix_down",
                &["mat", "vec"],
                &["out"],
                &src,
                "",
            )
            .expect("loopmix_down kernel creation failed")
        });
        let out_shape = [b2 / 2, rows, n_cols];
        let mut outs = unsafe {
            kernel.call(
                &[w16.clone(), z.clone()],
                &[(&out_shape[..], MlxDType::F32)],
                [sgs_total * 64, (b2 / 2) * rows, 1],
                [64 * sgs, 1, 1],
                &[
                    ("R", TemplateArg::Int(rows as i32)),
                    ("K", TemplateArg::Int(k as i32)),
                    ("N", TemplateArg::Int(n_cols as i32)),
                    ("U", TemplateArg::Int(u as i32)),
                    ("TN", TemplateArg::Int(tn as i32)),
                    ("SGS", TemplateArg::Int(sgs as i32)),
                ],
            )?
        };
        Ok(outs.remove(0))
    }

    /// Fused `loopmix_gates_mix`: gain/forget GEMV_T + `(1 - tanh(forget)) * x + tanh(gain) * msg`.
    pub fn loopmix_gates_mix(
        &self,
        low: &Array,
        w16: &Array,
        x: &Array,
        message: &Array,
    ) -> Result<Array> {
        let (n, rows, k) = (low.shape()[0], low.shape()[1], low.shape()[2]);
        let h = x.shape()[3];
        let (u, tn, sgs, sgs_half) = loopmix_launch(k, h);
        let kernel = self.loopmix_gates.get_or_init(|| {
            let src = loopmix_gates_mix_source();
            MetalKernel::new_strided(
                &self.stream,
                "et_inkling2_loopmix_gates",
                &["mat", "vec", "x", "msg"],
                &["out"],
                &src,
                "",
            )
            .expect("loopmix_gates kernel creation failed")
        });
        let out_shape = x.shape().to_vec();
        let mut outs = unsafe {
            kernel.call(
                &[w16.clone(), low.clone(), x.clone(), message.clone()],
                &[(&out_shape[..], MlxDType::F32)],
                [sgs_half * 64, n * rows, 1],
                [64 * sgs, 1, 1],
                &[
                    ("R", TemplateArg::Int(rows as i32)),
                    ("K", TemplateArg::Int(k as i32)),
                    ("N", TemplateArg::Int((2 * h) as i32)),
                    ("HH", TemplateArg::Int(h as i32)),
                    ("U", TemplateArg::Int(u as i32)),
                    ("TN", TemplateArg::Int(tn as i32)),
                    ("SGS", TemplateArg::Int(sgs as i32)),
                ],
            )?
        };
        Ok(outs.remove(0))
    }

    /// Two-pass parallel first-index argmax over 2-D `vals` `[R, N]`, matching `mx.argmax`.
    /// When `ttab` and `bias` are provided (`vals` BF16), compares `ttab[bits(v)] + bias[i]`.
    pub fn argmax_rows(
        &self,
        vals: &Array,
        ttab: Option<&Array>,
        bias: Option<&Array>,
    ) -> Result<Array> {
        let (rows, n) = (vals.shape()[0], vals.shape()[1]);
        let nc = n.div_ceil(ARGMAX_CH);
        let (p1, inputs) = if let (Some(tt), Some(bi)) = (ttab, bias) {
            let k = self.argmax_p1_table.get_or_init(|| {
                let src = argmax_p1_source("ttab[as_type<ushort>(vals[r * N + i])] + bias[i]");
                MetalKernel::new(
                    &self.stream,
                    "et_inkling2_argmax_p1_table",
                    &["vals", "ttab", "bias"],
                    &["pv", "pi"],
                    &src,
                    ARGMAX_BETTER,
                )
                .expect("argmax p1_table kernel creation failed")
            });
            (k, vec![vals.clone(), tt.clone(), bi.clone()])
        } else {
            let k = self.argmax_p1_plain.get_or_init(|| {
                let src = argmax_p1_source("static_cast<float>(vals[r * N + i])");
                MetalKernel::new(
                    &self.stream,
                    "et_inkling2_argmax_p1_plain",
                    &["vals"],
                    &["pv", "pi"],
                    &src,
                    ARGMAX_BETTER,
                )
                .expect("argmax p1_plain kernel creation failed")
            });
            (k, vec![vals.clone()])
        };
        let p_shape = [rows, nc];
        let mut p1_outs = unsafe {
            p1.call(
                &inputs,
                &[(&p_shape[..], MlxDType::F32), (&p_shape[..], MlxDType::U32)],
                [nc * ARGMAX_TG, rows, 1],
                [ARGMAX_TG, 1, 1],
                &[
                    ("N", TemplateArg::Int(n as i32)),
                    ("NC", TemplateArg::Int(nc as i32)),
                    ("CH", TemplateArg::Int(ARGMAX_CH as i32)),
                    ("TG", TemplateArg::Int(ARGMAX_TG as i32)),
                ],
            )?
        };
        let pi = p1_outs.pop().unwrap();
        let pv = p1_outs.pop().unwrap();

        let p2 = self.argmax_p2.get_or_init(|| {
            let src = argmax_p2_source();
            MetalKernel::new(
                &self.stream,
                "et_inkling2_argmax_p2",
                &["pv", "pi"],
                &["out"],
                &src,
                ARGMAX_BETTER,
            )
            .expect("argmax p2 kernel creation failed")
        });
        let out_shape = [rows];
        let mut outs = unsafe {
            p2.call(
                &[pv, pi],
                &[(&out_shape[..], MlxDType::U32)],
                [ARGMAX_TG, rows, 1],
                [ARGMAX_TG, 1, 1],
                &[
                    ("NC", TemplateArg::Int(nc as i32)),
                    ("TG", TemplateArg::Int(ARGMAX_TG as i32)),
                ],
            )?
        };
        Ok(outs.remove(0))
    }

    /// Certified q8 g64 LM-head screen argmax for BF16 `w` `[V, H]` and `x` `[R, H]`.
    /// Returns `[R]` I32 token IDs, or `-1` everywhere if the certificate fails.
    pub fn screen_argmax(&self, w: &Array, x: &Array, screen: &VerifyHeadScreen) -> Result<Array> {
        let (n_out, k) = (w.shape()[0], w.shape()[1]);
        let rows = x.shape()[0];
        let xw = if x.dtype() == w.dtype() {
            x.clone()
        } else {
            x.cast(w.dtype())?
        };
        let per_tg = SCREEN_TM * SCREEN_BM;
        let ntg = n_out.div_ceil(per_tg);

        let k_q8 = self.screen_q8.get_or_init(|| {
            MetalKernel::new(
                &self.stream,
                "et_inkling2_screen_q8",
                &["wq", "sc", "bi", "err", "x", "ca"],
                &["upper", "tgmax"],
                Q8_SCREEN_SOURCE,
                "",
            )
            .expect("screen_q8 kernel creation failed")
        });
        let upper_shape = [rows, n_out];
        let tgmax_shape = [rows, ntg];
        let mut q8_outs = unsafe {
            k_q8.call(
                &[
                    screen.wq.clone(),
                    screen.sc.clone(),
                    screen.bi.clone(),
                    screen.err.clone(),
                    xw.clone(),
                    screen.ca.clone(),
                ],
                &[
                    (&upper_shape[..], MlxDType::F32),
                    (&tgmax_shape[..], MlxDType::F32),
                ],
                [ntg * 32 * SCREEN_BM, 1, 1],
                [32 * SCREEN_BM, 1, 1],
                &[
                    ("R", TemplateArg::Int(rows as i32)),
                    ("H", TemplateArg::Int(k as i32)),
                    ("V", TemplateArg::Int(n_out as i32)),
                    ("VU", TemplateArg::Int(screen.unpadded as i32)),
                    ("TM", TemplateArg::Int(SCREEN_TM as i32)),
                    ("BM", TemplateArg::Int(SCREEN_BM as i32)),
                    ("NTG", TemplateArg::Int(ntg as i32)),
                ],
            )?
        };
        let tgmax = q8_outs.pop().unwrap();
        let upper = q8_outs.pop().unwrap();

        let k_lstar = self.screen_lstar.get_or_init(|| {
            MetalKernel::new(
                &self.stream,
                "et_inkling2_screen_lstar",
                &["tgmax", "ptab"],
                &["thr"],
                SCREEN_LSTAR_SOURCE,
                "",
            )
            .expect("screen_lstar kernel creation failed")
        });
        let thr_shape = [rows];
        let thr = unsafe {
            k_lstar.call(
                &[tgmax, screen.ptab.clone()],
                &[(&thr_shape[..], MlxDType::F32)],
                [1024, rows, 1],
                [1024, 1, 1],
                &[
                    ("NTG", TemplateArg::Int(ntg as i32)),
                    ("TG", TemplateArg::Int(1024)),
                ],
            )?
        }
        .remove(0);

        let k_select = self.screen_select.get_or_init(|| {
            MetalKernel::new_with_options(
                &self.stream,
                "et_inkling2_screen_select",
                &["upper", "thr"],
                &["count", "cand"],
                SCREEN_SELECT_SOURCE,
                "",
                true,
                true,
            )
            .expect("screen_select kernel creation failed")
        });
        let count_shape = [1usize];
        let cand_shape = [SCREEN_CAP];
        let mut sel_outs = unsafe {
            k_select.call_with_init(
                &[upper, thr.clone()],
                &[
                    (&count_shape[..], MlxDType::U32),
                    (&cand_shape[..], MlxDType::U32),
                ],
                [n_out, 1, 1],
                [256, 1, 1],
                &[
                    ("R", TemplateArg::Int(rows as i32)),
                    ("V", TemplateArg::Int(n_out as i32)),
                    ("CAP", TemplateArg::Int(SCREEN_CAP as i32)),
                ],
                Some(0.0),
            )?
        };
        let cand = sel_outs.pop().unwrap();
        let count = sel_outs.pop().unwrap();

        let (bm_g, bn_g, sm_g, sn_g, tm_g, tn_g) = gemv_params(n_out, k);
        let k_gather = self.screen_gather.get_or_init(|| {
            let src = gather_gemv_source();
            MetalKernel::new(
                &self.stream,
                "et_inkling2_screen_gather",
                &["mat", "vec", "cand", "count"],
                &["out"],
                &src,
                "",
            )
            .expect("screen_gather kernel creation failed")
        });
        let rech_shape = [rows, SCREEN_CAP];
        let g_per_tgp = bm_g * sm_g * tm_g;
        let g_ntgp = SCREEN_CAP.div_ceil(g_per_tgp);
        let rech = unsafe {
            k_gather.call(
                &[w.clone(), xw, cand.clone(), count.clone()],
                &[(&rech_shape[..], w.dtype())],
                [g_ntgp * 32 * bn_g * bm_g, 1, 1],
                [32 * bn_g * bm_g, 1, 1],
                &[
                    ("T", TemplateArg::DType(w.dtype())),
                    ("R", TemplateArg::Int(rows as i32)),
                    ("IN", TemplateArg::Int(k as i32)),
                    ("OUT", TemplateArg::Int(SCREEN_CAP as i32)),
                    ("BM", TemplateArg::Int(bm_g as i32)),
                    ("BN", TemplateArg::Int(bn_g as i32)),
                    ("SM", TemplateArg::Int(sm_g as i32)),
                    ("SN", TemplateArg::Int(sn_g as i32)),
                    ("TM", TemplateArg::Int(tm_g as i32)),
                    ("TN", TemplateArg::Int(tn_g as i32)),
                ],
            )?
        }
        .remove(0);

        let k_final = self.screen_final.get_or_init(|| {
            let src = screen_final_source();
            MetalKernel::new(
                &self.stream,
                "et_inkling2_screen_final",
                &["rech", "cand", "count", "ttab", "bias", "thr"],
                &["out"],
                &src,
                ARGMAX_BETTER,
            )
            .expect("screen_final kernel creation failed")
        });
        let out_shape = [rows];
        let mut outs = unsafe {
            k_final.call(
                &[
                    rech,
                    cand,
                    count,
                    screen.ttab.clone(),
                    screen.bias.clone(),
                    thr,
                ],
                &[(&out_shape[..], MlxDType::I32)],
                [256, rows, 1],
                [256, 1, 1],
                &[
                    ("CAP", TemplateArg::Int(SCREEN_CAP as i32)),
                    ("TG", TemplateArg::Int(256)),
                    ("R", TemplateArg::Int(rows as i32)),
                ],
            )?
        };
        Ok(outs.remove(0))
    }
}
