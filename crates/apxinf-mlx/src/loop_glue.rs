//! Fused L2 clip, RMS norm with register casts, causal short convolutions
//! (decode/verify + prefill), loop-message head/tail/RMS fusions, and prefill
//! gate/residual epilogues on Metal.
//!
//! Ported from EngineTailor's Inkling-2 MLX kernels (Apache-2.0) and
//! Apple MLX 0.32.4's `reduce_row.h` / `rms_norm.metal` (MIT).

use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::Result;
use std::cell::{OnceCell, RefCell};
use std::collections::HashMap;

const NO_FMA_HEADER: &str = "\n#pragma clang fp contract(off)\n";

const L2_CLIP_SOURCE: &str = r#"
    constexpr int NR = 4;
    const float thr = static_cast<float>(threshold[0]);
if (D_IS_SIMD) {
    const uint row = thread_position_in_grid.x / 32;
    const uint lane = thread_index_in_simdgroup;
    if (row >= NROWS) return;
    const uint h = row % HEADS;
    const uint bl = row / HEADS;
    const int base = int(bl / LROWS) * int(x_strides[0]) + int(bl % LROWS) * int(x_strides[1])
        + int(h) * int(x_strides[2]);
    const int sd = int(x_strides[3]);
    float xf[NR];
    for (int i = 0; i < NR; i++) {
      const float xv = static_cast<float>(x[base + int(lane * NR + i) * sd]);
      const float sv = SCALE_FULL ? static_cast<float>(scale[h * D + lane * NR + i])
                                  : static_cast<float>(scale[0]);
      xf[i] = xv * sv;
    }
    float sq[NR];
    for (int i = 0; i < NR; i++) {
      sq[i] = xf[i] * xf[i];
    }
    float total = 0.0f;
    for (int i = 0; i < NR; i++) {
      total = sq[i] + total;
    }
    total = simd_sum(total);
    const float nrm = metal::precise::sqrt(total);
    const float den = nrm + 1e-6f;
    const float qv = thr / den;
    const float m = metal::isnan(qv) ? qv : (qv < 1.0f ? qv : 1.0f);
    for (int i = 0; i < NR; i++) {
      const uint e = row * D + lane * NR + i;
      T o = static_cast<T>(xf[i] * m);
      if (HAS_TAU) {
        o = static_cast<T>(static_cast<float>(o) * tau[(row / HEADS) % LROWS]);
      }
      out[e] = o;
    }
} else {
    const uint row = thread_position_in_grid.x;
    if (row >= NROWS) return;
    const uint h = row % HEADS;
    const uint bl = row / HEADS;
    const int base = int(bl / LROWS) * int(x_strides[0]) + int(bl % LROWS) * int(x_strides[1])
        + int(h) * int(x_strides[2]);
    const int sd = int(x_strides[3]);
    float xf[D];
    for (int i = 0; i < D; i++) {
      const float xv = static_cast<float>(x[base + i * sd]);
      const float sv = SCALE_FULL ? static_cast<float>(scale[h * D + i])
                                  : static_cast<float>(scale[0]);
      xf[i] = xv * sv;
    }
    float sq[D];
    for (int i = 0; i < D; i++) {
      sq[i] = xf[i] * xf[i];
    }
    float total = 0.0f;
    for (int b = 0; b < D / NR; b++) {
      float vals[NR];
      for (int j = 0; j < NR; j++) {
        vals[j] = sq[b * NR + j];
      }
      for (int j = 0; j < NR; j++) {
        total = vals[j] + total;
      }
    }
    const float nrm = metal::precise::sqrt(total);
    const float den = nrm + 1e-6f;
    const float qv = thr / den;
    const float m = metal::isnan(qv) ? qv : (qv < 1.0f ? qv : 1.0f);
    for (int i = 0; i < D; i++) {
      const uint e = row * D + i;
      T o = static_cast<T>(xf[i] * m);
      if (HAS_TAU) {
        o = static_cast<T>(static_cast<float>(o) * tau[(row / HEADS) % LROWS]);
      }
      out[e] = o;
    }
}
"#;

const RMS_SOURCE: &str = r#"
    constexpr int NR = 4;
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    const uint gid = threadgroup_position_in_grid.x;
    const uint lid = thread_position_in_threadgroup.x;
    const uint simd_lane_id = thread_index_in_simdgroup;
    const uint simd_group_id = simdgroup_index_in_threadgroup;
    const float eps_v = static_cast<float>(eps[0]);
    const device TX* xr = x + size_t(gid) * D + lid * NR;
    float xv[NR];
    float acc = 0;
    if (lid * NR + NR <= D) {
      for (int i = 0; i < NR; i++) {
        float xi = static_cast<float>(static_cast<TC>(xr[i]));
        xv[i] = xi;
        acc += xi * xi;
      }
    } else {
      for (int i = 0; i < NR; i++) {
        if ((lid * NR + i) < D) {
          float xi = static_cast<float>(static_cast<TC>(xr[i]));
          xv[i] = xi;
          acc += xi * xi;
        }
      }
    }
    acc = simd_sum(acc);
    if (simd_group_id == 0) {
      local_sums[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
      local_sums[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
      acc = simd_sum(local_sums[simd_lane_id]);
      if (simd_lane_id == 0) {
        local_inv_mean[0] = metal::precise::rsqrt(acc / D + eps_v);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device TO* outr = out + size_t(gid) * D + lid * NR;
    for (int i = 0; i < NR; i++) {
      if ((lid * NR + i) < D) {
        const TC wi = HAS_W ? static_cast<TC>(w[lid * NR + i]) : static_cast<TC>(1);
        const TC t = static_cast<TC>(xv[i] * local_inv_mean[0]);
        outr[i] = static_cast<TO>(wi * t);
      }
    }
"#;

const CONV_WINDOW_SOURCE: &str = r#"
    const uint c = thread_position_in_grid.x;
    const uint r = thread_position_in_grid.y;
    if (c >= C) return;
    const uint o = r * C + c;
    if (r == 0) {
      out[o] = s0[c];
    } else if (r == 1) {
      out[o] = s1[c];
    } else if (r == 2) {
      out[o] = s2[c];
    } else {
      const int l = int(r) - 3;
      out[o] = static_cast<float>(x[l * int(x_strides[1]) + int(c) * int(x_strides[2])]);
    }
"#;

const CONVF_SOURCE: &str = r#"
    const uint c = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    if (c >= C) return;
    const int ci = int(c);
    const int ti = int(t);
    const int L = int(x_shape[1]);
    const int xr = int(x_strides[1]);
    const int xc = int(x_strides[2]);
    float xv[K];
    for (int j = 0; j < K; j++) {
      const int r = ti + j;
      xv[j] = r == 0 ? s0[ci] : (r == 1 ? s1[ci] : (r == 2 ? s2[ci]
            : static_cast<float>(x[(r - (K - 1)) * xr + ci * xc])));
    }
    if (ti == 0) {
      xp[ci] = s0[ci];
      xp[C + ci] = s1[ci];
      xp[2 * C + ci] = s2[ci];
    }
    xp[size_t(ti + K - 1) * C + ci] = xv[K - 1];
#if DYN
    const int head = ci / D;
    const int dr = int(dyn_strides[1]);
    const int dc = int(dyn_strides[2]);
    const int wr = int(w_strides[0]);
    const int wc = int(w_strides[1]);
    const int db = ti * dr + head * K * dc;
    float acc = 0.0f;
    for (int lag = 0; lag < K; lag++) {
      const float tap = static_cast<float>(w[lag * wr + ci * wc])
                      + static_cast<float>(dyn[db + lag * dc]);
      const float term = xv[K - 1 - lag] * tap;
      acc = lag == 0 ? term : acc + term;
    }
    float o = acc + xv[K - 1];
    o = o * scale[0];
#else
    const int wr = int(w_strides[0]);
    const int wc = int(w_strides[1]);
    float acc = static_cast<float>(w[ci * wr]) * xv[0];
    for (int j = 1; j < K; j++) {
      const float p = static_cast<float>(w[ci * wr + j * wc]) * xv[j];
      acc = acc + p;
    }
    float o = static_cast<float>(static_cast<TW>(acc));
    o = o + xv[K - 1];
    o = o * scale[0];
#endif
    const TO ob = static_cast<TO>(o);
    const size_t oi = size_t(ti) * C + c;
#if MODE == 0
    out[oi] = ob;
#elif MODE == 1
    if (BOUNDARY) {
      out[oi] = static_cast<float>(ob);
    } else {
      const float hv = static_cast<float>(h[ti * int(h_strides[1]) + ci * int(h_strides[2])]);
      const float av = static_cast<float>(ob);
      out[oi] = hv + av;
    }
#else
    const float mv = static_cast<float>(ob);
    const float rv = res[oi];
    out[oi] = static_cast<TH>(rv + mv);
#endif
"#;

const PREFILL_STATIC_CONV_SOURCE: &str = r#"
    const uint c = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    if (c >= C) return;
    const int L = int(x_shape[1]);
    const int ci = int(c);
    const int ti = int(t);
    const int xr = int(x_strides[1]);
    const int xc = int(x_strides[2]);
    const int sr = int(state_strides[1]);
    const int sc = int(state_strides[2]);
    const int wr = int(w_strides[0]);
    const int wc = int(w_strides[1]);
    float xv[K];
    for (int j = 0; j < K; j++) {
      const int r = ti + j;
      xv[j] = r < K - 1 ? static_cast<float>(state[r * sr + ci * sc])
                        : static_cast<float>(x[(r - (K - 1)) * xr + ci * xc]);
    }
    float acc = static_cast<float>(w[ci * wr]) * xv[0];
    for (int j = 1; j < K; j++) {
      const float p = static_cast<float>(w[ci * wr + j * wc]) * xv[j];
      acc = acc + p;
    }
    float o = static_cast<float>(static_cast<TW>(acc));
    o = o + xv[K - 1];
    o = o * scale[0];
    out[size_t(ti) * C + c] = static_cast<TO>(o);
    if (ti >= L - (K - 1)) {
      nstate[size_t(ti - (L - (K - 1))) * C + c] = xv[K - 1];
    }
"#;

const PREFILL_DYNAMIC_CONV_SOURCE: &str = r#"
    const uint c = thread_position_in_grid.x;
    const uint t = thread_position_in_grid.y;
    if (c >= C) return;
    const int L = int(x_shape[1]);
    const int ci = int(c);
    const int ti = int(t);
    const int head = ci / D;
    const int xr = int(x_strides[1]);
    const int xc = int(x_strides[2]);
    const int sr = int(state_strides[1]);
    const int sc = int(state_strides[2]);
    const int dr = int(dyn_strides[1]);
    const int dc = int(dyn_strides[2]);
    const int wr = int(sw_strides[0]);
    const int wc = int(sw_strides[1]);
    float xv[K];
    for (int j = 0; j < K; j++) {
      const int r = ti + j;
      xv[j] = r < K - 1 ? static_cast<float>(state[r * sr + ci * sc])
                        : static_cast<float>(x[(r - (K - 1)) * xr + ci * xc]);
    }
    const int db = ti * dr + head * K * dc;
    float acc = 0.0f;
    for (int lag = 0; lag < K; lag++) {
      const float tap = static_cast<float>(sw[lag * wr + ci * wc])
                      + static_cast<float>(dyn[db + lag * dc]);
      const float term = xv[K - 1 - lag] * tap;
      acc = lag == 0 ? term : acc + term;
    }
    float o = acc + xv[K - 1];
    o = o * scale[0];
    out[size_t(ti) * C + c] = static_cast<TO>(o);
    if (ti >= L - (K - 1)) {
      nstate[size_t(ti - (L - (K - 1))) * C + c] = xv[K - 1];
    }
"#;

const PREFILL_MIX_HEAD_SOURCE: &str = r#"
    constexpr int NR = 4;
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    const uint gid = threadgroup_position_in_grid.x;
    const uint lid = thread_position_in_threadgroup.x;
    const uint simd_lane_id = thread_index_in_simdgroup;
    const uint simd_group_id = simdgroup_index_in_threadgroup;
    const int L = int(x_shape[2]);
    const int rows = int(x_shape[1]) * L;
    const int s = int(gid) / rows;
    const int r = int(gid) - s * rows;
    const int ii = s >> 1;
    const int bi = r / L;
    const int li = r - bi * L;
    const device float* src;
    int es;
    if ((s & 1) != 0) {
      src = msg + ii * int(msg_strides[0]) + bi * int(msg_strides[1]) + li * int(msg_strides[2]);
      es = int(msg_strides[3]);
    } else {
      src = x + ii * int(x_strides[0]) + bi * int(x_strides[1]) + li * int(x_strides[2]);
      es = int(x_strides[3]);
    }
    const int c0 = int(lid) * NR;
    float xv[NR];
    float acc = 0;
    if (c0 + NR <= D) {
      for (int i = 0; i < NR; i++) {
        float xi = src[(c0 + i) * es];
        xv[i] = xi;
        acc += xi * xi;
      }
    } else {
      for (int i = 0; i < NR; i++) {
        if ((c0 + i) < D) {
          float xi = src[(c0 + i) * es];
          xv[i] = xi;
          acc += xi * xi;
        }
      }
    }
    acc = simd_sum(acc);
    if (simd_group_id == 0) {
      local_sums[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
      local_sums[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
      acc = simd_sum(local_sums[simd_lane_id]);
      if (simd_lane_id == 0) {
        local_inv_mean[0] = metal::precise::rsqrt(acc / D + eps[0]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* outr = out + size_t(gid) * D + c0;
    for (int i = 0; i < NR; i++) {
      if ((c0 + i) < D) {
        const float t = xv[i] * local_inv_mean[0];
        outr[i] = static_cast<float>(static_cast<TB>(t));
      }
    }
"#;

const PREFILL_MIX_HEAD_OPT_SOURCE: &str = r#"
    constexpr int NR = 4;
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    const uint gid = threadgroup_position_in_grid.x;
    const uint lid = thread_position_in_threadgroup.x;
    const uint simd_lane_id = thread_index_in_simdgroup;
    const uint simd_group_id = simdgroup_index_in_threadgroup;
    const int N = int(x_shape[0]);
    const int L = int(x_shape[2]);
    const int rows = int(x_shape[1]) * L;
    const int c0 = int(lid) * NR;

    int is_msg, ii, r;
    if (DEDUP_X) {
        const int slot = int(gid) / rows;
        r = int(gid) - slot * rows;
        if (slot < N) {
            is_msg = 1;
            ii = slot;
        } else {
            is_msg = 0;
            ii = 0;
        }
    } else {
        const int s = int(gid) / rows;
        r = int(gid) - s * rows;
        is_msg = s & 1;
        ii = s >> 1;
    }
    const int bi = r / L;
    const int li = r - bi * L;
    const device float* src;
    int es;
    bool src_vec4;
    if (is_msg) {
        src = msg + int64_t(ii) * msg_strides[0] + int64_t(bi) * msg_strides[1]
              + int64_t(li) * msg_strides[2];
        es = int(msg_strides[3]);
        src_vec4 = (D % 4 == 0) && (es == 1)
            && ((msg_strides[0] & 3) == 0) && ((msg_strides[1] & 3) == 0)
            && ((msg_strides[2] & 3) == 0);
    } else {
        src = x + int64_t(ii) * x_strides[0] + int64_t(bi) * x_strides[1]
              + int64_t(li) * x_strides[2];
        es = int(x_strides[3]);
        src_vec4 = (D % 4 == 0) && (es == 1)
            && ((x_strides[0] & 3) == 0) && ((x_strides[1] & 3) == 0)
            && ((x_strides[2] & 3) == 0);
    }
    float xv[NR];
    float acc = 0.0f;
    if (c0 + NR <= D) {
        if (src_vec4) {
            const float4 v4 = *(const device float4*)(src + c0);
            for (int i = 0; i < NR; ++i) {
                float xi = v4[i];
                xv[i] = xi;
                acc += xi * xi;
            }
        } else {
            for (int i = 0; i < NR; ++i) {
                float xi = src[(c0 + i) * es];
                xv[i] = xi;
                acc += xi * xi;
            }
        }
    } else {
        for (int i = 0; i < NR; ++i) {
            if (c0 + i < D) {
                float xi = src[(c0 + i) * es];
                xv[i] = xi;
                acc += xi * xi;
            }
        }
    }
    acc = simd_sum(acc);
    if (simd_group_id == 0) {
        local_sums[simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        local_sums[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(acc / D + eps[0]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = local_inv_mean[0];
    if (D % 4 == 0 && c0 + NR <= D) {
        float4 ov;
        for (int i = 0; i < NR; ++i) {
            ov[i] = static_cast<float>(static_cast<TB>(xv[i] * inv));
        }
        if (DEDUP_X && !is_msg) {
            for (int k = 0; k < N; ++k) {
                const size_t s_out = size_t(2 * k) * size_t(rows) + size_t(r);
                *(device float4*)(out + s_out * D + c0) = ov;
            }
        } else {
            const size_t s_out = size_t(2 * ii + is_msg) * size_t(rows) + size_t(r);
            *(device float4*)(out + s_out * D + c0) = ov;
        }
    } else {
        for (int i = 0; i < NR; ++i) {
            if (c0 + i < D) {
                const float o = static_cast<float>(static_cast<TB>(xv[i] * inv));
                if (DEDUP_X && !is_msg) {
                    for (int k = 0; k < N; ++k) {
                        const size_t s_out = size_t(2 * k) * size_t(rows) + size_t(r);
                        out[s_out * D + c0 + i] = o;
                    }
                } else {
                    const size_t s_out = size_t(2 * ii + is_msg) * size_t(rows) + size_t(r);
                    out[s_out * D + c0 + i] = o;
                }
            }
        }
    }
"#;

const PREFILL_MIX_TAIL_SOURCE: &str = r#"
    const uint j = thread_position_in_grid.x;
    const uint r = thread_position_in_grid.y;
    const uint ii = thread_position_in_grid.z;
    if (j >= uint(H)) return;
    const int L = int(x_shape[2]);
    const size_t rows = size_t(x_shape[1]) * size_t(L);
    const int ri = int(r);
    const int bi = ri / L;
    const int li = ri - bi * L;
    const int ji = int(j);
    const int ib = int(ii);
    const int gb = ib * int(gates_strides[0]) + ri * int(gates_strides[1]);
    const int gc = int(gates_strides[2]);
    const float gain = metal::precise::tanh(gates[gb + ji * gc]);
    const float forget = metal::precise::tanh(gates[gb + (H + ji) * gc]);
    const float xv = x[ib * int(x_strides[0]) + bi * int(x_strides[1]) + li * int(x_strides[2])
                       + ji * int(x_strides[3])];
    const float mv = msg[ib * int(msg_strides[0]) + bi * int(msg_strides[1])
                         + li * int(msg_strides[2]) + ji * int(msg_strides[3])];
    const float a = 1.0f - forget;
    const float b = a * xv;
    const float c = gain * mv;
    out[(size_t(ib) * rows + size_t(ri)) * H + j] = b + c;
"#;

const PREFILL_MIX_TAIL_RMS_HEADER: &str = r#"
inline float et_mix_elem_no_fma(float g_in, float f_in, float xv, float mv) {
#pragma clang fp contract(off)
    const float gain = metal::precise::tanh(g_in);
    const float forget = metal::precise::tanh(f_in);
    const float a = 1.0f - forget;
    const float b = a * xv;
    const float c = gain * mv;
    return b + c;
}
"#;

const PREFILL_MIX_TAIL_RMS_SOURCE: &str = r#"
    constexpr int NR = 4;
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    const uint gid = threadgroup_position_in_grid.x;
    const uint lid = thread_position_in_threadgroup.x;
    const uint simd_lane_id = thread_index_in_simdgroup;
    const uint simd_group_id = simdgroup_index_in_threadgroup;
    const int L = int(x_shape[2]);
    const int rows = int(x_shape[1]) * L;
    const int ib = int(gid) / rows;
    const int ri = int(gid) - ib * rows;
    const int bi = ri / L;
    const int li = ri - bi * L;
    const int c0 = int(lid) * NR;

    const device float* gp = gates + int64_t(ib) * gates_strides[0]
                           + int64_t(ri) * gates_strides[1];
    const int gc = int(gates_strides[2]);
    const device float* xp = x + int64_t(ib) * x_strides[0]
                           + int64_t(bi) * x_strides[1]
                           + int64_t(li) * x_strides[2];
    const int xc = int(x_strides[3]);
    const device float* mp = msg + int64_t(ib) * msg_strides[0]
                           + int64_t(bi) * msg_strides[1]
                           + int64_t(li) * msg_strides[2];
    const int mc = int(msg_strides[3]);

    const bool vec4 = (D % 4 == 0)
        && gc == 1 && xc == 1 && mc == 1
        && ((gates_strides[0] & 3) == 0) && ((gates_strides[1] & 3) == 0)
        && ((x_strides[0] & 3) == 0) && ((x_strides[1] & 3) == 0) && ((x_strides[2] & 3) == 0)
        && ((msg_strides[0] & 3) == 0) && ((msg_strides[1] & 3) == 0) && ((msg_strides[2] & 3) == 0);

    float xv[NR];
    float acc = 0.0f;
    if (c0 + NR <= D) {
        if (vec4) {
            const float4 g4 = *(const device float4*)(gp + c0);
            const float4 f4 = *(const device float4*)(gp + D + c0);
            const float4 x4 = *(const device float4*)(xp + c0);
            const float4 m4 = *(const device float4*)(mp + c0);
            for (int i = 0; i < NR; ++i) {
                float xi = et_mix_elem_no_fma(g4[i], f4[i], x4[i], m4[i]);
                xv[i] = xi;
                acc += xi * xi;
            }
        } else {
            for (int i = 0; i < NR; ++i) {
                const int ji = c0 + i;
                float xi = et_mix_elem_no_fma(gp[ji * gc], gp[(D + ji) * gc],
                                              xp[ji * xc], mp[ji * mc]);
                xv[i] = xi;
                acc += xi * xi;
            }
        }
    } else {
        for (int i = 0; i < NR; ++i) {
            const int ji = c0 + i;
            if (ji < D) {
                float xi = et_mix_elem_no_fma(gp[ji * gc], gp[(D + ji) * gc],
                                              xp[ji * xc], mp[ji * mc]);
                xv[i] = xi;
                acc += xi * xi;
            }
        }
    }
    acc = simd_sum(acc);
    if (simd_group_id == 0) {
        local_sums[simd_lane_id] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
        local_sums[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0) {
            local_inv_mean[0] = metal::precise::rsqrt(acc / D + static_cast<float>(eps[0]));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const device TW* wp = MULTI_W ? (ib == 0 ? w0 : (ib == 1 ? w1 : w2)) : w0;
    const int ws = MULTI_W
        ? (ib == 0 ? int(w0_strides[0]) : (ib == 1 ? int(w1_strides[0]) : int(w2_strides[0])))
        : int(w0_strides[0]);
    device TO* outr = out + size_t(gid) * D + c0;
    if (c0 + NR <= D) {
        for (int i = 0; i < NR; ++i) {
            const float wi = static_cast<float>(wp[(c0 + i) * ws]);
            const float t = xv[i] * local_inv_mean[0];
            outr[i] = static_cast<TO>(wi * t);
        }
    } else {
        for (int i = 0; i < NR; ++i) {
            if (c0 + i < D) {
                const float wi = static_cast<float>(wp[(c0 + i) * ws]);
                const float t = xv[i] * local_inv_mean[0];
                outr[i] = static_cast<TO>(wi * t);
            }
        }
    }
"#;

const PREFILL_GATE_OUT_SOURCE: &str = r#"
    const uint c = thread_position_in_grid.x;
    const uint r = thread_position_in_grid.y;
    if (c >= uint(C)) return;
    const int64_t L = gate_shape[1];
    const int64_t ri = int64_t(r);
    const int64_t bi = ri / L;
    const int64_t li = ri - bi * L;
    const int64_t ci = int64_t(c);
    const int64_t hi = ci / HD;
    const int64_t di = ci - hi * HD;
    const float ov = static_cast<float>(attn[bi * attn_strides[0] + hi * attn_strides[1]
                                             + li * attn_strides[2] + di * attn_strides[3]]);
    const float gv = static_cast<float>(gate[bi * gate_strides[0] + li * gate_strides[1]
                                             + ci * gate_strides[2]]);
    const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(gv)));
    const float sg = (gv < 0) ? y : 1.0f - y;
    out[size_t(r) * C + c] = static_cast<T>(ov * sg);
"#;

const PREFILL_RESIDUAL_RMS_SOURCE: &str = r#"
    constexpr int NR = 4;
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[32];
    const uint gid = threadgroup_position_in_grid.x;
    const uint lid = thread_position_in_threadgroup.x;
    const uint simd_lane_id = thread_index_in_simdgroup;
    const uint simd_group_id = simdgroup_index_in_threadgroup;
    const float eps_v = static_cast<float>(eps[0]);
    const size_t base = size_t(gid) * D + lid * NR;
    float xv[NR];
    float acc = 0;
    if (lid * NR + NR <= D) {
      for (int i = 0; i < NR; i++) {
        float xi = BOUNDARY ? static_cast<float>(a[base + i])
                            : static_cast<float>(h[base + i]) + static_cast<float>(a[base + i]);
        xv[i] = xi;
        acc += xi * xi;
      }
    } else {
      for (int i = 0; i < NR; i++) {
        if ((lid * NR + i) < D) {
          float xi = BOUNDARY ? static_cast<float>(a[base + i])
                              : static_cast<float>(h[base + i]) + static_cast<float>(a[base + i]);
          xv[i] = xi;
          acc += xi * xi;
        }
      }
    }
    acc = simd_sum(acc);
    if (simd_group_id == 0) {
      local_sums[simd_lane_id] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0) {
      local_sums[simd_group_id] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
      acc = simd_sum(local_sums[simd_lane_id]);
      if (simd_lane_id == 0) {
        local_inv_mean[0] = metal::precise::rsqrt(acc / D + eps_v);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < NR; i++) {
      if ((lid * NR + i) < D) {
        res[base + i] = xv[i];
        const float t = xv[i] * local_inv_mean[0];
        nrm[base + i] = 1.0f * t;
      }
    }
"#;

const PREFILL_RESIDUAL_OUT_SOURCE: &str = r#"
    const uint i = thread_position_in_grid.x;
    if (i >= uint(N)) return;
    out[i] = static_cast<T>(res[i] + static_cast<float>(m[i]));
"#;

fn promote_float(a: MlxDType, b: MlxDType) -> MlxDType {
    if a == b {
        a
    } else {
        MlxDType::F32
    }
}

fn rms_tg(d: usize) -> usize {
    32 * d.div_ceil(4).div_ceil(32)
}

/// Optional epilogue fused into decode/verify `conv_fused`.
pub enum ConvFusedEpilogue<'a> {
    None,
    /// MODE 1: `residual = boundary ? f32(a) : f32(h) + f32(a)` (F32 output).
    ResidualStart { h: &'a Array, boundary: bool },
    /// MODE 2: `(res + f32(m)).astype(out_dtype)`.
    ResidualOut { res: &'a Array, out_dtype: MlxDType },
}

/// Cached Metal kernels for L2 clip, RMS norm, short causal convs, and prefill glue fusions.
pub struct LoopGlueKernels {
    stream: Stream,
    l2_clip: OnceCell<MetalKernel>,
    rms: OnceCell<MetalKernel>,
    conv_window: OnceCell<MetalKernel>,
    convf: RefCell<HashMap<(bool, u8), MetalKernel>>,
    prefill_static_conv: OnceCell<MetalKernel>,
    prefill_dynamic_conv: OnceCell<MetalKernel>,
    prefill_mix_head: OnceCell<MetalKernel>,
    prefill_mix_head_opt: OnceCell<MetalKernel>,
    prefill_mix_tail: OnceCell<MetalKernel>,
    prefill_mix_tail_rms: OnceCell<MetalKernel>,
    prefill_gate_out: OnceCell<MetalKernel>,
    prefill_residual_rms: OnceCell<MetalKernel>,
    prefill_residual_out: OnceCell<MetalKernel>,
}

impl LoopGlueKernels {
    pub fn new(stream: Stream) -> Self {
        Self {
            stream,
            l2_clip: OnceCell::new(),
            rms: OnceCell::new(),
            conv_window: OnceCell::new(),
            convf: RefCell::new(HashMap::new()),
            prefill_static_conv: OnceCell::new(),
            prefill_dynamic_conv: OnceCell::new(),
            prefill_mix_head: OnceCell::new(),
            prefill_mix_head_opt: OnceCell::new(),
            prefill_mix_tail: OnceCell::new(),
            prefill_mix_tail_rms: OnceCell::new(),
            prefill_gate_out: OnceCell::new(),
            prefill_residual_rms: OnceCell::new(),
            prefill_residual_out: OnceCell::new(),
        }
    }

    /// Fused L2 norm clip on `[B, L, H, D]` (D=128 or D=16) with optional full-layer `tau` `[L]`.
    pub fn l2_clip_fused(
        &self,
        x: &Array,
        scale: &Array,
        threshold: &Array,
        tau: Option<&Array>,
    ) -> Result<Array> {
        let shape = x.shape().to_vec();
        let (b, length, heads, d) = (shape[0], shape[1], shape[2], shape[3]);
        let rows = b * length * heads;
        let has_tau = tau.is_some();
        let simd = d == 128;
        let thr = threshold.reshape(&[1])?;
        let tau_in = match tau {
            Some(t) => t.reshape(&[t.numel()])?.cast(MlxDType::F32)?,
            None => thr.clone(),
        };
        let scale_flat = scale.reshape(&[scale.numel()])?;
        let scale_full = scale.numel() != 1;
        let threads = if simd { rows * 32 } else { rows };
        let kernel = self.l2_clip.get_or_init(|| {
            MetalKernel::new_strided(
                &self.stream,
                "et_inkling2_l2_clip",
                &["x", "scale", "threshold", "tau"],
                &["out"],
                L2_CLIP_SOURCE,
                "",
            )
            .expect("l2_clip kernel creation failed")
        });
        let mut outs = unsafe {
            kernel.call(
                &[x.clone(), scale_flat, thr, tau_in],
                &[(&shape[..], x.dtype())],
                [threads, 1, 1],
                [threads.min(256), 1, 1],
                &[
                    ("T", TemplateArg::DType(x.dtype())),
                    ("D", TemplateArg::Int(d as i32)),
                    ("D_IS_SIMD", TemplateArg::Bool(simd)),
                    ("HEADS", TemplateArg::Int(heads as i32)),
                    ("NROWS", TemplateArg::Int(rows as i32)),
                    ("LROWS", TemplateArg::Int(length as i32)),
                    ("HAS_TAU", TemplateArg::Bool(has_tau)),
                    ("SCALE_FULL", TemplateArg::Bool(scale_full)),
                ],
            )?
        };
        Ok(outs.remove(0))
    }

    /// Fused `rms_norm(x.astype(compute_dtype), w, eps).astype(out_dtype)` in one launch (`D <= 4096`).
    pub fn rms_norm_fused(
        &self,
        x: &Array,
        w: Option<&Array>,
        eps: f32,
        out_dtype: MlxDType,
        compute_dtype: Option<MlxDType>,
    ) -> Result<Array> {
        let shape = x.shape().to_vec();
        let d = *shape.last().unwrap();
        if d > 4096 {
            let xc = match compute_dtype {
                Some(dt) if dt != x.dtype() => x.cast(dt)?,
                _ => x.clone(),
            };
            let normed = match w {
                Some(wt) => xc.fast_rms_norm(wt, eps)?,
                None => xc.fast_rms_norm_unweighted(eps)?,
            };
            return if normed.dtype() == out_dtype {
                Ok(normed)
            } else {
                normed.cast(out_dtype)
            };
        }
        let rows = x.numel() / d;
        let tg = rms_tg(d);
        let has_w = w.is_some();
        let mut tc = match (compute_dtype, w) {
            (Some(dt), _) => dt,
            (None, Some(wt)) => promote_float(x.dtype(), wt.dtype()),
            (None, None) => x.dtype(),
        };
        if let Some(wt) = w {
            tc = promote_float(tc, wt.dtype());
        }
        let eps_a = Array::from_f32(&self.stream, &[1], &[eps])?;
        let kernel = self.rms.get_or_init(|| {
            MetalKernel::new(
                &self.stream,
                "et_inkling2_rms_fused",
                &["x", "w", "eps"],
                &["out"],
                RMS_SOURCE,
                "",
            )
            .expect("rms_norm_fused kernel creation failed")
        });
        let mut outs = unsafe {
            kernel.call(
                &[x.clone(), w.cloned().unwrap_or_else(|| eps_a.clone()), eps_a],
                &[(&shape[..], out_dtype)],
                [rows * tg, 1, 1],
                [tg, 1, 1],
                &[
                    ("TX", TemplateArg::DType(x.dtype())),
                    ("TC", TemplateArg::DType(tc)),
                    ("TO", TemplateArg::DType(out_dtype)),
                    ("D", TemplateArg::Int(d as i32)),
                    ("HAS_W", TemplateArg::Bool(has_w)),
                ],
            )?
        };
        Ok(outs.remove(0))
    }

    /// Concatenates three `[1, 1, C]` F32 state rows and `[1, L, C]` `x` (cast to F32) into `[1, 3+L, C]`.
    pub fn conv_window(&self, x: &Array, state: &[Array; 3]) -> Result<Array> {
        let (_, length, c) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let kernel = self.conv_window.get_or_init(|| {
            MetalKernel::new_strided(
                &self.stream,
                "et_inkling2_conv_window",
                &["s0", "s1", "s2", "x"],
                &["out"],
                CONV_WINDOW_SOURCE,
                "",
            )
            .expect("conv_window kernel creation failed")
        });
        let out_shape = [1, 3 + length, c];
        let mut outs = unsafe {
            kernel.call(
                &[
                    state[0].reshape(&[c])?,
                    state[1].reshape(&[c])?,
                    state[2].reshape(&[c])?,
                    x.clone(),
                ],
                &[(&out_shape[..], MlxDType::F32)],
                [c, 3 + length, 1],
                [c.min(256), 1, 1],
                &[("C", TemplateArg::Int(c as i32))],
            )?
        };
        Ok(outs.remove(0))
    }

    /// Decode/verify fused 4-tap short causal conv (`[1, L, C]`, `L <= 8`), returning `(y, xp_window)`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_fused(
        &self,
        x: &Array,
        w: &Array,
        state: &[Array; 3],
        scale: f32,
        dyn_taps: Option<&Array>,
        heads: usize,
        epilogue: ConvFusedEpilogue<'_>,
    ) -> Result<(Array, Array)> {
        let (_, length, c) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let is_dyn = dyn_taps.is_some();
        let k = if is_dyn {
            w.shape()[0]
        } else {
            *w.shape().last().unwrap()
        };
        let mode: u8 = match epilogue {
            ConvFusedEpilogue::None => 0,
            ConvFusedEpilogue::ResidualStart { .. } => 1,
            ConvFusedEpilogue::ResidualOut { .. } => 2,
        };
        let key = (is_dyn, mode);
        let mut cache = self.convf.borrow_mut();
        if !cache.contains_key(&key) {
            let mut names = vec!["s0", "s1", "s2", "x", "w", "scale"];
            if is_dyn {
                names.push("dyn");
            }
            if mode == 1 {
                names.push("h");
            } else if mode == 2 {
                names.push("res");
            }
            let src = format!(
                "#define DYN {}\n#define MODE {}\n{CONVF_SOURCE}",
                is_dyn as i32, mode
            );
            let name = format!(
                "et_inkling2_convf_{}_{}",
                if is_dyn { "dyn" } else { "static" },
                mode
            );
            let kernel = MetalKernel::new_strided(
                &self.stream,
                &name,
                &names,
                &["out", "xp"],
                &src,
                NO_FMA_HEADER,
            )?;
            cache.insert(key, kernel);
        }
        let kernel = cache.get(&key).unwrap();
        let mut inputs = vec![
            state[0].reshape(&[c])?,
            state[1].reshape(&[c])?,
            state[2].reshape(&[c])?,
            x.clone(),
            w.clone(),
            Array::from_f32(&self.stream, &[1], &[scale])?,
        ];
        let mut templates = vec![
            ("C", TemplateArg::Int(c as i32)),
            ("K", TemplateArg::Int(k as i32)),
            ("TO", TemplateArg::DType(x.dtype())),
        ];
        if let Some(d) = dyn_taps {
            inputs.push(d.clone());
            templates.push(("D", TemplateArg::Int((c / heads) as i32)));
        } else {
            templates.push(("TW", TemplateArg::DType(w.dtype())));
        }
        let y_dtype = match epilogue {
            ConvFusedEpilogue::None => x.dtype(),
            ConvFusedEpilogue::ResidualStart { h, boundary } => {
                inputs.push(h.clone());
                templates.push(("BOUNDARY", TemplateArg::Bool(boundary)));
                MlxDType::F32
            }
            ConvFusedEpilogue::ResidualOut { res, out_dtype } => {
                inputs.push(res.clone());
                templates.push(("TH", TemplateArg::DType(out_dtype)));
                out_dtype
            }
        };
        let y_shape = [1, length, c];
        let xp_shape = [1, k - 1 + length, c];
        let mut outs = unsafe {
            kernel.call(
                &inputs,
                &[(&y_shape[..], y_dtype), (&xp_shape[..], MlxDType::F32)],
                [c, length, 1],
                [c.min(256), 1, 1],
                &templates,
            )?
        };
        let xp = outs.pop().unwrap();
        let y = outs.pop().unwrap();
        Ok((y, xp))
    }

    /// Prefill static causal conv (`[1, L, C]`, `L >= K - 1`), returning `(out, next_state)`.
    pub fn prefill_static_conv(
        &self,
        x: &Array,
        w: &Array,
        state: &Array,
        scale: f32,
    ) -> Result<(Array, Array)> {
        let (_, length, c) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let k = *w.shape().last().unwrap();
        let kernel = self.prefill_static_conv.get_or_init(|| {
            MetalKernel::new_strided(
                &self.stream,
                "et_inkling2_prefill_static_conv",
                &["x", "w", "state", "scale"],
                &["out", "nstate"],
                PREFILL_STATIC_CONV_SOURCE,
                NO_FMA_HEADER,
            )
            .expect("prefill_static_conv kernel creation failed")
        });
        let scale_a = Array::from_f32(&self.stream, &[1], &[scale])?;
        let out_shape = [1, length, c];
        let st_shape = [1, k - 1, c];
        let mut outs = unsafe {
            kernel.call(
                &[x.clone(), w.clone(), state.clone(), scale_a],
                &[(&out_shape[..], x.dtype()), (&st_shape[..], MlxDType::F32)],
                [c, length, 1],
                [c.min(256), 1, 1],
                &[
                    ("C", TemplateArg::Int(c as i32)),
                    ("K", TemplateArg::Int(k as i32)),
                    ("TW", TemplateArg::DType(w.dtype())),
                    ("TO", TemplateArg::DType(x.dtype())),
                ],
            )?
        };
        let nstate = outs.pop().unwrap();
        let out = outs.pop().unwrap();
        Ok((out, nstate))
    }

    /// Prefill dynamic causal conv (`[1, L, C]`, `L >= K - 1`), returning `(out, next_state)`.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_dynamic_conv(
        &self,
        x: &Array,
        dyn_taps: &Array,
        static_w: &Array,
        state: &Array,
        heads: usize,
        scale: f32,
    ) -> Result<(Array, Array)> {
        let (_, length, c) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let k = static_w.shape()[0];
        let kernel = self.prefill_dynamic_conv.get_or_init(|| {
            MetalKernel::new_strided(
                &self.stream,
                "et_inkling2_prefill_dynamic_conv",
                &["x", "dyn", "sw", "state", "scale"],
                &["out", "nstate"],
                PREFILL_DYNAMIC_CONV_SOURCE,
                NO_FMA_HEADER,
            )
            .expect("prefill_dynamic_conv kernel creation failed")
        });
        let scale_a = Array::from_f32(&self.stream, &[1], &[scale])?;
        let out_shape = [1, length, c];
        let st_shape = [1, k - 1, c];
        let mut outs = unsafe {
            kernel.call(
                &[
                    x.clone(),
                    dyn_taps.clone(),
                    static_w.clone(),
                    state.clone(),
                    scale_a,
                ],
                &[(&out_shape[..], x.dtype()), (&st_shape[..], MlxDType::F32)],
                [c, length, 1],
                [c.min(256), 1, 1],
                &[
                    ("C", TemplateArg::Int(c as i32)),
                    ("K", TemplateArg::Int(k as i32)),
                    ("D", TemplateArg::Int((c / heads) as i32)),
                    ("TO", TemplateArg::DType(x.dtype())),
                ],
            )?
        };
        let nstate = outs.pop().unwrap();
        let out = outs.pop().unwrap();
        Ok((out, nstate))
    }

    /// `stack([unit_rms(x), unit_rms(msg)], 1).astype(bf16).astype(f32)` reshaped to `[2N, B*L, D]`.
    pub fn prefill_mix_head(&self, x: &Array, message: &Array, eps: f32) -> Result<Array> {
        let (n, b, length, d) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
        let rows = b * length;
        let tg = rms_tg(d);
        let kernel = self.prefill_mix_head.get_or_init(|| {
            MetalKernel::new_strided(
                &self.stream,
                "et_inkling2_prefill_mix_head",
                &["x", "msg", "eps"],
                &["out"],
                PREFILL_MIX_HEAD_SOURCE,
                "",
            )
            .expect("prefill_mix_head kernel creation failed")
        });
        let eps_a = Array::from_f32(&self.stream, &[1], &[eps])?;
        let out_shape = [2 * n, rows, d];
        let mut outs = unsafe {
            kernel.call(
                &[x.clone(), message.clone(), eps_a],
                &[(&out_shape[..], MlxDType::F32)],
                [2 * n * rows * tg, 1, 1],
                [tg, 1, 1],
                &[
                    ("D", TemplateArg::Int(d as i32)),
                    ("TB", TemplateArg::DType(MlxDType::BF16)),
                ],
            )?
        };
        Ok(outs.remove(0))
    }

    /// Vectorized `float4` + optional sliding `dedup_x` variant of `prefill_mix_head`.
    pub fn prefill_mix_head_opt(
        &self,
        x: &Array,
        message: &Array,
        eps: f32,
        dedup_x: bool,
    ) -> Result<Array> {
        let (n, b, length, d) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
        let rows = b * length;
        let tg = rms_tg(d);
        let dedup = dedup_x && n > 1;
        let slots = if dedup { n + 1 } else { 2 * n };
        let kernel = self.prefill_mix_head_opt.get_or_init(|| {
            MetalKernel::new_strided(
                &self.stream,
                "et_inkling2_prefill_mix_head_opt",
                &["x", "msg", "eps"],
                &["out"],
                PREFILL_MIX_HEAD_OPT_SOURCE,
                "",
            )
            .expect("prefill_mix_head_opt kernel creation failed")
        });
        let eps_a = Array::from_f32(&self.stream, &[1], &[eps])?;
        let out_shape = [2 * n, rows, d];
        let mut outs = unsafe {
            kernel.call(
                &[x.clone(), message.clone(), eps_a],
                &[(&out_shape[..], MlxDType::F32)],
                [slots * rows * tg, 1, 1],
                [tg, 1, 1],
                &[
                    ("D", TemplateArg::Int(d as i32)),
                    ("TB", TemplateArg::DType(MlxDType::BF16)),
                    ("DEDUP_X", TemplateArg::Bool(dedup)),
                ],
            )?
        };
        Ok(outs.remove(0))
    }

    /// `(1 - tanh(gates[..., H:])) * x + tanh(gates[..., :H]) * message` in one pass.
    pub fn prefill_mix_tail(&self, gates: &Array, x: &Array, message: &Array) -> Result<Array> {
        let shape = x.shape().to_vec();
        let (n, b, length, h) = (shape[0], shape[1], shape[2], shape[3]);
        let rows = b * length;
        let kernel = self.prefill_mix_tail.get_or_init(|| {
            MetalKernel::new_strided(
                &self.stream,
                "et_inkling2_prefill_mix_tail",
                &["gates", "x", "msg"],
                &["out"],
                PREFILL_MIX_TAIL_SOURCE,
                NO_FMA_HEADER,
            )
            .expect("prefill_mix_tail kernel creation failed")
        });
        let mut outs = unsafe {
            kernel.call(
                &[gates.clone(), x.clone(), message.clone()],
                &[(&shape[..], MlxDType::F32)],
                [h, rows, n],
                [h.min(256), 1, 1],
                &[("H", TemplateArg::Int(h as i32))],
            )?
        };
        Ok(outs.remove(0))
    }

    /// Fused `rms_norm_fused(prefill_mix_tail(gates, x, msg), w, eps, out_dtype)`.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_mix_tail_rms(
        &self,
        gates: &Array,
        x: &Array,
        message: &Array,
        w0: &Array,
        eps: f32,
        out_dtype: MlxDType,
        w1_w2: Option<(&Array, &Array)>,
    ) -> Result<Array> {
        let shape = x.shape().to_vec();
        let (n, b, length, d) = (shape[0], shape[1], shape[2], shape[3]);
        let rows = b * length;
        let tg = rms_tg(d);
        let multi_w = w1_w2.is_some();
        let (w1, w2) = match w1_w2 {
            Some((a, b)) => (a.clone(), b.clone()),
            None => (w0.clone(), w0.clone()),
        };
        let kernel = self.prefill_mix_tail_rms.get_or_init(|| {
            MetalKernel::new_strided(
                &self.stream,
                "et_inkling2_prefill_mix_tail_rms",
                &["gates", "x", "msg", "w0", "w1", "w2", "eps"],
                &["out"],
                PREFILL_MIX_TAIL_RMS_SOURCE,
                PREFILL_MIX_TAIL_RMS_HEADER,
            )
            .expect("prefill_mix_tail_rms kernel creation failed")
        });
        let eps_a = Array::from_f32(&self.stream, &[1], &[eps])?;
        let mut outs = unsafe {
            kernel.call(
                &[
                    gates.clone(),
                    x.clone(),
                    message.clone(),
                    w0.clone(),
                    w1,
                    w2,
                    eps_a,
                ],
                &[(&shape[..], out_dtype)],
                [n * rows * tg, 1, 1],
                [tg, 1, 1],
                &[
                    ("D", TemplateArg::Int(d as i32)),
                    ("TW", TemplateArg::DType(w0.dtype())),
                    ("TO", TemplateArg::DType(out_dtype)),
                    ("MULTI_W", TemplateArg::Bool(multi_w)),
                ],
            )?
        };
        Ok(outs.remove(0))
    }

    /// Prefill attention output gate: `(attn.transpose(0,2,1,3).reshape(B,L,-1) * sigmoid(gate))`.
    pub fn prefill_gate_out(&self, attn: &Array, gate: &Array) -> Result<Array> {
        let (b, nh, length, hd) = (
            attn.shape()[0],
            attn.shape()[1],
            attn.shape()[2],
            attn.shape()[3],
        );
        let c = nh * hd;
        let kernel = self.prefill_gate_out.get_or_init(|| {
            MetalKernel::new_strided(
                &self.stream,
                "et_inkling2_prefill_gate_out",
                &["attn", "gate"],
                &["out"],
                PREFILL_GATE_OUT_SOURCE,
                NO_FMA_HEADER,
            )
            .expect("prefill_gate_out kernel creation failed")
        });
        let out_shape = [b, length, c];
        let mut outs = unsafe {
            kernel.call(
                &[attn.clone(), gate.clone()],
                &[(&out_shape[..], gate.dtype())],
                [c, b * length, 1],
                [c.min(256), 1, 1],
                &[
                    ("C", TemplateArg::Int(c as i32)),
                    ("HD", TemplateArg::Int(hd as i32)),
                    ("T", TemplateArg::DType(gate.dtype())),
                ],
            )?
        };
        Ok(outs.remove(0))
    }

    /// Prefill `residual = boundary ? f32(a) : f32(h) + f32(a)` and `unit_rms(residual)`.
    pub fn prefill_residual_rms(
        &self,
        h: &Array,
        a: &Array,
        boundary: bool,
        eps: f32,
    ) -> Result<(Array, Array)> {
        let shape = h.shape().to_vec();
        let d = *shape.last().unwrap();
        let rows = h.numel() / d;
        let tg = rms_tg(d);
        let kernel = self.prefill_residual_rms.get_or_init(|| {
            MetalKernel::new(
                &self.stream,
                "et_inkling2_prefill_residual_rms",
                &["h", "a", "eps"],
                &["res", "nrm"],
                PREFILL_RESIDUAL_RMS_SOURCE,
                "",
            )
            .expect("prefill_residual_rms kernel creation failed")
        });
        let eps_a = Array::from_f32(&self.stream, &[1], &[eps])?;
        let mut outs = unsafe {
            kernel.call(
                &[h.clone(), a.clone(), eps_a],
                &[(&shape[..], MlxDType::F32), (&shape[..], MlxDType::F32)],
                [rows * tg, 1, 1],
                [tg, 1, 1],
                &[
                    ("D", TemplateArg::Int(d as i32)),
                    ("BOUNDARY", TemplateArg::Bool(boundary)),
                ],
            )?
        };
        let nrm = outs.pop().unwrap();
        let res = outs.pop().unwrap();
        Ok((res, nrm))
    }

    /// Prefill `(residual + f32(m)).astype(out_dtype)`.
    pub fn prefill_residual_out(
        &self,
        residual: &Array,
        m: &Array,
        out_dtype: MlxDType,
    ) -> Result<Array> {
        let shape = residual.shape().to_vec();
        let n = residual.numel();
        let kernel = self.prefill_residual_out.get_or_init(|| {
            MetalKernel::new(
                &self.stream,
                "et_inkling2_prefill_residual_out",
                &["res", "m"],
                &["out"],
                PREFILL_RESIDUAL_OUT_SOURCE,
                NO_FMA_HEADER,
            )
            .expect("prefill_residual_out kernel creation failed")
        });
        let mut outs = unsafe {
            kernel.call(
                &[residual.clone(), m.clone()],
                &[(&shape[..], out_dtype)],
                [n, 1, 1],
                [256, 1, 1],
                &[
                    ("N", TemplateArg::Int(n as i32)),
                    ("T", TemplateArg::DType(out_dtype)),
                ],
            )?
        };
        Ok(outs.remove(0))
    }
}
