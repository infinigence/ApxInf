//! Fused MoE routing, quantized gather QMV (NVFP4/MXFP8), softcapped SwiGLU,
//! and sorted/unsorted combine kernels on Metal.
//!
//! Ported from EngineTailor's Inkling-2 MLX kernels (Apache-2.0) and
//! Apple MLX 0.32.4's `fp_quantized.h` (MIT).

use crate::{Array, MetalKernel, MlxDType, QuantMode, Stream, TemplateArg};
use apxinf_core::{Error, Result};
use std::cell::OnceCell;

const ROUTER_SOURCE: &str = r#"
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    constexpr uint PER = E / 32;
    float sc[PER];
    float key[PER];
    for (uint j = 0; j < PER; ++j) {
        uint e = lane + 32 * j;
        float s = 1.0f / (1.0f + metal::exp(-logits[row * E + e]));
        sc[j] = s;
        key[j] = s + bias[e];
    }
    float picked[K];
    uint chosen[K];
    float total = 0.0f;
    for (uint k = 0; k < K; ++k) {
        float best = -INFINITY;
        uint bj = 0;
        for (uint j = 0; j < PER; ++j) {
            if (key[j] > best) {
                best = key[j];
                bj = j;
            }
        }
        float top = simd_max(best);
        uint mine = (best == top) ? (lane + 32 * bj) : 0xFFFFFFFFu;
        uint win = simd_min(mine);
        float s = (lane == win % 32) ? sc[win / 32] : 0.0f;
        s = simd_sum(s);
        if (lane == win % 32) key[win / 32] = -INFINITY;
        picked[k] = s;
        chosen[k] = win;
        total += s;
    }
    if (lane == 0) {
        float norm = route_scale[0] / (total + 1e-20f);
        for (uint k = 0; k < K; ++k) {
            inds[row * K + k] = chosen[k];
            weights[row * K + k] = picked[k] * norm;
        }
    }
"#;

const TILE_QMV_HEADER: &str = r#"
struct et_fp4_e2m1 {
  uint8_t bits;
  operator half() thread {
    half converted = as_type<half>(ushort((bits & 7) << 9));
    converted *= 16384.0;
    return bits & 8 ? -converted : converted;
  }
  operator float() thread {
    return static_cast<float>(this->operator half());
  }
};

struct et_fp8_e4m3 {
  uint8_t bits;
  operator half() thread {
    uint16_t v = bits & 127;
    uint16_t sign_bit = ((uint16_t)((bits >> 7) & 1)) << 15;
    uint16_t u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
    half converted = as_type<half>(u);
    return converted * 256.0;
  }
  operator float() thread {
    return static_cast<float>(this->operator half());
  }
};

struct et_fp8_e8m0 {
  uint8_t bits;
  operator float() thread {
    uint32_t out = (bits == 0 ? 0x400000 : (static_cast<uint16_t>(bits) << 23));
    return as_type<float>(out);
  }
};

template <typename T, int group_size>
static inline T et_dequantize_scale(uint8_t s) {
  if constexpr (group_size == 16) {
    return T(*(thread et_fp8_e4m3*)(&s));
  } else {
    return T(*(thread et_fp8_e8m0*)(&s));
  }
}

template <int bits, typename U = float>
struct EtDequantize {
  U operator()(uint8_t x) thread {
    if constexpr (bits == 8) {
      return U(*(thread et_fp8_e4m3*)(&x));
    } else {
      return U(*(thread et_fp4_e2m1*)(&x));
    }
  }
};

template <typename U, int values_per_thread, int bits>
inline U et_qdot_dev(const device uint8_t* w, const thread U* x_thread, U scale) {
  U accum = 0;
  if constexpr (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * EtDequantize<4>{}(ws[i]) +
           x_thread[4 * i + 1] * EtDequantize<4>{}(ws[i] >> 4) +
           x_thread[4 * i + 2] * EtDequantize<4>{}(ws[i] >> 8) +
           x_thread[4 * i + 3] * EtDequantize<4>{}(ws[i] >> 12));
    }
  } else {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * EtDequantize<8>{}(w[i]);
    }
  }
  return scale * accum;
}

template <typename U, int values_per_thread, int bits>
inline U et_qdot_safe_dev(const device uint8_t* w, const thread U* x_thread, U scale, int N) {
  U accum = 0;
  if constexpr (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * EtDequantize<4>{}(ws[i]) +
           x_thread[4 * i + 1] * EtDequantize<4>{}(ws[i] >> 4) +
           x_thread[4 * i + 2] * EtDequantize<4>{}(ws[i] >> 8) +
           x_thread[4 * i + 3] * EtDequantize<4>{}(ws[i] >> 12));
    }
  } else {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * EtDequantize<8>{}(w[i]);
    }
  }
  return scale * accum;
}
"#;

const TILE_QMV_SOURCE: &str = r#"
    constexpr int PACKS = FAST ? 2 : 1;
    constexpr int PF = 32 / BITS;
    constexpr int VPT = PF * PACKS;
    constexpr int BLOCK = VPT * 32;
    constexpr int SSTEP = GS / VPT;
    constexpr int INW = IN * 4 / PF;
    constexpr int ING = IN / GS;
    constexpr int NP = LR * TOPK;
    typedef float U;

    const uint3 tid = threadgroup_position_in_grid;
    const int simd_gid = int(simdgroup_index_in_threadgroup);
    const int simd_lid = int(thread_index_in_simdgroup);
    const int tile = int(tid.y) / NP;
    const int p = int(tid.y) % NP;
    const uint e = idx[p];
    const int xr = XPER == 1 ? p / TOPK : p;
    const int out_row = tile * 8 + simd_gid * 4;

    const device uint8_t* ws = (const device uint8_t*)w
        + (size_t(e) * OUT + out_row) * INW + simd_lid * PACKS * 4;
    const device uint8_t* sc = scales + (size_t(e) * OUT + out_row) * ING + simd_lid / SSTEP;
    const device T* xp = x + size_t(xr) * IN + simd_lid * VPT;
    device T* yp = y + size_t(p) * OUT + out_row;

    thread U x_thread[VPT];
    thread U result[4] = {0};

    int k = 0;
    const int kend = FAST ? IN : IN - BLOCK;
    for (; k < kend; k += BLOCK) {
      for (int i = 0; i < VPT; i++) {
        x_thread[i] = xp[i];
      }
      for (int row = 0; row < 4; row++) {
        auto wl = (const device uint8_t*)(ws + row * INW);
        const device auto* sl = sc + row * ING;
        U s = et_dequantize_scale<U, GS>(sl[0]);
        result[row] += et_qdot_dev<U, VPT, BITS>(wl, x_thread, s);
      }
      ws += BLOCK * 4 / PF;
      sc += BLOCK / GS;
      xp += BLOCK;
    }
    if (!FAST) {
      const int remaining = clamp(IN - k - simd_lid * VPT, 0, VPT);
      if (remaining > 0) {
        for (int i = 0; i < remaining; i++) {
          x_thread[i] = xp[i];
        }
        for (int i = remaining; i < VPT; i++) {
          x_thread[i] = 0;
        }
        for (int row = 0; row < 4; row++) {
          auto wl = (const device uint8_t*)(ws + row * INW);
          const device auto* sl = sc + row * ING;
          U s = et_dequantize_scale<U, GS>(sl[0]);
          result[row] += et_qdot_safe_dev<U, VPT, BITS>(wl, x_thread, s, remaining);
        }
      }
    }
    for (int row = 0; row < 4; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        yp[row] = static_cast<T>(result[row]);
      }
    }
"#;

fn tile_qmv_swiglu_source() -> String {
    let old_yp = "    device T* yp = y + size_t(p) * OUT + out_row;\n";
    let old_store = r#"    for (int row = 0; row < 4; row++) {
      result[row] = simd_sum(result[row]);
      if (simd_lid == 0) {
        yp[row] = static_cast<T>(result[row]);
      }
    }
"#;
    let new_store = r#"    for (int row = 0; row < 4; row++) {
      result[row] = simd_sum(result[row]);
    }
    if (simd_lid < 2) {
#pragma clang fp contract(off)
      const int jj = simd_lid;
      const T yraw_g = static_cast<T>(result[2 * jj]);
      const T yraw_u = static_cast<T>(result[2 * jj + 1]);
      const float gc = static_cast<float>(gcap[0]);
      const float uc = static_cast<float>(ucap[0]);
      float g;
      float u;
      if (HAS_S) {
        const float s13 = static_cast<float>(gs[e]) * static_cast<float>(c[0]);
        const T sb = static_cast<T>(s13);
        const T yg = yraw_g * sb;
        const T yu = yraw_u * sb;
        g = static_cast<float>(yg);
        u = static_cast<float>(yu);
      } else {
        g = static_cast<float>(yraw_g);
        u = static_cast<float>(yraw_u);
      }
      const float a0 = g / gc;
      const float a1 = metal::precise::tanh(a0);
      const float a2 = a1 * gc;
      const float sy = 1 / (1 + metal::precise::exp(metal::abs(g)));
      const float sg = (g < 0) ? sy : 1 - sy;
      const float a3 = a2 * sg;
      const float b0 = u / uc;
      const float b1 = metal::precise::tanh(b0);
      const float b2 = b1 * uc;
      const float o = a3 * b2;
      act[size_t(p) * (OUT / 2) + out_row / 2 + jj] = static_cast<T>(o);
    }
"#;
    debug_assert_eq!(TILE_QMV_SOURCE.matches(old_yp).count(), 1);
    debug_assert_eq!(TILE_QMV_SOURCE.matches(old_store).count(), 1);
    TILE_QMV_SOURCE
        .replace(old_yp, "")
        .replace(old_store, new_store)
}

const NO_FMA_HEADER: &str = "\n#pragma clang fp contract(off)\n";

const MOE_COMBINE_SORTED_SOURCE: &str = r#"
    const uint j = thread_position_in_grid.x * VEC;
    const uint t = thread_position_in_grid.y;
    if (j >= HD) {
        return;
    }
    float acc[VEC];
    for (int e = 0; e < VEC; ++e) {
        acc[e] = 0.0f;
    }
    for (int k = 0; k < TOPK; ++k) {
        const uint pair = t * TOPK + k;
        float w = weights[pair];
        if (HAS_S2) {
            w = w * s2[pair];
        }
        const device T* zr = z + (size_t)inv[pair] * HD + j;
        for (int e = 0; e < VEC; ++e) {
            const float prod = static_cast<float>(zr[e]) * w;
            acc[e] = acc[e] + prod;
        }
    }
    device T* o = out + (size_t)t * HD + j;
    for (int e = 0; e < VEC; ++e) {
        o[e] = static_cast<T>(acc[e]);
    }
"#;

const MOE_SWIGLU_SOURCE: &str = r#"
    const uint j = thread_position_in_grid.x;
    const uint p = thread_position_in_grid.y;
    if (j >= I) {
        return;
    }
    const float gc = static_cast<float>(gcap[0]);
    const float uc = static_cast<float>(ucap[0]);
    const device T* yr = y + size_t(p) * (2 * I);
    float g;
    float u;
    if (HAS_S) {
        const float s13 = static_cast<float>(gs[idx[p]]) * static_cast<float>(c[0]);
        const T sb = static_cast<T>(s13);
        const T yg = yr[2 * j] * sb;
        const T yu = yr[2 * j + 1] * sb;
        g = static_cast<float>(yg);
        u = static_cast<float>(yu);
    } else {
        g = static_cast<float>(yr[2 * j]);
        u = static_cast<float>(yr[2 * j + 1]);
    }
    const float a0 = g / gc;
    const float a1 = metal::precise::tanh(a0);
    const float a2 = a1 * gc;
    const float sy = 1 / (1 + metal::precise::exp(metal::abs(g)));
    const float sg = (g < 0) ? sy : 1 - sy;
    const float a3 = a2 * sg;
    const float b0 = u / uc;
    const float b1 = metal::precise::tanh(b0);
    const float b2 = b1 * uc;
    const float o = a3 * b2;
    act[size_t(p) * I + j] = static_cast<T>(o);
"#;

const MOE_COMBINE_ROWS_SOURCE: &str = r#"
    const uint j = thread_position_in_grid.x * VEC;
    const uint t = thread_position_in_grid.y;
    if (j >= HD) {
        return;
    }
    float acc[VEC];
    for (int e = 0; e < VEC; ++e) {
        acc[e] = 0.0f;
    }
    for (int k = 0; k < TOPK; ++k) {
        const uint pair = t * TOPK + k;
        float w = weights[pair];
        if (HAS_S2) {
            const float s2 = static_cast<float>(gs[inds[pair]]) * static_cast<float>(c[0]);
            w = w * s2;
        }
        const device T* zr = z + (size_t)pair * HD + j;
        for (int e = 0; e < VEC; ++e) {
            const float prod = static_cast<float>(zr[e]) * w;
            acc[e] = acc[e] + prod;
        }
    }
    device T* o = out + (size_t)t * HD + j;
    for (int e = 0; e < VEC; ++e) {
        o[e] = static_cast<T>(acc[e]);
    }
"#;

const PREFILL_MOE_SWIGLU_V4_SOURCE: &str = r#"
    const uint j4 = thread_position_in_grid.x;
    const uint p = thread_position_in_grid.y;
    const uint j0 = j4 * 4;
    if (j0 >= I) {
        return;
    }
    const float gc = static_cast<float>(gcap[0]);
    const float uc = static_cast<float>(ucap[0]);
    T sb = T(1);
    if (HAS_S) {
        const float s13 = static_cast<float>(gs[idx[p]]) * static_cast<float>(c[0]);
        sb = static_cast<T>(s13);
    }
    const uint4 packed = *(const device uint4*)(y + size_t(p) * (2 * I) + 2 * j0);
    ushort raw[8];
    raw[0] = ushort(packed.x & 0xFFFFu);
    raw[1] = ushort(packed.x >> 16);
    raw[2] = ushort(packed.y & 0xFFFFu);
    raw[3] = ushort(packed.y >> 16);
    raw[4] = ushort(packed.z & 0xFFFFu);
    raw[5] = ushort(packed.z >> 16);
    raw[6] = ushort(packed.w & 0xFFFFu);
    raw[7] = ushort(packed.w >> 16);
    ushort out_raw[4];
    for (int k = 0; k < 4; ++k) {
        const T gv_t = as_type<T>(raw[2 * k]);
        const T uv_t = as_type<T>(raw[2 * k + 1]);
        float g;
        float u;
        if (HAS_S) {
            const T yg = gv_t * sb;
            const T yu = uv_t * sb;
            g = static_cast<float>(yg);
            u = static_cast<float>(yu);
        } else {
            g = static_cast<float>(gv_t);
            u = static_cast<float>(uv_t);
        }
        const float a0 = g / gc;
        const float a1 = metal::precise::tanh(a0);
        const float a2 = a1 * gc;
        const float sy = 1.0f / (1.0f + metal::precise::exp(metal::abs(g)));
        const float sg = (g < 0) ? sy : 1.0f - sy;
        const float a3 = a2 * sg;
        const float b0 = u / uc;
        const float b1 = metal::precise::tanh(b0);
        const float b2 = b1 * uc;
        out_raw[k] = as_type<ushort>(static_cast<T>(a3 * b2));
    }
    uint2 out_p;
    out_p.x = uint(out_raw[0]) | (uint(out_raw[1]) << 16);
    out_p.y = uint(out_raw[2]) | (uint(out_raw[3]) << 16);
    *(device uint2*)(act + size_t(p) * I + j0) = out_p;
"#;

/// Cached Metal kernels for MoE routing, quantized gather QMV, SwiGLU, and combine.
pub struct MoeKernels {
    stream: Stream,
    router: OnceCell<MetalKernel>,
    tile_qmv: OnceCell<MetalKernel>,
    qmv_swiglu: OnceCell<MetalKernel>,
    swiglu: OnceCell<MetalKernel>,
    swiglu_v4: OnceCell<MetalKernel>,
    combine_sorted: OnceCell<MetalKernel>,
    combine_rows: OnceCell<MetalKernel>,
}

impl MoeKernels {
    pub fn new(stream: Stream) -> Self {
        Self {
            stream,
            router: OnceCell::new(),
            tile_qmv: OnceCell::new(),
            qmv_swiglu: OnceCell::new(),
            swiglu: OnceCell::new(),
            swiglu_v4: OnceCell::new(),
            combine_sorted: OnceCell::new(),
            combine_rows: OnceCell::new(),
        }
    }

    /// Sigmoid scores, top-k by `score + bias`, renormalized unbiased weights.
    /// `logits`: `[N, E]` F32, `bias`: `[E]` F32. Returns `([N, K] U32, [N, K] F32)`.
    pub fn route_topk(
        &self,
        logits: &Array,
        bias: &Array,
        top_k: usize,
        route_scale: f32,
    ) -> Result<(Array, Array)> {
        let shape = logits.shape();
        if shape.len() != 2 || shape[1] % 32 != 0 {
            return Err(Error::Contract("route_topk requires [N, E] with E % 32 == 0"));
        }
        let (rows, experts) = (shape[0], shape[1]);
        let kernel = self.router.get_or_init(|| {
            MetalKernel::new(
                &self.stream,
                "omlx_inkling2_route_topk",
                &["logits", "bias", "route_scale"],
                &["inds", "weights"],
                ROUTER_SOURCE,
                "",
            )
            .expect("route_topk kernel creation failed")
        });
        let lf = if logits.dtype() == MlxDType::F32 {
            logits.clone()
        } else {
            logits.cast(MlxDType::F32)?
        };
        let bf = if bias.dtype() == MlxDType::F32 {
            bias.clone()
        } else {
            bias.cast(MlxDType::F32)?
        };
        let rs = Array::from_f32(&self.stream, &[1], &[route_scale])?;
        let out_shape = [rows, top_k];
        let mut outs = unsafe {
            kernel.call(
                &[lf, bf, rs],
                &[(&out_shape[..], MlxDType::U32), (&out_shape[..], MlxDType::F32)],
                [rows * 32, 1, 1],
                [32, 1, 1],
                &[
                    ("E", TemplateArg::Int(experts as i32)),
                    ("K", TemplateArg::Int(top_k as i32)),
                ],
            )?
        };
        let weights = outs.pop().unwrap();
        let inds = outs.pop().unwrap();
        Ok((inds, weights))
    }

    pub fn dedup_qmv_supported(
        mode: QuantMode,
        weight: &Array,
        bits: i32,
        x: &Array,
    ) -> bool {
        if !matches!(mode, QuantMode::Nvfp4 | QuantMode::Mxfp8) || x.dtype() != MlxDType::BF16 {
            return false;
        }
        let n_out = weight.shape()[1];
        let k = weight.shape()[2] * 32 / (bits as usize);
        n_out % 8 == 0 && k % 32 == 0 && k >= 32 * (32 / (bits as usize))
    }

    /// Tile-major exact NVFP4/MXFP8 gather QMV, bit-equal to MLX per-pair `gather_qmm`.
    /// `x`: `[L, K]` (`x_per_pair=false`) or `[L * top_k, K]` (`x_per_pair=true`) BF16.
    /// Returns `[L, top_k, 1, N]` BF16.
    #[allow(clippy::too_many_arguments)]
    pub fn gather_qmv_dedup(
        &self,
        x: &Array,
        weight: &Array,
        scales: &Array,
        idx: &Array,
        bits: i32,
        group_size: i32,
        x_per_pair: bool,
    ) -> Result<Array> {
        let (rows, top_k) = (idx.shape()[0], idx.shape()[1]);
        let n_out = weight.shape()[1];
        let k = weight.shape()[2] * 32 / (bits as usize);
        let block = 32 * (32 / (bits as usize)) * 2;
        let fast = n_out % 8 == 0 && k % block == 0;
        let tiles = n_out / 8;
        let kernel = self.tile_qmv.get_or_init(|| {
            MetalKernel::new(
                &self.stream,
                "et_inkling2_gather_qmv_tile",
                &["w", "scales", "x", "idx"],
                &["y"],
                TILE_QMV_SOURCE,
                TILE_QMV_HEADER,
            )
            .expect("gather_qmv_tile kernel creation failed")
        });
        let idx_u32 = if idx.dtype() == MlxDType::U32 {
            idx.clone()
        } else {
            idx.cast(MlxDType::U32)?
        };
        let out_shape = [rows * top_k, n_out];
        let out = unsafe {
            kernel.call(
                &[weight.clone(), scales.clone(), x.clone(), idx_u32],
                &[(&out_shape[..], x.dtype())],
                [32, 2 * tiles * rows * top_k, 1],
                [32, 2, 1],
                &[
                    ("T", TemplateArg::DType(x.dtype())),
                    ("BITS", TemplateArg::Int(bits)),
                    ("GS", TemplateArg::Int(group_size)),
                    ("IN", TemplateArg::Int(k as i32)),
                    ("OUT", TemplateArg::Int(n_out as i32)),
                    ("LR", TemplateArg::Int(rows as i32)),
                    ("TOPK", TemplateArg::Int(top_k as i32)),
                    ("XPER", TemplateArg::Int(if x_per_pair { top_k as i32 } else { 1 })),
                    ("FAST", TemplateArg::Bool(fast)),
                ],
            )?
        }
        .remove(0);
        out.reshape(&[rows, top_k, 1, n_out])
    }

    /// Fused w13 tile-major NVFP4/MXFP8 QMV + optional NVFP4 post-scale + softcapped SwiGLU.
    /// Returns `[L, top_k, 1, N / 2]` BF16.
    #[allow(clippy::too_many_arguments)]
    pub fn gather_qmv_swiglu(
        &self,
        x: &Array,
        weight: &Array,
        scales: &Array,
        idx: &Array,
        bits: i32,
        group_size: i32,
        gs: Option<&Array>,
        c: Option<&Array>,
        gcap: &Array,
        ucap: &Array,
    ) -> Result<Array> {
        let (rows, top_k) = (idx.shape()[0], idx.shape()[1]);
        let n_out = weight.shape()[1];
        let k = weight.shape()[2] * 32 / (bits as usize);
        let block = 32 * (32 / (bits as usize)) * 2;
        let fast = n_out % 8 == 0 && k % block == 0;
        let tiles = n_out / 8;
        let has_s = gs.is_some();
        let gcap1 = gcap.reshape(&[1])?;
        let ucap1 = ucap.reshape(&[1])?;
        let kernel = self.qmv_swiglu.get_or_init(|| {
            let src = tile_qmv_swiglu_source();
            MetalKernel::new(
                &self.stream,
                "et_inkling2_gather_qmv_tile_swiglu",
                &["w", "scales", "x", "idx", "gs", "c", "gcap", "ucap"],
                &["act"],
                &src,
                TILE_QMV_HEADER,
            )
            .expect("gather_qmv_swiglu kernel creation failed")
        });
        let idx_u32 = if idx.dtype() == MlxDType::U32 {
            idx.clone()
        } else {
            idx.cast(MlxDType::U32)?
        };
        let out_shape = [rows * top_k, n_out / 2];
        let out = unsafe {
            kernel.call(
                &[
                    weight.clone(),
                    scales.clone(),
                    x.clone(),
                    idx_u32,
                    gs.cloned().unwrap_or_else(|| gcap1.clone()),
                    c.cloned().unwrap_or_else(|| gcap1.clone()),
                    gcap1,
                    ucap1,
                ],
                &[(&out_shape[..], x.dtype())],
                [32, 2 * tiles * rows * top_k, 1],
                [32, 2, 1],
                &[
                    ("T", TemplateArg::DType(x.dtype())),
                    ("BITS", TemplateArg::Int(bits)),
                    ("GS", TemplateArg::Int(group_size)),
                    ("IN", TemplateArg::Int(k as i32)),
                    ("OUT", TemplateArg::Int(n_out as i32)),
                    ("LR", TemplateArg::Int(rows as i32)),
                    ("TOPK", TemplateArg::Int(top_k as i32)),
                    ("XPER", TemplateArg::Int(1)),
                    ("FAST", TemplateArg::Bool(fast)),
                    ("HAS_S", TemplateArg::Bool(has_s)),
                ],
            )?
        }
        .remove(0);
        out.reshape(&[rows, top_k, 1, n_out / 2])
    }

    /// Vectorized 4-pair softcapped SwiGLU (with optional NVFP4 post-scale) for sorted prefill.
    pub fn prefill_moe_swiglu_v4(
        &self,
        y: &Array,
        idx: &Array,
        gs: Option<&Array>,
        c: Option<&Array>,
        gcap: &Array,
        ucap: &Array,
    ) -> Result<Array> {
        let orig_shape = y.shape().to_vec();
        let two_i = *orig_shape.last().unwrap();
        let half = two_i / 2;
        let pairs = y.numel() / two_i;
        let has_s = gs.is_some();
        let gcap1 = gcap.reshape(&[1])?;
        let ucap1 = ucap.reshape(&[1])?;
        let gx = half / 4;
        let tg_x = if gx % 144 == 0 { 144 } else { gx.min(256) };
        let idx_flat = idx.reshape(&[idx.numel()])?;
        let idx_u32 = if idx_flat.dtype() == MlxDType::U32 {
            idx_flat
        } else {
            idx_flat.cast(MlxDType::U32)?
        };
        let kernel = self.swiglu_v4.get_or_init(|| {
            MetalKernel::new(
                &self.stream,
                "et_inkling2_prefill_moe_swiglu_v4",
                &["y", "idx", "gs", "c", "gcap", "ucap"],
                &["act"],
                PREFILL_MOE_SWIGLU_V4_SOURCE,
                NO_FMA_HEADER,
            )
            .expect("prefill_moe_swiglu_v4 kernel creation failed")
        });
        let out_shape = [pairs, half];
        let out = unsafe {
            kernel.call(
                &[
                    y.reshape(&[pairs, two_i])?,
                    idx_u32,
                    gs.cloned().unwrap_or_else(|| gcap1.clone()),
                    c.cloned().unwrap_or_else(|| gcap1.clone()),
                    gcap1,
                    ucap1,
                ],
                &[(&out_shape[..], y.dtype())],
                [gx, pairs, 1],
                [tg_x, 1, 1],
                &[
                    ("T", TemplateArg::DType(y.dtype())),
                    ("I", TemplateArg::Int(half as i32)),
                    ("HAS_S", TemplateArg::Bool(has_s)),
                ],
            )?
        }
        .remove(0);
        let mut final_shape = orig_shape;
        *final_shape.last_mut().unwrap() = half;
        out.reshape(&final_shape)
    }

    /// Unfused scalar softcapped SwiGLU with optional NVFP4 post-scale (for affine/fallback).
    pub fn moe_swiglu_scaled(
        &self,
        y: &Array,
        idx: &Array,
        gs: Option<&Array>,
        c: Option<&Array>,
        gcap: &Array,
        ucap: &Array,
    ) -> Result<Array> {
        let orig_shape = y.shape().to_vec();
        let two_i = *orig_shape.last().unwrap();
        let half = two_i / 2;
        let pairs = y.numel() / two_i;
        let has_s = gs.is_some();
        let gcap1 = gcap.reshape(&[1])?;
        let ucap1 = ucap.reshape(&[1])?;
        let idx_flat = idx.reshape(&[idx.numel()])?;
        let idx_u32 = if idx_flat.dtype() == MlxDType::U32 {
            idx_flat
        } else {
            idx_flat.cast(MlxDType::U32)?
        };
        let kernel = self.swiglu.get_or_init(|| {
            MetalKernel::new(
                &self.stream,
                "et_inkling2_moe_swiglu_scaled",
                &["y", "idx", "gs", "c", "gcap", "ucap"],
                &["act"],
                MOE_SWIGLU_SOURCE,
                NO_FMA_HEADER,
            )
            .expect("moe_swiglu_scaled kernel creation failed")
        });
        let out_shape = [pairs, half];
        let out = unsafe {
            kernel.call(
                &[
                    y.reshape(&[pairs, two_i])?,
                    idx_u32,
                    gs.cloned().unwrap_or_else(|| gcap1.clone()),
                    c.cloned().unwrap_or_else(|| gcap1.clone()),
                    gcap1,
                    ucap1,
                ],
                &[(&out_shape[..], y.dtype())],
                [half, pairs, 1],
                [half.min(256), 1, 1],
                &[
                    ("T", TemplateArg::DType(y.dtype())),
                    ("I", TemplateArg::Int(half as i32)),
                    ("HAS_S", TemplateArg::Bool(has_s)),
                ],
            )?
        }
        .remove(0);
        let mut final_shape = orig_shape;
        *final_shape.last_mut().unwrap() = half;
        out.reshape(&final_shape)
    }

    /// Fused sorted prefill combine: unsort via `inv`, multiply by `weights * s2`, sum top-k.
    pub fn moe_combine_sorted(
        &self,
        z: &Array,
        inv: &Array,
        weights: &Array,
        s2: Option<&Array>,
    ) -> Result<Array> {
        let (rows, top_k) = (weights.shape()[0], weights.shape()[1]);
        let hd = *z.shape().last().unwrap();
        let vec = 4usize;
        let has_s2 = s2.is_some();
        let kernel = self.combine_sorted.get_or_init(|| {
            MetalKernel::new(
                &self.stream,
                "et_inkling2_moe_combine_sorted",
                &["z", "inv", "weights", "s2"],
                &["out"],
                MOE_COMBINE_SORTED_SOURCE,
                NO_FMA_HEADER,
            )
            .expect("moe_combine_sorted kernel creation failed")
        });
        let wf = if weights.dtype() == MlxDType::F32 {
            weights.clone()
        } else {
            weights.cast(MlxDType::F32)?
        };
        let s2f = match s2 {
            Some(s) if s.dtype() == MlxDType::F32 => s.clone(),
            Some(s) => s.cast(MlxDType::F32)?,
            None => wf.clone(),
        };
        let inv_u32 = if inv.dtype() == MlxDType::U32 {
            inv.clone()
        } else {
            inv.cast(MlxDType::U32)?
        };
        let out_shape = [rows, hd];
        let mut outs = unsafe {
            kernel.call(
                &[z.reshape(&[rows * top_k, hd])?, inv_u32, wf, s2f],
                &[(&out_shape[..], z.dtype())],
                [hd / vec, rows, 1],
                [(hd / vec).min(256), 1, 1],
                &[
                    ("T", TemplateArg::DType(z.dtype())),
                    ("HD", TemplateArg::Int(hd as i32)),
                    ("TOPK", TemplateArg::Int(top_k as i32)),
                    ("VEC", TemplateArg::Int(vec as i32)),
                    ("HAS_S2", TemplateArg::Bool(has_s2)),
                ],
            )?
        };
        Ok(outs.remove(0))
    }

    /// Fused unsorted decode/verify MoE combine with optional NVFP4 `gs`/`c` gather.
    pub fn moe_combine_rows(
        &self,
        z: &Array,
        weights: &Array,
        inds: &Array,
        gs: Option<&Array>,
        c: Option<&Array>,
    ) -> Result<Array> {
        let (rows, top_k) = (weights.shape()[0], weights.shape()[1]);
        let hd = *z.shape().last().unwrap();
        let vec = 4usize;
        let has_s2 = gs.is_some();
        let kernel = self.combine_rows.get_or_init(|| {
            MetalKernel::new(
                &self.stream,
                "et_inkling2_moe_combine_rows",
                &["z", "weights", "inds", "gs", "c"],
                &["out"],
                MOE_COMBINE_ROWS_SOURCE,
                NO_FMA_HEADER,
            )
            .expect("moe_combine_rows kernel creation failed")
        });
        let inds_flat = inds.reshape(&[inds.numel()])?;
        let inds_u32 = if inds_flat.dtype() == MlxDType::U32 {
            inds_flat
        } else {
            inds_flat.cast(MlxDType::U32)?
        };
        let out_shape = [rows, hd];
        let mut outs = unsafe {
            kernel.call(
                &[
                    z.reshape(&[rows * top_k, hd])?,
                    weights.clone(),
                    inds_u32,
                    gs.cloned().unwrap_or_else(|| weights.clone()),
                    c.cloned().unwrap_or_else(|| weights.clone()),
                ],
                &[(&out_shape[..], z.dtype())],
                [hd / vec, rows, 1],
                [(hd / vec).min(256), 1, 1],
                &[
                    ("T", TemplateArg::DType(z.dtype())),
                    ("HD", TemplateArg::Int(hd as i32)),
                    ("TOPK", TemplateArg::Int(top_k as i32)),
                    ("VEC", TemplateArg::Int(vec as i32)),
                    ("HAS_S2", TemplateArg::Bool(has_s2)),
                ],
            )?
        };
        Ok(outs.remove(0))
    }
}
