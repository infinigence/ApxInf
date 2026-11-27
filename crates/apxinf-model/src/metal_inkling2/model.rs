//! Native Metal/MLX Inkling-2-Mini (`inkling2_loop`) execution engine.
//!
//! Implements prefill (NAX full attention, flash banded sliding attention,
//! sorted MoE with `prefill_moe_swiglu_v4` & `moe_combine_sorted`, fused prefill glue),
//! compiled single-token decode (`_step_pre` / eager `attend` / `_step_post`),
//! bit-exact multi-row MTP verify (`_xstep_pre` / `verify_attention_exact` / `_xstep_post`,
//! split submit every 4 visits, certified q8/g64 `screen_argmax`), and `LlmTrait`.

use super::{
    cache::{ConvSlotState, ModelCache, VisitCache},
    config::Config,
    draft::{DSparkDraft, DraftState},
    weights::{AttnLayoutWeights, Inkling2Weights, LayerWeights},
    Variant,
};
use crate::{
    llm_trait::{
        GeneratedToken, LlmInput, LlmTrait, SyncTokenSequence, TextCompilationScope,
        TextPreparationState, TextPreparationStatus, TokenSequence,
    },
    mlx_decode::{self, DecodeFinish, DeviceDecode, GreedySequence},
};
use apxinf_core::{
    Backend, Device, Error, Result, Tensor, TokenSampler, TokenSamplingParams,
};
use apxinf_loader::ModelConfig;
use apxinf_mlx::{
    banded_attn::BandedAttnKernels,
    exact_gemv::ExactGemvKernels,
    loop_glue::{ConvFusedEpilogue, LoopGlueKernels},
    moe_qmv::MoeKernels,
    Array, Compiled, MlxBackend, MlxDType, QuantMode, Stream,
};
use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    path::Path,
    rc::Rc,
    time::Instant,
};

const PREFILL_STEP: usize = 2048;
const COMPILED_MAX_TOKENS: usize = 16;
const VERIFY_MAX_ROWS: usize = 8;
const XATTN_MIN_KEYS: usize = 1024;
const XATTN_BLOCKS: usize = 128;
const V_SPLIT_VISITS: usize = 4;
const DEFAULT_MTP_DEPTH: usize = 2;

/// Shared handle to all custom Metal kernel caches.
struct Inkling2Kernels {
    banded: BandedAttnKernels,
    moe: MoeKernels,
    gemv: ExactGemvKernels,
    glue: LoopGlueKernels,
}

impl Inkling2Kernels {
    fn new(stream: &Stream) -> Self {
        Self {
            banded: BandedAttnKernels::new(stream.clone()),
            moe: MoeKernels::new(stream.clone()),
            gemv: ExactGemvKernels::new(stream.clone()),
            glue: LoopGlueKernels::new(stream.clone()),
        }
    }
}

/// Constants captured by compiled layer envelopes (evaluated device leaves).
struct SharedConstants {
    nvfp4_inv: Array,
    qk_clip: Array,
    r_clip: Array,
    one_f32: Array,
    gcap: Array,
    ucap: Array,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PreKey {
    layer_idx: usize,
    exact: bool,
    seq_len: usize,
    n_blocks: usize,
    has_messages: bool,
    has_tau: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct PostKey {
    layer_idx: usize,
    exact: bool,
    seq_len: usize,
    n_blocks: usize,
    has_messages: bool,
    fused_gate: bool,
}

#[derive(Clone, Debug, Default)]
pub struct MtpCycleStats {
    pub k: usize,
    pub accepted: usize,
    pub total_ms: f64,
    pub screen_fallback: bool,
}

#[derive(Clone, Debug, Default)]
pub struct MtpStats {
    pub depth: usize,
    pub cycles: Vec<MtpCycleStats>,
    pub screen_fallbacks: usize,
}

impl MtpStats {
    pub fn tokens_per_cycle(&self) -> Option<f64> {
        if self.cycles.is_empty() {
            return None;
        }
        let tot: usize = self.cycles.iter().map(|c| c.accepted + 1).sum();
        Some(tot as f64 / self.cycles.len() as f64)
    }

    pub fn median_cycle_ms(&self) -> Option<f64> {
        if self.cycles.is_empty() {
            return None;
        }
        let mut ms: Vec<f64> = self.cycles.iter().map(|c| c.total_ms).collect();
        ms.sort_by(|a, b| a.total_cmp(b));
        Some(ms[ms.len() / 2])
    }
}

/// Set `APXINF_INKLING2_DECODE_TRACE=1` to log which decode path each request
/// takes and what the MTP verifier accepted. Off by default: the line is
/// per-request, not per-token.
fn decode_trace_enabled() -> bool {
    std::env::var_os("APXINF_INKLING2_DECODE_TRACE").is_some_and(|value| value == "1")
}

/// Output of `generate_greedy_benchmark` (with MTP cycle and timing statistics).
#[derive(Clone, Debug)]
pub struct BenchmarkRun {
    pub prompt_tokens: usize,
    pub tokens: Vec<u32>,
    pub ttft_s: f64,
    pub decode_s: f64,
    pub tpot_ms: f64,
    pub peak_memory_gb: f64,
    pub mtp: Option<MtpStats>,
}

/// Native Metal/MLX Inkling-2-Mini model implementing `LlmTrait`.
pub struct MetalInkling2 {
    config: Config,
    variant: Variant,
    backend: MlxBackend,
    weights: Rc<Inkling2Weights>,
    consts: Rc<SharedConstants>,
    kernels: Rc<Inkling2Kernels>,
    schedule: Vec<(usize, usize)>,
    core_start: Option<usize>,
    cache: ModelCache,
    pre_compiled: RefCell<HashMap<PreKey, Compiled>>,
    post_compiled: RefCell<HashMap<PostKey, Compiled>>,
    lhs_cache: RefCell<HashMap<(usize, usize), Array>>,
    draft: Option<DSparkDraft>,
    draft_state: Option<DraftState>,
    mtp_enabled: bool,
    mtp_depth: usize,
    last_prefill_logits: Option<Array>,
    planned_tokens: Option<usize>,
    planned_max_new: usize,
    last_mtp_stats: Option<MtpStats>,
    /// Tokens from the last prefilled prompt, for prefix-cache reuse.
    cached_prompt_tokens: Vec<u32>,
    /// Number of tokens at the start of `cached_prompt_tokens` whose KV is
    /// still valid in `self.cache` (i.e. not overwritten by a later prefill).
    cached_kv_valid: usize,
    /// Total tokens ever prefilled into the current cache (including decode).
    cache_generation: usize,
    /// Tokens served from the KV prefix cache in the most recent prefill.
    last_prefix_reuse: usize,
}

fn log_tau(stream: &Stream, offset: usize, length: usize, floor: usize, alpha: f32) -> Result<Array> {
    let mut vals = Vec::with_capacity(length);
    let fl = floor as f32;
    for i in 0..length {
        let ratio = ((offset + i + 1) as f32 / fl).max(1.0);
        vals.push(1.0 + alpha * ratio.ln());
    }
    Array::from_f32(stream, &[1, length, 1, 1], &vals)
}

fn split_last_axis(x: &Array, cuts: &[usize]) -> Result<Vec<Array>> {
    let rank = x.shape().len();
    let last_dim = x.shape()[rank - 1];
    let mut out = Vec::with_capacity(cuts.len() + 1);
    let mut prev = 0;
    for &c in cuts {
        out.push(x.slice_axis(rank - 1, prev, c)?);
        prev = c;
    }
    out.push(x.slice_axis(rank - 1, prev, last_dim)?);
    Ok(out)
}

fn l2_clip_eager(x: &Array, scale: &Array, threshold: &Array) -> Result<Array> {
    let stream = x.stream();
    let xf = x.cast(MlxDType::F32)?.mul(&scale.cast(MlxDType::F32)?)?;
    let sq = xf.mul(&xf)?;
    let sum = sq.sum(&[-1], true)?;
    let nrm = sum.sqrt()?;
    let eps = Array::scalar(stream, 1e-6, MlxDType::F32)?;
    let one = Array::scalar(stream, 1.0, MlxDType::F32)?;
    let m = threshold.div(&nrm.add(&eps)?)?.minimum(&one)?;
    xf.mul(&m)?.cast(x.dtype())
}

fn softcapped_swiglu_eager(gate: &Array, up: &Array, gcap: &Array, ucap: &Array) -> Result<Array> {
    let dtype = gate.dtype();
    let g = gate.cast(MlxDType::F32)?;
    let u = up.cast(MlxDType::F32)?;
    let g_cap = g.div(gcap)?.tanh()?.mul(gcap)?;
    let u_cap = u.div(ucap)?.tanh()?.mul(ucap)?;
    g_cap.mul(&g.sigmoid()?)?.mul(&u_cap)?.cast(dtype)
}

fn static_conv_eager(
    x: &Array,
    w: &Array,
    state: &Array,
    scale: f32,
) -> Result<(Array, Array)> {
    let stream = x.stream();
    let xf = x.cast(MlxDType::F32)?;
    let k = *w.shape().last().unwrap();
    let length = xf.shape()[1];
    let xp = Array::concat(&[state, &xf], 1)?;
    let w32 = w.cast(MlxDType::F32)?;
    let mut acc = w32
        .slice_axis(1, 0, 1)?
        .reshape(&[1, 1, w.shape()[0]])?
        .mul(&xp.slice_axis(1, 0, length)?)?;
    for j in 1..k {
        let tap = w32.slice_axis(1, j, j + 1)?.reshape(&[1, 1, w.shape()[0]])?;
        acc = acc.add(&tap.mul(&xp.slice_axis(1, j, j + length)?)?)?;
    }
    let s = Array::scalar(stream, scale, MlxDType::F32)?;
    let out = acc
        .cast(w.dtype())?
        .cast(MlxDType::F32)?
        .add(&xf)?
        .mul(&s)?;
    Ok((out.cast(x.dtype())?, xp))
}

fn dynamic_conv_eager(
    x: &Array,
    dyn_taps: &Array,
    static_w: &Array,
    state: &Array,
    heads: usize,
    scale: f32,
) -> Result<(Array, Array)> {
    let stream = x.stream();
    let xf = x.cast(MlxDType::F32)?;
    let (b, length, c) = (xf.shape()[0], xf.shape()[1], xf.shape()[2]);
    let k = static_w.shape()[0];
    let hd = c / heads;
    let xp = Array::concat(&[state, &xf], 1)?;
    let dyn5 = dyn_taps
        .cast(MlxDType::F32)?
        .reshape(&[b, length, heads, 1, k])?;
    let sw32 = static_w.cast(MlxDType::F32)?;
    let mut acc: Option<Array> = None;
    for lag in 0..k {
        let start = k - 1 - lag;
        let win = xp
            .slice_axis(1, start, start + length)?
            .reshape(&[b, length, heads, hd])?;
        let sw_lag = sw32.slice_axis(0, lag, lag + 1)?.reshape(&[1, 1, heads, hd])?;
        let d_lag = dyn5
            .slice_axis(4, lag, lag + 1)?
            .reshape(&[b, length, heads, 1])?;
        let term = win.mul(&sw_lag.add(&d_lag)?)?;
        acc = Some(match acc {
            None => term,
            Some(prev) => prev.add(&term)?,
        });
    }
    let s = Array::scalar(stream, scale, MlxDType::F32)?;
    let out = acc.unwrap().reshape(&[b, length, c])?.add(&xf)?.mul(&s)?;
    Ok((out.cast(x.dtype())?, xp))
}

/// Core `_mix_messages` for decode / exact verify / prefill chunks.
#[allow(clippy::too_many_arguments)]
fn mix_messages(
    x: &Array,
    message: &Array,
    down_f32: &Array,
    gf_f32: &Array,
    w16: Option<&(Array, Array, Option<Array>)>,
    exact: bool,
    prefill: bool,
    dedup_x: bool,
    rms_norm: Option<(&Array, Option<(&Array, &Array)>, f32, MlxDType)>,
    kernels: &Inkling2Kernels,
) -> Result<Array> {
    let stream = x.stream();
    let (n, b, length, hidden) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
    let rows = b * length;
    let use16 = w16.is_some() && !prefill && (rows == 1 || exact);

    let z = if prefill && length > COMPILED_MAX_TOKENS {
        kernels.glue.prefill_mix_head_opt(x, message, 1e-6, dedup_x)?
    } else if rows == 1 || exact {
        kernels.glue.prefill_mix_head(x, message, 1e-6)?
    } else {
        let xu = x.fast_rms_norm_unweighted(1e-6)?;
        let mu = message.fast_rms_norm_unweighted(1e-6)?;
        let xu5 = xu.reshape(&[n, 1, b, length, hidden])?;
        let mu5 = mu.reshape(&[n, 1, b, length, hidden])?;
        Array::concat(&[&xu5, &mu5], 1)?
            .cast(MlxDType::BF16)?
            .cast(MlxDType::F32)?
            .reshape(&[2 * n, rows, hidden])?
    };

    if use16 {
        let (d16, g16, d_perm) = w16.unwrap();
        let low = if let Some(perm) = d_perm {
            kernels.gemv.loopmix_down_pair_perm(&z, d16, perm)?
        } else {
            kernels.gemv.loopmix_down_pair(&z, d16)?
        };
        return kernels.gemv.loopmix_gates_mix(&low, g16, x, message);
    }

    let rank = down_f32.shape()[3];
    let low = if exact && rows > 1 {
        let z4 = z.reshape(&[2 * n, rows, 1, hidden])?;
        let d4 = down_f32.reshape(&[2 * n, 1, hidden, rank])?;
        z4.matmul(&d4)?
            .reshape(&[n, 2, rows, rank])?
            .sum(&[1], false)?
            .cast(MlxDType::BF16)?
            .cast(MlxDType::F32)?
    } else {
        let d3 = down_f32.reshape(&[2 * n, hidden, rank])?;
        z.matmul(&d3)?
            .reshape(&[n, 2, rows, rank])?
            .sum(&[1], false)?
            .cast(MlxDType::BF16)?
            .cast(MlxDType::F32)?
    };

    if prefill && length > COMPILED_MAX_TOKENS {
        let gates = low.matmul(gf_f32)?;
        if let Some((w0, w1_w2, eps, out_dtype)) = rms_norm {
            return kernels
                .glue
                .prefill_mix_tail_rms(&gates, x, message, w0, eps, out_dtype, w1_w2);
        }
        return kernels.glue.prefill_mix_tail(&gates, x, message);
    }

    let gates = if exact && rows > 1 && n == 1 {
        let mut parts = Vec::with_capacity(rows);
        for r in 0..rows {
            parts.push(low.slice_axis(1, r, r + 1)?.contiguous()?.matmul(gf_f32)?);
        }
        let refs: Vec<&Array> = parts.iter().collect();
        Array::concat(&refs, 1)?
    } else if exact && rows > 1 {
        let gf4 = gf_f32.reshape(&[n, 1, rank, 2 * hidden])?;
        low.reshape(&[n, rows, 1, rank])?.matmul(&gf4)?
    } else {
        low.matmul(gf_f32)?
    }
    .reshape(&[n, b, length, 2 * hidden])?;

    let gain = gates.slice_axis(3, 0, hidden)?.tanh()?;
    let forget = gates.slice_axis(3, hidden, 2 * hidden)?.tanh()?;
    let one = Array::scalar(stream, 1.0, MlxDType::F32)?;
    one.sub(&forget)?.mul(x)?.add(&gain.mul(message)?)
}

/// Core `_pre` for decode (`!exact`, L=1) and exact verify (`exact`, 2<=L<=8), running inside `Compiled`.
#[allow(clippy::too_many_arguments)]
fn layer_pre_compiled_body(
    layer: &LayerWeights,
    consts: &SharedConstants,
    kernels: &Inkling2Kernels,
    h: &Array,
    blocks: &[Array],
    ks: &[Array; 3],
    vs: &[Array; 3],
    tau: Option<&Array>,
    messages: Option<&Array>,
    exact: bool,
) -> Result<Vec<Array>> {
    let dtype = h.dtype();
    let (b, length, hidden) = (h.shape()[0], h.shape()[1], h.shape()[2]);
    let h_n = kernels
        .glue
        .rms_norm_fused(h, None, 1e-6, MlxDType::F32, Some(MlxDType::F32))?;

    let (mut xq, mut xk, mut xv) = if layer.layer_idx == 0 {
        (h.clone(), None, None)
    } else {
        let q_w = layer.attn_input_query.as_ref().unwrap();
        let mixed = kernels
            .banded
            .attn_res_mix(blocks, &h_n, q_w, layer.attn_res_scale, true)?;
        if layer.sliding {
            let x0 = mixed.slice_axis(2, 0, 1)?.reshape(&[b, length, hidden])?;
            (x0, None, None)
        } else {
            let q0 = mixed.slice_axis(2, 0, 1)?.reshape(&[b, length, hidden])?;
            let k0 = mixed.slice_axis(2, 1, 2)?.reshape(&[b, length, hidden])?;
            let v0 = mixed.slice_axis(2, 2, 3)?.reshape(&[b, length, hidden])?;
            (q0, Some(k0), Some(v0))
        }
    };

    let mut x3_normed: Option<Array> = None;
    if let Some(msgs) = messages {
        let ctrl = layer.loop_ctrl.as_ref().unwrap();
        let x_in = if layer.sliding {
            let xq4 = xq.reshape(&[1, b, length, hidden])?;
            Array::concat(&[&xq4, &xq4, &xq4], 0)?
        } else {
            let q4 = xq.reshape(&[1, b, length, hidden])?;
            let k4 = xk.as_ref().unwrap().reshape(&[1, b, length, hidden])?;
            let v4 = xv.as_ref().unwrap().reshape(&[1, b, length, hidden])?;
            Array::concat(&[&q4, &k4, &v4], 0)?
        };
        let msg_in = msgs.slice_axis(2, 0, 3)?.transpose(&[2, 0, 1, 3])?;
        let mixed3 = mix_messages(
            &x_in,
            &msg_in,
            &ctrl.down_qkv_f32,
            &ctrl.gf_qkv_f32,
            ctrl.qkv_bf16.as_ref(),
            exact,
            false,
            false,
            None,
            kernels,
        )?;
        if layer.sliding {
            let attn_norm = match &layer.attn_layout {
                AttnLayoutWeights::Sliding { attn_norm, .. } => attn_norm,
                _ => unreachable!(),
            };
            let stacked = mixed3.reshape(&[3 * b, length, hidden])?;
            let normed = kernels
                .glue
                .rms_norm_fused(&stacked, Some(attn_norm), layer.rms_eps, dtype, None)?;
            xq = normed.slice_axis(0, 0, b)?;
            xk = Some(normed.slice_axis(0, b, 2 * b)?);
            xv = Some(normed.slice_axis(0, 2 * b, 3 * b)?);
            x3_normed = Some(normed);
        } else {
            let (q_norm, k_norm, v_norm) = match &layer.attn_layout {
                AttnLayoutWeights::Full {
                    q_norm,
                    k_norm,
                    v_norm,
                    ..
                } => (q_norm, k_norm, v_norm),
                _ => unreachable!(),
            };
            let mq = mixed3.slice_axis(0, 0, 1)?.reshape(&[b, length, hidden])?;
            let mk = mixed3.slice_axis(0, 1, 2)?.reshape(&[b, length, hidden])?;
            let mv = mixed3.slice_axis(0, 2, 3)?.reshape(&[b, length, hidden])?;
            xq = kernels
                .glue
                .rms_norm_fused(&mq, Some(q_norm), layer.rms_eps, dtype, None)?;
            xk = Some(
                kernels
                    .glue
                    .rms_norm_fused(&mk, Some(k_norm), layer.rms_eps, dtype, None)?,
            );
            xv = Some(
                kernels
                    .glue
                    .rms_norm_fused(&mv, Some(v_norm), layer.rms_eps, dtype, None)?,
            );
        }
    } else {
        match &layer.attn_layout {
            AttnLayoutWeights::Sliding { attn_norm, .. } => {
                xq = kernels
                    .glue
                    .rms_norm_fused(&xq, Some(attn_norm), layer.rms_eps, dtype, None)?;
            }
            AttnLayoutWeights::Full {
                q_norm,
                k_norm,
                v_norm,
                ..
            } => {
                xq = kernels
                    .glue
                    .rms_norm_fused(&xq, Some(q_norm), layer.rms_eps, dtype, None)?;
                xk = Some(kernels.glue.rms_norm_fused(
                    xk.as_ref().unwrap(),
                    Some(k_norm),
                    layer.rms_eps,
                    dtype,
                    None,
                )?);
                xv = Some(kernels.glue.rms_norm_fused(
                    xv.as_ref().unwrap(),
                    Some(v_norm),
                    layer.rms_eps,
                    dtype,
                    None,
                )?);
            }
        }
    }

    // Attention projections + fused short convs.
    let (q, k, v, r, gate, sink, xk_win, xv_win) = match &layer.attn_layout {
        AttnLayoutWeights::Sliding {
            in_proj,
            k_conv,
            v_conv,
            ..
        } => {
            let (q, k_raw, v_raw, r, gate, sink) = if xk.is_none() {
                let proj = in_proj.apply_exact(&xq, exact, &kernels.gemv)?;
                let mut parts = split_last_axis(&proj, &layer.cuts)?;
                let sink = parts.pop().unwrap();
                let gate = parts.pop().unwrap();
                let r = parts.pop().unwrap();
                let v = parts.pop().unwrap();
                let k = parts.pop().unwrap();
                let q = parts.pop().unwrap();
                (q, k, v, r, gate, sink)
            } else {
                let xs = match &x3_normed {
                    Some(x3) => x3.clone(),
                    None => Array::concat(&[&xq, xk.as_ref().unwrap(), xv.as_ref().unwrap()], 0)?,
                };
                let proj = if let Some(w_bf16) = in_proj.dense_weight() {
                    if w_bf16.dtype() == MlxDType::BF16 && b == 1 {
                        let x2 = xs.reshape(&[3 * length, hidden])?;
                        kernels
                            .gemv
                            .wide_gemv(w_bf16, &x2)?
                            .reshape(&[3, length, w_bf16.shape()[0]])?
                    } else if exact && length > 1 {
                        let mut cols = Vec::with_capacity(length);
                        for t in 0..length {
                            cols.push(in_proj.apply(&xs.slice_axis(1, t, t + 1)?.contiguous()?)?);
                        }
                        let refs: Vec<&Array> = cols.iter().collect();
                        Array::concat(&refs, 1)?
                    } else {
                        in_proj.apply(&xs)?
                    }
                } else if exact && length > 1 {
                    let mut cols = Vec::with_capacity(length);
                    for t in 0..length {
                        cols.push(in_proj.apply(&xs.slice_axis(1, t, t + 1)?.contiguous()?)?);
                    }
                    let refs: Vec<&Array> = cols.iter().collect();
                    Array::concat(&refs, 1)?
                } else {
                    in_proj.apply(&xs)?
                };
                let pq = proj.slice_axis(0, 0, b)?;
                let pk = proj.slice_axis(0, b, 2 * b)?;
                let pv = proj.slice_axis(0, 2 * b, 3 * b)?;
                let mut parts = split_last_axis(&pq, &layer.cuts)?;
                let sink = parts.pop().unwrap();
                let gate = parts.pop().unwrap();
                let r = parts.pop().unwrap();
                let _v_unused = parts.pop().unwrap();
                let _k_unused = parts.pop().unwrap();
                let q = parts.pop().unwrap();
                let k = pk.slice_axis(2, layer.cuts[0], layer.cuts[1])?;
                let v = pv.slice_axis(2, layer.cuts[1], layer.cuts[2])?;
                (q, k, v, r, gate, sink)
            };
            let (k, xk_win) = kernels.glue.conv_fused(
                &k_raw,
                k_conv,
                ks,
                layer.sconv_scale,
                None,
                0,
                ConvFusedEpilogue::None,
            )?;
            let (v, xv_win) = kernels.glue.conv_fused(
                &v_raw,
                v_conv,
                vs,
                layer.sconv_scale,
                None,
                0,
                ConvFusedEpilogue::None,
            )?;
            (q, k, v, r, gate, sink, xk_win, xv_win)
        }
        AttnLayoutWeights::Full {
            q_proj,
            k_proj,
            v_proj,
            k_static,
            v_static,
            ..
        } => {
            let qp = q_proj.apply_exact(&xq, exact, &kernels.gemv)?;
            let mut parts = split_last_axis(&qp, &layer.cuts)?;
            let sink = parts.pop().unwrap();
            let gate = parts.pop().unwrap();
            let r = parts.pop().unwrap();
            let q = parts.pop().unwrap();
            let kp = k_proj.apply_exact(xk.as_ref().unwrap(), exact, &kernels.gemv)?;
            let vp = v_proj.apply_exact(xv.as_ref().unwrap(), exact, &kernels.gemv)?;
            let k_x = kp.slice_axis(2, 0, layer.dkv)?;
            let k_dyn = kp.slice_axis(2, layer.dkv, kp.shape()[2])?;
            let v_x = vp.slice_axis(2, 0, layer.dkv)?;
            let v_dyn = vp.slice_axis(2, layer.dkv, vp.shape()[2])?;
            let (k, xk_win) = kernels.glue.conv_fused(
                &k_x,
                k_static,
                ks,
                layer.dconv_scale,
                Some(&k_dyn),
                layer.n_kv,
                ConvFusedEpilogue::None,
            )?;
            let (v, xv_win) = kernels.glue.conv_fused(
                &v_x,
                v_static,
                vs,
                layer.dconv_scale,
                Some(&v_dyn),
                layer.n_kv,
                ConvFusedEpilogue::None,
            )?;
            (q, k, v, r, gate, sink, xk_win, xv_win)
        }
    };

    let q4 = q.reshape(&[b, length, layer.n_heads, layer.head_dim])?;
    let k4 = k.reshape(&[b, length, layer.n_kv, layer.head_dim])?;
    let r4 = r.reshape(&[b, length, layer.n_heads, layer.d_rel])?;
    let q_c = kernels
        .glue
        .l2_clip_fused(&q4, &layer.q_scale, &consts.qk_clip, tau)?;
    let k_c = kernels
        .glue
        .l2_clip_fused(&k4, &consts.one_f32, &consts.qk_clip, None)?;
    let r_c = kernels
        .glue
        .l2_clip_fused(&r4, &layer.r_scale, &consts.r_clip, tau)?;
    let v4 = v.reshape(&[b, length, layer.n_kv, layer.head_dim])?;
    let v_n = kernels
        .glue
        .rms_norm_fused(&v4, None, 1e-6, dtype, Some(MlxDType::F32))?;
    let rel = r_c.matmul(&layer.rel_proj.cast(r_c.dtype())?)?;

    let q_t = q_c.transpose(&[0, 2, 1, 3])?;
    let k_t = k_c.transpose(&[0, 2, 1, 3])?;
    let v_t = v_n.transpose(&[0, 2, 1, 3])?;

    if !exact && length == 1 {
        Ok(vec![
            q_t,
            k_t,
            v_t,
            rel,
            gate,
            sink,
            h_n,
            xk_win.slice_axis(1, 1, 2)?,
            xk_win.slice_axis(1, 2, 3)?,
            xk_win.slice_axis(1, 3, 4)?,
            xv_win.slice_axis(1, 1, 2)?,
            xv_win.slice_axis(1, 2, 3)?,
            xv_win.slice_axis(1, 3, 4)?,
        ])
    } else {
        Ok(vec![q_t, k_t, v_t, rel, gate, sink, h_n, xk_win, xv_win])
    }
}

/// Core `_post` for decode (`!exact`, L=1) and exact verify (`exact`, 2<=L<=8), running inside `Compiled`.
#[allow(clippy::too_many_arguments)]
fn layer_post_compiled_body(
    layer: &LayerWeights,
    consts: &SharedConstants,
    kernels: &Inkling2Kernels,
    out: &Array,
    gate: &Array,
    h: &Array,
    mlp_blocks: &[Array],
    s2: &[Array; 3],
    s3: &[Array; 3],
    messages: Option<&Array>,
    exact: bool,
    fused_gate: bool,
    lhs13: &Array,
    lhs2: &Array,
) -> Result<Vec<Array>> {
    let dtype = h.dtype();
    let (b, length, hidden) = (h.shape()[0], h.shape()[1], h.shape()[2]);

    // Attention output gate + o_proj.
    let gated = if fused_gate {
        out.clone()
    } else {
        let flat = out
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[b, length, layer.n_heads * layer.head_dim])?;
        flat.cast(MlxDType::F32)?
            .mul(&gate.cast(MlxDType::F32)?.sigmoid()?)?
            .cast(gate.dtype())?
    };
    let a = layer.o_proj.apply_exact(&gated, exact, &kernels.gemv)?;

    // Fused attn_conv + residual start.
    let (residual, x2_win) = kernels.glue.conv_fused(
        &a,
        &layer.attn_conv,
        s2,
        layer.conv_scale,
        None,
        0,
        ConvFusedEpilogue::ResidualStart {
            h,
            boundary: layer.is_boundary,
        },
    )?;
    let r_n = residual.fast_rms_norm_unweighted(1e-6)?;

    // MLP AttnRes + optional loop message mixing.
    let mut y = kernels
        .banded
        .attn_res_mix(
            mlp_blocks,
            &r_n,
            &layer.mlp_input_query,
            layer.attn_res_scale,
            true,
        )?
        .slice_axis(2, 0, 1)?
        .reshape(&[b, length, hidden])?;
    if let Some(msgs) = messages {
        let ctrl = layer.loop_ctrl.as_ref().unwrap();
        let x_in = y.reshape(&[1, b, length, hidden])?;
        let msg_in = msgs
            .slice_axis(2, 3, 4)?
            .reshape(&[1, b, length, hidden])?;
        y = mix_messages(
            &x_in,
            &msg_in,
            &ctrl.down_mlp_f32,
            &ctrl.gf_mlp_f32,
            ctrl.mlp_bf16.as_ref(),
            exact,
            false,
            false,
            None,
            kernels,
        )?
        .reshape(&[b, length, hidden])?;
    }
    let xm = kernels
        .glue
        .rms_norm_fused(&y, Some(&layer.mlp_norm), layer.rms_eps, dtype, None)?;

    // MoE router + experts (unsorted on decode and exact verify).
    let flat = xm.reshape(&[b * length, hidden])?;
    let router_logits = if exact && length > 1 {
        if let Some(rw16) = &layer.router_weight_bf16 {
            kernels.gemv.multirow_gemv_router16(rw16, &flat)?
        } else {
            kernels
                .gemv
                .multirow_gemv(&layer.router_weight, &flat, true)?
        }
    } else {
        let rwt = layer.router_weight.transpose(&[1, 0])?;
        flat.cast(MlxDType::F32)?.matmul(&rwt)?
    };
    let (inds, weights) =
        kernels
            .moe
            .route_topk(&router_logits, &layer.router_bias, layer.top_k, layer.route_scale)?;

    let dedup = exact
        && length > 1
        && MoeKernels::dedup_qmv_supported(layer.w13.mode, &layer.w13.weight, layer.w13.bits, &flat)
        && MoeKernels::dedup_qmv_supported(layer.w2.mode, &layer.w2.weight, layer.w2.bits, &flat);
    let r2_epi = (dedup || (!exact && length == 1))
        && MoeKernels::dedup_qmv_supported(layer.w13.mode, &layer.w13.weight, layer.w13.bits, &flat);

    let act = if r2_epi {
        let nv13 = layer.w13.mode == QuantMode::Nvfp4;
        kernels.moe.gather_qmv_swiglu(
            &flat,
            &layer.w13.weight,
            &layer.w13.scales,
            &inds,
            layer.w13.bits,
            layer.w13.group_size,
            if nv13 {
                layer.w13.global_scale.as_ref()
            } else {
                None
            },
            if nv13 { Some(&consts.nvfp4_inv) } else { None },
            &consts.gcap,
            &consts.ucap,
        )?
    } else {
        let xe = flat.reshape(&[b * length, 1, 1, hidden])?;
        let y13 = layer
            .w13
            .gather_qmm(&xe, Some(lhs13), &inds, false)?;
        if layer.w13.mode == QuantMode::Nvfp4 {
            kernels.moe.moe_swiglu_scaled(
                &y13,
                &inds,
                layer.w13.global_scale.as_ref(),
                Some(&consts.nvfp4_inv),
                &consts.gcap,
                &consts.ucap,
            )?
        } else {
            let two_i = *y13.shape().last().unwrap();
            let s13 = layer.w13.post_scale(&inds, &consts.nvfp4_inv)?;
            let y13 = if let Some(sc) = s13 {
                let mut sc_sh = sc.shape().to_vec();
                sc_sh.push(1);
                sc_sh.push(1);
                y13.mul(&sc.reshape(&sc_sh)?.cast(y13.dtype())?)?
            } else {
                y13
            };
            let g = y13.slice_strided(
                &[0, 0, 0, 0],
                &[y13.shape()[0], y13.shape()[1], y13.shape()[2], two_i],
                &[1, 1, 1, 2],
            )?;
            let u = y13.slice_strided(
                &[0, 0, 0, 1],
                &[y13.shape()[0], y13.shape()[1], y13.shape()[2], two_i],
                &[1, 1, 1, 2],
            )?;
            softcapped_swiglu_eager(&g, &u, &consts.gcap, &consts.ucap)?
        }
    };

    let z = if dedup {
        let inter = *act.shape().last().unwrap();
        let act2 = act.reshape(&[b * length * layer.top_k, inter])?;
        kernels.moe.gather_qmv_dedup(
            &act2,
            &layer.w2.weight,
            &layer.w2.scales,
            &inds,
            layer.w2.bits,
            layer.w2.group_size,
            true,
        )?
    } else {
        layer.w2.gather_qmm(&act, Some(lhs2), &inds, false)?
    };

    let nv2 = layer.w2.mode == QuantMode::Nvfp4;
    let m = kernels
        .moe
        .moe_combine_rows(
            &z,
            &weights,
            &inds,
            if nv2 {
                layer.w2.global_scale.as_ref()
            } else {
                None
            },
            if nv2 { Some(&consts.nvfp4_inv) } else { None },
        )?
        .reshape(&[b, length, hidden])?;

    // Fused mlp_conv + residual output.
    let (out_h, x3_win) = kernels.glue.conv_fused(
        &m,
        &layer.mlp_conv,
        s3,
        layer.conv_scale,
        None,
        0,
        ConvFusedEpilogue::ResidualOut {
            res: &residual,
            out_dtype: dtype,
        },
    )?;

    if !exact && length == 1 {
        Ok(vec![
            out_h,
            x2_win.slice_axis(1, 1, 2)?,
            x2_win.slice_axis(1, 2, 3)?,
            x2_win.slice_axis(1, 3, 4)?,
            x3_win.slice_axis(1, 1, 2)?,
            x3_win.slice_axis(1, 2, 3)?,
            x3_win.slice_axis(1, 3, 4)?,
        ])
    } else {
        Ok(vec![out_h, x2_win, x3_win])
    }
}

impl MetalInkling2 {
    pub fn load(
        dir: &Path,
        device: Device,
        backend: MlxBackend,
        variant_opt: Option<&str>,
    ) -> Result<Self> {
        if !matches!(device, Device::Metal(_)) {
            return Err(Error::UnsupportedDevice(device));
        }
        let config = Config::from_json(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let variant = match variant_opt {
            None | Some("auto") => {
                if config.is_mlx_quantized() {
                    Variant::Oq4eMtp
                } else {
                    Variant::Nvfp4Mixed
                }
            }
            Some(other) => Variant::parse(Some(other))?,
        };
        let stream = backend.stream().clone();
        let weights = Rc::new(Inkling2Weights::load(&stream, dir, &config)?);
        let kernels = Rc::new(Inkling2Kernels::new(&stream));
        let consts = Rc::new(SharedConstants {
            nvfp4_inv: weights.nvfp4_inv.clone(),
            qk_clip: weights.qk_clip.clone(),
            r_clip: weights.r_clip.clone(),
            one_f32: weights.one_f32.clone(),
            gcap: weights.gcap.clone(),
            ucap: weights.ucap.clone(),
        });

        let schedule = config.layer_schedule();
        let n_kv_per_layer: Vec<usize> = weights.layers.iter().map(|l| l.n_kv).collect();
        let head_dim_per_layer: Vec<usize> = weights.layers.iter().map(|l| l.head_dim).collect();
        let cache = ModelCache::new(&schedule, &n_kv_per_layer, &head_dim_per_layer);
        let core_start = config.text_config.loop_config.as_ref().map(|lc| lc.core_start);

        let draft = DSparkDraft::load_if_present(&stream, dir, &config, &weights)?;
        let disable_mtp = std::env::var_os("APXINF_INKLING2_DISABLE_MTP")
            .is_some_and(|v| v == "1");
        let mtp_enabled = draft.is_some() && !disable_mtp;

        // Warm screen_argmax Metal pipelines for 2- and 3-row verify shapes at load time.
        if let (Some(screen), Some(w_bf16)) = (&weights.verify_screen, weights.lm_head.dense_weight()) {
            let h_dim = w_bf16.shape()[1];
            for rows in [2usize, 3usize] {
                let z = Array::zeros(&stream, &[rows, h_dim], w_bf16.dtype())?;
                let out = kernels.gemv.screen_argmax(w_bf16, &z, screen)?;
                out.eval()?;
            }
        }

        Ok(Self {
            config,
            variant,
            backend,
            weights,
            consts,
            kernels,
            schedule,
            core_start,
            cache,
            pre_compiled: RefCell::new(HashMap::new()),
            post_compiled: RefCell::new(HashMap::new()),
            lhs_cache: RefCell::new(HashMap::new()),
            draft,
            draft_state: None,
            mtp_enabled,
            mtp_depth: DEFAULT_MTP_DEPTH,
            last_prefill_logits: None,
            planned_tokens: None,
            planned_max_new: 0,
            last_mtp_stats: None,
            cached_prompt_tokens: Vec::new(),
            cached_kv_valid: 0,
            cache_generation: 0,
            last_prefix_reuse: 0,
        })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn variant(&self) -> Variant {
        self.variant
    }

    pub fn has_draft(&self) -> bool {
        self.draft.is_some()
    }

    pub fn set_mtp_enabled(&mut self, on: bool) {
        self.mtp_enabled = on && self.draft.is_some();
    }

    pub fn set_mtp_depth(&mut self, depth: usize) {
        if let Some(d) = &self.draft {
            self.mtp_depth = depth.clamp(1, d.depth);
        }
    }

    pub fn last_mtp_stats(&self) -> Option<&MtpStats> {
        self.last_mtp_stats.as_ref()
    }

    fn stream(&self) -> &Stream {
        self.backend.stream()
    }

    fn lhs_indices(&self, rows: usize, per_row: usize) -> Result<Array> {
        let mut cache = self.lhs_cache.borrow_mut();
        if let Some(a) = cache.get(&(rows, per_row)) {
            return Ok(a.clone());
        }
        let data: Vec<u32> = (0..(rows * per_row) as u32).collect();
        let a = Array::from_u32(self.stream(), &[rows, per_row], &data)?;
        a.eval()?;
        cache.insert((rows, per_row), a.clone());
        Ok(a)
    }

    /// Run a single-token decode or exact multi-row verify visit using cached `Compiled` envelopes.
    #[allow(clippy::too_many_arguments)]
    fn run_layer_compiled(
        &self,
        layer: &Rc<LayerWeights>,
        h: &Array,
        blocks: &mut Vec<Array>,
        visit_cache: &mut VisitCache,
        bank: Option<(&[Array], &Array)>,
        tau: Option<&Array>,
        exact: bool,
    ) -> Result<Array> {
        let stream = self.stream();
        let length = h.shape()[1];
        let hidden = h.shape()[2];
        let s0 = visit_cache.conv[0].as_rows(stream, layer.dkv)?;
        let s1 = visit_cache.conv[1].as_rows(stream, layer.dkv)?;
        let s2 = visit_cache.conv[2].as_rows(stream, hidden)?;
        let s3 = visit_cache.conv[3].as_rows(stream, hidden)?;

        let messages = if let Some((b_blocks, b_part)) = bank {
            let lmq = layer.loop_message_query.as_ref().unwrap();
            Some(
                self.kernels
                    .banded
                    .attn_res_mix(b_blocks, b_part, lmq, layer.attn_res_scale, true)?,
            )
        } else {
            None
        };

        // 1. Compiled `_pre`
        let pre_key = PreKey {
            layer_idx: layer.layer_idx,
            exact,
            seq_len: length,
            n_blocks: blocks.len(),
            has_messages: messages.is_some(),
            has_tau: tau.is_some(),
        };
        let mut pre_inputs = Vec::with_capacity(9 + blocks.len());
        pre_inputs.push(h.clone());
        pre_inputs.extend(s0.iter().cloned());
        pre_inputs.extend(s1.iter().cloned());
        pre_inputs.extend(blocks.iter().cloned());
        if let Some(t) = tau {
            pre_inputs.push(t.clone());
        }
        if let Some(m) = &messages {
            pre_inputs.push(m.clone());
        }
        let mut pre_outs = {
            let mut map = self.pre_compiled.borrow_mut();
            let compiled = if let Some(c) = map.get(&pre_key) {
                c
            } else {
                let layer_c = layer.clone();
                let consts_c = self.consts.clone();
                let kernels_c = self.kernels.clone();
                let n_blk = pre_key.n_blocks;
                let has_tau = pre_key.has_tau;
                let has_msg = pre_key.has_messages;
                let n_out = if !exact && length == 1 { 13 } else { 9 };
                let c = Compiled::new(stream, n_out, move |inp| {
                    let h_in = &inp[0];
                    let ks_in = [inp[1].clone(), inp[2].clone(), inp[3].clone()];
                    let vs_in = [inp[4].clone(), inp[5].clone(), inp[6].clone()];
                    let blks = &inp[7..7 + n_blk];
                    let mut idx = 7 + n_blk;
                    let tau_in = if has_tau {
                        let t = &inp[idx];
                        idx += 1;
                        Some(t)
                    } else {
                        None
                    };
                    let msg_in = if has_msg { Some(&inp[idx]) } else { None };
                    layer_pre_compiled_body(
                        &layer_c, &consts_c, &kernels_c, h_in, blks, &ks_in, &vs_in, tau_in,
                        msg_in, exact,
                    )
                })?;
                map.entry(pre_key).or_insert(c)
            };
            compiled.call(&pre_inputs)?
        };

        let q = pre_outs[0].clone();
        let k = pre_outs[1].clone();
        let v = pre_outs[2].clone();
        let rel = pre_outs[3].clone();
        let gate = pre_outs[4].clone();
        let sink = pre_outs[5].clone();
        let h_n = pre_outs[6].clone();

        // 2. Eager KV cache in-place publish (donation-safe, outside `Compiled`) + attention.
        let (keys, values) = visit_cache.kv.update_and_fetch(&k, &v)?;
        let (attn_out, fused_gate) = if !exact && length == 1 {
            let o = self.kernels.banded.decode_attention(
                &q,
                &keys,
                &values,
                &rel,
                &sink,
                layer.attn_scale,
                layer.window,
            )?;
            (o, false)
        } else {
            let base = keys.shape()[2] - length;
            let n_first = if layer.window > 0 {
                (base + 1).min(layer.window)
            } else {
                base + 1
            };
            if n_first >= XATTN_MIN_KEYS {
                let o = self.kernels.banded.verify_attention_exact(
                    &q,
                    &keys,
                    &values,
                    &rel,
                    &sink,
                    layer.attn_scale,
                    layer.window,
                    base,
                    XATTN_BLOCKS,
                    true,
                    Some(&gate),
                    !layer.sliding,
                )?;
                (o, true)
            } else {
                let mut outs = Vec::with_capacity(length);
                for t in 0..length {
                    let s_t = base + t + 1;
                    let qt = q.slice_axis(2, t, t + 1)?;
                    let kt = keys.slice_axis(2, 0, s_t)?;
                    let vt = values.slice_axis(2, 0, s_t)?;
                    let rt = rel.slice_axis(1, t, t + 1)?;
                    let st = sink.slice_axis(1, t, t + 1)?;
                    outs.push(self.kernels.banded.decode_attention(
                        &qt,
                        &kt,
                        &vt,
                        &rt,
                        &st,
                        layer.attn_scale,
                        layer.window,
                    )?);
                }
                let refs: Vec<&Array> = outs.iter().collect();
                (Array::concat(&refs, 2)?, false)
            }
        };

        if layer.is_boundary {
            blocks.push(h_n);
        }

        // 3. Compiled `_post`
        let post_key = PostKey {
            layer_idx: layer.layer_idx,
            exact,
            seq_len: length,
            n_blocks: blocks.len(),
            has_messages: messages.is_some(),
            fused_gate,
        };
        let mut post_inputs = Vec::with_capacity(10 + blocks.len());
        post_inputs.push(attn_out);
        post_inputs.push(gate);
        post_inputs.push(h.clone());
        post_inputs.extend(s2.iter().cloned());
        post_inputs.extend(s3.iter().cloned());
        post_inputs.extend(blocks.iter().cloned());
        if let Some(m) = &messages {
            post_inputs.push(m.clone());
        }

        let lhs13 = self.lhs_indices(length, 1)?;
        let lhs2 = self.lhs_indices(length, layer.top_k)?;
        let post_outs = {
            let mut map = self.post_compiled.borrow_mut();
            let compiled = if let Some(c) = map.get(&post_key) {
                c
            } else {
                let layer_c = layer.clone();
                let consts_c = self.consts.clone();
                let kernels_c = self.kernels.clone();
                let n_blk = post_key.n_blocks;
                let has_msg = post_key.has_messages;
                let n_out = if !exact && length == 1 { 7 } else { 3 };
                let c = Compiled::new(stream, n_out, move |inp| {
                    let out_in = &inp[0];
                    let gate_in = &inp[1];
                    let h_in = &inp[2];
                    let s2_in = [inp[3].clone(), inp[4].clone(), inp[5].clone()];
                    let s3_in = [inp[6].clone(), inp[7].clone(), inp[8].clone()];
                    let blks = &inp[9..9 + n_blk];
                    let msg_in = if has_msg { Some(&inp[9 + n_blk]) } else { None };
                    layer_post_compiled_body(
                        &layer_c,
                        &consts_c,
                        &kernels_c,
                        out_in,
                        gate_in,
                        h_in,
                        blks,
                        &s2_in,
                        &s3_in,
                        msg_in,
                        exact,
                        fused_gate,
                        &lhs13,
                        &lhs2,
                    )
                })?;
                map.entry(post_key).or_insert(c)
            };
            compiled.call(&post_inputs)?
        };

        let out_h = post_outs[0].clone();
        if !exact && length == 1 {
            visit_cache.conv[0] = ConvSlotState::Rows([
                pre_outs.remove(7),
                pre_outs.remove(7),
                pre_outs.remove(7),
            ]);
            visit_cache.conv[1] = ConvSlotState::Rows([
                pre_outs.remove(7),
                pre_outs.remove(7),
                pre_outs.remove(7),
            ]);
            visit_cache.conv[2] = ConvSlotState::Rows([
                post_outs[1].clone(),
                post_outs[2].clone(),
                post_outs[3].clone(),
            ]);
            visit_cache.conv[3] = ConvSlotState::Rows([
                post_outs[4].clone(),
                post_outs[5].clone(),
                post_outs[6].clone(),
            ]);
            visit_cache.verify_windows = None;
        } else {
            let xk_win = pre_outs[7].clone();
            let xv_win = pre_outs[8].clone();
            let x2_win = post_outs[1].clone();
            let x3_win = post_outs[2].clone();
            let windows = [xk_win, xv_win, x2_win, x3_win];
            for (slot, win) in windows.iter().enumerate() {
                visit_cache.conv[slot] = ConvSlotState::Rows([
                    win.slice_axis(1, length, length + 1)?,
                    win.slice_axis(1, length + 1, length + 2)?,
                    win.slice_axis(1, length + 2, length + 3)?,
                ]);
            }
            visit_cache.verify_windows = Some(windows);
        }
        Ok(out_h)
    }

    /// Eager prefill/general forward for a single layer visit (`L > 1` chunked prefill or unallocated KV).
    #[allow(clippy::too_many_arguments)]
    fn run_layer_eager(
        &self,
        layer: &LayerWeights,
        h: &Array,
        blocks: &mut Vec<Array>,
        visit_cache: &mut VisitCache,
        bank: Option<(&[Array], &Array)>,
        tau: Option<&Array>,
    ) -> Result<Array> {
        let stream = self.stream();
        let dtype = h.dtype();
        let (b, length, hidden) = (h.shape()[0], h.shape()[1], h.shape()[2]);
        let keep = layer.conv_k - 1;
        let pf = length > COMPILED_MAX_TOKENS;

        let s0 = visit_cache.conv[0].as_array(stream, keep, layer.dkv)?;
        let s1 = visit_cache.conv[1].as_array(stream, keep, layer.dkv)?;
        let s2 = visit_cache.conv[2].as_array(stream, keep, hidden)?;
        let s3 = visit_cache.conv[3].as_array(stream, keep, hidden)?;

        let messages = if let Some((b_blocks, b_part)) = bank {
            let lmq = layer.loop_message_query.as_ref().unwrap();
            Some(
                self.kernels
                    .banded
                    .attn_res_mix(b_blocks, b_part, lmq, layer.attn_res_scale, true)?,
            )
        } else {
            None
        };

        let h_n = if pf {
            self.kernels
                .glue
                .rms_norm_fused(h, None, 1e-6, MlxDType::F32, Some(MlxDType::F32))?
        } else {
            h.cast(MlxDType::F32)?.fast_rms_norm_unweighted(1e-6)?
        };

        let (mut xq, mut xk, mut xv, mixed_opt) = if layer.layer_idx == 0 {
            (h.clone(), None, None, None)
        } else {
            let q_w = layer.attn_input_query.as_ref().unwrap();
            let mixed = self
                .kernels
                .banded
                .attn_res_mix(blocks, &h_n, q_w, layer.attn_res_scale, true)?;
            if layer.sliding {
                let x0 = mixed.slice_axis(2, 0, 1)?.reshape(&[b, length, hidden])?;
                (x0, None, None, Some(mixed))
            } else {
                let q0 = mixed.slice_axis(2, 0, 1)?.reshape(&[b, length, hidden])?;
                let k0 = mixed.slice_axis(2, 1, 2)?.reshape(&[b, length, hidden])?;
                let v0 = mixed.slice_axis(2, 2, 3)?.reshape(&[b, length, hidden])?;
                (q0, Some(k0), Some(v0), Some(mixed))
            }
        };

        let mut x3_normed: Option<Array> = None;
        let mut p_tail_rms_done = false;
        if let Some(msgs) = &messages {
            let ctrl = layer.loop_ctrl.as_ref().unwrap();
            let p_head_view = pf && layer.layer_idx != 0;
            let x_in = if p_head_view {
                let mixed = mixed_opt.as_ref().unwrap();
                if layer.sliding {
                    mixed
                        .slice_axis(2, 0, 1)?
                        .transpose(&[2, 0, 1, 3])?
                        .broadcast_to(&[3, b, length, hidden])?
                } else {
                    mixed.slice_axis(2, 0, 3)?.transpose(&[2, 0, 1, 3])?
                }
            } else if layer.sliding {
                let xq4 = xq.reshape(&[1, b, length, hidden])?;
                Array::concat(&[&xq4, &xq4, &xq4], 0)?
            } else {
                let q4 = xq.reshape(&[1, b, length, hidden])?;
                let k4 = xk.as_ref().unwrap().reshape(&[1, b, length, hidden])?;
                let v4 = xv.as_ref().unwrap().reshape(&[1, b, length, hidden])?;
                Array::concat(&[&q4, &k4, &v4], 0)?
            };
            let msg_in = msgs.slice_axis(2, 0, 3)?.transpose(&[2, 0, 1, 3])?;
            let rms_norm_arg = if pf {
                p_tail_rms_done = true;
                match &layer.attn_layout {
                    AttnLayoutWeights::Sliding { attn_norm, .. } => {
                        Some((attn_norm, None, layer.rms_eps, dtype))
                    }
                    AttnLayoutWeights::Full {
                        q_norm,
                        k_norm,
                        v_norm,
                        ..
                    } => Some((q_norm, Some((k_norm, v_norm)), layer.rms_eps, dtype)),
                }
            } else {
                None
            };
            let mixed3 = mix_messages(
                &x_in,
                &msg_in,
                &ctrl.down_qkv_f32,
                &ctrl.gf_qkv_f32,
                None,
                false,
                pf,
                p_head_view && layer.sliding,
                rms_norm_arg,
                &self.kernels,
            )?;
            if p_tail_rms_done {
                if layer.sliding {
                    let x3 = mixed3.reshape(&[3 * b, length, hidden])?;
                    xq = x3.slice_axis(0, 0, b)?;
                    xk = Some(x3.slice_axis(0, b, 2 * b)?);
                    xv = Some(x3.slice_axis(0, 2 * b, 3 * b)?);
                    x3_normed = Some(x3);
                } else {
                    xq = mixed3.slice_axis(0, 0, 1)?.reshape(&[b, length, hidden])?;
                    xk = Some(mixed3.slice_axis(0, 1, 2)?.reshape(&[b, length, hidden])?);
                    xv = Some(mixed3.slice_axis(0, 2, 3)?.reshape(&[b, length, hidden])?);
                }
            } else {
                xq = mixed3.slice_axis(0, 0, 1)?.reshape(&[b, length, hidden])?;
                xk = Some(mixed3.slice_axis(0, 1, 2)?.reshape(&[b, length, hidden])?);
                xv = Some(mixed3.slice_axis(0, 2, 3)?.reshape(&[b, length, hidden])?);
            }
        }

        if !p_tail_rms_done {
            match &layer.attn_layout {
                AttnLayoutWeights::Sliding { attn_norm, .. } => {
                    if messages.is_none() {
                        xq = if pf {
                            self.kernels
                                .glue
                                .rms_norm_fused(&xq, Some(attn_norm), layer.rms_eps, dtype, None)?
                        } else {
                            xq.fast_rms_norm(&attn_norm.cast(xq.dtype())?, layer.rms_eps)?
                                .cast(dtype)?
                        };
                    } else if pf {
                        let stacked =
                            Array::concat(&[&xq, xk.as_ref().unwrap(), xv.as_ref().unwrap()], 0)?;
                        let x3 = self.kernels.glue.rms_norm_fused(
                            &stacked,
                            Some(attn_norm),
                            layer.rms_eps,
                            dtype,
                            None,
                        )?;
                        xq = x3.slice_axis(0, 0, b)?;
                        xk = Some(x3.slice_axis(0, b, 2 * b)?);
                        xv = Some(x3.slice_axis(0, 2 * b, 3 * b)?);
                        x3_normed = Some(x3);
                    } else {
                        let wn = attn_norm.cast(xq.dtype())?;
                        xq = xq.fast_rms_norm(&wn, layer.rms_eps)?.cast(dtype)?;
                        xk = Some(
                            xk.as_ref()
                                .unwrap()
                                .fast_rms_norm(&wn, layer.rms_eps)?
                                .cast(dtype)?,
                        );
                        xv = Some(
                            xv.as_ref()
                                .unwrap()
                                .fast_rms_norm(&wn, layer.rms_eps)?
                                .cast(dtype)?,
                        );
                    }
                }
                AttnLayoutWeights::Full {
                    q_norm,
                    k_norm,
                    v_norm,
                    ..
                } => {
                    if pf {
                        xq = self.kernels.glue.rms_norm_fused(
                            &xq,
                            Some(q_norm),
                            layer.rms_eps,
                            dtype,
                            None,
                        )?;
                        xk = Some(self.kernels.glue.rms_norm_fused(
                            xk.as_ref().unwrap(),
                            Some(k_norm),
                            layer.rms_eps,
                            dtype,
                            None,
                        )?);
                        xv = Some(self.kernels.glue.rms_norm_fused(
                            xv.as_ref().unwrap(),
                            Some(v_norm),
                            layer.rms_eps,
                            dtype,
                            None,
                        )?);
                    } else {
                        xq = xq
                            .fast_rms_norm(&q_norm.cast(xq.dtype())?, layer.rms_eps)?
                            .cast(dtype)?;
                        xk = Some(
                            xk.as_ref()
                                .unwrap()
                                .fast_rms_norm(&k_norm.cast(xq.dtype())?, layer.rms_eps)?
                                .cast(dtype)?,
                        );
                        xv = Some(
                            xv.as_ref()
                                .unwrap()
                                .fast_rms_norm(&v_norm.cast(xq.dtype())?, layer.rms_eps)?
                                .cast(dtype)?,
                        );
                    }
                }
            }
        }

        // Projections + short convs.
        let (q, k, v, r, gate, sink, ks_next, vs_next) = match &layer.attn_layout {
            AttnLayoutWeights::Sliding {
                in_proj,
                k_conv,
                v_conv,
                ..
            } => {
                let (q, k_raw, v_raw, r, gate, sink) = if xk.is_none() {
                    let proj = in_proj.apply(&xq)?;
                    let mut parts = split_last_axis(&proj, &layer.cuts)?;
                    let sink = parts.pop().unwrap();
                    let gate = parts.pop().unwrap();
                    let r = parts.pop().unwrap();
                    let v = parts.pop().unwrap();
                    let k = parts.pop().unwrap();
                    let q = parts.pop().unwrap();
                    (q, k, v, r, gate, sink)
                } else {
                    let xs = match &x3_normed {
                        Some(x3) => x3.clone(),
                        None => {
                            Array::concat(&[&xq, xk.as_ref().unwrap(), xv.as_ref().unwrap()], 0)?
                        }
                    };
                    let proj = in_proj.apply(&xs)?;
                    let pq = proj.slice_axis(0, 0, b)?;
                    let pk = proj.slice_axis(0, b, 2 * b)?;
                    let pv = proj.slice_axis(0, 2 * b, 3 * b)?;
                    let mut parts = split_last_axis(&pq, &layer.cuts)?;
                    let sink = parts.pop().unwrap();
                    let gate = parts.pop().unwrap();
                    let r = parts.pop().unwrap();
                    let _v_u = parts.pop().unwrap();
                    let _k_u = parts.pop().unwrap();
                    let q = parts.pop().unwrap();
                    let k = pk.slice_axis(2, layer.cuts[0], layer.cuts[1])?;
                    let v = pv.slice_axis(2, layer.cuts[1], layer.cuts[2])?;
                    (q, k, v, r, gate, sink)
                };
                let (k, ks_n) = if pf && b == 1 && length >= keep {
                    self.kernels
                        .glue
                        .prefill_static_conv(&k_raw, k_conv, &s0, layer.sconv_scale)?
                } else {
                    static_conv_eager(&k_raw, k_conv, &s0, layer.sconv_scale)?
                };
                let (v, vs_n) = if pf && b == 1 && length >= keep {
                    self.kernels
                        .glue
                        .prefill_static_conv(&v_raw, v_conv, &s1, layer.sconv_scale)?
                } else {
                    static_conv_eager(&v_raw, v_conv, &s1, layer.sconv_scale)?
                };
                (q, k, v, r, gate, sink, ks_n, vs_n)
            }
            AttnLayoutWeights::Full {
                q_proj,
                k_proj,
                v_proj,
                k_static,
                v_static,
                ..
            } => {
                let qp = q_proj.apply(&xq)?;
                let mut parts = split_last_axis(&qp, &layer.cuts)?;
                let sink = parts.pop().unwrap();
                let gate = parts.pop().unwrap();
                let r = parts.pop().unwrap();
                let q = parts.pop().unwrap();
                let kp = k_proj.apply(xk.as_ref().unwrap())?;
                let vp = v_proj.apply(xv.as_ref().unwrap())?;
                let k_x = kp.slice_axis(2, 0, layer.dkv)?;
                let k_dyn = kp.slice_axis(2, layer.dkv, kp.shape()[2])?;
                let v_x = vp.slice_axis(2, 0, layer.dkv)?;
                let v_dyn = vp.slice_axis(2, layer.dkv, vp.shape()[2])?;
                let (k, ks_n) = if pf && b == 1 && length >= keep {
                    self.kernels.glue.prefill_dynamic_conv(
                        &k_x,
                        &k_dyn,
                        k_static,
                        &s0,
                        layer.n_kv,
                        layer.dconv_scale,
                    )?
                } else {
                    dynamic_conv_eager(&k_x, &k_dyn, k_static, &s0, layer.n_kv, layer.dconv_scale)?
                };
                let (v, vs_n) = if pf && b == 1 && length >= keep {
                    self.kernels.glue.prefill_dynamic_conv(
                        &v_x,
                        &v_dyn,
                        v_static,
                        &s1,
                        layer.n_kv,
                        layer.dconv_scale,
                    )?
                } else {
                    dynamic_conv_eager(&v_x, &v_dyn, v_static, &s1, layer.n_kv, layer.dconv_scale)?
                };
                (q, k, v, r, gate, sink, ks_n, vs_n)
            }
        };

        let q4 = q.reshape(&[b, length, layer.n_heads, layer.head_dim])?;
        let k4 = k.reshape(&[b, length, layer.n_kv, layer.head_dim])?;
        let r4 = r.reshape(&[b, length, layer.n_heads, layer.d_rel])?;
        let (q_c, k_c, r_c, v_n) = if pf {
            let qc = self
                .kernels
                .glue
                .l2_clip_fused(&q4, &layer.q_scale, &self.consts.qk_clip, tau)?;
            let kc = self
                .kernels
                .glue
                .l2_clip_fused(&k4, &self.consts.one_f32, &self.consts.qk_clip, None)?;
            let rc = self
                .kernels
                .glue
                .l2_clip_fused(&r4, &layer.r_scale, &self.consts.r_clip, tau)?;
            let vn = self.kernels.glue.rms_norm_fused(
                &v.reshape(&[b, length, layer.n_kv, layer.head_dim])?,
                None,
                1e-6,
                dtype,
                Some(MlxDType::F32),
            )?;
            (qc, kc, rc, vn)
        } else {
            let mut qc = l2_clip_eager(&q4, &layer.q_scale, &self.consts.qk_clip)?;
            let kc = l2_clip_eager(&k4, &self.consts.one_f32, &self.consts.qk_clip)?;
            let mut rc = l2_clip_eager(&r4, &layer.r_scale, &self.consts.r_clip)?;
            if let Some(t) = &tau {
                qc = qc.cast(MlxDType::F32)?.mul(t)?.cast(qc.dtype())?;
                rc = rc.cast(MlxDType::F32)?.mul(t)?.cast(rc.dtype())?;
            }
            let vn = v
                .reshape(&[b, length, layer.n_kv, layer.head_dim])?
                .cast(MlxDType::F32)?
                .fast_rms_norm_unweighted(1e-6)?
                .cast(dtype)?;
            (qc, kc, rc, vn)
        };
        let rel = r_c.matmul(&layer.rel_proj.cast(r_c.dtype())?)?;

        let (keys, values) = visit_cache
            .kv
            .update_and_fetch(&k_c.transpose(&[0, 2, 1, 3])?, &v_n.transpose(&[0, 2, 1, 3])?)?;
        let attn_out = self.kernels.banded.banded_sink_attention(
            &q_c.transpose(&[0, 2, 1, 3])?,
            &keys,
            &values,
            &rel,
            &sink,
            layer.attn_scale,
            layer.window,
        )?;

        if layer.is_boundary {
            blocks.push(h_n);
        }

        let gated = if pf {
            self.kernels.glue.prefill_gate_out(&attn_out, &gate)?
        } else {
            let flat = attn_out
                .transpose(&[0, 2, 1, 3])?
                .reshape(&[b, length, layer.n_heads * layer.head_dim])?;
            flat.cast(MlxDType::F32)?
                .mul(&gate.cast(MlxDType::F32)?.sigmoid()?)?
                .cast(gate.dtype())?
        };
        let a = layer.o_proj.apply(&gated)?;

        let (a_c, s2_next) = if pf && b == 1 && length >= keep {
            self.kernels
                .glue
                .prefill_static_conv(&a, &layer.attn_conv, &s2, layer.conv_scale)?
        } else {
            static_conv_eager(&a, &layer.attn_conv, &s2, layer.conv_scale)?
        };
        let (residual, r_n) = if pf {
            self.kernels
                .glue
                .prefill_residual_rms(h, &a_c, layer.is_boundary, 1e-6)?
        } else {
            let res = if layer.is_boundary {
                a_c.cast(MlxDType::F32)?
            } else {
                h.cast(MlxDType::F32)?.add(&a_c.cast(MlxDType::F32)?)?
            };
            let rn = res.fast_rms_norm_unweighted(1e-6)?;
            (res, rn)
        };

        let mut y = self
            .kernels
            .banded
            .attn_res_mix(
                blocks,
                &r_n,
                &layer.mlp_input_query,
                layer.attn_res_scale,
                true,
            )?
            .slice_axis(2, 0, 1)?
            .reshape(&[b, length, hidden])?;
        let mut mlp_tail_rms = false;
        if let Some(msgs) = &messages {
            let ctrl = layer.loop_ctrl.as_ref().unwrap();
            let x_in = y.reshape(&[1, b, length, hidden])?;
            let msg_in = msgs
                .slice_axis(2, 3, 4)?
                .reshape(&[1, b, length, hidden])?;
            let rms_arg = if pf {
                mlp_tail_rms = true;
                Some((&layer.mlp_norm, None, layer.rms_eps, dtype))
            } else {
                None
            };
            y = mix_messages(
                &x_in,
                &msg_in,
                &ctrl.down_mlp_f32,
                &ctrl.gf_mlp_f32,
                None,
                false,
                pf,
                false,
                rms_arg,
                &self.kernels,
            )?
            .reshape(&[b, length, hidden])?;
        }
        let xm = if mlp_tail_rms {
            y
        } else if pf {
            self.kernels
                .glue
                .rms_norm_fused(&y, Some(&layer.mlp_norm), layer.rms_eps, dtype, None)?
        } else {
            y.fast_rms_norm(&layer.mlp_norm.cast(y.dtype())?, layer.rms_eps)?
                .cast(dtype)?
        };

        // MoE (sorted when rows * top_k >= 64).
        let flat = xm.reshape(&[b * length, hidden])?;
        let rwt = layer.router_weight.transpose(&[1, 0])?;
        let router_logits = flat.cast(MlxDType::F32)?.matmul(&rwt)?;
        let (inds, weights) = self.kernels.moe.route_topk(
            &router_logits,
            &layer.router_bias,
            layer.top_k,
            layer.route_scale,
        )?;
        let do_sort = inds.numel() >= 64;
        let (xe, idx, inv) = if do_sort {
            let flat_idx = inds.reshape(&[inds.numel()])?;
            let order = flat_idx.argsort(0)?;
            let inv_order = order.argsort(0)?;
            let tok_idx = order.floor_divide(&Array::from_u32(stream, &[], &[layer.top_k as u32])?)?;
            let xe_s = flat
                .take(&tok_idx, 0)?
                .reshape(&[inds.numel(), 1, hidden])?;
            let idx_s = flat_idx.take(&order, 0)?;
            (xe_s, idx_s, Some(inv_order))
        } else {
            (flat.reshape(&[b * length, 1, 1, hidden])?, inds.clone(), None)
        };

        let y13 = layer.w13.gather_qmm(&xe, None, &idx, do_sort)?;
        let act = if pf && do_sort {
            let nv13 = layer.w13.mode == QuantMode::Nvfp4;
            self.kernels.moe.prefill_moe_swiglu_v4(
                &y13,
                &idx,
                if nv13 {
                    layer.w13.global_scale.as_ref()
                } else {
                    None
                },
                if nv13 { Some(&self.consts.nvfp4_inv) } else { None },
                &self.consts.gcap,
                &self.consts.ucap,
            )?
        } else {
            let y13 = if let Some(sc) = layer.w13.post_scale(&idx, &self.consts.nvfp4_inv)? {
                let mut sc_sh = sc.shape().to_vec();
                sc_sh.push(1);
                sc_sh.push(1);
                y13.mul(&sc.reshape(&sc_sh)?.cast(y13.dtype())?)?
            } else {
                y13
            };
            let two_i = *y13.shape().last().unwrap();
            let sh = y13.shape();
            let mut st0 = vec![0usize; sh.len()];
            let mut st1 = vec![0usize; sh.len()];
            st1[sh.len() - 1] = 1;
            let mut strides = vec![1usize; sh.len()];
            strides[sh.len() - 1] = 2;
            let mut stop = sh.to_vec();
            stop[sh.len() - 1] = two_i;
            let g = y13.slice_strided(&st0, &stop, &strides)?;
            st0[sh.len() - 1] = 1;
            let u = y13.slice_strided(&st1, &stop, &strides)?;
            softcapped_swiglu_eager(&g, &u, &self.consts.gcap, &self.consts.ucap)?
        };

        let z = layer.w2.gather_qmm(&act, None, &idx, do_sort)?;
        let m_flat = if pf && do_sort {
            let s2_opt = layer.w2.post_scale(&inds, &self.consts.nvfp4_inv)?;
            self.kernels
                .moe
                .moe_combine_sorted(&z, inv.as_ref().unwrap(), &weights, s2_opt.as_ref())?
        } else {
            let z_u = if let Some(inv_o) = inv {
                z.take(&inv_o, 0)?
                    .reshape(&[b * length, layer.top_k, hidden])?
            } else {
                z.reshape(&[b * length, layer.top_k, hidden])?
            };
            let w_eff = if let Some(s2_p) = layer.w2.post_scale(&inds, &self.consts.nvfp4_inv)? {
                weights.mul(&s2_p)?
            } else {
                weights
            };
            z_u.cast(MlxDType::F32)?
                .mul(&w_eff.reshape(&[b * length, layer.top_k, 1])?)?
                .sum(&[1], false)?
                .cast(dtype)?
        };
        let m = m_flat.reshape(&[b, length, hidden])?;

        let (m_c, s3_next) = if pf && b == 1 && length >= keep {
            self.kernels
                .glue
                .prefill_static_conv(&m, &layer.mlp_conv, &s3, layer.conv_scale)?
        } else {
            static_conv_eager(&m, &layer.mlp_conv, &s3, layer.conv_scale)?
        };
        let out_h = if pf {
            self.kernels.glue.prefill_residual_out(&residual, &m_c, dtype)?
        } else {
            residual.add(&m_c.cast(MlxDType::F32)?)?.cast(dtype)?
        };

        let next_states = [ks_next, vs_next, s2_next, s3_next];
        for (slot, win) in next_states.into_iter().enumerate() {
            let wlen = win.shape()[1];
            let tail = if wlen > keep {
                win.slice_axis(1, wlen - keep, wlen)?.copy()?
            } else {
                win
            };
            visit_cache.conv[slot] = ConvSlotState::Array(tail);
        }
        visit_cache.verify_windows = None;
        Ok(out_h)
    }

    /// Run all 48 visits for `tokens` (`[1, L]` I32/U32).
    /// Returns `(final_hidden [1, L, H] BF16, captured_taps)` in `tap_visits` order.
    fn forward_backbone(
        &mut self,
        tokens: &Array,
        tap_visits: &[usize],
        exact_verify: bool,
    ) -> Result<(Array, Vec<Array>)> {
        let mut h = self
            .weights
            .embed
            .embed(tokens)?
            .fast_rms_norm(&self.weights.embed_norm, self.config.text_config.rms_norm_eps)?;
        let length = h.shape()[1];
        let use_compiled_decode =
            !exact_verify && length == 1 && self.cache.visits[0].kv.is_allocated();
        let use_exact_verify = exact_verify
            && length > 1
            && length <= VERIFY_MAX_ROWS
            && self.cache.visits[0].kv.is_allocated();
        let split = if use_exact_verify || use_compiled_decode {
            V_SPLIT_VISITS
        } else {
            0
        };
        let last_visit = self.schedule.len() - 1;
        let num_visits = self.schedule.len();
        // Cache full-attention `log_tau` once per step (shared by all 8 full visits).
        let step_tau: Option<Array> = if self.config.text_config.log_scaling_n_floor > 0 {
            Some(log_tau(
                self.stream(),
                self.cache.visits[0].kv.offset,
                length,
                self.config.text_config.log_scaling_n_floor,
                self.config.text_config.log_scaling_alpha,
            )?)
        } else {
            None
        };

        let mut blocks: Vec<Array> = Vec::with_capacity(6);
        let mut prelude: Option<(Array, Vec<Array>)> = None;
        let mut bank: Option<(Vec<Array>, Array)> = None;
        let mut captured: HashMap<usize, Array> = HashMap::with_capacity(tap_visits.len());

        for visit in 0..num_visits {
            let (layer_idx, loop_pass) = self.schedule[visit];
            if self.core_start == Some(layer_idx) {
                if prelude.is_none() {
                    prelude = Some((h.clone(), blocks.clone()));
                } else {
                    let (pre_h, pre_blks) = prelude.as_ref().unwrap();
                    let bank_blks = blocks[pre_blks.len() + 1..].to_vec();
                    let bank_part = h.cast(MlxDType::F32)?.fast_rms_norm_unweighted(1e-6)?;
                    bank = Some((bank_blks, bank_part));
                    h = pre_h.clone();
                    blocks = pre_blks.clone();
                }
            }
            if tap_visits.contains(&visit) {
                captured.insert(visit, h.clone());
            }
            let layer = self.weights.layers[layer_idx].clone();
            let bank_ref = if loop_pass > 0 {
                bank.as_ref().map(|(b, p)| (b.as_slice(), p))
            } else {
                None
            };
            // Borrow cache slot without conflicting with `&self` compiled caches.
            let mut visit_cache = std::mem::replace(
                &mut self.cache.visits[visit],
                VisitCache::new(layer.n_kv, layer.head_dim),
            );
            let layer_tau = if layer.log_floor.is_some() {
                step_tau.as_ref()
            } else {
                None
            };
            let res = if use_exact_verify {
                self.run_layer_compiled(
                    &layer,
                    &h,
                    &mut blocks,
                    &mut visit_cache,
                    bank_ref,
                    layer_tau,
                    true,
                )
            } else if use_compiled_decode {
                self.run_layer_compiled(
                    &layer,
                    &h,
                    &mut blocks,
                    &mut visit_cache,
                    bank_ref,
                    layer_tau,
                    false,
                )
            } else {
                self.run_layer_eager(&layer, &h, &mut blocks, &mut visit_cache, bank_ref, layer_tau)
            };
            self.cache.visits[visit] = visit_cache;
            h = res?;
            if split > 0 && visit < last_visit && (visit + 1) % split == 0 {
                self.stream().async_eval(&[h.clone()])?;
            }
        }

        let h_n = h.cast(MlxDType::F32)?.fast_rms_norm_unweighted(1e-6)?;
        let final_scale = (self.config.text_config.hidden_size as f32).powf(-0.5);
        let x = self
            .kernels
            .banded
            .attn_res_mix(&blocks, &h_n, &self.weights.final_input_query, final_scale, true)?
            .slice_axis(2, 0, 1)?
            .reshape(h.shape())?;
        let out = x
            .fast_rms_norm(&self.weights.norm.cast(x.dtype())?, self.config.text_config.rms_norm_eps)?
            .cast(h.dtype())?;
        if tap_visits.contains(&num_visits) {
            captured.insert(num_visits, out.clone());
        }
        let taps_out = tap_visits
            .iter()
            .map(|id| captured.remove(id).unwrap())
            .collect();
        Ok((out, taps_out))
    }

    fn logits_from_hidden(&self, hidden: &Array, exact: bool) -> Result<Array> {
        let raw = if exact && hidden.shape()[0] == 1 && hidden.shape()[1] > 1 {
            self.weights
                .lm_head
                .apply_exact(hidden, true, &self.kernels.gemv)?
        } else {
            self.weights.lm_head.apply(hidden)?
        };
        let mut logits = raw.cast(MlxDType::F32)?;
        if self.weights.softcap > 0.0 {
            let cap = Array::scalar(self.stream(), self.weights.softcap, MlxDType::F32)?;
            logits = logits.div(&cap)?.tanh()?.mul(&cap)?;
        }
        if let Some(vb) = &self.weights.vocab_bias {
            logits = logits.add(vb)?;
        }
        Ok(logits)
    }

    fn tap_rows(taps: &[Array], start: usize) -> Result<Array> {
        let refs: Vec<&Array> = taps.iter().collect();
        let rows = Array::concat(&refs, 2)?;
        if start > 0 {
            rows.slice_axis(1, start, rows.shape()[1])
        } else {
            Ok(rows)
        }
    }

    /// Run one MTP draft+verify cycle for `k` draft tokens (`1..=depth`).
    /// Returns `(accepted_tokens_plus_bonus, next_anchor, cycle_stats)`.
    fn run_mtp_cycle(&mut self, anchor: u32, k: usize) -> Result<(Vec<u32>, u32, MtpCycleStats)> {
        let t0 = Instant::now();
        let stream = self.stream().clone();
        let anchor_arr = Array::from_i32(&stream, &[1], &[anchor as i32])?;
        let (props, tap_ids) = {
            let draft = self
                .draft
                .as_ref()
                .ok_or(Error::Contract("MTP draft not loaded"))?;
            let dstate = self
                .draft_state
                .as_mut()
                .ok_or(Error::Contract("MTP draft state not initialized"))?;
            let props = draft.propose(dstate, &anchor_arr, k, &self.weights)?;
            (props, draft.target_layer_ids.clone())
        };
        // Overlap GPU draft execution with host verify graph construction.
        stream.async_eval(&[props.clone()])?;

        let verify_in = Array::concat(&[&anchor_arr, &props], 0)?.reshape(&[1, k + 1])?;
        let (hidden, taps) = self.forward_backbone(&verify_in, &tap_ids, true)?;
        let rows = Self::tap_rows(&taps, 0)?;

        // Certified q8/g64 screen_argmax when dense BF16 lm_head is available; else full logits argmax.
        let use_screen = self.weights.verify_screen.is_some()
            && self.weights.lm_head.is_dense_bf16()
            && hidden.shape()[1] <= VERIFY_MAX_ROWS;
        let target_arr = if use_screen {
            let screen = self.weights.verify_screen.as_ref().unwrap();
            let w_bf16 = self.weights.lm_head.dense_weight().unwrap();
            let h2 = hidden.reshape(&[k + 1, hidden.shape()[2]])?;
            self.kernels.gemv.screen_argmax(w_bf16, &h2, screen)?
        } else {
            let logits = self.logits_from_hidden(&hidden, true)?;
            logits
                .reshape(&[k + 1, logits.shape()[2]])?
                .argmax(-1)?
                .cast(MlxDType::I32)?
        };
        stream.async_eval(&[target_arr.clone(), props.clone(), rows.clone()])?;

        let mut tl = target_arr.to_i32_vec()?;
        let pl = props.to_i32_vec()?;
        let fell_back = tl.iter().any(|&t| t < 0);
        if fell_back {
            let logits = self.logits_from_hidden(&hidden, true)?;
            let full = logits
                .reshape(&[k + 1, logits.shape()[2]])?
                .argmax(-1)?
                .cast(MlxDType::I32)?;
            tl = full.to_i32_vec()?;
        }

        let mut a = 0usize;
        while a < k && pl[a] == tl[a] {
            a += 1;
        }
        let mut emit = Vec::with_capacity(a + 1);
        for &tok in &pl[..a] {
            emit.push(tok as u32);
        }
        let bonus = tl[a] as u32;
        emit.push(bonus);

        if !self.cache.mtp_partial_rollback(a, k)? {
            return Err(Error::Other("MTP verify rollback found no conv windows".into()));
        }
        let kept_rows = rows.slice_axis(1, 0, a + 1)?;
        if let (Some(draft), Some(dstate)) = (&self.draft, &mut self.draft_state) {
            draft.queue(dstate, kept_rows)?;
        }
        let stats = MtpCycleStats {
            k,
            accepted: a,
            total_ms: t0.elapsed().as_secs_f64() * 1000.0,
            screen_fallback: fell_back,
        };
        Ok((emit, bonus, stats))
    }

    /// Prefill `token_ids` and return `(last_logits [1, V] F32, requested_taps)`.
    /// Used by parity tests to compare all 11 visit taps against a reference snapshot.
    pub fn prefill_with_taps(
        &mut self,
        token_ids: &[u32],
        tap_visits: &[usize],
    ) -> Result<(Array, Vec<(usize, Array)>)> {
        if token_ids.is_empty() {
            return Err(Error::Contract("prefill requires non-empty prompt"));
        }
        self.reset();
        let n_prompt = token_ids.len();
        let stream = self.stream().clone();
        self.cache.reserve_capacity(&stream, n_prompt + 256, MlxDType::BF16)?;

        let mut i = 0usize;
        while i + PREFILL_STEP < n_prompt {
            let j = i + PREFILL_STEP;
            let chunk = Array::from_u32(&stream, &[1, j - i], &token_ids[i..j])?;
            let _ = self.forward_backbone(&chunk, &[], false)?;
            let state = self.cache.state_arrays();
            stream.eval(&state)?;
            i = j;
        }
        let tail = Array::from_u32(&stream, &[1, n_prompt - i], &token_ids[i..])?;
        let (hidden, taps) = self.forward_backbone(&tail, tap_visits, false)?;
        let h_last = hidden.slice(
            &[0, n_prompt - i - 1, 0],
            &[1, n_prompt - i, self.config.text_config.hidden_size],
        )?;
        let logits = self.logits_from_hidden(&h_last, false)?;
        let mut to_eval = vec![logits.clone()];
        to_eval.extend(taps.iter().cloned());
        stream.eval(&to_eval)?;
        let out_taps = tap_visits.iter().copied().zip(taps).collect();
        Ok((logits.reshape(&[1, logits.shape()[2]])?, out_taps))
    }

    /// Greedy benchmark driver with optional MTP speculative decoding.
    pub fn generate_greedy_benchmark(
        &mut self,
        prompt_ids: &[u32],
        max_new_tokens: usize,
        forced_prefix: &[u32],
        use_mtp: bool,
    ) -> Result<BenchmarkRun> {
        if prompt_ids.is_empty() {
            return Err(Error::Contract("empty prompt"));
        }
        let n_prompt = prompt_ids.len();
        let stream = self.stream().clone();
        // Benchmark runs must be independent: drop any cached prefix state so
        // back-to-back MTP-off/MTP-on runs start from identical cache state.
        self.invalidate_cache();
        self.reset();
        let _ = apxinf_mlx::reset_peak_memory();
        let start = Instant::now();
        self.cache
            .reserve_capacity(&stream, n_prompt + max_new_tokens, MlxDType::BF16)?;

        let run_mtp = use_mtp && self.draft.is_some();
        if run_mtp {
            let (ring_slots, tap_ids, depth) = {
                let d = self.draft.as_ref().unwrap();
                (d.ring_slots, d.target_layer_ids.clone(), self.mtp_depth.min(d.depth))
            };
            let keep_from = n_prompt.saturating_sub(ring_slots);
            self.draft_state = Some(self.draft.as_ref().unwrap().new_state(&stream, keep_from)?);

            let mut i = 0usize;
            while i + 1 < n_prompt {
                let j = (i + PREFILL_STEP).min(n_prompt - 1);
                let chunk = Array::from_u32(&stream, &[1, j - i], &prompt_ids[i..j])?;
                if j > keep_from {
                    let (_, taps) = self.forward_backbone(&chunk, &tap_ids, false)?;
                    let rows = Self::tap_rows(&taps, keep_from.saturating_sub(i))?;
                    let mut to_eval = self.cache.state_arrays();
                    to_eval.push(rows.clone());
                    stream.eval(&to_eval)?;
                    self.draft
                        .as_ref()
                        .unwrap()
                        .queue(self.draft_state.as_mut().unwrap(), rows)?;
                } else {
                    let _ = self.forward_backbone(&chunk, &[], false)?;
                    let state = self.cache.state_arrays();
                    stream.eval(&state)?;
                }
                i = j;
            }
            let last = Array::from_u32(&stream, &[1, 1], &prompt_ids[n_prompt - 1..])?;
            let (hidden, taps) = self.forward_backbone(&last, &tap_ids, false)?;
            let rows = Self::tap_rows(&taps, 0)?;
            self.draft
                .as_ref()
                .unwrap()
                .queue(self.draft_state.as_mut().unwrap(), rows)?;
            let logits = self.logits_from_hidden(&hidden, false)?;
            let mut y = logits.reshape(&[logits.shape()[2]])?.argmax(0)?;
            stream.async_eval(&[y.clone()])?;

            let mut out: Vec<u32> = Vec::with_capacity(max_new_tokens);
            let mut first_at: Option<Instant> = None;

            // Phase 1: forced prefix + first sampled token.
            while out.len() < max_new_tokens {
                let tok = if out.len() < forced_prefix.len() {
                    forced_prefix[out.len()]
                } else {
                    y.to_u32_scalar()?
                };
                out.push(tok);
                if first_at.is_none() {
                    first_at = Some(Instant::now());
                }
                if out.len() == max_new_tokens || out.len() > forced_prefix.len() {
                    break;
                }
                let tok_arr = Array::from_u32(&stream, &[1, 1], &[tok])?;
                let (h1, taps1) = self.forward_backbone(&tok_arr, &tap_ids, false)?;
                let rows1 = Self::tap_rows(&taps1, 0)?;
                self.draft
                    .as_ref()
                    .unwrap()
                    .queue(self.draft_state.as_mut().unwrap(), rows1)?;
                let l1 = self.logits_from_hidden(&h1, false)?;
                y = l1.reshape(&[l1.shape()[2]])?.argmax(0)?;
                stream.async_eval(&[y.clone()])?;
            }

            // Phase 2: MTP draft + exact verify cycles.
            let mut anchor = *out.last().unwrap();
            let mut stats = MtpStats {
                depth,
                cycles: Vec::new(),
                screen_fallbacks: 0,
            };
            while out.len() < max_new_tokens {
                let k = depth.min(max_new_tokens - out.len() - 1);
                if k == 0 {
                    let tok_arr = Array::from_u32(&stream, &[1, 1], &[anchor])?;
                    let (h1, _) = self.forward_backbone(&tok_arr, &[], false)?;
                    let l1 = self.logits_from_hidden(&h1, false)?;
                    let next_tok = l1.reshape(&[l1.shape()[2]])?.argmax(0)?.to_u32_scalar()?;
                    out.push(next_tok);
                    break;
                }
                let (emit, bonus, cstats) = self.run_mtp_cycle(anchor, k)?;
                if cstats.screen_fallback {
                    stats.screen_fallbacks += 1;
                }
                stats.cycles.push(cstats);
                let take = (max_new_tokens - out.len()).min(emit.len());
                out.extend_from_slice(&emit[..take]);
                anchor = bonus;
            }
            let end = Instant::now();
            let first = first_at.unwrap_or(end);
            let ttft_s = first.duration_since(start).as_secs_f64();
            let decode_s = end.duration_since(first).as_secs_f64();
            let tpot_ms = if out.len() > 1 {
                1000.0 * decode_s / (out.len() - 1) as f64
            } else {
                0.0
            };
            let peak_gb =
                apxinf_mlx::memory_stats()?.peak_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            self.last_mtp_stats = Some(stats.clone());
            return Ok(BenchmarkRun {
                prompt_tokens: n_prompt,
                tokens: out,
                ttft_s,
                decode_s,
                tpot_ms,
                peak_memory_gb: peak_gb,
                mtp: Some(stats),
            });
        }

        // MTP-off: chunked prefill + true lookahead single-token decode (Mechanism T).
        let mut i = 0usize;
        while i + 1 < n_prompt {
            let j = (i + PREFILL_STEP).min(n_prompt - 1);
            let chunk = Array::from_u32(&stream, &[1, j - i], &prompt_ids[i..j])?;
            let _ = self.forward_backbone(&chunk, &[], false)?;
            let state = self.cache.state_arrays();
            stream.eval(&state)?;
            i = j;
        }
        let last = Array::from_u32(&stream, &[1, 1], &prompt_ids[n_prompt - 1..])?;
        let (hidden, _) = self.forward_backbone(&last, &[], false)?;
        let logits = self.logits_from_hidden(&hidden, false)?;
        let mut y = logits.reshape(&[logits.shape()[2]])?.argmax(0)?;
        stream.async_eval(&[y.clone()])?;

        let mut out: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let mut first_at: Option<Instant> = None;
        for step in 0..max_new_tokens {
            if step < forced_prefix.len() || first_at.is_none() {
                let tok = if step < forced_prefix.len() {
                    forced_prefix[step]
                } else {
                    y.to_u32_scalar()?
                };
                out.push(tok);
                if first_at.is_none() {
                    first_at = Some(Instant::now());
                }
                if step + 1 == max_new_tokens {
                    break;
                }
                let tok_arr = Array::from_u32(&stream, &[1, 1], &[tok])?;
                let (h1, _) = self.forward_backbone(&tok_arr, &[], false)?;
                let l1 = self.logits_from_hidden(&h1, false)?;
                y = l1.reshape(&[l1.shape()[2]])?.argmax(0)?;
                stream.async_eval(&[y.clone()])?;
            } else {
                let nxt = if step + 1 < max_new_tokens {
                    let y2 = y.reshape(&[1, 1])?;
                    let (h1, _) = self.forward_backbone(&y2, &[], false)?;
                    let l1 = self.logits_from_hidden(&h1, false)?;
                    let n_arr = l1.reshape(&[l1.shape()[2]])?.argmax(0)?;
                    stream.async_eval(&[n_arr.clone()])?;
                    Some(n_arr)
                } else {
                    None
                };
                out.push(y.to_u32_scalar()?);
                match nxt {
                    Some(n_arr) => y = n_arr,
                    None => break,
                }
            }
        }
        let end = Instant::now();
        let first = first_at.unwrap_or(end);
        let ttft_s = first.duration_since(start).as_secs_f64();
        let decode_s = end.duration_since(first).as_secs_f64();
        let tpot_ms = if out.len() > 1 {
            1000.0 * decode_s / (out.len() - 1) as f64
        } else {
            0.0
        };
        let peak_gb = apxinf_mlx::memory_stats()?.peak_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        Ok(BenchmarkRun {
            prompt_tokens: n_prompt,
            tokens: out,
            ttft_s,
            decode_s,
            tpot_ms,
            peak_memory_gb: peak_gb,
            mtp: None,
        })
    }
}

impl MetalInkling2 {
    /// Hard-invalidate the prefix cache: drop all KV/conv state and the
    /// cached prompt. Callers that need deterministic per-run behavior
    /// (parity tests, greedy reference comparisons) must call this after
    /// `reset()`; serving-style callers skip it to allow cross-request reuse.
    pub fn invalidate_cache(&mut self) {
        self.cache.reset();
        self.cached_prompt_tokens.clear();
        self.cached_kv_valid = 0;
    }

    /// Prefix reuse is opt-in (`APXINF_INKLING2_PREFIX_CACHE=1`): reusing
    /// cached KV skips recompute but is not bit-exact with a fresh prefill
    /// (MTP draft-ring and conv-window state are rebuilt rather than restored),
    /// so it stays off unless a serving deployment accepts that trade-off.
    fn prefix_cache_enabled() -> bool {
        std::env::var("APXINF_INKLING2_PREFIX_CACHE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    }

    /// Length of the longest common prefix between `tokens` and the cached
    /// prompt whose KV state is still valid.
    fn common_prefix_len(&self, tokens: &[u32]) -> usize {
        if !Self::prefix_cache_enabled() {
            return 0;
        }
        if self.cached_kv_valid == 0 || self.cached_prompt_tokens.is_empty() {
            return 0;
        }
        let valid = self.cached_kv_valid.min(self.cached_prompt_tokens.len());
        let limit = valid.min(tokens.len());
        // For safety, require the reuse to be at least 4 tokens and exclude
        // the last token (the KV at position N-1 was consumed to produce the
        // first decode logits; the cache may have been extended since).
        if limit < 4 {
            return 0;
        }
        let mut prefix = 0;
        for i in 0..limit {
            if self.cached_prompt_tokens[i] != tokens[i] {
                break;
            }
            prefix = i + 1;
        }
        // Don't reuse the very last cached token: the cache offset after
        // generation includes decode tokens appended after the prompt, so
        // the boundary is ambiguous. Conservatively drop the last token.
        prefix.saturating_sub(1)
    }
}

impl DeviceDecode for MetalInkling2 {
    fn stream(&self) -> &Stream {
        self.backend.stream()
    }

    fn decode(&mut self, token: &Array) -> Result<Array> {
        let tok_2d = token.reshape(&[1, 1])?;
        let (hidden, _) = self.forward_backbone(&tok_2d, &[], false)?;
        let logits = self.logits_from_hidden(&hidden, false)?;
        logits.reshape(&[1, logits.shape()[2]])
    }

    fn finish_decode(&mut self, outcome: DecodeFinish) -> Result<()> {
        self.backend.stream().synchronize()?;
        if let DecodeFinish::Invalidated = outcome {
            self.reset();
        }
        Ok(())
    }
}

/// Request-owned greedy MTP `TokenSequence` wired into `LlmTrait::token_sequence`.
struct Inkling2MtpSequence<'a> {
    model: &'a mut MetalInkling2,
    first_pending: Option<Array>,
    anchor: Option<u32>,
    buffer: VecDeque<u32>,
    emitted: usize,
    max_new: usize,
    stats: MtpStats,
    failed: bool,
    finished: bool,
}

impl<'a> Inkling2MtpSequence<'a> {
    fn new(model: &'a mut MetalInkling2, first_logits: Array) -> Result<Self> {
        let depth = model
            .draft
            .as_ref()
            .map_or(DEFAULT_MTP_DEPTH, |d| model.mtp_depth.min(d.depth));
        let max_new = model.planned_max_new;
        let first_argmax = first_logits
            .reshape(&[first_logits.shape()[first_logits.shape().len() - 1]])?
            .argmax(0)?;
        model.stream().async_eval(&[first_argmax.clone()])?;
        Ok(Self {
            model,
            first_pending: Some(first_argmax),
            anchor: None,
            buffer: VecDeque::new(),
            emitted: 0,
            max_new,
            stats: MtpStats {
                depth,
                cycles: Vec::new(),
                screen_fallbacks: 0,
            },
            failed: false,
            finished: false,
        })
    }

    fn step(&mut self, lookahead: bool) -> Result<GeneratedToken> {
        if let Some(first_arr) = self.first_pending.take() {
            let tok = first_arr.to_u32_scalar()?;
            self.anchor = Some(tok);
            self.emitted += 1;
            return Ok(GeneratedToken {
                token_id: tok,
                logprob: None,
            });
        }
        if let Some(tok) = self.buffer.pop_front() {
            self.emitted += 1;
            return Ok(GeneratedToken {
                token_id: tok,
                logprob: None,
            });
        }
        let anchor = self
            .anchor
            .ok_or(Error::Contract("MTP sequence missing anchor"))?;
        let k = if !lookahead {
            0
        } else if self.max_new > self.emitted + 1 {
            self.stats.depth.min(self.max_new - self.emitted - 1)
        } else {
            self.stats.depth
        };
        if k == 0 {
            let stream = self.model.stream().clone();
            let tok_arr = Array::from_u32(&stream, &[1, 1], &[anchor])?;
            let (h1, _) = self.model.forward_backbone(&tok_arr, &[], false)?;
            let l1 = self.model.logits_from_hidden(&h1, false)?;
            let tok = l1.reshape(&[l1.shape()[2]])?.argmax(0)?.to_u32_scalar()?;
            self.anchor = Some(tok);
            self.emitted += 1;
            return Ok(GeneratedToken {
                token_id: tok,
                logprob: None,
            });
        }
        let (emit, bonus, cstats) = self.model.run_mtp_cycle(anchor, k)?;
        if cstats.screen_fallback {
            self.stats.screen_fallbacks += 1;
        }
        self.stats.cycles.push(cstats);
        self.anchor = Some(bonus);
        let mut iter = emit.into_iter();
        let first = iter.next().ok_or(Error::Contract("empty MTP emit"))?;
        self.buffer.extend(iter);
        self.emitted += 1;
        Ok(GeneratedToken {
            token_id: first,
            logprob: None,
        })
    }
}

impl TokenSequence for Inkling2MtpSequence<'_> {
    fn next(&mut self, lookahead: bool) -> Result<GeneratedToken> {
        let res = self.step(lookahead);
        self.failed |= res.is_err();
        res
    }

    fn finish(mut self: Box<Self>, published: usize) -> Result<()> {
        let res = self.model.stream().synchronize();
        let stats = std::mem::take(&mut self.stats);
        if decode_trace_enabled() {
            eprintln!(
                "inkling2 decode trace: sequence=mtp published={published} emitted={} cycles={} \
                 tpc={:?} screen_fallbacks={} median_cycle_ms={:?} failed={}",
                self.emitted,
                stats.cycles.len(),
                stats.tokens_per_cycle().map(|v| (v * 1000.0).round() / 1000.0),
                stats.screen_fallbacks,
                stats.median_cycle_ms().map(|v| (v * 1000.0).round() / 1000.0),
                self.failed,
            );
        }
        self.model.last_mtp_stats = Some(stats);
        if self.failed {
            self.model.reset();
        }
        self.finished = true;
        res
    }
}

impl Drop for Inkling2MtpSequence<'_> {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.model.stream().synchronize();
            self.model.reset();
        }
    }
}

impl LlmTrait for MetalInkling2 {
    fn load(_config: ModelConfig, _weights: HashMap<String, Tensor>, _device: Device) -> Result<Self>
    where
        Self: Sized,
    {
        Err(Error::Contract(
            "MetalInkling2 must be loaded via metal_inkling2::load",
        ))
    }

    fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Err(Error::Contract("forward requires at least one token"));
        }
        if start_pos == 0 && self.cache.offset() > 0 {
            self.reset();
        }
        let stream = self.stream().clone();
        let want = self.cache.offset() + token_ids.len();
        if self.cache.capacity() < want {
            self.cache.reserve_capacity(&stream, want, MlxDType::BF16)?;
        }
        let ids = Array::from_u32(&stream, &[1, token_ids.len()], token_ids)?;
        let (hidden, _) = self.forward_backbone(&ids, &[], false)?;
        let logits = self
            .logits_from_hidden(&hidden, false)?
            .reshape(&[token_ids.len(), self.config.text_config.vocab_size])?;
        self.backend.from_array(logits)
    }

    fn prefill(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        if input.image.is_some() {
            return Err(Error::Other(
                "metal_inkling2 text engine does not support image input yet".into(),
            ));
        }
        let token_ids = input.token_ids;
        if token_ids.is_empty() {
            return Err(Error::Contract("prefill requires at least one token"));
        }
        let n_prompt = token_ids.len();
        let stream = self.stream().clone();

        // Prefix-cache: find the longest common prefix with the last prefilled
        // prompt whose KV is still valid, and skip prefilling those tokens.
        let reuse = self.common_prefix_len(token_ids);
        self.last_prefix_reuse = reuse;
        if reuse > 0 {
            // Trim decode tokens appended after the previous prompt so the
            // cache offset matches the prefix we are about to reuse.
            self.cache.trim_to(reuse);
            eprintln!(
                "inkling2 prefix-cache: reusing {} / {} tokens",
                reuse, n_prompt
            );
        } else if self.cached_kv_valid > 0 || self.cache.offset() > 0 {
            // No reusable prefix; clear any stale cache (decode tokens from a
            // previous generation, or a previous prompt's KV).
            self.cache.reset();
        }
        self.cached_kv_valid = 0;

        let reserve = self
            .planned_tokens
            .unwrap_or(n_prompt + 256)
            .max(n_prompt + 16);
        self.cache.reserve_capacity(&stream, reserve, MlxDType::BF16)?;

        let capture_mtp = self.mtp_enabled && self.draft.is_some();
        let (keep_from, tap_ids) = if capture_mtp {
            let d = self.draft.as_ref().unwrap();
            let kf = n_prompt.saturating_sub(d.ring_slots);
            self.draft_state = Some(d.new_state(&stream, kf)?);
            (kf, d.target_layer_ids.clone())
        } else {
            self.draft_state = None;
            (n_prompt, Vec::new())
        };

        let mut i = reuse;
        let mut queued_rows = 0usize;
        while i + 1 < n_prompt {
            let j = (i + PREFILL_STEP).min(n_prompt - 1);
            let chunk = Array::from_u32(&stream, &[1, j - i], &token_ids[i..j])?;
            if capture_mtp && j > keep_from {
                let (_, taps) = self.forward_backbone(&chunk, &tap_ids, false)?;
                let rows = Self::tap_rows(&taps, keep_from.saturating_sub(i))?;
                queued_rows += rows.shape()[0];
                let mut to_eval = self.cache.state_arrays();
                to_eval.push(rows.clone());
                stream.eval(&to_eval)?;
                self.draft
                    .as_ref()
                    .unwrap()
                    .queue(self.draft_state.as_mut().unwrap(), rows)?;
            } else {
                let _ = self.forward_backbone(&chunk, &[], false)?;
                let state = self.cache.state_arrays();
                stream.eval(&state)?;
            }
            i = j;
        }

        let last = Array::from_u32(&stream, &[1, 1], &token_ids[n_prompt - 1..])?;
        let (hidden, taps) = self.forward_backbone(&last, &tap_ids, false)?;
        if capture_mtp {
            let rows = Self::tap_rows(&taps, 0)?;
            queued_rows += rows.shape()[0];
            self.draft
                .as_ref()
                .unwrap()
                .queue(self.draft_state.as_mut().unwrap(), rows)?;
        }
        if decode_trace_enabled() {
            let ring_slots = self.draft.as_ref().map_or(0, |d| d.ring_slots);
            eprintln!(
                "inkling2 decode trace: prefill n_prompt={n_prompt} reuse={reuse} \
                 capture_mtp={capture_mtp} keep_from={keep_from} ring_slots={ring_slots} \
                 queued_rows={queued_rows}",
            );
        }
        let logits = self
            .logits_from_hidden(&hidden, false)?
            .reshape(&[1, self.config.text_config.vocab_size])?;
        self.last_prefill_logits = Some(logits.clone());
        // Record the prompt for prefix-cache reuse. The full prompt's KV is
        // now in the cache, so `cached_kv_valid` covers all n_prompt tokens.
        self.cached_prompt_tokens = token_ids.to_vec();
        self.cached_kv_valid = n_prompt;
        self.cache_generation += 1;
        self.backend.from_array(logits)
    }

    fn prewarm_decode(&mut self, prompt_len: usize, max_new_tokens: usize) {
        self.planned_tokens = Some(prompt_len.saturating_add(max_new_tokens));
        self.planned_max_new = max_new_tokens;
    }

    fn last_reused_prompt_tokens(&self) -> usize {
        self.last_prefix_reuse
    }

    fn backend(&self) -> &dyn Backend {
        &self.backend
    }

    fn reset(&mut self) {
        // Preserve KV cache for potential prefix reuse. The framework calls
        // reset() before every generation; we defer the actual cache clear
        // to prefill() where we can compare the new prompt against the cached
        // one and skip the common prefix.
        self.draft_state = None;
        self.last_prefill_logits = None;
        self.planned_tokens = None;
        self.planned_max_new = 0;
    }

    fn vocab_size(&self) -> usize {
        self.config.text_config.vocab_size
    }

    fn preparation_status(&self) -> TextPreparationStatus {
        let mut scopes = vec![TextCompilationScope::DecoderBlock];
        if self.draft.is_some() {
            scopes.push(TextCompilationScope::DraftProposal);
        }
        TextPreparationStatus {
            state: TextPreparationState::Ready,
            implementation: "metal_inkling2",
            variant: Some(match self.variant {
                Variant::Nvfp4Mixed => "nvfp4-mixed",
                Variant::Oq4eMtp => "oq4e-mtp",
            }),
            compiled_scopes: scopes,
            prepared_prompt_tokens: self.planned_tokens,
            prepared_sequence_lengths: vec![1, 2, 3],
            kv_capacity: Some(self.config.text_config.model_max_length),
            max_decode_rows: Some(VERIFY_MAX_ROWS),
            error: None,
        }
    }

    fn token_sequence<'a>(
        &'a mut self,
        first: Tensor,
        sampler: Box<dyn TokenSampler>,
        params: &TokenSamplingParams,
        prompt_end: u32,
    ) -> Result<Box<dyn TokenSequence + 'a>> {
        let trace = decode_trace_enabled();
        if mlx_decode::enabled(params) {
            if let Some(first_arr) = self.last_prefill_logits.take() {
                if self.mtp_enabled && self.draft.is_some() && self.draft_state.is_some() {
                    if trace {
                        eprintln!(
                            "inkling2 decode trace: sequence=mtp selection={:?} max_new={} depth={}",
                            params.selection, self.planned_max_new, self.mtp_depth,
                        );
                    }
                    return Ok(Box::new(Inkling2MtpSequence::new(self, first_arr)?));
                }
                if trace {
                    eprintln!(
                        "inkling2 decode trace: sequence=greedy-async selection={:?} \
                         mtp_enabled={} draft={} draft_state={}",
                        params.selection,
                        self.mtp_enabled,
                        self.draft.is_some(),
                        self.draft_state.is_some(),
                    );
                }
                return Ok(Box::new(GreedySequence::new(self, first_arr, prompt_end)?));
            }
        }
        if trace {
            eprintln!(
                "inkling2 decode trace: sequence=sync selection={:?} penalties={:?} \
                 return_logprob={} mlx_decode={} prefill_logits={}",
                params.selection,
                params.penalties,
                params.return_logprob,
                mlx_decode::enabled(params),
                self.last_prefill_logits.is_some(),
            );
        }
        self.last_prefill_logits = None;
        Ok(Box::new(SyncTokenSequence::new(
            self, first, sampler, prompt_end,
        )))
    }
}
