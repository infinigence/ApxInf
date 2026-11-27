//! Checkpoint loading, weight sanitization, and Metal weight preparation for Inkling-2-Mini.
//!
//! Supports both:
//! - Original `model.llm.*` layout (BF16 dense attention/embed/lm_head + ModelOpt NVFP4/MXFP8 experts).
//! - Pre-sanitized `model.*` layout (e.g. `oQ4e-mtp`: affine Q8/g64 dense + NVFP4/affine Q4/g64 experts).

use super::config::{Config, QuantSpec};
use apxinf_core::{Error, Result};
use apxinf_mlx::{
    exact_gemv::{ExactGemvKernels, VerifyHeadScreen},
    load_safetensors, Array, MlxDType, QuantMode, Stream,
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    rc::Rc,
};

pub const NVFP4_DENOMINATOR: f32 = 448.0 * 6.0;
const ORIG_PREFIX: &str = "model.llm.";
const VISUAL_PREFIX: &str = "model.visual.";
const LOOP_INPUTS: [&str; 4] = ["attn", "k", "v", "mlp"];

/// Either a BF16 dense matrix `[Out, In]` or an MLX affine quantized matrix.
#[derive(Clone)]
pub enum LinearWeight {
    Dense(Array),
    Quantized {
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        mode: QuantMode,
        bits: i32,
        group_size: i32,
    },
}

impl LinearWeight {
    pub fn is_dense_bf16(&self) -> bool {
        matches!(self, Self::Dense(w) if w.dtype() == MlxDType::BF16)
    }

    pub fn dense_weight(&self) -> Option<&Array> {
        match self {
            Self::Dense(w) => Some(w),
            Self::Quantized { .. } => None,
        }
    }

    pub fn out_features(&self) -> usize {
        match self {
            Self::Dense(w) => w.shape()[0],
            Self::Quantized { weight, .. } => weight.shape()[0],
        }
    }

    /// Standard forward `x @ W.T`.
    pub fn apply(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Dense(w) => {
                let wt = w.transpose(&[1, 0])?;
                x.matmul(&wt)
            }
            Self::Quantized {
                weight,
                scales,
                biases,
                mode,
                bits,
                group_size,
            } => x.quantized_matmul_mode(
                weight,
                scales,
                biases.as_ref(),
                true,
                *group_size,
                *bits,
                *mode,
            ),
        }
    }

    /// Exact-verify linear: when `exact && x.shape()[0] == 1 && x.shape()[1] > 1`, uses
    /// `multirow_gemv` for BF16 dense weights (bit-equal to 1-row decode).
    pub fn apply_exact(&self, x: &Array, exact: bool, gemv: &ExactGemvKernels) -> Result<Array> {
        if exact && x.shape().len() == 3 && x.shape()[0] == 1 && x.shape()[1] > 1 {
            if let Self::Dense(w) = self {
                let x2 = x.reshape(&[x.shape()[1], x.shape()[2]])?;
                let y = gemv.multirow_gemv(w, &x2, false)?;
                return y.reshape(&[1, x.shape()[1], y.shape()[1]]);
            }
        }
        self.apply(x)
    }

    /// Embedding lookup (`[B, L]` int32/u32 -> `[B, L, H]` BF16).
    pub fn embed(&self, tokens: &Array) -> Result<Array> {
        match self {
            Self::Dense(w) => w.take(tokens, 0),
            Self::Quantized {
                weight,
                scales,
                biases,
                mode,
                bits,
                group_size,
            } => {
                let flat = tokens.reshape(&[tokens.numel()])?;
                let w_sel = weight.take(&flat, 0)?;
                let s_sel = scales.take(&flat, 0)?;
                let b_sel = match biases {
                    Some(b) => Some(b.take(&flat, 0)?),
                    None => None,
                };
                let deq = w_sel.dequantize_mode(
                    &s_sel,
                    b_sel.as_ref(),
                    *group_size,
                    *bits,
                    *mode,
                    Some(MlxDType::BF16),
                )?;
                let mut out_shape = tokens.shape().to_vec();
                out_shape.push(deq.shape()[1]);
                deq.reshape(&out_shape)
            }
        }
    }

    /// Dequantize to BF16 (or return dense weight directly) for folding into MTP Markov head.
    pub fn dequantize_slice(&self, start: usize, end: usize) -> Result<Array> {
        match self {
            Self::Dense(w) => w.slice_axis(0, start, end),
            Self::Quantized {
                weight,
                scales,
                biases,
                mode,
                bits,
                group_size,
            } => {
                let w_sub = weight.slice_axis(0, start, end)?;
                let s_sub = scales.slice_axis(0, start, end)?;
                let b_sub = match biases {
                    Some(b) => Some(b.slice_axis(0, start, end)?),
                    None => None,
                };
                w_sub.dequantize_mode(
                    &s_sub,
                    b_sub.as_ref(),
                    *group_size,
                    *bits,
                    *mode,
                    Some(MlxDType::BF16),
                )
            }
        }
    }
}

/// Stacked routed-expert projection (`[E, N, K_packed]` U32).
#[derive(Clone)]
pub struct ExpertProjWeight {
    pub weight: Array,
    pub scales: Array,
    pub biases: Option<Array>,
    pub global_scale: Option<Array>,
    pub mode: QuantMode,
    pub bits: i32,
    pub group_size: i32,
}

impl ExpertProjWeight {
    pub fn gather_qmm(
        &self,
        x: &Array,
        lhs_indices: Option<&Array>,
        rhs_indices: &Array,
        sorted: bool,
    ) -> Result<Array> {
        x.gather_qmm(
            &self.weight,
            &self.scales,
            self.biases.as_ref(),
            lhs_indices,
            Some(rhs_indices),
            true,
            self.group_size,
            self.bits,
            self.mode,
            None,
            sorted,
        )
    }

    pub fn post_scale(&self, idx: &Array, nvfp4_inv: &Array) -> Result<Option<Array>> {
        let Some(gs) = &self.global_scale else {
            return Ok(None);
        };
        let flat = idx.reshape(&[idx.numel()])?;
        let gathered = gs.take(&flat, 0)?.reshape(idx.shape())?;
        Ok(Some(gathered.mul(nvfp4_inv)?))
    }
}

/// Prepared loop controller weights for one core layer (`6..18`).
#[derive(Clone)]
pub struct LoopControllerWeights {
    pub down_qkv_f32: Array,         // [3, 2, H, r] F32
    pub gf_qkv_f32: Array,           // [3, r, 2H] F32
    pub down_mlp_f32: Array,         // [1, 2, H, r] F32
    pub gf_mlp_f32: Array,           // [1, r, 2H] F32
    pub qkv_bf16: Option<(Array, Array, Option<Array>)>, // ([6, H, r], [3, r, 2H], perm)
    pub mlp_bf16: Option<(Array, Array, Option<Array>)>, // ([2, H, r], [1, r, 2H], perm)
}

#[derive(Clone)]
pub enum AttnLayoutWeights {
    Sliding {
        in_proj: LinearWeight,
        k_conv: Array,     // [dkv, K] BF16
        v_conv: Array,     // [dkv, K] BF16
        k_conv_f32: Array, // [dkv, K] F32
        v_conv_f32: Array, // [dkv, K] F32
        attn_norm: Array,  // [H] BF16
    },
    Full {
        q_proj: LinearWeight,
        k_proj: LinearWeight,
        v_proj: LinearWeight,
        k_static: Array,     // [K, dkv] BF16
        v_static: Array,     // [K, dkv] BF16
        k_static_f32: Array, // [K, dkv] F32
        v_static_f32: Array, // [K, dkv] F32
        q_norm: Array,       // [H] BF16
        k_norm: Array,       // [H] BF16
        v_norm: Array,       // [H] BF16
    },
}

/// Complete weights and prepared copies for one physical Inkling-2 layer (`0..24`).
pub struct LayerWeights {
    pub layer_idx: usize,
    pub sliding: bool,
    pub is_boundary: bool,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub dkv: usize,
    pub d_rel: usize,
    pub window: usize,
    pub attn_scale: f32,
    pub log_floor: Option<usize>,
    pub log_alpha: f32,
    pub conv_k: usize,
    pub sconv_scale: f32,
    pub dconv_scale: f32,
    pub rms_eps: f32,
    pub attn_res_scale: f32,
    pub cuts: Vec<usize>,
    pub attn_input_query: Option<Array>, // [Q, H] F32 (None on layer 0)
    pub mlp_input_query: Array,          // [1, H] F32
    pub loop_message_query: Option<Array>, // [4, H] F32 (core layers only)
    pub loop_ctrl: Option<LoopControllerWeights>,
    pub attn_layout: AttnLayoutWeights,
    pub o_proj: LinearWeight,
    pub q_scale: Array,    // [Hq, D] BF16
    pub r_scale: Array,    // [Hq, d_rel] BF16
    pub rel_proj: Array,   // [d_rel, R] BF16
    pub attn_conv: Array,  // [H, K] BF16
    pub attn_conv_f32: Array,
    pub mlp_conv: Array,   // [H, K] BF16
    pub mlp_conv_f32: Array,
    pub conv_scale: f32,
    pub mlp_norm: Array,   // [H] BF16
    pub router_weight: Array,            // [E, H] F32
    pub router_weight_bf16: Option<Array>, // exact BF16 copy
    pub router_bias: Array,              // [E] F32
    pub top_k: usize,
    pub route_scale: f32,
    pub w13: ExpertProjWeight,
    pub w2: ExpertProjWeight,
}

/// Full loaded Inkling-2-Mini model weights on Metal.
pub struct Inkling2Weights {
    pub embed: LinearWeight,
    pub embed_norm: Array,
    pub layers: Vec<Rc<LayerWeights>>,
    pub final_input_query: Array,
    pub norm: Array,
    pub lm_head: LinearWeight,
    pub vocab_bias: Option<Array>,
    pub softcap: f32,
    pub nvfp4_inv: Array,
    pub qk_clip: Array,
    pub r_clip: Array,
    pub one_f32: Array,
    pub gcap: Array,
    pub ucap: Array,
    pub verify_screen: Option<VerifyHeadScreen>,
}

fn find_model_shards(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut shards = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if name.starts_with("model") && name.ends_with(".safetensors") {
            shards.push(path);
        }
    }
    shards.sort();
    if shards.is_empty() {
        return Err(Error::Other(format!(
            "no model*.safetensors found in {}",
            dir.display()
        )));
    }
    Ok(shards)
}

fn pop(map: &mut HashMap<String, Array>, key: &str) -> Result<Array> {
    map.remove(key)
        .ok_or_else(|| Error::Other(format!("missing Inkling2 tensor `{key}`")))
}

fn take_linear(
    map: &mut HashMap<String, Array>,
    prefix: &str,
    config: &Config,
) -> Result<LinearWeight> {
    let w_key = format!("{prefix}.weight");
    let s_key = format!("{prefix}.scales");
    if let Some(scales) = map.remove(&s_key) {
        let weight = pop(map, &w_key)?;
        let biases = map.remove(&format!("{prefix}.biases"));
        let spec = config
            .quant_spec_for(prefix)?
            .unwrap_or(QuantSpec::AFFINE_Q8_G64);
        Ok(LinearWeight::Quantized {
            weight,
            scales,
            biases,
            mode: spec.mode,
            bits: spec.bits,
            group_size: spec.group_size,
        })
    } else {
        Ok(LinearWeight::Dense(pop(map, &w_key)?))
    }
}

fn take_expert_proj(
    map: &mut HashMap<String, Array>,
    prefix: &str,
    config: &Config,
    layer_idx: usize,
    which: &str,
) -> Result<ExpertProjWeight> {
    let weight = pop(map, &format!("{prefix}.weight"))?;
    let scales = pop(map, &format!("{prefix}.scales"))?;
    let biases = map.remove(&format!("{prefix}.biases"));
    let global_scale = map.remove(&format!("{prefix}.global_scale"));
    let spec = if let Some(s) = config.quant_spec_for(prefix)? {
        s
    } else if global_scale.is_some() {
        QuantSpec::NVFP4
    } else if biases.is_some() {
        QuantSpec::AFFINE_Q4_G64
    } else if scales.dtype() == MlxDType::U8 {
        if layer_idx >= config.text_config.num_hidden_layers.saturating_sub(8) {
            QuantSpec::MXFP8
        } else {
            QuantSpec::NVFP4
        }
    } else {
        return Err(Error::Other(format!(
            "cannot infer expert quantization spec for `{prefix}` (layer {layer_idx} {which})"
        )));
    };
    Ok(ExpertProjWeight {
        weight,
        scales,
        biases,
        global_scale,
        mode: spec.mode,
        bits: spec.bits,
        group_size: spec.group_size,
    })
}

/// Sanitize original `model.llm.*` layout to runtime `model.*` layout (exact port of `vendor/loader.py`).
fn sanitize_original_weights(
    stream: &Stream,
    config: &Config,
    raw: HashMap<String, Array>,
) -> Result<HashMap<String, Array>> {
    let text = &config.text_config;
    let hidden = text.hidden_size;
    let k_conv = text.sconv_kernel_size;
    let mut out: HashMap<String, Array> = HashMap::new();
    let mut per_layer: HashMap<usize, HashMap<String, Array>> = HashMap::new();
    let mut loop_q: Option<Array> = None;
    let mut loop_kv: Option<Array> = None;

    for (key, value) in raw {
        if key.starts_with(VISUAL_PREFIX) || !key.starts_with(ORIG_PREFIX) {
            continue;
        }
        let name = &key[ORIG_PREFIX.len()..];
        if name == "embed.weight" {
            out.insert("model.embed.weight".into(), value);
        } else if name == "embed_norm.weight" {
            out.insert("model.embed_norm.weight".into(), value);
        } else if name == "norm.weight" {
            out.insert("model.norm.weight".into(), value);
        } else if name == "unembed.weight" {
            out.insert("lm_head.weight".into(), value);
        } else if let Some(rest) = name.strip_prefix("attn_res.attn_queries.") {
            let idx: usize = rest
                .parse()
                .map_err(|_| Error::Other(format!("bad attn_queries index in `{key}`")))?;
            let q_rows = value.numel() / hidden;
            out.insert(
                format!("model.layers.{}.attn_input.query", idx + 1),
                value.reshape(&[q_rows, hidden])?.cast(MlxDType::F32)?,
            );
        } else if let Some(rest) = name.strip_prefix("attn_res.mlp_queries.") {
            let idx: usize = rest
                .parse()
                .map_err(|_| Error::Other(format!("bad mlp_queries index in `{key}`")))?;
            let q_rows = value.numel() / hidden;
            out.insert(
                format!("model.layers.{idx}.mlp_input.query"),
                value.reshape(&[q_rows, hidden])?.cast(MlxDType::F32)?,
            );
        } else if name == "attn_res.output_query" {
            let q_rows = value.numel() / hidden;
            out.insert(
                "model.final_input.query".into(),
                value.reshape(&[q_rows, hidden])?.cast(MlxDType::F32)?,
            );
        } else if name == "middle_cycle_cross_pass_queries" {
            loop_q = Some(value);
        } else if name == "middle_cycle_cross_pass_kv_queries" {
            loop_kv = Some(value);
        } else if let Some(rest) = name.strip_prefix("layers.") {
            let mut parts = rest.splitn(2, '.');
            let idx: usize = parts
                .next()
                .ok_or_else(|| Error::Other(format!("bad layer key `{key}`")))?
                .parse()
                .map_err(|_| Error::Other(format!("bad layer index in `{key}`")))?;
            let sub = parts
                .next()
                .ok_or_else(|| Error::Other(format!("bad layer suffix in `{key}`")))?;
            per_layer.entry(idx).or_default().insert(sub.to_owned(), value);
        }
    }

    let nvfp4_num = Array::scalar(stream, NVFP4_DENOMINATOR, MlxDType::F32)?;
    let mut layer_ids: Vec<usize> = per_layer.keys().copied().collect();
    layer_ids.sort_unstable();
    for idx in layer_ids {
        let mut d = per_layer.remove(&idx).unwrap();
        let pre = format!("model.layers.{idx}.");
        if config.is_sliding(idx) {
            let wq = pop(&mut d, "attn.wq_du.weight")?;
            let wk = pop(&mut d, "attn.wk_dv.weight")?;
            let wv = pop(&mut d, "attn.wv_dv.weight")?;
            let wr = pop(&mut d, "attn.wr_du.weight")?;
            let wg = pop(&mut d, "attn.gate_attn.weight")?;
            let ws = pop(&mut d, "attn.attn_sink_proj.weight")?;
            let in_proj = Array::concat(&[&wq, &wk, &wv, &wr, &wg, &ws], 0)?;
            out.insert(format!("{pre}attn.in_proj.weight"), in_proj);
            let kc = pop(&mut d, "attn.k_sconv.weight")?;
            let vc = pop(&mut d, "attn.v_sconv.weight")?;
            out.insert(format!("{pre}attn.k_conv"), kc.reshape(&[kc.numel() / k_conv, k_conv])?);
            out.insert(format!("{pre}attn.v_conv"), vc.reshape(&[vc.numel() / k_conv, k_conv])?);
            out.insert(format!("{pre}attn_norm.weight"), pop(&mut d, "attn_norm.weight")?);
        } else {
            let wq = pop(&mut d, "attn.wq_du.weight")?;
            let wr = pop(&mut d, "attn.wr_du.weight")?;
            let wg = pop(&mut d, "attn.gate_attn.weight")?;
            let ws = pop(&mut d, "attn.attn_sink_proj.weight")?;
            out.insert(
                format!("{pre}attn.q_proj.weight"),
                Array::concat(&[&wq, &wr, &wg, &ws], 0)?,
            );
            let wk = pop(&mut d, "attn.wk_dv.weight")?;
            let wkp = pop(&mut d, "attn.k_sconv.weight_proj")?;
            out.insert(format!("{pre}attn.k_proj.weight"), Array::concat(&[&wk, &wkp], 0)?);
            let wv = pop(&mut d, "attn.wv_dv.weight")?;
            let wvp = pop(&mut d, "attn.v_sconv.weight_proj")?;
            out.insert(format!("{pre}attn.v_proj.weight"), Array::concat(&[&wv, &wvp], 0)?);
            out.insert(format!("{pre}attn.k_static"), pop(&mut d, "attn.k_sconv.static_w")?);
            out.insert(format!("{pre}attn.v_static"), pop(&mut d, "attn.v_sconv.static_w")?);
            for n in ["q", "k", "v"] {
                out.insert(
                    format!("{pre}{n}_input_norm.weight"),
                    pop(&mut d, &format!("{n}_input_norm.weight"))?,
                );
            }
        }
        out.insert(format!("{pre}attn.o_proj.weight"), pop(&mut d, "attn.wo_ud.weight")?);
        out.insert(format!("{pre}attn.q_scale"), pop(&mut d, "attn.q_l2_clip.scale")?);
        out.insert(format!("{pre}attn.r_scale"), pop(&mut d, "attn.rel_l2_clip.scale")?);
        out.insert(format!("{pre}attn.rel_proj"), pop(&mut d, "attn.rel_logits_proj.proj")?);
        out.insert(
            format!("{pre}attn_conv"),
            pop(&mut d, "attn_sconv.weight")?.reshape(&[hidden, k_conv])?,
        );
        out.insert(
            format!("{pre}mlp_conv"),
            pop(&mut d, "mlp_sconv.weight")?.reshape(&[hidden, k_conv])?,
        );
        out.insert(format!("{pre}mlp_norm.weight"), pop(&mut d, "mlp_norm.weight")?);
        out.insert(
            format!("{pre}mlp.router_weight"),
            pop(&mut d, "mlp.gate.weight")?.cast(MlxDType::F32)?,
        );
        out.insert(
            format!("{pre}mlp.router_bias"),
            pop(&mut d, "mlp.gate.bias")?.cast(MlxDType::F32)?,
        );

        for which in ["w13", "w2"] {
            let w = pop(&mut d, &format!("mlp.experts.{which}_weight"))?;
            let s = pop(&mut d, &format!("mlp.experts.{which}_weight_scale"))?;
            let base = format!("{pre}mlp.experts.{which}");
            out.insert(format!("{base}.weight"), w.view(MlxDType::U32)?);
            out.insert(format!("{base}.scales"), s);
            if let Some(amax) = d.remove(&format!("mlp.experts.{which}_weight_amax")) {
                let denom = pop(&mut d, &format!("mlp.experts.{which}_weight_outer_scale_denominator"))?;
                let scale = amax
                    .cast(MlxDType::F32)?
                    .mul(&nvfp4_num.div(&denom.cast(MlxDType::F32)?)?)?;
                out.insert(format!("{base}.global_scale"), scale);
            }
        }

        if d.contains_key("controllers.attn.down_weight") {
            let mut downs = Vec::with_capacity(4);
            let mut gates = Vec::with_capacity(4);
            for m in LOOP_INPUTS {
                let down = pop(&mut d, &format!("controllers.{m}.down_weight"))?.cast(MlxDType::F32)?;
                let rank = down.shape()[0];
                let half = down.shape()[1] / 2;
                let down_h = down.reshape(&[rank, 2, half])?.transpose(&[1, 2, 0])?;
                downs.push(down_h);
                let gain = pop(&mut d, &format!("controllers.{m}.gain_weight"))?
                    .cast(MlxDType::F32)?
                    .transpose(&[1, 0])?;
                let forget = pop(&mut d, &format!("controllers.{m}.forget_weight"))?
                    .cast(MlxDType::F32)?
                    .transpose(&[1, 0])?;
                gates.push(Array::concat(&[&gain, &forget], 1)?);
            }
            let down_refs: Vec<Array> = downs
                .iter()
                .map(|a| a.reshape(&[1, a.shape()[0], a.shape()[1], a.shape()[2]]))
                .collect::<Result<_>>()?;
            let gate_refs: Vec<Array> = gates
                .iter()
                .map(|a| a.reshape(&[1, a.shape()[0], a.shape()[1]]))
                .collect::<Result<_>>()?;
            let d_r: Vec<&Array> = down_refs.iter().collect();
            let g_r: Vec<&Array> = gate_refs.iter().collect();
            out.insert(format!("{pre}loop_down"), Array::concat(&d_r, 0)?);
            out.insert(format!("{pre}loop_gain_forget"), Array::concat(&g_r, 0)?);
        }

        if !d.is_empty() {
            let mut extra: Vec<_> = d.keys().cloned().collect();
            extra.sort();
            return Err(Error::Other(format!(
                "unmapped Inkling2 tensors in layer {idx}: {extra:?}"
            )));
        }
    }

    if let (Some(q), Some(kv)) = (loop_q, loop_kv) {
        if let Some(loop_cfg) = &text.loop_config {
            for (i, idx) in (loop_cfg.core_start..loop_cfg.core_end).enumerate() {
                let r0 = q.slice_axis(0, 2 * i, 2 * i + 1)?.cast(MlxDType::F32)?;
                let r1 = kv.slice_axis(0, 2 * i, 2 * i + 1)?.cast(MlxDType::F32)?;
                let r2 = kv.slice_axis(0, 2 * i + 1, 2 * i + 2)?.cast(MlxDType::F32)?;
                let r3 = q.slice_axis(0, 2 * i + 1, 2 * i + 2)?.cast(MlxDType::F32)?;
                out.insert(
                    format!("model.layers.{idx}.loop_message.query"),
                    Array::concat(&[&r0, &r1, &r2, &r3], 0)?,
                );
            }
        }
    }

    Ok(out)
}

/// Build the q8/g64 verify head screen for MTP candidate filtering.
pub fn build_verify_head_screen(
    stream: &Stream,
    lm_head_bf16: &Array,
    softcap: f32,
    vocab_bias: Option<&Array>,
    unpadded_vocab: usize,
) -> Result<VerifyHeadScreen> {
    // 1. All 65536 BF16 patterns -> exact production softcap table `ttab` via MLX.
    let patterns_f32: Vec<f32> = (0u32..65536u32)
        .map(|u| f32::from_bits(u << 16))
        .collect();
    let vals = Array::from_f32(stream, &[65536], &patterns_f32)?.cast(MlxDType::BF16)?;
    let cap_a = Array::scalar(stream, softcap, MlxDType::F32)?;
    let ttab = if softcap > 0.0 {
        vals.cast(MlxDType::F32)?.div(&cap_a)?.tanh()?.mul(&cap_a)?
    } else {
        vals.cast(MlxDType::F32)?
    };
    let ttab_host = ttab.to_f32_vec()?;

    // 2. Threshold table `ptab` (exact port of `_threshold_table`).
    let mut order: Vec<usize> = (0..65536)
        .filter(|&i| !patterns_f32[i].is_nan())
        .collect();
    order.sort_by(|&a, &b| patterns_f32[a].total_cmp(&patterns_f32[b]));
    let n_ok = order.len();
    let mut fs = Vec::with_capacity(n_ok);
    let mut ts = Vec::with_capacity(n_ok);
    for &idx in &order {
        let tv = ttab_host[idx];
        if tv.is_nan() {
            return Err(Error::Other("softcap table has NaN for non-NaN BF16 input".into()));
        }
        fs.push(patterns_f32[idx]);
        ts.push(tv);
    }
    let mut tup = vec![0.0f32; n_ok];
    let mut cur_max = f32::NEG_INFINITY;
    for i in 0..n_ok {
        if ts[i] > cur_max {
            cur_max = ts[i];
        }
        tup[i] = cur_max;
    }
    let mut tdn = vec![0.0f32; n_ok];
    let mut cur_min = f32::INFINITY;
    for i in (0..n_ok).rev() {
        if ts[i] < cur_min {
            cur_min = ts[i];
        }
        tdn[i] = cur_min;
    }
    let mut first_eq = vec![0usize; n_ok];
    for i in 0..n_ok {
        first_eq[i] = if i > 0 && fs[i] == fs[i - 1] {
            first_eq[i - 1]
        } else {
            i
        };
    }
    let mut ptab_host = vec![f32::NEG_INFINITY; 65536];
    for i in 0..n_ok {
        let target = tdn[i];
        // lower_bound in sorted `tup`
        let pos = tup.partition_point(|&v| v < target);
        let thr_pos = pos.min(n_ok - 1);
        let feq = first_eq[thr_pos];
        let p_val = if feq > 0 { fs[feq - 1] } else { f32::NEG_INFINITY };
        ptab_host[order[i]] = p_val;
    }
    let ptab = Array::from_f32(stream, &[65536], &ptab_host)?;

    let v_dim = lm_head_bf16.shape()[0];
    let h_dim = lm_head_bf16.shape()[1];
    let bias = match vocab_bias {
        Some(b) => b.clone(),
        None => Array::zeros(stream, &[v_dim], MlxDType::F32)?,
    };

    // 3. Quantize lm_head to q8/g64 and compute rigorous per-group error bound `err` and `ca`.
    let [wq, sc, bi] = lm_head_bf16.quantize(64, 8)?;
    stream.eval(&[wq.clone(), sc.clone(), bi.clone()])?;

    let c255 = Array::scalar(stream, 255.0, MlxDType::F32)?;
    let eps_20 = Array::scalar(stream, 2.0f32.powi(-20), MlxDType::F32)?;
    let one_plus_20 = Array::scalar(stream, 1.0 + 2.0f32.powi(-20), MlxDType::F32)?;
    let one_plus_6 = Array::scalar(stream, 1.0 + 2.0f32.powi(-6), MlxDType::F32)?;
    let mut errs = Vec::new();
    let mut mmax = 0.0f32;
    let mut wmax = 0.0f32;
    let chunk = 16384usize;
    let mut r0 = 0usize;
    while r0 < v_dim {
        let r1 = (r0 + chunk).min(v_dim);
        let nr = r1 - r0;
        let w32 = lm_head_bf16.slice_axis(0, r0, r1)?.cast(MlxDType::F32)?;
        let s32 = sc.slice_axis(0, r0, r1)?.cast(MlxDType::F32)?;
        let b32 = bi.slice_axis(0, r0, r1)?.cast(MlxDType::F32)?;
        let deq = wq
            .slice_axis(0, r0, r1)?
            .dequantize_mode(&s32, Some(&b32), 64, 8, QuantMode::Affine, Some(MlxDType::F32))?;
        let e = w32
            .sub(&deq)?
            .abs()?
            .reshape(&[nr, h_dim / 64, 64])?
            .max(2)?
            .reshape(&[nr, h_dim / 64])?;
        let m_g = s32.abs()?.mul(&c255)?.add(&b32.abs()?)?;
        let e = e.mul(&one_plus_20)?.add(&m_g.mul(&eps_20)?)?;
        let eb = e.mul(&one_plus_6)?.cast(MlxDType::BF16)?;
        let chunk_mmax = m_g.reshape(&[m_g.numel()])?.max(0)?.to_f32_vec()?[0];
        let chunk_wmax = w32.abs()?.reshape(&[w32.numel()])?.max(0)?.to_f32_vec()?[0];
        if chunk_mmax > mmax {
            mmax = chunk_mmax;
        }
        if chunk_wmax > wmax {
            wmax = chunk_wmax;
        }
        eb.eval()?;
        errs.push(eb);
        r0 = r1;
    }
    let err_refs: Vec<&Array> = errs.iter().collect();
    let err = Array::concat(&err_refs, 0)?;
    let ca_val = 2.0f32 * 2.0f32.powi(-24) * (77.0 * wmax + 25.0 * mmax) * 2.0;
    let ca = Array::from_f32(stream, &[1], &[ca_val])?;
    stream.eval(&[ttab.clone(), ptab.clone(), bias.clone(), err.clone(), ca.clone()])?;

    Ok(VerifyHeadScreen {
        wq,
        sc,
        bi,
        err,
        ca,
        ttab,
        ptab,
        bias,
        unpadded: unpadded_vocab,
    })
}

impl Inkling2Weights {
    pub fn load(stream: &Stream, dir: &Path, config: &Config) -> Result<Self> {
        let shards = find_model_shards(dir)?;
        // Never load vision_tower / model.visual tensors during text-only loading.
        let raw = load_safetensors(stream, &shards, |name| {
            !name.starts_with(VISUAL_PREFIX) && !name.starts_with("vision_tower.")
        })?;

        let is_orig = raw.keys().any(|k| k.starts_with(ORIG_PREFIX));
        let mut weights = if is_orig {
            sanitize_original_weights(stream, config, raw)?
        } else {
            raw
        };

        let text = &config.text_config;
        let embed = take_linear(&mut weights, "model.embed", config)?;
        let embed_norm = pop(&mut weights, "model.embed_norm.weight")?;
        let final_input_query = pop(&mut weights, "model.final_input.query")?.cast(MlxDType::F32)?;
        let norm = pop(&mut weights, "model.norm.weight")?;
        let lm_head = take_linear(&mut weights, "lm_head", config)?;

        let mut layers = Vec::with_capacity(text.num_hidden_layers);
        let mut prepared_arrays: Vec<Array> = Vec::new();
        let core_range = text
            .loop_config
            .as_ref()
            .map(|lc| lc.core_start..lc.core_end)
            .unwrap_or(0..0);

        for idx in 0..text.num_hidden_layers {
            let pre = format!("model.layers.{idx}.");
            let sliding = config.is_sliding(idx);
            let is_boundary = idx % text.n_layers_per_attn_res_block == 0;
            let head_dim = if sliding {
                text.swa_head_dim.unwrap_or(text.head_dim)
            } else {
                text.head_dim
            };
            let n_heads = if sliding {
                text.swa_num_attention_heads.unwrap_or(text.num_attention_heads)
            } else {
                text.num_attention_heads
            };
            let n_kv = if sliding {
                text.swa_num_key_value_heads.unwrap_or(text.num_key_value_heads)
            } else {
                text.num_key_value_heads
            };
            let dq = n_heads * head_dim;
            let dkv = n_kv * head_dim;
            let dr = n_heads * text.d_rel;
            let conv_k = text.sconv_kernel_size;

            let attn_input_query = if idx > 0 {
                Some(pop(&mut weights, &format!("{pre}attn_input.query"))?.cast(MlxDType::F32)?)
            } else {
                None
            };
            let mlp_input_query = pop(&mut weights, &format!("{pre}mlp_input.query"))?.cast(MlxDType::F32)?;

            let (loop_message_query, loop_ctrl) = if core_range.contains(&idx) {
                let lmq = pop(&mut weights, &format!("{pre}loop_message.query"))?.cast(MlxDType::F32)?;
                let down = pop(&mut weights, &format!("{pre}loop_down"))?.cast(MlxDType::F32)?;
                let gf = pop(&mut weights, &format!("{pre}loop_gain_forget"))?.cast(MlxDType::F32)?;
                let h = down.shape()[2];
                let r = down.shape()[3];

                let down_qkv_f32 = down.slice_axis(0, 0, 3)?;
                let gf_qkv_f32 = gf.slice_axis(0, 0, 3)?;
                let down_mlp_f32 = down.slice_axis(0, 3, 4)?;
                let gf_mlp_f32 = gf.slice_axis(0, 3, 4)?;

                // Exact BF16 copy + permuted down weight for `loopmix_down_pair_perm`.
                let d16 = down.cast(MlxDType::BF16)?;
                let g16 = gf.cast(MlxDType::BF16)?;
                let d_eq = d16.cast(MlxDType::F32)?.array_equal(&down)?;
                let g_eq = g16.cast(MlxDType::F32)?.array_equal(&gf)?;
                let exact_bf16 = d_eq.to_bool_scalar()? && g_eq.to_bool_scalar()?;
                let (qkv_bf16, mlp_bf16) = if exact_bf16 {
                    let dq16 = d16.slice_axis(0, 0, 3)?.reshape(&[6, h, r])?.contiguous()?;
                    let gq16 = g16.slice_axis(0, 0, 3)?.contiguous()?;
                    let dq_perm = if ExactGemvKernels::loopmix_perm_supported(&dq16) {
                        Some(ExactGemvKernels::loopmix_perm_weight(&dq16)?)
                    } else {
                        None
                    };
                    let dm16 = d16.slice_axis(0, 3, 4)?.reshape(&[2, h, r])?.contiguous()?;
                    let gm16 = g16.slice_axis(0, 3, 4)?.contiguous()?;
                    let dm_perm = if ExactGemvKernels::loopmix_perm_supported(&dm16) {
                        Some(ExactGemvKernels::loopmix_perm_weight(&dm16)?)
                    } else {
                        None
                    };
                    prepared_arrays.push(dq16.clone());
                    prepared_arrays.push(gq16.clone());
                    if let Some(p) = &dq_perm {
                        prepared_arrays.push(p.clone());
                    }
                    prepared_arrays.push(dm16.clone());
                    prepared_arrays.push(gm16.clone());
                    if let Some(p) = &dm_perm {
                        prepared_arrays.push(p.clone());
                    }
                    (Some((dq16, gq16, dq_perm)), Some((dm16, gm16, dm_perm)))
                } else {
                    (None, None)
                };
                prepared_arrays.push(down_qkv_f32.clone());
                prepared_arrays.push(gf_qkv_f32.clone());
                prepared_arrays.push(down_mlp_f32.clone());
                prepared_arrays.push(gf_mlp_f32.clone());
                (
                    Some(lmq),
                    Some(LoopControllerWeights {
                        down_qkv_f32,
                        gf_qkv_f32,
                        down_mlp_f32,
                        gf_mlp_f32,
                        qkv_bf16,
                        mlp_bf16,
                    }),
                )
            } else {
                (None, None)
            };

            let (cuts, attn_layout) = if sliding {
                let splits = [dq, dkv, dkv, dr, dq, n_heads];
                let mut cuts = Vec::with_capacity(5);
                let mut acc = 0;
                for &s in &splits[..5] {
                    acc += s;
                    cuts.push(acc);
                }
                let in_proj = take_linear(&mut weights, &format!("{pre}attn.in_proj"), config)?;
                let k_conv = pop(&mut weights, &format!("{pre}attn.k_conv"))?;
                let v_conv = pop(&mut weights, &format!("{pre}attn.v_conv"))?;
                let k_conv_f32 = k_conv.cast(MlxDType::F32)?;
                let v_conv_f32 = v_conv.cast(MlxDType::F32)?;
                let attn_norm = pop(&mut weights, &format!("{pre}attn_norm.weight"))?;
                prepared_arrays.push(k_conv_f32.clone());
                prepared_arrays.push(v_conv_f32.clone());
                (
                    cuts,
                    AttnLayoutWeights::Sliding {
                        in_proj,
                        k_conv,
                        v_conv,
                        k_conv_f32,
                        v_conv_f32,
                        attn_norm,
                    },
                )
            } else {
                let splits = [dq, dr, dq, n_heads];
                let mut cuts = Vec::with_capacity(3);
                let mut acc = 0;
                for &s in &splits[..3] {
                    acc += s;
                    cuts.push(acc);
                }
                let q_proj = take_linear(&mut weights, &format!("{pre}attn.q_proj"), config)?;
                let k_proj = take_linear(&mut weights, &format!("{pre}attn.k_proj"), config)?;
                let v_proj = take_linear(&mut weights, &format!("{pre}attn.v_proj"), config)?;
                let k_static = pop(&mut weights, &format!("{pre}attn.k_static"))?;
                let v_static = pop(&mut weights, &format!("{pre}attn.v_static"))?;
                let k_static_f32 = k_static.cast(MlxDType::F32)?;
                let v_static_f32 = v_static.cast(MlxDType::F32)?;
                let q_norm = pop(&mut weights, &format!("{pre}q_input_norm.weight"))?;
                let k_norm = pop(&mut weights, &format!("{pre}k_input_norm.weight"))?;
                let v_norm = pop(&mut weights, &format!("{pre}v_input_norm.weight"))?;
                prepared_arrays.push(k_static_f32.clone());
                prepared_arrays.push(v_static_f32.clone());
                (
                    cuts,
                    AttnLayoutWeights::Full {
                        q_proj,
                        k_proj,
                        v_proj,
                        k_static,
                        v_static,
                        k_static_f32,
                        v_static_f32,
                        q_norm,
                        k_norm,
                        v_norm,
                    },
                )
            };

            let o_proj = take_linear(&mut weights, &format!("{pre}attn.o_proj"), config)?;
            let q_scale = pop(&mut weights, &format!("{pre}attn.q_scale"))?;
            let r_scale = pop(&mut weights, &format!("{pre}attn.r_scale"))?;
            let rel_proj = pop(&mut weights, &format!("{pre}attn.rel_proj"))?;
            let attn_conv = pop(&mut weights, &format!("{pre}attn_conv"))?;
            let mlp_conv = pop(&mut weights, &format!("{pre}mlp_conv"))?;
            let attn_conv_f32 = attn_conv.cast(MlxDType::F32)?;
            let mlp_conv_f32 = mlp_conv.cast(MlxDType::F32)?;
            prepared_arrays.push(attn_conv_f32.clone());
            prepared_arrays.push(mlp_conv_f32.clone());

            let mlp_norm = pop(&mut weights, &format!("{pre}mlp_norm.weight"))?;
            let router_weight = pop(&mut weights, &format!("{pre}mlp.router_weight"))?.cast(MlxDType::F32)?;
            let router_bias = pop(&mut weights, &format!("{pre}mlp.router_bias"))?.cast(MlxDType::F32)?;
            let rw16 = router_weight.cast(MlxDType::BF16)?;
            let router_weight_bf16 = if rw16.cast(MlxDType::F32)?.array_equal(&router_weight)?.to_bool_scalar()? {
                prepared_arrays.push(rw16.clone());
                Some(rw16)
            } else {
                None
            };

            let w13 = take_expert_proj(
                &mut weights,
                &format!("{pre}mlp.experts.w13"),
                config,
                idx,
                "w13",
            )?;
            let w2 = take_expert_proj(
                &mut weights,
                &format!("{pre}mlp.experts.w2"),
                config,
                idx,
                "w2",
            )?;

            layers.push(Rc::new(LayerWeights {
                layer_idx: idx,
                sliding,
                is_boundary,
                head_dim,
                n_heads,
                n_kv,
                dkv,
                d_rel: text.d_rel,
                window: if sliding { text.sliding_window_size } else { 0 },
                attn_scale: (head_dim as f32).powf(-0.5),
                log_floor: if sliding { None } else { Some(text.log_scaling_n_floor) },
                log_alpha: text.log_scaling_alpha,
                conv_k,
                sconv_scale: 1.0 / ((conv_k + 1) as f32).sqrt(),
                dconv_scale: 1.0 / ((2 * conv_k + 1) as f32).sqrt(),
                rms_eps: text.rms_norm_eps,
                attn_res_scale: (text.hidden_size as f32).powf(-0.5),
                cuts,
                attn_input_query,
                mlp_input_query,
                loop_message_query,
                loop_ctrl,
                attn_layout,
                o_proj,
                q_scale,
                r_scale,
                rel_proj,
                attn_conv,
                attn_conv_f32,
                mlp_conv,
                mlp_conv_f32,
                conv_scale: 1.0 / ((conv_k + 1) as f32).sqrt(),
                mlp_norm,
                router_weight,
                router_weight_bf16,
                router_bias,
                top_k: text.num_experts_per_tok,
                route_scale: text.route_scale,
                w13,
                w2,
            }));
        }

        let vocab = text.vocab_size;
        let unpadded = config.unpadded_vocab_size();
        let vocab_bias = if unpadded < vocab {
            let ar = Array::arange(stream, 0.0, vocab as f32, 1.0, MlxDType::F32)?;
            let thr = Array::scalar(stream, unpadded as f32, MlxDType::F32)?;
            let mask = ar.less(&thr)?;
            let zero = Array::scalar(stream, 0.0, MlxDType::F32)?;
            let neg_inf = Array::scalar(stream, f32::NEG_INFINITY, MlxDType::F32)?;
            let vb = mask.where_select(&zero, &neg_inf)?;
            prepared_arrays.push(vb.clone());
            Some(vb)
        } else {
            None
        };

        let nvfp4_inv = Array::from_f32(stream, &[1], &[1.0 / NVFP4_DENOMINATOR])?;
        let qk_clip = Array::scalar(stream, text.qk_l2_clip_threshold, MlxDType::F32)?;
        let r_clip = Array::scalar(stream, text.rel_l2_clip_threshold, MlxDType::F32)?;
        let one_f32 = Array::scalar(stream, 1.0, MlxDType::F32)?;
        let gcap = Array::scalar(stream, text.mlp_gate_softcap, MlxDType::F32)?;
        let ucap = Array::scalar(stream, text.mlp_up_softcap, MlxDType::F32)?;
        prepared_arrays.extend([
            nvfp4_inv.clone(),
            qk_clip.clone(),
            r_clip.clone(),
            one_f32.clone(),
            gcap.clone(),
            ucap.clone(),
            embed_norm.clone(),
            final_input_query.clone(),
            norm.clone(),
        ]);
        stream.eval(&prepared_arrays)?;

        let softcap = text.final_logit_softcapping.unwrap_or(0.0);
        let verify_screen = if let Some(w_bf16) = lm_head.dense_weight() {
            if w_bf16.dtype() == MlxDType::BF16 && w_bf16.shape()[1] % 256 == 0 {
                Some(build_verify_head_screen(
                    stream,
                    w_bf16,
                    softcap,
                    vocab_bias.as_ref(),
                    unpadded,
                )?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self {
            embed,
            embed_norm,
            layers,
            final_input_query,
            norm,
            lm_head,
            vocab_bias,
            softcap,
            nvfp4_inv,
            qk_clip,
            r_clip,
            one_f32,
            gcap,
            ucap,
            verify_screen,
        })
    }
}
