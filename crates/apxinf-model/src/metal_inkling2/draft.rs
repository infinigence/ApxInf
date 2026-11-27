//! DSpark 3-layer MTP draft network with RNN Markov head (`mtp.safetensors`).
//!
//! Ported from EngineTailor's Inkling-2 MTP draft implementation (Apache-2.0).

use super::config::Config;
use super::weights::Inkling2Weights;
use apxinf_core::{Error, Result};
use apxinf_mlx::{load_safetensors, Array, MlxDType, QuantMode, Stream};
use std::{collections::HashMap, path::Path};

pub const DRAFT_FILENAME: &str = "mtp.safetensors";
const DRAFT_QUANT_GROUP: i32 = 64;
const DRAFT_QUANT_BITS: i32 = 8;
const D_HEAD_GROUP: i32 = 64;
pub const RUNTIME_D_HEAD_BITS: i32 = 4;

#[derive(Clone, Debug)]
pub struct DSparkConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub sliding_window: usize,
    pub block_size: usize,
    pub markov_rank: usize,
    pub target_layer_ids: Vec<usize>,
    pub vocab_size: usize,
    pub unpadded_vocab_size: usize,
    pub final_logit_softcapping: f32,
}

impl DSparkConfig {
    pub fn from_config(config: &Config) -> Result<Self> {
        let mtp = config
            .mtp_config
            .as_ref()
            .ok_or(Error::Contract("Inkling2 config has no mtp_config"))?;
        if mtp.q_bias || mtp.o_bias {
            return Err(Error::Contract(
                "Inkling2 DSpark does not support projection biases",
            ));
        }
        if mtp.markov_head_type.as_deref() != Some("rnn_hidden") {
            return Err(Error::Contract(
                "Inkling2 DSpark expects markov_head_type == 'rnn_hidden'",
            ));
        }
        let block_size = mtp
            .block_size
            .ok_or(Error::Contract("mtp_config missing block_size"))?;
        let markov_rank = mtp
            .markov_rank
            .ok_or(Error::Contract("mtp_config missing markov_rank"))?;
        if mtp.target_layer_ids.is_empty() || block_size < 2 || mtp.sliding_window_size < 2 {
            return Err(Error::Contract("invalid Inkling2 mtp_config dimensions"));
        }
        let text = &config.text_config;
        Ok(Self {
            hidden_size: mtp.hidden_size,
            num_hidden_layers: mtp.num_hidden_layers,
            num_attention_heads: mtp.num_attention_heads,
            num_key_value_heads: mtp.num_key_value_heads,
            head_dim: mtp.head_dim,
            intermediate_size: mtp.intermediate_size,
            rms_norm_eps: mtp.rms_norm_eps,
            rope_theta: mtp.rope_theta,
            sliding_window: mtp.sliding_window_size,
            block_size,
            markov_rank,
            target_layer_ids: mtp.target_layer_ids.clone(),
            vocab_size: text.vocab_size,
            unpadded_vocab_size: config.unpadded_vocab_size(),
            final_logit_softcapping: text.final_logit_softcapping.unwrap_or(0.0),
        })
    }
}

/// 8-bit affine quantized linear (group 64) for draft transformer layers.
struct DraftQuantLinear {
    weight: Array,
    scales: Array,
    biases: Array,
}

impl DraftQuantLinear {
    fn from_dense(w: &Array) -> Result<Self> {
        let [weight, scales, biases] = w.quantize(DRAFT_QUANT_GROUP, DRAFT_QUANT_BITS)?;
        Ok(Self {
            weight,
            scales,
            biases,
        })
    }

    fn apply(&self, x: &Array) -> Result<Array> {
        x.quantized_matmul(
            &self.weight,
            &self.scales,
            &self.biases,
            true,
            DRAFT_QUANT_GROUP,
            DRAFT_QUANT_BITS,
        )
    }

    fn push_arrays(&self, out: &mut Vec<Array>) {
        out.push(self.weight.clone());
        out.push(self.scales.clone());
        out.push(self.biases.clone());
    }
}

struct DraftAttention {
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    eps: f32,
    q_proj: DraftQuantLinear,
    k_proj: DraftQuantLinear,
    v_proj: DraftQuantLinear,
    o_proj: DraftQuantLinear,
    k_proj_ctx: DraftQuantLinear,
    v_proj_ctx: DraftQuantLinear,
    q_norm: Array,
    k_norm: Array,
    k_norm_ctx: Array,
    alpha: Array,      // [T] BF16
    mix_weight: Array, // [H] BF16 (dflare_norm.weight)
}

impl DraftAttention {
    fn project_context_kv(&self, context: &Array) -> Result<(Array, Array)> {
        let mixed = mix_rmsnorm(context, &self.alpha, &self.mix_weight, self.eps)?;
        Ok((self.k_proj_ctx.apply(&mixed)?, self.v_proj_ctx.apply(&mixed)?))
    }
}

struct DraftMlp {
    gate_proj: DraftQuantLinear,
    up_proj: DraftQuantLinear,
    down_proj: DraftQuantLinear,
}

impl DraftMlp {
    fn apply(&self, x: &Array) -> Result<Array> {
        let g = self.gate_proj.apply(x)?;
        let u = self.up_proj.apply(x)?;
        let act = g.mul(&g.sigmoid()?)?.mul(&u)?;
        self.down_proj.apply(&act)
    }
}

struct DraftLayer {
    self_attn: DraftAttention,
    mlp: DraftMlp,
    input_layernorm: Array,
    post_attention_layernorm: Array,
}

struct MarkovHead {
    rank: usize,
    eps: f32,
    softcap: f32,
    markov_w1: Array,          // [V, r] BF16
    rnn_weight: Array,         // [3r, 2r + 2H] BF16
    rnn_bias: Array,           // [3r] BF16
    dflare_norm: Array,        // [H] BF16
    alpha: Array,              // [T] BF16
    fold_weight: Array,        // [V, r] BF16
    fold_bias: Array,          // [V] BF16
    fold_q4: Option<[Array; 3]>,
    vocab_bias: Option<Array>, // [V] F32
}

impl MarkovHead {
    fn committed_row(&self, last: &Array) -> Result<Array> {
        let mixed = mix_rmsnorm(last, &self.alpha, &self.dflare_norm, self.eps)?;
        let h = mixed.shape()[2];
        mixed.reshape(&[1, h])
    }

    fn fold_matvec(&self, latent: &Array) -> Result<Array> {
        if let Some([wq, sc, bi]) = &self.fold_q4 {
            latent.quantized_matmul(wq, sc, bi, true, D_HEAD_GROUP, RUNTIME_D_HEAD_BITS)
        } else {
            let wt = self.fold_weight.transpose(&[1, 0])?;
            latent.matmul(&wt)
        }
    }

    fn step_logits(
        &self,
        base_logits: &Array,
        prev_tokens: &Array,
        state: &Array,
        hidden_row: &Array,
        rows_f: &Array,
    ) -> Result<(Array, Array)> {
        let stream = state.stream();
        let dtype = state.dtype();
        let prev_flat = prev_tokens.reshape(&[prev_tokens.numel()])?;
        let w1_tok = self.markov_w1.take(&prev_flat, 0)?.cast(dtype)?;
        let h_row = if hidden_row.dtype() == dtype {
            hidden_row.clone()
        } else {
            hidden_row.cast(dtype)?
        };
        let rf = if rows_f.dtype() == dtype {
            rows_f.clone()
        } else {
            rows_f.cast(dtype)?
        };
        let inputs = Array::concat(&[state, &w1_tok, &h_row, &rf], 1)?;
        let wt = self.rnn_weight.transpose(&[1, 0])?;
        let proj = inputs.matmul(&wt)?.add(&self.rnn_bias)?.cast(MlxDType::F32)?;
        let r = self.rank;
        let p0 = proj.slice_axis(1, 0, r)?;
        let p1 = proj.slice_axis(1, r, 2 * r)?;
        let p2 = proj.slice_axis(1, 2 * r, 3 * r)?;
        let gate = p0.sigmoid()?.cast(dtype)?;
        let kept = gate.mul(state)?.cast(dtype)?;
        let one = Array::scalar(stream, 1.0, dtype)?;
        let blend = one.sub(&gate)?.mul(&p1.tanh()?.cast(dtype)?)?.cast(dtype)?;
        let new_state = kept.add(&blend)?;
        let latent = p2.tanh()?.cast(dtype)?;
        let bias = self.fold_matvec(&latent)?.add(&self.fold_bias)?;
        let mut logits = base_logits.cast(MlxDType::F32)?.add(&bias.cast(MlxDType::F32)?)?;
        if self.softcap > 0.0 {
            let cap = Array::scalar(stream, self.softcap, MlxDType::F32)?;
            logits = logits.div(&cap)?.tanh()?.mul(&cap)?;
        }
        if let Some(vb) = &self.vocab_bias {
            logits = logits.add(vb)?;
        }
        Ok((logits, new_state))
    }
}

fn mix_rmsnorm(context: &Array, alpha: &Array, weight: &Array, eps: f32) -> Result<Array> {
    let shape = context.shape();
    let (b, n, width) = (shape[0], shape[1], shape[2]);
    let taps = alpha.shape()[0];
    let x = context
        .reshape(&[b, n, taps, width / taps])?
        .cast(MlxDType::F32)?;
    let a = alpha.cast(MlxDType::F32)?.reshape(&[1, 1, taps, 1])?;
    let mixed = x.mul(&a)?.sum(&[2], false)?;
    let w32 = weight.cast(MlxDType::F32)?;
    mixed.fast_rms_norm(&w32, eps)?.cast(context.dtype())
}

/// Write `src` (`[1, n_kv, n, head_dim]`) into ring buffer `[1, n_kv, capacity, head_dim]`
/// starting at `base % capacity` with wrap-around via contiguous `slice_update_static`.
fn write_ring_kv(ring: &Array, src: &Array, base: usize, n: usize, capacity: usize) -> Result<Array> {
    let n_kv = ring.shape()[1];
    let head_dim = ring.shape()[3];
    let start = base % capacity;
    let ring = ring.stop_gradient()?;
    if start + n <= capacity {
        ring.slice_update_static(src, &[0, 0, start, 0], &[1, n_kv, start + n, head_dim])
    } else {
        let first = capacity - start;
        let second = n - first;
        let s1 = src.slice_axis(2, 0, first)?;
        let s2 = src.slice_axis(2, first, n)?;
        let r1 = ring.slice_update_static(&s1, &[0, 0, start, 0], &[1, n_kv, capacity, head_dim])?;
        r1.slice_update_static(&s2, &[0, 0, 0, 0], &[1, n_kv, second, head_dim])
    }
}

/// Write absolute positions `[base .. base + n)` into ring `pos` (`[1, capacity]` I32).
fn write_ring_pos(pos: &Array, base: usize, n: usize, capacity: usize) -> Result<Array> {
    let stream = pos.stream();
    let start = base % capacity;
    let vals: Vec<i32> = (base..base + n).map(|p| p as i32).collect();
    let pos = pos.stop_gradient()?;
    if start + n <= capacity {
        let upd = Array::from_i32(stream, &[1, n], &vals)?;
        pos.slice_update_static(&upd, &[0, start], &[1, start + n])
    } else {
        let first = capacity - start;
        let second = n - first;
        let u1 = Array::from_i32(stream, &[1, first], &vals[..first])?;
        let u2 = Array::from_i32(stream, &[1, second], &vals[first..])?;
        let p1 = pos.slice_update_static(&u1, &[0, start], &[1, capacity])?;
        p1.slice_update_static(&u2, &[0, 0], &[1, second])
    }
}

/// Per-request DSpark ring KV and pending tap state.
pub struct DraftState {
    pub keys: Vec<Array>,
    pub values: Vec<Array>,
    pub pos: Array,
    pub fed: usize,
    pub pending: Vec<Array>,
    pub pending_len: usize,
    pub f_rnn: Option<Array>,
}

/// Batch-1 greedy DSpark MTP drafter bound to target Inkling-2-Mini weights.
pub struct DSparkDraft {
    pub config: DSparkConfig,
    layers: Vec<DraftLayer>,
    norm: Array,
    mask_embed: Array,
    markov_head: MarkovHead,
    /// Draft-only Q4/g64 copy of target `lm_head` (when target `lm_head` is dense BF16).
    head_q4: Option<[Array; 3]>,
    pub target_layer_ids: Vec<usize>,
    pub block_size: usize,
    pub depth: usize,
    pub ring_slots: usize,
    pub capacity: usize,
    kv_shape: [usize; 4],
}

fn pop_draft(map: &mut HashMap<String, Array>, key: &str) -> Result<Array> {
    map.remove(key)
        .ok_or_else(|| Error::Other(format!("missing mtp.safetensors tensor `{key}`")))
}

impl DSparkDraft {
    /// Load `<dir>/mtp.safetensors` if present, quantize draft layers to Q8/g64,
    /// fold target `lm_head` into Markov head, and build Q4/g64 draft head copies.
    pub fn load_if_present(
        stream: &Stream,
        dir: &Path,
        config: &Config,
        target: &Inkling2Weights,
    ) -> Result<Option<Self>> {
        let path = dir.join(DRAFT_FILENAME);
        if !path.is_file() || config.mtp_config.is_none() {
            return Ok(None);
        }
        let dcfg = DSparkConfig::from_config(config)?;
        let mut raw = load_safetensors(stream, &[&path], |_| true)?;

        let fuse_weight = pop_draft(&mut raw, "fuse_weight")?;
        let mask_embed = pop_draft(&mut raw, "mask_embed")?;
        let norm = pop_draft(&mut raw, "norm.weight")?;
        let alpha_all = fuse_weight
            .cast(MlxDType::F32)?
            .softmax(-1)?
            .cast(MlxDType::BF16)?;

        let mut to_eval: Vec<Array> = vec![mask_embed.clone(), norm.clone(), alpha_all.clone()];
        let mut layers = Vec::with_capacity(dcfg.num_hidden_layers);
        for i in 0..dcfg.num_hidden_layers {
            let pre = format!("layers.{i}.");
            let q_proj = DraftQuantLinear::from_dense(&pop_draft(&mut raw, &format!("{pre}self_attn.q_proj.weight"))?)?;
            let k_proj = DraftQuantLinear::from_dense(&pop_draft(&mut raw, &format!("{pre}self_attn.k_proj.weight"))?)?;
            let v_proj = DraftQuantLinear::from_dense(&pop_draft(&mut raw, &format!("{pre}self_attn.v_proj.weight"))?)?;
            let o_proj = DraftQuantLinear::from_dense(&pop_draft(&mut raw, &format!("{pre}self_attn.o_proj.weight"))?)?;
            let k_proj_ctx = DraftQuantLinear::from_dense(&pop_draft(&mut raw, &format!("{pre}self_attn.k_proj_ctx.weight"))?)?;
            let v_proj_ctx = DraftQuantLinear::from_dense(&pop_draft(&mut raw, &format!("{pre}self_attn.v_proj_ctx.weight"))?)?;
            let q_norm = pop_draft(&mut raw, &format!("{pre}self_attn.q_norm.weight"))?;
            let k_norm = pop_draft(&mut raw, &format!("{pre}self_attn.k_norm.weight"))?;
            let k_norm_ctx = pop_draft(&mut raw, &format!("{pre}self_attn.k_norm_ctx.weight"))?;
            let input_layernorm = pop_draft(&mut raw, &format!("{pre}input_layernorm.weight"))?;
            let post_attention_layernorm = pop_draft(&mut raw, &format!("{pre}post_attention_layernorm.weight"))?;
            let dflare_norm = pop_draft(&mut raw, &format!("{pre}dflare_norm.weight"))?;
            let alpha = alpha_all
                .slice_axis(0, i, i + 1)?
                .reshape(&[dcfg.target_layer_ids.len()])?;

            let gate_proj = DraftQuantLinear::from_dense(&pop_draft(&mut raw, &format!("{pre}mlp.gate_proj.weight"))?)?;
            let up_proj = DraftQuantLinear::from_dense(&pop_draft(&mut raw, &format!("{pre}mlp.up_proj.weight"))?)?;
            let down_proj = DraftQuantLinear::from_dense(&pop_draft(&mut raw, &format!("{pre}mlp.down_proj.weight"))?)?;

            for ql in [&q_proj, &k_proj, &v_proj, &o_proj, &k_proj_ctx, &v_proj_ctx, &gate_proj, &up_proj, &down_proj] {
                ql.push_arrays(&mut to_eval);
            }
            to_eval.extend([
                q_norm.clone(),
                k_norm.clone(),
                k_norm_ctx.clone(),
                input_layernorm.clone(),
                post_attention_layernorm.clone(),
                dflare_norm.clone(),
                alpha.clone(),
            ]);

            layers.push(DraftLayer {
                self_attn: DraftAttention {
                    n_heads: dcfg.num_attention_heads,
                    n_kv_heads: dcfg.num_key_value_heads,
                    head_dim: dcfg.head_dim,
                    scale: (dcfg.head_dim as f32).powf(-0.5),
                    eps: dcfg.rms_norm_eps,
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    k_proj_ctx,
                    v_proj_ctx,
                    q_norm,
                    k_norm,
                    k_norm_ctx,
                    alpha,
                    mix_weight: dflare_norm,
                },
                mlp: DraftMlp {
                    gate_proj,
                    up_proj,
                    down_proj,
                },
                input_layernorm,
                post_attention_layernorm,
            });
        }

        let markov_w1 = pop_draft(&mut raw, "markov_head.markov_w1.weight")?;
        let rnn_weight = pop_draft(&mut raw, "markov_head.rnn_proj.weight")?;
        let rnn_bias = pop_draft(&mut raw, "markov_head.rnn_proj.bias")?;
        let hidden_out_weight = pop_draft(&mut raw, "markov_head.hidden_out.weight")?;
        let hidden_out_bias = pop_draft(&mut raw, "markov_head.hidden_out.bias")?;
        let mh_dflare_norm = pop_draft(&mut raw, "markov_head.dflare_norm.weight")?;
        let mh_alpha = alpha_all
            .slice_axis(0, dcfg.num_hidden_layers, dcfg.num_hidden_layers + 1)?
            .reshape(&[dcfg.target_layer_ids.len()])?;

        // Fold target lm_head @ hidden_out in 16384-row chunks (matches mtp_draft.py:212-222).
        let proj = hidden_out_weight.cast(MlxDType::F32)?;
        let h_bias = hidden_out_bias.cast(MlxDType::F32)?;
        let vocab = target.lm_head.out_features();
        let mut fold_rows = Vec::new();
        let mut fold_biases = Vec::new();
        let mut r0 = 0usize;
        while r0 < vocab {
            let r1 = (r0 + 16384).min(vocab);
            let head = target.lm_head.dequantize_slice(r0, r1)?.cast(MlxDType::F32)?;
            let rw = head.matmul(&proj)?.cast(MlxDType::BF16)?;
            let rb = head.matmul(&h_bias)?.cast(MlxDType::BF16)?;
            stream.eval(&[rw.clone(), rb.clone()])?;
            fold_rows.push(rw);
            fold_biases.push(rb);
            r0 = r1;
        }
        let rw_refs: Vec<&Array> = fold_rows.iter().collect();
        let rb_refs: Vec<&Array> = fold_biases.iter().collect();
        let fold_weight = Array::concat(&rw_refs, 0)?;
        let fold_bias = Array::concat(&rb_refs, 0)?;
        stream.eval(&[fold_weight.clone(), fold_bias.clone()])?;

        // Draft-only Q4/g64 copies when target lm_head is dense BF16 (4-bit draft head).
        let (head_q4, fold_q4) = if let Some(w_bf16) = target.lm_head.dense_weight() {
            if w_bf16.dtype() == MlxDType::BF16 && RUNTIME_D_HEAD_BITS > 0 {
                let hq = w_bf16.quantize(D_HEAD_GROUP, RUNTIME_D_HEAD_BITS)?;
                let fq = fold_weight.quantize(D_HEAD_GROUP, RUNTIME_D_HEAD_BITS)?;
                to_eval.extend(hq.iter().cloned());
                to_eval.extend(fq.iter().cloned());
                (Some(hq), Some(fq))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        let vocab_bias = target.vocab_bias.clone();
        to_eval.extend([
            markov_w1.clone(),
            rnn_weight.clone(),
            rnn_bias.clone(),
            mh_dflare_norm.clone(),
            mh_alpha.clone(),
        ]);
        stream.eval(&to_eval)?;

        let block_size = dcfg.block_size;
        let depth = block_size - 1;
        let ring_slots = dcfg.sliding_window - 1;
        let capacity = ring_slots + block_size;
        let kv_shape = [1, dcfg.num_key_value_heads, capacity, dcfg.head_dim];
        let target_layer_ids = dcfg.target_layer_ids.clone();

        Ok(Some(Self {
            config: dcfg.clone(),
            layers,
            norm,
            mask_embed,
            markov_head: MarkovHead {
                rank: dcfg.markov_rank,
                eps: dcfg.rms_norm_eps,
                softcap: dcfg.final_logit_softcapping,
                markov_w1,
                rnn_weight,
                rnn_bias,
                dflare_norm: mh_dflare_norm,
                alpha: mh_alpha,
                fold_weight,
                fold_bias,
                fold_q4,
                vocab_bias,
            },
            head_q4,
            target_layer_ids,
            block_size,
            depth,
            ring_slots,
            capacity,
            kv_shape,
        }))
    }

    pub fn new_state(&self, stream: &Stream, start: usize) -> Result<DraftState> {
        let mut keys = Vec::with_capacity(self.layers.len());
        let mut values = Vec::with_capacity(self.layers.len());
        for _ in 0..self.layers.len() {
            keys.push(Array::zeros(stream, &self.kv_shape, MlxDType::BF16)?);
            values.push(Array::zeros(stream, &self.kv_shape, MlxDType::BF16)?);
        }
        let neg_one = Array::from_i32(stream, &[], &[-1])?;
        let pos = Array::full(stream, &[1, self.capacity], &neg_one, MlxDType::I32)?;
        Ok(DraftState {
            keys,
            values,
            pos,
            fed: start,
            pending: Vec::new(),
            pending_len: 0,
            f_rnn: None,
        })
    }

    pub fn queue(&self, state: &mut DraftState, rows: Array) -> Result<()> {
        let added = rows.shape()[1];
        state.pending.push(rows);
        state.pending_len += added;
        if state.pending_len <= self.ring_slots {
            return Ok(());
        }
        let mut drop = state.pending_len - self.ring_slots;
        state.fed += drop;
        state.pending_len -= drop;
        while drop > 0 {
            let head_len = state.pending[0].shape()[1];
            if head_len <= drop {
                drop -= head_len;
                state.pending.remove(0);
            } else {
                let tail = state.pending[0].slice_axis(1, drop, head_len)?;
                state.pending[0] = tail;
                drop = 0;
            }
        }
        Ok(())
    }

    fn draft_logits(&self, hidden: &Array, target: &Inkling2Weights) -> Result<Array> {
        if let Some([wq, sc, bi]) = &self.head_q4 {
            hidden.quantized_matmul_mode(
                wq,
                sc,
                Some(bi),
                true,
                D_HEAD_GROUP,
                RUNTIME_D_HEAD_BITS,
                QuantMode::Affine,
            )
        } else {
            target.lm_head.apply(hidden)
        }
    }

    /// Draft `steps` (`1..=self.depth`) proposal tokens anchored on `anchor` (`[1]` I32).
    /// Returns `[steps]` I32 lazy token IDs.
    pub fn propose(
        &self,
        state: &mut DraftState,
        anchor: &Array,
        steps: usize,
        target: &Inkling2Weights,
    ) -> Result<Array> {
        if state.pending.is_empty() {
            return Err(Error::Other("DSpark draft has no queued context".into()));
        }
        let stream = anchor.stream();
        let pending = std::mem::take(&mut state.pending);
        state.pending_len = 0;
        let mut context = if pending.len() == 1 {
            pending.into_iter().next().unwrap()
        } else {
            let refs: Vec<&Array> = pending.iter().collect();
            Array::concat(&refs, 1)?
        };
        let ctx_len = context.shape()[1];
        let skip = ctx_len.saturating_sub(self.ring_slots);
        if skip > 0 {
            context = context.slice_axis(1, skip, ctx_len)?;
        }
        let n = context.shape()[1];
        let base = state.fed + skip;

        let anchor_flat = anchor.reshape(&[anchor.numel()])?;
        let anchor_last = anchor_flat
            .slice_axis(0, anchor_flat.numel() - 1, anchor_flat.numel())?
            .cast(MlxDType::I32)?;
        let anchor_2d = anchor_last.reshape(&[1, 1])?;
        let h0 = target
            .embed
            .embed(&anchor_2d)?
            .fast_rms_norm(&target.embed_norm, self.config.rms_norm_eps)?;
        let fill = self
            .mask_embed
            .cast(h0.dtype())?
            .broadcast_to(&[1, self.block_size - 1, h0.shape()[2]])?;
        let mut h = Array::concat(&[&h0, &fill], 1)?;

        let total = base + n;
        let base_arr = Array::from_i32(stream, &[1], &[base as i32])?;
        let query_offsets = Array::from_i32(stream, &[1], &[total as i32])?;
        let pos = write_ring_pos(&state.pos, base, n, self.capacity)?;

        let zero_i = Array::from_i32(stream, &[], &[0])?;
        let min_valid = Array::from_i32(
            stream,
            &[1, 1],
            &[(total as i64 - self.ring_slots as i64) as i32],
        )?;
        let max_valid = Array::from_i32(stream, &[1, 1], &[total as i32])?;
        let ring_valid = pos
            .greater_equal(&zero_i)?
            .logical_and(&pos.greater_equal(&min_valid)?)?
            .logical_and(&pos.less(&max_valid)?)?;
        let true_scalar = Array::from_i32(stream, &[], &[1])?.cast(MlxDType::Bool)?;
        let ones_block = Array::full(stream, &[1, self.block_size], &true_scalar, MlxDType::Bool)?;
        let mask = Array::concat(&[&ring_valid, &ones_block], 1)?
            .reshape(&[1, 1, 1, self.capacity + self.block_size])?;

        let mut new_keys = Vec::with_capacity(self.layers.len());
        let mut new_values = Vec::with_capacity(self.layers.len());
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let attn = &layer.self_attn;
            let residual = h.clone();
            let x = h.fast_rms_norm(&layer.input_layernorm, self.config.rms_norm_eps)?;
            let queries = attn.q_proj.apply(&x)?;
            let (ctx_k, ctx_v) = attn.project_context_kv(&context)?;
            let prop_k = attn.k_proj.apply(&x)?;
            let prop_v = attn.v_proj.apply(&x)?;

            let ctx_k = ctx_k
                .reshape(&[1, n, attn.n_kv_heads, attn.head_dim])?
                .fast_rms_norm(&attn.k_norm_ctx, attn.eps)?
                .transpose(&[0, 2, 1, 3])?
                .rope(&base_arr, attn.head_dim, false, self.config.rope_theta, 1.0)?;
            let ctx_v = ctx_v
                .reshape(&[1, n, attn.n_kv_heads, attn.head_dim])?
                .transpose(&[0, 2, 1, 3])?;

            let ring_k = write_ring_kv(&state.keys[layer_index], &ctx_k, base, n, self.capacity)?;
            let ring_v = write_ring_kv(&state.values[layer_index], &ctx_v, base, n, self.capacity)?;
            new_keys.push(ring_k.clone());
            new_values.push(ring_v.clone());

            let queries = queries
                .reshape(&[1, self.block_size, attn.n_heads, attn.head_dim])?
                .fast_rms_norm(&attn.q_norm, attn.eps)?
                .transpose(&[0, 2, 1, 3])?
                .rope(&query_offsets, attn.head_dim, false, self.config.rope_theta, 1.0)?;
            let prop_k = prop_k
                .reshape(&[1, self.block_size, attn.n_kv_heads, attn.head_dim])?
                .fast_rms_norm(&attn.k_norm, attn.eps)?
                .transpose(&[0, 2, 1, 3])?
                .rope(&query_offsets, attn.head_dim, false, self.config.rope_theta, 1.0)?;
            let prop_v = prop_v
                .reshape(&[1, self.block_size, attn.n_kv_heads, attn.head_dim])?
                .transpose(&[0, 2, 1, 3])?;

            let keys = Array::concat(&[&ring_k, &prop_k], 2)?;
            let values = Array::concat(&[&ring_v, &prop_v], 2)?;
            let attended = queries
                .sdpa(&keys, &values, attn.scale, false, Some(&mask))?
                .transpose(&[0, 2, 1, 3])?
                .reshape(&[1, self.block_size, attn.n_heads * attn.head_dim])?;
            let attended = attn.o_proj.apply(&attended)?;
            h = residual.add(&attended)?;

            let residual = h.clone();
            let x = h.fast_rms_norm(&layer.post_attention_layernorm, self.config.rms_norm_eps)?;
            let x = layer.mlp.apply(&x)?;
            h = residual.add(&x)?;
        }

        state.keys = new_keys;
        state.values = new_values;
        state.pos = pos;
        state.fed = total;

        let normed = h.fast_rms_norm(&self.norm, self.config.rms_norm_eps)?;
        let draft_hidden = normed.slice_axis(1, 1, self.block_size)?;
        let n_steps = steps.clamp(1, self.depth);
        let logits = if self.head_q4.is_some() {
            let sub = draft_hidden.slice_axis(1, 0, n_steps)?;
            self.draft_logits(&sub, target)?
        } else {
            self.draft_logits(&draft_hidden, target)?
        };

        let last_ctx = context.slice_axis(1, n - 1, n)?;
        let rows_f = self.markov_head.committed_row(&last_ctx)?;
        state.f_rnn = Some(rows_f.clone());
        let mut rnn = Array::zeros(stream, &[1, self.markov_head.rank], draft_hidden.dtype())?;
        let mut prev = anchor_last;
        let mut tokens = Vec::with_capacity(n_steps);
        let h_dim = draft_hidden.shape()[2];
        let v_dim = logits.shape()[2];
        for step in 0..n_steps {
            let base_l = logits.slice_axis(1, step, step + 1)?.reshape(&[1, v_dim])?;
            let h_row = draft_hidden
                .slice_axis(1, step, step + 1)?
                .reshape(&[1, h_dim])?;
            let (step_logits, next_rnn) =
                self.markov_head
                    .step_logits(&base_l, &prev, &rnn, &h_row, &rows_f)?;
            rnn = next_rnn;
            prev = step_logits.argmax(-1)?;
            tokens.push(prev.clone());
        }
        let refs: Vec<&Array> = tokens.iter().collect();
        Array::concat(&refs, 0)?.cast(MlxDType::I32)
    }
}
