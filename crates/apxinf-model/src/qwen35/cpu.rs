//! Correctness-first single-thread CPU reference forward for Qwen3.5 text.
//!
//! Everything runs in f32. Quantized weights are dequantized into reused
//! per-layer buffers (caching the full model in f32 would need ~100 GB). This
//! path validates the loader + dequant + math end to end; the fast path is
//! separate CUDA code.

use half::bf16;
use rayon::prelude::*;

use crate::qwen35::weights::{Bf16Mat, LayerWeights, MatKind, Q4Linear, Qwen35Weights};

const MAX_LEN: usize = 4096;

#[inline]
fn nib4(v: i32, idx: usize) -> i32 {
    ((v >> (4 * (idx & 7))) & 0xF) - 8
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn q4_to_f32(w: &Q4Linear, dst: &mut Vec<f32>) {
    let out = w.out;
    let inp = w.inp;
    dst.clear();
    dst.resize(out * inp, 0.0);
    let groups = inp / 32;
    let packed_cols = inp.div_ceil(8);
    dst.par_chunks_mut(inp).enumerate().for_each(|(o, row)| {
        let zp_row = (o / 8) * groups;
        let o_off = o % 8;
        for g in 0..groups {
            let zpv = nib4(w.zp[zp_row + g], o_off) as f32;
            let s = w.scale[o * groups + g].to_f32();
            let b = g * 32;
            let pb = o * packed_cols + (g * 32) / 8;
            for j in 0..32 {
                let w4 = nib4(w.packed[pb + j / 8], j % 8) as f32;
                row[b + j] = s * (w4 - zpv);
            }
        }
    });
}

fn bf16mat_to_f32(m: &Bf16Mat, dst: &mut Vec<f32>) {
    dst.clear();
    dst.reserve(m.data.len());
    dst.extend(m.data.iter().map(|v| v.to_f32()));
}

fn bf16vec_to_f32(v: &[bf16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

/// dst[m,n] = a[m,k] * w[n,k]^T, all f32. Each output element is computed
/// independently, so the whole matrix is parallelized with rayon.
fn matmul_nt(dst: &mut [f32], a: &[f32], w: &[f32], m: usize, k: usize, n: usize) {
    debug_assert!(dst.len() >= m * n);
    let _ = m;
    dst.par_iter_mut().enumerate().for_each(|(f, out)| {
        let mi = f / n;
        let o = f % n;
        let arow = &a[mi * k..(mi + 1) * k];
        let wrow = &w[o * k..(o + 1) * k];
        let mut acc = 0.0f32;
        for j in 0..k {
            acc += arow[j] * wrow[j];
        }
        *out = acc;
    });
}

/// dst[m,n] = a[m,k] * w_bf16[n,k]^T casting weights to f32 on the fly.
fn matmul_nt_bf16w(dst: &mut [f32], a: &[f32], w: &[bf16], m: usize, k: usize, n: usize) {
    let _ = m;
    dst.par_iter_mut().enumerate().for_each(|(f, out)| {
        let mi = f / n;
        let o = f % n;
        let arow = &a[mi * k..(mi + 1) * k];
        let wrow = &w[o * k..(o + 1) * k];
        let mut acc = 0.0f32;
        for j in 0..k {
            acc += arow[j] * wrow[j].to_f32();
        }
        *out = acc;
    });
}

/// In-place RMSNorm with the Qwen3.5 `(1 + weight)` convention.
/// `x` is `[rows, d]` row-major; one call handles every row.
fn rms_norm_1plus(x: &mut [f32], weight: &[f32], eps: f32, d: usize) {
    let rows = x.len() / d;
    for r in 0..rows {
        let row = &mut x[r * d..(r + 1) * d];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (mean_sq + eps).sqrt();
        for i in 0..d {
            row[i] = row[i] * inv * (1.0 + weight[i]);
        }
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bestv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bestv {
            bestv = x;
            best = i;
        }
    }
    best
}

pub struct Scratch {
    pub x_norm: Vec<f32>,
    pub qkv: Vec<f32>,
    pub conv: Vec<f32>,
    pub z: Vec<f32>,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub q_heads: Vec<f32>,
    pub k_heads: Vec<f32>,
    pub v_heads: Vec<f32>,
    pub z_heads: Vec<f32>,
    pub o_heads: Vec<f32>,
    pub qhat: Vec<f32>,
    pub khat: Vec<f32>,
    pub state: Vec<f32>,
    pub aq: Vec<f32>,
    pub ak: Vec<f32>,
    pub av: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub mlp_gate: Vec<f32>,
    pub mlp_up: Vec<f32>,
    pub mlp_down: Vec<f32>,
    pub out_proj_tmp: Vec<f32>,
    pub w_qkv: Vec<f32>,
    pub w_z: Vec<f32>,
    pub w_a: Vec<f32>,
    pub w_b: Vec<f32>,
    pub w_out: Vec<f32>,
    pub w_conv: Vec<f32>,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_o: Vec<f32>,
    pub logits: Vec<f32>,
}

pub struct CpuQwen35 {
    pub hidden: usize,
    pub intermediate: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub nk: usize,
    pub nv: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    pub conv_dim: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub rotary_dim: usize,
    pub eps: f32,
    pub weights: Qwen35Weights,
    pub rope_cos: Vec<f32>,
    pub rope_sin: Vec<f32>,
    pub eos: u32,
    pub scratch: Scratch,
}

impl CpuQwen35 {
    pub fn load(model_dir: &std::path::Path) -> Result<Self, String> {
        let weights = Qwen35Weights::load(model_dir)?;
        let c = &weights.config;
        let hidden = c.hidden_size;
        let intermediate = c.intermediate_size;
        let head_dim = c.head_dim;
        let nk = c.linear_num_key_heads;
        let nv = c.linear_num_value_heads;
        let key_head_dim = c.linear_key_head_dim;
        let value_head_dim = c.linear_value_head_dim;
        let k_dim = nk * key_head_dim;
        let v_dim = nv * value_head_dim;
        let conv_dim = k_dim * 2 + v_dim;
        let rotary_dim = c.rotary_dim();
        let theta = c.rope_theta() as f32;

        let n_attn_q = c.num_attention_heads * head_dim * 2;
        let n_attn_kv = c.num_key_value_heads * head_dim;

        let mut rope_cos = vec![0.0f32; MAX_LEN * (rotary_dim / 2)];
        let mut rope_sin = vec![0.0f32; MAX_LEN * (rotary_dim / 2)];
        let half = rotary_dim / 2;
        for p in 0..MAX_LEN {
            let pos = p as f32;
            for i in 0..half {
                let freq = pos * theta.powf(-(i as f32) / (half as f32));
                rope_cos[p * half + i] = freq.cos();
                rope_sin[p * half + i] = freq.sin();
            }
        }

        let eos = c.eos_token_id.unwrap_or(248044) as u32;

        let scratch = Scratch {
            x_norm: vec![0.0; MAX_LEN * hidden],
            qkv: vec![0.0; MAX_LEN * conv_dim],
            conv: vec![0.0; MAX_LEN * conv_dim],
            z: vec![0.0; MAX_LEN * v_dim],
            a: vec![0.0; MAX_LEN * nv],
            b: vec![0.0; MAX_LEN * nv],
            q_heads: vec![0.0; MAX_LEN * nv * value_head_dim],
            k_heads: vec![0.0; MAX_LEN * nv * value_head_dim],
            v_heads: vec![0.0; MAX_LEN * nv * value_head_dim],
            z_heads: vec![0.0; MAX_LEN * nv * value_head_dim],
            o_heads: vec![0.0; MAX_LEN * nv * value_head_dim],
            qhat: vec![0.0; MAX_LEN * nv * value_head_dim],
            khat: vec![0.0; MAX_LEN * nv * value_head_dim],
            state: vec![0.0; nv * key_head_dim * value_head_dim],
            aq: vec![0.0; MAX_LEN * n_attn_q],
            ak: vec![0.0; MAX_LEN * n_attn_kv],
            av: vec![0.0; MAX_LEN * n_attn_kv],
            attn_out: vec![0.0; MAX_LEN * c.num_attention_heads * head_dim],
            mlp_gate: vec![0.0; MAX_LEN * intermediate],
            mlp_up: vec![0.0; MAX_LEN * intermediate],
            mlp_down: vec![0.0; MAX_LEN * hidden],
            out_proj_tmp: vec![0.0; MAX_LEN * hidden],
            w_qkv: vec![0.0; conv_dim * hidden],
            w_z: vec![0.0; v_dim * hidden],
            w_a: vec![0.0; nv * hidden],
            w_b: vec![0.0; nv * hidden],
            w_out: vec![0.0; hidden * v_dim],
            w_conv: vec![0.0; conv_dim * 4],
            w_gate: vec![0.0; intermediate * hidden],
            w_up: vec![0.0; intermediate * hidden],
            w_down: vec![0.0; hidden * intermediate],
            w_q: vec![0.0; n_attn_q * hidden],
            w_k: vec![0.0; n_attn_kv * hidden],
            w_v: vec![0.0; n_attn_kv * hidden],
            w_o: vec![0.0; hidden * c.num_attention_heads * head_dim],
            logits: vec![0.0; c.vocab_size],
        };

        Ok(Self {
            hidden,
            intermediate,
            n_layers: c.num_hidden_layers,
            n_heads: c.num_attention_heads,
            n_kv_heads: c.num_key_value_heads,
            head_dim,
            nk,
            nv,
            k_dim,
            v_dim,
            conv_dim,
            key_head_dim,
            value_head_dim,
            rotary_dim,
            eps: c.rms_norm_eps as f32,
            weights,
            rope_cos,
            rope_sin,
            eos,
            scratch,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.weights.config.vocab_size
    }

    pub fn forward_last_logits(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let l = ids.len();
        if l == 0 || l > MAX_LEN {
            return Err(format!("unsupported sequence length {l}"));
        }
        let d = self.hidden;
        let mut x = vec![0.0f32; l * d];
        for (t, &id) in ids.iter().enumerate() {
            let row = &self.weights.embed.data[id as usize * d..(id as usize + 1) * d];
            for (j, v) in row.iter().enumerate() {
                x[t * d + j] = v.to_f32();
            }
        }

        let Self {
            hidden,
            intermediate,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            nk,
            nv,
            k_dim,
            v_dim,
            conv_dim,
            key_head_dim,
            value_head_dim,
            rotary_dim,
            eps,
            weights,
            rope_cos,
            rope_sin,
            eos: _,
            scratch,
        } = self;

        for li in 0..*n_layers {
            if li % 8 == 0 {
                eprintln!("  layer {li}/{}", *n_layers);
            }
            match &weights.layers[li] {
                LayerWeights::Full { .. } => full_attention_layer(
                    scratch,
                    &weights.layers[li],
                    &mut x,
                    l,
                    *hidden,
                    *intermediate,
                    *n_heads,
                    *n_kv_heads,
                    *head_dim,
                    *rotary_dim,
                    *eps,
                    rope_cos,
                    rope_sin,
                ),
                LayerWeights::Linear { .. } => gated_delta_layer(
                    scratch,
                    &weights.layers[li],
                    &mut x,
                    l,
                    *hidden,
                    *intermediate,
                    *nk,
                    *nv,
                    *k_dim,
                    *v_dim,
                    *conv_dim,
                    *key_head_dim,
                    *value_head_dim,
                    *eps,
                ),
            }
        }

        let final_norm = bf16vec_to_f32(&weights.final_norm);
        rms_norm_1plus(&mut x, &final_norm, *eps, d);

        // lm_head for the last row only
        let last = &x[(l - 1) * d..l * d];
        matmul_nt_bf16w(
            &mut scratch.logits,
            last,
            &weights.lm_head.data,
            1,
            d,
            weights.config.vocab_size,
        );
        Ok(scratch.logits.clone())
    }

    /// Greedy sampling over a full generation (recomputes hidden state each
    /// step; fine for short smoke tests).
    pub fn generate_greedy(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        ignore_eos: bool,
    ) -> Result<Vec<u32>, String> {
        let mut ids = prompt.to_vec();
        let mut out = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let logits = self.forward_last_logits(&ids)?;
            let next = argmax(&logits) as u32;
            out.push(next);
            if !ignore_eos && next == self.eos {
                break;
            }
            ids.push(next);
            if ids.len() > MAX_LEN {
                break;
            }
        }
        Ok(out)
    }
}

fn full_attention_layer(
    s: &mut Scratch,
    layer: &LayerWeights,
    x: &mut [f32],
    l: usize,
    hidden: usize,
    intermediate: usize,
    heads: usize,
    kv: usize,
    hd: usize,
    rotary_dim: usize,
    eps: f32,
    rope_cos: &[f32],
    rope_sin: &[f32],
) {
    let (q, k, v, o, q_norm, k_norm, gate, up, down, in_norm, post_norm) = match layer {
        LayerWeights::Full {
            q,
            k,
            v,
            o,
            q_norm,
            k_norm,
            gate,
            up,
            down,
            in_norm,
            post_norm,
        } => (q, k, v, o, q_norm, k_norm, gate, up, down, in_norm, post_norm),
        _ => return,
    };
    let in_norm_w = bf16vec_to_f32(in_norm);
    let post_norm_w = bf16vec_to_f32(post_norm);
    let qn = bf16vec_to_f32(q_norm);
    let kn = bf16vec_to_f32(k_norm);

    s.x_norm[..l * hidden].copy_from_slice(x);
    rms_norm_1plus(&mut s.x_norm[..l * hidden], &in_norm_w, eps, hidden);

    q4_to_f32(q, &mut s.w_q);
    q4_to_f32(k, &mut s.w_k);
    q4_to_f32(v, &mut s.w_v);
    let n_q = heads * hd * 2;
    let n_kv = kv * hd;
    matmul_nt(&mut s.aq[..l * n_q], &s.x_norm, &s.w_q, l, hidden, n_q);
    matmul_nt(&mut s.ak[..l * n_kv], &s.x_norm, &s.w_k, l, hidden, n_kv);
    matmul_nt(&mut s.av[..l * n_kv], &s.x_norm, &s.w_v, l, hidden, n_kv);

    // Per-head q/k RMSNorm (eps 1e-6) over hd dims, on the shared buffers.
    for t in 0..l {
        for h in 0..heads {
            let base = t * n_q + h * hd * 2;
            let mut mean = 0.0f32;
            for j in 0..hd {
                mean += s.aq[base + j] * s.aq[base + j];
            }
            let inv = 1.0 / (mean / hd as f32 + 1e-6).sqrt();
            for j in 0..hd {
                s.aq[base + j] = s.aq[base + j] * inv * (1.0 + qn[j]);
            }
        }
        for h in 0..kv {
            let base = t * n_kv + h * hd;
            let mut mean = 0.0f32;
            for j in 0..hd {
                mean += s.ak[base + j] * s.ak[base + j];
            }
            let inv = 1.0 / (mean / hd as f32 + 1e-6).sqrt();
            for j in 0..hd {
                s.ak[base + j] = s.ak[base + j] * inv * (1.0 + kn[j]);
            }
        }
    }

    // RoPE on first rotary_dim dims of q (stride n_q, offset h*hd*2) and k.
    let half = rotary_dim / 2;
    for t in 0..l {
        let cp = &rope_cos[t * half..(t + 1) * half];
        let sp = &rope_sin[t * half..(t + 1) * half];
        for h in 0..heads {
            let base = t * n_q + h * hd * 2;
            for i in 0..half {
                let a = s.aq[base + i];
                let b = s.aq[base + half + i];
                s.aq[base + i] = a * cp[i] - b * sp[i];
                s.aq[base + half + i] = a * sp[i] + b * cp[i];
            }
        }
        for h in 0..kv {
            let base = t * n_kv + h * hd;
            for i in 0..half {
                let a = s.ak[base + i];
                let b = s.ak[base + half + i];
                s.ak[base + i] = a * cp[i] - b * sp[i];
                s.ak[base + half + i] = a * sp[i] + b * cp[i];
            }
        }
    }

    // Attention with causal mask + GQA.
    let scale = 1.0 / (hd as f32).sqrt();
    let group = heads / kv;
    let mut scores = vec![0.0f32; l];
    for t in 0..l {
        for h in 0..heads {
            let kh = h / group;
            let qbase = t * n_q + h * hd * 2;
            let mut maxv = f32::NEG_INFINITY;
            for pos in 0..=t {
                let kbase = pos * n_kv + kh * hd;
                let mut acc = 0.0f32;
                for j in 0..hd {
                    acc += s.aq[qbase + j] * s.ak[kbase + j];
                }
                let sc = acc * scale;
                scores[pos] = sc;
                if sc > maxv {
                    maxv = sc;
                }
            }
            let mut sum = 0.0f32;
            for pos in 0..=t {
                let e = (scores[pos] - maxv).exp();
                scores[pos] = e;
                sum += e;
            }
            let obase = t * heads * hd + h * hd;
            for j in 0..hd {
                let mut acc = 0.0f32;
                for pos in 0..=t {
                    let vbase = pos * n_kv + kh * hd;
                    acc += scores[pos] * s.av[vbase + j];
                }
                s.attn_out[obase + j] = acc / sum;
            }
        }
    }

    // attn_out *= sigmoid(gate) where gate lives in the upper hd half of aq.
    for t in 0..l {
        for h in 0..heads {
            let gbase = t * n_q + h * hd * 2 + hd;
            let obase = t * heads * hd + h * hd;
            for j in 0..hd {
                s.attn_out[obase + j] *= sigmoid(s.aq[gbase + j]);
            }
        }
    }

    q4_to_f32(o, &mut s.w_o);
    matmul_nt(
        &mut s.out_proj_tmp[..l * hidden],
        &s.attn_out[..l * heads * hd],
        &s.w_o,
        l,
        heads * hd,
        hidden,
    );
    for i in 0..l * hidden {
        x[i] += s.out_proj_tmp[i];
    }

    mlp_block(s, x, l, hidden, intermediate, gate, up, down, &post_norm_w, false, eps);
}

fn mlp_block(
    s: &mut Scratch,
    x: &mut [f32],
    l: usize,
    hidden: usize,
    intermediate: usize,
    gate: &Q4Linear,
    up: &Q4Linear,
    down: &Q4Linear,
    post_norm_w: &[f32],
    _is_linear: bool,
    eps: f32,
) {
    s.x_norm[..l * hidden].copy_from_slice(x);
    rms_norm_1plus(&mut s.x_norm[..l * hidden], post_norm_w, eps, hidden);
    q4_to_f32(gate, &mut s.w_gate);
    q4_to_f32(up, &mut s.w_up);
    q4_to_f32(down, &mut s.w_down);
    matmul_nt(
        &mut s.mlp_gate[..l * intermediate],
        &s.x_norm,
        &s.w_gate,
        l,
        hidden,
        intermediate,
    );
    matmul_nt(
        &mut s.mlp_up[..l * intermediate],
        &s.x_norm,
        &s.w_up,
        l,
        hidden,
        intermediate,
    );
    for i in 0..l * intermediate {
        s.mlp_gate[i] = silu(s.mlp_gate[i]) * s.mlp_up[i];
    }
    matmul_nt(
        &mut s.mlp_down[..l * hidden],
        &s.mlp_gate,
        &s.w_down,
        l,
        intermediate,
        hidden,
    );
    for i in 0..l * hidden {
        x[i] += s.mlp_down[i];
    }
}

#[allow(clippy::too_many_arguments)]
fn gated_delta_layer(
    s: &mut Scratch,
    layer: &LayerWeights,
    x: &mut [f32],
    l: usize,
    hidden: usize,
    intermediate: usize,
    nk: usize,
    nv: usize,
    k_dim: usize,
    v_dim: usize,
    conv_dim: usize,
    kd: usize,
    vd: usize,
    eps: f32,
) {
    let (in_qkv, in_z, in_a, in_b, conv, a_log, dt_bias, norm, out_proj, gate, up, down, in_norm, post_norm) =
        match layer {
            LayerWeights::Linear {
                in_qkv,
                in_z,
                in_a,
                in_b,
                conv,
                a_log,
                dt_bias,
                norm,
                out_proj,
                gate,
                up,
                down,
                in_norm,
                post_norm,
            } => (
                in_qkv, in_z, in_a, in_b, conv, a_log, dt_bias, norm, out_proj, gate, up, down,
                in_norm, post_norm,
            ),
            _ => return,
        };
    let in_norm_w = bf16vec_to_f32(in_norm);
    let post_norm_w = bf16vec_to_f32(post_norm);
    let norm_w = bf16vec_to_f32(norm);

    s.x_norm[..l * hidden].copy_from_slice(x);
    rms_norm_1plus(&mut s.x_norm[..l * hidden], &in_norm_w, eps, hidden);

    q4_to_f32(in_qkv, &mut s.w_qkv);
    q4_to_f32(in_z, &mut s.w_z);
    bf16mat_to_f32(in_a, &mut s.w_a);
    bf16mat_to_f32(in_b, &mut s.w_b);
    bf16mat_to_f32(conv, &mut s.w_conv);
    match out_proj {
        MatKind::Q4(q) => q4_to_f32(q, &mut s.w_out),
        MatKind::Dense(den) => bf16mat_to_f32(den, &mut s.w_out),
    }

    matmul_nt(&mut s.qkv[..l * conv_dim], &s.x_norm, &s.w_qkv, l, hidden, conv_dim);
    matmul_nt(&mut s.z[..l * v_dim], &s.x_norm, &s.w_z, l, hidden, v_dim);
    matmul_nt(&mut s.a[..l * nv], &s.x_norm, &s.w_a, l, hidden, nv);
    matmul_nt(&mut s.b[..l * nv], &s.x_norm, &s.w_b, l, hidden, nv);

    // Causal depthwise conv (kernel 4) + silu, per conv_dim channel.
    for c in 0..conv_dim {
        for t in 0..l {
            let mut acc = 0.0f32;
            for j in 0..4usize {
                if t >= j {
                    acc += s.w_conv[c * 4 + j] * s.qkv[(t - j) * conv_dim + c];
                }
            }
            s.conv[t * conv_dim + c] = silu(acc);
        }
    }

    let group = nv / nk;
    for t in 0..l {
        for h in 0..nv {
            let r = h / group;
            let out_base = (t * nv + h) * vd;
            for j in 0..vd {
                s.v_heads[out_base + j] = s.conv[t * conv_dim + 2 * k_dim + h * vd + j];
                s.z_heads[out_base + j] = s.z[t * v_dim + h * vd + j];
            }
            for j in 0..kd {
                s.q_heads[out_base + j] = s.conv[t * conv_dim + r * kd + j];
                s.k_heads[out_base + j] = s.conv[t * conv_dim + k_dim + r * kd + j];
            }
        }
    }

    // L2-normalize q/k (eps 1e-6); q additionally scaled by kd^-0.5.
    let qscale = 1.0 / (kd as f32).sqrt();
    for t in 0..l {
        for h in 0..nv {
            let base = (t * nv + h) * kd;
            let mut ss = 0.0f32;
            for j in 0..kd {
                let vv = s.q_heads[base + j];
                ss += vv * vv;
            }
            let inv = 1.0 / (ss + 1e-6).sqrt();
            for j in 0..kd {
                s.qhat[base + j] = s.q_heads[base + j] * inv * qscale;
            }
            let mut ss2 = 0.0f32;
            for j in 0..kd {
                let vv = s.k_heads[base + j];
                ss2 += vv * vv;
            }
            let inv2 = 1.0 / (ss2 + 1e-6).sqrt();
            for j in 0..kd {
                s.khat[base + j] = s.k_heads[base + j] * inv2;
            }
        }
    }

    // Gated Delta Rule recurrence (state [nv, kd, vd], fp32).
    for i in 0..s.state.len() {
        s.state[i] = 0.0;
    }
    let mut kv_mem = vec![0.0f32; vd];
    let mut delta = vec![0.0f32; vd];
    for t in 0..l {
        for h in 0..nv {
            let sh = &mut s.state[h * kd * vd..(h + 1) * kd * vd];
            let beta = sigmoid(s.b[t * nv + h]);
            let g = -a_log[h].exp() * softplus(s.a[t * nv + h] + dt_bias[h]);
            let decay = g.exp();
            for vv in sh.iter_mut() {
                *vv *= decay;
            }
            let kb = (t * nv + h) * kd;
            for j in 0..vd {
                let mut acc = 0.0f32;
                for kk in 0..kd {
                    acc += sh[kk * vd + j] * s.khat[kb + kk];
                }
                kv_mem[j] = acc;
            }
            let vb = (t * nv + h) * vd;
            for j in 0..vd {
                delta[j] = (s.v_heads[vb + j] - kv_mem[j]) * beta;
            }
            for j in 0..vd {
                for kk in 0..kd {
                    sh[kk * vd + j] += s.khat[kb + kk] * delta[j];
                }
            }
            for j in 0..vd {
                let mut acc = 0.0f32;
                for kk in 0..kd {
                    acc += sh[kk * vd + j] * s.qhat[kb + kk];
                }
                s.o_heads[vb + j] = acc;
            }
        }
    }

    // RMSNormGated over vd: rms(o)*(1+w) * silu(z).
    for t in 0..l {
        for h in 0..nv {
            let base = (t * nv + h) * vd;
            let mut mean = 0.0f32;
            for j in 0..vd {
                mean += s.o_heads[base + j] * s.o_heads[base + j];
            }
            let inv = 1.0 / (mean / vd as f32 + eps).sqrt();
            for j in 0..vd {
                s.o_heads[base + j] =
                    s.o_heads[base + j] * inv * (1.0 + norm_w[j]) * silu(s.z_heads[base + j]);
            }
        }
    }

    matmul_nt(
        &mut s.out_proj_tmp[..l * hidden],
        &s.o_heads[..l * v_dim],
        &s.w_out,
        l,
        v_dim,
        hidden,
    );
    for i in 0..l * hidden {
        x[i] += s.out_proj_tmp[i];
    }

    mlp_block(s, x, l, hidden, intermediate, gate, up, down, &post_norm_w, true, eps);
}
