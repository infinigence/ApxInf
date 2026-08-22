//! CUDA eager + incremental forward for Qwen3.5.
//!
//! Weights are uploaded once. Prefill runs the parallel batched path
//! (C4 flash attention / tiled INT4 GEMM) and stores per-layer incremental
//! state: K/V cache for full-attention layers and the GatedDeltaNet state
//! plus the last 3 causal-conv taps for linear layers. Decode then advances
//! one token at a time through the caches (C5).

#![cfg(feature = "cuda")]

use half::bf16;

use apxinf_core::DType;
use apxinf_cuda::kernels::qwen35 as k;
use apxinf_cuda::{CudaBuffer, CudaContext};

use crate::qwen35::weights::{Bf16Mat, LayerWeights, MatKind, Q4Linear, Qwen35Weights};

const MAX_LEN: usize = 1024; // parallel prefill length cap (workspace)
const MAX_SEQ: usize = 4096; // total sequence cap (rope + KV cache)

struct GpuQ4 {
    packed: CudaBuffer,
    scale: CudaBuffer,
    zp: CudaBuffer,
    out: usize,
    inp: usize,
}

struct GpuMlp {
    gate: GpuQ4,
    up: GpuQ4,
    down: GpuQ4,
}

enum GpuOutProj {
    Q4(GpuQ4),
    Dense(CudaBuffer), // transposed [v_dim, hidden]
}

struct GpuFull {
    wq: GpuQ4,
    wk: GpuQ4,
    wv: GpuQ4,
    wo: GpuQ4,
    q_norm_w: CudaBuffer,
    k_norm_w: CudaBuffer,
    in_norm_w: CudaBuffer,
    post_norm_w: CudaBuffer,
    mlp: GpuMlp,
    kv_k: CudaBuffer, // [MAX_SEQ, kv_heads * hd] bf16
    kv_v: CudaBuffer,
}

struct GpuLinear {
    in_qkv: GpuQ4,
    in_z: GpuQ4,
    in_a: CudaBuffer,
    in_b: CudaBuffer,
    conv_w: CudaBuffer,
    a_log: CudaBuffer, // [nv] f32
    dt_bias: CudaBuffer,
    norm_w: CudaBuffer,
    out_proj: GpuOutProj,
    in_norm_w: CudaBuffer,
    post_norm_w: CudaBuffer,
    mlp: GpuMlp,
    state: CudaBuffer,     // [nv, kd, vd] f32
    conv_hist: CudaBuffer, // [3, conv_dim] bf16; hist[0] = t-1
}

enum GpuLayer {
    Full(GpuFull),
    Linear(GpuLinear),
}

pub struct CudaQwen35 {
    ctx: CudaContext,
    hidden: usize,
    inter: usize,
    n_layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    nk: usize,
    nv: usize,
    kd: usize,
    vd: usize,
    conv_dim: usize,
    vocab: usize,
    eps: f32,
    rotary_half: usize,
    lm_head: CudaBuffer, // [hidden, vocab]
    final_w: CudaBuffer,
    rope_cos: CudaBuffer, // [MAX_SEQ, half]
    rope_sin: CudaBuffer,
    layers: Vec<GpuLayer>,
    ws: Workspace,
    embed: Vec<bf16>,
    seq_len: usize,
}

struct Workspace {
    x: CudaBuffer,
    xn: CudaBuffer,
    xn2: CudaBuffer,
    fnorm: CudaBuffer,
    qkv: CudaBuffer,
    conv: CudaBuffer,
    z: CudaBuffer,
    z2: CudaBuffer,
    o_delta: CudaBuffer,
    o_norm: CudaBuffer,
    a: CudaBuffer,
    b: CudaBuffer,
    beta: CudaBuffer,
    g: CudaBuffer,
    qh: CudaBuffer,
    kh: CudaBuffer,
    v_delta: CudaBuffer,
    qg: CudaBuffer,
    q: CudaBuffer,
    gate: CudaBuffer,
    qn: CudaBuffer,
    kn: CudaBuffer,
    attn_v: CudaBuffer,
    attn_out: CudaBuffer,
    o_out: CudaBuffer,
    mlp_gate: CudaBuffer,
    mlp_up: CudaBuffer,
    mlp_hid: CudaBuffer,
    mlp_down: CudaBuffer,
    logits: CudaBuffer,
}

fn bf16_bytes(v: &[bf16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn i32_bytes(v: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn upload(dev: usize, bytes: &[u8]) -> Result<CudaBuffer, String> {
    let b = CudaBuffer::alloc(bytes.len(), dev)?;
    b.copy_from_host(bytes)?;
    Ok(b)
}

fn upload_transposed(dev: usize, m: &Bf16Mat) -> Result<CudaBuffer, String> {
    let mut out = vec![0u8; m.rows * m.cols * 2];
    for i in 0..m.cols {
        for o in 0..m.rows {
            let src = m.data[o * m.cols + i];
            out[(i * m.rows + o) * 2..(i * m.rows + o) * 2 + 2]
                .copy_from_slice(&src.to_le_bytes());
        }
    }
    upload(dev, &out)
}

fn norm_plus1(v: &[bf16]) -> Vec<bf16> {
    v.iter().map(|x| bf16::from_f32(x.to_f32() + 1.0)).collect()
}

fn upload_q4(dev: usize, q: &Q4Linear) -> Result<GpuQ4, String> {
    Ok(GpuQ4 {
        packed: upload(dev, &i32_bytes(&q.packed))?,
        scale: upload(dev, &bf16_bytes(&q.scale))?,
        zp: upload(dev, &i32_bytes(&q.zp))?,
        out: q.out,
        inp: q.inp,
    })
}

fn rope_tables(theta: f64, half: usize, max_len: usize) -> (Vec<bf16>, Vec<bf16>) {
    let mut cos = Vec::with_capacity(max_len * half);
    let mut sin = Vec::with_capacity(max_len * half);
    for t in 0..max_len {
        for i in 0..half {
            let angle = (t as f64) * theta.powf(-(i as f64) / (half as f64));
            cos.push(bf16::from_f32(angle.cos() as f32));
            sin.push(bf16::from_f32(angle.sin() as f32));
        }
    }
    (cos, sin)
}

impl CudaQwen35 {
    pub fn load(model_dir: &std::path::Path) -> Result<Self, String> {
        let w = Qwen35Weights::load(model_dir)?;
        let cfg = &w.config;
        let hidden = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let n_layers = cfg.num_hidden_layers;
        let heads = cfg.num_attention_heads;
        let kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;
        let nk = cfg.linear_num_key_heads;
        let nv = cfg.linear_num_value_heads;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let conv_dim = 2 * nk * kd + nv * vd;
        let vocab = cfg.vocab_size;
        let eps = cfg.rms_norm_eps as f32;
        let rotary_dim = cfg.rotary_dim();
        let half = rotary_dim / 2;

        let ctx = CudaContext::new(0)?;
        let dev = ctx.device_id();

        let lm_head = upload_transposed(dev, &w.lm_head)?;
        let final_w = upload(dev, &bf16_bytes(&norm_plus1(&w.final_norm)))?;
        let (cos, sin) = rope_tables(cfg.rope_theta(), half, MAX_SEQ);
        let rope_cos = upload(dev, &bf16_bytes(&cos))?;
        let rope_sin = upload(dev, &bf16_bytes(&sin))?;

        let mut layers = Vec::with_capacity(n_layers);
        for lw in &w.layers {
            layers.push(match lw {
                LayerWeights::Full { q, k, v, o, q_norm, k_norm, gate, up, down, in_norm, post_norm } => {
                    GpuLayer::Full(GpuFull {
                        wq: upload_q4(dev, q)?,
                        wk: upload_q4(dev, k)?,
                        wv: upload_q4(dev, v)?,
                        wo: upload_q4(dev, o)?,
                        q_norm_w: upload(dev, &bf16_bytes(&norm_plus1(q_norm)))?,
                        k_norm_w: upload(dev, &bf16_bytes(&norm_plus1(k_norm)))?,
                        in_norm_w: upload(dev, &bf16_bytes(&norm_plus1(in_norm)))?,
                        post_norm_w: upload(dev, &bf16_bytes(&norm_plus1(post_norm)))?,
                        mlp: GpuMlp {
                            gate: upload_q4(dev, gate)?,
                            up: upload_q4(dev, up)?,
                            down: upload_q4(dev, down)?,
                        },
                        kv_k: CudaBuffer::alloc(MAX_SEQ * kv_heads * head_dim * 2, dev)?,
                        kv_v: CudaBuffer::alloc(MAX_SEQ * kv_heads * head_dim * 2, dev)?,
                    })
                }
                LayerWeights::Linear { in_qkv, in_z, in_a, in_b, conv, a_log, dt_bias, norm, out_proj, gate, up, down, in_norm, post_norm } => {
                    GpuLayer::Linear(GpuLinear {
                        in_qkv: upload_q4(dev, in_qkv)?,
                        in_z: upload_q4(dev, in_z)?,
                        in_a: upload_transposed(dev, in_a)?,
                        in_b: upload_transposed(dev, in_b)?,
                        conv_w: upload(dev, &bf16_bytes(&conv.data))?,
                        a_log: upload(dev, &f32_bytes(a_log))?,
                        dt_bias: upload(dev, &f32_bytes(dt_bias))?,
                        norm_w: upload(dev, &bf16_bytes(&norm_plus1(norm)))?,
                        out_proj: match out_proj {
                            MatKind::Q4(qq) => GpuOutProj::Q4(upload_q4(dev, qq)?),
                            MatKind::Dense(d) => GpuOutProj::Dense(upload_transposed(dev, d)?),
                        },
                        in_norm_w: upload(dev, &bf16_bytes(&norm_plus1(in_norm)))?,
                        post_norm_w: upload(dev, &bf16_bytes(&norm_plus1(post_norm)))?,
                        mlp: GpuMlp {
                            gate: upload_q4(dev, gate)?,
                            up: upload_q4(dev, up)?,
                            down: upload_q4(dev, down)?,
                        },
                        state: CudaBuffer::alloc(nv * kd * vd * 4, dev)?,
                        conv_hist: CudaBuffer::alloc_zeros(3 * conv_dim * 2, dev)?,
                    })
                }
            });
        }

        let n = |x: usize| CudaBuffer::alloc(x, dev);
        let ws = Workspace {
            x: n(MAX_LEN * hidden * 2)?,
            xn: n(MAX_LEN * hidden * 2)?,
            xn2: n(MAX_LEN * hidden * 2)?,
            fnorm: n(MAX_LEN * hidden * 2)?,
            qkv: n(MAX_LEN * conv_dim * 2)?,
            conv: n(MAX_LEN * conv_dim * 2)?,
            z: n(MAX_LEN * nv * vd * 2)?,
            z2: n(MAX_LEN * nv * vd * 2)?,
            o_delta: n(MAX_LEN * nv * vd * 2)?,
            o_norm: n(MAX_LEN * nv * vd * 2)?,
            a: n(MAX_LEN * nv * 2)?,
            b: n(MAX_LEN * nv * 2)?,
            beta: n(MAX_LEN * nv * 2)?,
            g: n(MAX_LEN * nv * 2)?,
            qh: n(MAX_LEN * nv * kd * 2)?,
            kh: n(MAX_LEN * nv * kd * 2)?,
            v_delta: n(MAX_LEN * nv * vd * 2)?,
            qg: n(MAX_LEN * heads * 2 * head_dim * 2)?,
            q: n(MAX_LEN * heads * head_dim * 2)?,
            gate: n(MAX_LEN * heads * head_dim * 2)?,
            qn: n(MAX_LEN * heads * head_dim * 2)?,
            kn: n(MAX_LEN * kv_heads * head_dim * 2)?,
            attn_v: n(MAX_LEN * kv_heads * head_dim * 2)?,
            attn_out: n(MAX_LEN * heads * head_dim * 2)?,
            o_out: n(MAX_LEN * hidden * 2)?,
            mlp_gate: n(MAX_LEN * inter * 2)?,
            mlp_up: n(MAX_LEN * inter * 2)?,
            mlp_hid: n(MAX_LEN * inter * 2)?,
            mlp_down: n(MAX_LEN * hidden * 2)?,
            logits: n(vocab * 2)?,
        };

        Ok(Self {
            ctx,
            hidden,
            inter,
            n_layers,
            heads,
            kv_heads,
            head_dim,
            nk,
            nv,
            kd,
            vd,
            conv_dim,
            vocab,
            eps,
            rotary_half: half,
            lm_head,
            final_w,
            rope_cos,
            rope_sin,
            layers,
            ws,
            embed: w.embed.data,
            seq_len: 0,
        })
    }

    fn gemm(&self, m: usize, n: usize, kk: usize, a: &CudaBuffer, b: &CudaBuffer, c: &CudaBuffer) -> Result<(), String> {
        self.ctx.cublas().gemm(DType::BF16, m, n, kk, 1.0f32, a, b, 0.0f32, c)
    }

    fn gemm_q4(&self, q: &GpuQ4, m: usize, a: &CudaBuffer, c: &CudaBuffer) -> Result<(), String> {
        k::gemm_w4a16_bf16(&self.ctx, a, &q.packed, &q.scale, &q.zp, c, m, q.out, q.inp)
            .map_err(|e| e.to_string())
    }

    fn rms(&self, x: &CudaBuffer, w: &CudaBuffer, out: &CudaBuffer, rows: usize, cols: usize) -> Result<(), String> {
        k::rms_norm_bf16_into(&self.ctx, x, w, out, rows, cols, self.eps).map_err(|e| e.to_string())
    }

    fn l2norm(&self, x: &CudaBuffer, out: &CudaBuffer, rows: usize, cols: usize, scale: f32) -> Result<(), String> {
        k::l2norm_bf16_into(&self.ctx, x, out, rows, cols, 1e-6f32, scale).map_err(|e| e.to_string())
    }

    fn mlp_run(&self, mlp: &GpuMlp, post_w: &CudaBuffer, l: usize) -> Result<(), String> {
        self.rms(&self.ws.x, post_w, &self.ws.xn2, l, self.hidden)?;
        self.gemm_q4(&mlp.gate, l, &self.ws.xn2, &self.ws.mlp_gate)?;
        self.gemm_q4(&mlp.up, l, &self.ws.xn2, &self.ws.mlp_up)?;
        k::silu_bf16_into(&self.ctx, &self.ws.mlp_gate, &self.ws.mlp_gate, l * self.inter).map_err(|e| e.to_string())?;
        k::mul_bf16_into(&self.ctx, &self.ws.mlp_gate, &self.ws.mlp_up, &self.ws.mlp_hid, l * self.inter).map_err(|e| e.to_string())?;
        self.gemm_q4(&mlp.down, l, &self.ws.mlp_hid, &self.ws.mlp_down)?;
        k::accum_bf16_into(&self.ctx, &self.ws.x, &self.ws.mlp_down, l * self.hidden).map_err(|e| e.to_string())?;
        Ok(())
    }

    fn full_layer_prefill(&self, layer: &GpuFull, l: usize) -> Result<(), String> {
        self.rms(&self.ws.x, &layer.in_norm_w, &self.ws.xn, l, self.hidden)?;
        let hd = self.head_dim;
        self.gemm_q4(&layer.wq, l, &self.ws.xn, &self.ws.qg)?;
        self.gemm_q4(&layer.wk, l, &self.ws.xn, &self.ws.kn)?;
        self.gemm_q4(&layer.wv, l, &self.ws.xn, &self.ws.attn_v)?;
        k::qg_split_bf16_into(&self.ctx, &self.ws.qg, &self.ws.q, &self.ws.gate, l * self.heads * hd, self.heads, hd).map_err(|e| e.to_string())?;
        self.rms(&self.ws.q, &layer.q_norm_w, &self.ws.qn, l * self.heads, hd)?;
        self.rms(&self.ws.kn, &layer.k_norm_w, &self.ws.kn, l * self.kv_heads, hd)?;
        k::partial_rope_bf16_inplace(&self.ctx, &self.ws.qn, &self.rope_cos, &self.rope_sin, l * self.heads, self.heads, hd, self.rotary_half, MAX_SEQ, 0).map_err(|e| e.to_string())?;
        k::partial_rope_bf16_inplace(&self.ctx, &self.ws.kn, &self.rope_cos, &self.rope_sin, l * self.kv_heads, self.kv_heads, hd, self.rotary_half, MAX_SEQ, 0).map_err(|e| e.to_string())?;
        // cache K/V for incremental decode
        k::copy_bf16(&self.ctx, &self.ws.kn, &layer.kv_k, l * self.kv_heads * hd).map_err(|e| e.to_string())?;
        k::copy_bf16(&self.ctx, &self.ws.attn_v, &layer.kv_v, l * self.kv_heads * hd).map_err(|e| e.to_string())?;
        k::attention_bf16_into(&self.ctx, &self.ws.qn, &self.ws.kn, &self.ws.attn_v, &self.ws.gate, &self.ws.attn_out, l, self.heads, self.kv_heads, hd).map_err(|e| e.to_string())?;
        self.gemm_q4(&layer.wo, l, &self.ws.attn_out, &self.ws.o_out)?;
        k::accum_bf16_into(&self.ctx, &self.ws.x, &self.ws.o_out, l * self.hidden).map_err(|e| e.to_string())?;
        self.mlp_run(&layer.mlp, &layer.post_norm_w, l)
    }

    fn linear_layer_prefill(&self, layer: &GpuLinear, l: usize) -> Result<(), String> {
        self.rms(&self.ws.x, &layer.in_norm_w, &self.ws.xn, l, self.hidden)?;
        self.gemm_q4(&layer.in_qkv, l, &self.ws.xn, &self.ws.qkv)?;
        k::conv_silu_bf16_into(&self.ctx, &self.ws.qkv, &layer.conv_w, &self.ws.conv, l, self.conv_dim).map_err(|e| e.to_string())?;
        self.gemm_q4(&layer.in_z, l, &self.ws.xn, &self.ws.z)?;
        self.gemm(l, self.nv, self.hidden, &self.ws.xn, &layer.in_a, &self.ws.a)?;
        self.gemm(l, self.nv, self.hidden, &self.ws.xn, &layer.in_b, &self.ws.b)?;
        k::beta_g_bf16(&self.ctx, &self.ws.a, &self.ws.b, &layer.a_log, &layer.dt_bias, &self.ws.beta, &self.ws.g, l * self.nv, self.nv).map_err(|e| e.to_string())?;
        k::conv_split_bf16_into(&self.ctx, &self.ws.conv, &self.ws.qh, &self.ws.kh, &self.ws.v_delta, l, self.nk, self.nv, self.kd, self.vd, self.conv_dim).map_err(|e| e.to_string())?;
        let qscale = 1.0f32 / (self.kd as f32).sqrt();
        self.l2norm(&self.ws.qh, &self.ws.qh, l * self.nv, self.kd, qscale)?;
        self.l2norm(&self.ws.kh, &self.ws.kh, l * self.nv, self.kd, 1.0f32)?;
        // recurrence into the per-layer persistent state (kernel zeroes first)
        k::delta_recurrence_bf16_into(&self.ctx, &self.ws.qh, &self.ws.kh, &self.ws.v_delta, &self.ws.beta, &self.ws.g, &layer.state, &self.ws.o_delta, l, self.nv, self.kd, self.vd).map_err(|e| e.to_string())?;
        self.rms(&self.ws.o_delta, &layer.norm_w, &self.ws.o_norm, l * self.nv, self.vd)?;
        k::silu_bf16_into(&self.ctx, &self.ws.z, &self.ws.z2, l * self.nv * self.vd).map_err(|e| e.to_string())?;
        k::mul_bf16_into(&self.ctx, &self.ws.o_norm, &self.ws.z2, &self.ws.z, l * self.nv * self.vd).map_err(|e| e.to_string())?;
        match &layer.out_proj {
            GpuOutProj::Q4(q) => self.gemm_q4(q, l, &self.ws.z, &self.ws.o_out)?,
            GpuOutProj::Dense(b) => self.gemm(l, self.hidden, self.nv * self.vd, &self.ws.z, b, &self.ws.o_out)?,
        }
        k::accum_bf16_into(&self.ctx, &self.ws.x, &self.ws.o_out, l * self.hidden).map_err(|e| e.to_string())?;
        // stash the last 3 pre-conv qkv rows for the incremental conv taps
        // (hist[0] = qkv[l-1] = newest, hist[2] = qkv[l-3])
        for (j, slot) in [0usize, 1, 2].iter().enumerate() {
            if j >= l {
                break;
            }
            let src = self.ws.qkv.view((l - 1 - j) * self.conv_dim * 2, self.conv_dim * 2)?;
            let dst = layer.conv_hist.view(slot * self.conv_dim * 2, self.conv_dim * 2)?;
            k::copy_bf16(&self.ctx, &src, &dst, self.conv_dim).map_err(|e| e.to_string())?;
        }
        self.mlp_run(&layer.mlp, &layer.post_norm_w, l)
    }

    fn full_layer_decode(&self, layer: &GpuFull, pos: usize) -> Result<(), String> {
        let hd = self.head_dim;
        self.rms(&self.ws.x, &layer.in_norm_w, &self.ws.xn, 1, self.hidden)?;
        self.gemm_q4(&layer.wq, 1, &self.ws.xn, &self.ws.qg)?;
        self.gemm_q4(&layer.wk, 1, &self.ws.xn, &self.ws.kn)?;
        self.gemm_q4(&layer.wv, 1, &self.ws.xn, &self.ws.attn_v)?;
        k::qg_split_bf16_into(&self.ctx, &self.ws.qg, &self.ws.q, &self.ws.gate, self.heads * hd, self.heads, hd).map_err(|e| e.to_string())?;
        self.rms(&self.ws.q, &layer.q_norm_w, &self.ws.qn, self.heads, hd)?;
        self.rms(&self.ws.kn, &layer.k_norm_w, &self.ws.kn, self.kv_heads, hd)?;
        k::partial_rope_bf16_inplace(&self.ctx, &self.ws.qn, &self.rope_cos, &self.rope_sin, self.heads, self.heads, hd, self.rotary_half, MAX_SEQ, pos).map_err(|e| e.to_string())?;
        k::partial_rope_bf16_inplace(&self.ctx, &self.ws.kn, &self.rope_cos, &self.rope_sin, self.kv_heads, self.kv_heads, hd, self.rotary_half, MAX_SEQ, pos).map_err(|e| e.to_string())?;
        // append this token's K/V
        let kslot = layer.kv_k.view(pos * self.kv_heads * hd * 2, self.kv_heads * hd * 2)?;
        let vslot = layer.kv_v.view(pos * self.kv_heads * hd * 2, self.kv_heads * hd * 2)?;
        k::copy_bf16(&self.ctx, &self.ws.kn, &kslot, self.kv_heads * hd).map_err(|e| e.to_string())?;
        k::copy_bf16(&self.ctx, &self.ws.attn_v, &vslot, self.kv_heads * hd).map_err(|e| e.to_string())?;
        // attend over cache [0..=pos]
        k::attention_decode_bf16_into(&self.ctx, &self.ws.qn, &layer.kv_k, &layer.kv_v, &self.ws.gate, &self.ws.attn_out, pos + 1, self.heads, self.kv_heads, hd).map_err(|e| e.to_string())?;
        self.gemm_q4(&layer.wo, 1, &self.ws.attn_out, &self.ws.o_out)?;
        k::accum_bf16_into(&self.ctx, &self.ws.x, &self.ws.o_out, self.hidden).map_err(|e| e.to_string())?;
        self.mlp_run(&layer.mlp, &layer.post_norm_w, 1)
    }

    fn linear_layer_decode(&self, layer: &GpuLinear, pos: usize) -> Result<(), String> {
        let _ = pos;
        self.rms(&self.ws.x, &layer.in_norm_w, &self.ws.xn, 1, self.hidden)?;
        self.gemm_q4(&layer.in_qkv, 1, &self.ws.xn, &self.ws.qkv)?;
        k::conv_step_silu_bf16_into(&self.ctx, &self.ws.qkv, &layer.conv_hist, &layer.conv_w, &self.ws.conv, self.conv_dim).map_err(|e| e.to_string())?;
        self.gemm_q4(&layer.in_z, 1, &self.ws.xn, &self.ws.z)?;
        self.gemm(1, self.nv, self.hidden, &self.ws.xn, &layer.in_a, &self.ws.a)?;
        self.gemm(1, self.nv, self.hidden, &self.ws.xn, &layer.in_b, &self.ws.b)?;
        k::beta_g_bf16(&self.ctx, &self.ws.a, &self.ws.b, &layer.a_log, &layer.dt_bias, &self.ws.beta, &self.ws.g, self.nv, self.nv).map_err(|e| e.to_string())?;
        k::conv_split_bf16_into(&self.ctx, &self.ws.conv, &self.ws.qh, &self.ws.kh, &self.ws.v_delta, 1, self.nk, self.nv, self.kd, self.vd, self.conv_dim).map_err(|e| e.to_string())?;
        let qscale = 1.0f32 / (self.kd as f32).sqrt();
        self.l2norm(&self.ws.qh, &self.ws.qh, self.nv, self.kd, qscale)?;
        self.l2norm(&self.ws.kh, &self.ws.kh, self.nv, self.kd, 1.0f32)?;
        k::delta_step_bf16_into(&self.ctx, &self.ws.qh, &self.ws.kh, &self.ws.v_delta, &self.ws.beta, &self.ws.g, &layer.state, &self.ws.o_delta, self.nv, self.kd, self.vd).map_err(|e| e.to_string())?;
        self.rms(&self.ws.o_delta, &layer.norm_w, &self.ws.o_norm, self.nv, self.vd)?;
        k::silu_bf16_into(&self.ctx, &self.ws.z, &self.ws.z2, self.nv * self.vd).map_err(|e| e.to_string())?;
        k::mul_bf16_into(&self.ctx, &self.ws.o_norm, &self.ws.z2, &self.ws.z, self.nv * self.vd).map_err(|e| e.to_string())?;
        match &layer.out_proj {
            GpuOutProj::Q4(q) => self.gemm_q4(q, 1, &self.ws.z, &self.ws.o_out)?,
            GpuOutProj::Dense(b) => self.gemm(1, self.hidden, self.nv * self.vd, &self.ws.z, b, &self.ws.o_out)?,
        }
        k::accum_bf16_into(&self.ctx, &self.ws.x, &self.ws.o_out, self.hidden).map_err(|e| e.to_string())?;
        self.mlp_run(&layer.mlp, &layer.post_norm_w, 1)
    }

    fn read_logits(&self) -> Result<Vec<f32>, String> {
        let mut bytes = vec![0u8; self.vocab * 2];
        self.ws.logits.copy_to_host(&mut bytes)?;
        let mut out = Vec::with_capacity(self.vocab);
        for j in 0..self.vocab {
            let v = bf16::from_le_bytes([bytes[j * 2], bytes[j * 2 + 1]]).to_f32();
            out.push(v);
        }
        Ok(out)
    }

    fn apply_logits(&self, l: usize) -> Result<(), String> {
        self.rms(&self.ws.x, &self.final_w, &self.ws.fnorm, l, self.hidden)?;
        let last = self.ws.fnorm.view((l - 1) * self.hidden * 2, self.hidden * 2)?;
        self.gemm(1, self.vocab, self.hidden, &last, &self.lm_head, &self.ws.logits)?;
        self.ctx.synchronize()?;
        Ok(())
    }

    /// Parallel prefill; caches per-layer incremental state and returns the
    /// logits of the final token (bf16 -> f32).
    pub fn prefill_logits(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let l = ids.len();
        if l == 0 || l > MAX_LEN {
            return Err(format!("unsupported prefill length {l} (max {MAX_LEN})"));
        }
        self.seq_len = 0;
        let mut host_x = vec![0u8; l * self.hidden * 2];
        for (t, id) in ids.iter().enumerate() {
            let row = *id as usize;
            if row >= self.embed.len() / self.hidden {
                return Err(format!("token id {id} out of range"));
            }
            for j in 0..self.hidden {
                let src = self.embed[row * self.hidden + j];
                host_x[(t * self.hidden + j) * 2..(t * self.hidden + j) * 2 + 2]
                    .copy_from_slice(&src.to_le_bytes());
            }
        }
        self.ws.x.copy_from_host(&host_x)?;
        // reset per-layer conv histories (stale rows would leak into a short prefill)
        let zeros = vec![0u8; 3 * self.conv_dim * 2];
        for i in 0..self.n_layers {
            match &self.layers[i] {
                GpuLayer::Full(fl) => self.full_layer_prefill(fl, l)?,
                GpuLayer::Linear(li) => {
                    li.conv_hist.copy_from_host(&zeros)?;
                    self.linear_layer_prefill(li, l)?;
                }
            }
            if std::env::var_os("Q35_DUMP").is_some() {
                let mut bytes = vec![0u8; l * self.hidden * 2];
                let _ = self.ws.x.copy_to_host(&mut bytes);
                let _ = std::fs::write(format!("/tmp/gpu_x_{i}.bin"), &bytes);
            }
        }
        self.apply_logits(l)?;
        self.seq_len = l;
        self.read_logits()
    }

    /// One incremental decode step; appends to caches and returns the next
    /// token logits.
    pub fn decode_logits(&mut self, token: u32) -> Result<Vec<f32>, String> {
        if self.seq_len >= MAX_SEQ {
            return Err(format!("sequence longer than MAX_SEQ ({MAX_SEQ})"));
        }
        let pos = self.seq_len;
        let row = token as usize;
        if row >= self.embed.len() / self.hidden {
            return Err(format!("token id {token} out of range"));
        }
        let mut host_x = vec![0u8; self.hidden * 2];
        for j in 0..self.hidden {
            let src = self.embed[row * self.hidden + j];
            host_x[j * 2..j * 2 + 2].copy_from_slice(&src.to_le_bytes());
        }
        self.ws.x.copy_from_host(&host_x)?;
        for i in 0..self.n_layers {
            match &self.layers[i] {
                GpuLayer::Full(fl) => self.full_layer_decode(fl, pos)?,
                GpuLayer::Linear(li) => self.linear_layer_decode(li, pos)?,
            }
        }
        self.apply_logits(1)?;
        self.seq_len += 1;
        self.read_logits()
    }

    /// Compatibility wrapper: prefill and return the final-token logits.
    pub fn forward_last_logits(&mut self, ids: &[u32]) -> Result<Vec<f32>, String> {
        self.prefill_logits(ids)
    }
}
