//! CUDA eager forward for Qwen3.5 (C1 correctness path).
//!
//! AWQ INT4 weights stay packed on device (i32 packed + bf16 scale + i32
//! zero points) and are dequantized per projection into a reusable bf16
//! `[in, out]` staging buffer before each cuBLAS GEMM. Dense bf16 weights are
//! uploaded transposed so every GEMM is `C[M,K] @ B[K,N]` (row-major).

#![cfg(feature = "cuda")]

use half::bf16;

use apxinf_core::DType;
use apxinf_cuda::kernels::qwen35 as k;
use apxinf_cuda::{CudaBuffer, CudaContext};

use crate::qwen35::weights::{Bf16Mat, LayerWeights, MatKind, Q4Linear, Qwen35Weights};

const MAX_LEN: usize = 128;

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
    q_norm_w: CudaBuffer,  // [hd] (1+w)
    k_norm_w: CudaBuffer,  // [hd] (1+w)
    in_norm_w: CudaBuffer, // [hidden] (1+w)
    post_norm_w: CudaBuffer,
    mlp: GpuMlp,
}

struct GpuLinear {
    in_qkv: GpuQ4,
    in_z: GpuQ4,
    in_a: CudaBuffer,   // transposed [hidden, nv]
    in_b: CudaBuffer,
    conv_w: CudaBuffer, // [conv_dim, 4] flat
    a_log: CudaBuffer, // [nv] f32
    dt_bias: CudaBuffer, // [nv] f32
    norm_w: CudaBuffer, // [vd] (1+w)
    out_proj: GpuOutProj,
    in_norm_w: CudaBuffer,
    post_norm_w: CudaBuffer,
    mlp: GpuMlp,
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
    rotary_dim: usize,
    lm_head: CudaBuffer, // [hidden, vocab]
    final_w: CudaBuffer,
    rope_cos: CudaBuffer,
    rope_sin: CudaBuffer,
    layers: Vec<GpuLayer>,
    ws: Workspace,
    embed: Vec<bf16>,
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
    qg: CudaBuffer,
    q: CudaBuffer,
    gate: CudaBuffer,
    qn: CudaBuffer,
    kn: CudaBuffer,
    attn_v: CudaBuffer,
    v_delta: CudaBuffer,
    attn_out: CudaBuffer,
    o_out: CudaBuffer,
    mlp_gate: CudaBuffer,
    mlp_up: CudaBuffer,
    mlp_hid: CudaBuffer,
    mlp_down: CudaBuffer,
    logits: CudaBuffer,
    state: CudaBuffer,
}

fn bf16_bytes(v: &[bf16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 2);
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

fn i32_bytes(v: &[i32]) -> Vec<u8> {
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
        let (cos, sin) = rope_tables(cfg.rope_theta(), half, MAX_LEN);
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
                            MatKind::Q4(q) => GpuOutProj::Q4(upload_q4(dev, q)?),
                            MatKind::Dense(d) => GpuOutProj::Dense(upload_transposed(dev, d)?),
                        },
                        in_norm_w: upload(dev, &bf16_bytes(&norm_plus1(in_norm)))?,
                        post_norm_w: upload(dev, &bf16_bytes(&norm_plus1(post_norm)))?,
                        mlp: GpuMlp {
                            gate: upload_q4(dev, gate)?,
                            up: upload_q4(dev, up)?,
                            down: upload_q4(dev, down)?,
                        },
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
            qg: n(MAX_LEN * heads * 2 * head_dim * 2)?,
            q: n(MAX_LEN * heads * head_dim * 2)?,
            gate: n(MAX_LEN * heads * head_dim * 2)?,
            qn: n(MAX_LEN * heads * head_dim * 2)?,
            kn: n(MAX_LEN * kv_heads * head_dim * 2)?,
            attn_v: n(MAX_LEN * kv_heads * head_dim * 2)?,
            v_delta: n(MAX_LEN * nv * vd * 2)?,
            attn_out: n(MAX_LEN * heads * head_dim * 2)?,
            o_out: n(MAX_LEN * hidden * 2)?,
            mlp_gate: n(MAX_LEN * inter * 2)?,
            mlp_up: n(MAX_LEN * inter * 2)?,
            mlp_hid: n(MAX_LEN * inter * 2)?,
            mlp_down: n(MAX_LEN * hidden * 2)?,
            logits: n(vocab * 2)?,
            state: n(nv * kd * vd * 4)?,
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
            rotary_dim,
            lm_head,
            final_w,
            rope_cos,
            rope_sin,
            layers,
            ws,
            embed: w.embed.data,
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
        k::silu_bf16_into(&self.ctx, &self.ws.mlp_gate, &self.ws.mlp_gate, (l * self.inter) as usize).map_err(|e| e.to_string())?;
        k::mul_bf16_into(&self.ctx, &self.ws.mlp_gate, &self.ws.mlp_up, &self.ws.mlp_hid, (l * self.inter) as usize).map_err(|e| e.to_string())?;
        self.gemm_q4(&mlp.down, l, &self.ws.mlp_hid, &self.ws.mlp_down)?;
        k::accum_bf16_into(&self.ctx, &self.ws.x, &self.ws.mlp_down, (l * self.hidden) as usize).map_err(|e| e.to_string())?;
        Ok(())
    }

    fn linear_layer(&self, layer: &GpuLinear, l: usize, dump0: bool) -> Result<(), String> {
        self.rms(&self.ws.x, &layer.in_norm_w, &self.ws.xn, l, self.hidden)?;
        if dump0 { self.dump_buf(&self.ws.xn, "xn", l * self.hidden * 2); }
        // QKV + conv
        self.gemm_q4(&layer.in_qkv, l, &self.ws.xn, &self.ws.qkv)?;
        if dump0 { self.dump_buf(&self.ws.qkv, "qkv", l * self.conv_dim * 2); }
        k::conv_silu_bf16_into(&self.ctx, &self.ws.qkv, &layer.conv_w, &self.ws.conv, l, self.conv_dim).map_err(|e| e.to_string())?;
        if dump0 { self.dump_buf(&self.ws.conv, "conv", l * self.conv_dim * 2); }
        // z, a, b
        self.gemm_q4(&layer.in_z, l, &self.ws.xn, &self.ws.z)?;
        if dump0 { self.dump_buf(&self.ws.z, "z", l * self.nv * self.vd * 2); }
        self.gemm(l, self.nv, self.hidden, &self.ws.xn, &layer.in_a, &self.ws.a)?;
        self.gemm(l, self.nv, self.hidden, &self.ws.xn, &layer.in_b, &self.ws.b)?;
        if dump0 { self.dump_buf(&self.ws.a, "a", l * self.nv * 2); self.dump_buf(&self.ws.b, "b", l * self.nv * 2); }
        k::beta_g_bf16(&self.ctx, &self.ws.a, &self.ws.b, &layer.a_log, &layer.dt_bias, &self.ws.beta, &self.ws.g, l * self.nv, self.nv).map_err(|e| e.to_string())?;
        if dump0 { self.dump_buf(&self.ws.beta, "beta", l * self.nv * 2); self.dump_buf(&self.ws.g, "g", l * self.nv * 2); }
        // split conv -> q/k expanded + v
        k::conv_split_bf16_into(
            &self.ctx, &self.ws.conv, &self.ws.qh, &self.ws.kh, &self.ws.v_delta,
            l, self.nk, self.nv, self.kd, self.vd, self.conv_dim,
        ).map_err(|e| e.to_string())?;
        if dump0 { self.dump_buf(&self.ws.v_delta, "v", l * self.nv * self.vd * 2); }
        // L2 normalize q, k
        let qscale = 1.0f32 / (self.kd as f32).sqrt();
        self.l2norm(&self.ws.qh, &self.ws.qh, l * self.nv, self.kd, qscale)?;
        self.l2norm(&self.ws.kh, &self.ws.kh, l * self.nv, self.kd, 1.0f32)?;
        if dump0 { self.dump_buf(&self.ws.qh, "q", l * self.nv * self.kd * 2); self.dump_buf(&self.ws.kh, "k", l * self.nv * self.kd * 2); }
        // recurrence
        k::delta_recurrence_bf16_into(
            &self.ctx, &self.ws.qh, &self.ws.kh, &self.ws.v_delta,
            &self.ws.beta, &self.ws.g, &self.ws.state, &self.ws.o_delta,
            l, self.nv, self.kd, self.vd,
        ).map_err(|e| e.to_string())?;
        if dump0 { self.dump_buf(&self.ws.o_delta, "O", l * self.nv * self.vd * 2); }
        // RMSNormGated: o_norm = rms(o)* (1+w) ; z2 = silu(z); out = mul
        self.rms(&self.ws.o_delta, &layer.norm_w, &self.ws.o_norm, l * self.nv, self.vd)?;
        if dump0 { self.dump_buf(&self.ws.o_norm, "o_norm", l * self.nv * self.vd * 2); }
        k::silu_bf16_into(&self.ctx, &self.ws.z, &self.ws.z2, (l * self.nv * self.vd) as usize).map_err(|e| e.to_string())?;
        if dump0 { self.dump_buf(&self.ws.z2, "z_silu", l * self.nv * self.vd * 2); }
        k::mul_bf16_into(&self.ctx, &self.ws.o_norm, &self.ws.z2, &self.ws.z, (l * self.nv * self.vd) as usize).map_err(|e| e.to_string())?;
        if dump0 { self.dump_buf(&self.ws.z, "o_gated", l * self.nv * self.vd * 2); }
        // out projection + residual
        match &layer.out_proj {
            GpuOutProj::Q4(q) => self.gemm_q4(q, l, &self.ws.z, &self.ws.o_out)?,
            GpuOutProj::Dense(b) => self.gemm(l, self.hidden, self.nv * self.vd, &self.ws.z, b, &self.ws.o_out)?,
        }
        if dump0 { self.dump_buf(&self.ws.o_out, "out_proj", l * self.hidden * 2); }
        k::accum_bf16_into(&self.ctx, &self.ws.x, &self.ws.o_out, (l * self.hidden) as usize).map_err(|e| e.to_string())?;
        // MLP
        self.mlp_run(&layer.mlp, &layer.post_norm_w, l)
    }

    fn full_layer(&self, layer: &GpuFull, l: usize, _dump0: bool) -> Result<(), String> {
        self.rms(&self.ws.x, &layer.in_norm_w, &self.ws.xn, l, self.hidden)?;
        let hdim = self.head_dim;
        self.gemm_q4(&layer.wq, l, &self.ws.xn, &self.ws.qg)?;
        self.gemm_q4(&layer.wk, l, &self.ws.xn, &self.ws.kn)?; // kn buffer [L*kv*hd]
        self.gemm_q4(&layer.wv, l, &self.ws.xn, &self.ws.attn_v)?;
        k::qg_split_bf16_into(&self.ctx, &self.ws.qg, &self.ws.q, &self.ws.gate, (l * self.heads * hdim) as usize, self.heads, hdim).map_err(|e| e.to_string())?;
        // per-head q/k norm + partial RoPE
        self.rms(&self.ws.q, &layer.q_norm_w, &self.ws.qn, l * self.heads, hdim)?;
        self.rms(&self.ws.kn, &layer.k_norm_w, &self.ws.kn, l * self.kv_heads, hdim)?;
        let half = self.rotary_dim / 2;
        k::partial_rope_bf16_inplace(&self.ctx, &self.ws.qn, &self.rope_cos, &self.rope_sin, l * self.heads, self.heads, hdim, half, MAX_LEN).map_err(|e| e.to_string())?;
        k::partial_rope_bf16_inplace(&self.ctx, &self.ws.kn, &self.rope_cos, &self.rope_sin, l * self.kv_heads, self.kv_heads, hdim, half, MAX_LEN).map_err(|e| e.to_string())?;
        // attention (qn, kn, attn_v, gate -> attn_out) with sigmoid gate
        k::attention_bf16_into(
            &self.ctx, &self.ws.qn, &self.ws.kn, &self.ws.attn_v, &self.ws.gate,
            &self.ws.attn_out, l, self.heads, self.kv_heads, hdim,
        ).map_err(|e| e.to_string())?;
        // o projection + residual
        self.gemm_q4(&layer.wo, l, &self.ws.attn_out, &self.ws.o_out)?;
        k::accum_bf16_into(&self.ctx, &self.ws.x, &self.ws.o_out, (l * self.hidden) as usize).map_err(|e| e.to_string())?;
        // MLP
        self.mlp_run(&layer.mlp, &layer.post_norm_w, l)
    }

    fn dump_x(&self, name: &str, l: usize) {
        let bytes = l * self.hidden * 2;
        let mut h = vec![0u8; bytes];
        let _ = self.ws.x.copy_to_host(&mut h);
        let _ = std::fs::write(format!("/tmp/{name}"), &h);
    }

    fn dump_buf(&self, b: &CudaBuffer, name: &str, bytes: usize) {
        let mut h = vec![0u8; bytes];
        let _ = b.copy_to_host(&mut h);
        let _ = std::fs::write(format!("/tmp/g0_{name}.bin"), &h);
    }

    /// Full-sequence forward; returns logits (bf16 -> f32) for the last token.
    pub fn forward_last_logits(&self, ids: &[u32]) -> Result<Vec<f32>, String> {
        let l = ids.len();
        if l == 0 || l > MAX_LEN {
            return Err(format!("unsupported sequence length {l} (max {MAX_LEN})"));
        }
        // Host-side embedding lookup, then upload.
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
        let dump = std::env::var_os("Q35_DUMP").is_some();
        self.ws.x.copy_from_host(&host_x)?;
        if dump {
            let _ = std::fs::write("/tmp/gpu_x_emb.bin", &host_x);
        }

        for i in 0..self.n_layers {
            match &self.layers[i] {
                GpuLayer::Full(fl) => self.full_layer(fl, l, i == 0)?,
                GpuLayer::Linear(li) => self.linear_layer(li, l, i == 0)?,
            }
            if dump {
                self.dump_x(&format!("gpu_x_{i}.bin"), l);
            }
        }

        // final norm (all rows) then GEMM on the last row only.
        self.rms(&self.ws.x, &self.final_w, &self.ws.fnorm, l, self.hidden)?;
        let last = self.ws.fnorm.view((l - 1) * self.hidden * 2, self.hidden * 2)?;
        self.gemm(1, self.vocab, self.hidden, &last, &self.lm_head, &self.ws.logits)?;
        self.ctx.synchronize()?;

        let mut bytes = vec![0u8; self.vocab * 2];
        self.ws.logits.copy_to_host(&mut bytes)?;
        let mut out = Vec::with_capacity(self.vocab);
        for j in 0..self.vocab {
            let v = bf16::from_le_bytes([bytes[j * 2], bytes[j * 2 + 1]]).to_f32();
            out.push(v);
        }
        Ok(out)
    }
}
