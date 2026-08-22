//! GPU-resident Qwen3.5 forward path.
//!
//! Mirrors the CPU reference math in [`super::general`] but keeps activations,
//! KV caches, and the linear-attention recurrent states on the device:
//!
//! - packed INT4 GEMMs use the fused dequant-GEMM kernel for single-row
//!   (decode) shapes and dequant + cublas for multi-row prefill chunks;
//! - the linear-attention layers run the causal conv + SiLU and the gated
//!   delta-rule recurrence as one kernel launch per layer;
//! - full-attention layers write q_norm/k_norm + partial-RoPE directly into
//!   per-layer K/V caches and use flash attention (decode or split-warp
//!   prefill);
//! - prompts longer than [`CHUNK`] are processed in chunks. Chunking is
//!   exact: the linear layers are sequential recurrences and the full
//!   attention is causal, so chunk boundaries change nothing but buffer sizes.

use std::collections::HashMap;
use std::sync::Arc;

use apxinf_core::{Backend, DType, Error, Result, Shape, Tensor};
use apxinf_cuda::buffer::{CudaBuffer, HostMappedBuffer};
use apxinf_cuda::kernels;
use apxinf_cuda::{CudaBackend, CudaContext};

use super::{LayerKind, Qwen35Config};

/// Prefill chunk size: bounds all activation workspace buffers.
const CHUNK: usize = 512;
/// KV cache rows per full-attention layer. The base evaluation never
/// exceeds 16384 prompt tokens + 128 output; 16640 leaves a small margin
/// while freeing ~1 GB of VRAM versus the declared 32768 (longer requests
/// are rejected by the service as a capacity error).
pub const MAX_SEQ_LEN: usize = 16640;

struct Gemm {
    packed: Option<GemmPacked>,
    dense: Option<CudaBuffer>,
    out_cols: usize,
    in_cols: usize,
}

struct GemmPacked {
    w: CudaBuffer,
    scale: CudaBuffer,
    zp: CudaBuffer,
    groups: usize,
}

struct LinearLayer {
    in_norm_w: CudaBuffer,
    post_norm_w: CudaBuffer,
    qkv: Gemm,
    z: Gemm,
    a: Gemm,
    b: Gemm,
    out: Gemm,
    conv_w: CudaBuffer,
    a_log: CudaBuffer,
    dt_bias: CudaBuffer,
    gate_norm_w: CudaBuffer,
    conv_state: CudaBuffer,
    recurrent: CudaBuffer,
    gate: Gemm,
    up: Gemm,
    down: Gemm,
    conv_dim: usize,
    kdim: usize,
    vdim: usize,
    v_heads: usize,
    k_heads: usize,
    conv_kernel: usize,
}

struct FullLayer {
    in_norm_w: CudaBuffer,
    post_norm_w: CudaBuffer,
    q: Gemm,
    k: Gemm,
    v: Gemm,
    o: Gemm,
    q_norm_w: CudaBuffer,
    k_norm_w: CudaBuffer,
    k_cache: CudaBuffer,
    v_cache: CudaBuffer,
    gate: Gemm,
    up: Gemm,
    down: Gemm,
}

enum CudaLayer {
    Linear(LinearLayer),
    Full(FullLayer),
}

pub struct Qwen35Cuda {
    backend: Arc<dyn Backend>,
    embed: CudaBuffer,
    lm_head: CudaBuffer,
    final_norm_w: CudaBuffer,
    layers: Vec<CudaLayer>,
    hidden: usize,
    intermediate: usize,
    vocab: usize,
    eps: f32,
    rope_theta: f32,
    rotary_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    // workspace
    x: CudaBuffer,
    normed: CudaBuffer,
    normed2: CudaBuffer,
    qkv: CudaBuffer,
    z: CudaBuffer,
    a: CudaBuffer,
    b: CudaBuffer,
    delta_out: CudaBuffer,
    qk_scratch: CudaBuffer,
    attn_scores: CudaBuffer,
    attn_l: CudaBuffer,
    attn_kt: CudaBuffer,
    attn_vf32: CudaBuffer,
    attn_pv: CudaBuffer,
    gated: CudaBuffer,
    attn: CudaBuffer,
    attn2: CudaBuffer,
    gate_proj: CudaBuffer,
    up_proj: CudaBuffer,
    mlp_act: CudaBuffer,
    q_gate: CudaBuffer,
    q_buf: CudaBuffer,
    gate_buf: CudaBuffer,
    k_buf: CudaBuffer,
    v_buf: CudaBuffer,
    logits: CudaBuffer,
    dense_scratch: CudaBuffer,
    ids: CudaBuffer,
    pos: CudaBuffer,
    argmax_out: HostMappedBuffer,
}

/// Buffers extracted from a linear layer for one run call.
struct LinearRun {
    in_norm_w: CudaBuffer,
    qkv: Gemm,
    z: Gemm,
    a: Gemm,
    b: Gemm,
    out: Gemm,
    conv_w: CudaBuffer,
    a_log: CudaBuffer,
    dt_bias: CudaBuffer,
    gate_norm_w: CudaBuffer,
    conv_state: CudaBuffer,
    recurrent: CudaBuffer,
    conv_dim: usize,
    kdim: usize,
    vdim: usize,
    v_heads: usize,
    k_heads: usize,
    conv_kernel: usize,
    gate: Gemm,
    up: Gemm,
    down: Gemm,
    post_norm_w: CudaBuffer,
}

/// Buffers extracted from a full-attention layer for one run call.
struct FullRun {
    in_norm_w: CudaBuffer,
    q: Gemm,
    k: Gemm,
    v: Gemm,
    o: Gemm,
    q_norm_w: CudaBuffer,
    k_norm_w: CudaBuffer,
    k_cache: CudaBuffer,
    v_cache: CudaBuffer,
    gate: Gemm,
    up: Gemm,
    down: Gemm,
    post_norm_w: CudaBuffer,
}

impl Qwen35Cuda {
    /// Build the GPU state from the model's device tensors. `tensors` holds
    /// the CPU-side map for small 1-D parameters (norms, A_log, dt_bias).
    #[allow(clippy::too_many_lines)]
    pub fn new(
        backend: Arc<dyn Backend>,
        config: &Qwen35Config,
        tensors: &HashMap<String, Tensor>,
        device_tensors: &HashMap<String, Tensor>,
    ) -> Result<Self> {
        let cb = backend
            .as_any()
            .downcast_ref::<CudaBackend>()
            .ok_or_else(|| Error::Other("qwen3_5 CUDA state requires a CudaBackend".into()))?;
        let device = cb.device_id();
        let tc = &config.text;
        let hidden = tc.hidden_size;
        let intermediate = tc.intermediate_size;
        let vocab = tc.vocab_size;
        let rotary_dim = (tc.head_dim as f32 * tc.partial_rotary_factor) as usize & !1usize;

        let mut layers = Vec::with_capacity(tc.n_layers);
        for l in 0..tc.n_layers {
            let prefix = format!("model.language_model.layers.{l}");
            match tc.layer_types[l] {
                LayerKind::LinearAttention => {
                    let attn = format!("{prefix}.linear_attn");
                    let k_heads = tc.linear_num_key_heads;
                    let v_heads = tc.linear_num_value_heads;
                    let kdim = tc.linear_key_head_dim;
                    let vdim = tc.linear_value_head_dim;
                    let key_dim = k_heads * kdim;
                    let value_dim = v_heads * vdim;
                    let conv_dim = key_dim * 2 + value_dim;
                    let kernel = tc.linear_conv_kernel_dim;
                    layers.push(CudaLayer::Linear(LinearLayer {
                        in_norm_w: upload_norm_plus_one(
                            device,
                            tensors,
                            &format!("{prefix}.input_layernorm.weight"),
                            hidden,
                        )?,
                        post_norm_w: upload_norm_plus_one(
                            device,
                            tensors,
                            &format!("{prefix}.post_attention_layernorm.weight"),
                            hidden,
                        )?,
                        qkv: gemm(device_tensors, &format!("{attn}.in_proj_qkv"), conv_dim, hidden)?,
                        z: gemm(device_tensors, &format!("{attn}.in_proj_z"), value_dim, hidden)?,
                        a: gemm(device_tensors, &format!("{attn}.in_proj_a"), v_heads, hidden)?,
                        b: gemm(device_tensors, &format!("{attn}.in_proj_b"), v_heads, hidden)?,
                        out: gemm(device_tensors, &format!("{attn}.out_proj"), hidden, value_dim)?,
                        conv_w: upload_bf16_flat(device, tensors, &format!("{attn}.conv1d.weight"))?,
                        a_log: upload_bf16(device, tensors, &format!("{attn}.A_log"))?,
                        dt_bias: upload_bf16(device, tensors, &format!("{attn}.dt_bias"))?,
                        gate_norm_w: upload_bf16(device, tensors, &format!("{attn}.norm.weight"))?,
                        conv_state: CudaBuffer::alloc_zeros((kernel - 1) * conv_dim * 4, device)
                            .map_err(Error::Cuda)?,
                        recurrent: CudaBuffer::alloc_zeros(v_heads * kdim * vdim * 4, device)
                            .map_err(Error::Cuda)?,
                        gate: gemm(
                            device_tensors,
                            &format!("{prefix}.mlp.gate_proj"),
                            intermediate,
                            hidden,
                        )?,
                        up: gemm(
                            device_tensors,
                            &format!("{prefix}.mlp.up_proj"),
                            intermediate,
                            hidden,
                        )?,
                        down: gemm(
                            device_tensors,
                            &format!("{prefix}.mlp.down_proj"),
                            hidden,
                            intermediate,
                        )?,
                        conv_dim,
                        kdim,
                        vdim,
                        v_heads,
                        k_heads,
                        conv_kernel: kernel,
                    }));
                }
                LayerKind::FullAttention => {
                    let attn = format!("{prefix}.self_attn");
                    let head_dim = tc.head_dim;
                    let n_kv = tc.n_kv_heads;
                    let cache_bytes = n_kv * MAX_SEQ_LEN * head_dim * 2;
                    layers.push(CudaLayer::Full(FullLayer {
                        in_norm_w: upload_norm_plus_one(
                            device,
                            tensors,
                            &format!("{prefix}.input_layernorm.weight"),
                            hidden,
                        )?,
                        post_norm_w: upload_norm_plus_one(
                            device,
                            tensors,
                            &format!("{prefix}.post_attention_layernorm.weight"),
                            hidden,
                        )?,
                        q: gemm(
                            device_tensors,
                            &format!("{attn}.q_proj"),
                            tc.n_heads * head_dim * 2,
                            hidden,
                        )?,
                        k: gemm(
                            device_tensors,
                            &format!("{attn}.k_proj"),
                            n_kv * head_dim,
                            hidden,
                        )?,
                        v: gemm(
                            device_tensors,
                            &format!("{attn}.v_proj"),
                            n_kv * head_dim,
                            hidden,
                        )?,
                        o: gemm(
                            device_tensors,
                            &format!("{attn}.o_proj"),
                            hidden,
                            tc.n_heads * head_dim,
                        )?,
                        q_norm_w: upload_norm_plus_one(
                            device,
                            tensors,
                            &format!("{attn}.q_norm.weight"),
                            head_dim,
                        )?,
                        k_norm_w: upload_norm_plus_one(
                            device,
                            tensors,
                            &format!("{attn}.k_norm.weight"),
                            head_dim,
                        )?,
                        k_cache: CudaBuffer::alloc_zeros(cache_bytes, device)
                            .map_err(Error::Cuda)?,
                        v_cache: CudaBuffer::alloc_zeros(cache_bytes, device)
                            .map_err(Error::Cuda)?,
                        gate: gemm(
                            device_tensors,
                            &format!("{prefix}.mlp.gate_proj"),
                            intermediate,
                            hidden,
                        )?,
                        up: gemm(
                            device_tensors,
                            &format!("{prefix}.mlp.up_proj"),
                            intermediate,
                            hidden,
                        )?,
                        down: gemm(
                            device_tensors,
                            &format!("{prefix}.mlp.down_proj"),
                            hidden,
                            intermediate,
                        )?,
                    }));
                }
            }
        }

        let ws = |rows: usize, cols: usize| {
            CudaBuffer::alloc_zeros(rows * cols * 2, device).map_err(Error::Cuda)
        };
        Ok(Self {
            backend,
            embed: buffer_from(device_tensors, "model.language_model.embed_tokens.weight")?,
            lm_head: buffer_from(device_tensors, "lm_head.weight")?,
            final_norm_w: upload_norm_plus_one(
                device,
                tensors,
                "model.language_model.norm.weight",
                hidden,
            )?,
            layers,
            hidden,
            intermediate,
            vocab,
            eps: tc.rms_norm_eps,
            rope_theta: tc.rope_theta,
            rotary_dim,
            n_heads: tc.n_heads,
            n_kv_heads: tc.n_kv_heads,
            head_dim: tc.head_dim,
            x: ws(CHUNK, hidden)?,
            normed: ws(CHUNK, hidden)?,
            normed2: ws(CHUNK, hidden)?,
            qkv: ws(CHUNK, 10240)?,
            z: ws(CHUNK, 6144)?,
            a: ws(CHUNK, 48)?,
            b: ws(CHUNK, 48)?,
            delta_out: ws(CHUNK, 6144)?,
            qk_scratch: ws(CHUNK, 4096)?, // 16 k_heads * 2 * 128 kdim bf16
            attn_scores: CudaBuffer::alloc_zeros(
                6 * CHUNK * MAX_SEQ_LEN * 4,
                device,
            )
            .map_err(|e| Error::Other(format!("attn scores alloc: {e}")))?,
            attn_l: CudaBuffer::alloc_zeros(
                CHUNK * tc.n_heads * 4,
                device,
            )
            .map_err(|e| Error::Other(format!("attn l alloc: {e}")))?,
            attn_kt: CudaBuffer::alloc_zeros(
                tc.n_kv_heads * tc.head_dim * MAX_SEQ_LEN * 2,
                device,
            )
            .map_err(|e| Error::Other(format!("attn kt alloc: {e}")))?,
            attn_vf32: CudaBuffer::alloc_zeros(
                tc.n_kv_heads * MAX_SEQ_LEN * tc.head_dim * 4,
                device,
            )
            .map_err(|e| Error::Other(format!("attn vf32 alloc: {e}")))?,
            attn_pv: CudaBuffer::alloc_zeros(
                CHUNK * tc.n_heads * tc.head_dim * 4,
                device,
            )
            .map_err(|e| Error::Other(format!("attn pv alloc: {e}")))?,
            gated: ws(CHUNK, 6144)?,
            attn: ws(CHUNK, tc.n_heads * tc.head_dim)?,
            attn2: ws(CHUNK, hidden)?,
            gate_proj: ws(CHUNK, intermediate)?,
            up_proj: ws(CHUNK, intermediate)?,
            mlp_act: ws(CHUNK, intermediate)?,
            q_gate: ws(CHUNK, tc.n_heads * tc.head_dim * 2)?,
            q_buf: ws(CHUNK, tc.n_heads * tc.head_dim)?,
            gate_buf: ws(CHUNK, tc.n_heads * tc.head_dim)?,
            k_buf: ws(CHUNK, tc.n_kv_heads * tc.head_dim)?,
            v_buf: ws(CHUNK, tc.n_kv_heads * tc.head_dim)?,
            logits: ws(1, vocab)?,
            dense_scratch: CudaBuffer::alloc_zeros(
                intermediate * hidden * 2,
                device,
            )
            .map_err(Error::Cuda)?,
            ids: CudaBuffer::alloc(CHUNK * 4, device).map_err(Error::Cuda)?,
            pos: CudaBuffer::alloc(4, device).map_err(Error::Cuda)?,
            argmax_out: HostMappedBuffer::alloc(4, device).map_err(Error::Cuda)?,
        })
    }

    fn ctx(&self) -> &CudaContext {
        self.backend
            .as_any()
            .downcast_ref::<CudaBackend>()
            .expect("qwen3_5 CUDA backend downcast")
            .context()
    }

    /// Zero the recurrent/conv state (KV caches are position-indexed and do
    /// not need clearing).
    pub fn reset(&mut self) -> Result<()> {
        let backend = self.backend.clone();
        let stream = backend
            .as_any()
            .downcast_ref::<CudaBackend>()
            .expect("qwen3_5 CUDA backend downcast")
            .context()
            .stream()
            .clone();
        for layer in &mut self.layers {
            if let CudaLayer::Linear(l) = layer {
                l.conv_state.zero_async(&stream).map_err(Error::Cuda)?;
                l.recurrent.zero_async(&stream).map_err(Error::Cuda)?;
            }
        }
        Ok(())
    }

    /// Full GPU forward for `token_ids` starting at absolute position
    /// `start_pos`. Chunks long prompts; returns logits for the last token as
    /// a device tensor.
    pub fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        let seq = token_ids.len();
        if seq == 0 {
            return Err(Error::Other("qwen3_5 GPU forward: empty input".into()));
        }
        let perf = std::env::var_os("APXINF_PERF").is_some();
        let mut offset = 0usize;
        let mut last_chunk = 0usize;
        while offset < seq {
            let chunk = (seq - offset).min(CHUNK);
            let pos = start_pos + offset as u32;
            let t0 = std::time::Instant::now();
            self.forward_chunk(&token_ids[offset..offset + chunk], chunk, pos)?;
            let host_ms = t0.elapsed().as_secs_f32() * 1000.0;
            if perf {
                self.ctx().synchronize().map_err(Error::Cuda)?;
                let gpu_ms = t0.elapsed().as_secs_f32() * 1000.0;
                eprintln!("[chunk] len={chunk} : host {host_ms:.2} ms, gpu {gpu_ms:.2} ms");
            }
            last_chunk = chunk;
            offset += chunk;
        }
        self.final_logits(last_chunk)
    }

    /// Decode fast path: one token in, GPU argmax out.
    pub fn decode_token(&mut self, token: u32, pos: u32) -> Result<u32> {
        let perf = std::env::var_os("APXINF_PERF").is_some();
        let t0 = std::time::Instant::now();
        self.forward_chunk(std::slice::from_ref(&token), 1, pos)?;
        let t1 = std::time::Instant::now();
        let logits = self.final_logits(1)?;
        let logits_buf = CudaBuffer::from_tensor(&logits).map_err(Error::Cuda)?;
        kernels::qwen35::argmax_bf16(self.ctx(), &logits_buf, self.vocab, &self.argmax_out)?;
        self.ctx().synchronize().map_err(Error::Cuda)?;
        let t2 = std::time::Instant::now();
        let tok = self.argmax_out.read_u32(0).map_err(Error::Cuda)?;
        if perf {
            eprintln!(
                "[decode] layers={:.2}ms final+argmax={:.2}ms",
                (t1 - t0).as_secs_f32() * 1000.0,
                (t2 - t1).as_secs_f32() * 1000.0
            );
        }
        Ok(tok)
    }

    /// Debug helper: like [`Self::forward`] but writes the hidden state after
    /// every decoder layer to `dir` as raw f32 (`layer_NN.f32`).
    pub fn forward_dump(
        &mut self,
        token_ids: &[u32],
        start_pos: u32,
        dir: &std::path::Path,
    ) -> Result<Tensor> {
        let seq = token_ids.len();
        if seq == 0 {
            return Err(Error::Other("qwen3_5 GPU forward: empty input".into()));
        }
        let mut offset = 0usize;
        let mut last_chunk = 0usize;
        while offset < seq {
            let chunk = (seq - offset).min(CHUNK);
            last_chunk = chunk;
            let pos = start_pos + offset as u32;
            let ids = &token_ids[offset..offset + chunk];
            self.upload_ids(ids)?;
            self.upload_pos(pos)?;
            kernels::embedding::lookup_into(
                self.ctx(),
                DType::BF16,
                &self.embed,
                self.ids.address(),
                &self.x,
                self.hidden,
                chunk,
            )?;
            for l in 0..self.layers.len() {
                self.run_layer(l, chunk, pos)?;
                let mut bytes = vec![0u8; chunk * self.hidden * 2];
                self.x.copy_to_host(&mut bytes).map_err(Error::Cuda)?;
                let values: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect();
                let path = dir.join(format!("layer_{l:02}.f32"));
                let mut raw = Vec::with_capacity(values.len() * 4);
                for v in &values {
                    raw.extend_from_slice(&v.to_le_bytes());
                }
                std::fs::write(path, raw)
                    .map_err(|e| Error::Other(format!("dump layer {l}: {e}")))?;
            }
            offset += chunk;
        }
        self.final_logits(last_chunk)
    }

    fn forward_chunk(&mut self, ids: &[u32], seq: usize, start_pos: u32) -> Result<()> {
        self.upload_ids(ids)?;
        self.upload_pos(start_pos)?;
        let ctx = self.ctx();
        kernels::embedding::lookup_into(
            ctx,
            DType::BF16,
            &self.embed,
            self.ids.address(),
            &self.x,
            self.hidden,
            seq,
        )?;
        for l in 0..self.layers.len() {
            self.run_layer(l, seq, start_pos)?;
        }
        Ok(())
    }
    fn run_layer(&mut self, l: usize, seq: usize, start_pos: u32) -> Result<()> {
        let perf = std::env::var_os("APXINF_LAYER_PROF").is_some();
        let t0 = std::time::Instant::now();
        let result = match self.layers[l] {
            CudaLayer::Linear(_) => self.run_linear(l, seq),
            CudaLayer::Full(_) => self.run_full(l, seq, start_pos),
        };
        if perf {
            self.ctx().synchronize().map_err(Error::Cuda)?;
            let ms = t0.elapsed().as_secs_f32() * 1000.0;
            if ms > 0.3 {
                eprintln!("[layer {l}] seq={seq} : {ms:.2} ms");
            }
        }
        result
    }


    fn take_linear(&self, l: usize) -> LinearRun {
        let layer = match &self.layers[l] {
            CudaLayer::Linear(layer) => layer,
            _ => unreachable!(),
        };
        LinearRun {
            in_norm_w: layer.in_norm_w.clone(),
            qkv: gemm_clone(&layer.qkv),
            z: gemm_clone(&layer.z),
            a: gemm_clone(&layer.a),
            b: gemm_clone(&layer.b),
            out: gemm_clone(&layer.out),
            conv_w: layer.conv_w.clone(),
            a_log: layer.a_log.clone(),
            dt_bias: layer.dt_bias.clone(),
            gate_norm_w: layer.gate_norm_w.clone(),
            conv_state: layer.conv_state.clone(),
            recurrent: layer.recurrent.clone(),
            conv_dim: layer.conv_dim,
            kdim: layer.kdim,
            vdim: layer.vdim,
            v_heads: layer.v_heads,
            k_heads: layer.k_heads,
            conv_kernel: layer.conv_kernel,
            gate: gemm_clone(&layer.gate),
            up: gemm_clone(&layer.up),
            down: gemm_clone(&layer.down),
            post_norm_w: layer.post_norm_w.clone(),
        }
    }

    fn run_linear(&mut self, l: usize, seq: usize) -> Result<()> {
        let ctx = self.ctx();
        let mut run = self.take_linear(l);
        if l == 0 {
            trace_buf("gpu_x_pre", &self.x, seq * self.hidden);
        }
        kernels::norm::rms_into(
            ctx, DType::BF16, &self.x, &run.in_norm_w, &self.normed, self.hidden, seq, self.eps,
        )?;
        if l == 0 {
            trace_buf("gpu_normed", &self.normed, seq * self.hidden);
        }
        gemm_run(ctx, &run.qkv, &self.normed, &self.qkv, seq, &self.dense_scratch)?;
        if l == 0 {
            trace_buf("gpu_qkv_pre", &self.qkv, seq * run.conv_dim);
        }
        gemm_run(ctx, &run.z, &self.normed, &self.z, seq, &self.dense_scratch)?;
        gemm_run(ctx, &run.a, &self.normed, &self.a, seq, &self.dense_scratch)?;
        gemm_run(ctx, &run.b, &self.normed, &self.b, seq, &self.dense_scratch)?;
        if l == 0 {
            trace_buf("gpu_qkv_pre2", &self.qkv, seq * run.conv_dim);
        }

        kernels::qwen35::conv_silu(
            ctx,
            &self.qkv,
            &run.conv_w,
            &self.qkv,
            &mut run.conv_state,
            seq,
            run.conv_dim,
            run.conv_kernel,
        )?;
        if l == 0 {
            trace_buf("gpu_qkv_post", &self.qkv, seq * run.conv_dim);
        }
        kernels::qwen35::delta_norm_prepass(
            ctx,
            &self.qkv,
            &self.qk_scratch,
            seq,
            run.k_heads,
            run.v_heads,
            run.kdim,
            run.vdim,
        )?;
        kernels::qwen35::delta_step(
            ctx,
            &self.qkv,
            &self.qk_scratch,
            &self.a,
            &self.b,
            &run.a_log,
            &run.dt_bias,
            &mut run.recurrent,
            &self.delta_out,
            seq,
            run.k_heads,
            run.v_heads,
            run.kdim,
            run.vdim,
        )?;
        if l == 0 {
            trace_buf("gpu_delta_out", &self.delta_out, seq * run.v_heads * run.vdim);
        }
        kernels::qwen35::gated_norm(
            ctx,
            &self.delta_out,
            &self.z,
            &run.gate_norm_w,
            &self.gated,
            seq,
            run.v_heads,
            run.vdim,
            self.eps,
        )?;
        if l == 0 {
            trace_buf("gpu_gated", &self.gated, seq * run.v_heads * run.vdim);
        }
        gemm_run(ctx, &run.out, &self.gated, &self.attn, seq, &self.dense_scratch)?;
        if l == 0 {
            trace_buf("gpu_attn", &self.attn, seq * self.hidden);
        }
        kernels::elementwise::add_into(
            ctx, DType::BF16, &self.x, &self.attn, &self.x, seq * self.hidden,
        )?;
        if l == 0 {
            trace_buf("gpu_x", &self.x, seq * self.hidden);
        }
        self.run_mlp(
            &run.gate, &run.up, &run.down, &run.post_norm_w, seq,
        )?;
        if let CudaLayer::Linear(layer) = &mut self.layers[l] {
            layer.conv_state = run.conv_state;
            layer.recurrent = run.recurrent;
        }
        Ok(())
    }


    fn take_full(&self, l: usize) -> FullRun {
        let layer = match &self.layers[l] {
            CudaLayer::Full(layer) => layer,
            _ => unreachable!(),
        };
        FullRun {
            in_norm_w: layer.in_norm_w.clone(),
            q: gemm_clone(&layer.q),
            k: gemm_clone(&layer.k),
            v: gemm_clone(&layer.v),
            o: gemm_clone(&layer.o),
            q_norm_w: layer.q_norm_w.clone(),
            k_norm_w: layer.k_norm_w.clone(),
            k_cache: layer.k_cache.clone(),
            v_cache: layer.v_cache.clone(),
            gate: gemm_clone(&layer.gate),
            up: gemm_clone(&layer.up),
            down: gemm_clone(&layer.down),
            post_norm_w: layer.post_norm_w.clone(),
        }
    }

    fn run_full(&mut self, l: usize, seq: usize, start_pos: u32) -> Result<()> {
        self.upload_pos(start_pos)?;
        let ctx = self.ctx();
        let mut run = self.take_full(l);
        kernels::norm::rms_into(
            ctx, DType::BF16, &self.x, &run.in_norm_w, &self.normed, self.hidden, seq, self.eps,
        )?;
        if l == 3 {
            trace_buf("f3_normed", &self.normed, seq * self.hidden);
        }
        gemm_run(ctx, &run.q, &self.normed, &self.q_gate, seq, &self.dense_scratch)?;
        if l == 3 {
            trace_buf("f3_qgate", &self.q_gate, seq * self.n_heads * self.head_dim * 2);
        }
        gemm_run(ctx, &run.k, &self.normed, &self.k_buf, seq, &self.dense_scratch)?;
        gemm_run(ctx, &run.v, &self.normed, &self.v_buf, seq, &self.dense_scratch)?;

        kernels::qwen35::q_split_norm_rope(
            ctx,
            &self.q_gate,
            &run.q_norm_w,
            &self.q_buf,
            &self.gate_buf,
            seq,
            self.n_heads,
            self.head_dim,
            self.rotary_dim,
            self.rope_theta,
            start_pos,
        )?;
        kernels::qwen35::k_norm_rope_append(
            ctx,
            &self.k_buf,
            &run.k_norm_w,
            &mut run.k_cache,
            seq,
            self.n_kv_heads,
            self.head_dim,
            self.rotary_dim,
            self.rope_theta,
            start_pos,
            MAX_SEQ_LEN,
        )?;
        if l == 3 {
            trace_buf("f3_q", &self.q_buf, seq * self.n_heads * self.head_dim);
            trace_buf("f3_v", &self.v_buf, seq * self.n_kv_heads * self.head_dim);
            trace_buf("f3_gate", &self.gate_buf, seq * self.n_heads * self.head_dim);
        }

        // Append V before attention reads it.
        if seq == 1 {
            kernels::cache::append_at(
                ctx,
                DType::BF16,
                &run.v_cache,
                &self.v_buf,
                self.n_kv_heads,
                self.head_dim,
                MAX_SEQ_LEN,
                self.pos.address(),
            )?;
        } else {
            let v_tensor = self.v_buf.clone().into_tensor(
                Shape::from(vec![seq, self.n_kv_heads, self.head_dim]),
                DType::BF16,
            );
            kernels::cache::append(
                ctx,
                &run.v_cache,
                &v_tensor,
                self.n_kv_heads,
                self.head_dim,
                MAX_SEQ_LEN,
                start_pos as usize,
                seq,
            )?;
        }

        if l == 3 {
            trace_buf("f3g_vcache", &run.v_cache, 11 * self.n_kv_heads * self.head_dim);
            trace_buf("f3g_kcache", &run.k_cache, 11 * self.n_kv_heads * self.head_dim);
        }
        // Attention: decode steps (seq=1) use the proven fused flash
        // kernel; prefill chunks use GEMM-based scores = q @ k^T, causal
        // softmax, then out = sigmoid(gate) * (p @ v) / l.
        let visible = start_pos as usize + seq;
        if seq == 1 {
            kernels::qwen35::flash_prefill(
                ctx,
                &self.q_buf,
                &run.k_cache,
                &run.v_cache,
                &self.attn,
                seq,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                1.0 / (self.head_dim as f32).sqrt(),
                start_pos,
                MAX_SEQ_LEN,
            )?;
            kernels::qwen35::sigmoid_mul(
                ctx,
                &self.gate_buf,
                &self.attn,
                &self.attn,
                seq * self.n_heads * self.head_dim,
            )?;
            if l == 3 {
                trace_buf("f3_attn_pre", &self.attn, seq * self.n_heads * self.head_dim);
            }
        } else {
        let per_kv = self.n_heads / self.n_kv_heads;
        for kv in 0..self.n_kv_heads {
            let k_slice = run.k_cache.len() / self.n_kv_heads;
            let k_view = run
                .k_cache
                .view(kv * k_slice, k_slice)
                .map_err(Error::Cuda)?;
            let v_slice = run.v_cache.len() / self.n_kv_heads;
            let v_view = run
                .v_cache
                .view(kv * v_slice, v_slice)
                .map_err(Error::Cuda)?;
            kernels::qwen35::attention_gqa_dot(
                ctx,
                &self.q_buf,
                &k_view,
                &self.attn_kt,
                &self.attn_scores,
                kv,
                seq,
                visible,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                MAX_SEQ_LEN,
            )?;
            kernels::qwen35::attention_softmax_rows(
                ctx,
                &self.attn_scores,
                &self.attn_l,
                kv * per_kv,
                seq,
                per_kv,
                visible,
                MAX_SEQ_LEN,
                start_pos,
                1.0 / (self.head_dim as f32).sqrt(),
            )?;
            let vf32_view = self
                .attn_vf32
                .view(
                    kv * MAX_SEQ_LEN * self.head_dim * 4,
                    MAX_SEQ_LEN * self.head_dim * 4,
                )
                .map_err(Error::Cuda)?;
            kernels::qwen35::v_to_f32(
                ctx,
                &v_view,
                &vf32_view,
                visible,
                self.head_dim,
            )?;
            kernels::qwen35::attention_gqa_pv(
                ctx,
                &self.attn_scores,
                &vf32_view,
                &self.attn_pv,
                kv,
                seq,
                visible,
                self.n_heads,
                self.n_kv_heads,
                self.head_dim,
                MAX_SEQ_LEN,
            )?;
        }
        if l == 3 {
            trace_buf_f32("f3g_scores_f32", &self.attn_scores, 6 * seq * MAX_SEQ_LEN);
        }
        if l == 3 {
            trace_buf_f32("f3g_pv_f32", &self.attn_pv, seq * self.n_heads * self.head_dim);
        }
        kernels::qwen35::scale_out(
            ctx,
            &self.attn_pv,
            &self.attn_l,
            &self.attn,
            seq,
            self.n_heads,
            self.head_dim,
        )?;
        kernels::qwen35::sigmoid_mul(
            ctx,
            &self.gate_buf,
            &self.attn,
            &self.attn,
            seq * self.n_heads * self.head_dim,
        )?;
            if l == 3 {
                trace_buf("f3_attn_pre", &self.attn, seq * self.n_heads * self.head_dim);
            }
        }
        gemm_run(ctx, &run.o, &self.attn, &self.attn2, seq, &self.dense_scratch)?;
        if l == 3 {
            trace_buf("f3_attn2", &self.attn2, seq * self.hidden);
        }
        kernels::elementwise::add_into(
            ctx, DType::BF16, &self.x, &self.attn2, &self.x, seq * self.hidden,
        )?;
        if l == 3 {
            trace_buf("f3_x", &self.x, seq * self.hidden);
        }
        self.run_mlp(&run.gate, &run.up, &run.down, &run.post_norm_w, seq)?;
        if let CudaLayer::Full(layer) = &mut self.layers[l] {
            layer.k_cache = run.k_cache;
            layer.v_cache = run.v_cache;
        }
        Ok(())
    }

    /// MLP shared by both layer kinds: norm, gate/up GEMMs, SiLU·up, down.
    fn run_mlp(
        &mut self,
        gate: &Gemm,
        up: &Gemm,
        down: &Gemm,
        post_norm_w: &CudaBuffer,
        seq: usize,
    ) -> Result<()> {
        let ctx = self.ctx();
        kernels::norm::rms_into(
            ctx, DType::BF16, &self.x, post_norm_w, &self.normed2,
            self.hidden, seq, self.eps,
        )?;
        gemm_run(ctx, gate, &self.normed2, &self.gate_proj, seq, &self.dense_scratch)?;
        gemm_run(ctx, up, &self.normed2, &self.up_proj, seq, &self.dense_scratch)?;
        kernels::qwen35::silu_mul(
            ctx, &self.gate_proj, &self.up_proj, &self.mlp_act,
            seq * self.intermediate,
        )?;
        gemm_run(ctx, down, &self.mlp_act, &self.attn2, seq, &self.dense_scratch)?;
        kernels::elementwise::add_into(
            ctx, DType::BF16, &self.x, &self.attn2, &self.x, seq * self.hidden,
        )
    }

    /// Copy token ids into the device ids buffer (sync memcpy; kernels on the
    /// same stream read it afterwards).
    fn upload_ids(&mut self, ids: &[u32]) -> Result<()> {
        let bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.ids.copy_from_host(&bytes).map_err(Error::Cuda)
    }

    fn upload_pos(&mut self, pos: u32) -> Result<()> {
        self.pos
            .copy_from_host(&pos.to_le_bytes())
            .map_err(Error::Cuda)
    }

    /// Final RMSNorm (last row) + lm_head, returning `[1, vocab]` logits as a
    fn final_logits(&mut self, seq: usize) -> Result<Tensor> {
        let ctx = self.ctx();
        // Norm the LAST row of the final chunk: `x` holds the whole chunk
        // [seq, hidden], not just the final token.
        let last_row = self
            .x
            .view((seq - 1) * self.hidden * 2, self.hidden * 2)
            .map_err(Error::Cuda)?;
        kernels::norm::rms_into(
            ctx, DType::BF16, &last_row, &self.final_norm_w, &self.normed,
            self.hidden, 1, self.eps,
        )?;
        kernels::gemm::write_ex(
            ctx,
            DType::BF16,
            apxinf_cuda::CublasTranspose::None,
            apxinf_cuda::CublasTranspose::Transpose,
            1,
            self.vocab,
            self.hidden,
            1.0,
            &self.normed,
            self.hidden as i32,
            &self.lm_head,
            self.hidden as i32,
            0.0,
            &self.logits,
            self.vocab as i32,
        )?;
        Ok(self.logits.clone().into_tensor(Shape::from(vec![1, self.vocab]), DType::BF16))
    }
}

fn gemm_clone(gemm: &Gemm) -> Gemm {
    Gemm {
        packed: gemm.packed.as_ref().map(|p| GemmPacked {
            w: p.w.clone(),
            scale: p.scale.clone(),
            zp: p.zp.clone(),
            groups: p.groups,
        }),
        dense: gemm.dense.clone(),
        out_cols: gemm.out_cols,
        in_cols: gemm.in_cols,
    }
}

/// Dispatch a GEMM: fused dequant kernel for single-row activations, dequant +
/// cublas otherwise.
fn gemm_run(
    ctx: &CudaContext,
    gemm: &Gemm,
    act: &CudaBuffer,
    out: &CudaBuffer,
    seq: usize,
    scratch: &CudaBuffer,
) -> Result<()> {
    if let Some(dense) = &gemm.dense {
        return kernels::gemm::write_ex(
            ctx,
            DType::BF16,
            apxinf_cuda::CublasTranspose::None,
            apxinf_cuda::CublasTranspose::Transpose,
            seq,
            gemm.out_cols,
            gemm.in_cols,
            1.0,
            act,
            gemm.in_cols as i32,
            dense,
            gemm.in_cols as i32,
            0.0,
            out,
            gemm.out_cols as i32,
        );
    }
    let packed = gemm
        .packed
        .as_ref()
        .ok_or_else(|| Error::Other("GEMM has no weights".into()))?;
    let act_tensor = act
        .clone()
        .into_tensor(Shape::from(vec![seq, gemm.in_cols]), DType::BF16);
    let w_tensor = packed.w.clone().into_tensor(
        Shape::from(vec![gemm.out_cols, gemm.in_cols.div_ceil(8)]),
        DType::I32,
    );
    let s_tensor = packed
        .scale
        .clone()
        .into_tensor(Shape::from(vec![gemm.out_cols, packed.groups]), DType::BF16);
    let zp_tensor = packed.zp.clone().into_tensor(
        Shape::from(vec![gemm.out_cols.div_ceil(8), packed.groups]),
        DType::I32,
    );
    if seq == 1 && gemm.in_cols % 16 == 0 && gemm.out_cols % 64 == 0 {
        // Tensor-core decode GEMM (m16n8k16 bf16 MMA), allocation-free.
        return kernels::quantization::matmul_bf16_w4a16_asym_tc(
            ctx,
            act,
            &packed.w,
            &packed.scale,
            &packed.zp,
            out,
            gemm.in_cols,
            gemm.out_cols,
            packed.groups,
        );
    }
    if seq == 1 {
        // Allocation-free fused dequant-GEMM straight into the workspace slot.
        return kernels::quantization::matmul_bf16_w4a16_asym_into(
            ctx,
            act,
            &packed.w,
            &packed.scale,
            &packed.zp,
            out,
            1,
            gemm.in_cols,
            gemm.out_cols,
            packed.groups,
        );
    }
    if gemm.out_cols * gemm.in_cols <= 17408 * 5120 {
        // Prefill: dequant into the persistent scratch + cublas, alloc-free.
        return kernels::quantization::matmul_bf16_w4a16_asym_prefill_into(
            ctx,
            act,
            &packed.w,
            &packed.scale,
            &packed.zp,
            scratch,
            out,
            seq,
            gemm.in_cols,
            gemm.out_cols,
            packed.groups,
        );
    }
    let act_tensor = act
        .clone()
        .into_tensor(Shape::from(vec![seq, gemm.in_cols]), DType::BF16);
    let w_tensor = packed.w.clone().into_tensor(
        Shape::from(vec![gemm.out_cols, gemm.in_cols.div_ceil(8)]),
        DType::I32,
    );
    let s_tensor = packed
        .scale
        .clone()
        .into_tensor(Shape::from(vec![gemm.out_cols, packed.groups]), DType::BF16);
    let zp_tensor = packed.zp.clone().into_tensor(
        Shape::from(vec![gemm.out_cols.div_ceil(8), packed.groups]),
        DType::I32,
    );
    let result = kernels::quantization::matmul_bf16_w4a16_asym(
        ctx,
        &act_tensor,
        &w_tensor,
        &s_tensor,
        &zp_tensor,
        gemm.out_cols,
        gemm.in_cols,
        packed.groups,
    )?;
    let result_buf = CudaBuffer::from_tensor(&result).map_err(Error::Cuda)?;
    out.copy_d2d_async(&result_buf, seq * gemm.out_cols * 2, ctx.stream())
        .map_err(Error::Cuda)?;
    Ok(())
}

/// Look up a GPU tensor by name and borrow its buffer.
fn buffer_from(device_tensors: &HashMap<String, Tensor>, name: &str) -> Result<CudaBuffer> {
    let tensor = device_tensors
        .get(name)
        .ok_or_else(|| Error::Other(format!("qwen3_5 CUDA: missing device tensor {name}")))?;
    CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)
}

/// Build a GEMM handle from the device tensors: packed when the compressed
/// tensor set is present, otherwise dense.
fn gemm(
    device_tensors: &HashMap<String, Tensor>,
    prefix: &str,
    out_cols: usize,
    in_cols: usize,
) -> Result<Gemm> {
    let packed_name = format!("{prefix}.weight_packed");
    if device_tensors.contains_key(&packed_name) {
        let scale = buffer_from(device_tensors, &format!("{prefix}.weight_scale"))?;
        let groups = scale.len() / (out_cols * 2);
        Ok(Gemm {
            packed: Some(GemmPacked {
                w: buffer_from(device_tensors, &packed_name)?,
                scale,
                zp: buffer_from(device_tensors, &format!("{prefix}.weight_zero_point"))?,
                groups,
            }),
            dense: None,
            out_cols,
            in_cols,
        })
    } else {
        Ok(Gemm {
            packed: None,
            dense: Some(buffer_from(device_tensors, &format!("{prefix}.weight"))?),
            out_cols,
            in_cols,
        })
    }
}

/// Upload a 1-D bf16 weight with the Qwen3.5 `(1 + weight)` convention.
fn upload_norm_plus_one(
    device: usize,
    tensors: &HashMap<String, Tensor>,
    name: &str,
    len: usize,
) -> Result<CudaBuffer> {
    let tensor = tensors
        .get(name)
        .ok_or_else(|| Error::Other(format!("qwen3_5 CUDA: missing tensor {name}")))?;
    let values = tensor
        .as_bf16()
        .map(|v| v.iter().map(|x| x.to_f32() + 1.0).collect::<Vec<_>>())
        .or_else(|_| tensor.as_f32().map(|v| v.iter().map(|x| x + 1.0).collect::<Vec<_>>()))
        .map_err(|_| Error::Other(format!("qwen3_5 CUDA: {name} is not bf16/f32")))?;
    if values.len() != len {
        return Err(Error::Other(format!(
            "qwen3_5 CUDA: {name} has {} values, expected {len}",
            values.len()
        )));
    }
    let bf16: Vec<u8> = values
        .iter()
        .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
        .collect();
    let buf = CudaBuffer::alloc(bf16.len(), device).map_err(Error::Cuda)?;
    buf.copy_from_host(&bf16).map_err(Error::Cuda)?;
    Ok(buf)
}

/// Upload a small 1-D tensor as bf16.
fn upload_bf16(
    device: usize,
    tensors: &HashMap<String, Tensor>,
    name: &str,
) -> Result<CudaBuffer> {
    let tensor = tensors
        .get(name)
        .ok_or_else(|| Error::Other(format!("qwen3_5 CUDA: missing tensor {name}")))?;
    let values = tensor
        .as_bf16()
        .map(|v| v.iter().map(|x| x.to_f32()).collect::<Vec<_>>())
        .or_else(|_| tensor.as_f32().map(|v| v.to_vec()))
        .map_err(|_| Error::Other(format!("qwen3_5 CUDA: {name} is not bf16/f32")))?;
    let bf16: Vec<u8> = values
        .iter()
        .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
        .collect();
    let buf = CudaBuffer::alloc(bf16.len(), device).map_err(Error::Cuda)?;
    buf.copy_from_host(&bf16).map_err(Error::Cuda)?;
    Ok(buf)
}

/// Upload a CPU tensor of any shape, flattened in row-major order, as bf16.
fn upload_bf16_flat(
    device: usize,
    tensors: &HashMap<String, Tensor>,
    name: &str,
) -> Result<CudaBuffer> {
    let tensor = tensors
        .get(name)
        .ok_or_else(|| Error::Other(format!("qwen3_5 CUDA: missing tensor {name}")))?;
    let values = tensor
        .as_bf16()
        .map(|v| v.iter().map(|x| x.to_f32()).collect::<Vec<_>>())
        .or_else(|_| tensor.as_f32().map(|v| v.to_vec()))
        .map_err(|_| Error::Other(format!("qwen3_5 CUDA: {name} is not bf16/f32")))?;
    let bf16: Vec<u8> = values
        .iter()
        .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
        .collect();
    let buf = CudaBuffer::alloc(bf16.len(), device).map_err(Error::Cuda)?;
    buf.copy_from_host(&bf16).map_err(Error::Cuda)?;
    Ok(buf)
}

/// Debug: copy an f32 GPU buffer to /tmp/qwen35_trace/<name>.f32 when APXINF_TRACE is set.
fn trace_buf_f32(name: &str, buf: &CudaBuffer, count: usize) {
    if std::env::var_os("APXINF_TRACE").is_none() {
        return;
    }
    let dir = std::path::Path::new("/tmp/qwen35_trace");
    let _ = std::fs::create_dir_all(dir);
    let mut bytes = vec![0u8; count * 4];
    if buf.copy_to_host(&mut bytes).is_err() {
        return;
    }
    let _ = std::fs::write(dir.join(format!("{name}.f32")), bytes);
}

/// Debug: copy a GPU buffer to /tmp/qwen35_trace/<name>.f32 when APXINF_TRACE is set.
fn trace_buf(name: &str, buf: &CudaBuffer, count: usize) {
    if std::env::var_os("APXINF_TRACE").is_none() {
        return;
    }
    let dir = std::path::Path::new("/tmp/qwen35_trace");
    let _ = std::fs::create_dir_all(dir);
    let mut bytes = vec![0u8; count * 2];
    if buf.copy_to_host(&mut bytes).is_err() {
        return;
    }
    let mut raw = Vec::with_capacity(count * 4);
    for chunk in bytes.chunks_exact(2) {
        let v = half::bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let _ = std::fs::write(dir.join(format!("{name}.f32")), raw);
}
