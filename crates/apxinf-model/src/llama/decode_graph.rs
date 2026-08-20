//! Decode workspace + graph capture — the allocation-free decode fast path.
//!
//! **Design note:** This module lives in `apxinf-model` (not `apxinf-cuda`)
//! because it encodes model-structure knowledge: the transformer layer
//! forward order (norm → QKV → RoPE → attention → MLP → residual), the
//! GQA layout, and fusion choices (packed QKV, flash attention, etc.).
//! The backend crate only provides single-kernel wrappers; the model
//! composes them here. See `doc/design.md`.
//!
//! CUDA-only. The portable `dyn Backend` path in `GeneralLlama::forward`
//! handles CPU + non-workspace backends.

#![cfg(feature = "cuda")]

use apxinf_core::{Backend, Device, DType, Error, Graph, KvCache, Result, RopeKind, Shape, Tensor};

use crate::accelerator::cuda::{
    kernels, Context as CudaContext, CublasTranspose,
    DeviceBuffer as CudaBuffer, KvCache as CudaKVCache, MappedBuffer as HostMappedBuffer,
    RuntimeBackend as CudaBackend,
};

// ── Config + weight view types (model-level, not backend) ───────────────

/// Model dimensions needed to build a decode workspace/graph.
/// Primitive-only so the model layer has no backend-type dependency.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct DecodeGraphConfig {
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub dtype: DType,
    pub rope_kind: RopeKind,
    pub qk_norm: bool,
    pub tie_embeddings: bool,
}

impl DecodeGraphConfig {
    pub fn llama_like(
        n_layers: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize,
        hidden_size: usize, intermediate_size: usize, vocab_size: usize,
        max_seq_len: usize, rope_theta: f32, rms_norm_eps: f32, dtype: DType,
    ) -> Self {
        Self {
            n_layers, n_heads, n_kv_heads, head_dim, hidden_size,
            intermediate_size, vocab_size, max_seq_len, rope_theta,
            rms_norm_eps, dtype,
            rope_kind: RopeKind::OneD { theta: rope_theta },
            qk_norm: false, tie_embeddings: false,
        }
    }
}

/// Weight tensors for a decode forward, passed by reference.
/// All tensors must already reside on the target backend's device.
pub struct DecodeGraphWeights<'a> {
    pub token_embedding: &'a Tensor,
    pub layers: Vec<DecodeLayerWeights<'a>>,
    pub output_norm_weight: &'a Tensor,
    pub output_weight: &'a Tensor,
}

/// Weights for one transformer layer (Llama-style: RMSNorm + GQA attention + SwiGLU MLP).
pub struct DecodeLayerWeights<'a> {
    pub attn_norm_weight: &'a Tensor,
    pub wq: &'a Tensor,
    pub wk: &'a Tensor,
    pub wv: &'a Tensor,
    pub wo: &'a Tensor,
    pub ffn_norm_weight: &'a Tensor,
    pub w_gate: &'a Tensor,
    pub w_up: &'a Tensor,
    pub w_down: &'a Tensor,
    pub q_norm_weight: Option<&'a Tensor>,
    pub k_norm_weight: Option<&'a Tensor>,
    pub qkv_packed: Option<&'a Tensor>,
    pub gate_up_packed: Option<&'a Tensor>,
}

// ── Workspace ────────────────────────────────────────────────────────────

/// Pre-allocated activation buffers for allocation-free decode (seq_len=1).
/// Reused across tokens and layers. Zero `cudaMalloc` per token.
struct DecodeWorkspace {
    dtype: DType,
    x: CudaBuffer,            // [hidden] flowing residual
    logits: CudaBuffer,       // [vocab] — device mem (GPU→host mapped write is
                              // slower on Tegra due to coherency; D2H copy wins)
    norm_out: CudaBuffer,     // [hidden] pre-attn normed (reused across layers)
    q: CudaBuffer,            // [hidden]
    k: CudaBuffer,            // [n_kv_heads*head_dim]
    v: CudaBuffer,            // [n_kv_heads*head_dim]
    qkv: CudaBuffer,          // [hidden + 2*kv_proj] fused QKV output
    gate_up: CudaBuffer,      // [2*intermediate] fused Gate/Up output
    q_rope: CudaBuffer,       // [hidden]
    k_rope: CudaBuffer,       // [n_kv_heads*head_dim]
    scores: CudaBuffer,       // [n_heads * max_seq_len]
    attn_weights: CudaBuffer, // [n_heads * max_seq_len]
    attn_out: CudaBuffer,     // [hidden]
    attn_proj: CudaBuffer,    // [hidden] (wo output)
    ffn_norm_out: CudaBuffer, // [hidden]
    gate: CudaBuffer,         // [intermediate]
    gate_silu: CudaBuffer,    // [intermediate]
    up: CudaBuffer,           // [intermediate]
    mlp_hidden: CudaBuffer,   // [intermediate]
    mlp_out: CudaBuffer,      // [hidden]
    token_buf: HostMappedBuffer,    // [1] u32 — zero-copy host-write, device-read
    pos_buf: HostMappedBuffer,      // [1] u32
}

impl DecodeWorkspace {
    fn new(device_id: usize, cfg: &DecodeGraphConfig) -> std::result::Result<Self, String> {
        let h = cfg.hidden_size;
        let kv = cfg.n_kv_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;
        let scores_n = cfg.n_heads * cfg.max_seq_len;
        let vocab = cfg.vocab_size;
        let elem = cfg.dtype.size_in_bytes();
        let f = |n: usize| CudaBuffer::alloc_zeros(n * elem, device_id);
        Ok(Self {
            dtype: cfg.dtype,
            x: f(h)?,
            logits: f(vocab)?,
            norm_out: f(h)?,
            q: f(h)?,
            k: f(kv)?,
            v: f(kv)?,
            qkv: f(h + 2 * kv)?,
            gate_up: f(2 * inter)?,
            q_rope: f(h)?,
            k_rope: f(kv)?,
            scores: f(scores_n)?,
            attn_weights: f(scores_n)?,
            attn_out: f(h)?,
            attn_proj: f(h)?,
            ffn_norm_out: f(h)?,
            gate: f(inter)?,
            gate_silu: f(inter)?,
            up: f(inter)?,
            mlp_hidden: f(inter)?,
            mlp_out: f(h)?,
            token_buf: HostMappedBuffer::alloc(4, device_id)?,
            pos_buf: HostMappedBuffer::alloc(4, device_id)?,
        })
    }

    fn dtype(&self) -> DType { self.dtype }
}

// ── Helpers ──────────────────────────────────────────────────────────────

fn weight_view(tensor: &Tensor, device_id: usize) -> Result<CudaBuffer> {
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    if buffer.device() != device_id {
        return Err(Error::Other(format!(
            "weight is on CUDA {}, expected CUDA {device_id}",
            buffer.device()
        )));
    }
    Ok(buffer)
}

/// Write a u32 into a zero-copy host-mapped buffer. The CPU store lands in
/// shared physical memory (Tegra/Thor UMA) so the GPU reads it with no
/// `cudaMemcpy`. `cudaGraphLaunch` provides the host→device ordering
/// boundary; the fence keeps the compiler from hoisting the store and
/// flushes the store buffer on ARM.
fn write_u32_mapped(buf: &HostMappedBuffer, val: u32) {
    buf.write_u32(val)
        .expect("decode control buffer is allocated as one u32");
}

/// Read the logits row back to the CPU for sampling. Synchronous D2H — on
/// Tegra this beats a host-mapped write (the GPU→host coherency flush for a
/// 64 kB row costs more than the explicit copy).
fn read_logits(ws: &DecodeWorkspace, vocab: usize) -> Result<Tensor> {
    let dtype = ws.dtype();
    let nbytes = vocab * dtype.size_in_bytes();
    let mut host = vec![0u8; nbytes];
    ws.logits.copy_to_host(&mut host).map_err(Error::Cuda)?;
    match dtype {
        DType::F32 => Tensor::from_raw(Shape::new(vec![1, vocab]), DType::F32, Device::Cpu, host),
        DType::BF16 => {
            // Return the bf16 bytes as-is — no f32 conversion here. The
            // argmax in llm_trait scans bf16 directly, saving a 32k-element
            // conversion + Vec alloc per token.
            Tensor::from_raw(Shape::new(vec![1, vocab]), DType::BF16, Device::Cpu, host)
        }
        dtype => Err(Error::Other(format!("Llama decode logits do not support {dtype}"))),
    }
}

// ── Decode forward (graph body) ──────────────────────────────────────────

fn decode_forward_capturable(
    ctx: &CudaContext,
    ws: &DecodeWorkspace,
    weights: &DecodeGraphWeights,
    kv: &mut dyn KvCache,
    cfg: &DecodeGraphConfig,
    bucket_kv_len: usize,
) -> Result<()> {
    let device_id = ctx.device_id();
    let hidden = cfg.hidden_size;
    let n_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let head_dim = cfg.head_dim;
    let gqa_ratio = n_heads / n_kv_heads;
    let inter = cfg.intermediate_size;
    let vocab = cfg.vocab_size;
    let max_seq = cfg.max_seq_len;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let position = ws.pos_buf.address();
    let elem = cfg.dtype.size_in_bytes();
    let dtype = cfg.dtype;
    let is_bf16 = dtype == DType::BF16;

    let cache = kv.as_any_mut().downcast_mut::<CudaKVCache>()
        .ok_or_else(|| Error::Other("expected CudaKVCache".into()))?;

    // Embedding lookup
    let token_embedding = weight_view(weights.token_embedding, device_id)?;
    kernels::embedding::lookup_into(
        ctx,
        dtype,
        &token_embedding,
        ws.token_buf.address(),
        &ws.x,
        hidden,
        1,
    )?;

    // Layer 0 pre-attention norm (standalone — subsequent layers' norms are
    // fused into the previous layer's post-FFN rms_norm_add).
    let first_norm = weight_view(weights.layers[0].attn_norm_weight, device_id)?;
    kernels::norm::rms_into(
        ctx,
        dtype,
        &ws.x,
        &first_norm,
        &ws.norm_out,
        hidden,
        1,
        cfg.rms_norm_eps,
    )?;

    for (i, layer) in weights.layers.iter().enumerate() {
        let kv_proj = n_kv_heads * head_dim;
        let (q_view, k_view, v_view) = if let Some(qkv_w) = layer.qkv_packed {
            let norm_view = ws.norm_out.view(0, ws.norm_out.len()).map_err(Error::Cuda)?;
            let qkv_wv = weight_view(qkv_w, device_id)?;
            let fused_n = hidden + 2 * kv_proj;
            kernels::gemm::write(ctx, dtype, 1, fused_n, hidden, 1.0, &norm_view, &qkv_wv, 0.0, &ws.qkv)
                ?;
            (
                ws.qkv.view(0, hidden * elem).map_err(Error::Cuda)?,
                ws.qkv
                    .view(hidden * elem, kv_proj * elem)
                    .map_err(Error::Cuda)?,
                ws.qkv
                    .view((hidden + kv_proj) * elem, kv_proj * elem)
                    .map_err(Error::Cuda)?,
            )
        } else {
            let norm_view = ws.norm_out.view(0, ws.norm_out.len()).map_err(Error::Cuda)?;
            let wq_v = weight_view(layer.wq, device_id)?;
            let wk_v = weight_view(layer.wk, device_id)?;
            let wv_v = weight_view(layer.wv, device_id)?;
            kernels::gemm::write(ctx, dtype, 1, hidden, hidden, 1.0, &norm_view, &wq_v, 0.0, &ws.q)?;
            kernels::gemm::write(ctx, dtype, 1, kv_proj, hidden, 1.0, &norm_view, &wk_v, 0.0, &ws.k)?;
            kernels::gemm::write(ctx, dtype, 1, kv_proj, hidden, 1.0, &norm_view, &wv_v, 0.0, &ws.v)?;
            (
                ws.q.view(0, ws.q.len()).map_err(Error::Cuda)?,
                ws.k.view(0, ws.k.len()).map_err(Error::Cuda)?,
                ws.v.view(0, ws.v.len()).map_err(Error::Cuda)?,
            )
        };

        // RoPE + KV cache append. bf16: fused rope_k_write for K; Q uses
        // rope_decode. V has no RoPE. fp32: 4-kernel fallback.
        kernels::rope::apply_into(
            ctx,
            dtype,
            &q_view,
            &ws.q_rope,
            head_dim,
            n_heads,
            cfg.rope_theta,
            position,
        )?;
        if is_bf16 {
            kernels::rope::apply_k_write_cache_bf16(
                ctx,
                &k_view,
                cache.k_buffer(i),
                head_dim,
                n_kv_heads,
                max_seq,
                cfg.rope_theta,
                position,
            )?;
        } else {
            kernels::rope::apply_into(
                ctx,
                dtype,
                &k_view,
                &ws.k_rope,
                head_dim,
                n_kv_heads,
                cfg.rope_theta,
                position,
            )?;
            kernels::cache::append_at(
                ctx,
                dtype,
                cache.k_buffer(i),
                &ws.k_rope,
                n_kv_heads,
                head_dim,
                max_seq,
                position,
            )?;
        }
        kernels::cache::append_at(
            ctx,
            dtype,
            cache.v_buffer(i),
            &v_view,
            n_kv_heads,
            head_dim,
            max_seq,
            position,
        )?;

        // Attention. bf16: flash attention decode (1 kernel). fp32: 17-kernel fallback.
        if is_bf16 {
            kernels::attention::flash_bf16_into(
                ctx,
                &ws.q_rope,
                cache.k_buffer(i),
                cache.v_buffer(i),
                &ws.attn_out,
                n_heads,
                n_kv_heads,
                head_dim,
                bucket_kv_len,
                max_seq,
                scale,
                position,
            )?;
        } else {
            for kv_h in 0..n_kv_heads {
                let q_offset = (kv_h * gqa_ratio) * head_dim;
                let scores_offset = (kv_h * gqa_ratio) * bucket_kv_len;
                let k_offset = kv_h * max_seq * head_dim;
                let q_group = ws.q_rope.view(q_offset * elem, gqa_ratio * head_dim * elem).map_err(Error::Cuda)?;
                let k_block = cache.k_buffer(i).view(k_offset * elem, bucket_kv_len * head_dim * elem).map_err(Error::Cuda)?;
                let scores_block = ws.scores.view(scores_offset * elem, gqa_ratio * bucket_kv_len * elem).map_err(Error::Cuda)?;
                kernels::gemm::write_ex(ctx, dtype, CublasTranspose::None, CublasTranspose::Transpose,
                    gqa_ratio, bucket_kv_len, head_dim, scale,
                    &q_group, head_dim as i32, &k_block, head_dim as i32, 0.0, &scores_block, bucket_kv_len as i32)?;
            }
            kernels::attention::softmax_f32_into(
                ctx,
                &ws.scores,
                &ws.attn_weights,
                bucket_kv_len,
                n_heads,
                position,
            )?;
            for kv_h in 0..n_kv_heads {
                let attn_offset = (kv_h * gqa_ratio) * bucket_kv_len;
                let v_offset = kv_h * max_seq * head_dim;
                let out_offset = (kv_h * gqa_ratio) * head_dim;
                let attn_group = ws.attn_weights.view(attn_offset * elem, gqa_ratio * bucket_kv_len * elem).map_err(Error::Cuda)?;
                let v_block = cache.v_buffer(i).view(v_offset * elem, bucket_kv_len * head_dim * elem).map_err(Error::Cuda)?;
                let out_block = ws.attn_out.view(out_offset * elem, gqa_ratio * head_dim * elem).map_err(Error::Cuda)?;
                kernels::gemm::write_ex(ctx, dtype, CublasTranspose::None, CublasTranspose::None,
                    gqa_ratio, head_dim, bucket_kv_len, 1.0,
                    &attn_group, bucket_kv_len as i32, &v_block, head_dim as i32, 0.0, &out_block, head_dim as i32)?;
            }
        }

        // wo projection
        let ao_view = ws.attn_out.view(0, ws.attn_out.len()).map_err(Error::Cuda)?;
        let wo_v = weight_view(layer.wo, device_id)?;
        kernels::gemm::write(ctx, dtype, 1, hidden, hidden, 1.0, &ao_view, &wo_v, 0.0, &ws.attn_proj)?;

        // Fused post-attn residual add + pre-FFN norm
        let ffn_norm_weight = weight_view(layer.ffn_norm_weight, device_id)?;
        if is_bf16 {
            kernels::norm::residual_add_rms_bf16_into(
                ctx,
                &ws.x,
                &ws.attn_proj,
                &ffn_norm_weight,
                &ws.ffn_norm_out,
                hidden,
                1,
                cfg.rms_norm_eps,
            )?;
        } else {
            kernels::elementwise::add_into(ctx, dtype, &ws.x, &ws.attn_proj, &ws.x, hidden)?;
            kernels::norm::rms_into(
                ctx,
                dtype,
                &ws.x,
                &ffn_norm_weight,
                &ws.ffn_norm_out,
                hidden,
                1,
                cfg.rms_norm_eps,
            )?;
        }

        let fn_view = ws.ffn_norm_out.view(0, ws.ffn_norm_out.len()).map_err(Error::Cuda)?;
        if let Some(gate_up_w) = layer.gate_up_packed {
            let gu_wv = weight_view(gate_up_w, device_id)?;
            kernels::gemm::write(ctx, dtype, 1, 2 * inter, hidden, 1.0, &fn_view, &gu_wv, 0.0, &ws.gate_up)
                ?;
            let up_view = ws
                .gate_up
                .view(inter * elem, inter * elem)
                .map_err(Error::Cuda)?;
            if is_bf16 {
                kernels::activation::silu_mul_bf16_into(ctx, &ws.gate_up, &ws.mlp_hidden, inter)?;
            } else {
                kernels::activation::silu_into(ctx, dtype, &ws.gate_up, &ws.gate_silu, inter)?;
                kernels::elementwise::mul_into(
                    ctx,
                    dtype,
                    &ws.gate_silu,
                    &up_view,
                    &ws.mlp_hidden,
                    inter,
                )?;
            }
        } else {
            let wg_v = weight_view(layer.w_gate, device_id)?;
            let wu_v = weight_view(layer.w_up, device_id)?;
            kernels::gemm::write(ctx, dtype, 1, inter, hidden, 1.0, &fn_view, &wg_v, 0.0, &ws.gate)?;
            kernels::gemm::write(ctx, dtype, 1, inter, hidden, 1.0, &fn_view, &wu_v, 0.0, &ws.up)?;
            kernels::activation::silu_into(ctx, dtype, &ws.gate, &ws.gate_silu, inter)?;
            kernels::elementwise::mul_into(
                ctx,
                dtype,
                &ws.gate_silu,
                &ws.up,
                &ws.mlp_hidden,
                inter,
            )?;
        }
        let mh_view = ws.mlp_hidden.view(0, ws.mlp_hidden.len()).map_err(Error::Cuda)?;
        let wd_v = weight_view(layer.w_down, device_id)?;
        kernels::gemm::write(ctx, dtype, 1, hidden, inter, 1.0, &mh_view, &wd_v, 0.0, &ws.mlp_out)?;

        // Fused post-FFN residual add + next layer's pre-attn norm (or final output norm)
        let next_norm_w = if i + 1 < weights.layers.len() {
            weights.layers[i + 1].attn_norm_weight
        } else {
            weights.output_norm_weight
        };
        if is_bf16 {
            let next_norm_weight = weight_view(next_norm_w, device_id)?;
            kernels::norm::residual_add_rms_bf16_into(
                ctx,
                &ws.x,
                &ws.mlp_out,
                &next_norm_weight,
                &ws.norm_out,
                hidden,
                1,
                cfg.rms_norm_eps,
            )?;
        } else {
            let next_norm_weight = weight_view(next_norm_w, device_id)?;
            kernels::elementwise::add_into(ctx, dtype, &ws.x, &ws.mlp_out, &ws.x, hidden)?;
            kernels::norm::rms_into(
                ctx,
                dtype,
                &ws.x,
                &next_norm_weight,
                &ws.norm_out,
                hidden,
                1,
                cfg.rms_norm_eps,
            )?;
        }
    }

    // lm_head matmul (norm_out already holds the final normed output)
    let no_view = ws.norm_out.view(0, ws.norm_out.len()).map_err(Error::Cuda)?;
    let ow_v = weight_view(weights.output_weight, device_id)?;
    kernels::gemm::write(ctx, dtype, 1, vocab, hidden, 1.0, &no_view, &ow_v, 0.0, &ws.logits)?;

    // NOTE: a single-block GPU argmax over [vocab] was tried here but is
    // net-negative on Thor (14 SMs): one block under-occupies the GPU and
    // costs more than the CPU scan it would replace. A multi-block 2-stage
    // argmax would tip it positive — left as a TODO. CPU argmax wins for now.

    Ok(())
}

// ── Graph capture/replay ─────────────────────────────────────────────────

struct BucketGraph {
    bucket_kv_len: usize,
    graph: Box<dyn Graph>,
}

/// Manages one captured CUDA Graph per kv_len bucket (powers of two).
/// First visit to a bucket warms up (real forward → cuBLAS workspace alloc
/// + this token's logits) then captures; subsequent tokens in the bucket
/// replay.
pub struct DecodeGraph {
    config: DecodeGraphConfig,
    workspace: DecodeWorkspace,
    buckets: Vec<BucketGraph>,
}

impl DecodeGraph {
    pub fn new(backend: &CudaBackend, cfg: DecodeGraphConfig) -> Result<Self> {
        let ws = DecodeWorkspace::new(backend.device_id(), &cfg)
            .map_err(Error::Other)?;
        Ok(Self { config: cfg, workspace: ws, buckets: Vec::new() })
    }

    fn bucket_for(&self, _pos: u32) -> usize {
        // Single bucket: the attention kernel loops to `valid_len` (read from
        // pos_ptr at replay time), so bucket_kv_len is just an upper bound.
        // One captured graph serves every decode position — no per-bucket
        // recapture, and no wasted attention iterations on padded tails.
        self.config.max_seq_len.max(1)
    }

    /// Pre-capture the CUDA graph for every distinct bucket that decode will
    /// hit in positions `0..=max_pos`. Capture records the kernel launches
    /// but does NOT execute them, so this leaves the workspace and KV cache
    /// untouched. Paying the (expensive, ~ms-per-bucket) `cudaGraphInstantiate`
    /// cost up front keeps it out of the per-token TPOT — without this, the
    /// first token of each new power-of-two bucket pays capture cost and
    /// drags the average decode latency well above steady-state.
    pub fn prewarm(
        &mut self,
        backend: &CudaBackend,
        weights: &DecodeGraphWeights,
        kv: &mut dyn KvCache,
        max_pos: u32,
    ) -> Result<()> {
        let ctx = backend.context();
        let mut seen: std::collections::HashSet<usize> =
            self.buckets.iter().map(|b| b.bucket_kv_len).collect();
        for pos in 0..=max_pos {
            let bkl = self.bucket_for(pos);
            if !seen.insert(bkl) {
                continue;
            }
            backend.begin_capture_relaxed()?;
            let cap_res = decode_forward_capturable(
                ctx, &self.workspace, weights, kv, &self.config, bkl);
            let graph = backend.end_capture()?;
            cap_res?;
            self.buckets.push(BucketGraph { bucket_kv_len: bkl, graph });
        }
        Ok(())
    }

    pub fn decode(
        &mut self,
        backend: &CudaBackend,
        weights: &DecodeGraphWeights,
        kv: &mut dyn KvCache,
        token: u32,
        pos: u32,
    ) -> Result<Tensor> {
        let ctx = backend.context();
        let bucket_kv_len = self.bucket_for(pos);
        let vocab = self.config.vocab_size;
        let have = self.buckets.iter().any(|b| b.bucket_kv_len == bucket_kv_len);

        if !have {
            write_u32_mapped(&self.workspace.token_buf, token);
            write_u32_mapped(&self.workspace.pos_buf, pos);
            decode_forward_capturable(ctx, &self.workspace, weights, kv, &self.config, bucket_kv_len)?;
            backend.synchronize()?;
            let logits = read_logits(&self.workspace, vocab)?;

            backend.begin_capture_relaxed()?;
            let cap_res = decode_forward_capturable(ctx, &self.workspace, weights, kv, &self.config, bucket_kv_len);
            let graph = backend.end_capture()?;
            cap_res?;
            self.buckets.push(BucketGraph { bucket_kv_len, graph });
            return Ok(logits);
        }

        write_u32_mapped(&self.workspace.token_buf, token);
        write_u32_mapped(&self.workspace.pos_buf, pos);
        // APXINF_NO_GRAPH: bypass graph replay and re-run the layer loop live.
        // Lets nsys see every per-token kernel (graph-replayed kernels are
        // undercounted in the CUPTI activity trace). Profiling-only — do not
        // use in production (loses the launch-overhead amortization).
        if std::env::var("APXINF_NO_GRAPH").map(|v| !v.is_empty()).unwrap_or(false) {
            decode_forward_capturable(ctx, &self.workspace, weights, kv, &self.config, bucket_kv_len)?;
            backend.synchronize()?;
            return read_logits(&self.workspace, vocab);
        }
        let graph = &self
            .buckets
            .iter()
            .find(|bucket| bucket.bucket_kv_len == bucket_kv_len)
            .unwrap()
            .graph;
        graph.replay()?;
        backend.synchronize()?;
        read_logits(&self.workspace, vocab)
    }
}
