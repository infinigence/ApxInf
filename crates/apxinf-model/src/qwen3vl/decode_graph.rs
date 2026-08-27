//! Qwen3-VL decode workspace + graph capture.
//!
//! Adapts the fused decode path (mirrors `llama/decode_graph.rs`) for
//! Qwen3-VL specifics:
//! 1. **mRoPE** (multimodal RoPE) instead of 1-D RoPE — reads a `[3]` u32
//!    device buffer of `(t, h, w)` per-token axes.
//! 2. **QK-norm** — per-head RMSNorm on Q and K after projection, before
//!    RoPE. Uses the existing `apxinf_rms_norm_bf16` kernel with rows =
//!    n_heads (or n_kv_heads).
//! 3. **Tied embeddings** — the lm_head matmul uses the pre-transposed
//!    token_embedding (cached at model load), passed via
//!    `Qwen3VLDecodeGraphWeights::lm_head`.
//!
//! CUDA + bf16 only. Uses the same fused kernels as Llama:
//! flash_attn_decode_bf16, rms_norm_add_bf16, silu_mul_bf16, plus
//! rope_mrope_decode_bf16 for the mRoPE step. Cannot use rope_k_write_bf16
//! (K goes through QK-norm first, so RoPE can't fuse with cache write).

#![cfg(feature = "cuda")]

use apxinf_core::{Backend, DType, Error, Graph, KvCache, Result, Shape, Tensor};

use crate::accelerator::cuda::{
    kernels, Context as CudaContext, DeviceAddress,
    DeviceBuffer as CudaBuffer, KvCache as CudaKVCache, MappedBuffer as HostMappedBuffer,
    RuntimeBackend as CudaBackend,
};

// ── Config + weight views ────────────────────────────────────────────────

/// Model dimensions for the Qwen3-VL decode graph. Layout is the same as
/// Llama's `DecodeGraphConfig` but without the config flags (`rope_kind`,
/// `qk_norm`, `tie_embeddings`) — this graph is Qwen3-VL-specific, so all
/// three are implicit (mRoPE, qk_norm on, tie on).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Qwen3VLDecodeGraphConfig {
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    /// mRoPE section split: [T, H, W] over the head_dim/2 frequency pairs.
    pub mrope_section: [usize; 3],
    pub rms_norm_eps: f32,
    /// Must be BF16 — Qwen3-VL fast path is bf16-only.
    pub dtype: DType,
}

pub struct Qwen3VLDecodeGraphWeights<'a> {
    pub token_embedding: &'a Tensor,
    pub layers: Vec<Qwen3VLDecodeLayerWeights<'a>>,
    pub output_norm_weight: &'a Tensor,
    /// Pre-transposed token_embedding for the tied lm_head matmul.
    /// Shape `[hidden, vocab]`.
    pub lm_head: &'a Tensor,
}

pub struct Qwen3VLDecodeLayerWeights<'a> {
    pub attn_norm_weight: &'a Tensor,
    pub wq: &'a Tensor,
    pub wk: &'a Tensor,
    pub wv: &'a Tensor,
    pub wo: &'a Tensor,
    pub ffn_norm_weight: &'a Tensor,
    pub w_gate: &'a Tensor,
    pub w_up: &'a Tensor,
    pub w_down: &'a Tensor,
    pub q_norm_weight: &'a Tensor,
    pub k_norm_weight: &'a Tensor,
    /// Fused QKV weight (optional; `None` uses 3 separate GEMMs).
    pub qkv_packed: Option<&'a Tensor>,
    /// Fused Gate/Up weight (optional; `None` uses 2 separate GEMMs).
    pub gate_up_packed: Option<&'a Tensor>,
}

// ── Workspace ────────────────────────────────────────────────────────────

struct DecodeWorkspace {
    dtype: DType,
    x: CudaBuffer,            // [hidden] flowing residual
    logits: CudaBuffer,       // [vocab]
    norm_out: CudaBuffer,     // [hidden] pre-attn normed (reused across layers)
    q: CudaBuffer,            // [hidden]
    k: CudaBuffer,            // [n_kv_heads*head_dim]
    v: CudaBuffer,            // [n_kv_heads*head_dim]
    qkv: CudaBuffer,          // [hidden + 2*kv_proj] fused QKV output
    gate_up: CudaBuffer,      // [2*intermediate] fused Gate/Up output
    q_normed: CudaBuffer,     // [hidden] post-QK-norm Q
    k_normed: CudaBuffer,     // [n_kv_heads*head_dim] post-QK-norm K
    q_rope: CudaBuffer,       // [hidden]
    k_rope: CudaBuffer,       // [n_kv_heads*head_dim]
    attn_out: CudaBuffer,     // [hidden]
    attn_proj: CudaBuffer,    // [hidden]
    ffn_norm_out: CudaBuffer, // [hidden]
    mlp_hidden: CudaBuffer,   // [intermediate]
    mlp_out: CudaBuffer,      // [hidden]
    token_buf: HostMappedBuffer, // [1] u32
    /// `[3]` u32 for mRoPE (t, h, w). Written by the caller before each
    /// decode; read by the rope_mrope_decode_bf16 kernel.
    pos_ids_buf: HostMappedBuffer, // [t, h, w, cache_pos]
}

impl DecodeWorkspace {
    fn new(device_id: usize, cfg: &Qwen3VLDecodeGraphConfig) -> std::result::Result<Self, String> {
        let h = cfg.hidden_size;
        let kv = cfg.n_kv_heads * cfg.head_dim;
        let inter = cfg.intermediate_size;
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
            q_normed: f(h)?,
            k_normed: f(kv)?,
            q_rope: f(h)?,
            k_rope: f(kv)?,
            attn_out: f(h)?,
            attn_proj: f(h)?,
            ffn_norm_out: f(h)?,
            mlp_hidden: f(inter)?,
            mlp_out: f(h)?,
            token_buf: HostMappedBuffer::alloc(4, device_id)?,
            // [t, h, w, cache_pos] as 4 u32s (16 bytes).
            pos_ids_buf: HostMappedBuffer::alloc(16, device_id)?,
        })
    }
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

fn device_logits(ws: &DecodeWorkspace, vocab: usize) -> Result<Tensor> {
    ws.logits
        .as_tensor(Shape::new(vec![1, vocab]), ws.dtype)
        .map_err(Error::Cuda)
}

// ── Decode forward (graph body) ──────────────────────────────────────────

fn decode_forward_capturable(
    ctx: &CudaContext,
    ws: &DecodeWorkspace,
    weights: &Qwen3VLDecodeGraphWeights,
    kv: &mut dyn KvCache,
    cfg: &Qwen3VLDecodeGraphConfig,
    bucket_kv_len: usize,
) -> Result<()> {
    let device_id = ctx.device_id();
    let hidden = cfg.hidden_size;
    let n_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let head_dim = cfg.head_dim;
    let inter = cfg.intermediate_size;
    let vocab = cfg.vocab_size;
    let max_seq = cfg.max_seq_len;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let positions = ws.pos_ids_buf.address();
    let elem = cfg.dtype.size_in_bytes();
    let dtype = cfg.dtype;
    let kv_proj = n_kv_heads * head_dim;

    if dtype != DType::BF16 {
        return Err(Error::Other("Qwen3-VL decode graph: bf16 only".into()));
    }

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

    // Layer 0 pre-attention norm (standalone). Subsequent layers' pre-attn
    // norms are fused into the previous layer's post-FFN rms_norm_add.
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
        // ── QKV projection (fused if qkv_packed is Some) ──
        let (q_view, k_view, v_view) = if let Some(qkv_w) = layer.qkv_packed {
            let norm_view = ws.norm_out.view(0, ws.norm_out.len()).map_err(Error::Cuda)?;
            let qkv_wv = weight_view(qkv_w, device_id)?;
            let fused_n = hidden + 2 * kv_proj;
            kernels::gemm::write(ctx, dtype, 1, fused_n, hidden, 1.0, &norm_view, &qkv_wv, 0.0, &ws.qkv)
                ?;
            (
                ws.qkv.view(0, hidden * elem).map_err(Error::Cuda)?,
                ws.qkv.view(hidden * elem, kv_proj * elem).map_err(Error::Cuda)?,
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

        // ── QK-norm (per-head RMSNorm on Q and K) ──
        // Reshape [hidden] → [n_heads, head_dim] for Q; [kv_proj] → [n_kv_heads, head_dim] for K.
        // The rms_norm_bf16 kernel takes cols=head_dim, rows=n_heads and applies
        // the head_dim-length weight per row. Output to ws.q_normed / ws.k_normed.
        let q_norm_weight = weight_view(layer.q_norm_weight, device_id)?;
        let k_norm_weight = weight_view(layer.k_norm_weight, device_id)?;
        kernels::norm::rms_into(
            ctx,
            dtype,
            &q_view,
            &q_norm_weight,
            &ws.q_normed,
            head_dim,
            n_heads,
            cfg.rms_norm_eps,
        )?;
        kernels::norm::rms_into(
            ctx,
            dtype,
            &k_view,
            &k_norm_weight,
            &ws.k_normed,
            head_dim,
            n_kv_heads,
            cfg.rms_norm_eps,
        )?;

        // ── mRoPE on Q and K ──
        // sections is [T, H, W] but the kernel takes sec_h, sec_w (T is implicit).
        kernels::rope::apply_mrope_bf16_into(
            ctx,
            &ws.q_normed,
            &ws.q_rope,
            head_dim,
            n_heads,
            cfg.rope_theta,
            positions,
            cfg.mrope_section[1],
            cfg.mrope_section[2],
        )?;
        kernels::rope::apply_mrope_bf16_into(
            ctx,
            &ws.k_normed,
            &ws.k_rope,
            head_dim,
            n_kv_heads,
            cfg.rope_theta,
            positions,
            cfg.mrope_section[1],
            cfg.mrope_section[2],
        )?;

        // ── KV cache append (V from qkv, K from k_rope) ──
        // Position for the append: derived from the mRoPE-linear position.
        // The kv_cache_append_decode kernel takes a [1] u32 pos_ptr. For the
        // graph-captured path we need the linear cache position, which is
        // pos_ids[0] for text-only decode (or the effective linear pos for
        // multimodal decode — the caller writes it into pos_ids_buf[0]).
        //
        // WORKAROUND: We use pos_ids_buf's first u32 as the linear position.
        // For text-only after multimodal prefill, the caller sets pos_ids to
        // (linear+rope_delta, linear+rope_delta, linear+rope_delta) but we need
        // the linear cache index. To keep this simple, we track the cache
        // position separately: pos_ids_buf[3..7] holds the linear cache position
        // as a 4th u32.
        //
        // Actually: keep it simple — the caller writes pos_ids as [mrope_t,
        // mrope_h, mrope_w] and separately writes the linear cache index into
        // token_buf offset... no, that's ugly. Instead: use pos_ids_buf as
        // 4 u32s (16 bytes) — 3 for mRoPE, 1 for cache index. Update
        // pos_ids_buf allocation.
        //
        // For this first version, we assume mrope_t == cache index (text-only
        // case where all three axes equal the linear position + rope_delta,
        // and the cache index is the pure linear position without the delta).
        // The caller MUST write pos_ids[3] as the linear cache index.
        //
        // Cleanest: pass a separate `cache_pos_ptr` from a dedicated buffer.
        // I'll extend the workspace with `cache_pos_buf`.
        // (See DecodeGraph::decode below for how the caller populates it.)
        let cache_position = cache_pos_view(ws)?;
        kernels::cache::append_at(
            ctx,
            dtype,
            cache.k_buffer(i),
            &ws.k_rope,
            n_kv_heads,
            head_dim,
            max_seq,
            cache_position,
        )?;
        kernels::cache::append_at(
            ctx,
            dtype,
            cache.v_buffer(i),
            &v_view,
            n_kv_heads,
            head_dim,
            max_seq,
            cache_position,
        )?;

        // ── Flash Attention decode ──
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
            cache_position,
        )?;

        // ── wo projection ──
        let ao_view = ws.attn_out.view(0, ws.attn_out.len()).map_err(Error::Cuda)?;
        let wo_v = weight_view(layer.wo, device_id)?;
        kernels::gemm::write(ctx, dtype, 1, hidden, hidden, 1.0, &ao_view, &wo_v, 0.0, &ws.attn_proj)?;

        // ── Fused post-attn residual add + pre-FFN norm ──
        let ffn_norm_weight = weight_view(layer.ffn_norm_weight, device_id)?;
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

        // ── Gate/Up GEMM + fused SiLU*Mul (fused if gate_up_packed is Some) ──
        let fn_view = ws.ffn_norm_out.view(0, ws.ffn_norm_out.len()).map_err(Error::Cuda)?;
        if let Some(gate_up_w) = layer.gate_up_packed {
            let gu_wv = weight_view(gate_up_w, device_id)?;
            kernels::gemm::write(ctx, dtype, 1, 2 * inter, hidden, 1.0, &fn_view, &gu_wv, 0.0, &ws.gate_up)
                ?;
            kernels::activation::silu_mul_bf16_into(ctx, &ws.gate_up, &ws.mlp_hidden, inter)?;
        } else {
            // Fallback path: separate gate/up GEMMs + silu + mul. We don't
            // have a dedicated `ws.gate` in the Qwen3-VL workspace (packed is
            // the fast path), so this fallback requires packed to be present.
            return Err(Error::Other("Qwen3-VL decode graph requires packed Gate/Up weights".into()));
        }

        // ── down GEMM ──
        let mh_view = ws.mlp_hidden.view(0, ws.mlp_hidden.len()).map_err(Error::Cuda)?;
        let wd_v = weight_view(layer.w_down, device_id)?;
        kernels::gemm::write(ctx, dtype, 1, hidden, inter, 1.0, &mh_view, &wd_v, 0.0, &ws.mlp_out)?;

        // ── Fused post-FFN residual add + next layer's pre-attn norm
        //    (or the final output norm for the last layer) ──
        let next_norm_w = if i + 1 < weights.layers.len() {
            weights.layers[i + 1].attn_norm_weight
        } else {
            weights.output_norm_weight
        };
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
    }

    // ── lm_head matmul (tied embeddings — lm_head IS the transposed
    //    token_embedding, passed in Qwen3VLDecodeGraphWeights) ──
    let no_view = ws.norm_out.view(0, ws.norm_out.len()).map_err(Error::Cuda)?;
    let lm_v = weight_view(weights.lm_head, device_id)?;
    kernels::gemm::write(ctx, dtype, 1, vocab, hidden, 1.0, &no_view, &lm_v, 0.0, &ws.logits)?;

    Ok(())
}

/// Pointer to the linear cache position (4th u32 in pos_ids_buf).
/// Layout: `[mrope_t, mrope_h, mrope_w, linear_cache_pos]`.
fn cache_pos_view(ws: &DecodeWorkspace) -> Result<DeviceAddress> {
    ws.pos_ids_buf.address_at(12, 4).map_err(Error::Cuda)
}

// ── Graph capture/replay ─────────────────────────────────────────────────

struct BucketGraph {
    bucket_kv_len: usize,
    graph: Box<dyn Graph>,
}

pub struct Qwen3VLDecodeGraph {
    config: Qwen3VLDecodeGraphConfig,
    workspace: DecodeWorkspace,
    buckets: Vec<BucketGraph>,
}

impl Qwen3VLDecodeGraph {
    pub fn new(backend: &CudaBackend, cfg: Qwen3VLDecodeGraphConfig) -> Result<Self> {
        let ws = DecodeWorkspace::new(backend.device_id(), &cfg)
            .map_err(Error::Other)?;
        Ok(Self { config: cfg, workspace: ws, buckets: Vec::new() })
    }

    fn bucket_for(&self, cache_pos: u32) -> usize {
        let kv_len = cache_pos as usize + 1;
        kv_len.next_power_of_two().min(self.config.max_seq_len).max(1)
    }

    /// Decode one token. `pos_ids` is the mRoPE (t, h, w) coordinate;
    /// `cache_pos` is the linear KV cache index for this token.
    pub fn decode(
        &mut self,
        backend: &CudaBackend,
        weights: &Qwen3VLDecodeGraphWeights,
        kv: &mut dyn KvCache,
        token: u32,
        pos_ids: [u32; 3],
        cache_pos: u32,
    ) -> Result<Tensor> {
        let ctx = backend.context();
        let bucket_kv_len = self.bucket_for(cache_pos);
        let vocab = self.config.vocab_size;
        let have = self.buckets.iter().any(|b| b.bucket_kv_len == bucket_kv_len);

        // Reallocate the pos_ids buffer to 16 bytes if needed (lazily grow
        // to include the 4th u32 for cache position — the constructor
        // allocated 12 bytes above; we need 16). Cheaper to just reallocate
        // once at first use.
        // Actually — we always need 16 bytes. Fix the workspace ctor to
        // allocate 16 (see DecodeWorkspace::new).

        if !have {
            self.workspace.token_buf.write_u32(token).map_err(Error::Cuda)?;
            self.workspace
                .pos_ids_buf
                .write_u32s(&[pos_ids[0], pos_ids[1], pos_ids[2], cache_pos])
                .map_err(Error::Cuda)?;
            decode_forward_capturable(ctx, &self.workspace, weights, kv, &self.config, bucket_kv_len)?;
            backend.synchronize()?;
            let logits = device_logits(&self.workspace, vocab)?;

            backend.begin_capture_relaxed()?;
            let cap_res = decode_forward_capturable(ctx, &self.workspace, weights, kv, &self.config, bucket_kv_len);
            let graph = backend.end_capture()?;
            cap_res?;
            self.buckets.push(BucketGraph { bucket_kv_len, graph });
            return Ok(logits);
        }

        self.workspace.token_buf.write_u32(token).map_err(Error::Cuda)?;
        self.workspace
            .pos_ids_buf
            .write_u32s(&[pos_ids[0], pos_ids[1], pos_ids[2], cache_pos])
            .map_err(Error::Cuda)?;
        let graph = &self
            .buckets
            .iter()
            .find(|bucket| bucket.bucket_kv_len == bucket_kv_len)
            .unwrap()
            .graph;
        graph.replay()?;
        // Sampling follows on this same stream and synchronizes only when its
        // 16-byte result is copied to the host.
        device_logits(&self.workspace, vocab)
    }
}
