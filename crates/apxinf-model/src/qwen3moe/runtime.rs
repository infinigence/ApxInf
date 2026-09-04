//! CUDA execution for Qwen3-MoE AutoAWQ INT4.
//!
//! Two paths share the weights and the KV cache:
//!
//! * **Prefill** (`M = seq_len` tokens, eager): dense projections dequantize
//!   the packed weights to a BF16 scratch and run cuBLAS GEMMs; attention is
//!   the causal FlashAttention-2 FP16 kernel; experts are executed as
//!   per-expert GEMMs over token rows grouped by routed expert (a
//!   correctness scaffold whose exit criterion is a fused W4A16 grouped GEMM).
//! * **Decode** (`M = 1`, capturable): every projection is a split-K W4A16
//!   GEMV on the packed weights; the eight routed experts are evaluated by one
//!   multi-slot launch that reads the expert indices on the device, so the
//!   whole token step has a fixed shape and can be captured into a CUDA
//!   graph bucketed by KV length.
//!
//! All activations are BF16 with FP32 accumulation; the FP16 attention inputs
//! are converted on the fly.

use apxinf_core::{Backend, DType, Error, Result, Shape, Tensor};

use super::config::Qwen3MoeConfig;
use super::weights::{AwqLinear, Qwen3MoeWeights};
use crate::accelerator::cuda::{
    kernels, Context as CudaContext, CublasTranspose, DeviceAddress, DeviceBuffer,
    MappedBuffer, RuntimeBackend as CudaBackend,
};
use kernels::gemm::w4a16::{self, AwqWeightView, GemvSlots};

fn cuda(error: String) -> Error {
    Error::Cuda(error)
}

fn alloc(device: usize, bytes: usize) -> Result<DeviceBuffer> {
    DeviceBuffer::alloc_zeros(bytes.max(16), device).map_err(cuda)
}

fn view(buffer: &DeviceBuffer, offset: usize, len: usize) -> Result<DeviceBuffer> {
    buffer.view(offset, len).map_err(cuda)
}

const BF16: usize = 2;

/// Fixed-address BF16 KV cache: `[n_kv_heads, max_seq_len, head_dim]` per
/// layer, the layout consumed by `kernels::cache` and the decode flash kernel.
/// `clear` only resets the logical length so captured graphs stay valid.
pub struct KvCache {
    k: Vec<DeviceBuffer>,
    v: Vec<DeviceBuffer>,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
    pub seq_len: usize,
}

impl KvCache {
    /// Total bytes for `max_seq_len` positions across every layer.
    fn bytes(cfg: &Qwen3MoeConfig, max_seq_len: usize) -> usize {
        2 * cfg.n_layers * cfg.n_kv_heads * max_seq_len * cfg.head_dim * BF16
    }

    fn new(device: usize, cfg: &Qwen3MoeConfig, max_seq_len: usize) -> Result<Self> {
        let layer_bytes = cfg.n_kv_heads * max_seq_len * cfg.head_dim * BF16;
        let mut k = Vec::with_capacity(cfg.n_layers);
        let mut v = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            k.push(alloc(device, layer_bytes)?);
            v.push(alloc(device, layer_bytes)?);
        }
        Ok(Self {
            k,
            v,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            max_seq_len,
            seq_len: 0,
        })
    }
}

/// Split-K factors chosen so every decode GEMV launch has enough blocks to
/// saturate LPDDR5X bandwidth on a ~20-SM part.
#[derive(Clone, Copy, Debug)]
struct DecodeSplits {
    qkv: usize,
    o: usize,
    gate_up: usize,
    down: usize,
}

impl DecodeSplits {
    fn choose(cfg: &Qwen3MoeConfig, sm_count: usize) -> Self {
        // Blocks to aim for per launch, as a multiple of the SM count. The
        // GEMV is latency-bound, so what matters is requests in flight —
        // but the depth comes from `W4A16_GEMV_UNROLL` inside the kernel, not
        // from more blocks. Sweeping this on Thor-U (14 SMs) at ISL 128 gave
        // TPOT 15.98 / 16.65 / 17.18 / 18.99 ms for 4 / 8 / 16 / 32: past 4
        // the extra splits cost more in duplicated activation staging and
        // partial-sum reduction than the parallelism is worth. The knob stays
        // so a part with a different SM count can be re-tuned without a build.
        let per_sm = std::env::var("APXINF_QWEN3MOE_BLOCKS_PER_SM")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(4);
        let target = (sm_count * per_sm).max(64);
        let pick = |n_out: usize, k: usize, slots: usize| -> usize {
            let tiles = n_out.div_ceil(256) * slots;
            let wanted = target.div_ceil(tiles).max(1);
            // Keep at least 64 rows (8 per warp) per block.
            wanted.min(k / 64).max(1)
        };
        Self {
            qkv: pick(cfg.qkv_dim(), cfg.hidden_size, 1),
            o: pick(cfg.hidden_size, cfg.n_heads * cfg.head_dim, 1),
            gate_up: pick(2 * cfg.moe_intermediate_size, cfg.hidden_size, cfg.num_experts_per_tok),
            down: pick(cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts_per_tok),
        }
    }
}

/// Stable-address buffers for one decode step.
struct DecodeWorkspace {
    x: DeviceBuffer,           // [hidden] residual stream
    normed: DeviceBuffer,      // [hidden]
    qkv: DeviceBuffer,         // [q | k | v]
    q_normed: DeviceBuffer,    // [q_dim]
    k_normed: DeviceBuffer,    // [kv_dim]
    q_rope: DeviceBuffer,      // [q_dim]
    k_rope: DeviceBuffer,      // [kv_dim]
    attn_out: DeviceBuffer,    // [q_dim]
    attn_proj: DeviceBuffer,   // [hidden]
    ffn_normed: DeviceBuffer,  // [hidden]
    router_logits: DeviceBuffer, // [experts] bf16
    topk_idx: DeviceBuffer,    // [k] i32
    topk_w: DeviceBuffer,      // [k] f32
    p_qkv: DeviceBuffer,       // f32 partials
    p_o: DeviceBuffer,
    p_gate_up: DeviceBuffer,
    h: DeviceBuffer,           // [k, inter] bf16
    p_down: DeviceBuffer,
    moe_out: DeviceBuffer,     // [hidden]
    logits: DeviceBuffer,      // [vocab] bf16
    token: MappedBuffer,       // u32
    position: MappedBuffer,    // u32
    splits: DecodeSplits,
}

impl DecodeWorkspace {
    fn new(device: usize, cfg: &Qwen3MoeConfig, splits: DecodeSplits) -> Result<Self> {
        let h = cfg.hidden_size;
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.kv_dim();
        let inter = cfg.moe_intermediate_size;
        let k = cfg.num_experts_per_tok;
        Ok(Self {
            x: alloc(device, h * BF16)?,
            normed: alloc(device, h * BF16)?,
            qkv: alloc(device, cfg.qkv_dim() * BF16)?,
            q_normed: alloc(device, q_dim * BF16)?,
            k_normed: alloc(device, kv_dim * BF16)?,
            q_rope: alloc(device, q_dim * BF16)?,
            k_rope: alloc(device, kv_dim * BF16)?,
            attn_out: alloc(device, q_dim * BF16)?,
            attn_proj: alloc(device, h * BF16)?,
            ffn_normed: alloc(device, h * BF16)?,
            router_logits: alloc(device, cfg.num_experts * BF16)?,
            topk_idx: alloc(device, k * 4)?,
            topk_w: alloc(device, k * 4)?,
            p_qkv: alloc(device, w4a16::partial_bytes(cfg.qkv_dim(), splits.qkv, 1))?,
            p_o: alloc(device, w4a16::partial_bytes(h, splits.o, 1))?,
            p_gate_up: alloc(device, w4a16::partial_bytes(2 * inter, splits.gate_up, k))?,
            h: alloc(device, k * inter * BF16)?,
            p_down: alloc(device, w4a16::partial_bytes(h, splits.down, k))?,
            moe_out: alloc(device, h * BF16)?,
            logits: alloc(device, cfg.vocab_size * BF16)?,
            token: MappedBuffer::alloc(4, device).map_err(cuda)?,
            position: MappedBuffer::alloc(4, device).map_err(cuda)?,
            splits,
        })
    }
}

/// BF16 scratch for dequantized weights during prefill (allocated lazily on
/// the first prefill; ~1.2 GB for Qwen3-30B-A3B).
struct DequantWorkspace {
    qkv: DeviceBuffer,     // [hidden, qkv_dim]
    o: DeviceBuffer,       // [q_dim, hidden]
    gate_up: DeviceBuffer, // [experts, hidden, 2*inter]
    down: DeviceBuffer,    // [experts, inter, hidden]
}

impl DequantWorkspace {
    fn bytes(cfg: &Qwen3MoeConfig) -> usize {
        let q_dim = cfg.n_heads * cfg.head_dim;
        let inter = cfg.moe_intermediate_size;
        (cfg.hidden_size * cfg.qkv_dim()
            + q_dim * cfg.hidden_size
            + cfg.num_experts * cfg.hidden_size * 2 * inter
            + cfg.num_experts * inter * cfg.hidden_size)
            * BF16
    }
}

/// Per-prefill activation buffers sized for `seq_len` tokens.
struct PrefillWorkspace {
    seq_len: usize,
    ids: DeviceBuffer,
    x: DeviceBuffer,
    normed: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    q_normed: DeviceBuffer,
    k_normed: DeviceBuffer,
    q_f16: DeviceBuffer,
    k_f16: DeviceBuffer,
    v_f16: DeviceBuffer,
    o_f16: DeviceBuffer,
    lse: DeviceBuffer,
    attn_out: DeviceBuffer,
    attn_proj: DeviceBuffer,
    ffn_normed: DeviceBuffer,
    router_logits: DeviceBuffer,
    topk_idx: DeviceBuffer,
    topk_w: DeviceBuffer,
    source_rows: DeviceBuffer,
    slot_rows: DeviceBuffer,
    gathered: DeviceBuffer,
    gate_up_out: DeviceBuffer,
    h: DeviceBuffer,
    y: DeviceBuffer,
    moe_out: DeviceBuffer,
    logits: DeviceBuffer,
}

impl PrefillWorkspace {
    fn new(device: usize, cfg: &Qwen3MoeConfig, seq_len: usize) -> Result<Self> {
        let s = seq_len;
        let hd = cfg.hidden_size;
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.kv_dim();
        let inter = cfg.moe_intermediate_size;
        let k = cfg.num_experts_per_tok;
        let routed = s * k;
        Ok(Self {
            seq_len,
            ids: alloc(device, s * 4)?,
            x: alloc(device, s * hd * BF16)?,
            normed: alloc(device, s * hd * BF16)?,
            q: alloc(device, s * q_dim * BF16)?,
            k: alloc(device, s * kv_dim * BF16)?,
            v: alloc(device, s * kv_dim * BF16)?,
            q_normed: alloc(device, s * q_dim * BF16)?,
            k_normed: alloc(device, s * kv_dim * BF16)?,
            q_f16: alloc(device, s * q_dim * 2)?,
            k_f16: alloc(device, s * kv_dim * 2)?,
            v_f16: alloc(device, s * kv_dim * 2)?,
            o_f16: alloc(device, s * q_dim * 2)?,
            lse: alloc(device, cfg.n_heads * s * 4)?,
            attn_out: alloc(device, s * q_dim * BF16)?,
            attn_proj: alloc(device, s * hd * BF16)?,
            ffn_normed: alloc(device, s * hd * BF16)?,
            router_logits: alloc(device, s * cfg.num_experts * BF16)?,
            topk_idx: alloc(device, routed * 4)?,
            topk_w: alloc(device, routed * 4)?,
            source_rows: alloc(device, routed * 4)?,
            slot_rows: alloc(device, routed * 4)?,
            gathered: alloc(device, routed * hd * BF16)?,
            gate_up_out: alloc(device, routed * 2 * inter * BF16)?,
            h: alloc(device, routed * inter * BF16)?,
            y: alloc(device, routed * hd * BF16)?,
            moe_out: alloc(device, s * hd * BF16)?,
            logits: alloc(device, cfg.vocab_size * BF16)?,
        })
    }
}

fn awq_view<'a>(linear: &'a AwqLinear, group_size: usize) -> AwqWeightView<'a> {
    AwqWeightView {
        qweight: &linear.qweight,
        qzeros: &linear.qzeros,
        scales: &linear.scales,
        in_dim: linear.in_dim,
        out_dim: linear.out_dim,
        group_size,
        experts: linear.experts,
        stride_q: linear.stride_q(),
        stride_z: linear.stride_z(group_size),
        stride_s: linear.stride_s(group_size),
    }
}

struct BucketGraph {
    bucket_kv_len: usize,
    graph: Box<dyn apxinf_core::Graph>,
}

pub struct Qwen3MoeRuntime {
    cfg: Qwen3MoeConfig,
    weights: Qwen3MoeWeights,
    kv: KvCache,
    decode: DecodeWorkspace,
    dequant: Option<DequantWorkspace>,
    prefill: Option<PrefillWorkspace>,
    graphs: Vec<BucketGraph>,
    use_graphs: bool,
}

impl Qwen3MoeRuntime {
    /// Bytes of the `cudaMalloc` pool this runtime needs once the weights are
    /// in place. The weight loader subtracts this from its budget, because the
    /// KV cache and the dequant scratch have nowhere else to go: they are
    /// written every step, so putting them in mapped host memory would cost
    /// more than spilling cold weights there.
    ///
    /// The dequant scratch dominates and is the clearest sign that the prefill
    /// path is a scaffold — a fused W4A16 grouped GEMM would read the packed
    /// weights in place and delete this term.
    pub fn pool_reserve_bytes(cfg: &Qwen3MoeConfig, max_seq_len: usize) -> usize {
        let kv = KvCache::bytes(cfg, max_seq_len);
        let dequant = DequantWorkspace::bytes(cfg);
        // Prefill activations scale with the sequence; 1024 tokens is the ISL
        // this model is tuned for and the buffers are small next to the rest.
        let activations = 384 << 20;
        kv + dequant + activations
    }

    pub fn new(
        backend: &CudaBackend,
        cfg: Qwen3MoeConfig,
        weights: Qwen3MoeWeights,
        max_seq_len: usize,
    ) -> Result<Self> {
        let ctx = backend.context();
        let device = ctx.device_id();
        let kv = KvCache::new(device, &cfg, max_seq_len)?;
        let splits = DecodeSplits::choose(&cfg, ctx.caps().multiprocessor_count as usize);
        let decode = DecodeWorkspace::new(device, &cfg, splits)?;
        Ok(Self {
            cfg,
            weights,
            kv,
            decode,
            dequant: None,
            prefill: None,
            graphs: Vec::new(),
            // Graph capture is the fast path, but it fixes the launch
            // configuration for the whole run, so numerics bisection wants a
            // way to fall back to eager launches without a rebuild.
            use_graphs: !matches!(
                std::env::var("APXINF_QWEN3MOE_GRAPHS").as_deref(),
                Ok("0") | Ok("off") | Ok("false")
            ),
        })
    }

    pub fn config(&self) -> &Qwen3MoeConfig {
        &self.cfg
    }

    pub fn weights(&self) -> &Qwen3MoeWeights {
        &self.weights
    }

    pub fn kv_len(&self) -> usize {
        self.kv.seq_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.kv.max_seq_len
    }

    pub fn set_use_graphs(&mut self, enabled: bool) {
        self.use_graphs = enabled;
    }

    pub fn reset(&mut self) {
        self.kv.seq_len = 0;
    }

    // ── Prefill ──────────────────────────────────────────────────────────

    fn ensure_dequant(&mut self, device: usize) -> Result<()> {
        if self.dequant.is_some() {
            return Ok(());
        }
        let c = &self.cfg;
        let q_dim = c.n_heads * c.head_dim;
        self.dequant = Some(DequantWorkspace {
            qkv: alloc(device, c.hidden_size * c.qkv_dim() * BF16)?,
            o: alloc(device, q_dim * c.hidden_size * BF16)?,
            gate_up: alloc(device, c.num_experts * c.hidden_size * 2 * c.moe_intermediate_size * BF16)?,
            down: alloc(device, c.num_experts * c.moe_intermediate_size * c.hidden_size * BF16)?,
        });
        Ok(())
    }

    fn ensure_prefill(&mut self, device: usize, seq_len: usize) -> Result<()> {
        if self.prefill.as_ref().is_some_and(|ws| ws.seq_len >= seq_len) {
            return Ok(());
        }
        self.prefill = None;
        self.prefill = Some(PrefillWorkspace::new(device, &self.cfg, seq_len)?);
        Ok(())
    }

    /// Run a full prompt from position 0 and return the logits of the last
    /// token as a `[1, vocab]` BF16 device tensor.
    pub fn prefill(&mut self, backend: &CudaBackend, token_ids: &[u32]) -> Result<Tensor> {
        let ctx = backend.context();
        let device = ctx.device_id();
        let s = token_ids.len();
        if s == 0 {
            return Err(Error::Other("qwen3moe prefill: empty prompt".into()));
        }
        if self.kv.seq_len != 0 {
            return Err(Error::Other(
                "qwen3moe prefill: KV cache is not empty (call reset first)".into(),
            ));
        }
        if s > self.kv.max_seq_len {
            return Err(Error::Other(format!(
                "qwen3moe prefill: {s} tokens exceed max_seq_len {}",
                self.kv.max_seq_len
            )));
        }
        self.ensure_dequant(device)?;
        self.ensure_prefill(device, s)?;
        let c = self.cfg.clone();
        let group = c.group_size();
        let hd = c.hidden_size;
        let q_dim = c.n_heads * c.head_dim;
        let kv_dim = c.kv_dim();
        let inter = c.moe_intermediate_size;
        let topk = c.num_experts_per_tok;
        let experts = c.num_experts;
        let eps = c.rms_norm_eps;
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        let ws = self.prefill.as_ref().unwrap();
        let dq = self.dequant.as_ref().unwrap();
        let w = &self.weights;

        // Token ids -> embeddings.
        let id_bytes: Vec<u8> = token_ids.iter().flat_map(|t| t.to_ne_bytes()).collect();
        ws.ids.copy_from_host(&id_bytes).map_err(cuda)?;
        let x = view(&ws.x, 0, s * hd * BF16)?;
        kernels::embedding::lookup_into(
            ctx,
            DType::BF16,
            &w.embed_tokens,
            ws.ids.address(),
            &x,
            hd,
            s,
        )?;
        let normed = view(&ws.normed, 0, s * hd * BF16)?;
        kernels::norm::rms_into(ctx, DType::BF16, &x, &w.layers[0].attn_norm, &normed, hd, s, eps)?;

        let q = view(&ws.q, 0, s * q_dim * BF16)?;
        let k = view(&ws.k, 0, s * kv_dim * BF16)?;
        let v = view(&ws.v, 0, s * kv_dim * BF16)?;
        let q_normed = view(&ws.q_normed, 0, s * q_dim * BF16)?;
        let k_normed = view(&ws.k_normed, 0, s * kv_dim * BF16)?;
        let q_f16 = view(&ws.q_f16, 0, s * q_dim * 2)?;
        let k_f16 = view(&ws.k_f16, 0, s * kv_dim * 2)?;
        let v_f16 = view(&ws.v_f16, 0, s * kv_dim * 2)?;
        let o_f16 = view(&ws.o_f16, 0, s * q_dim * 2)?;
        let lse = view(&ws.lse, 0, c.n_heads * s * 4)?;
        let attn_out = view(&ws.attn_out, 0, s * q_dim * BF16)?;
        let attn_proj = view(&ws.attn_proj, 0, s * hd * BF16)?;
        let ffn_normed = view(&ws.ffn_normed, 0, s * hd * BF16)?;
        let router_logits = view(&ws.router_logits, 0, s * experts * BF16)?;
        let topk_idx = view(&ws.topk_idx, 0, s * topk * 4)?;
        let topk_w = view(&ws.topk_w, 0, s * topk * 4)?;
        let moe_out = view(&ws.moe_out, 0, s * hd * BF16)?;
        let mut host_idx = vec![0u8; s * topk * 4];

        for (li, layer) in w.layers.iter().enumerate() {
            // ── attention projections: dequant packed [q|k|v] then 3 strided GEMMs
            w4a16::dequant_bf16_into(ctx, awq_view(&layer.qkv, group), &dq.qkv)?;
            let ldb = c.qkv_dim() as i32;
            let mut col = 0usize;
            for (out, n) in [(&q, q_dim), (&k, kv_dim), (&v, kv_dim)] {
                let b = view(&dq.qkv, col * BF16, dq.qkv.len() - col * BF16)?;
                kernels::gemm::write_ex(
                    ctx,
                    DType::BF16,
                    CublasTranspose::None,
                    CublasTranspose::None,
                    s,
                    n,
                    hd,
                    1.0,
                    &normed,
                    hd as i32,
                    &b,
                    ldb,
                    0.0,
                    out,
                    n as i32,
                )?;
                col += n;
            }
            // ── per-head QK-norm, RoPE (positions 0..s)
            kernels::norm::rms_into(ctx, DType::BF16, &q, &layer.q_norm, &q_normed, c.head_dim, s * c.n_heads, eps)?;
            kernels::norm::rms_into(ctx, DType::BF16, &k, &layer.k_norm, &k_normed, c.head_dim, s * c.n_kv_heads, eps)?;
            let q_t = q_normed.as_tensor(Shape::new(vec![s, c.n_heads, c.head_dim]), DType::BF16).map_err(cuda)?;
            let k_t = k_normed.as_tensor(Shape::new(vec![s, c.n_kv_heads, c.head_dim]), DType::BF16).map_err(cuda)?;
            let q_rope = kernels::rope::apply_batched(ctx, &q_t, c.n_heads, c.head_dim, c.rope_theta, 0)?;
            let k_rope = kernels::rope::apply_batched(ctx, &k_t, c.n_kv_heads, c.head_dim, c.rope_theta, 0)?;
            let q_rope_buf = DeviceBuffer::from_tensor(&q_rope).map_err(cuda)?;
            let k_rope_buf = DeviceBuffer::from_tensor(&k_rope).map_err(cuda)?;
            // ── KV cache write (positions 0..s)
            let v_t = v.as_tensor(Shape::new(vec![s, c.n_kv_heads, c.head_dim]), DType::BF16).map_err(cuda)?;
            kernels::cache::append(ctx, &self.kv.k[li], &k_rope, c.n_kv_heads, c.head_dim, self.kv.max_seq_len, 0, s)?;
            kernels::cache::append(ctx, &self.kv.v[li], &v_t, c.n_kv_heads, c.head_dim, self.kv.max_seq_len, 0, s)?;
            // ── causal attention (FA2, fp16)
            kernels::elementwise::convert_bf16_to_f16_into(ctx, &q_rope_buf, &q_f16, s * q_dim)?;
            kernels::elementwise::convert_bf16_to_f16_into(ctx, &k_rope_buf, &k_f16, s * kv_dim)?;
            kernels::elementwise::convert_bf16_to_f16_into(ctx, &v, &v_f16, s * kv_dim)?;
            kernels::attention::causal_prefill_f16_into(
                ctx, &q_f16, &k_f16, &v_f16, &o_f16, &lse, s, s, c.n_heads, c.n_kv_heads, scale,
            )?;
            kernels::elementwise::convert_f16_to_bf16_into(ctx, &o_f16, &attn_out, s * q_dim)?;
            // ── output projection + residual + FFN norm
            w4a16::dequant_bf16_into(ctx, awq_view(&layer.o, group), &dq.o)?;
            kernels::gemm::write(ctx, DType::BF16, s, hd, q_dim, 1.0, &attn_out, &dq.o, 0.0, &attn_proj)?;
            kernels::norm::residual_add_rms_bf16_into(ctx, &x, &attn_proj, &layer.ffn_norm, &ffn_normed, hd, s, eps)?;
            // ── router
            kernels::gemm::write(ctx, DType::BF16, s, experts, hd, 1.0, &ffn_normed, &layer.router, 0.0, &router_logits)?;
            kernels::moe::router_topk_into(ctx, &router_logits, s, experts, topk, c.norm_topk_prob, &topk_idx, &topk_w)?;
            // ── group routed rows by expert (host-side permutation)
            ctx.synchronize().map_err(cuda)?;
            topk_idx.copy_to_host(&mut host_idx).map_err(cuda)?;
            let assignments: Vec<usize> = host_idx
                .chunks_exact(4)
                .map(|b| i32::from_ne_bytes([b[0], b[1], b[2], b[3]]) as usize)
                .collect();
            let mut counts = vec![0usize; experts];
            for &e in &assignments {
                if e >= experts {
                    return Err(Error::Other(format!("qwen3moe: router produced expert {e}")));
                }
                counts[e] += 1;
            }
            let mut offsets = vec![0usize; experts + 1];
            for e in 0..experts {
                offsets[e + 1] = offsets[e] + counts[e];
            }
            let mut cursor = offsets.clone();
            let mut source_rows = vec![0i32; s * topk];
            let mut slot_rows = vec![0i32; s * topk];
            for (slot, &e) in assignments.iter().enumerate() {
                let row = cursor[e];
                cursor[e] += 1;
                source_rows[row] = (slot / topk) as i32;
                slot_rows[slot] = row as i32;
            }
            let to_bytes = |v: &[i32]| v.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<u8>>();
            let source_rows_dev = view(&ws.source_rows, 0, s * topk * 4)?;
            let slot_rows_dev = view(&ws.slot_rows, 0, s * topk * 4)?;
            source_rows_dev.copy_from_host(&to_bytes(&source_rows)).map_err(cuda)?;
            slot_rows_dev.copy_from_host(&to_bytes(&slot_rows)).map_err(cuda)?;
            let gathered = view(&ws.gathered, 0, s * topk * hd * BF16)?;
            kernels::moe::gather_rows_into(ctx, &ffn_normed, s, &source_rows_dev, s * topk, hd, &gathered)?;
            // ── experts: dequantize the whole layer once, then one GEMM pair per used expert
            w4a16::dequant_bf16_into(ctx, awq_view(&layer.gate_up, group), &dq.gate_up)?;
            w4a16::dequant_bf16_into(ctx, awq_view(&layer.down, group), &dq.down)?;
            let gu_expert = hd * 2 * inter * BF16;
            let down_expert = inter * hd * BF16;
            for e in 0..experts {
                let rows = counts[e];
                if rows == 0 {
                    continue;
                }
                let off = offsets[e];
                let a = view(&gathered, off * hd * BF16, rows * hd * BF16)?;
                let wg = view(&dq.gate_up, e * gu_expert, gu_expert)?;
                let gu = view(&ws.gate_up_out, off * 2 * inter * BF16, rows * 2 * inter * BF16)?;
                kernels::gemm::write(ctx, DType::BF16, rows, 2 * inter, hd, 1.0, &a, &wg, 0.0, &gu)?;
                let hbuf = view(&ws.h, off * inter * BF16, rows * inter * BF16)?;
                kernels::moe::silu_mul_rows_into(ctx, &gu, rows, inter, &hbuf)?;
                let wd = view(&dq.down, e * down_expert, down_expert)?;
                let y = view(&ws.y, off * hd * BF16, rows * hd * BF16)?;
                kernels::gemm::write(ctx, DType::BF16, rows, hd, inter, 1.0, &hbuf, &wd, 0.0, &y)?;
            }
            let y_all = view(&ws.y, 0, s * topk * hd * BF16)?;
            kernels::moe::weighted_gather_sum_into(ctx, &y_all, &slot_rows_dev, &topk_w, s, topk, hd, &moe_out)?;
            // ── residual + next norm (final norm after the last layer)
            let next_norm = if li + 1 < w.layers.len() {
                &w.layers[li + 1].attn_norm
            } else {
                &w.final_norm
            };
            kernels::norm::residual_add_rms_bf16_into(ctx, &x, &moe_out, next_norm, &normed, hd, s, eps)?;
        }
        self.kv.seq_len = s;

        // Logits for the last token only: [1, vocab] = normed[s-1] · lm_head^T.
        let last = view(&normed, (s - 1) * hd * BF16, hd * BF16)?;
        let logits = view(&ws.logits, 0, c.vocab_size * BF16)?;
        kernels::gemm::write_ex(
            ctx,
            DType::BF16,
            CublasTranspose::None,
            CublasTranspose::Transpose,
            1,
            c.vocab_size,
            hd,
            1.0,
            &last,
            hd as i32,
            &w.lm_head,
            hd as i32,
            0.0,
            &logits,
            c.vocab_size as i32,
        )?;
        logits
            .as_tensor(Shape::new(vec![1, c.vocab_size]), DType::BF16)
            .map_err(cuda)
    }

    // ── Decode ───────────────────────────────────────────────────────────

    fn bucket_for(&self, kv_len: usize) -> usize {
        kv_len.next_power_of_two().clamp(1, self.kv.max_seq_len)
    }

    /// Fixed-shape decode body; every launch reads token/position/expert
    /// selection from device memory so it can be captured.
    fn decode_body(&self, ctx: &CudaContext, bucket_kv_len: usize) -> Result<()> {
        let c = &self.cfg;
        let ws = &self.decode;
        let w = &self.weights;
        let group = c.group_size();
        let hd = c.hidden_size;
        let q_dim = c.n_heads * c.head_dim;
        let kv_dim = c.kv_dim();
        let inter = c.moe_intermediate_size;
        let topk = c.num_experts_per_tok;
        let eps = c.rms_norm_eps;
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        let position: DeviceAddress = ws.position.address();
        let sp = ws.splits;

        kernels::embedding::lookup_into(ctx, DType::BF16, &w.embed_tokens, ws.token.address(), &ws.x, hd, 1)?;
        kernels::norm::rms_into(ctx, DType::BF16, &ws.x, &w.layers[0].attn_norm, &ws.normed, hd, 1, eps)?;

        for (li, layer) in w.layers.iter().enumerate() {
            // qkv = normed · W_qkv
            w4a16::gemv_partial_into(ctx, &ws.normed, awq_view(&layer.qkv, group), GemvSlots::dense(), sp.qkv, &ws.p_qkv)?;
            w4a16::partial_sum_into(ctx, &ws.p_qkv, sp.qkv, c.qkv_dim(), &ws.qkv)?;
            let q = view(&ws.qkv, 0, q_dim * BF16)?;
            let k = view(&ws.qkv, q_dim * BF16, kv_dim * BF16)?;
            let v = view(&ws.qkv, (q_dim + kv_dim) * BF16, kv_dim * BF16)?;
            kernels::norm::rms_into(ctx, DType::BF16, &q, &layer.q_norm, &ws.q_normed, c.head_dim, c.n_heads, eps)?;
            kernels::norm::rms_into(ctx, DType::BF16, &k, &layer.k_norm, &ws.k_normed, c.head_dim, c.n_kv_heads, eps)?;
            kernels::rope::apply_into(ctx, DType::BF16, &ws.q_normed, &ws.q_rope, c.head_dim, c.n_heads, c.rope_theta, position)?;
            kernels::rope::apply_into(ctx, DType::BF16, &ws.k_normed, &ws.k_rope, c.head_dim, c.n_kv_heads, c.rope_theta, position)?;
            kernels::cache::append_at(ctx, DType::BF16, &self.kv.k[li], &ws.k_rope, c.n_kv_heads, c.head_dim, self.kv.max_seq_len, position)?;
            kernels::cache::append_at(ctx, DType::BF16, &self.kv.v[li], &v, c.n_kv_heads, c.head_dim, self.kv.max_seq_len, position)?;
            kernels::attention::flash_bf16_into(
                ctx, &ws.q_rope, &self.kv.k[li], &self.kv.v[li], &ws.attn_out,
                c.n_heads, c.n_kv_heads, c.head_dim, bucket_kv_len, self.kv.max_seq_len, scale, position,
            )?;
            w4a16::gemv_partial_into(ctx, &ws.attn_out, awq_view(&layer.o, group), GemvSlots::dense(), sp.o, &ws.p_o)?;
            w4a16::partial_sum_into(ctx, &ws.p_o, sp.o, hd, &ws.attn_proj)?;
            kernels::norm::residual_add_rms_bf16_into(ctx, &ws.x, &ws.attn_proj, &layer.ffn_norm, &ws.ffn_normed, hd, 1, eps)?;
            // router (cuBLAS M=1) + top-k on device
            kernels::gemm::write(ctx, DType::BF16, 1, c.num_experts, hd, 1.0, &ws.ffn_normed, &layer.router, 0.0, &ws.router_logits)?;
            kernels::moe::router_topk_into(ctx, &ws.router_logits, 1, c.num_experts, topk, c.norm_topk_prob, &ws.topk_idx, &ws.topk_w)?;
            // experts: gate|up for the 8 routed experts, fused SwiGLU, down with router weights folded in
            let routed = GemvSlots {
                slots: topk,
                expert_ids: Some(ws.topk_idx.address()),
                per_slot_activation: false,
                slot_scale: None,
            };
            w4a16::gemv_partial_into(ctx, &ws.ffn_normed, awq_view(&layer.gate_up, group), routed, sp.gate_up, &ws.p_gate_up)?;
            w4a16::partial_silu_mul_into(ctx, &ws.p_gate_up, topk, sp.gate_up, inter, &ws.h)?;
            let routed_down = GemvSlots {
                slots: topk,
                expert_ids: Some(ws.topk_idx.address()),
                per_slot_activation: true,
                slot_scale: Some(ws.topk_w.address()),
            };
            w4a16::gemv_partial_into(ctx, &ws.h, awq_view(&layer.down, group), routed_down, sp.down, &ws.p_down)?;
            w4a16::partial_sum_into(ctx, &ws.p_down, topk * sp.down, hd, &ws.moe_out)?;
            let next_norm = if li + 1 < w.layers.len() { &w.layers[li + 1].attn_norm } else { &w.final_norm };
            kernels::norm::residual_add_rms_bf16_into(ctx, &ws.x, &ws.moe_out, next_norm, &ws.normed, hd, 1, eps)?;
        }
        kernels::gemm::write_ex(
            ctx, DType::BF16, CublasTranspose::None, CublasTranspose::Transpose,
            1, c.vocab_size, hd, 1.0, &ws.normed, hd as i32, &w.lm_head, hd as i32, 0.0, &ws.logits, c.vocab_size as i32,
        )?;
        Ok(())
    }

    /// Decode one token at the next KV position. Returns `[1, vocab]` logits.
    pub fn decode(&mut self, backend: &CudaBackend, token: u32) -> Result<Tensor> {
        let ctx = backend.context();
        let position = self.kv.seq_len;
        if position >= self.kv.max_seq_len {
            return Err(Error::Other(format!(
                "qwen3moe decode: KV cache full ({} positions)",
                self.kv.max_seq_len
            )));
        }
        let kv_len = position + 1;
        let bucket = self.bucket_for(kv_len);
        self.decode.token.write_u32(token).map_err(cuda)?;
        self.decode.position.write_u32(position as u32).map_err(cuda)?;

        if self.use_graphs {
            if let Some(entry) = self.graphs.iter().find(|g| g.bucket_kv_len == bucket) {
                entry.graph.replay()?;
            } else {
                // Warm up eagerly (allocates cuBLAS plans), then capture.
                self.decode_body(ctx, bucket)?;
                backend.synchronize()?;
                backend.begin_capture_relaxed()?;
                let body = self.decode_body(ctx, bucket);
                let graph = backend.end_capture()?;
                body?;
                self.graphs.push(BucketGraph { bucket_kv_len: bucket, graph });
                // The eager run already produced this step's logits and KV
                // entry; the capture replayed the identical work in place.
            }
        } else {
            self.decode_body(ctx, bucket)?;
        }
        self.kv.seq_len = kv_len;
        self.decode
            .logits
            .as_tensor(Shape::new(vec![1, self.cfg.vocab_size]), DType::BF16)
            .map_err(cuda)
    }
}
