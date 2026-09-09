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
use super::trace::{LayerBuffers, LayerTrace};
use super::weights::{AwqLinear, Qwen3MoeWeights};
use crate::accelerator::cuda::{
    kernels, Context as CudaContext, CublasTranspose, DeviceAddress, DeviceBuffer, MappedBuffer,
    RuntimeBackend as CudaBackend,
};
use kernels::gemm::w4a16::{self, AwqWeightView, GemvSlots};
use kernels::gemm::w4a16_marlin as marlin;

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
const PREFILL_CHUNK: usize = 1024;

/// Fixed-address 16-bit KV cache (BF16 for decode, optional FP16 for prefill): `[n_kv_heads, max_seq_len, head_dim]` per
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
            gate_up: pick(
                2 * cfg.moe_intermediate_size,
                cfg.hidden_size,
                cfg.num_experts_per_tok,
            ),
            down: pick(
                cfg.hidden_size,
                cfg.moe_intermediate_size,
                cfg.num_experts_per_tok,
            ),
        }
    }
}

/// Stable-address buffers for one decode step.
struct DecodeWorkspace {
    x: DeviceBuffer,             // [hidden] residual stream
    normed: DeviceBuffer,        // [hidden]
    qkv: DeviceBuffer,           // [q | k | v]
    q_normed: DeviceBuffer,      // [q_dim]
    k_normed: DeviceBuffer,      // [kv_dim]
    q_rope: DeviceBuffer,        // [q_dim]
    k_rope: DeviceBuffer,        // [kv_dim]
    attn_out: DeviceBuffer,      // [q_dim]
    attn_partial: DeviceBuffer,  // [attention_splits, q_heads, 130] FP32
    attn_proj: DeviceBuffer,     // [hidden]
    ffn_normed: DeviceBuffer,    // [hidden]
    router_logits: DeviceBuffer, // [experts] bf16
    topk_idx: DeviceBuffer,      // [k] i32
    topk_w: DeviceBuffer,        // [k] f32
    p_qkv: DeviceBuffer,         // f32 partials
    p_o: DeviceBuffer,
    p_gate_up: DeviceBuffer,
    h: DeviceBuffer, // [k, inter] bf16
    p_down: DeviceBuffer,
    moe_out: DeviceBuffer,  // [hidden]
    logits: DeviceBuffer,   // [vocab] bf16
    token: MappedBuffer,    // u32
    position: MappedBuffer, // u32
    splits: DecodeSplits,
}

impl DecodeWorkspace {
    fn new(
        device: usize,
        cfg: &Qwen3MoeConfig,
        splits: DecodeSplits,
        logits_dtype: DType,
        attention_splits: usize,
    ) -> Result<Self> {
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
            attn_partial: alloc(device, attention_splits * cfg.n_heads * 130 * 4)?,
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
            logits: alloc(device, cfg.vocab_size * logits_dtype.size_in_bytes())?,
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
        let experts = if matches!(
            std::env::var("APXINF_QWEN3MOE_GROUPED").as_deref(),
            Ok("1") | Ok("marlin")
        ) {
            0
        } else {
            cfg.num_experts
        };
        (cfg.hidden_size * cfg.qkv_dim()
            + q_dim * cfg.hidden_size
            + experts * cfg.hidden_size * 2 * inter
            + experts * inter * cfg.hidden_size)
            * BF16
    }
}

/// Per-prefill activation buffers sized for `seq_len` tokens.
struct PrefillWorkspace {
    seq_len: usize,
    ids: DeviceBuffer,
    used_k: DeviceBuffer,
    x: DeviceBuffer,
    normed: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    q_normed: DeviceBuffer,
    k_normed: DeviceBuffer,
    q_rope: DeviceBuffer,
    k_rope: DeviceBuffer,
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
    grouped_tiles: DeviceBuffer,
    gate_up_out: DeviceBuffer,
    h: DeviceBuffer,
    y: DeviceBuffer,
    moe_out: DeviceBuffer,
    logits: DeviceBuffer,
    short_gate_up: DeviceBuffer,
    short_down: DeviceBuffer,
}

impl PrefillWorkspace {
    fn bytes(&self) -> usize {
        [
            self.ids.len(),
            self.used_k.len(),
            self.x.len(),
            self.normed.len(),
            self.q.len(),
            self.k.len(),
            self.v.len(),
            self.q_normed.len(),
            self.k_normed.len(),
            self.q_rope.len(),
            self.k_rope.len(),
            self.q_f16.len(),
            self.k_f16.len(),
            self.v_f16.len(),
            self.o_f16.len(),
            self.lse.len(),
            self.attn_out.len(),
            self.attn_proj.len(),
            self.ffn_normed.len(),
            self.router_logits.len(),
            self.topk_idx.len(),
            self.topk_w.len(),
            self.source_rows.len(),
            self.slot_rows.len(),
            self.gathered.len(),
            self.grouped_tiles.len(),
            self.gate_up_out.len(),
            self.h.len(),
            self.y.len(),
            self.moe_out.len(),
            self.logits.len(),
            self.short_gate_up.len(),
            self.short_down.len(),
        ]
        .into_iter()
        .sum()
    }

    fn new(
        device: usize,
        cfg: &Qwen3MoeConfig,
        seq_len: usize,
        short_threshold: usize,
        splits: DecodeSplits,
        logits_dtype: DType,
    ) -> Result<Self> {
        let s = seq_len;
        let hd = cfg.hidden_size;
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.kv_dim();
        let inter = cfg.moe_intermediate_size;
        let k = cfg.num_experts_per_tok;
        let routed = s * k;
        let short_slots = s.min(short_threshold) * k;
        Ok(Self {
            seq_len,
            ids: alloc(device, s * 4)?,
            used_k: alloc(device, 4)?,
            x: alloc(device, s * hd * BF16)?,
            normed: alloc(device, s * hd * BF16)?,
            q: alloc(device, s * q_dim * BF16)?,
            k: alloc(device, s * kv_dim * BF16)?,
            v: alloc(device, s * kv_dim * BF16)?,
            q_normed: alloc(device, s * q_dim * BF16)?,
            k_normed: alloc(device, s * kv_dim * BF16)?,
            q_rope: alloc(device, s * q_dim * BF16)?,
            k_rope: alloc(device, s * kv_dim * BF16)?,
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
            grouped_tiles: alloc(device, (routed.div_ceil(64) + cfg.num_experts) * 12)?,
            gate_up_out: alloc(device, routed * 2 * inter * BF16)?,
            h: alloc(device, routed * inter * BF16)?,
            y: alloc(device, routed * hd * BF16)?,
            moe_out: alloc(device, s * hd * BF16)?,
            logits: alloc(device, cfg.vocab_size * logits_dtype.size_in_bytes())?,
            short_gate_up: alloc(
                device,
                w4a16::partial_bytes(2 * inter, splits.gate_up, short_slots),
            )?,
            short_down: alloc(device, w4a16::partial_bytes(hd, splits.down, short_slots))?,
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

struct MarlinRuntime {
    use_f16: bool,
    fused_silu: bool,
    silu_table: Option<kernels::moe::SiluBf16Table>,
    layers: Vec<(marlin::Weights, marlin::Weights)>,
    workspace: marlin::Workspace,
}

struct CompensatedHead {
    input: DeviceBuffer,    // [2, hidden] BF16 high/low normalized residual
    partials: DeviceBuffer, // [2, vocab] FP32 products, shared by prefill/decode
}

impl CompensatedHead {
    fn bytes(cfg: &Qwen3MoeConfig) -> usize {
        2 * cfg.hidden_size * BF16 + 2 * cfg.vocab_size * 4
    }
}

pub struct Qwen3MoeRuntime {
    cfg: Qwen3MoeConfig,
    weights: Qwen3MoeWeights,
    kv: KvCache,
    prefill_kv: Option<KvCache>,
    chunk_size: usize,
    short_gemv_threshold: usize,
    gqa_mma: bool,
    gqa_vector: bool,
    gqa_balanced: bool,
    logits_dtype: DType,
    compensated_head: Option<CompensatedHead>,
    trace: Option<LayerTrace>,
    decode: DecodeWorkspace,
    dequant: Option<DequantWorkspace>,
    prefill: Option<PrefillWorkspace>,
    graphs: Vec<BucketGraph>,
    prefill_graphs: Vec<BucketGraph>,
    use_graphs: bool,
    marlin: Option<MarlinRuntime>,
    blocked: Option<Vec<[w4a16::BlockedWeights; 4]>>,
    dense_prefill: Option<Vec<[DeviceBuffer; 2]>>,
    rope_table: Option<kernels::rope::Table128>,
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
        let kv = KvCache::bytes(cfg, max_seq_len)
            * if std::env::var("APXINF_QWEN3MOE_CHUNKED").as_deref() == Ok("1") {
                2
            } else {
                1
            };
        let dequant = DequantWorkspace::bytes(cfg);
        // Prefill activations scale with the sequence; 1024 tokens is the ISL
        // this model is tuned for and the buffers are small next to the rest.
        let activations = 384 << 20;
        let dense_cache = if std::env::var("APXINF_QWEN3MOE_DENSE_CACHE").as_deref() == Ok("2") {
            cfg.n_layers
                * (cfg.hidden_size * cfg.qkv_dim() + cfg.n_heads * cfg.head_dim * cfg.hidden_size)
                * BF16
        } else {
            0
        };
        let head = if std::env::var("APXINF_QWEN3MOE_HEAD_COMPENSATED").as_deref() == Ok("1") {
            CompensatedHead::bytes(cfg)
        } else {
            0
        };
        kv + dequant + activations + dense_cache + head
    }

    pub fn new(
        backend: &CudaBackend,
        cfg: Qwen3MoeConfig,
        mut weights: Qwen3MoeWeights,
        max_seq_len: usize,
    ) -> Result<Self> {
        let short_gemv_threshold = match std::env::var("APXINF_QWEN3MOE_SHORT_GEMV") {
            Ok(value) => value
                .parse::<usize>()
                .ok()
                .filter(|&n| n <= 256)
                .ok_or_else(|| {
                    Error::Other(
                        "APXINF_QWEN3MOE_SHORT_GEMV must be an integer from 0 through 256".into(),
                    )
                })?,
            Err(std::env::VarError::NotPresent) => 0,
            Err(_) => {
                return Err(Error::Other(
                    "APXINF_QWEN3MOE_SHORT_GEMV is not valid Unicode".into(),
                ))
            }
        };
        let ctx = backend.context();
        let device = ctx.device_id();
        let kv = KvCache::new(device, &cfg, max_seq_len)?;
        let splits = DecodeSplits::choose(&cfg, ctx.caps().multiprocessor_count as usize);
        let compensated_head =
            if std::env::var("APXINF_QWEN3MOE_HEAD_COMPENSATED").as_deref() == Ok("1") {
                Some(CompensatedHead {
                    input: alloc(device, 2 * cfg.hidden_size * BF16)?,
                    partials: alloc(device, 2 * cfg.vocab_size * 4)?,
                })
            } else {
                None
            };
        let logits_dtype = if compensated_head.is_some()
            || std::env::var("APXINF_QWEN3MOE_F32_LOGITS").as_deref() == Ok("1")
        {
            DType::F32
        } else {
            DType::BF16
        };
        let gqa_mma = std::env::var("APXINF_QWEN3MOE_GQA_MMA").as_deref() == Ok("1");
        let gqa_balanced = std::env::var("APXINF_QWEN3MOE_GQA_BALANCED").as_deref() == Ok("1");
        let gqa_vector =
            gqa_balanced || std::env::var("APXINF_QWEN3MOE_GQA_VECTOR").as_deref() == Ok("1");
        if gqa_mma && gqa_vector {
            return Err(Error::Other(
                "GQA_MMA cannot be combined with GQA_VECTOR or GQA_BALANCED".into(),
            ));
        }
        let decode = DecodeWorkspace::new(
            device,
            &cfg,
            splits,
            logits_dtype,
            if gqa_mma { 32 } else { 16 },
        )?;
        let chunk_size = if std::env::var("APXINF_QWEN3MOE_CHUNKED").as_deref() == Ok("1") {
            PREFILL_CHUNK
        } else {
            0
        };
        let marlin = if std::env::var("APXINF_QWEN3MOE_GROUPED").as_deref() == Ok("marlin") {
            let use_f16 = std::env::var("APXINF_QWEN3MOE_MARLIN_F16").as_deref() == Ok("1");
            let fused_silu =
                use_f16 && std::env::var("APXINF_QWEN3MOE_MARLIN_SILU").as_deref() == Ok("1");
            let mut layers = Vec::with_capacity(weights.layers.len());
            for layer in &weights.layers {
                layers.push((
                    marlin::Weights::repack_with_silu(
                        ctx,
                        awq_view(&layer.gate_up, cfg.group_size()),
                        true,
                        fused_silu,
                    )?,
                    marlin::Weights::repack(ctx, awq_view(&layer.down, cfg.group_size()), true)?,
                ));
            }
            ctx.synchronize().map_err(cuda)?;
            eprintln!("[apxinf] qwen3moe: experimental Marlin expert copies use additional mapped memory and preserve FP16 scales");
            let silu_table = if use_f16
                && !fused_silu
                && std::env::var("APXINF_QWEN3MOE_SILU_LUT").as_deref() == Ok("1")
            {
                Some(kernels::moe::SiluBf16Table::new(ctx)?)
            } else {
                None
            };
            Some(MarlinRuntime {
                use_f16,
                fused_silu,
                silu_table,
                layers,
                workspace: marlin::Workspace::new(
                    ctx,
                    (if chunk_size > 0 {
                        max_seq_len.min(chunk_size)
                    } else {
                        max_seq_len
                    }) * cfg.num_experts_per_tok,
                    cfg.num_experts,
                    if std::env::var("APXINF_QWEN3MOE_MARLIN_M").as_deref() == Ok("64") {
                        64
                    } else {
                        32
                    },
                    use_f16,
                )?,
            })
        } else {
            None
        };
        let blocked_mode = std::env::var("APXINF_QWEN3MOE_BLOCKED").unwrap_or_default();
        let blocked = if matches!(blocked_mode.as_str(), "1" | "2") {
            let mut copies = Vec::with_capacity(weights.layers.len());
            let repack = |linear: &mut AwqLinear| -> Result<w4a16::BlockedWeights> {
                if blocked_mode == "2" {
                    let (repacked, backup) = w4a16::BlockedWeights::repack_reusing_storage(
                        ctx,
                        awq_view(linear, cfg.group_size()),
                    )?;
                    linear.qweight = backup;
                    Ok(repacked)
                } else {
                    w4a16::BlockedWeights::repack(ctx, awq_view(linear, cfg.group_size()), true)
                }
            };
            for layer in &mut weights.layers {
                copies.push([
                    repack(&mut layer.qkv)?,
                    repack(&mut layer.o)?,
                    repack(&mut layer.gate_up)?,
                    repack(&mut layer.down)?,
                ]);
            }
            ctx.synchronize().map_err(cuda)?;
            Some(copies)
        } else {
            None
        };
        // Optional persistent dense attention weights remove repeated dequant
        // traffic from prefill. Mode 2 reserves CUDA-pool space before packing
        // weights; mode 1 uses mapped storage without moving packed weights.
        let dense_cache_mode = std::env::var("APXINF_QWEN3MOE_DENSE_CACHE").unwrap_or_default();
        let dense_prefill = if matches!(dense_cache_mode.as_str(), "1" | "2") {
            let mapped = dense_cache_mode == "1";
            let allocate = |bytes| {
                if mapped {
                    DeviceBuffer::alloc_mapped(bytes, device).map_err(cuda)
                } else {
                    DeviceBuffer::alloc(bytes, device).map_err(cuda)
                }
            };
            let mut copies = Vec::with_capacity(weights.layers.len());
            let mut bytes = 0usize;
            for layer in &weights.layers {
                let qkv = allocate(layer.qkv.in_dim * layer.qkv.out_dim * BF16)?;
                let o = allocate(layer.o.in_dim * layer.o.out_dim * BF16)?;
                w4a16::dequant_bf16_into(ctx, awq_view(&layer.qkv, cfg.group_size()), &qkv)?;
                w4a16::dequant_bf16_into(ctx, awq_view(&layer.o, cfg.group_size()), &o)?;
                bytes += qkv.len() + o.len();
                copies.push([qkv, o]);
            }
            ctx.synchronize().map_err(cuda)?;
            eprintln!(
                "[apxinf] qwen3moe: persistent BF16 attention cache mapped={mapped} bytes={bytes}"
            );
            Some(copies)
        } else {
            None
        };
        let rope_table = if cfg.head_dim == 128
            && (chunk_size > 0
                || std::env::var("APXINF_QWEN3MOE_FUSED_QKV").as_deref() == Ok("1")
                || std::env::var("APXINF_QWEN3MOE_DECODE_QKV").as_deref() == Ok("1"))
        {
            Some(kernels::rope::Table128::new(
                ctx,
                max_seq_len,
                cfg.rope_theta,
            )?)
        } else {
            None
        };
        let prefill_kv = if chunk_size > 0 {
            if marlin.is_none() || rope_table.is_none() {
                return Err(Error::Other(
                    "chunked prefill requires Marlin and head dimension 128".into(),
                ));
            }
            Some(KvCache::new(device, &cfg, max_seq_len)?)
        } else {
            None
        };
        Ok(Self {
            cfg,
            weights,
            kv,
            prefill_kv,
            chunk_size,
            short_gemv_threshold,
            gqa_mma,
            gqa_vector,
            gqa_balanced,
            logits_dtype,
            trace: LayerTrace::from_env()?,
            compensated_head,
            decode,
            marlin,
            blocked,
            dense_prefill,
            rope_table,
            dequant: None,
            prefill: None,
            graphs: Vec::new(),
            prefill_graphs: Vec::new(),
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
        let experts = if self.marlin.is_some()
            || std::env::var("APXINF_QWEN3MOE_GROUPED").as_deref() == Ok("1")
        {
            0
        } else {
            c.num_experts
        };
        let dense_scratch = usize::from(self.dense_prefill.is_none());
        self.dequant = Some(DequantWorkspace {
            qkv: alloc(device, dense_scratch * c.hidden_size * c.qkv_dim() * BF16)?,
            o: alloc(device, dense_scratch * q_dim * c.hidden_size * BF16)?,
            gate_up: alloc(
                device,
                experts * c.hidden_size * 2 * c.moe_intermediate_size * BF16,
            )?,
            down: alloc(
                device,
                experts * c.moe_intermediate_size * c.hidden_size * BF16,
            )?,
        });
        Ok(())
    }

    fn ensure_prefill(&mut self, device: usize, seq_len: usize) -> Result<()> {
        let seq_len = if self.chunk_size > 0 {
            self.chunk_size.min(self.kv.max_seq_len)
        } else {
            seq_len
        };
        if self
            .prefill
            .as_ref()
            .is_some_and(|ws| ws.seq_len >= seq_len)
        {
            return Ok(());
        }
        self.prefill_graphs.clear();
        self.prefill = None;
        self.prefill = Some(PrefillWorkspace::new(
            device,
            &self.cfg,
            seq_len,
            self.short_gemv_threshold,
            self.decode.splits,
            self.logits_dtype,
        )?);
        let ws = self.prefill.as_ref().unwrap();
        eprintln!("[apxinf] qwen3moe: prefill capacity={} activation_workspace_bytes={} routing_workspace_bytes={} dequant_workspace_bytes={}",
            ws.seq_len, ws.bytes() + self.compensated_head.as_ref().map_or(0, |_| CompensatedHead::bytes(&self.cfg)), self.marlin.as_ref().map_or(0, |m| m.workspace.bytes()),
            self.dequant.as_ref().map_or(0, |d| d.qkv.len()+d.o.len()+d.gate_up.len()+d.down.len()));
        Ok(())
    }

    /// Run a full prompt from position 0 and return the logits of the last
    /// token as a `[1, vocab]` BF16 device tensor.
    pub fn prefill(&mut self, backend: &CudaBackend, token_ids: &[u32]) -> Result<Tensor> {
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
        let chunk = if self.chunk_size > 0 {
            self.chunk_size
        } else {
            s
        };
        for (index, tokens) in token_ids.chunks(chunk).enumerate() {
            self.prefill_chunk(backend, tokens, index * chunk)?;
        }
        self.kv.seq_len = s;
        self.prefill
            .as_ref()
            .unwrap()
            .logits
            .as_tensor(Shape::new(vec![1, self.cfg.vocab_size]), self.logits_dtype)
            .map_err(cuda)
    }

    fn prefill_chunk(
        &mut self,
        backend: &CudaBackend,
        token_ids: &[u32],
        offset: usize,
    ) -> Result<()> {
        let ctx = backend.context();
        let device = ctx.device_id();
        let s = token_ids.len();
        self.ensure_dequant(device)?;
        self.ensure_prefill(device, s)?;
        let ws = self.prefill.as_ref().unwrap();
        let id_bytes: Vec<u8> = token_ids.iter().flat_map(|t| t.to_ne_bytes()).collect();
        ws.ids.copy_from_host(&id_bytes).map_err(cuda)?;
        ws.used_k
            .copy_from_host(&((offset + s) as i32).to_ne_bytes())
            .map_err(cuda)?;
        let short_gemv = s <= self.short_gemv_threshold;
        if short_gemv {
            let sources: Vec<u8> = (0..s * self.cfg.num_experts_per_tok)
                .flat_map(|slot| ((slot / self.cfg.num_experts_per_tok) as i32).to_ne_bytes())
                .collect();
            ws.source_rows.copy_from_host(&sources).map_err(cuda)?;
        }
        let device_routing = short_gemv
            || self.marlin.as_ref().is_some_and(|m| {
                self.chunk_size > 0
                    || m.use_f16
                    || std::env::var("APXINF_QWEN3MOE_DEVICE_ROUTING").as_deref() == Ok("1")
            });
        if device_routing {
            let identity: Vec<u8> = (0..s * self.cfg.num_experts_per_tok)
                .flat_map(|i| (i as i32).to_ne_bytes())
                .collect();
            ws.slot_rows.copy_from_host(&identity).map_err(cuda)?;
        }
        let capture = self.trace.is_none()
            && device_routing
            && (self.chunk_size > 0
                || std::env::var("APXINF_QWEN3MOE_PREFILL_GRAPH").as_deref() == Ok("1"));
        if capture {
            if let Some(entry) = self.prefill_graphs.iter().find(|g| g.bucket_kv_len == s) {
                entry.graph.replay()?;
            } else {
                self.prefill_body(ctx, s, offset)?;
                backend.synchronize()?;
                backend.begin_capture_relaxed()?;
                let body = self.prefill_body(ctx, s, offset);
                let graph = backend.end_capture()?;
                body?;
                if self.prefill_graphs.len() == 4 {
                    let oldest = self
                        .prefill_graphs
                        .iter()
                        .position(|g| g.bucket_kv_len != PREFILL_CHUNK)
                        .unwrap_or(0);
                    self.prefill_graphs.remove(oldest);
                }
                self.prefill_graphs.push(BucketGraph {
                    bucket_kv_len: s,
                    graph,
                });
            }
        } else {
            self.prefill_body(ctx, s, offset)?;
        }
        Ok(())
    }

    /// Fixed-address device-routed prefill body. Input upload and allocations
    /// happen before entry, allowing graph capture after cuBLAS warmup.
    fn prefill_body(&self, ctx: &CudaContext, s: usize, offset: usize) -> Result<()> {
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
        let residual_norm = if w.rms_weights_f16 {
            kernels::norm::residual_add_rms_f16_weight_into
        } else {
            kernels::norm::residual_add_rms_bf16_into
        };
        let routed_norm = if w.rms_weights_f16 {
            kernels::moe::routed_residual_rms_f16_weight_into
        } else {
            kernels::moe::routed_residual_rms_into
        };
        let partial_rows_norm = if w.rms_weights_f16 {
            w4a16::partial_residual_rms_rows_f16_weight_into
        } else {
            w4a16::partial_residual_rms_rows_into
        };

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
        self.layer_rms(ctx, &x, &w.layers[0].attn_norm, &normed, s)?;

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
        let marlin_f16 = self.marlin.as_ref().is_some_and(|m| m.use_f16);
        let short_gemv = s <= self.short_gemv_threshold;
        let device_routing = self.marlin.is_some()
            && (self.chunk_size > 0
                || marlin_f16
                || std::env::var("APXINF_QWEN3MOE_DEVICE_ROUTING").as_deref() == Ok("1"));
        let slot_rows_dev = view(&ws.slot_rows, 0, s * topk * 4)?;
        for (li, layer) in w.layers.iter().enumerate() {
            // ── attention projections: dequant packed [q|k|v] then 3 strided GEMMs
            let qkv_weight = if let Some(cache) = &self.dense_prefill {
                &cache[li][0]
            } else {
                w4a16::dequant_bf16_into(ctx, awq_view(&layer.qkv, group), &dq.qkv)?;
                &dq.qkv
            };
            let ldb = c.qkv_dim() as i32;
            let mut col = 0usize;
            for (out, n) in [(&q, q_dim), (&k, kv_dim), (&v, kv_dim)] {
                let b = view(qkv_weight, col * BF16, qkv_weight.len() - col * BF16)?;
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
            if let Some(prefill_kv) = &self.prefill_kv {
                kernels::rope::qk_norm_append_cached_f16_into(
                    ctx,
                    &q,
                    &k,
                    &v,
                    &layer.q_norm,
                    &layer.k_norm,
                    &q_f16,
                    &prefill_kv.k[li],
                    &prefill_kv.v[li],
                    &self.kv.k[li],
                    &self.kv.v[li],
                    s,
                    c.n_heads,
                    c.n_kv_heads,
                    self.kv.max_seq_len,
                    ws.used_k.address(),
                    eps,
                    self.rope_table.as_ref().unwrap(),
                )?;
            } else if let Some(table) = self
                .rope_table
                .as_ref()
                .filter(|_| std::env::var("APXINF_QWEN3MOE_FUSED_QKV").as_deref() == Ok("1"))
            {
                kernels::rope::qk_norm_append_f16_into(
                    ctx,
                    &q,
                    &k,
                    &v,
                    &layer.q_norm,
                    &layer.k_norm,
                    &q_f16,
                    &k_f16,
                    &v_f16,
                    &self.kv.k[li],
                    &self.kv.v[li],
                    s,
                    c.n_heads,
                    c.n_kv_heads,
                    self.kv.max_seq_len,
                    0,
                    eps,
                    table,
                )?;
            } else {
                // ── per-head QK-norm, RoPE (positions 0..s)
                kernels::norm::rms_into(
                    ctx,
                    DType::BF16,
                    &q,
                    &layer.q_norm,
                    &q_normed,
                    c.head_dim,
                    s * c.n_heads,
                    eps,
                )?;
                kernels::norm::rms_into(
                    ctx,
                    DType::BF16,
                    &k,
                    &layer.k_norm,
                    &k_normed,
                    c.head_dim,
                    s * c.n_kv_heads,
                    eps,
                )?;
                let q_rope_buf = view(&ws.q_rope, 0, s * q_dim * BF16)?;
                let k_rope_buf = view(&ws.k_rope, 0, s * kv_dim * BF16)?;
                kernels::rope::apply_batched_bf16_into(
                    ctx,
                    &q_normed,
                    &q_rope_buf,
                    c.n_heads,
                    c.head_dim,
                    s,
                    c.rope_theta,
                    0,
                )?;
                kernels::rope::apply_batched_bf16_into(
                    ctx,
                    &k_normed,
                    &k_rope_buf,
                    c.n_kv_heads,
                    c.head_dim,
                    s,
                    c.rope_theta,
                    0,
                )?;
                let k_rope = k_rope_buf
                    .as_tensor(Shape::new(vec![s, c.n_kv_heads, c.head_dim]), DType::BF16)
                    .map_err(cuda)?;
                // ── KV cache write (positions 0..s)
                let v_t = v
                    .as_tensor(Shape::new(vec![s, c.n_kv_heads, c.head_dim]), DType::BF16)
                    .map_err(cuda)?;
                kernels::cache::append(
                    ctx,
                    &self.kv.k[li],
                    &k_rope,
                    c.n_kv_heads,
                    c.head_dim,
                    self.kv.max_seq_len,
                    0,
                    s,
                )?;
                kernels::cache::append(
                    ctx,
                    &self.kv.v[li],
                    &v_t,
                    c.n_kv_heads,
                    c.head_dim,
                    self.kv.max_seq_len,
                    0,
                    s,
                )?;
                // ── causal attention (FA2, fp16)
                kernels::elementwise::convert_bf16_to_f16_into(
                    ctx,
                    &q_rope_buf,
                    &q_f16,
                    s * q_dim,
                )?;
                kernels::elementwise::convert_bf16_to_f16_into(
                    ctx,
                    &k_rope_buf,
                    &k_f16,
                    s * kv_dim,
                )?;
                kernels::elementwise::convert_bf16_to_f16_into(ctx, &v, &v_f16, s * kv_dim)?;
            }
            if let Some(prefill_kv) = &self.prefill_kv {
                kernels::attention::causal_prefill_cached_f16_into(
                    ctx,
                    &q_f16,
                    &prefill_kv.k[li],
                    &prefill_kv.v[li],
                    &o_f16,
                    &lse,
                    s,
                    self.kv.max_seq_len,
                    ws.used_k.address(),
                    c.n_heads,
                    c.n_kv_heads,
                    scale,
                )?;
            } else {
                kernels::attention::causal_prefill_f16_into(
                    ctx,
                    &q_f16,
                    &k_f16,
                    &v_f16,
                    &o_f16,
                    &lse,
                    s,
                    s,
                    c.n_heads,
                    c.n_kv_heads,
                    scale,
                )?;
            }
            kernels::elementwise::convert_f16_to_bf16_into(ctx, &o_f16, &attn_out, s * q_dim)?;
            // ── output projection + residual + FFN norm
            let o_weight = if let Some(cache) = &self.dense_prefill {
                &cache[li][1]
            } else {
                w4a16::dequant_bf16_into(ctx, awq_view(&layer.o, group), &dq.o)?;
                &dq.o
            };
            kernels::gemm::write(
                ctx,
                DType::BF16,
                s,
                hd,
                q_dim,
                1.0,
                &attn_out,
                o_weight,
                0.0,
                &attn_proj,
            )?;
            residual_norm(
                ctx,
                &x,
                &attn_proj,
                &layer.ffn_norm,
                &ffn_normed,
                hd,
                s,
                eps,
            )?;
            // ── router
            kernels::gemm::write(
                ctx,
                DType::BF16,
                s,
                experts,
                hd,
                1.0,
                &ffn_normed,
                &layer.router,
                0.0,
                &router_logits,
            )?;
            kernels::moe::router_topk_into(
                ctx,
                &router_logits,
                s,
                experts,
                topk,
                c.norm_topk_prob,
                &topk_idx,
                &topk_w,
            )?;
            if short_gemv {
                // Reuse the decode expert arithmetic for small token batches.
                // Slots are token-major; no padding or host expert routing is needed.
                let slots = s * topk;
                let gathered = view(&ws.gathered, 0, slots * hd * BF16)?;
                let sources = view(&ws.source_rows, 0, slots * 4)?;
                kernels::moe::gather_rows_into(
                    ctx,
                    &ffn_normed,
                    s,
                    &sources,
                    slots,
                    hd,
                    &gathered,
                )?;
                let routed = GemvSlots {
                    slots,
                    expert_ids: Some(topk_idx.address()),
                    per_slot_activation: true,
                    slot_scale: None,
                };
                w4a16::gemv_partial_repacked_into(
                    ctx,
                    &gathered,
                    awq_view(&layer.gate_up, group),
                    self.blocked.as_ref().map(|layers| &layers[li][2]),
                    routed,
                    self.decode.splits.gate_up,
                    &ws.short_gate_up,
                )?;
                w4a16::partial_silu_mul_into(
                    ctx,
                    &ws.short_gate_up,
                    slots,
                    self.decode.splits.gate_up,
                    inter,
                    &ws.h,
                )?;
                let routed_down = GemvSlots {
                    slot_scale: Some(topk_w.address()),
                    ..routed
                };
                w4a16::gemv_partial_repacked_into(
                    ctx,
                    &ws.h,
                    awq_view(&layer.down, group),
                    self.blocked.as_ref().map(|layers| &layers[li][3]),
                    routed_down,
                    self.decode.splits.down,
                    &ws.short_down,
                )?;
            } else if device_routing {
                let marlin = self
                    .marlin
                    .as_ref()
                    .expect("device routing requires Marlin");
                let schedule = marlin
                    .workspace
                    .permute(ctx, &topk_idx, s * topk, experts)?;
                if marlin.use_f16 {
                    let f16_input = view(&ws.gathered, 0, s * hd * 2)?;
                    kernels::elementwise::convert_bf16_to_f16_into(
                        ctx,
                        &ffn_normed,
                        &f16_input,
                        s * hd,
                    )?;
                    if marlin.fused_silu {
                        marlin::indexed_into(
                            ctx,
                            &f16_input,
                            &marlin.layers[li].0,
                            &schedule,
                            &ws.h,
                            topk,
                        )?;
                    } else {
                        marlin::indexed_into(
                            ctx,
                            &f16_input,
                            &marlin.layers[li].0,
                            &schedule,
                            &ws.gate_up_out,
                            topk,
                        )?;
                        if let Some(table) = &marlin.silu_table {
                            kernels::moe::silu_mul_rows_f16_lut_into(
                                ctx,
                                &ws.gate_up_out,
                                &ws.h,
                                table,
                                s * topk,
                                inter,
                            )?;
                        } else {
                            kernels::moe::silu_mul_rows_f16_rounded_into(
                                ctx,
                                &ws.gate_up_out,
                                &ws.h,
                                s * topk,
                                inter,
                            )?;
                        }
                    }
                } else {
                    marlin::indexed_into(
                        ctx,
                        &ffn_normed,
                        &marlin.layers[li].0,
                        &schedule,
                        &ws.gate_up_out,
                        topk,
                    )?;
                    kernels::moe::silu_mul_rows_into(ctx, &ws.gate_up_out, s * topk, inter, &ws.h)?;
                }
                marlin::grouped_into(ctx, &ws.h, &marlin.layers[li].1, &schedule, &ws.y)?;
            } else {
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
                        return Err(Error::Other(format!(
                            "qwen3moe: router produced expert {e}"
                        )));
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
                let to_bytes =
                    |v: &[i32]| v.iter().flat_map(|x| x.to_ne_bytes()).collect::<Vec<u8>>();
                let source_rows_dev = view(&ws.source_rows, 0, s * topk * 4)?;
                source_rows_dev
                    .copy_from_host(&to_bytes(&source_rows))
                    .map_err(cuda)?;
                slot_rows_dev
                    .copy_from_host(&to_bytes(&slot_rows))
                    .map_err(cuda)?;
                let gathered = view(&ws.gathered, 0, s * topk * hd * BF16)?;
                kernels::moe::gather_rows_into(
                    ctx,
                    &ffn_normed,
                    s,
                    &source_rows_dev,
                    s * topk,
                    hd,
                    &gathered,
                )?;
                // Grouped W4A16 consumes packed weights without a global BF16 copy.
                // Keep the original path selectable while accepting this candidate.
                if let Some(marlin) = &self.marlin {
                    let schedule = marlin.workspace.schedule(&offsets)?;
                    marlin::grouped_into(
                        ctx,
                        &gathered,
                        &marlin.layers[li].0,
                        &schedule,
                        &ws.gate_up_out,
                    )?;
                    kernels::moe::silu_mul_rows_into(ctx, &ws.gate_up_out, s * topk, inter, &ws.h)?;
                    marlin::grouped_into(ctx, &ws.h, &marlin.layers[li].1, &schedule, &ws.y)?;
                } else if std::env::var("APXINF_QWEN3MOE_GROUPED").as_deref() == Ok("1") {
                    let schedule = w4a16::GroupedRows::new(ctx, &offsets, &ws.grouped_tiles)?;
                    w4a16::grouped_bf16_into(
                        ctx,
                        &gathered,
                        awq_view(&layer.gate_up, group),
                        &schedule,
                        &ws.gate_up_out,
                    )?;
                    kernels::moe::silu_mul_rows_into(ctx, &ws.gate_up_out, s * topk, inter, &ws.h)?;
                    w4a16::grouped_bf16_into(
                        ctx,
                        &ws.h,
                        awq_view(&layer.down, group),
                        &schedule,
                        &ws.y,
                    )?;
                } else {
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
                        let gu = view(
                            &ws.gate_up_out,
                            off * 2 * inter * BF16,
                            rows * 2 * inter * BF16,
                        )?;
                        kernels::gemm::write(
                            ctx,
                            DType::BF16,
                            rows,
                            2 * inter,
                            hd,
                            1.0,
                            &a,
                            &wg,
                            0.0,
                            &gu,
                        )?;
                        let hbuf = view(&ws.h, off * inter * BF16, rows * inter * BF16)?;
                        kernels::moe::silu_mul_rows_into(ctx, &gu, rows, inter, &hbuf)?;
                        let wd = view(&dq.down, e * down_expert, down_expert)?;
                        let y = view(&ws.y, off * hd * BF16, rows * hd * BF16)?;
                        kernels::gemm::write(
                            ctx,
                            DType::BF16,
                            rows,
                            hd,
                            inter,
                            1.0,
                            &hbuf,
                            &wd,
                            0.0,
                            &y,
                        )?;
                    }
                }
            }
            let y_all = view(&ws.y, 0, s * topk * hd * BF16)?;
            // ── residual + next norm (final norm after the last layer)
            let next_norm = if li + 1 < w.layers.len() {
                &w.layers[li + 1].attn_norm
            } else {
                &w.final_norm
            };
            if short_gemv {
                partial_rows_norm(
                    ctx,
                    &x,
                    &ws.short_down,
                    next_norm,
                    &normed,
                    hd,
                    s,
                    topk * self.decode.splits.down,
                    eps,
                )?;
            } else if marlin_f16
                || std::env::var("APXINF_QWEN3MOE_PREFILL_COMBINE").as_deref() == Ok("1")
            {
                routed_norm(
                    ctx,
                    &x,
                    &y_all,
                    &slot_rows_dev,
                    &topk_w,
                    next_norm,
                    &normed,
                    hd,
                    s,
                    topk,
                    eps,
                    marlin_f16,
                )?;
            } else {
                kernels::moe::weighted_gather_sum_into(
                    ctx,
                    &y_all,
                    &slot_rows_dev,
                    &topk_w,
                    s,
                    topk,
                    hd,
                    &moe_out,
                )?;
                residual_norm(ctx, &x, &moe_out, next_norm, &normed, hd, s, eps)?;
            }
            if let Some(trace) = &self.trace {
                trace.write(
                    ctx,
                    "prefill",
                    offset + s - 1,
                    li,
                    s - 1,
                    hd,
                    experts,
                    topk,
                    LayerBuffers {
                        residual: &x,
                        ffn_normed: &ffn_normed,
                        router_logits: &router_logits,
                        topk_idx: &topk_idx,
                        topk_weight: &topk_w,
                    },
                )?;
            }
        }
        // Logits for the last token only: [1, vocab] = normed[s-1] · lm_head^T.
        let last = view(&normed, (s - 1) * hd * BF16, hd * BF16)?;
        let residual = view(&x, (s - 1) * hd * BF16, hd * BF16)?;
        self.output_logits(ctx, &last, &residual, &ws.logits)?;
        Ok(())
    }

    fn layer_rms(
        &self,
        ctx: &CudaContext,
        input: &DeviceBuffer,
        weight: &DeviceBuffer,
        output: &DeviceBuffer,
        rows: usize,
    ) -> Result<()> {
        if self.weights.rms_weights_f16 {
            kernels::norm::rms_f16_weight_into(
                ctx,
                input,
                weight,
                output,
                self.cfg.hidden_size,
                rows,
                self.cfg.rms_norm_eps,
            )
        } else {
            kernels::norm::rms_into(
                ctx,
                DType::BF16,
                input,
                weight,
                output,
                self.cfg.hidden_size,
                rows,
                self.cfg.rms_norm_eps,
            )
        }
    }

    fn output_logits(
        &self,
        ctx: &CudaContext,
        input: &DeviceBuffer,
        residual: &DeviceBuffer,
        output: &DeviceBuffer,
    ) -> Result<()> {
        let c = &self.cfg;
        if let Some(head) = &self.compensated_head {
            kernels::norm::rms_bf16_split_into(
                ctx,
                residual,
                &self.weights.final_norm,
                if self.weights.rms_weights_f16 {
                    DType::F16
                } else {
                    DType::BF16
                },
                &head.input,
                c.hidden_size,
                1,
                c.rms_norm_eps,
            )?;
            kernels::gemm::bf16_f32_output_into(
                ctx,
                &head.input,
                &self.weights.lm_head,
                &head.partials,
                2,
                c.vocab_size,
                c.hidden_size,
            )?;
            let high = view(&head.partials, 0, c.vocab_size * 4)?;
            let low = view(&head.partials, c.vocab_size * 4, c.vocab_size * 4)?;
            kernels::elementwise::add_into(ctx, DType::F32, &high, &low, output, c.vocab_size)
        } else if self.logits_dtype == DType::F32 {
            kernels::gemm::bf16_f32_output_into(
                ctx,
                input,
                &self.weights.lm_head,
                output,
                1,
                c.vocab_size,
                c.hidden_size,
            )
        } else {
            kernels::gemm::write_ex(
                ctx,
                DType::BF16,
                CublasTranspose::None,
                CublasTranspose::Transpose,
                1,
                c.vocab_size,
                c.hidden_size,
                1.0,
                input,
                c.hidden_size as i32,
                &self.weights.lm_head,
                c.hidden_size as i32,
                0.0,
                output,
                c.vocab_size as i32,
            )
        }
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
        let residual_norm = if w.rms_weights_f16 {
            kernels::norm::residual_add_rms_f16_weight_into
        } else {
            kernels::norm::residual_add_rms_bf16_into
        };
        let partial_norm = if w.rms_weights_f16 {
            w4a16::partial_residual_rms_f16_weight_into
        } else {
            w4a16::partial_residual_rms_into
        };
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

        kernels::embedding::lookup_into(
            ctx,
            DType::BF16,
            &w.embed_tokens,
            ws.token.address(),
            &ws.x,
            hd,
            1,
        )?;
        self.layer_rms(ctx, &ws.x, &w.layers[0].attn_norm, &ws.normed, 1)?;

        for (li, layer) in w.layers.iter().enumerate() {
            // qkv = normed · W_qkv
            w4a16::gemv_partial_repacked_into(
                ctx,
                &ws.normed,
                awq_view(&layer.qkv, group),
                self.blocked.as_ref().map(|layers| &layers[li][0]),
                GemvSlots::dense(),
                sp.qkv,
                &ws.p_qkv,
            )?;
            if let Some(table) = self
                .rope_table
                .as_ref()
                .filter(|_| std::env::var("APXINF_QWEN3MOE_DECODE_QKV").as_deref() == Ok("1"))
            {
                kernels::rope::qkv_partial_cache_bf16_into(
                    ctx,
                    &ws.p_qkv,
                    &layer.q_norm,
                    &layer.k_norm,
                    &ws.q_rope,
                    &self.kv.k[li],
                    &self.kv.v[li],
                    position,
                    c.n_heads,
                    c.n_kv_heads,
                    self.kv.max_seq_len,
                    sp.qkv,
                    eps,
                    table,
                )?;
            } else {
                w4a16::partial_sum_into(ctx, &ws.p_qkv, sp.qkv, c.qkv_dim(), &ws.qkv)?;
                let q = view(&ws.qkv, 0, q_dim * BF16)?;
                let k = view(&ws.qkv, q_dim * BF16, kv_dim * BF16)?;
                let v = view(&ws.qkv, (q_dim + kv_dim) * BF16, kv_dim * BF16)?;
                kernels::norm::rms_into(
                    ctx,
                    DType::BF16,
                    &q,
                    &layer.q_norm,
                    &ws.q_normed,
                    c.head_dim,
                    c.n_heads,
                    eps,
                )?;
                kernels::norm::rms_into(
                    ctx,
                    DType::BF16,
                    &k,
                    &layer.k_norm,
                    &ws.k_normed,
                    c.head_dim,
                    c.n_kv_heads,
                    eps,
                )?;
                kernels::rope::apply_into(
                    ctx,
                    DType::BF16,
                    &ws.q_normed,
                    &ws.q_rope,
                    c.head_dim,
                    c.n_heads,
                    c.rope_theta,
                    position,
                )?;
                kernels::rope::apply_into(
                    ctx,
                    DType::BF16,
                    &ws.k_normed,
                    &ws.k_rope,
                    c.head_dim,
                    c.n_kv_heads,
                    c.rope_theta,
                    position,
                )?;
                kernels::cache::append_at(
                    ctx,
                    DType::BF16,
                    &self.kv.k[li],
                    &ws.k_rope,
                    c.n_kv_heads,
                    c.head_dim,
                    self.kv.max_seq_len,
                    position,
                )?;
                kernels::cache::append_at(
                    ctx,
                    DType::BF16,
                    &self.kv.v[li],
                    &v,
                    c.n_kv_heads,
                    c.head_dim,
                    self.kv.max_seq_len,
                    position,
                )?;
            }
            if c.head_dim == 128
                && c.n_heads == c.n_kv_heads * 8
                && (self.gqa_mma
                    || self.gqa_vector
                    || std::env::var("APXINF_QWEN3MOE_GQA").as_deref() == Ok("1"))
            {
                if self.gqa_vector {
                    let attention = if self.gqa_balanced {
                        kernels::attention::gqa_balanced_bf16_into
                    } else {
                        kernels::attention::gqa_vector_bf16_into
                    };
                    attention(
                        ctx,
                        &ws.q_rope,
                        &self.kv.k[li],
                        &self.kv.v[li],
                        &ws.attn_partial,
                        &ws.attn_out,
                        c.n_kv_heads,
                        self.kv.max_seq_len,
                        16,
                        scale,
                        position,
                    )?;
                } else if self.gqa_mma {
                    // Fixed per capture bucket. Long buckets increase sequence
                    // splits while short buckets avoid their combine overhead.
                    kernels::attention::gqa_mma_bf16_into(
                        ctx,
                        &ws.q_rope,
                        &self.kv.k[li],
                        &self.kv.v[li],
                        &ws.attn_partial,
                        &ws.attn_out,
                        c.n_kv_heads,
                        self.kv.max_seq_len,
                        if bucket_kv_len > 4096 { 32 } else { 16 },
                        if bucket_kv_len <= 512 { 32 } else { 64 },
                        scale,
                        position,
                    )?;
                } else {
                    kernels::attention::gqa_bf16_into(
                        ctx,
                        &ws.q_rope,
                        &self.kv.k[li],
                        &self.kv.v[li],
                        &ws.attn_partial,
                        &ws.attn_out,
                        c.n_kv_heads,
                        self.kv.max_seq_len,
                        16,
                        scale,
                        position,
                    )?;
                }
            } else {
                kernels::attention::flash_bf16_into(
                    ctx,
                    &ws.q_rope,
                    &self.kv.k[li],
                    &self.kv.v[li],
                    &ws.attn_out,
                    c.n_heads,
                    c.n_kv_heads,
                    c.head_dim,
                    bucket_kv_len,
                    self.kv.max_seq_len,
                    scale,
                    position,
                )?;
            }
            w4a16::gemv_partial_repacked_into(
                ctx,
                &ws.attn_out,
                awq_view(&layer.o, group),
                self.blocked.as_ref().map(|layers| &layers[li][1]),
                GemvSlots::dense(),
                sp.o,
                &ws.p_o,
            )?;
            if std::env::var("APXINF_QWEN3MOE_DECODE_RESIDUAL").as_deref() == Ok("1") {
                partial_norm(
                    ctx,
                    &ws.x,
                    &ws.p_o,
                    &layer.ffn_norm,
                    &ws.ffn_normed,
                    hd,
                    sp.o,
                    eps,
                )?;
            } else {
                w4a16::partial_sum_into(ctx, &ws.p_o, sp.o, hd, &ws.attn_proj)?;
                residual_norm(
                    ctx,
                    &ws.x,
                    &ws.attn_proj,
                    &layer.ffn_norm,
                    &ws.ffn_normed,
                    hd,
                    1,
                    eps,
                )?;
            }
            // router (cuBLAS M=1) + top-k on device
            kernels::gemm::write(
                ctx,
                DType::BF16,
                1,
                c.num_experts,
                hd,
                1.0,
                &ws.ffn_normed,
                &layer.router,
                0.0,
                &ws.router_logits,
            )?;
            kernels::moe::router_topk_into(
                ctx,
                &ws.router_logits,
                1,
                c.num_experts,
                topk,
                c.norm_topk_prob,
                &ws.topk_idx,
                &ws.topk_w,
            )?;
            // experts: gate|up for the 8 routed experts, fused SwiGLU, down with router weights folded in
            let routed = GemvSlots {
                slots: topk,
                expert_ids: Some(ws.topk_idx.address()),
                per_slot_activation: false,
                slot_scale: None,
            };
            w4a16::gemv_partial_repacked_into(
                ctx,
                &ws.ffn_normed,
                awq_view(&layer.gate_up, group),
                self.blocked.as_ref().map(|layers| &layers[li][2]),
                routed,
                sp.gate_up,
                &ws.p_gate_up,
            )?;
            w4a16::partial_silu_mul_into(ctx, &ws.p_gate_up, topk, sp.gate_up, inter, &ws.h)?;
            let routed_down = GemvSlots {
                slots: topk,
                expert_ids: Some(ws.topk_idx.address()),
                per_slot_activation: true,
                slot_scale: Some(ws.topk_w.address()),
            };
            w4a16::gemv_partial_repacked_into(
                ctx,
                &ws.h,
                awq_view(&layer.down, group),
                self.blocked.as_ref().map(|layers| &layers[li][3]),
                routed_down,
                sp.down,
                &ws.p_down,
            )?;
            let next_norm = if li + 1 < w.layers.len() {
                &w.layers[li + 1].attn_norm
            } else {
                &w.final_norm
            };
            if std::env::var("APXINF_QWEN3MOE_DECODE_RESIDUAL").as_deref() == Ok("1") {
                partial_norm(
                    ctx,
                    &ws.x,
                    &ws.p_down,
                    next_norm,
                    &ws.normed,
                    hd,
                    topk * sp.down,
                    eps,
                )?;
            } else {
                w4a16::partial_sum_into(ctx, &ws.p_down, topk * sp.down, hd, &ws.moe_out)?;
                residual_norm(ctx, &ws.x, &ws.moe_out, next_norm, &ws.normed, hd, 1, eps)?;
            }
            if let Some(trace) = &self.trace {
                trace.write(
                    ctx,
                    "decode",
                    self.kv.seq_len,
                    li,
                    0,
                    hd,
                    c.num_experts,
                    topk,
                    LayerBuffers {
                        residual: &ws.x,
                        ffn_normed: &ws.ffn_normed,
                        router_logits: &ws.router_logits,
                        topk_idx: &ws.topk_idx,
                        topk_weight: &ws.topk_w,
                    },
                )?;
            }
        }
        self.output_logits(ctx, &ws.normed, &ws.x, &ws.logits)?;
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
        self.decode
            .position
            .write_u32(position as u32)
            .map_err(cuda)?;

        if self.use_graphs && self.trace.is_none() {
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
                self.graphs.push(BucketGraph {
                    bucket_kv_len: bucket,
                    graph,
                });
                // The eager run already produced this step's logits and KV
                // entry; the capture replayed the identical work in place.
            }
        } else {
            self.decode_body(ctx, bucket)?;
        }
        self.kv.seq_len = kv_len;
        self.decode
            .logits
            .as_tensor(Shape::new(vec![1, self.cfg.vocab_size]), self.logits_dtype)
            .map_err(cuda)
    }
}
