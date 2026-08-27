//! GPU-resident Qwen3.5 forward path.
//!
//! Mirrors the CPU reference math in [`super::general`] but keeps activations,
//! KV caches, and the linear-attention recurrent states on the device:
//!
//! - when `APXINF_MARLIN=1`, eligible packed INT4 GEMMs use the same persistent
//!   Marlin device layout for both decode and prefill; unsupported weights keep
//!   the established raw fused/dequantized paths;
//! - linear-attention layers run causal conv + SiLU and the gated delta-rule
//!   recurrence as one kernel launch per layer;
//! - full-attention layers write q_norm/k_norm + partial-RoPE directly into
//!   per-layer K/V caches and use flash attention (decode or split-warp
//!   prefill);
//! - prompts longer than [`CHUNK`] are processed in chunks. Chunking is
//!   exact: the linear layers are sequential recurrences and the full
//!   attention is causal, so chunk boundaries change nothing but buffer sizes.

use std::collections::HashMap;
use std::sync::Arc;

use apxinf_core::{Backend, DType, Error, Graph, Result, Shape, Tensor};
use apxinf_cuda::buffer::{CudaBuffer, CudaDeviceAddress, HostMappedBuffer};
use apxinf_cuda::kernels;
use apxinf_cuda::{CudaBackend, CudaContext, CudaEvent};
use apxinf_loader::compressed_tensors::{
    W4_MARLIN_AWQ_U4_G32_V1_SUFFIX, W4_REPACKED_N64_K16_V1_SUFFIX,
    W4_TRANSFORM_CACHE_SUFFIX,
};

use super::{LayerKind, Qwen35Config};

/// Prefill chunk size: bounds all activation workspace buffers.
const CHUNK: usize = 512;
/// BF16 K/V cache capacity for every full-attention layer. This covers the
/// leaderboard's 65,536-token prompt tier plus its required 128-token output.
/// The service admits prompt plus completion against this exact row count.
pub const MAX_SEQ_LEN: usize = 65_664;
/// Bound the f32 V tile used by long-context attention output GEMMs.
const V_TILE_ROWS: usize = 16_384;

/// One startup allocation owns every mutable Qwen runtime buffer. Views keep
/// the allocation alive and are 256-byte aligned for CUDA/Marlin vector loads.
struct PersistentArena {
    storage: CudaBuffer,
    offset: usize,
}

impl PersistentArena {
    const ALIGNMENT: usize = 256;

    fn planned_bytes(sizes: impl IntoIterator<Item = usize>) -> Result<usize> {
        sizes.into_iter().try_fold(0usize, |offset, len| {
            let aligned = offset
                .checked_add(Self::ALIGNMENT - 1)
                .map(|value| value & !(Self::ALIGNMENT - 1))
                .ok_or_else(|| Error::Other("Qwen CUDA arena alignment overflow".into()))?;
            aligned
                .checked_add(len)
                .ok_or_else(|| Error::Other("Qwen CUDA arena size overflow".into()))
        })
    }

    fn new(device: usize, bytes: usize) -> Result<Self> {
        Ok(Self {
            storage: CudaBuffer::alloc_zeros(bytes, device).map_err(Error::Cuda)?,
            offset: 0,
        })
    }

    fn take(&mut self, len: usize) -> Result<CudaBuffer> {
        let offset = self
            .offset
            .checked_add(Self::ALIGNMENT - 1)
            .map(|value| value & !(Self::ALIGNMENT - 1))
            .ok_or_else(|| Error::Other("Qwen CUDA arena alignment overflow".into()))?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| Error::Other("Qwen CUDA arena view overflow".into()))?;
        let view = self.storage.view(offset, len).map_err(Error::Cuda)?;
        self.offset = end;
        Ok(view)
    }

    fn take_bf16(&mut self, rows: usize, cols: usize) -> Result<CudaBuffer> {
        let bytes = rows
            .checked_mul(cols)
            .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
            .ok_or_else(|| Error::Other("Qwen CUDA BF16 arena size overflow".into()))?;
        self.take(bytes)
    }
}

struct Gemm {
    packed: Option<GemmPacked>,
    dense: Option<CudaBuffer>,
    out_cols: usize,
    in_cols: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum W4DeviceLayout {
    RawCompressedTensors,
    RepackedN64K16V1,
    TransformCacheN64K16V1,
    MarlinAwqU4G32V1,
}

struct GemmPacked {
    w: CudaBuffer,
    scale: CudaBuffer,
    zp: CudaBuffer,
    groups: usize,
    layout: W4DeviceLayout,
    padded_out_cols: usize,
    padded_in_cols: usize,
    /// Logical N offset inside a shared physical Marlin representation.
    source_n_offset: usize,
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
    gate_up_concat: Option<Gemm>,
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
    qkv_concat: Option<Gemm>,
    gate_up_concat: Option<Gemm>,
}

enum CudaLayer {
    Linear(LinearLayer),
    Full(FullLayer),
}

pub struct Qwen35Cuda {
    /// Decode-only graphs for maximal contiguous runs of linear-attention
    /// layers. Indexed by the first layer in each run; other entries stay
    /// `None`. Declared before every captured allocation so graph executables
    /// are destroyed first.
    decode_linear_graphs: Vec<Option<Box<dyn Graph>>>,
    /// Opt-in position-safe graph per full-attention layer for ordinary
    /// slot-zero decode. All position consumers dereference stable device data.
    decode_full_graphs: Vec<Option<Box<dyn Graph>>>,
    backend: Arc<dyn Backend>,
    embed: CudaBuffer,
    lm_head: Gemm,
    final_norm_w: CudaBuffer,
    layers: Vec<CudaLayer>,
    /// Single startup allocation backing all mutable model state, KV caches,
    /// activation workspaces, Marlin locks/scratch, and selection buffers.
    runtime_arena: CudaBuffer,
    hidden: usize,
    intermediate: usize,
    vocab: usize,
    eps: f32,
    rope_theta: f32,
    rotary_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    decode_linear_graphs_enabled: bool,
    exact_gdn_graphs_enabled: bool,
    gdn_fused_enabled: bool,
    pair_graph_enabled: bool,
    gdn_projection_overlap: bool,
    gdn_aux_stream: Option<apxinf_cuda::CudaStream>,
    gdn_norm_ready: Option<CudaEvent>,
    gdn_aux_done: Option<CudaEvent>,
    /// Decode-only exact GDN graph per model layer. Entries for full-attention
    /// layers remain `None`.
    exact_gdn_graphs: Vec<Option<ExactGdnGraph>>,
    /// Decode-only raw W4 pair graphs, indexed by `layer * 3 + projection site`.
    pair_graphs: Vec<Option<PairW4Graph>>,
    /// One graph for the complete ordinary slot-zero decode step. Keeping the
    /// entire step in one executable removes the per-token launch/API gap
    /// between the layer graphs, final LM head, and argmax.
    decode_step_graph: Option<Box<dyn Graph>>,
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
    gated: CudaBuffer,
    attn: CudaBuffer,
    attn_partials: CudaBuffer,
    attn2: CudaBuffer,
    gate_proj: CudaBuffer,
    up_proj: CudaBuffer,
    gate_up_proj: CudaBuffer,
    mlp_act: CudaBuffer,
    q_gate: CudaBuffer,
    q_buf: CudaBuffer,
    gate_buf: CudaBuffer,
    k_buf: CudaBuffer,
    v_buf: CudaBuffer,
    full_qkv_proj: CudaBuffer,
    logits: CudaBuffer,
    argmax_partials: CudaBuffer,
    argmax_arrivals: CudaBuffer,
    argmax_single_launch_enabled: bool,
    dense_scratch: CudaBuffer,
    /// Contiguous backup for every linear-attention conv/recurrent state.
    /// This is a persistent view into the unused decode tail of
    /// `dense_scratch`, not a second device allocation.
    linear_state_snapshot: Option<CudaBuffer>,
    ids: CudaBuffer,
    pos: CudaBuffer,
    ids_host: Vec<u8>,
    /// Device control slots: ids `[0..8)`, positions `[8..16)`.
    decode_control: CudaBuffer,

    /// Slot zero serves ordinary decode; slots 0..8 hold one draft block.
    argmax_out: HostMappedBuffer,
    argmax_ready: Option<CudaEvent>,
    decode_event_audit_enabled: bool,
}

/// A replayable exact GDN sequence plus owners for every captured device
/// pointer. The graph is declared first so its executable is destroyed before
/// any captured allocation can be released.
struct ExactGdnGraph {
    graph: Box<dyn Graph>,
    _buffers: Vec<CudaBuffer>,
}

impl ExactGdnGraph {
    fn replay(&self) -> Result<()> {
        self.graph.replay()
    }

    fn new(graph: Box<dyn Graph>, run: &LinearRun, model: &Qwen35Cuda) -> Self {
        Self {
            graph,
            _buffers: vec![
                model.qkv.clone(),
                model.qk_scratch.clone(),
                model.a.clone(),
                model.b.clone(),
                model.z.clone(),
                model.delta_out.clone(),
                model.gated.clone(),
                run.conv_w.clone(),
                run.conv_state.clone(),
                run.a_log.clone(),
                run.dt_bias.clone(),
                run.recurrent.clone(),
                run.gate_norm_w.clone(),
            ],
        }
    }
}
/// A replayable raw W4 pair launch plus owners for every captured device
/// pointer. The graph is declared first so its executable is destroyed before
/// any captured allocation can be released.
struct PairW4Graph {
    graph: Box<dyn Graph>,
    _buffers: Vec<CudaBuffer>,
}

impl PairW4Graph {
    fn replay(&self) -> Result<()> {
        self.graph.replay()
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        graph: Box<dyn Graph>,
        activation: &CudaBuffer,
        first_packed: &CudaBuffer,
        first_scale: &CudaBuffer,
        first_zp: &CudaBuffer,
        first_output: &CudaBuffer,
        second_packed: &CudaBuffer,
        second_scale: &CudaBuffer,
        second_zp: &CudaBuffer,
        second_output: &CudaBuffer,
    ) -> Self {
        Self {
            graph,
            _buffers: vec![
                activation.clone(),
                first_packed.clone(),
                first_scale.clone(),
                first_zp.clone(),
                first_output.clone(),
                second_packed.clone(),
                second_scale.clone(),
                second_zp.clone(),
                second_output.clone(),
            ],
        }
    }
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
    gate_up_concat: Option<Gemm>,
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
    qkv_concat: Option<Gemm>,
    gate_up_concat: Option<Gemm>,
}
/// Enqueue the established GDN state transition without changing its kernel
/// boundaries or arithmetic. CUDA graph capture records these exact launches;
/// replay therefore preserves both bf16 materialization boundaries and the
/// conv/recurrent state ordering.
fn launch_exact_gdn(
    ctx: &CudaContext,
    run: &mut LinearRun,
    qkv: &CudaBuffer,
    qk_scratch: &CudaBuffer,
    a: &CudaBuffer,
    b: &CudaBuffer,
    z: &CudaBuffer,
    delta_out: &CudaBuffer,
    gated: &CudaBuffer,
    seq: usize,
    eps: f32,
) -> Result<()> {
    kernels::qwen35::conv_silu(
        ctx,
        qkv,
        &run.conv_w,
        qkv,
        &mut run.conv_state,
        seq,
        run.conv_dim,
        run.conv_kernel,
    )?;
    kernels::qwen35::delta_norm_prepass(
        ctx,
        qkv,
        qk_scratch,
        seq,
        run.k_heads,
        run.v_heads,
        run.kdim,
        run.vdim,
    )?;
    kernels::qwen35::delta_step(
        ctx,
        qkv,
        qk_scratch,
        a,
        b,
        &run.a_log,
        &run.dt_bias,
        &mut run.recurrent,
        delta_out,
        seq,
        run.k_heads,
        run.v_heads,
        run.kdim,
        run.vdim,
    )?;
    kernels::qwen35::gated_norm(
        ctx,
        delta_out,
        z,
        &run.gate_norm_w,
        gated,
        seq,
        run.v_heads,
        run.vdim,
        eps,
    )
}

static PREFILL_GDN_WARPS: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
    std::env::var("APXINF_PREFILL_GDN_WARPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| matches!(value, 0 | 2 | 4))
        .unwrap_or(2)
});

/// Prefill path for the exact 128x128 geometry. Convolution and q/k norm keep
/// their existing globally materialized bf16 boundaries; the recurrence then
/// partitions only independent value-state columns into warp tiles before the
/// unchanged gated normalization. Within each tile, every token completes its
/// decay, delta update, and output before the next token begins.
#[allow(clippy::too_many_arguments)]
fn launch_prefill_gdn(
    ctx: &CudaContext,
    run: &mut LinearRun,
    qkv: &CudaBuffer,
    qk_scratch: &CudaBuffer,
    a: &CudaBuffer,
    b: &CudaBuffer,
    z: &CudaBuffer,
    delta_out: &CudaBuffer,
    gated: &CudaBuffer,
    seq: usize,
    eps: f32,
) -> Result<()> {
    kernels::qwen35::conv_silu(
        ctx,
        qkv,
        &run.conv_w,
        qkv,
        &mut run.conv_state,
        seq,
        run.conv_dim,
        run.conv_kernel,
    )?;
    kernels::qwen35::delta_norm_prepass(
        ctx,
        qkv,
        qk_scratch,
        seq,
        run.k_heads,
        run.v_heads,
        run.kdim,
        run.vdim,
    )?;
    let gdn_warps = *PREFILL_GDN_WARPS;
    if gdn_warps == 4 && kernels::qwen35::prepare_prefill_delta_step_4w()? {
        kernels::qwen35::prefill_delta_step_4w(
            ctx, qkv, qk_scratch, a, b, &run.a_log, &run.dt_bias,
            &mut run.recurrent, delta_out, seq, run.k_heads, run.v_heads,
            run.kdim, run.vdim,
        )?;
    } else if gdn_warps == 2 && kernels::qwen35::prepare_prefill_delta_step_2w()? {
        kernels::qwen35::prefill_delta_step_2w(
            ctx, qkv, qk_scratch, a, b, &run.a_log, &run.dt_bias,
            &mut run.recurrent, delta_out, seq, run.k_heads, run.v_heads,
            run.kdim, run.vdim,
        )?;
    } else {
        kernels::qwen35::prefill_delta_step(
            ctx, qkv, qk_scratch, a, b, &run.a_log, &run.dt_bias,
            &mut run.recurrent, delta_out, seq, run.k_heads, run.v_heads,
            run.kdim, run.vdim,
        )?;
    }
    kernels::qwen35::gated_norm(
        ctx,
        delta_out,
        z,
        &run.gate_norm_w,
        gated,
        seq,
        run.v_heads,
        run.vdim,
        eps,
    )
}

/// Maximal exact GDN launch reduction. Convolution stays separate because its
/// complete bf16 output is a dependency of every grouped recurrent block.
#[allow(clippy::too_many_arguments)]
fn launch_fused_gdn(
    ctx: &CudaContext,
    run: &mut LinearRun,
    qkv: &CudaBuffer,
    qk_scratch: &CudaBuffer,
    a: &CudaBuffer,
    b: &CudaBuffer,
    z: &CudaBuffer,
    delta_out: &CudaBuffer,
    gated: &CudaBuffer,
    seq: usize,
    eps: f32,
) -> Result<()> {
    kernels::qwen35::conv_silu(
        ctx,
        qkv,
        &run.conv_w,
        qkv,
        &mut run.conv_state,
        seq,
        run.conv_dim,
        run.conv_kernel,
    )?;
    kernels::qwen35::norm_delta_gated(
        ctx,
        qkv,
        qk_scratch,
        a,
        b,
        &run.a_log,
        &run.dt_bias,
        z,
        &run.gate_norm_w,
        &mut run.recurrent,
        delta_out,
        gated,
        seq,
        run.k_heads,
        run.v_heads,
        run.kdim,
        run.vdim,
        eps,
    )
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

        let bf16_bytes = |rows: usize, cols: usize| -> Result<usize> {
            rows.checked_mul(cols)
                .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
                .ok_or_else(|| Error::Other("Qwen CUDA arena BF16 size overflow".into()))
        };
        let mut arena_sizes = Vec::new();
        for layer_type in &tc.layer_types {
            match layer_type {
                LayerKind::LinearAttention => {
                    let key_dim = tc.linear_num_key_heads * tc.linear_key_head_dim;
                    let value_dim = tc.linear_num_value_heads * tc.linear_value_head_dim;
                    let conv_dim = key_dim * 2 + value_dim;
                    arena_sizes.push((tc.linear_conv_kernel_dim - 1) * conv_dim * 4);
                    arena_sizes.push(
                        tc.linear_num_value_heads
                            * tc.linear_key_head_dim
                            * tc.linear_value_head_dim
                            * 4,
                    );
                }
                LayerKind::FullAttention => {
                    let cache_bytes = tc.n_kv_heads * MAX_SEQ_LEN * tc.head_dim * 2;
                    arena_sizes.extend([cache_bytes, cache_bytes]);
                }
            }
        }
        let linear_key_dim = tc.linear_num_key_heads * tc.linear_key_head_dim;
        let linear_value_dim = tc.linear_num_value_heads * tc.linear_value_head_dim;
        let linear_qkv_dim = linear_key_dim * 2 + linear_value_dim;
        // The raw linear-QKV projection is the largest exact dequant+cublas
        // fallback. Its output lives immediately after the dequantized matrix.
        let dense_scratch_bytes = bf16_bytes(linear_qkv_dim, hidden)?;
        let qkv_output_bytes = bf16_bytes(CHUNK, linear_qkv_dim)?;
        let q_buf_bytes = bf16_bytes(CHUNK, tc.n_heads * tc.head_dim)?;
        let workspace_tail_bytes = qkv_output_bytes.max(
            q_buf_bytes
                .checked_mul(3)
                .ok_or_else(|| Error::Other("Qwen CUDA attention tail overflow".into()))?,
        );
        let workspace_bytes = dense_scratch_bytes
            .checked_add(workspace_tail_bytes)
            .ok_or_else(|| Error::Other("Qwen CUDA compact workspace overflow".into()))?;
        arena_sizes.extend([
            bf16_bytes(CHUNK, hidden)?,
            bf16_bytes(CHUNK, hidden)?,
            workspace_bytes,
            CHUNK * std::mem::size_of::<u32>(),
            CHUNK * std::mem::size_of::<u32>(),
            16 * std::mem::size_of::<u32>(),
            std::mem::size_of::<u32>(),
        ]);
        let arena_bytes = PersistentArena::planned_bytes(arena_sizes)?;
        let mut arena = PersistentArena::new(device, arena_bytes)?;

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
                    let mlp_base = format!("{prefix}.mlp.");
                    let (gate, up, gate_up_concat) = marlin_concat_group(
                        device_tensors,
                        &format!("{mlp_base}gate_up_concat"),
                        &[(format!("{mlp_base}gate_proj"), intermediate),
                          (format!("{mlp_base}up_proj"), intermediate)],
                        hidden,
                    )?.map_or_else(
                        || -> Result<(Gemm, Gemm, Option<Gemm>)> { Ok((
                            gemm(device_tensors, &format!("{mlp_base}gate_proj"), intermediate, hidden)?,
                            gemm(device_tensors, &format!("{mlp_base}up_proj"), intermediate, hidden)?,
                            None,
                        )) },
                        |(combined, mut members)| -> Result<(Gemm, Gemm, Option<Gemm>)> {
                            Ok((members.remove(0), members.remove(0), Some(combined)))
                        },
                    )?;
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
                        conv_state: arena.take((kernel - 1) * conv_dim * 4)?,
                        recurrent: arena.take(v_heads * kdim * vdim * 4)?,
                        gate,
                        up,
                        down: gemm(
                            device_tensors,
                            &format!("{prefix}.mlp.down_proj"),
                            hidden,
                            intermediate,
                        )?,
                        gate_up_concat,
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
                    let (q, k, v, qkv_concat) = marlin_concat_group(
                        device_tensors,
                        &format!("{attn}.qkv_concat"),
                        &[(format!("{attn}.q_proj"), tc.n_heads * head_dim * 2),
                          (format!("{attn}.k_proj"), n_kv * head_dim),
                          (format!("{attn}.v_proj"), n_kv * head_dim)],
                        hidden,
                    )?.map_or_else(
                        || -> Result<(Gemm, Gemm, Gemm, Option<Gemm>)> { Ok((
                            gemm(device_tensors, &format!("{attn}.q_proj"), tc.n_heads * head_dim * 2, hidden)?,
                            gemm(device_tensors, &format!("{attn}.k_proj"), n_kv * head_dim, hidden)?,
                            gemm(device_tensors, &format!("{attn}.v_proj"), n_kv * head_dim, hidden)?,
                            None,
                        )) },
                        |(combined, mut members)| -> Result<(Gemm, Gemm, Gemm, Option<Gemm>)> {
                            Ok((members.remove(0), members.remove(0), members.remove(0), Some(combined)))
                        },
                    )?;
                    let mlp_base = format!("{prefix}.mlp.");
                    let (gate, up, gate_up_concat) = marlin_concat_group(
                        device_tensors,
                        &format!("{mlp_base}gate_up_concat"),
                        &[(format!("{mlp_base}gate_proj"), intermediate),
                          (format!("{mlp_base}up_proj"), intermediate)],
                        hidden,
                    )?.map_or_else(
                        || -> Result<(Gemm, Gemm, Option<Gemm>)> { Ok((
                            gemm(device_tensors, &format!("{mlp_base}gate_proj"), intermediate, hidden)?,
                            gemm(device_tensors, &format!("{mlp_base}up_proj"), intermediate, hidden)?,
                            None,
                        )) },
                        |(combined, mut members)| -> Result<(Gemm, Gemm, Option<Gemm>)> {
                            Ok((members.remove(0), members.remove(0), Some(combined)))
                        },
                    )?;
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
                        q,
                        k,
                        v,
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
                        k_cache: arena.take(cache_bytes)?,
                        v_cache: arena.take(cache_bytes)?,
                        gate,
                        up,
                        down: gemm(
                            device_tensors,
                            &format!("{prefix}.mlp.down_proj"),
                            hidden,
                            intermediate,
                        )?,
                        qkv_concat,
                        gate_up_concat,
                    }));
                }
            }
        }

        let decode_async_enabled = std::env::var("APXINF_DECODE_ASYNC")
            .map_or(true, |value| value != "0");
        let decode_event_audit_enabled =
            std::env::var_os("APXINF_DECODE_EVENT_AUDIT").is_some();
        let gdn_projection_overlap = std::env::var_os("APXINF_GDN_PROJ_OVERLAP")
            .is_some_and(|value| value == "1");
        let gdn_aux_stream = gdn_projection_overlap
            .then(|| apxinf_cuda::CudaStream::new().map_err(Error::Cuda))
            .transpose()?;
        let gdn_norm_ready = gdn_projection_overlap
            .then(|| CudaEvent::new().map_err(Error::Cuda))
            .transpose()?;
        let gdn_aux_done = gdn_projection_overlap
            .then(|| CudaEvent::new().map_err(Error::Cuda))
            .transpose()?;
        let mut marlin_decode_head = 0usize;
        let lock_bytes = cb.context().caps().multiprocessor_count as usize
            * std::mem::size_of::<i32>();
        let mut account_gemm = |gemm: &Gemm| -> Result<()> {
            let Some(packed) = gemm.packed.as_ref() else {
                return Ok(());
            };
            if packed.layout == W4DeviceLayout::MarlinAwqU4G32V1 {
                let bytes = packed.padded_in_cols
                    .checked_add(packed.padded_out_cols)
                    .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
                    .and_then(|bytes| bytes.checked_add(lock_bytes))
                    .ok_or_else(|| Error::Other("Marlin decode scratch size overflow".into()))?;
                marlin_decode_head = marlin_decode_head.max(bytes);
            }
            Ok(())
        };
        for layer in &layers {
            match layer {
                CudaLayer::Linear(layer) => {
                    for gemm in [&layer.qkv, &layer.z, &layer.a, &layer.b, &layer.out,
                        &layer.gate, &layer.up, &layer.down]
                    {
                        account_gemm(gemm)?;
                    }
                    if let Some(gemm) = &layer.gate_up_concat {
                        account_gemm(gemm)?;
                    }
                }
                CudaLayer::Full(layer) => {
                    for gemm in [&layer.q, &layer.k, &layer.v, &layer.o,
                        &layer.gate, &layer.up, &layer.down]
                    {
                        account_gemm(gemm)?;
                    }
                    for gemm in [&layer.qkv_concat, &layer.gate_up_concat].into_iter().flatten() {
                        account_gemm(gemm)?;
                    }
                }
            }
        }
        if marlin_decode_head > dense_scratch_bytes {
            return Err(Error::Other(format!(
                "Marlin decode scratch requires {marlin_decode_head} bytes, compact scratch has {dense_scratch_bytes}"
            )));
        }

        let x = arena.take_bf16(CHUNK, hidden)?;
        let normed = arena.take_bf16(CHUNK, hidden)?;
        let workspace = arena.take(workspace_bytes)?;
        let view = |offset: usize, len: usize| workspace.view(offset, len).map_err(Error::Cuda);
        let dense_scratch = view(0, dense_scratch_bytes)?;

        let qkv_bytes = bf16_bytes(CHUNK, linear_qkv_dim)?;
        let value_bytes = bf16_bytes(CHUNK, linear_value_dim)?;
        let qkv = view(dense_scratch_bytes, qkv_bytes)?;
        let z_weight_bytes = bf16_bytes(linear_value_dim, hidden)?;
        let z = view(z_weight_bytes, value_bytes)?;
        let scalar_head_bytes = bf16_bytes(CHUNK, tc.linear_num_value_heads)?;
        let a = view(z_weight_bytes + value_bytes, scalar_head_bytes)?;
        let b = view(z_weight_bytes + value_bytes + scalar_head_bytes, scalar_head_bytes)?;

        let qk_scratch_bytes = bf16_bytes(CHUNK, linear_key_dim * 2)?;
        let qk_scratch = view(0, qk_scratch_bytes)?;
        let delta_out = view(qk_scratch_bytes, value_bytes)?;
        let gated = view(qk_scratch_bytes + value_bytes, value_bytes)?;

        // Projection outputs begin at 24 MiB, beyond the largest 512-row
        // Marlin activation/output scratch partition and its lock words. Layer
        // categories and MLP phases execute serially on one stream, so these
        // aliases cannot be live at the same time.
        let phase_offset = 24 * 1024 * 1024;
        let full_qkv_cols = tc.n_heads * tc.head_dim * 2 + 2 * tc.n_kv_heads * tc.head_dim;
        let full_qkv_bytes = bf16_bytes(CHUNK, full_qkv_cols)?;
        let full_qkv_proj = view(phase_offset, full_qkv_bytes)?;
        let q_gate_bytes = bf16_bytes(CHUNK, tc.n_heads * tc.head_dim * 2)?;
        let kv_bytes = bf16_bytes(CHUNK, tc.n_kv_heads * tc.head_dim)?;
        let q_gate = view(phase_offset, q_gate_bytes)?;
        let k_buf = view(phase_offset + q_gate_bytes, kv_bytes)?;
        let v_buf = view(phase_offset + q_gate_bytes + kv_bytes, kv_bytes)?;
        let q_buf = view(dense_scratch_bytes, q_buf_bytes)?;
        let gate_buf = view(dense_scratch_bytes + q_buf_bytes, q_buf_bytes)?;
        let attn = view(dense_scratch_bytes + q_buf_bytes * 2, q_buf_bytes)?;
        let attn2 = view(phase_offset + q_buf_bytes, bf16_bytes(CHUNK, hidden)?)?;

        let gate_up_bytes = bf16_bytes(CHUNK, intermediate * 2)?;
        let gate_bytes = bf16_bytes(CHUNK, intermediate)?;
        let gate_up_proj = view(phase_offset, gate_up_bytes)?;
        let gate_proj = view(phase_offset, gate_bytes)?;
        let up_proj = view(phase_offset + gate_bytes, gate_bytes)?;
        let mlp_act = view(phase_offset + gate_up_bytes, gate_bytes)?;
        let logits = view(phase_offset, bf16_bytes(1, vocab)?)?;
        let aux_offset = q_buf_bytes * 2;
        let attn_partials = view(aux_offset, 24 * 8 * (2 + 8 * 32) * 4)?;
        let argmax_partials = view(aux_offset, 128 * 8)?;
        let ids = arena.take(CHUNK * 4)?;
        let pos = arena.take(CHUNK * 4)?;
        let decode_control = arena.take(16 * 4)?;
        let argmax_arrivals = arena.take(4)?;
        let model = Self {
            backend,
            embed: buffer_from(device_tensors, "model.language_model.embed_tokens.weight")?,
            lm_head: gemm(device_tensors, "lm_head", vocab, hidden)?,
            final_norm_w: upload_norm_plus_one(
                device,
                tensors,
                "model.language_model.norm.weight",
                hidden,
            )?,
            layers,
            runtime_arena: arena.storage.clone(),
            hidden,
            intermediate,
            vocab,
            eps: tc.rms_norm_eps,
            rope_theta: tc.rope_theta,
            rotary_dim,
            n_heads: tc.n_heads,
            n_kv_heads: tc.n_kv_heads,
            head_dim: tc.head_dim,
            decode_linear_graphs_enabled: !gdn_projection_overlap
                && std::env::var("APXINF_DECODE_LAYER_GRAPH")
                    .map_or(true, |value| value != "0")
                && std::env::var_os("APXINF_GEMM_PROF").is_none()
                && std::env::var_os("APXINF_KERNEL_PROF").is_none()
                && std::env::var_os("APXINF_LAYER_PROF").is_none()
                && std::env::var_os("APXINF_TRACE").is_none(),
            exact_gdn_graphs_enabled: std::env::var_os("APXINF_EXACT_GDN_GRAPH").is_some()
                && std::env::var_os("APXINF_TRACE").is_none()
                && std::env::var_os("APXINF_KERNEL_PROF").is_none(),
            gdn_fused_enabled: std::env::var("APXINF_GDN_FUSED")
                .map_or(true, |value| value != "0"),
            // Pair graphs are the decode default; disable them for profiling so
            // event instrumentation observes the real eager kernel launches.
            pair_graph_enabled: std::env::var("APXINF_PAIR_GRAPH").map_or(true, |value| value != "0")
                && std::env::var_os("APXINF_GEMM_PROF").is_none()
                && std::env::var_os("APXINF_KERNEL_PROF").is_none()
                && std::env::var_os("APXINF_TRACE").is_none(),
            gdn_projection_overlap,
            gdn_aux_stream,
            gdn_norm_ready,
            gdn_aux_done,
            exact_gdn_graphs: (0..tc.n_layers).map(|_| None).collect(),
            decode_step_graph: None,
            pair_graphs: (0..tc.n_layers * 3).map(|_| None).collect(),
            decode_full_graphs: (0..tc.n_layers).map(|_| None).collect(),
            decode_linear_graphs: (0..tc.n_layers).map(|_| None).collect(),
            x,
            normed: normed.clone(),
            normed2: normed,
            qkv,
            z,
            a,
            b,
            delta_out,
            qk_scratch,
            gate_up_proj,
            gated,
            attn,
            attn_partials,
            attn2,
            gate_proj,
            up_proj,
            mlp_act,
            full_qkv_proj,
            q_gate,
            q_buf,
            gate_buf,
            k_buf,
            v_buf,
            logits,
            argmax_partials,
            argmax_arrivals,
            argmax_single_launch_enabled: std::env::var("APXINF_ARGMAX_SINGLE_LAUNCH")
                .map_or(true, |value| value != "0"),
            dense_scratch,
            linear_state_snapshot: None,
            ids,
            ids_host: vec![0; CHUNK * 4],
            pos,
            decode_control,
            argmax_out: HostMappedBuffer::alloc(8 * 4, device).map_err(Error::Cuda)?,
            argmax_ready: if decode_async_enabled && !decode_event_audit_enabled {
                Some(CudaEvent::new().map_err(Error::Cuda)?)
            } else {
                None
            },
            decode_event_audit_enabled,
        };
        Ok(model)
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
            .stream();
        for layer in &mut self.layers {
            if let CudaLayer::Linear(l) = layer {
                l.conv_state.zero_async(stream).map_err(Error::Cuda)?;
                l.recurrent.zero_async(stream).map_err(Error::Cuda)?;
            }
        }
        Ok(())
    }

    fn prefill_chunk_len(&self, remaining: usize, start_pos: u32) -> usize {
        let limit = remaining.min(CHUNK);
        (1..=limit)
            .rev()
            .find(|&seq| {
                let visible = start_pos as usize + seq;
                let vf32 = visible.min(V_TILE_ROWS) * self.head_dim * 4;
                let scores = seq * visible * 4;
                let kt = self.head_dim * visible * 2;
                let pv = seq * self.n_heads * self.head_dim * 4;
                let l_sums = seq * self.n_heads * 4;
                scores + kt + vf32 + pv + l_sums <= self.dense_scratch.len()
            })
            .unwrap_or(1)
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
            let pos = start_pos + offset as u32;
            let chunk = self.prefill_chunk_len(seq - offset, pos);
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

    /// Full prompt prefill followed by exact GPU BF16 argmax. Only the 4-byte
    /// selected token is made visible to the host.
    pub fn prefill_token(&mut self, token_ids: &[u32]) -> Result<u32> {
        if token_ids.is_empty() {
            return Err(Error::Other("qwen3_5 GPU prefill: empty input".into()));
        }
        let mut offset = 0usize;
        let mut last_chunk = 0usize;
        while offset < token_ids.len() {
            let chunk = self.prefill_chunk_len(token_ids.len() - offset, offset as u32);
            self.forward_chunk(&token_ids[offset..offset + chunk], chunk, offset as u32)?;
            last_chunk = chunk;
            offset += chunk;
        }
        // Prefill aliases dense scratch for inverse weights and attention. Its
        // final stream-ordered clear initializes every possible Marlin lock
        // offset once; M=1 kernels restore used locks to zero on completion.
        self.dense_scratch
            .zero_async(self.ctx().stream())
            .map_err(Error::Cuda)?;
        self.final_logits_into(last_chunk)?;
        self.select_token_into()?;
        self.wait_for_selected_token()?;
        self.argmax_out.read_u32(0).map_err(Error::Cuda)
    }

    /// Decode fast path: one token in, GPU argmax out. Token ID and position
    /// share one compact H2D control upload.
    pub fn decode_token(&mut self, token: u32, pos: u32) -> Result<u32> {
        let perf = std::env::var_os("APXINF_PERF").is_some();
        let t0 = std::time::Instant::now();
        self.forward_chunk(std::slice::from_ref(&token), 1, pos)?;
        self.final_logits_into(1)?;
        self.select_token_into()?;
        let t1 = std::time::Instant::now();
        self.wait_for_selected_token()?;
        let tok = self.argmax_out.read_u32(0).map_err(Error::Cuda)?;
        if perf {
            eprintln!(
                "[decode] enqueue={:.2}ms wait={:.2}ms",
                (t1 - t0).as_secs_f32() * 1000.0,
                t1.elapsed().as_secs_f32() * 1000.0,
            );
        }
        Ok(tok)
    }

    /// Enqueue one ordinary seq=1 decode step from mapped input/position slots
    /// without waiting for its greedy token or issuing a CUDA transfer.
    fn enqueue_decode_step(&mut self, pos: u32, slot: usize) -> Result<()> {
        // Slot zero is the ordinary autoregressive path. Draft verification
        // uses independent mapped slots and must retain the eager layer loop.
        // The graph is captured lazily after the first real prefill, so all
        // CUDA library/kernel state is initialized before capture.
        if slot == 0 && self.decode_step_graph_enabled() {
            return self.run_decode_step_graph(pos);
        }
        self.enqueue_decode_step_eager(pos, slot)
    }

    fn decode_step_graph_enabled(&self) -> bool {
        self.decode_linear_graphs_enabled
            && std::env::var("APXINF_DECODE_STEP_GRAPH")
                .map_or(true, |value| value != "0")
    }

    fn run_decode_step_graph(&mut self, pos: u32) -> Result<()> {
        if let Some(graph) = self.decode_step_graph.as_ref() {
            return graph.replay();
        }

        let backend = self.backend.clone();
        let cuda_backend = backend
            .as_any()
            .downcast_ref::<CudaBackend>()
            .expect("qwen3_5 CUDA backend downcast");
        if cuda_backend.begin_capture_relaxed().is_err() {
            return self.enqueue_decode_step_eager(pos, 0);
        }

        // Disable nested per-layer/paired graph captures while recording the
        // complete step. The captured nodes are the same eager kernels; only
        // their launch boundary changes.
        let linear_graphs_enabled = self.decode_linear_graphs_enabled;
        let pair_graph_enabled = self.pair_graph_enabled;
        let exact_gdn_graphs_enabled = self.exact_gdn_graphs_enabled;
        self.decode_linear_graphs_enabled = false;
        self.pair_graph_enabled = false;
        self.exact_gdn_graphs_enabled = false;
        let capture_result = self.enqueue_decode_step_eager(pos, 0);
        let graph_result = backend.end_capture();
        self.decode_linear_graphs_enabled = linear_graphs_enabled;
        self.pair_graph_enabled = pair_graph_enabled;
        self.exact_gdn_graphs_enabled = exact_gdn_graphs_enabled;

        match (capture_result, graph_result) {
            (Ok(()), Ok(graph)) => {
                // Capture records launches without executing them. Replay once
                // now to produce the token corresponding to this decode call.
                graph.replay()?;
                self.decode_step_graph = Some(graph);
                Ok(())
            }
            _ => self.enqueue_decode_step_eager(pos, 0),
        }
    }

    fn enqueue_decode_step_eager(&mut self, pos: u32, slot: usize) -> Result<()> {
        let id = self.decode_control
            .view(slot * 4, 4)
            .map_err(Error::Cuda)?
            .address();
        let position = self.decode_control
            .view((8 + slot) * 4, 4)
            .map_err(Error::Cuda)?
            .address();
        kernels::embedding::lookup_into(
            self.ctx(), DType::BF16, &self.embed, id, &self.x, self.hidden, 1,
        )?;
        if self.decode_linear_graphs_enabled {
            let mut l = 0;
            while l < self.layers.len() {
                if matches!(self.layers[l], CudaLayer::Full(_)) {
                    if slot == 0
                        && std::env::var("APXINF_FULL_LAYER_GRAPH")
                            .map_or(true, |value| value != "0")
                    {
                        self.run_decode_full_layer_graph(l, position)?;
                    } else {
                        self.run_layer(l, 1, pos, position)?;
                    }
                    l += 1;
                } else {
                    let start = l;
                    while l < self.layers.len()
                        && matches!(self.layers[l], CudaLayer::Linear(_))
                    {
                        l += 1;
                    }
                    self.run_decode_linear_segment(start, l)?;
                }
            }
        } else {
            for l in 0..self.layers.len() {
                self.run_layer(l, 1, pos, position)?;
            }
        }
        self.final_logits_into(1)?;
        kernels::qwen35::argmax_bf16_single_launch_at(
            self.ctx(), &self.logits, self.vocab, &self.argmax_partials,
            &self.argmax_arrivals, &self.argmax_out, slot,
        )
    }

    fn run_decode_full_layer_graph(
        &mut self,
        layer: usize,
        position: CudaDeviceAddress,
    ) -> Result<()> {
        if let Some(graph) = self.decode_full_graphs[layer].as_ref() {
            return graph.replay();
        }
        let backend = self.backend.clone();
        let cuda_backend = backend.as_any().downcast_ref::<CudaBackend>()
            .expect("qwen3_5 CUDA backend downcast");
        if cuda_backend.begin_capture_relaxed().is_err() {
            return self.run_layer(layer, 1, 0, position);
        }
        let pair_graph_enabled = self.pair_graph_enabled;
        self.pair_graph_enabled = false;
        let capture_result = self.run_layer(layer, 1, 0, position);
        let graph_result = backend.end_capture();
        self.pair_graph_enabled = pair_graph_enabled;
        match (capture_result, graph_result) {
            (Ok(()), Ok(graph)) => {
                graph.replay()?;
                self.decode_full_graphs[layer] = Some(graph);
                Ok(())
            }
            _ => self.run_layer(layer, 1, 0, position),
        }
    }

    /// Replay one committed state transition without LM-head/argmax work.
    fn replay_decode_state(&mut self, token: u32, pos: u32) -> Result<()> {
        self.upload_decode_controls(&[token], &[pos])?;
        let id = self.decode_control.view(0, 4).map_err(Error::Cuda)?.address();
        let position = self.decode_control
            .view(8 * 4, 4)
            .map_err(Error::Cuda)?
            .address();
        kernels::embedding::lookup_into(
            self.ctx(), DType::BF16, &self.embed, id, &self.x, self.hidden, 1,
        )?;
        if self.decode_linear_graphs_enabled {
            let mut l = 0;
            while l < self.layers.len() {
                if matches!(self.layers[l], CudaLayer::Full(_)) {
                    self.run_layer(l, 1, pos, position)?;
                    l += 1;
                } else {
                    let start = l;
                    while l < self.layers.len()
                        && matches!(self.layers[l], CudaLayer::Linear(_))
                    {
                        l += 1;
                    }
                    self.run_decode_linear_segment(start, l)?;
                }
            }
        } else {
            for l in 0..self.layers.len() {
                self.run_layer(l, 1, pos, position)?;
            }
        }
        Ok(())
    }

    pub fn draft_block_supported(&self) -> bool {
        self.linear_state_snapshot.is_some()
    }

    /// Verify a prompt-lookup draft with the ordinary single-token kernels.
    ///
    /// The snapshot covers all cumulatively mutated linear conv/recurrent
    /// state. Full-attention KV rows are position-indexed, so speculative rows
    /// past the committed position remain invisible and are overwritten by a
    /// later decode. On rejection we restore the snapshot and replay exactly
    /// `current_token` followed by the accepted draft prefix. Consequently the
    /// state on return is byte-for-byte the state produced by the same ordinary
    /// serial `decode_token` calls; no `seq > 1` arithmetic is used.
    pub fn decode_draft_block(
        &mut self,
        current_token: u32,
        start_pos: u32,
        draft: &[u32],
        verified: &mut Vec<u32>,
    ) -> Result<crate::llm_trait::DraftBlockResult> {
        verified.clear();
        if draft.is_empty() || draft.len() > 8 {
            return Err(Error::Other("qwen3_5 draft block must contain 1..=8 tokens".into()));
        }
        if self.linear_state_snapshot.is_none() {
            return Err(Error::Other(
                "qwen3_5 draft verification snapshot does not fit decode scratch".into(),
            ));
        }

        // Every speculative input is known up front: current_token followed by
        // prior draft proposals. Stage all scalar inputs before enqueueing any
        // model work so host copies cannot introduce per-step synchronization.
        let mut inputs = [0u32; 8];
        let mut positions = [0u32; 8];
        inputs[0] = current_token;
        for index in 1..draft.len() {
            inputs[index] = draft[index - 1];
        }
        for (index, position) in positions[..draft.len()].iter_mut().enumerate() {
            *position = start_pos
                .checked_add(index as u32)
                .ok_or_else(|| Error::Other("draft decode position overflow".into()))?;
        }
        self.upload_decode_controls(&inputs[..draft.len()], &positions[..draft.len()])?;
        self.copy_linear_state(true)?;
        for index in 0..draft.len() {
            self.enqueue_decode_step(positions[index], index)?;
        }
        self.ctx().synchronize().map_err(Error::Cuda)?;
        for index in 0..draft.len() {
            verified.push(self.argmax_out.read_u32(index).map_err(Error::Cuda)?);
        }

        let accepted = verified
            .iter()
            .zip(draft)
            .take_while(|(greedy, proposed)| greedy == proposed)
            .count();
        if accepted == draft.len() {
            return Ok(crate::llm_trait::DraftBlockResult {
                consumed_draft: draft.len(),
                accepted_prefix: accepted,
            });
        }

        // Restore the cumulative linear state, then replay exactly the inputs
        // whose greedy outputs are committed. KV rows are position-indexed;
        // replay overwrites committed rows and later speculative rows remain
        // invisible until their positions are reached and overwritten.
        self.copy_linear_state(false)?;
        for index in 0..=accepted {
            self.replay_decode_state(inputs[index], positions[index])?;
        }
        verified.truncate(accepted + 1);
        Ok(crate::llm_trait::DraftBlockResult {
            consumed_draft: accepted + 1,
            accepted_prefix: accepted,
        })
    }

    /// Copy all linear state to (`save`) or from the contiguous scratch tail.
    /// Copies share the model stream, preserving ordering with decode kernels.
    fn copy_linear_state(&mut self, save: bool) -> Result<()> {
        let snapshot = self.linear_state_snapshot.as_ref()
            .expect("draft snapshot availability checked")
            .clone();
        let backend = self.backend.clone();
        let stream = backend.as_any()
            .downcast_ref::<CudaBackend>()
            .expect("qwen3_5 CUDA backend downcast")
            .context()
            .stream();
        let mut offset = 0usize;
        for layer in &self.layers {
            let CudaLayer::Linear(layer) = layer else { continue };
            for state in [&layer.conv_state, &layer.recurrent] {
                let backup = snapshot.view(offset, state.len()).map_err(Error::Cuda)?;
                if save {
                    backup.copy_d2d_async(state, state.len(), stream).map_err(Error::Cuda)?;
                } else {
                    state.copy_d2d_async(&backup, state.len(), stream).map_err(Error::Cuda)?;
                }
                offset += state.len();
            }
        }
        debug_assert_eq!(offset, snapshot.len());
        Ok(())
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
            let pos = start_pos + offset as u32;
            let chunk = self.prefill_chunk_len(seq - offset, pos);
            last_chunk = chunk;
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
                self.run_layer(l, chunk, pos, self.pos.address())?;
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
        let end_pos = (start_pos as usize)
            .checked_add(seq)
            .ok_or_else(|| Error::Other("qwen3_5 GPU forward: sequence position overflow".into()))?;
        if end_pos > MAX_SEQ_LEN {
            return Err(Error::Other(format!(
                "qwen3_5 GPU forward: position {start_pos} + length {seq} exceeds cache capacity {MAX_SEQ_LEN}"
            )));
        }
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
        if seq == 1 && self.decode_linear_graphs_enabled {
            let mut l = 0;
            while l < self.layers.len() {
                if matches!(self.layers[l], CudaLayer::Full(_)) {
                    self.run_layer(l, seq, start_pos, self.pos.address())?;
                    l += 1;
                    continue;
                }
                let start = l;
                while l < self.layers.len() && matches!(self.layers[l], CudaLayer::Linear(_)) {
                    l += 1;
                }
                self.run_decode_linear_segment(start, l)?;
            }
        } else {
            for l in 0..self.layers.len() {
                self.run_layer(l, seq, start_pos, self.pos.address())?;
            }
        }
        Ok(())
    }

    /// Replay or create one graph for a maximal contiguous linear-attention
    /// segment. Full-attention layers remain outside because their device
    /// position varies per token.
    fn run_decode_linear_segment(&mut self, start: usize, end: usize) -> Result<()> {
        let _gdn_range = apxinf_cuda::nvtx::range_static(b"Qwen/GDN\0");
        if let Some(graph) = self.decode_linear_graphs[start].as_ref() {
            return graph.replay();
        }
        let _ = kernels::qwen35::prepare_packed_delta_gated()?;
        let _ = kernels::qwen35::prepare_norm_delta_gated()?;
        kernels::qwen35::prepare_exact_gdn()?;
        let backend = self.backend.clone();
        let cuda_backend = backend
            .as_any()
            .downcast_ref::<CudaBackend>()
            .expect("qwen3_5 CUDA backend downcast");
        if cuda_backend.begin_capture_relaxed().is_err() {
            for l in start..end {
                self.run_layer(l, 1, 0, self.pos.address())?;
            }
            return Ok(());
        }
        let pair_graph_enabled = self.pair_graph_enabled;
        let exact_gdn_graphs_enabled = self.exact_gdn_graphs_enabled;
        self.pair_graph_enabled = false;
        self.exact_gdn_graphs_enabled = false;
        let capture_result: Result<()> = (|| {
            for l in start..end {
                self.run_layer(l, 1, 0, self.pos.address())?;
            }
            Ok(())
        })();
        let graph_result = backend.end_capture();
        self.pair_graph_enabled = pair_graph_enabled;
        self.exact_gdn_graphs_enabled = exact_gdn_graphs_enabled;
        match (capture_result, graph_result) {
            (Ok(()), Ok(graph)) => {
                graph.replay()?;
                self.decode_linear_graphs[start] = Some(graph);
            }
            _ => {
                for l in start..end {
                    self.run_layer(l, 1, 0, self.pos.address())?;
                }
            }
        }
        Ok(())
    }

    fn run_layer(
        &mut self,
        l: usize,
        seq: usize,
        start_pos: u32,
        position: CudaDeviceAddress,
    ) -> Result<()> {
        let perf = std::env::var_os("APXINF_LAYER_PROF").is_some();
        let t0 = std::time::Instant::now();
        let result = match self.layers[l] {
            CudaLayer::Linear(_) => self.run_linear(l, seq),
            CudaLayer::Full(_) => self.run_full(l, seq, start_pos, position),
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
            gate_up_concat: layer.gate_up_concat.as_ref().map(gemm_clone),
        }
    }

    fn run_linear(&mut self, l: usize, seq: usize) -> Result<()> {
        let backend = self.backend.clone();
        let cuda_backend = backend
            .as_any()
            .downcast_ref::<CudaBackend>()
            .expect("qwen3_5 CUDA backend downcast");
        let ctx = cuda_backend.context();
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
        if seq == 1 && self.gdn_projection_overlap {
            let aux_stream = self.gdn_aux_stream.as_ref().expect("overlap stream");
            let norm_ready = self.gdn_norm_ready.as_ref().expect("overlap ready event");
            norm_ready.record(ctx.stream()).map_err(Error::Cuda)?;
            aux_stream.wait_event(norm_ready).map_err(Error::Cuda)?;
        }
        let raw_qkv_decode = seq == 1
            && run.qkv.in_cols == 5120
            && run.qkv.out_cols == 10240
            && run.qkv.packed.as_ref().is_some_and(|packed| {
                packed.layout == W4DeviceLayout::RawCompressedTensors && packed.groups == 160
            })
            && (self.gdn_projection_overlap
                || std::env::var("APXINF_W4_QKV_EXACT").is_ok_and(|value| value == "1"));
        if raw_qkv_decode {
            let packed = run.qkv.packed.as_ref().expect("checked raw qkv weights");
            if let Ok(mode) = std::env::var("APXINF_QKV_SCHED") {
                let mode = mode.parse::<usize>().map_err(|_| {
                    Error::Other("APXINF_QKV_SCHED must be an integer 0..7".into())
                })?;
                kernels::quantization::matmul_bf16_w4a16_asym_qkv_sched(
                    ctx, &self.normed, &packed.w, &packed.scale, &packed.zp,
                    &self.qkv, run.qkv.in_cols, run.qkv.out_cols, packed.groups,
                    mode,
                )?;
            } else if std::env::var("APXINF_QKV_PERSISTENT").is_ok_and(|value| value == "1") {
                kernels::quantization::matmul_bf16_w4a16_asym_tc_persistent(
                    ctx, &self.normed, &packed.w, &packed.scale, &packed.zp,
                    &self.qkv, run.qkv.in_cols, run.qkv.out_cols, packed.groups,
                )?;
            } else {
                kernels::quantization::matmul_bf16_w4a16_asym_qkv_10240x5120(
                    ctx, &self.normed, &packed.w, &packed.scale, &packed.zp,
                    &self.qkv, run.qkv.in_cols, run.qkv.out_cols, packed.groups,
                )?;
            }
        }
        if raw_qkv_decode && self.gdn_projection_overlap {
            let zp = run.z.packed.as_ref().ok_or_else(|| {
                Error::Other("GDN overlap requires raw z weights".into())
            })?;
            if zp.layout != W4DeviceLayout::RawCompressedTensors || zp.groups != 160 {
                return Err(Error::Other("GDN overlap z layout mismatch".into()));
            }
            kernels::quantization::matmul_bf16_w4a16_asym_tc_tile_alt_on_stream(
                ctx, &self.normed, &zp.w, &zp.scale, &zp.zp, &self.z,
                run.z.in_cols, run.z.out_cols, zp.groups,
                self.gdn_aux_stream.as_ref().expect("overlap stream"),
            )?;
            self.gdn_aux_done.as_ref().expect("overlap done event")
                .record(self.gdn_aux_stream.as_ref().expect("overlap stream"))
                .map_err(Error::Cuda)?;
            gemm_run(ctx, &run.a, &self.normed, &self.a, seq, &self.dense_scratch)?;
            gemm_run(ctx, &run.b, &self.normed, &self.b, seq, &self.dense_scratch)?;
        } else if raw_qkv_decode {
            if !gemm_run_marlin_batch(
                ctx,
                [(&run.z, &self.z), (&run.a, &self.a), (&run.b, &self.b)],
                &self.normed,
                seq,
                &self.dense_scratch,
            )? {
                gemm_run(ctx, &run.z, &self.normed, &self.z, seq, &self.dense_scratch)?;
                if !gemm_run_marlin_batch(
                    ctx,
                    [(&run.a, &self.a), (&run.b, &self.b)],
                    &self.normed,
                    seq,
                    &self.dense_scratch,
                )? {
                    gemm_run(ctx, &run.a, &self.normed, &self.a, seq, &self.dense_scratch)?;
                    gemm_run(ctx, &run.b, &self.normed, &self.b, seq, &self.dense_scratch)?;
                }
            }
        } else if !gemm_run_marlin_batch(
            ctx,
            [
                (&run.qkv, &self.qkv),
                (&run.z, &self.z),
                (&run.a, &self.a),
                (&run.b, &self.b),
            ],
            &self.normed,
            seq,
            &self.dense_scratch,
        )? && !gemm_run_multi(
            ctx,
            [&run.qkv, &run.z, &run.a],
            Some(&run.b),
            &self.normed,
            [&self.qkv, &self.z, &self.a],
            Some(&self.b),
            seq,
        )? {
            if !gemm_run_marlin_batch(
                ctx,
                [(&run.qkv, &self.qkv), (&run.z, &self.z)],
                &self.normed,
                seq,
                &self.dense_scratch,
            )? && !gemm_run_prefill_pair(
                ctx, &run.qkv, &run.z, &self.normed, &self.qkv, &self.z, seq,
                &self.dense_scratch,
            )? && !gemm_run_pair(
                ctx,
                &run.qkv,
                &run.z,
                &self.normed,
                &self.qkv,
                &self.z,
                seq,
                self.pair_graph_enabled.then_some((
                    cuda_backend,
                    &mut self.pair_graphs[l * 3],
                )),
            )? {
                gemm_run(ctx, &run.qkv, &self.normed, &self.qkv, seq, &self.dense_scratch)?;
                gemm_run(ctx, &run.z, &self.normed, &self.z, seq, &self.dense_scratch)?;
            }
            if !gemm_run_marlin_batch(
                ctx,
                [(&run.a, &self.a), (&run.b, &self.b)],
                &self.normed,
                seq,
                &self.dense_scratch,
            )? {
                gemm_run(ctx, &run.a, &self.normed, &self.a, seq, &self.dense_scratch)?;
                gemm_run(ctx, &run.b, &self.normed, &self.b, seq, &self.dense_scratch)?;
            }
        }
        if self.gdn_projection_overlap {
            ctx.stream().wait_event(
                self.gdn_aux_done.as_ref().expect("overlap done event")
            ).map_err(Error::Cuda)?;
        }
        if l == 0 {
            trace_buf("gpu_qkv_pre2", &self.qkv, seq * run.conv_dim);
        }

        // Decode keeps its packed/fused routing unchanged. For short prefill,
        // use the split four-launch arithmetic with a tiled recurrent kernel:
        // it exposes four independent value-column blocks per head and needs
        // no dynamic-shared-memory preflight. Other geometry and seq > 512
        // retain the established fused/eager fallback.
        let use_packed_gdn = seq == 1
            && self.gdn_fused_enabled
            && run.kdim == 128
            && run.vdim == 128
            && run.conv_kernel == 4
            && run.k_heads != 0
            && run.v_heads % run.k_heads == 0
            && kernels::qwen35::prepare_packed_delta_gated()?;
        let use_prefill_gdn = (2..=512).contains(&seq)
            && self.gdn_fused_enabled
            && run.kdim == 128
            && run.vdim == 128
            && run.k_heads != 0
            && run.v_heads % run.k_heads == 0;
        let use_fused_gdn = !use_packed_gdn
            && !use_prefill_gdn
            && self.gdn_fused_enabled
            && run.kdim == 128
            && run.vdim == 128
            && run.k_heads != 0
            && run.v_heads % run.k_heads == 0
            && kernels::qwen35::prepare_norm_delta_gated()?;
        if use_packed_gdn {
            kernels::qwen35::packed_delta_gated(
                ctx,
                &self.qkv,
                &run.conv_w,
                &mut run.conv_state,
                &self.a,
                &self.b,
                &run.a_log,
                &run.dt_bias,
                &self.z,
                &run.gate_norm_w,
                &mut run.recurrent,
                &self.gated,
                seq,
                run.k_heads,
                run.v_heads,
                run.kdim,
                run.vdim,
                run.conv_kernel,
                self.eps,
            )?;
        } else if use_prefill_gdn {
            launch_prefill_gdn(
                ctx,
                &mut run,
                &self.qkv,
                &self.qk_scratch,
                &self.a,
                &self.b,
                &self.z,
                &self.delta_out,
                &self.gated,
                seq,
                self.eps,
            )?;
        } else if use_fused_gdn {
            launch_fused_gdn(
                ctx,
                &mut run,
                &self.qkv,
                &self.qk_scratch,
                &self.a,
                &self.b,
                &self.z,
                &self.delta_out,
                &self.gated,
                seq,
                self.eps,
            )?;
        // Decode has stable activation and per-layer state addresses. Capture
        // only the established four launches, after preparing the delta
        // kernel's one-time launch attribute outside capture. No kernel body,
        // launch geometry, intermediate bf16 boundary, or state pointer is
        // changed by replay. Prefill retains the same eager sequence.
        } else if seq == 1 && self.exact_gdn_graphs_enabled {
            if let Some(graph) = self.exact_gdn_graphs[l].as_ref() {
                graph.replay()?;
            } else {
                kernels::qwen35::prepare_exact_gdn()?;
                cuda_backend.begin_capture_relaxed()?;
                let capture_result = launch_exact_gdn(
                    ctx,
                    &mut run,
                    &self.qkv,
                    &self.qk_scratch,
                    &self.a,
                    &self.b,
                    &self.z,
                    &self.delta_out,
                    &self.gated,
                    seq,
                    self.eps,
                );
                // Ending capture is mandatory even when an adapter rejects a
                // launch. Capture records but does not execute the sequence,
                // so launch it once for this token before caching it.
                let graph_result = backend.end_capture();
                capture_result?;
                let graph = graph_result?;
                graph.replay()?;
                self.exact_gdn_graphs[l] = Some(ExactGdnGraph::new(graph, &run, self));
            }
        } else {
            launch_exact_gdn(
                ctx,
                &mut run,
                &self.qkv,
                &self.qk_scratch,
                &self.a,
                &self.b,
                &self.z,
                &self.delta_out,
                &self.gated,
                seq,
                self.eps,
            )?;
        }
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
        let norm_ready = false;
        if l == 0 {
            trace_buf("gpu_x", &self.x, seq * self.hidden);
        }
        self.run_mlp(
            l, &run.gate, &run.up, run.gate_up_concat.as_ref(), &run.down,
            &run.post_norm_w, seq, norm_ready,
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
            qkv_concat: layer.qkv_concat.as_ref().map(gemm_clone),
            gate_up_concat: layer.gate_up_concat.as_ref().map(gemm_clone),
        }
    }
    fn run_full(
        &mut self,
        l: usize,
        seq: usize,
        start_pos: u32,
        position: CudaDeviceAddress,
    ) -> Result<()> {
        let _attention_range = apxinf_cuda::nvtx::range_static(b"Qwen/attention\0");
        let backend = self.backend.clone();
        let cuda_backend = backend
            .as_any()
            .downcast_ref::<CudaBackend>()
            .expect("qwen3_5 CUDA backend downcast");
        let ctx = cuda_backend.context();
        let mut run = self.take_full(l);
        kernels::norm::rms_into(
            ctx, DType::BF16, &self.x, &run.in_norm_w, &self.normed,
            self.hidden, seq, self.eps,
        )?;
        if l == 3 {
            trace_buf("f3_normed", &self.normed, seq * self.hidden);
        }
        let combined_decode = seq == 1 && run.qkv_concat.is_some();
        if combined_decode {
            gemm_run(
                ctx, run.qkv_concat.as_ref().expect("checked combined QKV"),
                &self.normed, &self.full_qkv_proj, seq, &self.dense_scratch,
            )?;
        } else if !gemm_run_marlin_batch(
            ctx,
            [(&run.q, &self.q_gate), (&run.k, &self.k_buf), (&run.v, &self.v_buf)],
            &self.normed, seq, &self.dense_scratch,
        )? && !gemm_run_multi(
            ctx, [&run.q, &run.k, &run.v], None, &self.normed,
            [&self.q_gate, &self.k_buf, &self.v_buf], None, seq,
        )? {
            gemm_run(ctx, &run.q, &self.normed, &self.q_gate, seq, &self.dense_scratch)?;
            if !gemm_run_marlin_batch(
                ctx, [(&run.k, &self.k_buf), (&run.v, &self.v_buf)],
                &self.normed, seq, &self.dense_scratch,
            )? && !gemm_run_prefill_pair(
                ctx, &run.k, &run.v, &self.normed, &self.k_buf, &self.v_buf,
                seq, &self.dense_scratch,
            )? && !gemm_run_pair(
                ctx, &run.k, &run.v, &self.normed, &self.k_buf, &self.v_buf, seq,
                self.pair_graph_enabled.then_some((cuda_backend, &mut self.pair_graphs[l * 3 + 1])),
            )? {
                gemm_run(ctx, &run.k, &self.normed, &self.k_buf, seq, &self.dense_scratch)?;
                gemm_run(ctx, &run.v, &self.normed, &self.v_buf, seq, &self.dense_scratch)?;
            }
        }
        let q_cols = self.n_heads * self.head_dim * 2;
        let kv_cols = self.n_kv_heads * self.head_dim;
        let combined_q;
        let combined_k;
        let combined_v;
        let (q_gate, k_buf, v_buf) = if combined_decode {
            combined_q = self.full_qkv_proj.view(0, q_cols * 2).map_err(Error::Cuda)?;
            combined_k = self.full_qkv_proj.view(q_cols * 2, kv_cols * 2).map_err(Error::Cuda)?;
            combined_v = self.full_qkv_proj
                .view((q_cols + kv_cols) * 2, kv_cols * 2).map_err(Error::Cuda)?;
            (&combined_q, &combined_k, &combined_v)
        } else {
            (&self.q_gate, &self.k_buf, &self.v_buf)
        };
        if seq == 1 {
            kernels::qwen35::qk_norm_rope_append(
                ctx, q_gate, &run.q_norm_w, k_buf, &run.k_norm_w,
                &self.q_buf, &self.gate_buf, &mut run.k_cache, seq,
                self.n_heads, self.n_kv_heads, self.head_dim, self.rotary_dim,
                self.rope_theta, position, MAX_SEQ_LEN,
            )?;
        } else {
            kernels::qwen35::q_split_norm_rope(
                ctx, q_gate, &run.q_norm_w, &self.q_buf, &self.gate_buf,
                seq, self.n_heads, self.head_dim, self.rotary_dim,
                self.rope_theta, start_pos,
            )?;
            kernels::qwen35::k_norm_rope_append(
                ctx, k_buf, &run.k_norm_w, &mut run.k_cache, seq,
                self.n_kv_heads, self.head_dim, self.rotary_dim,
                self.rope_theta, start_pos, MAX_SEQ_LEN,
            )?;
        }
        if l == 3 {
            trace_buf("f3_q", &self.q_buf, seq * self.n_heads * self.head_dim);
            trace_buf("f3_v", v_buf, seq * self.n_kv_heads * self.head_dim);
            trace_buf("f3_gate", &self.gate_buf, seq * self.n_heads * self.head_dim);
        }

        // Append V before attention reads it.
        if seq == 1 {
            kernels::cache::append_at(
                ctx, DType::BF16, &run.v_cache, v_buf, self.n_kv_heads,
                self.head_dim, MAX_SEQ_LEN, position,
            )?;
        } else {
            let v_tensor = v_buf.clone().into_tensor(
                Shape::from(vec![seq, self.n_kv_heads, self.head_dim]), DType::BF16,
            );
            kernels::cache::append(
                ctx, &run.v_cache, &v_tensor, self.n_kv_heads, self.head_dim,
                MAX_SEQ_LEN, start_pos as usize, seq,
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
        let gqa_group = std::env::var("APXINF_GQA_GROUP")
            .map_or(Ok(6usize), |value| value.parse::<usize>().map_err(|_| ()))
            .map_err(|_| Error::Other("APXINF_GQA_GROUP must be 0, 2, 3, or 6".into()))?;
        if seq == 1
            && gqa_group != 0
            && self.n_heads == 24 && self.n_kv_heads == 4 && self.head_dim == 256
        {
            kernels::qwen35::flash_decode_gated_256_gqa(
                ctx, &self.q_buf, &run.k_cache, &run.v_cache, &self.gate_buf,
                &self.attn, &self.attn_partials,
                1.0 / (self.head_dim as f32).sqrt(), position, MAX_SEQ_LEN,
                gqa_group,
            )?;
        } else if seq == 1
            && std::env::var("APXINF_FLASH_SPLIT_256_1W").map_or(true, |value| value != "0")
            && self.n_heads == 24 && self.n_kv_heads == 4 && self.head_dim == 256
        {
            kernels::qwen35::flash_decode_gated_256_split_1w(
                ctx, &self.q_buf, &run.k_cache, &run.v_cache, &self.gate_buf,
                &self.attn, &self.attn_partials,
                1.0 / (self.head_dim as f32).sqrt(), position, MAX_SEQ_LEN,
            )?;
        } else if seq == 1
            && std::env::var_os("APXINF_FLASH_SPLIT_256_2W").is_some_and(|value| value == "1")
            && self.n_heads == 24 && self.n_kv_heads == 4 && self.head_dim == 256
        {
            kernels::qwen35::flash_decode_gated_256_split_2w(
                ctx, &self.q_buf, &run.k_cache, &run.v_cache, &self.gate_buf,
                &self.attn, &self.attn_partials,
                1.0 / (self.head_dim as f32).sqrt(), position, MAX_SEQ_LEN,
            )?;
        } else if seq == 1
            && std::env::var("APXINF_FLASH_SPLIT_256").map_or(true, |value| value != "0")
            && self.n_heads == 24 && self.n_kv_heads == 4 && self.head_dim == 256
        {
            kernels::qwen35::flash_decode_gated_256_split(
                ctx, &self.q_buf, &run.k_cache, &run.v_cache, &self.gate_buf,
                &self.attn, &self.attn_partials,
                1.0 / (self.head_dim as f32).sqrt(), position, MAX_SEQ_LEN,
            )?;
        } else if seq == 1
            && std::env::var("APXINF_FLASH_DECODE_256").map_or(true, |value| value != "0")
            && self.n_heads == 24 && self.n_kv_heads == 4 && self.head_dim == 256
        {
            kernels::qwen35::flash_decode_gated_256(
                ctx, &self.q_buf, &run.k_cache, &run.v_cache, &self.gate_buf,
                &self.attn, self.n_heads, self.n_kv_heads, self.head_dim,
                1.0 / (self.head_dim as f32).sqrt(), position, MAX_SEQ_LEN,
            )?;
            if l == 3 {
                trace_buf("f3_attn_pre", &self.attn, seq * self.n_heads * self.head_dim);
            }
        }
        else if seq == 1 {
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
            // Full-attention projections have completed on this stream, and
            // o_proj/MLP do not start until the sequence below is enqueued.
            // Reuse the compact dequant scratch as five disjoint attention
            // views. One query head at a time bounds scores to one
            // [CHUNK, visible] matrix without changing that head's
            // q@k^T -> causal softmax -> p@v operation order.
            let score_bytes = seq
                .checked_mul(visible)
                .and_then(|value| value.checked_mul(4))
                .ok_or_else(|| Error::Other("attention score workspace overflow".into()))?;
            let kt_bytes = self
                .head_dim
                .checked_mul(visible)
                .and_then(|value| value.checked_mul(2))
                .ok_or_else(|| Error::Other("attention Kt workspace overflow".into()))?;
            let v_tile_rows = visible.min(V_TILE_ROWS);
            let vf32_bytes = v_tile_rows * self.head_dim * 4;
            let pv_bytes = seq
                .checked_mul(self.n_heads)
                .and_then(|value| value.checked_mul(self.head_dim))
                .and_then(|value| value.checked_mul(4))
                .ok_or_else(|| Error::Other("attention PV workspace overflow".into()))?;
            let l_bytes = seq
                .checked_mul(self.n_heads)
                .and_then(|value| value.checked_mul(4))
                .ok_or_else(|| Error::Other("attention L workspace overflow".into()))?;
            let kt_offset = score_bytes;
            let vf32_offset = kt_offset
                .checked_add(kt_bytes)
                .ok_or_else(|| Error::Other("attention arena offset overflow".into()))?;
            let pv_offset = vf32_offset
                .checked_add(vf32_bytes)
                .ok_or_else(|| Error::Other("attention arena offset overflow".into()))?;
            let l_offset = pv_offset
                .checked_add(pv_bytes)
                .ok_or_else(|| Error::Other("attention arena offset overflow".into()))?;
            let arena_bytes = l_offset
                .checked_add(l_bytes)
                .ok_or_else(|| Error::Other("attention arena size overflow".into()))?;
            if arena_bytes > self.dense_scratch.len() {
                // Retain the established GEMM path for every base-score cell.
                // At longer contexts its O(queries * visible) workspace no
                // longer fits, so use the existing allocation-free causal
                // flash kernel without changing KV representation or RoPE.
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
            } else {
            let scores = self
                .dense_scratch
                .view(0, score_bytes)
                .map_err(Error::Cuda)?;
            let kt = self
                .dense_scratch
                .view(kt_offset, kt_bytes)
                .map_err(Error::Cuda)?;
            let pv = self
                .dense_scratch
                .view(pv_offset, pv_bytes)
                .map_err(Error::Cuda)?;
            let vf32 = self
                .dense_scratch
                .view(vf32_offset, vf32_bytes)
                .map_err(Error::Cuda)?;
            let l_sums = self
                .dense_scratch
                .view(l_offset, l_bytes)
                .map_err(Error::Cuda)?;

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
                kernels::qwen35::transpose_kt(ctx, &k_view, &kt, visible, self.head_dim)?;
                for local_head in 0..per_kv {
                    let head = kv * per_kv + local_head;
                    kernels::qwen35::attention_gqa_dot(
                        ctx,
                        &self.q_buf,
                        &kt,
                        &scores,
                        head,
                        seq,
                        visible,
                        self.n_heads,
                        self.head_dim,
                        visible,
                    )?;
                    kernels::qwen35::attention_softmax_rows(
                        ctx,
                        &scores,
                        &l_sums,
                        head,
                        seq,
                        1,
                        visible,
                        visible,
                        start_pos,
                        1.0 / (self.head_dim as f32).sqrt(),
                    )?;
                    let mut tile_start = 0usize;
                    while tile_start < visible {
                        let tile_rows = (visible - tile_start).min(v_tile_rows);
                        let v_offset = tile_start * self.head_dim * 2;
                        let v_tile = v_view
                            .view(v_offset, tile_rows * self.head_dim * 2)
                            .map_err(Error::Cuda)?;
                        let p_offset = tile_start * 4;
                        kernels::qwen35::v_to_f32(
                            ctx,
                            &v_tile,
                            &vf32,
                            tile_rows,
                            self.head_dim,
                        )?;
                        let p_tile = scores
                            .view(p_offset, scores.len() - p_offset)
                            .map_err(Error::Cuda)?;
                        kernels::qwen35::attention_gqa_pv(
                            ctx,
                            &p_tile,
                            &vf32,
                            &pv,
                            head,
                            seq,
                            tile_rows,
                            self.n_heads,
                            self.head_dim,
                            visible,
                            if tile_start == 0 { 0.0 } else { 1.0 },
                        )?;
                        tile_start += tile_rows;
                    }
                }
            }
            if l == 3 {
                trace_buf_f32("f3g_scores_f32", &scores, seq * visible);
                trace_buf_f32("f3g_pv_f32", &pv, seq * self.n_heads * self.head_dim);
            }
            // The fused launch is scoped to the profiled Qwen3.5 head shape.
            // Other model shapes retain the established two-kernel path.
            if self.head_dim == 256 {
                kernels::qwen35::scale_out_gated(
                    ctx,
                    &pv,
                    &l_sums,
                    &self.gate_buf,
                    &self.attn,
                    seq,
                    self.n_heads,
                    self.head_dim,
                )?;
            } else {
                kernels::qwen35::scale_out(
                    ctx,
                    &pv,
                    &l_sums,
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
            }
            if l == 3 {
                trace_buf("f3_attn_pre", &self.attn, seq * self.n_heads * self.head_dim);
            }
            }
        }
        gemm_run(ctx, &run.o, &self.attn, &self.attn2, seq, &self.dense_scratch)?;
        if l == 3 {
            trace_buf("f3_attn2", &self.attn2, seq * self.hidden);
        }
        kernels::elementwise::add_into(
            ctx, DType::BF16, &self.x, &self.attn2, &self.x, seq * self.hidden,
        )?;
        let norm_ready = false;
        self.run_mlp(
            l, &run.gate, &run.up, run.gate_up_concat.as_ref(), &run.down,
            &run.post_norm_w, seq, norm_ready,
        )?;
        if let CudaLayer::Full(layer) = &mut self.layers[l] {
            layer.k_cache = run.k_cache;
            layer.v_cache = run.v_cache;
        }
        Ok(())
    }

    /// MLP shared by both layer kinds: norm, gate/up GEMMs, SiLU·up, down.
    fn run_mlp(
        &mut self,
        l: usize,
        gate: &Gemm,
        up: &Gemm,
        gate_up: Option<&Gemm>,
        down: &Gemm,
        post_norm_w: &CudaBuffer,
        seq: usize,
        norm_ready: bool,
    ) -> Result<()> {
        let _mlp_range = apxinf_cuda::nvtx::range_static(b"Qwen/MLP\0");
        let backend = self.backend.clone();
        let cuda_backend = backend.as_any().downcast_ref::<CudaBackend>()
            .expect("qwen3_5 CUDA backend downcast");
        let ctx = cuda_backend.context();
        if !norm_ready {
            kernels::norm::rms_into(
                ctx, DType::BF16, &self.x, post_norm_w, &self.normed2,
                self.hidden, seq, self.eps,
            )?;
        }
        if seq == 1
            && std::env::var_os("APXINF_MLP_FUSED_RAW").is_some_and(|value| value == "1")
            && gate.in_cols == 5120
            && gate.out_cols == 17408
            && up.in_cols == gate.in_cols
            && up.out_cols == gate.out_cols
            && gate.packed.as_ref().is_some_and(|packed| {
                packed.layout == W4DeviceLayout::RawCompressedTensors && packed.groups == 160
            })
            && up.packed.as_ref().is_some_and(|packed| {
                packed.layout == W4DeviceLayout::RawCompressedTensors && packed.groups == 160
            })
        {
            let gate = gate.packed.as_ref().expect("fused gate preflight");
            let up = up.packed.as_ref().expect("fused up preflight");
            kernels::quantization::matmul_bf16_w4a16_gate_up_silu(
                ctx, &self.normed2,
                &gate.w, &gate.scale, &gate.zp,
                &up.w, &up.scale, &up.zp,
                &self.mlp_act, 5120, 17408, 160,
            )?;
            gemm_run(ctx, down, &self.mlp_act, &self.attn2, seq, &self.dense_scratch)?;
            return kernels::elementwise::add_into(
                ctx, DType::BF16, &self.x, &self.attn2, &self.x, self.hidden,
            );
        }
        let combined_gate;
        let combined_up;
        let (gate_out, up_out) = if seq == 1 && gate_up.is_some() {
            gemm_run(ctx, gate_up.expect("checked gate/up"), &self.normed2,
                &self.gate_up_proj, seq, &self.dense_scratch)?;
            combined_gate = self.gate_up_proj.view(0, self.intermediate * 2).map_err(Error::Cuda)?;
            combined_up = self.gate_up_proj.view(self.intermediate * 2, self.intermediate * 2).map_err(Error::Cuda)?;
            (&combined_gate, &combined_up)
        } else {
            if !gemm_run_marlin_batch(
                ctx, [(gate, &self.gate_proj), (up, &self.up_proj)],
                &self.normed2, seq, &self.dense_scratch,
            )? && !gemm_run_prefill_pair(
                ctx, gate, up, &self.normed2, &self.gate_proj, &self.up_proj,
                seq, &self.dense_scratch,
            )? && !gemm_run_pair(
                ctx, gate, up, &self.normed2, &self.gate_proj, &self.up_proj, seq,
                self.pair_graph_enabled.then_some((cuda_backend, &mut self.pair_graphs[l * 3 + 2])),
            )? {
                gemm_run(ctx, gate, &self.normed2, &self.gate_proj, seq, &self.dense_scratch)?;
                gemm_run(ctx, up, &self.normed2, &self.up_proj, seq, &self.dense_scratch)?;
            }
            (&self.gate_proj, &self.up_proj)
        };
        kernels::qwen35::silu_mul(
            ctx, gate_out, up_out, &self.mlp_act, seq * self.intermediate,
        )?;
        gemm_run(ctx, down, &self.mlp_act, &self.attn2, seq, &self.dense_scratch)?;
        kernels::elementwise::add_into(
            ctx, DType::BF16, &self.x, &self.attn2, &self.x, seq * self.hidden,
        )
    }

    fn upload_decode_controls(
        &mut self,
        inputs: &[u32],
        positions: &[u32],
    ) -> Result<()> {
        if inputs.len() != positions.len() || inputs.len() > 8 {
            return Err(Error::Other(
                "decode controls require equally sized token/position blocks of at most 8".into(),
            ));
        }
        let bytes = &mut self.ids_host[..16 * 4];
        bytes.fill(0);
        for (index, value) in inputs.iter().enumerate() {
            let offset = index * 4;
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        for (index, value) in positions.iter().enumerate() {
            let offset = (8 + index) * 4;
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        self.decode_control
            .copy_from_host(bytes)
            .map_err(Error::Cuda)
    }

    /// Upload token ids through reusable host staging.
    fn upload_ids(&mut self, ids: &[u32]) -> Result<()> {
        let bytes = &mut self.ids_host[..ids.len() * 4];
        for (dst, id) in bytes.chunks_exact_mut(4).zip(ids) {
            dst.copy_from_slice(&id.to_le_bytes());
        }
        self.ids.copy_from_host(bytes).map_err(Error::Cuda)
    }

    fn upload_pos(&mut self, pos: u32) -> Result<()> {
        self.pos
            .copy_from_host(&pos.to_le_bytes())
            .map_err(Error::Cuda)
    }

    fn select_token_into(&self) -> Result<()> {
        if self.argmax_single_launch_enabled {
            kernels::qwen35::argmax_bf16_single_launch(
                self.ctx(), &self.logits, self.vocab, &self.argmax_partials,
                &self.argmax_arrivals, &self.argmax_out,
            )?;
        } else {
            kernels::qwen35::argmax_bf16_parallel(
                self.ctx(), &self.logits, self.vocab, &self.argmax_partials, &self.argmax_out,
            )?;
        }
        if let Some(event) = &self.argmax_ready {
            event.record(self.ctx().stream()).map_err(Error::Cuda)?;
        }
        Ok(())
    }

    fn wait_for_selected_token(&self) -> Result<()> {
        if self.decode_event_audit_enabled {
            // The token is produced on the model's sole stream. Waiting on that
            // stream preserves the same host-read dependency without recording
            // a completion event for every decode step.
            self.ctx().synchronize().map_err(Error::Cuda)
        } else if let Some(event) = &self.argmax_ready {
            event.synchronize().map_err(Error::Cuda)
        } else {
            self.ctx().synchronize().map_err(Error::Cuda)
        }
    }

    fn final_logits_into(&mut self, seq: usize) -> Result<()> {
        let _lm_head_range = apxinf_cuda::nvtx::range_static(b"Qwen/LM head\0");
        let ctx = self.ctx();
        let last_row = self.x.view((seq - 1) * self.hidden * 2, self.hidden * 2)
            .map_err(Error::Cuda)?;
        kernels::norm::rms_into(
            ctx, DType::BF16, &last_row, &self.final_norm_w, &self.normed,
            self.hidden, 1, self.eps,
        )?;

        gemm_run(
            ctx,
            &self.lm_head,
            &self.normed,
            &self.logits,
            1,
            &self.dense_scratch,
        )
    }

    /// Final RMSNorm (last row) + lm_head, returning `[1, vocab]` logits.
    fn final_logits(&mut self, seq: usize) -> Result<Tensor> {
        self.final_logits_into(seq)?;
        Ok(self.logits.clone().into_tensor(Shape::from(vec![1, self.vocab]), DType::BF16))
    }
}

impl Drop for Qwen35Cuda {
    fn drop(&mut self) {
        // The gated stream-wait path still exposes mapped host storage to the
        // argmax kernel, but intentionally has no completion event to drain.
        // Synchronize before fields are released if an earlier CUDA error
        // returned between launch and the normal wait.
        if self.argmax_ready.is_some() || self.decode_event_audit_enabled {
            let _ = self.ctx().synchronize();
        }
    }
}
fn gemm_clone(gemm: &Gemm) -> Gemm {
    Gemm {
        packed: gemm.packed.as_ref().map(|p| GemmPacked {
            w: p.w.clone(),
            scale: p.scale.clone(),
            zp: p.zp.clone(),
            groups: p.groups,
            layout: p.layout,
            padded_out_cols: p.padded_out_cols,
            padded_in_cols: p.padded_in_cols,
            source_n_offset: p.source_n_offset,
        }),
        dense: gemm.dense.clone(),
        out_cols: gemm.out_cols,
        in_cols: gemm.in_cols,
    }
}

/// Batch two to four independently weighted Marlin projections that consume
/// the same activation. Mixed layouts and incompatible physical K strides use
/// the established per-projection paths.
fn gemm_run_marlin_batch<const N: usize>(
    ctx: &CudaContext,
    projections: [(&Gemm, &CudaBuffer); N],
    act: &CudaBuffer,
    seq: usize,
    scratch: &CudaBuffer,
) -> Result<bool> {
    if seq != 1 {
        return Ok(false);
    }
    if !(2..=4).contains(&N) {
        return Ok(false);
    }
    let Some((first_gemm, _)) = projections.first() else {
        return Ok(false);
    };
    let Some(first_packed) = first_gemm.packed.as_ref() else {
        return Ok(false);
    };
    if first_packed.layout != W4DeviceLayout::MarlinAwqU4G32V1
        || projections.iter().any(|(gemm, _)| {
            gemm.in_cols != first_gemm.in_cols
                || gemm.packed.as_ref().is_none_or(|packed| {
                    packed.layout != W4DeviceLayout::MarlinAwqU4G32V1
                        || packed.padded_in_cols != first_packed.padded_in_cols
                })
        })
    {
        return Ok(false);
    }
    let batch = projections.map(|(gemm, output)| {
        let packed = gemm.packed.as_ref().expect("Marlin batch preflight");
        kernels::quantization::MarlinAwqU4G32V1Projection {
            marlin_qweight: &packed.w,
            scales: &packed.scale,
            zero_points: &packed.zp,
            output,
            logical_n: gemm.out_cols,
            padded_n: packed.padded_out_cols,
        }
    });
    kernels::quantization::matmul_bf16_marlin_awq_u4_g32_v1_batch_into(
        ctx,
        act,
        &batch,
        scratch,
        seq,
        first_gemm.in_cols,
        first_packed.padded_in_cols,
    )?;
    Ok(true)
}
/// Dispatch three or four compatible single-row packed projections through the
/// shared-weight staged decode kernel. Returns `false` for dense, prefill,
/// explicitly disabled, or unsupported quantized geometries.
fn gemm_run_multi(
    ctx: &CudaContext,
    projections: [&Gemm; 3],
    fourth: Option<&Gemm>,
    act: &CudaBuffer,
    outputs: [&CudaBuffer; 3],
    fourth_out: Option<&CudaBuffer>,
    seq: usize,
) -> Result<bool> {
    if std::env::var_os("APXINF_W4_MULTI").is_some_and(|value| value == "0") {
        return Ok(false);
    }
    let [first, second, third] = projections;
    let [first_out, second_out, third_out] = outputs;
    let (Some(first_packed), Some(second_packed), Some(third_packed)) =
        (&first.packed, &second.packed, &third.packed)
    else {
        return Ok(false);
    };
    let fourth_packed = match fourth {
        Some(gemm) => match &gemm.packed {
            Some(packed) => Some(packed),
            None => return Ok(false),
        },
        None => None,
    };
    if [first_packed, second_packed, third_packed]
        .into_iter()
        .chain(fourth_packed)
        .any(|packed| packed.layout != W4DeviceLayout::RawCompressedTensors)
    {
        return Ok(false);
    }
    let common_geometry = [second, third]
        .into_iter()
        .chain(fourth)
        .all(|gemm| gemm.in_cols == first.in_cols);
    let common_groups = [second_packed, third_packed]
        .into_iter()
        .chain(fourth_packed)
        .all(|packed| packed.groups == first_packed.groups);
    let valid_outputs = [first, second, third]
        .into_iter()
        .chain(fourth)
        .all(|gemm| gemm.out_cols != 0 && gemm.out_cols % 8 == 0);
    if seq != 1
        || !common_geometry
        || !common_groups
        || !valid_outputs
        || fourth.is_some() != fourth_out.is_some()
        || first.in_cols % 128 != 0
        || first_packed.groups == 0
        || first.in_cols % first_packed.groups != 0
        || first.in_cols / first_packed.groups != 32
    {
        return Ok(false);
    }
    let fourth_projection = match (fourth_packed, fourth, fourth_out) {
        (Some(packed), Some(gemm), Some(output)) => Some((
            &packed.w,
            &packed.scale,
            &packed.zp,
            output,
            gemm.out_cols,
        )),
        (None, None, None) => None,
        _ => return Ok(false),
    };
    kernels::quantization::matmul_bf16_w4a16_asym_tc_multi(
        ctx,
        act,
        (&first_packed.w, &first_packed.scale, &first_packed.zp, first_out, first.out_cols),
        (&second_packed.w, &second_packed.scale, &second_packed.zp, second_out, second.out_cols),
        (&third_packed.w, &third_packed.scale, &third_packed.zp, third_out, third.out_cols),
        fourth_projection,
        first.in_cols,
        first_packed.groups,
    )?;
    Ok(true)
}

/// Co-launch raw W4 dequantization for an eligible multi-row projection pair,
/// then retain the established independent cuBLAS calls and BF16 stores.
fn gemm_run_prefill_pair(
    ctx: &CudaContext,
    first: &Gemm,
    second: &Gemm,
    act: &CudaBuffer,
    first_out: &CudaBuffer,
    second_out: &CudaBuffer,
    seq: usize,
    scratch: &CudaBuffer,
) -> Result<bool> {
    if !std::env::var_os("APXINF_PREFILL_BATCH").is_some_and(|value| value == "1")
        || seq <= 1
    {
        return Ok(false);
    }
    let (Some(first_packed), Some(second_packed)) = (&first.packed, &second.packed) else {
        return Ok(false);
    };
    if first_packed.layout != W4DeviceLayout::RawCompressedTensors
        || second_packed.layout != W4DeviceLayout::RawCompressedTensors
        || first.in_cols != second.in_cols
        || first_packed.groups != second_packed.groups
    {
        return Ok(false);
    }
    kernels::quantization::try_matmul_bf16_w4a16_asym_prefill_pair_into(
        ctx,
        act,
        &first_packed.w,
        &first_packed.scale,
        &first_packed.zp,
        first_out,
        first.out_cols,
        &second_packed.w,
        &second_packed.scale,
        &second_packed.zp,
        second_out,
        second.out_cols,
        scratch,
        seq,
        first.in_cols,
        first_packed.groups,
    )
}

/// Dispatch two compatible single-row packed GEMMs.
/// Returns `false` when the ordinary per-GEMM path is required.
fn gemm_run_pair(
    ctx: &CudaContext,
    first: &Gemm,
    second: &Gemm,
    act: &CudaBuffer,
    first_out: &CudaBuffer,
    second_out: &CudaBuffer,
    seq: usize,
    pair_graph: Option<(&CudaBackend, &mut Option<PairW4Graph>)>,
) -> Result<bool> {
    let (Some(first_packed), Some(second_packed)) = (&first.packed, &second.packed) else {
        return Ok(false);
    };
    if first_packed.layout != W4DeviceLayout::RawCompressedTensors
        || second_packed.layout != W4DeviceLayout::RawCompressedTensors
    {
        return Ok(false);
    }
    if seq != 1
        || first.in_cols != second.in_cols
        || first_packed.groups != second_packed.groups
        || first.in_cols % 128 != 0
        || first_packed.groups == 0
        || first.in_cols % first_packed.groups != 0
        || (first.in_cols / first_packed.groups) % 16 != 0
    {
        return Ok(false);
    }
    if std::env::var_os("APXINF_W4_PAIR_COARSEN").is_some_and(|value| value == "1")
        && first_packed.groups != 0
        && first.in_cols % 128 == 0
        && first.in_cols % first_packed.groups == 0
        && first.in_cols / first_packed.groups == 32
        && first.out_cols % 128 == 0
        && second.out_cols % 128 == 0
    {
        kernels::quantization::matmul_bf16_w4a16_asym_tc_pair_coarsen(
            ctx,
            act,
            &first_packed.w,
            &first_packed.scale,
            &first_packed.zp,
            first_out,
            first.out_cols,
            &second_packed.w,
            &second_packed.scale,
            &second_packed.zp,
            second_out,
            second.out_cols,
            first.in_cols,
            first_packed.groups,
        )?;
        return Ok(true);
    }
    if std::env::var_os("APXINF_W4_PAIR_ALT").is_some_and(|value| value == "1")
        && first_packed.groups != 0
        && first.in_cols % 128 == 0
        && first.in_cols % first_packed.groups == 0
        && first.in_cols / first_packed.groups == 32
        && first.out_cols % 32 == 0
        && second.out_cols % 32 == 0
    {
        kernels::quantization::matmul_bf16_w4a16_asym_tc_pair_alt(
            ctx,
            act,
            &first_packed.w,
            &first_packed.scale,
            &first_packed.zp,
            first_out,
            first.out_cols,
            &second_packed.w,
            &second_packed.scale,
            &second_packed.zp,
            second_out,
            second.out_cols,
            first.in_cols,
            first_packed.groups,
        )?;
        return Ok(true);
    }
    if std::env::var_os("APXINF_STORE_ALT").is_some_and(|value| value == "1")
        && first_packed.groups != 0
        && first.in_cols % 128 == 0
        && first.in_cols % first_packed.groups == 0
        && first.in_cols / first_packed.groups == 32
        && first.out_cols % 64 == 0
        && second.out_cols % 64 == 0
        && first_out.address().is_aligned(4)
        && second_out.address().is_aligned(4)
    {
        kernels::quantization::matmul_bf16_w4a16_asym_tc_store_alt(
            ctx,
            act,
            &first_packed.w,
            &first_packed.scale,
            &first_packed.zp,
            first_out,
            first.out_cols,
            &second_packed.w,
            &second_packed.scale,
            &second_packed.zp,
            second_out,
            second.out_cols,
            first.in_cols,
            first_packed.groups,
        )?;
        return Ok(true);
    }
    if std::env::var_os("APXINF_W4_PAIR_2W").is_some_and(|value| value == "1")
        && first_packed.groups != 0
        && first.in_cols % 128 == 0
        && first.in_cols % first_packed.groups == 0
        && first.in_cols / first_packed.groups == 32
        && first.out_cols % 16 == 0
        && second.out_cols % 16 == 0
    {
        kernels::quantization::matmul_bf16_w4a16_asym_tc_pair_2w(
            ctx,
            act,
            &first_packed.w,
            &first_packed.scale,
            &first_packed.zp,
            first_out,
            first.out_cols,
            &second_packed.w,
            &second_packed.scale,
            &second_packed.zp,
            second_out,
            second.out_cols,
            first.in_cols,
            first_packed.groups,
        )?;
        return Ok(true);
    }
    if std::env::var_os("APXINF_W4_PAIR_6W").is_some_and(|value| value == "1")
        && first_packed.groups != 0
        && first.in_cols % 128 == 0
        && first.in_cols % first_packed.groups == 0
        && first.in_cols / first_packed.groups == 32
        && first.out_cols % 8 == 0
        && second.out_cols % 8 == 0
    {
        kernels::quantization::matmul_bf16_w4a16_asym_tc_pair_6w(
            ctx,
            act,
            &first_packed.w,
            &first_packed.scale,
            &first_packed.zp,
            first_out,
            first.out_cols,
            &second_packed.w,
            &second_packed.scale,
            &second_packed.zp,
            second_out,
            second.out_cols,
            first.in_cols,
            first_packed.groups,
        )?;
        return Ok(true);
    }
    if first.out_cols % 64 != 0 || second.out_cols % 64 != 0 {
        return Ok(false);
    }
    if std::env::var("APXINF_W4_PAIR_PREFETCH").map_or(true, |value| value != "0")
        && first_packed.groups != 0
        && first.in_cols % 128 == 0
        && first.in_cols % first_packed.groups == 0
        && first.in_cols / first_packed.groups == 32
        && act.address().is_aligned(16)
    {
        kernels::quantization::matmul_bf16_w4a16_asym_tc_pair_prefetch(
            ctx,
            act,
            &first_packed.w,
            &first_packed.scale,
            &first_packed.zp,
            first_out,
            first.out_cols,
            &second_packed.w,
            &second_packed.scale,
            &second_packed.zp,
            second_out,
            second.out_cols,
            first.in_cols,
            first_packed.groups,
        )?;
        return Ok(true);
    }
    let pair_warp_geometry = first_packed.groups != 0
        && first.in_cols % first_packed.groups == 0
        && first.in_cols / first_packed.groups == 32
        && matches!(first.in_cols, 5120 | 6144 | 17408)
        && matches!(first.out_cols, 1024 | 5120 | 6144 | 10240 | 12288 | 17408)
        && matches!(second.out_cols, 1024 | 5120 | 6144 | 10240 | 12288 | 17408);
    if std::env::var_os("APXINF_W4_PAIR_WARP").is_some_and(|value| value == "1")
        && pair_warp_geometry
    {
        kernels::quantization::matmul_bf16_w4a16_asym_tc_pair_warp(
            ctx,
            act,
            &first_packed.w,
            &first_packed.scale,
            &first_packed.zp,
            first_out,
            first.out_cols,
            &second_packed.w,
            &second_packed.scale,
            &second_packed.zp,
            second_out,
            second.out_cols,
            first.in_cols,
            first_packed.groups,
        )?;
        return Ok(true);
    }
    let pair_occupancy_geometry = first_packed.groups != 0
        && first.in_cols % 128 == 0
        && first.in_cols % first_packed.groups == 0
        && first.in_cols / first_packed.groups == 32
        && matches!(first.in_cols, 5120 | 6144 | 17408)
        && matches!(first.out_cols, 1024 | 5120 | 6144 | 10240 | 12288 | 17408)
        && matches!(second.out_cols, 1024 | 5120 | 6144 | 10240 | 12288 | 17408);
    if std::env::var_os("APXINF_PAIR_OCCUPANCY").is_some_and(|value| value == "1")
        && pair_occupancy_geometry
    {
        kernels::quantization::matmul_bf16_w4a16_asym_tc_pair_occupancy(
            ctx,
            act,
            &first_packed.w,
            &first_packed.scale,
            &first_packed.zp,
            first_out,
            first.out_cols,
            &second_packed.w,
            &second_packed.scale,
            &second_packed.zp,
            second_out,
            second.out_cols,
            first.in_cols,
            first_packed.groups,
        )?;
        return Ok(true);
    }
    if std::env::var_os("APXINF_W4_PAIR_REG").is_some_and(|value| value == "1")
        && first_packed.groups != 0
        && first.in_cols % 128 == 0
        && first.in_cols % first_packed.groups == 0
        && first.in_cols / first_packed.groups == 32
    {
        kernels::quantization::matmul_bf16_w4a16_asym_tc_pair_reg(
            ctx,
            act,
            &first_packed.w,
            &first_packed.scale,
            &first_packed.zp,
            first_out,
            first.out_cols,
            &second_packed.w,
            &second_packed.scale,
            &second_packed.zp,
            second_out,
            second.out_cols,
            first.in_cols,
            first_packed.groups,
        )?;
        return Ok(true);
    }
    let tile_alt_geometry = first_packed.groups != 0
        && first.in_cols % 128 == 0
        && first.in_cols % first_packed.groups == 0
        && first.in_cols / first_packed.groups == 32
        && matches!(first.in_cols, 5120 | 6144 | 17408)
        && matches!(first.out_cols, 1024 | 5120 | 6144 | 10240 | 12288 | 17408)
        && matches!(second.out_cols, 1024 | 5120 | 6144 | 10240 | 12288 | 17408);
    let separate_outputs_fit = first
        .out_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .is_some_and(|bytes| first_out.len() >= bytes)
        && second
            .out_cols
            .checked_mul(DType::BF16.size_in_bytes())
            .is_some_and(|bytes| second_out.len() >= bytes);
    if std::env::var_os("APXINF_W4_PAIR_SEPARATE").is_some_and(|value| value == "1")
        && tile_alt_geometry
        && separate_outputs_fit
    {
        kernels::quantization::matmul_bf16_w4a16_asym_tc_tile_alt(
            ctx,
            act,
            &first_packed.w,
            &first_packed.scale,
            &first_packed.zp,
            first_out,
            first.in_cols,
            first.out_cols,
            first_packed.groups,
        )?;
        kernels::quantization::matmul_bf16_w4a16_asym_tc_tile_alt(
            ctx,
            act,
            &second_packed.w,
            &second_packed.scale,
            &second_packed.zp,
            second_out,
            second.in_cols,
            second.out_cols,
            second_packed.groups,
        )?;
        return Ok(true);
    }
    if std::env::var_os("APXINF_W4_VECTOR_MMA").is_some_and(|value| value == "1")
        && first_packed.groups != 0
        && first.in_cols % 128 == 0
        && first.in_cols % first_packed.groups == 0
        && first.in_cols / first_packed.groups == 32
        && matches!(first.in_cols, 5120 | 6144 | 17408)
        && matches!(first.out_cols, 1024 | 5120 | 6144 | 10240 | 12288 | 17408)
        && matches!(second.out_cols, 1024 | 5120 | 6144 | 10240 | 12288 | 17408)
    {
        kernels::quantization::matmul_bf16_w4a16_asym_tc_pair_vector_mma(
            ctx,
            act,
            &first_packed.w,
            &first_packed.scale,
            &first_packed.zp,
            first_out,
            first.out_cols,
            &second_packed.w,
            &second_packed.scale,
            &second_packed.zp,
            second_out,
            second.out_cols,
            first.in_cols,
            first_packed.groups,
        )?;
        return Ok(true);
    }
    let pair_act_aligned = act.address().is_aligned(16);
    if std::env::var_os("APXINF_W4_PAIR_ACT").is_some_and(|value| value == "1")
        && first_packed.groups != 0
        && first.in_cols % 128 == 0
        && first.in_cols % first_packed.groups == 0
        && first.in_cols / first_packed.groups == 32
        && first.out_cols % 64 == 0
        && second.out_cols % 64 == 0
        && pair_act_aligned
    {
        kernels::quantization::matmul_bf16_w4a16_asym_tc_pair_act(
            ctx,
            act,
            &first_packed.w,
            &first_packed.scale,
            &first_packed.zp,
            first_out,
            first.out_cols,
            &second_packed.w,
            &second_packed.scale,
            &second_packed.zp,
            second_out,
            second.out_cols,
            first.in_cols,
            first_packed.groups,
        )?;
        return Ok(true);
    }
    if std::env::var_os("APXINF_W4_PAIR_SHARED").is_some_and(|value| value == "1")
        && first_packed.groups != 0
        && first.in_cols % 128 == 0
        && first.in_cols % first_packed.groups == 0
        && first.in_cols / first_packed.groups == 32
        && first.out_cols % 64 == 0
        && second.out_cols % 64 == 0
    {
        kernels::quantization::matmul_bf16_w4a16_asym_tc_pair_shared(
            ctx,
            act,
            &first_packed.w,
            &first_packed.scale,
            &first_packed.zp,
            first_out,
            first.out_cols,
            &second_packed.w,
            &second_packed.scale,
            &second_packed.zp,
            second_out,
            second.out_cols,
            first.in_cols,
            first_packed.groups,
        )?;
        return Ok(true);
    }
    if std::env::var_os("APXINF_W4_PAIR_REUSE").is_some_and(|value| value == "1")
        && first_packed.groups != 0
        && first.in_cols % 128 == 0
        && first.in_cols % first_packed.groups == 0
        && first.in_cols / first_packed.groups == 32
        && first.out_cols % 64 == 0
        && second.out_cols % 64 == 0
    {
        kernels::quantization::matmul_bf16_w4a16_asym_tc_pair_reuse(
            ctx,
            act,
            &first_packed.w,
            &first_packed.scale,
            &first_packed.zp,
            first_out,
            first.out_cols,
            &second_packed.w,
            &second_packed.scale,
            &second_packed.zp,
            second_out,
            second.out_cols,
            first.in_cols,
            first_packed.groups,
        )?;
        return Ok(true);
    }

    let pair_cache_aligned = act.address().is_aligned(2)
        && first_packed.w.address().is_aligned(4)
        && second_packed.w.address().is_aligned(4)
        && first_packed.scale.address().is_aligned(2)
        && second_packed.scale.address().is_aligned(2)
        && first_packed.zp.address().is_aligned(4)
        && second_packed.zp.address().is_aligned(4);
    let weight_stage_geometry = first_packed.groups != 0
        && first.in_cols % 128 == 0
        && first.in_cols % first_packed.groups == 0
        && first.in_cols / first_packed.groups == 32
        && first.out_cols % 64 == 0
        && second.out_cols % 64 == 0;
    let use_weight_stage = std::env::var("APXINF_W4_WEIGHT_STAGE")
        .map_or(true, |value| value != "0")
        && weight_stage_geometry;
    let launch_pair = || -> Result<()> {
        if use_weight_stage {
            kernels::quantization::matmul_bf16_w4a16_asym_tc_pair_weight_stage(
                ctx,
                act,
                &first_packed.w,
                &first_packed.scale,
                &first_packed.zp,
                first_out,
                first.out_cols,
                &second_packed.w,
                &second_packed.scale,
                &second_packed.zp,
                second_out,
                second.out_cols,
                first.in_cols,
                first_packed.groups,
            )
        } else {
            kernels::quantization::matmul_bf16_w4a16_asym_tc_pair(
                ctx,
                act,
                &first_packed.w,
                &first_packed.scale,
                &first_packed.zp,
                first_out,
                first.out_cols,
                &second_packed.w,
                &second_packed.scale,
                &second_packed.zp,
                second_out,
                second.out_cols,
                first.in_cols,
                first_packed.groups,
            )
        }
    };
    if use_weight_stage && pair_graph.is_none() {
        launch_pair()?;
        return Ok(true);
    }

    if std::env::var_os("APXINF_W4_PAIR_CACHE").is_some_and(|value| value == "1")
        && first_packed.groups != 0
        && first.in_cols % 128 == 0
        && first.in_cols % first_packed.groups == 0
        && first.in_cols / first_packed.groups == 32
        && pair_cache_aligned
    {
        kernels::quantization::matmul_bf16_w4a16_asym_tc_pair_cache(
            ctx,
            act,
            &first_packed.w,
            &first_packed.scale,
            &first_packed.zp,
            first_out,
            first.out_cols,
            &second_packed.w,
            &second_packed.scale,
            &second_packed.zp,
            second_out,
            second.out_cols,
            first.in_cols,
            first_packed.groups,
        )?;
        return Ok(true);
    }
    if std::env::var_os("APXINF_W4_PAIR_META").is_some_and(|value| value == "1")
        && first_packed.groups != 0
        && first.in_cols % 128 == 0
        && first.in_cols % first_packed.groups == 0
        && first.in_cols / first_packed.groups == 32
    {
        kernels::quantization::matmul_bf16_w4a16_asym_tc_pair_meta(
            ctx,
            act,
            &first_packed.w,
            &first_packed.scale,
            &first_packed.zp,
            first_out,
            first.out_cols,
            &second_packed.w,
            &second_packed.scale,
            &second_packed.zp,
            second_out,
            second.out_cols,
            first.in_cols,
            first_packed.groups,
        )?;
    } else if let Some((backend, graph_slot)) = pair_graph {
        if let Some(graph) = graph_slot.as_ref() {
            graph.replay()?;
        } else if std::env::var_os("APXINF_GEMM_PROF").is_none()
            && std::env::var_os("APXINF_KERNEL_PROF").is_none()
            && std::env::var_os("APXINF_TRACE").is_none()
        {
            if backend.begin_capture_relaxed().is_err() {
                launch_pair()?;
            } else {
                let capture_result = launch_pair();
                let graph_result = backend.end_capture();
                // Capture records but does not execute. On any capture failure,
                // issue one eager launch so the normal path remains reachable.
                if capture_result.is_err() || graph_result.is_err() {
                    let _ = capture_result;
                    let _ = graph_result;
                    launch_pair()?;
                } else {
                    let graph = graph_result.expect("checked graph capture result");
                    capture_result.expect("checked pair capture result");
                    graph.replay()?;
                    *graph_slot = Some(PairW4Graph::new(
                        graph,
                        act,
                        &first_packed.w,
                        &first_packed.scale,
                        &first_packed.zp,
                        first_out,
                        &second_packed.w,
                        &second_packed.scale,
                        &second_packed.zp,
                        second_out,
                    ));
                }
            }
        } else {
            launch_pair()?;
        }
    } else {
        launch_pair()?;
    }
    Ok(true)
}

/// Dispatch a GEMM. Marlin-packed weights keep standalone Marlin for decode;
/// prefill reverses the persistent layout into dense BF16 scratch for cuBLAS.
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
    if packed.layout == W4DeviceLayout::MarlinAwqU4G32V1 {
        if seq > 1 {
            let dense_prefill_bytes = gemm
                .out_cols
                .checked_mul(gemm.in_cols)
                .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
                .ok_or_else(|| Error::Other("Marlin dense prefill size overflow".into()))?;
            if (seq == 512 || dense_prefill_bytes > scratch.len())
                && packed.source_n_offset == 0
                && std::env::var("APXINF_DIRECT_MARLIN_PREFILL")
                    .map_or(true, |value| value != "0")
            {
                let bf16 = DType::BF16.size_in_bytes();
                let act_bytes = seq * packed.padded_in_cols * bf16;
                let out_bytes = seq * packed.padded_out_cols * bf16;
                let lock_bytes = ctx.caps().multiprocessor_count as usize
                    * std::mem::size_of::<i32>();
                let output_offset = act_bytes;
                let lock_offset = output_offset + out_bytes;
                let activation_scratch = scratch.view(0, act_bytes).map_err(Error::Cuda)?;
                let output_scratch = scratch.view(output_offset, out_bytes).map_err(Error::Cuda)?;
                let workspace = scratch.view(lock_offset, lock_bytes).map_err(Error::Cuda)?;
                return kernels::quantization::matmul_bf16_marlin_awq_u4_g32_v1_into(
                    ctx, act, &packed.w, &packed.scale, &packed.zp, out,
                    &activation_scratch, &output_scratch, &workspace, seq,
                    gemm.out_cols, gemm.in_cols, packed.padded_out_cols,
                    packed.padded_in_cols,
                );
            }
            return kernels::quantization::matmul_bf16_marlin_awq_u4_g32_v1_prefill_into(
                ctx,
                act,
                &packed.w,
                &packed.scale,
                &packed.zp,
                scratch,
                out,
                seq,
                gemm.out_cols,
                gemm.in_cols,
                packed.padded_out_cols,
                packed.padded_in_cols,
                packed.source_n_offset,
            );
        }

        let bf16 = DType::BF16.size_in_bytes();
        let act_bytes = seq
            .checked_mul(packed.padded_in_cols)
            .and_then(|v| v.checked_mul(bf16))
            .ok_or_else(|| Error::Other("Marlin activation scratch overflow".into()))?;
        let out_bytes = seq
            .checked_mul(packed.padded_out_cols)
            .and_then(|v| v.checked_mul(bf16))
            .ok_or_else(|| Error::Other("Marlin output scratch overflow".into()))?;
        let lock_bytes = ctx.caps().multiprocessor_count as usize
            * std::mem::size_of::<i32>();
        let output_offset = act_bytes;
        let lock_offset = output_offset
            .checked_add(out_bytes)
            .ok_or_else(|| Error::Other("Marlin scratch offset overflow".into()))?;
        let activation_scratch = scratch.view(0, act_bytes).map_err(Error::Cuda)?;
        let output_scratch = scratch.view(output_offset, out_bytes).map_err(Error::Cuda)?;
        let workspace = scratch.view(lock_offset, lock_bytes).map_err(Error::Cuda)?;
        return kernels::quantization::matmul_bf16_marlin_awq_u4_g32_v1_into(
            ctx, act, &packed.w, &packed.scale, &packed.zp, out,
            &activation_scratch, &output_scratch, &workspace, seq,
            gemm.out_cols, gemm.in_cols, packed.padded_out_cols,
            packed.padded_in_cols,
        );
    }
    if std::env::var_os("APXINF_W4_PREFILL_NATIVE").is_some_and(|value| value == "1")
        && (2..=kernels::quantization::W4A16_PREFILL_NATIVE_MAX_ROWS).contains(&seq)
        && packed.layout == W4DeviceLayout::RawCompressedTensors
        && kernels::quantization::try_matmul_bf16_w4a16_asym_prefill_native_into(
            ctx,
            act,
            &packed.w,
            &packed.scale,
            &packed.zp,
            out,
            seq,
            gemm.in_cols,
            gemm.out_cols,
            packed.groups,
        )?
    {
        return Ok(());
    }
    if std::env::var_os("APXINF_PREFILL_PACKED").is_some_and(|value| value == "1")
        && seq > 1
        && seq <= 256
        && kernels::quantization::try_matmul_bf16_w4a16_asym_prefill_packed_into(
            ctx,
            act,
            &packed.w,
            &packed.scale,
            &packed.zp,
            out,
            seq,
            gemm.in_cols,
            gemm.out_cols,
            packed.padded_out_cols,
            packed.groups,
            matches!(packed.layout, W4DeviceLayout::RepackedN64K16V1 | W4DeviceLayout::TransformCacheN64K16V1),
        )?
    {
        return Ok(());
    }
    if seq == 1 && packed.layout == W4DeviceLayout::RepackedN64K16V1 {
        return kernels::quantization::matmul_bf16_w4a16_asym_tc_repacked_v1(
            ctx, act, &packed.w, &packed.scale, &packed.zp, out,
            gemm.in_cols, gemm.out_cols, packed.padded_out_cols, packed.groups,
        );
    }
    if std::env::var_os("APXINF_W4_TRANSFORM_CACHE").is_some_and(|value| value == "1")
        && seq == 1
        && packed.layout == W4DeviceLayout::TransformCacheN64K16V1
        && gemm.in_cols % 128 == 0
        && gemm.out_cols % 64 == 0
        && packed.groups != 0
        && gemm.in_cols % packed.groups == 0
        && gemm.in_cols / packed.groups == 32
    {
        return kernels::quantization::matmul_bf16_w4a16_asym_tc_w4_transform_cache(
            ctx, act, &packed.w, &packed.scale, &packed.zp, out,
            gemm.in_cols, gemm.out_cols, packed.padded_out_cols, packed.groups,
        );
    }
    if seq == 1 && packed.layout == W4DeviceLayout::TransformCacheN64K16V1 {
        return kernels::quantization::matmul_bf16_w4a16_asym_tc_repacked_v1(
            ctx, act, &packed.w, &packed.scale, &packed.zp, out,
            gemm.in_cols, gemm.out_cols, packed.padded_out_cols, packed.groups,
        );
    }
    if std::env::var_os("APXINF_PREFILL_FAST").is_some_and(|value| value == "1")
        && seq > 1 && seq <= 256
        && kernels::quantization::try_matmul_bf16_w4a16_asym_prefill_fast_into(
            ctx,
            act,
            &packed.w,
            &packed.scale,
            &packed.zp,
            out,
            seq,
            gemm.in_cols,
            gemm.out_cols,
            packed.padded_out_cols,
            packed.groups,
            matches!(packed.layout, W4DeviceLayout::RepackedN64K16V1 | W4DeviceLayout::TransformCacheN64K16V1),
        )?
    {
        return Ok(());
    }
    let cache_hint_geometry = matches!(gemm.in_cols, 5120 | 6144 | 17408)
        && matches!(gemm.out_cols, 1024 | 5120 | 6144 | 10240 | 12288 | 17408)
        && packed.groups == gemm.in_cols / 32;
    if std::env::var_os("APXINF_W4_CACHE_HINT").is_some_and(|value| value == "1")
        && seq == 1
        && packed.layout == W4DeviceLayout::RawCompressedTensors
        && cache_hint_geometry
    {
        return kernels::quantization::matmul_bf16_w4a16_asym_tc_cache_hint(
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

    let vector_mma_geometry = matches!(gemm.in_cols, 5120 | 6144 | 17408)
        && matches!(gemm.out_cols, 1024 | 5120 | 6144 | 10240 | 12288 | 17408)
        && packed.groups == gemm.in_cols / 32;
    if std::env::var_os("APXINF_W4_VECTOR_MMA").is_some_and(|value| value == "1")
        && seq == 1
        && packed.layout == W4DeviceLayout::RawCompressedTensors
        && vector_mma_geometry
    {
        return kernels::quantization::matmul_bf16_w4a16_asym_tc_vector_mma(
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
    let scale_epilogue_geometry = matches!(gemm.in_cols, 5120 | 6144 | 17408)
        && matches!(gemm.out_cols, 1024 | 5120 | 6144 | 10240 | 12288 | 17408)
        && packed.groups == gemm.in_cols / 32;
    if std::env::var_os("APXINF_W4_SCALE_EPILOGUE").is_some_and(|value| value == "1")
        && seq == 1
        && packed.layout == W4DeviceLayout::RawCompressedTensors
        && scale_epilogue_geometry
    {
        return kernels::quantization::matmul_bf16_w4a16_asym_tc_scale_epilogue(
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

    if std::env::var_os("APXINF_STORE_ALT").is_some_and(|value| value == "1")
        && seq == 1
        && packed.layout == W4DeviceLayout::RawCompressedTensors
        && gemm.in_cols % 128 == 0
        && gemm.out_cols % 64 == 0
        && packed.groups != 0
        && gemm.in_cols % packed.groups == 0
        && gemm.in_cols / packed.groups == 32
        && out.address().is_aligned(4)
    {
        return kernels::quantization::matmul_bf16_w4a16_asym_tc_store_alt_single(
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
    let persistent_geometry = matches!(gemm.in_cols, 5120 | 6144 | 17408)
        && matches!(gemm.out_cols, 1024 | 5120 | 6144 | 10240 | 12288 | 17408)
        && packed.groups == gemm.in_cols / 32;
    if std::env::var_os("APXINF_W4_PERSISTENT").is_some_and(|value| value == "1")
        && seq == 1
        && packed.layout == W4DeviceLayout::RawCompressedTensors
        && persistent_geometry
    {
        return kernels::quantization::matmul_bf16_w4a16_asym_tc_persistent(
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
    let tile_alt_geometry = matches!(gemm.in_cols, 5120 | 6144 | 17408)
        && matches!(gemm.out_cols, 1024 | 5120 | 6144 | 10240 | 12288 | 17408)
        && packed.groups == gemm.in_cols / 32;
    if std::env::var("APXINF_W4_TILE_ALT").map_or(true, |value| value != "0")
        && seq == 1
        && packed.layout == W4DeviceLayout::RawCompressedTensors
        && tile_alt_geometry
    {
        return kernels::quantization::matmul_bf16_w4a16_asym_tc_tile_alt(
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
    if std::env::var("APXINF_W4_META_SHARED").map_or(true, |value| value != "0")
        && seq == 1
        && packed.layout == W4DeviceLayout::RawCompressedTensors
        && gemm.in_cols % 128 == 0
        && gemm.out_cols % 64 == 0
        && packed.groups != 0
        && gemm.in_cols % packed.groups == 0
        && gemm.in_cols / packed.groups == 32
    {
        return kernels::quantization::matmul_bf16_w4a16_asym_tc_meta_shared(
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
    if seq == 1
        && packed.layout == W4DeviceLayout::RawCompressedTensors
        && gemm.in_cols % 128 == 0
        && gemm.out_cols % 64 == 0
        && packed.groups != 0
        && gemm.in_cols % packed.groups == 0
        && (gemm.in_cols / packed.groups) % 16 == 0
    {
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
    if seq == 1
        && packed.layout == W4DeviceLayout::RawCompressedTensors
        && gemm.in_cols % 8 == 0
        && packed.groups != 0
        && gemm.in_cols % packed.groups == 0
        && (gemm.in_cols / packed.groups) % 8 == 0
        && gemm.out_cols % 128 == 0
    {
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
    if matches!(packed.layout, W4DeviceLayout::RepackedN64K16V1 | W4DeviceLayout::TransformCacheN64K16V1) {
        return kernels::quantization::matmul_bf16_w4a16_asym_prefill_repacked_v1_into(
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
            packed.padded_out_cols,
            packed.groups,
        );
    }

    if packed.layout == W4DeviceLayout::RawCompressedTensors
        && gemm.out_cols * gemm.in_cols <= 17408 * 5120
    {
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
    let repacked_suffix = W4_REPACKED_N64_K16_V1_SUFFIX;
    let cache_suffix = W4_TRANSFORM_CACHE_SUFFIX;
    let marlin_suffix = W4_MARLIN_AWQ_U4_G32_V1_SUFFIX;
    let marlin_name = format!("{prefix}.weight_packed.{marlin_suffix}");
    let marlin_scale_name = format!("{prefix}.weight_scale.{marlin_suffix}");
    let marlin_zp_name = format!("{prefix}.weight_zero_point.{marlin_suffix}");
    let repacked_name = format!("{prefix}.weight_packed.{repacked_suffix}");
    let repacked_scale_name = format!("{prefix}.weight_scale.{repacked_suffix}");
    let repacked_zp_name = format!("{prefix}.weight_zero_point.{repacked_suffix}");
    let cache_name = format!("{prefix}.weight_packed.{cache_suffix}");
    let cache_scale_name = format!("{prefix}.weight_scale.{cache_suffix}");
    let cache_zp_name = format!("{prefix}.weight_zero_point.{cache_suffix}");
    let raw_name = format!("{prefix}.weight_packed");
    let raw_scale_name = format!("{prefix}.weight_scale");
    let raw_zp_name = format!("{prefix}.weight_zero_point");
    let cache_requested = std::env::var_os("APXINF_W4_TRANSFORM_CACHE")
        .is_some_and(|value| value == "1");
    let repacked_requested = std::env::var_os("APXINF_QWEN35_W4_REPACKED")
        .is_some_and(|value| value == "1");
    let marlin_requested = std::env::var("APXINF_MARLIN").map_or(true, |value| value != "0");
    let raw_complete = [&raw_name, &raw_scale_name, &raw_zp_name]
        .iter().all(|name| device_tensors.contains_key(*name));
    let repacked_names = [&repacked_name, &repacked_scale_name, &repacked_zp_name];
    let cache_names = [&cache_name, &cache_scale_name, &cache_zp_name];
    let marlin_names = [&marlin_name, &marlin_scale_name, &marlin_zp_name];
    let repacked_complete = repacked_names.iter().all(|name| device_tensors.contains_key(*name));
    let cache_complete = cache_names.iter().all(|name| device_tensors.contains_key(*name));
    let marlin_complete = marlin_names.iter().all(|name| device_tensors.contains_key(*name));
    let repacked_present = repacked_names.iter().any(|name| device_tensors.contains_key(*name));
    let cache_present = cache_names.iter().any(|name| device_tensors.contains_key(*name));
    let marlin_present = marlin_names.iter().any(|name| device_tensors.contains_key(*name));
    let (packed_name, scale_name, zp_name, layout, padded_out_cols, padded_in_cols) =
        if marlin_requested && marlin_complete {
            let dims = device_tensors
                .get(&marlin_scale_name)
                .expect("complete Marlin tensor set")
                .shape()
                .dims();
            if dims.len() != 2 {
                return Err(Error::Other(format!("qwen3_5 CUDA: {prefix} Marlin scale must be [K/32,N]")));
            }
            let padded_k = dims[0] * 32;
            let padded_n = dims[1];
            (marlin_name, marlin_scale_name, marlin_zp_name,
             W4DeviceLayout::MarlinAwqU4G32V1, padded_n, padded_k)
        } else if cache_requested && cache_complete {
            (cache_name, cache_scale_name, cache_zp_name,
             W4DeviceLayout::TransformCacheN64K16V1, out_cols.div_ceil(64) * 64, in_cols)
        } else if repacked_requested && repacked_complete && raw_complete {
            (repacked_name, repacked_scale_name, repacked_zp_name,
             W4DeviceLayout::RepackedN64K16V1, out_cols.div_ceil(64) * 64, in_cols)
        } else if raw_complete {
            (raw_name, raw_scale_name, raw_zp_name,
             W4DeviceLayout::RawCompressedTensors, out_cols, in_cols)
        } else if device_tensors.contains_key(&raw_name) || repacked_present || cache_present || marlin_present {
            return Err(Error::Other(format!(
                "qwen3_5 CUDA: {prefix} W4 tensors lack a complete raw fallback"
            )));
        } else {
            return Ok(Gemm {
                packed: None,
                dense: Some(buffer_from(device_tensors, &format!("{prefix}.weight"))?),
                out_cols,
                in_cols,
            });
        };
    let scale = buffer_from(device_tensors, &scale_name)?;
    let groups = scale.len() / (padded_out_cols * 2);
    let expected_w = padded_out_cols
        .checked_mul(padded_in_cols.div_ceil(8))
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| Error::Other(format!("qwen3_5 CUDA: {prefix} packed byte overflow")))?;
    let expected_zp = if layout == W4DeviceLayout::MarlinAwqU4G32V1 {
        padded_out_cols.checked_mul(groups).and_then(|v| v.checked_div(2))
    } else {
        padded_out_cols.div_ceil(8).checked_mul(groups)
            .and_then(|elements| elements.checked_mul(4))
    }.ok_or_else(|| Error::Other(format!("qwen3_5 CUDA: {prefix} zero-point byte overflow")))?;
    let w = buffer_from(device_tensors, &packed_name)?;
    let zp = buffer_from(device_tensors, &zp_name)?;
    if groups == 0 || w.len() != expected_w || zp.len() != expected_zp {
        return Err(Error::Other(format!(
            "qwen3_5 CUDA: {prefix} {:?} W4 byte accounting mismatch",
            layout
        )));
    }
    Ok(Gemm {
        packed: Some(GemmPacked {
            w,
            scale,
            zp,

            groups,
            layout,
            padded_out_cols,
            padded_in_cols,
            source_n_offset: 0,
        }),
        dense: None,
        out_cols,
        in_cols,
    })
}
/// Build one physical Marlin GEMM and logical member handles sharing its
/// allocation. Non-Marlin or absent combined tensors leave existing paths.
fn marlin_concat_group(
    device_tensors: &HashMap<String, Tensor>,
    combined_prefix: &str,
    members: &[(String, usize)],
    in_cols: usize,
) -> Result<Option<(Gemm, Vec<Gemm>)>> {
    let suffix = W4_MARLIN_AWQ_U4_G32_V1_SUFFIX;
    if !device_tensors.contains_key(&format!("{combined_prefix}.weight_packed.{suffix}")) {
        return Ok(None);
    }
    let total_n = members.iter().map(|(_, n)| n).sum();
    let combined = gemm(device_tensors, combined_prefix, total_n, in_cols)?;
    let packed = combined.packed.as_ref()
        .ok_or_else(|| Error::Other("combined Marlin GEMM is not packed".into()))?;
    let mut offset = 0usize;
    let logical = members.iter().map(|(_, out_cols)| {
        let gemm = Gemm {
            packed: Some(GemmPacked {
                w: packed.w.clone(),
                scale: packed.scale.clone(),
                zp: packed.zp.clone(),
                groups: packed.groups,
                layout: packed.layout,
                padded_out_cols: packed.padded_out_cols,
                padded_in_cols: packed.padded_in_cols,
                source_n_offset: offset,
            }),
            dense: None,
            out_cols: *out_cols,
            in_cols,
        };
        offset += *out_cols;
        gemm
    }).collect();
    Ok(Some((combined, logical)))
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
