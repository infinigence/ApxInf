//! Qwen-Drive native model: Qwen3.5 hybrid VLM (gated-delta linear attention
//! + gated full attention with partial interleaved mRoPE) plus the planning
//! expert driver, on the CUDA kernel path.
//!
//! The hybrid cache is model-owned: full-attention layers append post-rotary
//! K/V into per-layer `[capacity, kv_heads, head_dim]` BF16 buffers (exactly
//! the post-rotary caches the planning expert reads), while GDN layers
//! advance a causal-conv state and an fp32 recurrent state. The type
//! implements `LlmTrait` for the maintained registry/AutoModel surface and
//! exposes the dedicated VQA / direct-planning / reasoning-planning flows
//! used by the PyO3 binding. All layer mathematics run on device.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use apxinf_core::{
    Backend, DType, Device, Error, NextTokenLogits, Result, RngKey, Shape, Tensor,
    TokenSamplingInit, TokenSamplingParams, TokenSamplingSpec,
};
use apxinf_loader::ModelConfig;

use crate::accelerator::create_backend;
use crate::llm_trait::{LlmCapabilities, LlmInput, LlmTrait};

use super::backend::{
    downcast_arc, kernels, transfers, tuning, Context, CublasTranspose, DeviceBuffer,
    RuntimeBackend,
};
use kernels::{activation, attention, elementwise, embedding, gemm, linear_attention as la};

use super::config::QwenDriveConfig;
use super::device_weights::{MixerWeights, QwenDriveDeviceWeights};
use super::expert::{self, ExpertPlan};
use super::planner::ExpertConditioning;
use super::vision;
use super::weights::{QwenDriveExpertWeights, QwenDriveVlmWeights};

const GDN_CHUNK: usize = 64;

/// Gate for the TEMP-DIAG development probes.
///
/// Several of those probes copy hidden state back to the host once per layer
/// during prefill, and every copy synchronizes the stream, so leaving them on
/// costs real decode throughput on top of the printing itself. They are
/// debugging scaffolding for the divergence hunt, not part of the model, and
/// nothing reads their output for control flow -- `diag_digest` is only ever
/// pushed to and printed. Off unless `APXINF_QWEN_DIAG` is set, so the probes
/// remain available verbatim when the hunt needs them.
pub(super) fn diagnostics_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("APXINF_QWEN_DIAG").is_some())
}

/// Attribute one stage of the GDN prefill scan.
///
/// The five scan kernels launch asynchronously, so attributing time between
/// them means synchronizing the stream at each boundary. That serializes the
/// launches and is not free, which is why every call site sits behind
/// [`diagnostics_enabled`].
fn gdn_stage_mark(
    ctx: &Context,
    layer_idx: usize,
    name: &str,
    since: &mut std::time::Instant,
) -> Result<()> {
    ctx.synchronize().map_err(Error::Cuda)?;
    eprintln!(
        "[qwen_drive] gdn_stage layer={} {} ms={:.2}",
        layer_idx,
        name,
        since.elapsed().as_secs_f64() * 1000.0
    );
    *since = std::time::Instant::now();
    Ok(())
}

/// Arena for the captured GDN decode bodies. One layer's decode intermediates
/// are well under a megabyte at seq 1; this leaves room without tuning.
const DECODE_GRAPH_ARENA_BYTES: usize = 64 * 1024 * 1024;

/// A tensor outside the graph arena, so its address survives the arena being
/// reused by the next captured layer.
fn persistent_tensor(ctx: &Context, shape: &[usize], dtype: DType) -> Result<Tensor> {
    let elements: usize = shape.iter().product();
    let bytes = elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| Error::Other("qwen_drive: persistent tensor size overflow".into()))?;
    DeviceBuffer::alloc(bytes.max(1), ctx.device_id())
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape.to_vec()), dtype)
        .map_err(Error::Cuda)
}

/// Stream-ordered device copy between two tensors of the same byte length.
fn device_copy(ctx: &Context, destination: &Tensor, source: &Tensor, bytes: usize) -> Result<()> {
    let dst = DeviceBuffer::from_tensor(destination).map_err(Error::Cuda)?;
    let src = DeviceBuffer::from_tensor(source).map_err(Error::Cuda)?;
    dst.copy_from_device_async(&src, bytes, ctx.stream())
        .map_err(Error::Cuda)
}

/// Whether decode may run the GDN layers from captured CUDA graphs.
///
/// Only the GDN layers qualify. The full-attention layers pass `cache_len` and
/// `kv_len` to their kernels as host scalars, and a capture bakes those in, so
/// a replay would write the same KV slot and attend over the same extent on
/// every step -- wrong without failing. GDN carries a fixed-size recurrent
/// state updated in place, and its conv state double-buffers, so one capture
/// per layer per parity covers it exactly.
/// `APXINF_QWEN_DECODE_GRAPH` selects which GDN layers take the captured path:
/// `all`, or a single layer index to isolate one while the rest stay eager.
/// Restricting it to one layer separates a fault inside a captured body from
/// one in how the bodies share the arena and the staging buffers.
fn decode_graph_layers() -> Option<Option<usize>> {
    static SETTING: std::sync::OnceLock<Option<Option<usize>>> = std::sync::OnceLock::new();
    *SETTING.get_or_init(|| match std::env::var("APXINF_QWEN_DECODE_GRAPH") {
        Err(_) => None,
        Ok(value) => match value.trim().parse::<usize>() {
            Ok(layer) => Some(Some(layer)),
            Err(_) => Some(None),
        },
    })
}

fn decode_graph_enabled_for(layer_idx: usize) -> bool {
    match decode_graph_layers() {
        None => false,
        Some(None) => true,
        Some(Some(only)) => only == layer_idx,
    }
}

/// Decode steps to run eagerly before capturing.
///
/// cuBLASLt plan preparation allocates, which is illegal inside a capture, and
/// `may_prepare_native_resources` reports false once a workspace is bound. Two
/// eager passes leave every plan on this path already built.
const DECODE_GRAPH_WARMUP_STEPS: usize = 2;

/// The decode step the divergence probes should fire at, from
/// `APXINF_QWEN_TRACE_DECODE_STEP`. Unset means no step.
fn trace_decode_step() -> Option<usize> {
    static STEP: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *STEP.get_or_init(|| {
        std::env::var("APXINF_QWEN_TRACE_DECODE_STEP")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
    })
}

// Private takeover checkpoint probe; opt-in and removed before product acceptance.
pub(super) fn trace_rows(name: &str, tensor: &Tensor) -> Result<()> {
    let Some(root) = std::env::var_os("APXINF_QWEN_TRACE_DIR") else { return Ok(()); };
    let width = *tensor.shape().dims().last().unwrap();
    let count = tensor.numel() / width;
    // Four sampled rows cannot tell a layer that is itself nondeterministic
    // from one that merely mixes in a neighbour's unsampled row, and every
    // sequence mixer here does mix across positions. APXINF_QWEN_TRACE_FULL
    // dumps the whole tensor so the first affected layer is the real one.
    let full = std::env::var_os("APXINF_QWEN_TRACE_FULL").is_some();
    let rows = if full || count < 4 {
        (0..count).collect::<Vec<_>>()
    } else {
        vec![0, 1, 2, count - 1]
    };
    let buffer = DeviceBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    let mut bytes = Vec::new();
    for row in rows {
        let view = buffer.view(row * width * tensor.dtype().size_in_bytes(), width * tensor.dtype().size_in_bytes()).map_err(Error::Cuda)?;
        let t = view.as_tensor(Shape::new(vec![1, width]), tensor.dtype()).map_err(Error::Cuda)?;
        let values = transfers::to_cpu(&t)?.to_f32_vec().map_err(|e| Error::Other(e.to_string()))?;
        for value in values { bytes.extend_from_slice(&value.to_le_bytes()); }
    }
    let root = Path::new(&root);
    std::fs::create_dir_all(root).map_err(|e| Error::Other(e.to_string()))?;
    std::fs::write(root.join(format!("{name}.f32")), bytes).map_err(|e| Error::Other(e.to_string()))?;
    Ok(())
}

// TEMP-DIAG (implement_r13): zero-vision ablation switch (vision_loader_r12 V3);
// when true the image-embedding scatter consumes a zeros tensor instead of vis.primary
// AFTER the vis_fp fingerprint has read the true tower output; deactivated at r14
// (question answered: vision content exonerated as the constant-198 driver; true-vision
// measurement required for comparability); revert in the acceptance-bound revision.
// RE-ACTIVATED at successor-mission implement_r2 (synthesis_r2 directive): one-revision
// receipt-test of the r13 exoneration record, which survives only as this comment; bundled
// with the step-1 + content-row logits discriminators in the same revision; revert in the
// acceptance-bound revision.
// DEACTIVATED at successor-mission implement_r3 (synthesis_r3 directive): the R2 receipt
// answered branch B2 (zero-vision ablation active with the vis_fp l2=1103.4933 sanity gate
// intact, constant-198 collapse persisted at steps 0/1 -- vision-content poisoning refuted);
// R3 restores true-vision geometry because the A1 scatter-fidelity and C1 attention-value
// discriminators need it; the flag itself is REMOVED in the acceptance-bound revision.
const DIAG_ZERO_VISION: bool = false;

fn bf16_round(value: f32) -> f32 {
    half::bf16::from_f32(value).to_f32()
}

fn device_tensor(ctx: &Context, shape: &[usize], dtype: DType) -> Result<Tensor> {
    let elements: usize = shape.iter().product();
    let bytes = elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| Error::Other("qwen_drive: tensor size overflow".into()))?;
    let buffer = kernels::scratch_buffer(ctx, bytes.max(1))?;
    buffer
        .as_tensor(Shape::new(shape.to_vec()), dtype)
        .map_err(Error::Cuda)
}

/// Allocate zeroed device scratch on the compute stream.
///
/// The GDN scan takes ten of these per layer, and `DeviceBuffer::alloc_zeros`
/// clears them with the blocking `cudaMemset`, so each layer forced the device
/// to drain several times over. The async form is stream-ordered against the
/// kernels that read the memory, which is the only ordering that matters here.
fn alloc_zeros(ctx: &Context, bytes: usize) -> Result<DeviceBuffer> {
    kernels::scratch_buffer_zeroed(ctx, bytes.max(1))
}

/// GDN scan scratch.
///
/// The scan pads the sequence up to a chunk boundary and the prep kernels only
/// reach the real tokens, so the padding tail has to start at zero. When there
/// is no padding -- every decode step, where seq_pad and seq are both 1 -- the
/// clear covers nothing and is pure cost: it accounted for most of the 14,064
/// memsets one VQA inference issued after operator outputs stopped being
/// cleared.
fn alloc_scan_scratch(ctx: &Context, bytes: usize, padded: bool) -> Result<DeviceBuffer> {
    if padded {
        alloc_zeros(ctx, bytes)
    } else {
        kernels::scratch_buffer(ctx, bytes.max(1))
    }
}

/// Apply a BF16 linear weight in its original checkpoint [out,in] layout.
/// Whether the projection GEMMs go through the tuned path.
///
/// `write_ex` is a raw cuBLAS call: no tactic lookup, no cuBLASLt plan, and
/// invisible to autotune. Every projection except gate_up used to take it,
/// which is why an autotune pass over this model wrote exactly one record.
/// Through `gemm::bf16` they are tunable, and the loader stores them as
/// `[in, out]` so the GEMM is row-major NN rather than NT.
///
/// It is not bit-exact: a different kernel is selected, so the BF16 reduction
/// order changes. Measured with prefill made deterministic, so the comparison
/// means something: identical through text layers 0-2, first difference at
/// layer 3 at 1.8e-3 relative, ULP-scale at the source and amplified by the
/// residual stream to 2.6e-2 by layer 31. Nothing observable moved -- the
/// 130-token VQA probe is identical, the gate still diverges at index 121 with
/// token 357, and both trajectory modes report the same error to every digit.
/// `APXINF_QWEN_LINEAR_TUNED=0` restores the raw path for anyone who needs the
/// old arithmetic exactly.
pub(super) fn tuned_projection() -> bool {
    static TUNED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TUNED.get_or_init(|| {
        !matches!(
            std::env::var("APXINF_QWEN_LINEAR_TUNED").as_deref(),
            Ok("0") | Ok("off") | Ok("false")
        )
    })
}

fn linear_checkpoint(ctx: &Context, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    // Under the gate the loader has already stored this weight as [in, out],
    // so the raw path's [out, in] check does not apply to it.
    if tuned_projection() {
        return gemm::bf16(ctx, input, weight);
    }
    let x = input.shape().dims();
    let w = weight.shape().dims();
    if x.len() != 2
        || w.len() != 2
        || x[1] != w[1]
        || input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
    {
        return Err(Error::Other(
            "qwen_drive: checkpoint linear shape/dtype mismatch".into(),
        ));
    }
    let stride =
        i32::try_from(x[1]).map_err(|_| Error::Other("linear input stride overflow".into()))?;
    let columns =
        i32::try_from(w[0]).map_err(|_| Error::Other("linear output stride overflow".into()))?;
    let output = device_tensor(ctx, &[x[0], w[0]], DType::BF16)?;
    gemm::write_ex(
        ctx,
        DType::BF16,
        CublasTranspose::None,
        CublasTranspose::Transpose,
        x[0],
        w[0],
        x[1],
        1.0,
        &DeviceBuffer::from_tensor(input).map_err(Error::Cuda)?,
        stride,
        &DeviceBuffer::from_tensor(weight).map_err(Error::Cuda)?,
        stride,
        0.0,
        &DeviceBuffer::from_tensor(&output).map_err(Error::Cuda)?,
        columns,
    )?;
    Ok(output)
}

fn project_and_pack(ctx: &Context, input: &Tensor, weights: &[&Tensor]) -> Result<Tensor> {
    let rows = input.shape().dims()[0];
    let mut outputs = Vec::with_capacity(weights.len());
    for weight in weights {
        let value = linear_checkpoint(ctx, input, weight)?;
        outputs.push(value.reshape(vec![rows, value.shape().dims()[1], 1, 1])?);
    }
    let packed = elementwise::concat_channels_bf16(ctx, &outputs.iter().collect::<Vec<_>>())?;
    packed.reshape(vec![rows, packed.shape().dims()[1]])
}

fn upload_u32(ctx: &Context, values: &[u32]) -> Result<DeviceBuffer> {
    let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_ne_bytes()).collect();
    let buffer = DeviceBuffer::alloc(bytes.len().max(1), ctx.device_id()).map_err(Error::Cuda)?;
    buffer.copy_from_host(&bytes).map_err(Error::Cuda)?;
    Ok(buffer)
}

/// `[kv_len, kv_heads, head_dim]` prefix view of a cache tensor.
fn cache_view(cache: &Tensor, kv_len: usize) -> Result<Tensor> {
    let dims = cache.shape().dims().to_vec();
    if dims.len() != 3 || kv_len == 0 || kv_len > dims[0] {
        return Err(Error::Other("qwen_drive: cache view shape mismatch".into()));
    }
    let buffer = DeviceBuffer::from_tensor(cache).map_err(Error::Cuda)?;
    let bytes = kv_len * dims[1] * dims[2] * DType::BF16.size_in_bytes();
    let view = buffer.view(0, bytes).map_err(Error::Cuda)?;
    view.as_tensor(Shape::new(vec![kv_len, dims[1], dims[2]]), DType::BF16)
        .map_err(Error::Cuda)
}

// TEMP-DIAG (implement_r3 / synthesis_r3, folded A2): hidden-row {0, rows-1} l2 pair for
// the layer_delta probe; revert in the acceptance-bound revision.
fn hidden_row_l2_pair(hidden: &Tensor) -> Result<(f64, f64)> {
    let dims = hidden.shape().dims().to_vec();
    let width = *dims.last().unwrap_or(&1);
    let rows = hidden.numel() / width.max(1);
    let row_bytes = width * DType::BF16.size_in_bytes();
    let buf = DeviceBuffer::from_tensor(hidden).map_err(Error::Cuda)?;
    let l2_row = |row: usize| -> Result<f64> {
        let view = buf.view(row * row_bytes, row_bytes).map_err(Error::Cuda)?;
        let rt = view.as_tensor(Shape::new(vec![1, width]), DType::BF16).map_err(Error::Cuda)?;
        let vals = transfers::to_cpu(&rt)?.to_f32_vec()
            .map_err(|e| Error::Other(format!("qwen_drive layer_delta: {e}")))?;
        Ok(vals.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt())
    };
    Ok((l2_row(0)?, l2_row(rows - 1)?))
}

enum LayerCache {
    FullAttention { k: Tensor, v: Tensor },
    Gdn {
        conv_state_a: Tensor,
        conv_state_b: Tensor,
        /// false -> a is current, b is scratch; true -> b is current.
        flip: bool,
        recurrent: Tensor,
    },
}

/// The native Qwen-Drive model (VLM + optional planning expert).
pub struct QwenDriveModel {
    config: QwenDriveConfig,
    backend: Arc<dyn Backend>,
    cuda: Arc<RuntimeBackend>,
    weights: QwenDriveDeviceWeights,
    caches: Vec<LayerCache>,
    cache_len: usize,
    rope_delta: i64,
    max_seq_len: usize,
    /// mRoPE position of the last processed token (all three axes equal for
    /// the text tokens that close every supported prompt).
    last_position: i64,
    // TEMP-DIAG (implement_r13): digest sink for the evidence-accessibility re-emission;
    // the captured r12 marker strings are re-printed just before decode_exit so the
    // decisive values land inside the receipt's trailing capture window; revert in the repair revision.
    diag_digest: Vec<String>,
    /// Decode step currently being executed, so the per-layer trace can fire at
    /// one step instead of overwriting itself on every one of them. `None`
    /// outside the decode loop.
    decode_step: Option<usize>,
    /// Captured GDN decode graphs, by layer and by the conv double-buffer
    /// parity the capture baked in.
    gdn_graphs: Vec<[Option<Box<dyn apxinf_core::Graph>>; 2]>,
    /// Whether one eager pass has already run under the arena for this layer
    /// and parity. cuBLASLt plans bind buffer addresses when they are prepared,
    /// and `may_prepare_native_resources` reports false once a workspace is
    /// bound unless the pass is a declared preflight, so a capture taken before
    /// that preflight records kernels set up against the driver-allocated
    /// buffers the eager steps used.
    gdn_graph_prepared: Vec<[bool; 2]>,
    /// Where a captured layer leaves its result: a workspace view, valid only
    /// until the arena is reused, so it is copied out after every replay.
    gdn_graph_out: Vec<[Option<Tensor>; 2]>,
    /// Fixed-address input a captured layer reads, and the per-layer landing
    /// buffer a replayed result is copied into.
    gdn_graph_in: Option<Tensor>,
    gdn_graph_result: Vec<Option<Tensor>>,
    /// Stable-address arena the captured bodies allocate from.
    gdn_graph_workspace: Option<kernels::GraphWorkspace>,
}

/// Install the CUDA GEMM tactic store for this device.
///
/// Every model reached through `AutoModel` picks up its tactic database in
/// `configure_cuda_tuning`, but `QwenDriveModel::load` builds its backend
/// directly and the Python binding calls it directly too, so these GEMMs ran
/// on cuBLASLt heuristics and never consulted
/// `configs/tuning/nvidia/<device>/tactics.json` at all.
///
/// `APXINF_QWEN_TUNING_DIR` overrides the search root, which otherwise matches
/// `AutoModel`: `configs/tuning`, resolved against the working directory. A
/// missing tactics file is not an error -- the kernels fall back to the same
/// heuristics they used before. `APXINF_QWEN_AUTOTUNE` selects online
/// autotuning, which measures the shapes real requests produce and writes them
/// back to that directory; leave it unset for inference.
fn configure_gemm_tuning(cuda: &RuntimeBackend) -> Result<()> {
    let root = std::env::var_os("APXINF_QWEN_TUNING_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("configs/tuning"));
    let paths = tuning::TuningPaths::resolve_for_cuda(
        &root,
        cuda.context().caps(),
        cuda.context().library_versions(),
    );
    let database = paths
        .tactics
        .is_file()
        .then(|| tuning::TuningDb::from_json_file(&paths.tactics))
        .transpose()?;
    let autotune = std::env::var_os("APXINF_QWEN_AUTOTUNE").is_some();
    // Installing an empty store is not the same as installing none: with no
    // record to match, the resolver pins each shape to its safe inference
    // fallback instead of letting cuBLASLt pick, which measured 54% slower on
    // Orin (19.3s vs 12.5s per 64-token VQA scene). So configure tuning only
    // when there is something to configure it with.
    if database.is_none() && !autotune {
        return Ok(());
    }
    let mode = if autotune {
        tuning::TuningMode::AutoTune
    } else {
        tuning::TuningMode::Inference
    };
    eprintln!(
        "[qwen_drive] gemm tuning: mode={:?} store={} loaded={}",
        mode,
        paths.tactics.display(),
        database.is_some()
    );
    gemm::configure_tuning(
        cuda.context(),
        mode,
        database.as_ref().map(std::slice::from_ref).unwrap_or(&[]),
        Some(paths),
    )
}

impl QwenDriveModel {
    /// Load the VLM (and optional planner head) onto a CUDA device.
    pub fn load(model_dir: &Path, planner_dir: Option<&Path>, device: Device) -> Result<Self> {
        let backend = create_backend(device)?;
        Self::load_with_backend(model_dir, planner_dir, backend)
    }

    pub fn load_with_backend(
        model_dir: &Path,
        planner_dir: Option<&Path>,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        let cuda = downcast_arc(backend.clone()).ok_or_else(|| {
            Error::Other(
                "qwen_drive requires the CUDA backend (native device execution); \
                 the CPU backend is not a deployment target"
                    .into(),
            )
        })?;
        configure_gemm_tuning(&cuda)?;
        let config = QwenDriveConfig::from_json_file(&model_dir.join("config.json"))?;
        eprintln!("[qwen_drive] loading VLM weights from {}", model_dir.display());
        let (tensors, _meta) = apxinf_loader::safetensors::load_native_path(model_dir)
            .map_err(|e| Error::Other(format!("qwen_drive: load VLM weights: {e}")))?;
        eprintln!("[qwen_drive] VLM safetensors loaded: {} tensors", tensors.len());
        let vlm = QwenDriveVlmWeights::from_map(tensors)?;
        eprintln!(
            "[qwen_drive] VLM weights classified: {} language tensors, {} visual tensors",
            vlm.language_tensor_count(),
            vlm.visual_tensor_count()
        );
        let expert = planner_dir
            .map(|dir| -> Result<QwenDriveExpertWeights> {
                let (tensors, _meta) = apxinf_loader::safetensors::load_native_path(dir)
                    .map_err(|e| Error::Other(format!("qwen_drive: load planner weights: {e}")))?;
                QwenDriveExpertWeights::from_map(&config, &tensors)
            })
            .transpose()?;
        let weights = QwenDriveDeviceWeights::from_maps(&config, vlm, expert, &*backend)?;
        eprintln!(
            "[qwen_drive] device weights resident (planner={}); allocating caches",
            weights.expert.is_some()
        );
        let max_seq_len = config.text.max_position_embeddings.min(16384); // FIX (implement_r5): 8192 < measured 10457-token VQA prompt (12x868 image tokens + text); 16384 covers the post-clamp-revert worst case 10457+2048=12505 under the config ceiling 32768 (+256MiB full-attn KV cache).
        let caches = Self::fresh_caches(&config, &cuda, max_seq_len)?;
        Ok(Self {
            config,
            backend,
            cuda,
            weights,
            caches,
            cache_len: 0,
            rope_delta: 0,
            max_seq_len,
            last_position: 0,
            diag_digest: Vec::new(),
            decode_step: None,
            gdn_graphs: Vec::new(),
            gdn_graph_prepared: Vec::new(),
            gdn_graph_out: Vec::new(),
            gdn_graph_in: None,
            gdn_graph_result: Vec::new(),
            gdn_graph_workspace: None,
        })
    }

    fn fresh_caches(
        config: &QwenDriveConfig,
        cuda: &RuntimeBackend,
        max_seq_len: usize,
    ) -> Result<Vec<LayerCache>> {
        let text = &config.text;
        let device = cuda.device_id();
        let mut caches = Vec::with_capacity(text.n_layers);
        for index in 0..text.n_layers {
            if text.is_full_attention(index) {
                let bytes = max_seq_len * text.n_kv_heads * text.head_dim * DType::BF16.size_in_bytes();
                let k = DeviceBuffer::alloc_zeros(bytes, device).map_err(Error::Cuda)?;
                let v = DeviceBuffer::alloc_zeros(bytes, device).map_err(Error::Cuda)?;
                let shape = Shape::new(vec![max_seq_len, text.n_kv_heads, text.head_dim]);
                caches.push(LayerCache::FullAttention {
                    k: k.as_tensor(shape.clone(), DType::BF16).map_err(Error::Cuda)?,
                    v: v.as_tensor(shape, DType::BF16).map_err(Error::Cuda)?,
                });
            } else {
                let conv_dim = 2 * text.linear_num_key_heads * text.linear_key_head_dim
                    + text.linear_num_value_heads * text.linear_value_head_dim;
                let conv_bytes = conv_dim * text.linear_conv_kernel_dim * DType::BF16.size_in_bytes();
                let conv_shape = Shape::new(vec![conv_dim, text.linear_conv_kernel_dim]);
                let conv_a = DeviceBuffer::alloc_zeros(conv_bytes, device).map_err(Error::Cuda)?;
                let conv_b = DeviceBuffer::alloc_zeros(conv_bytes, device).map_err(Error::Cuda)?;
                let rec_bytes = text.linear_num_value_heads
                    * text.linear_key_head_dim
                    * text.linear_value_head_dim
                    * DType::F32.size_in_bytes();
                let recurrent = DeviceBuffer::alloc_zeros(rec_bytes, device).map_err(Error::Cuda)?;
                caches.push(LayerCache::Gdn {
                    conv_state_a: conv_a
                        .as_tensor(conv_shape.clone(), DType::BF16)
                        .map_err(Error::Cuda)?,
                    conv_state_b: conv_b
                        .as_tensor(conv_shape, DType::BF16)
                        .map_err(Error::Cuda)?,
                    flip: false,
                    recurrent: recurrent
                        .as_tensor(
                            Shape::new(vec![
                                text.linear_num_value_heads,
                                text.linear_key_head_dim,
                                text.linear_value_head_dim,
                            ]),
                            DType::F32,
                        )
                        .map_err(Error::Cuda)?,
                });
            }
        }
        Ok(caches)
    }

    fn reset_state(&mut self) -> Result<()> {
        self.caches = Self::fresh_caches(&self.config, &self.cuda, self.max_seq_len)?;
        // Captured GDN graphs hold the addresses of the cache buffers they were
        // recorded against, and those buffers are about to be replaced. Left
        // in place they read and write freed memory: with the allocation cache
        // on, that memory is handed straight back out to the fresh caches, so
        // the replay corrupts live state instead of faulting. Drop them and let
        // the next sequence re-prepare and re-capture.
        self.gdn_graphs.clear();
        self.gdn_graph_out.clear();
        self.gdn_graph_result.clear();
        self.gdn_graph_prepared.clear();
        self.cache_len = 0;
        self.rope_delta = 0;
        self.last_position = 0;
        self.diag_digest.clear(); // TEMP-DIAG (implement_r13): per-generate digest sink.
        Ok(())
    }

    fn ctx(&self) -> &Context {
        self.cuda.context()
    }

    pub fn device(&self) -> Device {
        Device::Cuda(self.cuda.device_id())
    }

    pub fn has_planner(&self) -> bool {
        self.weights.expert.is_some()
    }

    pub fn config(&self) -> &QwenDriveConfig {
        &self.config
    }

    // ---- positions / rope tables ------------------------------------------

    /// Qwen3.5 `get_rope_index`: text tokens take continuing linear
    /// positions; image-token runs take 3D grid positions (T, H/merge,
    /// W/merge) offset by the running position, which then advances by
    /// `max(H, W) // merge`.
    fn rope_index(&self, token_ids: &[u32], grid_thw: &[[u32; 3]]) -> Result<Vec<[u32; 3]>> {
        let merge = self.config.vision.spatial_merge_size as u32;
        let image_tok = self.config.image_token_id;
        let mut out: Vec<[u32; 3]> = Vec::with_capacity(token_ids.len());
        let mut grids = grid_thw.iter();
        let mut current_pos: u32 = 0;
        let mut index = 0usize;
        while index < token_ids.len() {
            if token_ids[index] == image_tok {
                let start = index;
                while index < token_ids.len() && token_ids[index] == image_tok {
                    index += 1;
                }
                let run = index - start;
                let grid = grids.next().ok_or_else(|| {
                    Error::Other("qwen_drive: more image-token runs than image grids".into())
                })?;
                let (t, h, w) = (grid[0], grid[1] / merge, grid[2] / merge);
                if run != (t * h * w) as usize {
                    return Err(Error::Other(format!(
                        "qwen_drive: image-token run of {run} != grid tokens {}",
                        t * h * w
                    )));
                }
                for ti in 0..t {
                    for hi in 0..h {
                        for wi in 0..w {
                            out.push([current_pos + ti, current_pos + hi, current_pos + wi]);
                        }
                    }
                }
                current_pos += grid[1].max(grid[2]) / merge;
            } else {
                out.push([current_pos, current_pos, current_pos]);
                current_pos += 1;
                index += 1;
            }
        }
        if grids.next().is_some() {
            return Err(Error::Other(
                "qwen_drive: more image grids than image-token runs".into(),
            ));
        }
        Ok(out)
    }

    /// Interleaved mRoPE cos/sin tables for a position list, computed on the
    /// host in fp32 and rounded to bf16 on upload (the reference's rounding).
    fn mrope_tables(&self, positions: &[[u32; 3]]) -> Result<(Tensor, Tensor)> {
        let rotary = self.config.text.rotary_dim();
        let pairs = rotary / 2;
        let theta = self.config.text.rope_theta;
        let section = self.config.text.mrope_section;
        let mut inv_freq = vec![0.0f32; pairs];
        for (i, slot) in inv_freq.iter_mut().enumerate() {
            *slot = 1.0 / theta.powf(2.0 * i as f32 / rotary as f32);
        }
        let mut cos = Vec::with_capacity(positions.len() * rotary);
        let mut sin = Vec::with_capacity(positions.len() * rotary);
        for token in positions {
            let mut merged = vec![0.0f32; pairs];
            for p in 0..pairs {
                let axis = if p % 3 == 1 && p < section[1] * 3 {
                    1
                } else if p % 3 == 2 && p < section[2] * 3 {
                    2
                } else {
                    0
                };
                merged[p] = token[axis] as f32 * inv_freq[p];
            }
            let mut cos_row = vec![0.0f32; rotary];
            let mut sin_row = vec![0.0f32; rotary];
            for p in 0..pairs {
                cos_row[p] = bf16_round(merged[p].cos());
                cos_row[pairs + p] = cos_row[p];
                sin_row[p] = bf16_round(merged[p].sin());
                sin_row[pairs + p] = sin_row[p];
            }
            cos.extend_from_slice(&cos_row);
            sin.extend_from_slice(&sin_row);
        }
        let ctx = self.ctx();
        let cos_t = {
            let rounded: Vec<half::bf16> = cos.iter().map(|&v| half::bf16::from_f32(v)).collect();
            transfers::to_cuda(&Tensor::from_bf16(vec![positions.len(), rotary], &rounded)?, ctx.device_id())?
        };
        let sin_t = {
            let rounded: Vec<half::bf16> = sin.iter().map(|&v| half::bf16::from_f32(v)).collect();
            transfers::to_cuda(&Tensor::from_bf16(vec![positions.len(), rotary], &rounded)?, ctx.device_id())?
        };
        Ok((cos_t, sin_t))
    }

    // ---- text stack --------------------------------------------------------

    fn lm_head(&self, x: &Tensor) -> Result<Tensor> {
        let ctx = self.ctx();
        let (m, k) = {
            let dims = x.shape().dims();
            if dims.len() != 2 {
                return Err(Error::Other("qwen_drive: lm_head input must be 2D".into()));
            }
            (dims[0], dims[1])
        };
        let vocab = self.config.text.vocab_size;
        if m == 1 {
            if let Some(file) = std::env::var_os("APXINF_QWEN_HEAD_INPUT") {
                let values = transfers::to_cpu(x)?.to_f32_vec()
                    .map_err(|error| Error::Other(error.to_string()))?;
                let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
                std::fs::write(file, bytes).map_err(|error| Error::Other(error.to_string()))?;
            }
        }
        let out = device_tensor(ctx, &[m, vocab], DType::BF16)?;
        let a = DeviceBuffer::from_tensor(x).map_err(Error::Cuda)?;
        let b = DeviceBuffer::from_tensor(&self.weights.embed_tokens).map_err(Error::Cuda)?;
        let c = DeviceBuffer::from_tensor(&out).map_err(Error::Cuda)?;
        gemm::write_ex(
            ctx,
            DType::BF16,
            CublasTranspose::None,
            CublasTranspose::Transpose,
            m,
            vocab,
            k,
            1.0,
            &a,
            k as i32,
            &b,
            k as i32,
            0.0,
            &c,
            vocab as i32,
        )?;
        Ok(out)
    }

    // TEMP-DIAG (implement_r19): full-stack echo-margin trajectory lens (lens_trajectory_r17) --
    // applies the model's own final readout (rms_norm_plus1 + lm_head) to the last 3 hidden
    // rows at one site, returning one digest line with per-row top-2 + own-token logit and the
    // pre-norm row l2 values; the k=31 site is the same computation as the final readout
    // (probe-validity gate) and the lens l2 must equal the live hidden_norm l2 (probe-bug
    // detector); digest-routed so every value lands inside the 80-line capture window; revert
    // in the acceptance-bound revision.
    fn shiftlens_probe(&self, hidden: &Tensor, tail3: &[u32; 3], k_tag: &str, kind: &str) -> Result<String> {
        let dims = hidden.shape().dims().to_vec();
        let width = *dims.last().unwrap_or(&1);
        let rows = hidden.numel() / width.max(1);
        if rows < 3 {
            return Ok(format!("[qwen_drive] shiftlens k={k_tag} kind={kind} rows={rows} (skipped: rows < 3)"));
        }
        let row_bytes = width * DType::BF16.size_in_bytes();
        let hbuf = DeviceBuffer::from_tensor(hidden).map_err(Error::Cuda)?;
        let slab = hbuf.view((rows - 3) * row_bytes, 3 * row_bytes).map_err(Error::Cuda)?;
        let slab_t = slab.as_tensor(Shape::new(vec![3, width]), DType::BF16).map_err(Error::Cuda)?;
        let hvals = transfers::to_cpu(&slab_t)?.to_f32_vec()
            .map_err(|e| Error::Other(format!("qwen_drive shiftlens l2: {e}")))?;
        let mut l2s: Vec<f64> = Vec::new();
        for r in 0..3usize {
            l2s.push(hvals[r * width..(r + 1) * width].iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt());
        }
        let lens_normed = la::rms_norm_plus1(self.ctx(), &slab_t, &self.weights.final_norm, self.config.text.rms_norm_eps)?;
        let lens_logits = self.lm_head(&lens_normed)?;
        let vocab = self.config.text.vocab_size;
        let lvals = transfers::to_cpu(&lens_logits)?.to_f32_vec()
            .map_err(|e| Error::Other(format!("qwen_drive shiftlens: {e}")))?;
        let mut cells: Vec<(u32, f32, u32, f32, f32)> = Vec::new();
        for r in 0..3usize {
            let row = &lvals[r * vocab..(r + 1) * vocab];
            let mut best: (u32, f32) = (0, f32::NEG_INFINITY);
            let mut runner: (u32, f32) = (0, f32::NEG_INFINITY);
            for (idx, &value) in row.iter().enumerate() {
                if value > best.1 {
                    runner = best;
                    best = (idx as u32, value);
                } else if value > runner.1 {
                    runner = (idx as u32, value);
                }
            }
            cells.push((best.0, best.1, runner.0, runner.1, row[tail3[r] as usize]));
        }
        // r16 row convention: rowN1 = last hidden row (slab row 2, own token tail3[2]),
        // rowN2 = slab row 1 (tail3[1]), rowN3 = slab row 0 (tail3[0]); l2 in the same order.
        Ok(format!(
            "[qwen_drive] shiftlens k={} kind={} rowN1={:?} rowN2={:?} rowN3={:?} l2=({:.4},{:.4},{:.4})",
            k_tag, kind, cells[2], cells[1], cells[0], l2s[2], l2s[1], l2s[0]))
    }

    fn forward_mlp(&self, x: Tensor, post_norm: &Tensor, gate_up_w: &Tensor, down_w: &Tensor, trace: bool) -> Result<Tensor> {
        let ctx = self.ctx();
        let eps = self.config.text.rms_norm_eps;
        let normed = la::rms_norm_plus1(ctx, &x, post_norm, eps)?;
        let gu = gemm::bf16(ctx, &normed, gate_up_w)?;
        let act = activation::swiglu_bf16_rounded(ctx, &gu)?;
        let down = linear_checkpoint(ctx, &act, down_w)?;
        if trace {
            trace_rows("text0_post_norm", &normed)?;
            trace_rows("text0_gate_up", &gu)?;
            trace_rows("text0_swiglu", &act)?;
            trace_rows("text0_down", &down)?;
        }
        elementwise::add(ctx, &x, &down)
    }

    fn forward_full_attention(
        &mut self,
        x: Tensor,
        layer_idx: usize,
        cos: &Tensor,
        sin: &Tensor,
        seq: usize,
    ) -> Result<Tensor> {
        let cuda = Arc::clone(&self.cuda);
        let ctx = cuda.context();
        let text = &self.config.text;
        let w = match &self.weights.layers[layer_idx] {
            MixerWeights::FullAttention(w) => w,
            _ => return Err(Error::Other("qwen_drive: layer kind mismatch".into())),
        };
        let eps = text.rms_norm_eps;
        let heads = text.n_heads;
        let kv_heads = text.n_kv_heads;
        let head_dim = text.head_dim;
        let rotary = text.rotary_dim();
        let normed = la::rms_norm_plus1(ctx, &x, &w.input_norm, eps)?;
        let fused = project_and_pack(ctx, &normed, &[&w.q_w, &w.k_w, &w.v_w])?;
        if layer_idx == 3 && seq > 1 {
            trace_rows("text3_input_norm", &normed)?;
            trace_rows("text3_fused_qkv", &fused)?;
            trace_rows("text3_cos", cos)?;
            trace_rows("text3_sin", sin)?;
        }
        let q_out = device_tensor(ctx, &[seq, heads, head_dim], DType::BF16)?;
        let (k_cache, v_cache) = match &self.caches[layer_idx] {
            LayerCache::FullAttention { k, v } => (k.clone(), v.clone()),
            _ => return Err(Error::Other("qwen_drive: cache kind mismatch".into())),
        };
        la::full_attn_prepare(
            ctx,
            &fused,
            &w.q_norm,
            &w.k_norm,
            cos,
            sin,
            &q_out,
            &k_cache,
            &v_cache,
            self.cache_len,
            heads,
            kv_heads,
            head_dim,
            rotary,
            eps,
        )?;
        let kv_len = self.cache_len + seq;
        let k_view = cache_view(&k_cache, kv_len)?;
        let v_view = cache_view(&v_cache, kv_len)?;
        let attn = attention::causal_gqa_bf16(ctx, &q_out, &k_view, &v_view, kv_len)?;
        if layer_idx == 3 && seq > 1 {
            trace_rows("text3_q", &q_out.reshape(vec![seq, heads * head_dim])?)?;
            trace_rows("text3_k", &k_view.reshape(vec![kv_len, kv_heads * head_dim])?)?;
            trace_rows("text3_v", &v_view.reshape(vec![kv_len, kv_heads * head_dim])?)?;
            trace_rows("text3_attention", &attn.reshape(vec![seq, heads * head_dim])?)?;
        }
        let attn = attn.reshape(vec![seq, heads * head_dim])?;
        la::sigmoid_gate_mul(ctx, &attn, &fused, heads, head_dim)?;
        let proj = linear_checkpoint(ctx, &attn, &w.o_w)?;
        if layer_idx == 3 && seq > 1 {
            trace_rows("text3_gated_attention", &attn)?;
            trace_rows("text3_out_proj", &proj)?;
        }
        let hidden = elementwise::add(ctx, &x, &proj)?;
        self.forward_mlp(hidden, &w.post_norm, &w.gate_up_w, &w.down_w, false)
    }

    /// Run one GDN decode layer, from a captured graph when one is available.
    ///
    /// Order matters. A capture records without executing, but the host-side
    /// conv parity flip inside the body runs either way, so the sequence is
    /// capture (parity advances, device does nothing) then replay (device does
    /// the work once). Running the body eagerly first and capturing afterwards
    /// would advance the parity twice and bake the wrong buffers.
    fn forward_gdn(&mut self, x: Tensor, layer_idx: usize, seq: usize) -> Result<Tensor> {
        let step = self.decode_step.unwrap_or(0);
        if seq != 1
            || self.cache_len == 0
            || step < DECODE_GRAPH_WARMUP_STEPS
            || !decode_graph_enabled_for(layer_idx)
        {
            return self.forward_gdn_eager(x, layer_idx, seq);
        }
        let parity = match &self.caches[layer_idx] {
            LayerCache::Gdn { flip, .. } => usize::from(*flip),
            _ => return self.forward_gdn_eager(x, layer_idx, seq),
        };
        self.forward_gdn_captured(x, layer_idx, parity)
    }

    fn forward_gdn_captured(
        &mut self,
        x: Tensor,
        layer_idx: usize,
        parity: usize,
    ) -> Result<Tensor> {
        let cuda = Arc::clone(&self.cuda);
        let ctx = cuda.context();
        let dims = x.shape().dims().to_vec();
        let bytes = x.numel() * DType::BF16.size_in_bytes();

        if self.gdn_graphs.len() != self.weights.layers.len() {
            self.gdn_graphs = (0..self.weights.layers.len())
                .map(|_| [None, None])
                .collect();
            self.gdn_graph_out = (0..self.weights.layers.len())
                .map(|_| [None, None])
                .collect();
            self.gdn_graph_result = vec![None; self.weights.layers.len()];
            self.gdn_graph_prepared = vec![[false, false]; self.weights.layers.len()];
        }
        if self.gdn_graph_workspace.is_none() {
            self.gdn_graph_workspace = Some(kernels::GraphWorkspace::new(
                DECODE_GRAPH_ARENA_BYTES,
                ctx.device_id(),
            )?);
        }
        // Diagnostic arm: take the arena and then run the body exactly as the
        // eager path does, with no workspace bound, no staging, no capture. The
        // only difference from a clean run is that a 64MB block now exists.
        // Every hypothesis that blames what the arena path *does* predicts this
        // is correct; a layout hypothesis -- some kernel writing past the end of
        // a buffer whose neighbour this allocation changed -- predicts it is
        // wrong, and that would explain a full-attention layer going bad while
        // the GDN layer that uses the arena stays exact.
        if std::env::var("APXINF_QWEN_DECODE_GRAPH_ARENA_ONLY")
            .map(|value| value.contains("alloc"))
            .unwrap_or(false)
        {
            return self.forward_gdn_eager(x, layer_idx, 1);
        }

        if self.gdn_graph_in.is_none() {
            self.gdn_graph_in = Some(persistent_tensor(ctx, &dims, DType::BF16)?);
        }
        if self.gdn_graph_result[layer_idx].is_none() {
            self.gdn_graph_result[layer_idx] = Some(persistent_tensor(ctx, &dims, DType::BF16)?);
        }

        // The capture reads this address, so every step stages its input here.
        let staged = self.gdn_graph_in.clone().expect("staging input");
        device_copy(ctx, &staged, &x, bytes)?;

        // Diagnostic arm: run the body inside the arena with no capture and no
        // replay. If this is already wrong, the fault is in the workspace
        // allocation path rather than in anything CUDA graphs do.
        if std::env::var_os("APXINF_QWEN_DECODE_GRAPH_ARENA_ONLY").is_some() {
            // `raw` drops the staging copies as well, which separates the arena
            // itself from the copies in and out of it.
            let raw = std::env::var("APXINF_QWEN_DECODE_GRAPH_ARENA_ONLY")
                .map(|value| value.contains("raw"))
                .unwrap_or(false);
            let input = if raw { x.clone() } else { staged.clone() };
            let prepare = std::env::var("APXINF_QWEN_DECODE_GRAPH_ARENA_ONLY")
                .map(|value| value.contains("prepare"))
                .unwrap_or(false);
            let workspace = self.gdn_graph_workspace.take().expect("graph arena");
            let produced = if prepare {
                kernels::prepare_with_workspace(&workspace, || {
                    self.forward_gdn_eager(input, layer_idx, 1)
                })
            } else {
                kernels::with_workspace(&workspace, || self.forward_gdn_eager(input, layer_idx, 1))
            };
            self.gdn_graph_workspace = Some(workspace);
            let output = produced?;
            if raw {
                return Ok(output);
            }
            let landing = self.gdn_graph_result[layer_idx]
                .clone()
                .expect("landing buffer");
            device_copy(ctx, &landing, &output, bytes)?;
            return Ok(landing);
        }

        if !self.gdn_graph_prepared[layer_idx][parity] {
            // Declared preflight: executes, so its output is this step's real
            // result and the recurrent and conv state advance exactly once.
            let workspace = self.gdn_graph_workspace.take().expect("graph arena");
            let prepared = kernels::prepare_with_workspace(&workspace, || {
                self.forward_gdn_eager(staged.clone(), layer_idx, 1)
            });
            self.gdn_graph_workspace = Some(workspace);
            let output = prepared?;
            cuda.synchronize()?;
            self.gdn_graph_prepared[layer_idx][parity] = true;
            let landing = self.gdn_graph_result[layer_idx]
                .clone()
                .expect("landing buffer");
            device_copy(ctx, &landing, &output, bytes)?;
            return Ok(landing);
        }

        // A capture executes the Rust body, so the host-side conv flip advances
        // on that step and must not be advanced again after the replay below.
        let mut captured_now = false;
        if self.gdn_graphs[layer_idx][parity].is_none() {
            captured_now = true;
            cuda.synchronize()?;
            let workspace = self.gdn_graph_workspace.take().expect("graph arena");
            cuda.begin_capture()?;
            let captured = kernels::with_workspace(&workspace, || {
                self.forward_gdn_eager(staged.clone(), layer_idx, 1)
            });
            let output = match captured {
                Ok(output) => output,
                Err(error) => {
                    let _ = cuda.end_capture();
                    self.gdn_graph_workspace = Some(workspace);
                    return Err(error);
                }
            };
            let graph = cuda.end_capture()?;
            self.gdn_graph_workspace = Some(workspace);
            self.gdn_graph_out[layer_idx][parity] = Some(output);
            self.gdn_graphs[layer_idx][parity] = Some(graph);
        }

        self.gdn_graphs[layer_idx][parity]
            .as_ref()
            .expect("captured graph")
            .replay()?;

        // The conv state is a ping-pong pair chosen by a host-side flip, and
        // `parity` is that flip. The eager body toggles it on its way through;
        // a replay runs no Rust at all, so without this the flip freezes, the
        // same graph is selected on every step, and it reads the same conv
        // buffer forever while writing the one nothing reads. The layer then
        // sees a convolution window frozen at the step of the last eager pass,
        // which is why the generation collapses into repetition rather than
        // drifting. Advancing it here restores the alternation the two captured
        // graphs were built for: parity 0 reads A and writes B, parity 1 reads
        // B and writes A.
        if !captured_now {
            if let LayerCache::Gdn { flip, .. } = &mut self.caches[layer_idx] {
                *flip = !*flip;
            }
        }

        // The captured output lives in the arena, which the next captured layer
        // reuses, so it is copied into this layer's own buffer before returning.
        let produced = self.gdn_graph_out[layer_idx][parity]
            .clone()
            .expect("captured output");
        let landing = self.gdn_graph_result[layer_idx]
            .clone()
            .expect("landing buffer");
        device_copy(ctx, &landing, &produced, bytes)?;
        Ok(landing)
    }

    fn forward_gdn_eager(
        &mut self,
        x: Tensor,
        layer_idx: usize,
        seq: usize,
    ) -> Result<Tensor> {
        let cuda = Arc::clone(&self.cuda);
        let ctx = cuda.context();
        let text = &self.config.text;
        let w = match &self.weights.layers[layer_idx] {
            MixerWeights::Gdn(w) => w,
            _ => return Err(Error::Other("qwen_drive: layer kind mismatch".into())),
        };
        let eps = text.rms_norm_eps;
        let num_k_heads = text.linear_num_key_heads;
        let num_v_heads = text.linear_num_value_heads;
        let head_k = text.linear_key_head_dim;
        let head_v = text.linear_value_head_dim;
        let key_dim = num_k_heads * head_k;
        let value_dim = num_v_heads * head_v;
        let conv_dim = 2 * key_dim + value_dim;
        let z_col = conv_dim;
        let b_col = conv_dim + value_dim;
        let a_col = b_col + num_v_heads;
        let kernel = text.linear_conv_kernel_dim;
        let has_state = self.cache_len > 0;

        // The scan is only part of a GDN layer; attribute what precedes it too.
        let gdn_timed = seq > 1 && diagnostics_enabled();
        let mut gdn_since = std::time::Instant::now();
        let normed = la::rms_norm_plus1(ctx, &x, &w.input_norm, eps)?;
        if gdn_timed {
            gdn_stage_mark(ctx, layer_idx, "pre_norm", &mut gdn_since)?;
        }
        // Preserve all four reference GEMM geometries. Transposing or fusing
        // these weights selects different BF16 reductions in cuBLAS.
        // One GEMM over the packed projection. Four separate ones read the same
        // weight bytes, but two were only [32, hidden] and dragged the group to
        // about 70GB/s where the MLP projections reach 172GB/s; the packed
        // weight also arrives in the layout the on-device pack used to build.
        let zba = linear_checkpoint(ctx, &normed, &w.zba_w)?
            .reshape(vec![seq, conv_dim + value_dim + 2 * num_v_heads])?;
        if gdn_timed {
            gdn_stage_mark(ctx, layer_idx, "project", &mut gdn_since)?;
        }
        if layer_idx == 0 && seq > 1 {
            trace_rows("text0_input", &x)?;
            trace_rows("text0_input_norm", &normed)?;
            trace_rows("text0_zba", &zba)?;
        }
        let conv_out = device_tensor(ctx, &[seq, conv_dim], DType::BF16)?;
        let (state_current, state_next) = match &self.caches[layer_idx] {
            LayerCache::Gdn { conv_state_a, conv_state_b, flip, .. } => {
                if *flip {
                    (conv_state_b.clone(), conv_state_a.clone())
                } else {
                    (conv_state_a.clone(), conv_state_b.clone())
                }
            }
            _ => return Err(Error::Other("qwen_drive: cache kind mismatch".into())),
        };
        la::causal_conv1d_silu_bf16(
            ctx,
            &zba,
            &w.conv_w,
            if has_state { Some(&state_current) } else { None },
            &conv_out,
            &state_next,
            kernel,
        )?;
        if let LayerCache::Gdn { flip, .. } = &mut self.caches[layer_idx] {
            *flip = !*flip;
        }
        if layer_idx == 0 && seq > 1 { trace_rows("text0_conv_silu", &conv_out)?; }
        if gdn_timed {
            gdn_stage_mark(ctx, layer_idx, "conv_silu", &mut gdn_since)?;
        }

        let recurrent_decode = has_state && seq == 1;
        let seq_pad = if recurrent_decode { 1 } else { seq.div_ceil(GDN_CHUNK) * GDN_CHUNK };
        let scan_padded = seq_pad != seq;
        let q_buf = alloc_scan_scratch(ctx, num_v_heads * seq_pad * head_k * DType::F32.size_in_bytes(), scan_padded)?;
        let k_buf = alloc_scan_scratch(ctx, num_v_heads * seq_pad * head_k * DType::F32.size_in_bytes(), scan_padded)?;
        let v_buf = alloc_scan_scratch(ctx, num_v_heads * seq_pad * head_v * DType::F32.size_in_bytes(), scan_padded)?;
        let beta_buf = alloc_scan_scratch(ctx, num_v_heads * seq_pad * DType::F32.size_in_bytes(), scan_padded)?;
        let g_buf = alloc_scan_scratch(ctx, num_v_heads * seq_pad * DType::F32.size_in_bytes(), scan_padded)?;
        if gdn_timed {
            gdn_stage_mark(ctx, layer_idx, "qkvbg_alloc", &mut gdn_since)?;
        }
        la::gdn_qk_prep(ctx, &conv_out, &q_buf, &k_buf, seq_pad, num_k_heads, num_v_heads, head_k, key_dim, recurrent_decode, 1e-6)?;
        la::gdn_vb_prep(
            ctx,
            &conv_out,
            &zba,
            b_col,
            a_col,
            &w.dt_bias,
            &w.a_log,
            &v_buf,
            &beta_buf,
            &g_buf,
            seq_pad,
            num_v_heads,
            head_v,
            2 * key_dim,
        )?;
        let gdn_out = device_tensor(ctx, &[seq, value_dim], DType::BF16)?;
        let rec_state = match &self.caches[layer_idx] {
            LayerCache::Gdn { recurrent, .. } => recurrent.clone(),
            _ => return Err(Error::Other("qwen_drive: cache kind mismatch".into())),
        };
        let rec_buf = DeviceBuffer::from_tensor(&rec_state).map_err(Error::Cuda)?;
        if recurrent_decode {
            la::gdn_recurrent(
                ctx,
                &q_buf,
                &k_buf,
                &v_buf,
                &beta_buf,
                &g_buf,
                &rec_buf,
                &gdn_out,
                num_v_heads,
                head_k,
                head_v,
            )?;
        } else {
            let chunks = seq_pad / GDN_CHUNK;
            let g_cum = alloc_zeros(ctx, num_v_heads * seq_pad * DType::F32.size_in_bytes())?;
            let a_buf = alloc_zeros(ctx, num_v_heads * chunks * GDN_CHUNK * GDN_CHUNK * DType::F32.size_in_bytes())?;
            let t_buf = alloc_zeros(ctx, num_v_heads * chunks * GDN_CHUNK * GDN_CHUNK * DType::F32.size_in_bytes())?;
            let vt_buf = alloc_zeros(ctx, num_v_heads * chunks * GDN_CHUNK * head_v * DType::F32.size_in_bytes())?;
            let kcd_buf = alloc_zeros(ctx, num_v_heads * chunks * GDN_CHUNK * head_k * DType::F32.size_in_bytes())?;
            // Covers the q/k/v/beta/g preparation kernels and the scan's own
            // zeroed allocations, which sit between conv_silu and cumsum.
            if gdn_timed {
                gdn_stage_mark(ctx, layer_idx, "prep_alloc", &mut gdn_since)?;
            }
            la::gdn_cumsum(ctx, &g_buf, &g_cum, seq_pad, num_v_heads, GDN_CHUNK)?;
            if gdn_timed {
                gdn_stage_mark(ctx, layer_idx, "cumsum", &mut gdn_since)?;
            }
            la::gdn_attn_raw(ctx, &q_buf, &k_buf, &beta_buf, &g_cum, &a_buf, &t_buf, seq_pad, num_v_heads, head_k, GDN_CHUNK)?;
            if gdn_timed {
                gdn_stage_mark(ctx, layer_idx, "attn_raw", &mut gdn_since)?;
            }
            la::gdn_tri_solve(ctx, &a_buf, num_v_heads * chunks, GDN_CHUNK)?;
            if gdn_timed {
                gdn_stage_mark(ctx, layer_idx, "tri_solve", &mut gdn_since)?;
            }
            la::gdn_chunk_gemm(ctx, &a_buf, &v_buf, &k_buf, &beta_buf, &g_cum, &vt_buf, &kcd_buf, seq_pad, num_v_heads, head_k, head_v, GDN_CHUNK)?;
            if gdn_timed {
                gdn_stage_mark(ctx, layer_idx, "chunk_gemm", &mut gdn_since)?;
            }
            la::gdn_chunk_state(
                ctx,
                &q_buf,
                &k_buf,
                &g_cum,
                &t_buf,
                &vt_buf,
                &kcd_buf,
                &rec_buf,
                &gdn_out,
                seq_pad,
                num_v_heads,
                head_k,
                head_v,
                GDN_CHUNK,
            )?;
            if gdn_timed {
                gdn_stage_mark(ctx, layer_idx, "chunk_state", &mut gdn_since)?;
            }
        }
        if layer_idx == 0 && seq > 1 { trace_rows("text0_core", &gdn_out)?; }
        // TEMP-DIAG (implement_r16): GDN layer-0 zero-history fingerprint (shift_gdn_r15 P2) --
        // row 0 is the unique zero-history row: correct math yields a finite normal-magnitude
        // row, while ANY one-position-lag mechanism (conv window, T-diagonal exclusion, lagged
        // prep) forces gdn_out row 0 to exactly 0; the conv_out row-0 companion splits
        // conv-window lag from diagonal-exclusion. Digest-routed (a live print here would sit
        // above the 80-line capture window); revert in the acceptance-bound revision.
        if layer_idx == 0 && seq > 1 && diagnostics_enabled() {
            let gbuf = DeviceBuffer::from_tensor(&gdn_out).map_err(Error::Cuda)?;
            let gview = gbuf.view(0, 2 * value_dim * DType::BF16.size_in_bytes()).map_err(Error::Cuda)?;
            let gt = gview.as_tensor(Shape::new(vec![2, value_dim]), DType::BF16).map_err(Error::Cuda)?;
            let gvals = transfers::to_cpu(&gt)?.to_f32_vec()
                .map_err(|e| Error::Other(format!("qwen_drive gdn_row: {e}")))?;
            let gl2 = |r: usize| -> f64 {
                gvals[r * value_dim..(r + 1) * value_dim].iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt()
            };
            let g_line0 = format!("[qwen_drive] gdn_row0 l2={:.4} first4={:?}", gl2(0), &gvals[..gvals.len().min(4)]);
            let g_line1 = format!("[qwen_drive] gdn_row1 l2={:.4} first4={:?}", gl2(1), &gvals[value_dim..value_dim + 4]);
            let cbuf = DeviceBuffer::from_tensor(&conv_out).map_err(Error::Cuda)?;
            let cview = cbuf.view(0, conv_dim * DType::BF16.size_in_bytes()).map_err(Error::Cuda)?;
            let ct = cview.as_tensor(Shape::new(vec![1, conv_dim]), DType::BF16).map_err(Error::Cuda)?;
            let cvals = transfers::to_cpu(&ct)?.to_f32_vec()
                .map_err(|e| Error::Other(format!("qwen_drive conv_row: {e}")))?;
            let cl2: f64 = cvals.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
            let c_line = format!("[qwen_drive] conv_row0 l2={:.4} first4={:?}", cl2, &cvals[..cvals.len().min(4)]);
            self.diag_digest.push(g_line0);
            self.diag_digest.push(g_line1);
            self.diag_digest.push(c_line);
        }
        let gated = la::gated_rms_silu(
            ctx,
            &gdn_out.reshape(vec![seq * num_v_heads, head_v])?,
            &zba,
            z_col,
            num_v_heads,
            &w.gated_norm,
            1e-6,
        )?;
        let gated = gated.reshape(vec![seq, value_dim])?;
        let proj = linear_checkpoint(ctx, &gated, &w.out_w)?;
        let hidden = elementwise::add(ctx, &x, &proj)?;
        if layer_idx == 0 && seq > 1 {
            trace_rows("text0_gated_norm", &gated)?;
            trace_rows("text0_out_proj", &proj)?;
            trace_rows("text0_residual", &hidden)?;
        }
        self.forward_mlp(hidden, &w.post_norm, &w.gate_up_w, &w.down_w, layer_idx == 0 && seq > 1)
    }

    /// Run the text transformer over one token span, appending to the hybrid
    /// cache. `positions` is the per-token mRoPE position triple.
    fn run_text(&mut self, x: Tensor, positions: &[[u32; 3]], probe_tail3: Option<[u32; 3]>) -> Result<Tensor> {
        let seq = positions.len();
        if seq == 0 {
            return Err(Error::Other("qwen_drive: empty forward span".into()));
        }
        let last = positions[seq - 1];
        self.last_position = last[0].max(last[1]).max(last[2]) as i64;
        let (cos, sin) = self.mrope_tables(positions)?;
        let mut hidden = x;
        // TEMP-DIAG (implement_r19): pre-stack echo-margin lens site k=pre (the assembled
        // embedding/scatter output before layer 0); digest-routed; revert in the
        // acceptance-bound revision.
        if seq > 1 && diagnostics_enabled() {
            if let Some(t3) = probe_tail3 {
                let line = self.shiftlens_probe(&hidden, &t3, "pre", "embed")?;
                self.diag_digest.push(line);
            }
        }
        // TEMP-DIAG (implement_r10): per-layer ms attribution for the measured 43.37s
        // prefill (over the ~29s/inference allowance); ms_since_prev on line k ~= layer
        // k-1's duration; revert in the acceptance-bound revision.
        let mut layer_t0 = std::time::Instant::now();
        for layer_idx in 0..self.config.text.n_layers {
            let is_full = matches!(
                &self.weights.layers[layer_idx],
                MixerWeights::FullAttention(_)
            );
            // TEMP-DIAG (implement_r2): per-layer prefill heartbeat; the last printed k names the hanging layer; revert in the acceptance-bound revision.
            // TEMP-DIAG (implement_r11): fire only during prefill (seq > 1); the 64 single-token
            // decode forwards flooded the head-limited receipt (~2048 lines) and truncated
            // gen_entry/vision_entry/mha signature/prefill_done/decode_step in r10.
            if seq > 1 && diagnostics_enabled() {
                eprintln!("[qwen_drive] prefill_layer k={} kind={} ms_since_prev={:.1}", layer_idx, if is_full { "full" } else { "gdn" }, layer_t0.elapsed().as_secs_f64() * 1000.0);
            }
            layer_t0 = std::time::Instant::now();
            // TEMP-DIAG (implement_r3 / synthesis_r3, folded A2): mixer-delta probe -- hidden
            // rows {0, rows-1} l2 before/after layers {2(gdn),3(full),30(gdn),31(full)};
            // digest-routed; revert in the acceptance-bound revision.
            let layer_delta_probe =
                seq > 1 && diagnostics_enabled() && matches!(layer_idx, 2 | 3 | 30 | 31);
            let pre_l2 = if layer_delta_probe { Some(hidden_row_l2_pair(&hidden)?) } else { None };
            hidden = if is_full {
                self.forward_full_attention(hidden, layer_idx, &cos, &sin, seq)?
            } else {
                self.forward_gdn(hidden, layer_idx, seq)?
            };
            if seq > 1 { trace_rows(&format!("model_language_model_layers_{layer_idx}"), &hidden)?; }
            // Per-layer decode trace for the divergence hunt. Without the step
            // gate every step overwrote the previous one, leaving only the last.
            if seq == 1 && self.decode_step.is_some() && self.decode_step == trace_decode_step() {
                trace_rows(&format!("decode_layer_{layer_idx}"), &hidden)?;
            }
            if let Some((pre0, pre1)) = pre_l2 {
                let (post0, post1) = hidden_row_l2_pair(&hidden)?;
                self.diag_digest.push(format!(
                    "[qwen_drive] layer_delta k={} kind={} row0 {:.4}->{:.4} rowN1 {:.4}->{:.4}",
                    layer_idx, if is_full { "full" } else { "gdn" }, pre0, post0, pre1, post1));
            }
            // TEMP-DIAG (implement_r12): per-layer prefill hidden-row norm fingerprint; the first
            // anomalous k names the corrupting layer; revert in the acceptance-bound revision.
            if seq > 1 && diagnostics_enabled() {
                let dims = hidden.shape().dims().to_vec();
                let width = *dims.last().unwrap_or(&1);
                let rows = hidden.numel() / width.max(1);
                let row_bytes = width * DType::BF16.size_in_bytes();
                let buf = DeviceBuffer::from_tensor(&hidden).map_err(Error::Cuda)?;
                let view = buf.view((rows - 1) * row_bytes, row_bytes).map_err(Error::Cuda)?;
                let row_t = view.as_tensor(Shape::new(vec![1, width]), DType::BF16).map_err(Error::Cuda)?;
                let vals = transfers::to_cpu(&row_t)?.to_f32_vec()
                    .map_err(|e| Error::Other(format!("qwen_drive hidden_norm: {e}")))?;
                let l2: f64 = vals.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
                let mean: f64 = vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len().max(1) as f64;
                eprintln!("[qwen_drive] hidden_norm k={} kind={} l2={:.4} mean={:.6} first2={:?}",
                    layer_idx, if is_full { "full" } else { "gdn" }, l2, mean, &vals[..vals.len().min(2)]);
                // TEMP-DIAG (implement_r13): capture k=0 and the final layer for the digest
                // re-emission inside the capture window (same format as the live marker).
                if layer_idx == 0 || layer_idx + 1 == self.config.text.n_layers {
                    self.diag_digest.push(format!("[qwen_drive] hidden_norm k={} kind={} l2={:.4} mean={:.6} first2={:?}",
                        layer_idx, if is_full { "full" } else { "gdn" }, l2, mean, &vals[..vals.len().min(2)]));
                }
            }
            // TEMP-DIAG (implement_r19, thinned at successor implement_r5 / synthesis_r4): the
            // 33-site erosion trajectory is a settled receipt; keep only the final-layer site
            // here (k=31) -- with the k=pre site before the layer loop it preserves the
            // probe-validity gate pair (lens l2 == hidden_norm l2); the 'shiftlens k=' prefix
            // and tuple format are unchanged; revert in the acceptance-bound revision.
            if seq > 1 && diagnostics_enabled() && layer_idx + 1 == self.config.text.n_layers {
                if let Some(t3) = probe_tail3 {
                    let line = self.shiftlens_probe(&hidden, &t3, &layer_idx.to_string(), if is_full { "full" } else { "gdn" })?;
                    self.diag_digest.push(line);
                }
            }
        }
        self.cache_len += seq;
        let normed = la::rms_norm_plus1(self.ctx(), &hidden, &self.weights.final_norm, self.config.text.rms_norm_eps)?;
        // Only the final row feeds the next token. Prefill ran the head over
        // every position anyway -- at the shipped prompt that is a
        // [3095, 248064] BF16 output, 1.5GB written to read one row, and the
        // single largest GEMM in the profile at 164ms. The probes that read the
        // other rows sit behind the diagnostics gate, so keep the full head for
        // them and slice for everyone else.
        let rows = normed.shape().dims()[0];
        if rows > 1 && !diagnostics_enabled() {
            let width = normed.shape().dims()[1];
            let row_bytes = width * DType::BF16.size_in_bytes();
            let buffer = DeviceBuffer::from_tensor(&normed).map_err(Error::Cuda)?;
            let last = buffer
                .view((rows - 1) * row_bytes, row_bytes)
                .map_err(Error::Cuda)?
                .as_tensor(Shape::new(vec![1, width]), DType::BF16)
                .map_err(Error::Cuda)?;
            return self.lm_head(&last);
        }
        self.lm_head(&normed)
    }

    fn embed_tokens(&self, token_ids: &[u32]) -> Result<Tensor> {
        let ids = upload_u32(self.ctx(), token_ids)?;
        // Qwen uses the raw embedding table; lookup_bf16 applies Gemma scaling.
        let output = embedding::lookup(self.ctx(), &self.weights.embed_tokens, &ids, token_ids.len())?;
        trace_rows("model_language_model_embed_tokens", &output)?;
        Ok(output)
    }

    fn upload_pixels(&self, pixels: &Tensor) -> Result<Tensor> {
        let ctx = self.ctx();
        let on_device = if pixels.device() != Device::Cuda(ctx.device_id()) {
            transfers::to_cuda(pixels, ctx.device_id())?
        } else {
            pixels.clone()
        };
        match on_device.dtype() {
            DType::BF16 => Ok(on_device),
            DType::F32 => {
                let dims = on_device.shape().dims().to_vec();
                let out = device_tensor(ctx, &dims, DType::BF16)?;
                la::cast_f32_to_bf16(ctx, &on_device, &out)?;
                Ok(out)
            }
            dtype => Err(Error::Other(format!(
                "qwen_drive: pixel_values must be f32 or bf16, got {dtype}"
            ))),
        }
    }

    /// Multimodal prefill: vision tower, image-embedding scatter, text stack.
    fn prefill_impl(
        &mut self,
        token_ids: &[u32],
        image: Option<(&Tensor, &[[u32; 3]])>,
    ) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Err(Error::Other("qwen_drive: empty prompt".into()));
        }
        if self.cache_len != 0 {
            return Err(Error::Other(
                "qwen_drive: prefill requires a fresh cache (call reset first)".into(),
            ));
        }
        let mut x = self.embed_tokens(token_ids)?;
        let empty_grids: &[[u32; 3]] = &[];
        let grids = if let Some((pixels, grid_thw)) = image {
            let pixels = self.upload_pixels(pixels)?;
            let vis = vision::forward(&self.config, &self.weights.vision, self.ctx(), &pixels, grid_thw)?;
            // TEMP-DIAG (implement_r2): vision-tower completion marker; revert in the acceptance-bound revision.
            eprintln!("[qwen_drive] vision_done vision_rows={}", vis.primary.shape().dims()[0]);
            trace_rows("model_visual", &vis.primary)?;
            let vis_primary = &vis.primary;
            let image_tok = self.config.image_token_id;
            let image_positions = token_ids
                .iter()
                .filter(|&&token| token == image_tok)
                .count();
            let vision_rows = vis_primary.shape().dims()[0]; // TEMP-DIAG (implement_r13): rows of the (possibly ablated) scatter source
            if image_positions != vision_rows {
                return Err(Error::Other(format!(
                    "qwen_drive: {image_positions} image tokens but vision produced {vision_rows} rows"
                )));
            }
            let mut ordinal = 0u32;
            let row_map: Vec<u32> = token_ids
                .iter()
                .map(|&token| {
                    if token == image_tok {
                        let row = ordinal;
                        ordinal += 1;
                        row
                    } else {
                        u32::MAX
                    }
                })
                .collect();
            // TEMP-DIAG (implement_r15): scatter-mapping tail for the receipt (host data, zero
            // GPU cost); expected at the measured prompt: image_rows=10416,
            // last_image_prompt_row=10441, last_src_row=10415; revert in the acceptance-bound revision.
            {
                let mapped = row_map.iter().filter(|&&m| m != u32::MAX).count();
                let (last_pr, last_src) = row_map.iter().enumerate().rev()
                    .find(|(_, &m)| m != u32::MAX).map(|(i, &m)| (i, m)).unwrap_or((usize::MAX, u32::MAX));
                self.diag_digest.push(format!(
                    "[qwen_drive] scatter_tail image_rows={} last_image_prompt_row={} last_src_row={}",
                    mapped, last_pr, last_src));
            }
            let row_map_dev = upload_u32(self.ctx(), &row_map)?;
            x = elementwise::replace_rows_bf16(self.ctx(), &x, vis_primary, &row_map_dev)?; // TEMP-DIAG (implement_r13): scatter source retargeted to the (possibly ablated) tensor
            // TEMP-DIAG (implement_r3 / synthesis_r3 A1 discriminator): post-scatter row
            // readback -- row 0 (text control) and the first image row; under true vision
            // rimg_first4 must equal the vis_fp row0 fingerprint; digest-routed; revert in
            // the acceptance-bound revision.
            {
                let dims = x.shape().dims().to_vec();
                let width = *dims.last().unwrap_or(&1);
                let row_bytes = width * DType::BF16.size_in_bytes();
                let xbuf = DeviceBuffer::from_tensor(&x).map_err(Error::Cuda)?;
                let read_row = |row: usize| -> Result<(f64, Vec<f32>)> {
                    let view = xbuf.view(row * row_bytes, row_bytes).map_err(Error::Cuda)?;
                    let rt = view.as_tensor(Shape::new(vec![1, width]), DType::BF16).map_err(Error::Cuda)?;
                    let vals = transfers::to_cpu(&rt)?.to_f32_vec()
                        .map_err(|e| Error::Other(format!("qwen_drive scatter_rows: {e}")))?;
                    let l2: f64 = vals.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
                    Ok((l2, vals[..vals.len().min(4)].to_vec()))
                };
                let (r0_l2, r0_first4) = read_row(0)?;
                if let Some(img_row) = token_ids.iter().position(|&token| token == image_tok) {
                    let (rimg_l2, rimg_first4) = read_row(img_row)?;
                    self.diag_digest.push(format!(
                        "[qwen_drive] scatter_rows r0_l2={:.4} r0_first4={:?} rimg={} rimg_l2={:.4} rimg_first4={:?}",
                        r0_l2, r0_first4, img_row, rimg_l2, rimg_first4));
                }
            }
            // TEMP-DIAG (implement_r5, successor synthesis_r4 bundle 3 L6): pre-merger frame-0
            // localization leg -- one ~7.1MB readback of the vis.pre_merger frame-0 slice;
            // degenerate pre_merger => tower, diverse => merger chain; digest-routed; revert in
            // the acceptance-bound revision.
            {
                let pdims = vis.pre_merger.shape().dims().to_vec();
                let pwidth = *pdims.last().unwrap_or(&1);
                let pframe = grid_thw
                    .first()
                    .map(|g| (g[0] as usize) * (g[1] as usize) * (g[2] as usize))
                    .unwrap_or(0)
                    .min(pdims[0]);
                if pframe > 0 && pwidth >= 4 {
                    let pbuf = DeviceBuffer::from_tensor(&vis.pre_merger).map_err(Error::Cuda)?;
                    let pview = pbuf.view(0, pframe * pwidth * DType::BF16.size_in_bytes()).map_err(Error::Cuda)?;
                    let pt = pview.as_tensor(Shape::new(vec![pframe, pwidth]), DType::BF16).map_err(Error::Cuda)?;
                    let pvals = transfers::to_cpu(&pt)?.to_f32_vec()
                        .map_err(|e| Error::Other(format!("qwen_drive vis_premerge: {e}")))?;
                    let mut pl2: Vec<f64> = Vec::with_capacity(pframe);
                    for r in 0..pframe {
                        let row = &pvals[r * pwidth..(r + 1) * pwidth];
                        pl2.push(row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt());
                    }
                    let mut pl2s = pl2.clone();
                    pl2s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let pmean = pl2.iter().sum::<f64>() / pl2.len().max(1) as f64;
                    let mut row_parts: Vec<String> = Vec::new();
                    for &r in &[0usize, 867, 1736, 2603, 3471] {
                        let r = r.min(pframe - 1);
                        row_parts.push(format!("r{}={:?}", r, &pvals[r * pwidth..r * pwidth + 4]));
                    }
                    self.diag_digest.push(format!(
                        "[qwen_drive] vis_premerge l2=[{:.4},{:.4},{:.4}] {}",
                        pl2s[0], pl2s[pl2s.len() - 1], pmean, row_parts.join(" ")));
                }
            }
            // TEMP-DIAG (implement_r2): image-embedding scatter completion marker; revert in the acceptance-bound revision.
            eprintln!("[qwen_drive] embed_scatter_done");
            grid_thw
        } else {
            empty_grids
        };
        // TEMP-DIAG (implement_r15): input-assembly row-alignment discriminator. The r14
        // logits_tail3 sweep measured a per-row one-position content shift (row i argmaxes
        // token_ids[i]); the prompt tail carries repeated ids (198 at N-1/N-4, 279 at
        // N-9/N-13), so under a row-aligned embed+scatter the assembled rows of equal ids
        // are bit-identical while a one-position shift makes them differ. Digest-routed (a
        // live print here would sit above the 80-line capture window); revert in the
        // acceptance-bound revision.
        if token_ids.len() >= 16 {
            let n = token_ids.len();
            let dims = x.shape().dims().to_vec();
            let width = *dims.last().unwrap_or(&1);
            let row_bytes = width * DType::BF16.size_in_bytes();
            let xbuf = DeviceBuffer::from_tensor(&x).map_err(Error::Cuda)?;
            let slab = xbuf.view((n - 16) * row_bytes, 16 * row_bytes).map_err(Error::Cuda)?;
            let slab_t = slab.as_tensor(Shape::new(vec![16, width]), DType::BF16).map_err(Error::Cuda)?;
            let vals = transfers::to_cpu(&slab_t)?.to_f32_vec()
                .map_err(|e| Error::Other(format!("qwen_drive embed_pair: {e}")))?;
            let max_diff = |a: usize, b: usize| -> f32 {
                let (ra, rb) = (&vals[a * width..(a + 1) * width], &vals[b * width..(b + 1) * width]);
                ra.iter().zip(rb.iter()).fold(0.0f32, |m, (u, v)| m.max((u - v).abs()))
            };
            let (d198, d279) = (max_diff(15, 12), max_diff(7, 3));
            self.diag_digest.push(format!(
                "[qwen_drive] embed_pair n={} ids=({},{},{},{}) eq_198={} eq_279={} maxdiff_198={:.6} maxdiff_279={:.6}",
                n, token_ids[n - 1], token_ids[n - 4], token_ids[n - 9], token_ids[n - 13],
                d198 == 0.0, d279 == 0.0, d198, d279));
        }
        let positions = self.rope_index(token_ids, grids)?;
        let max_pos = positions
            .iter()
            .map(|triple| triple[0].max(triple[1]).max(triple[2]))
            .max()
            .unwrap_or(0) as i64;
        self.rope_delta = max_pos + 1 - token_ids.len() as i64;
        // TEMP-DIAG (implement_r14): position-id tail captured into the digest sink (a live
        // eprintln here would sit above the 80-line capture window); revert in the
        // acceptance-bound revision.
        self.diag_digest.push(format!("[qwen_drive] pos_tail last4={:?} max_pos={} rope_delta={}", &positions[positions.len().saturating_sub(4)..], max_pos, self.rope_delta));
        if self.cache_len + token_ids.len() > self.max_seq_len {
            return Err(Error::Other("qwen_drive: prompt exceeds the cache capacity".into()));
        }
        // TEMP-DIAG (implement_r19): last-3 prompt ids for the echo-margin lens own-token
        // logits (token N-1/N-2/N-3 at the measured prompt); revert in the acceptance-bound
        // revision.
        let probe_tail3 = if token_ids.len() >= 3 {
            Some([token_ids[token_ids.len() - 3], token_ids[token_ids.len() - 2], token_ids[token_ids.len() - 1]])
        } else {
            None
        };
        // TEMP-DIAG (implement_r5, successor synthesis_r4 bundle 2 support): publish the
        // per-row segment map (1=prefix-text, 2=image, 3=vision-marker, 4=tail-text) for the
        // attn_mix probe's token-id-driven segment masses; revert in the acceptance-bound revision.
        {
            let img_tok = self.config.image_token_id;
            let vs_tok = self.config.vision_start_token_id;
            let ve_tok = self.config.vision_end_token_id;
            let last_img = token_ids.iter().rposition(|&tok| tok == img_tok).unwrap_or(0);
            let seg_map: Vec<u8> = token_ids
                .iter()
                .enumerate()
                .map(|(r, &tok)| {
                    if tok == vs_tok || tok == ve_tok {
                        3u8
                    } else if tok == img_tok {
                        2u8
                    } else if r > last_img {
                        4u8
                    } else {
                        1u8
                    }
                })
                .collect();
            attention::set_attn_seg_map(seg_map);
        }
        self.run_text(x, &positions, probe_tail3)
    }

    /// Continuation/decode forward over already-tokenized ids.
    fn forward_tokens(&mut self, token_ids: &[u32]) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Err(Error::Other("qwen_drive: empty continuation".into()));
        }
        if self.cache_len + token_ids.len() > self.max_seq_len {
            return Err(Error::Other("qwen_drive: sequence exceeds the cache capacity".into()));
        }
        let positions: Vec<[u32; 3]> = (0..token_ids.len())
            .map(|i| {
                let p = (self.cache_len + i) as i64 + self.rope_delta;
                [p.max(0) as u32; 3]
            })
            .collect();
        let x = self.embed_tokens(token_ids)?;
        self.run_text(x, &positions, None)
    }

    fn scene_caches(&self) -> Result<Vec<(Tensor, Tensor)>> {
        let mut out = Vec::new();
        for layer_idx in self.config.text.full_attention_layers() {
            match &self.caches[layer_idx] {
                LayerCache::FullAttention { k, v } => {
                    out.push((cache_view(k, self.cache_len)?, cache_view(v, self.cache_len)?));
                }
                _ => return Err(Error::Other("qwen_drive: cache kind mismatch".into())),
            }
        }
        Ok(out)
    }

    /// Greedy generation with optional min-new-token EOS suppression. Returns
    /// the generated token ids (including any terminator, like HF generate).
    pub fn generate(
        &mut self,
        token_ids: &[u32],
        pixel_values: Option<(&Tensor, &[[u32; 3]])>,
        max_new_tokens: usize,
        min_new_tokens: usize,
        eos_token_ids: &[u32],
    ) -> Result<Vec<u32>> {
        // TEMP-DIAG (implement_r2): generate() entry marker (channel vs pre-entry stall disambiguator); revert in the acceptance-bound revision.
        eprintln!("[qwen_drive] gen_entry prompt_tokens={} has_pixels={} max_new_tokens={}", token_ids.len(), pixel_values.is_some(), max_new_tokens);
        // TEMP-DIAG (implement_r12): last-16 prompt ids close the prompt-divergence candidate;
        // revert in the acceptance-bound revision.
        eprintln!("[qwen_drive] prompt_tail ids={:?}", &token_ids[token_ids.len().saturating_sub(16)..]);
        self.reset_state()?;
        // TEMP-DIAG (implement_r1): generation-entry clock for prefill/decode timing; revert in the acceptance-bound revision.
        let diag_start = std::time::Instant::now();
        let mut sampler = self.backend.create_token_sampler(TokenSamplingSpec {
            vocab_size: self.config.text.vocab_size,
            max_sequence_len: token_ids.len() + max_new_tokens + 1,
        })?;
        sampler.begin(TokenSamplingInit {
            prompt_token_ids: token_ids,
            params: &TokenSamplingParams::greedy(),
            rng: RngKey::default(),
        })?;
        let mut logits = self.prefill_impl(token_ids, pixel_values)?;
        // TEMP-DIAG (implement_r1): prefill completion marker + decode heartbeat clock; revert in the acceptance-bound revision.
        eprintln!("[qwen_drive] prefill_done prompt_tokens={} elapsed_ms={:.1}", token_ids.len(), diag_start.elapsed().as_secs_f64() * 1000.0);
        // TEMP-DIAG (implement_r3 / synthesis_r3): drain the composed-causal attention
        // P-invariant probe lines (emitted during prefill at layer 3) into the digest so
        // they land inside the receipt window; revert in the acceptance-bound revision.
        for line in attention::take_attn_diag_lines() {
            self.diag_digest.push(line);
        }
        // TEMP-DIAG (implement_r5, successor synthesis_r4 bundle 3 L5): drain the vision
        // pos-embed corner-anchor line into the digest; revert in the acceptance-bound revision.
        for line in vision::take_vision_diag_lines() {
            self.diag_digest.push(line);
        }
        // TEMP-DIAG (implement_final_r20): intrinsic self-similarity floor calibration --
        // lm_head over bare rms_norm_plus1'd embedding rows for a bounded id sample
        // (5 key ids + prompt-tail unique ids + 24 fixed-stride ids). Bisects the residual
        // echo-repair classes: F1 self_argmax ~= n with +30-class margins => the k=pre +35
        // echo is the tied table's intrinsic floor (stack content-strength convicted) vs F2
        // id 198's margin anomalous => embedding content/scale defect. Read-only on
        // weights; ~30ms-class cost lands in the post-prefill window; digest-routed
        // (captured 43 -> 49, header ~64th-from-end, in-window); revert in the
        // acceptance-bound revision.
        if diagnostics_enabled() {
            let vocab = self.config.text.vocab_size;
            let mut sample: Vec<u32> = vec![198, 220, 266, 74455, 248045];
            for &id in &token_ids[token_ids.len().saturating_sub(16)..] {
                if !sample.contains(&id) {
                    sample.push(id);
                }
            }
            for i in 0..24u64 {
                let id = ((i * 10301) % (vocab as u64)) as u32;
                if !sample.contains(&id) {
                    sample.push(id);
                }
            }
            let k = sample.len();
            let ids_dev = upload_u32(self.ctx(), &sample)?;
            let emb = embedding::lookup(self.ctx(), &self.weights.embed_tokens, &ids_dev, k)?;
            let normed = la::rms_norm_plus1(self.ctx(), &emb, &self.weights.final_norm, self.config.text.rms_norm_eps)?;
            let lens = self.lm_head(&normed)?;
            let lbuf = DeviceBuffer::from_tensor(&lens).map_err(Error::Cuda)?;
            let lview = lbuf.view(0, k * vocab * DType::BF16.size_in_bytes()).map_err(Error::Cuda)?;
            let lt = lview.as_tensor(Shape::new(vec![k, vocab]), DType::BF16).map_err(Error::Cuda)?;
            let lvals = transfers::to_cpu(&lt)?.to_f32_vec()
                .map_err(|e| Error::Other(format!("qwen_drive floor_sample: {e}")))?;
            let key_ids = [198u32, 220, 266, 74455, 248045];
            let mut self_argmax = 0usize;
            let mut margins: Vec<f32> = Vec::with_capacity(k);
            let mut key_lines: Vec<String> = Vec::new();
            for (i, &id) in sample.iter().enumerate() {
                let row = &lvals[i * vocab..(i + 1) * vocab];
                let self_logit = row[id as usize];
                let mut best: (u32, f32) = (0, f32::NEG_INFINITY);
                let mut runner: (u32, f32) = (0, f32::NEG_INFINITY);
                for (j, &value) in row.iter().enumerate() {
                    if value > best.1 {
                        runner = best;
                        best = (j as u32, value);
                    } else if value > runner.1 {
                        runner = (j as u32, value);
                    }
                }
                if best.0 == id {
                    self_argmax += 1;
                }
                margins.push(self_logit - runner.1);
                if key_ids.contains(&id) {
                    key_lines.push(format!(
                        "[qwen_drive] floor_key id={} self={:.3} top=({},{:.3}) runner=({},{:.3})",
                        id, self_logit, best.0, best.1, runner.0, runner.1));
                }
            }
            margins.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = margins[margins.len() / 2];
            self.diag_digest.push(format!(
                "[qwen_drive] floor_sample n={} self_argmax={} median_self_margin={:.3} max={:.3} min={:.3}",
                k, self_argmax, median, margins[margins.len() - 1], margins[0]));
            for line in key_lines {
                self.diag_digest.push(line);
            }
        }
        let mut diag_last_step = std::time::Instant::now();
        let eos_dev = upload_u32(self.ctx(), eos_token_ids)?;
        let mut generated = Vec::new();
        let mut diag_eos = false;
        // Opt-in readback at the first divergent decode step. Sampling and
        // generated token history are unchanged by this diagnostic.
        let trace_decode_step = std::env::var("APXINF_QWEN_TRACE_DECODE_STEP")
            .ok().and_then(|value| value.parse::<usize>().ok());
        for step in 0..max_new_tokens {
            self.decode_step = Some(step);
            if step < min_new_tokens {
                let row = logits.shape().dims()[0] - 1;
                la::suppress_logits(self.ctx(), &logits, row, &eos_dev)?;
            }
            let sample = sampler.sample(NextTokenLogits::last(&logits, self.config.text.vocab_size)?)?;
            generated.push(sample.token_id);
            // TEMP-DIAG (implement_r11): first-token logits fingerprint (top-8 ids+values, row
            // max/min) — the single discriminating observation for the constant-token-198 defect;
            // one ~0.5MB readback + one device sync per generate() call; revert in the
            // acceptance-bound revision. Extended at successor-mission implement_r2 (synthesis_r2
            // directive): fires at step 0 AND step 1 (step-1 top1 separates strict lag from echo)
            // and, at successor-mission implement_r5, is complemented by the digest-routed row_health sweep (the r2 logits_rows line was subsumed and removed).
            // The step 0/1 readbacks are unconditional scaffolding and follow the
            // diagnostics gate; the APXINF_QWEN_TRACE_DECODE_STEP readback is an
            // explicit opt-in and still fires on its own.
            if ((step == 0 || step == 1) && diagnostics_enabled()) || trace_decode_step == Some(step)
            {
                let vocab = self.config.text.vocab_size;
                let rows = logits.numel() / vocab;
                let row_bytes = vocab * DType::BF16.size_in_bytes();
                let logits_buf = DeviceBuffer::from_tensor(&logits).map_err(Error::Cuda)?;
                let row_view = logits_buf.view((rows - 1) * row_bytes, row_bytes).map_err(Error::Cuda)?;
                let row_tensor = row_view.as_tensor(Shape::new(vec![1, vocab]), DType::BF16).map_err(Error::Cuda)?;
                let row_cpu = transfers::to_cpu(&row_tensor)?;
                let values = row_cpu.to_f32_vec().map_err(|e| Error::Other(format!("qwen_drive logits_top8: {e}")))?;
                let mut top8: Vec<(u32, f32)> = Vec::new();
                let mut row_min = f32::INFINITY;
                let mut row_max = f32::NEG_INFINITY;
                for (idx, &value) in values.iter().enumerate() {
                    row_min = row_min.min(value);
                    row_max = row_max.max(value);
                    if top8.len() < 8 || value > top8[7].1 {
                        top8.push((idx as u32, value));
                        top8.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        top8.truncate(8);
                    }
                }
                eprintln!("[qwen_drive] logits_top8 step={} rows={} max={:.4} min={:.4} top8={:?}", step, rows, row_max, row_min, top8);
                // TEMP-DIAG (implement_r14): logits-row argmax sweep over rows N-1/N-2/N-3
                // (reuses the r11 row-readback idiom; two extra ~0.5MB readbacks at step 0
                // only; successor implement_r2: step-1 decode logits hold a single row, so the
                // sweep is guarded to rows>=3 and skipped at step 1); revert in the
                // acceptance-bound revision.
                if rows >= 3 {
                let mut tail: Vec<Vec<(u32, f32)>> = Vec::new();
                for back in 1..=2usize {
                    let row_view = logits_buf.view((rows - 1 - back) * row_bytes, row_bytes).map_err(Error::Cuda)?;
                    let row_tensor = row_view.as_tensor(Shape::new(vec![1, vocab]), DType::BF16).map_err(Error::Cuda)?;
                    let row_cpu = transfers::to_cpu(&row_tensor)?;
                    let values = row_cpu.to_f32_vec().map_err(|e| Error::Other(format!("qwen_drive logits_tail3: {e}")))?;
                    let mut top3: Vec<(u32, f32)> = Vec::new();
                    for (idx, &value) in values.iter().enumerate() {
                        if top3.len() < 3 || value > top3[2].1 {
                            top3.push((idx as u32, value));
                            top3.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                            top3.truncate(3);
                        }
                    }
                    tail.push(top3);
                }
                eprintln!("[qwen_drive] logits_tail3 step={} rows={} n1={:?} n2_top3={:?} n3_top3={:?}", step, rows, top8[0], tail[0], tail[1]);
                }
                // TEMP-DIAG (implement_r5, successor synthesis_r4 bundle 1): row_health 64-row
                // step-0 teacher-forced sweep, digest-routed as 4 lines (pre/mid/tail/aggregate).
                // Rows: dense 0..=25, the self-describing vision-marker ladder (token_ids in
                // {vision_start, vision_end}), tail rows-16..rows-1, deduped; per row
                // (row, own_id, own_logit, top1_id, top1_logit, next_id, next_logit, flag) with
                // flag precedence D(own==next) > H(top1==next) > E(top1==own) > O; the aggregate
                // line reports hit/echo/other/degen counts plus first_break row/kind/region.
                // Replaces the subsumed r2 logits_rows sweep; revert in the acceptance-bound revision.
                if step == 0 && rows > 16 && diagnostics_enabled() {
                    let vs_tok = self.config.vision_start_token_id;
                    let ve_tok = self.config.vision_end_token_id;
                    let img_tok = self.config.image_token_id;
                    let last_img = token_ids.iter().rposition(|&tok| tok == img_tok).unwrap_or(0);
                    let mut sweep_rows: Vec<usize> = (0..=25usize).collect();
                    for (r, &tok) in token_ids.iter().enumerate().take(rows) {
                        if tok == vs_tok || tok == ve_tok {
                            sweep_rows.push(r);
                        }
                    }
                    for row in (rows - 16)..rows {
                        sweep_rows.push(row);
                    }
                    sweep_rows.sort_unstable();
                    sweep_rows.dedup();
                    let mut pre_txt: Vec<String> = Vec::new();
                    let mut mid_txt: Vec<String> = Vec::new();
                    let mut tail_txt: Vec<String> = Vec::new();
                    let (mut hit, mut echo, mut other, mut degen) = (0usize, 0usize, 0usize, 0usize);
                    let mut first_break: Option<(usize, char)> = None;
                    for &row in &sweep_rows {
                        let row_view = logits_buf.view(row * row_bytes, row_bytes).map_err(Error::Cuda)?;
                        let row_tensor = row_view.as_tensor(Shape::new(vec![1, vocab]), DType::BF16).map_err(Error::Cuda)?;
                        let row_cpu = transfers::to_cpu(&row_tensor)?;
                        let values = row_cpu.to_f32_vec().map_err(|e| Error::Other(format!("qwen_drive row_health: {e}")))?;
                        let mut top1: (u32, f32) = (0, f32::NEG_INFINITY);
                        for (idx, &value) in values.iter().enumerate() {
                            if value > top1.1 {
                                top1 = (idx as u32, value);
                            }
                        }
                        let own_id = token_ids.get(row).copied().unwrap_or(0);
                        let next_id = token_ids.get(row + 1).copied();
                        let own_logit = values[own_id as usize];
                        let next_logit = next_id.map(|id| values[id as usize]).unwrap_or(f32::NAN);
                        let flag = if next_id == Some(own_id) {
                            'D'
                        } else if next_id == Some(top1.0) {
                            'H'
                        } else if top1.0 == own_id {
                            'E'
                        } else {
                            'O'
                        };
                        match flag {
                            'H' => hit += 1,
                            'E' => echo += 1,
                            'D' => degen += 1,
                            _ => other += 1,
                        }
                        if (flag == 'E' || flag == 'O') && first_break.is_none() {
                            first_break = Some((row, flag));
                        }
                        let part = format!(
                            "({},{},{:.4},{},{:.4},{},{:.4},{})",
                            row,
                            own_id,
                            own_logit,
                            top1.0,
                            top1.1,
                            next_id.map(|id| id.to_string()).unwrap_or_else(|| "na".to_string()),
                            next_logit,
                            flag,
                        );
                        if row <= 25 {
                            pre_txt.push(part);
                        } else if row >= rows - 16 {
                            tail_txt.push(part);
                        } else {
                            mid_txt.push(part);
                        }
                    }
                    let (fb_row, fb_kind) = first_break
                        .map(|(r, f)| (r.to_string(), f.to_string()))
                        .unwrap_or_else(|| ("none".to_string(), "none".to_string()));
                    let fb_region = match first_break {
                        None => "none",
                        Some((r, _)) if r < 4 => "prefix",
                        Some((r, _)) if r > last_img => "tail",
                        Some(_) => "marker",
                    };
                    self.diag_digest.push(format!("[qwen_drive] row_health_pre step=0 [{}]", pre_txt.join(" ")));
                    self.diag_digest.push(format!("[qwen_drive] row_health_mid step=0 [{}]", mid_txt.join(" ")));
                    self.diag_digest.push(format!("[qwen_drive] row_health_tail step=0 [{}]", tail_txt.join(" ")));
                    self.diag_digest.push(format!(
                        "[qwen_drive] row_health step=0 probed={} hit={} echo={} other={} degen={} first_break={} kind={} region={}",
                        sweep_rows.len(), hit, echo, other, degen, fb_row, fb_kind, fb_region));
                }
            }
            // TEMP-DIAG (implement_r1): heartbeat at step 0 and every 25 steps; revert in the acceptance-bound revision.
            if step % 25 == 0 {
                eprintln!("[qwen_drive] decode_step k={} token_id={} ms_since_last={:.1}", step, sample.token_id, diag_last_step.elapsed().as_secs_f64() * 1000.0);
                diag_last_step = std::time::Instant::now();
            }
            if eos_token_ids.contains(&sample.token_id) || step + 1 == max_new_tokens {
                diag_eos = eos_token_ids.contains(&sample.token_id);
                break;
            }
            logits = self.forward_tokens(&[sample.token_id])?;
        }
        // TEMP-DIAG (implement_r13): evidence-accessibility digest -- re-emit the r12 vis_fp,
        // prompt_tail and hidden_norm k=0/k=31 values as short lines here so they land inside the
        // receipt's trailing capture window; revert in the repair revision.
        eprintln!("[qwen_drive] diag_digest zero_vision={} captured={}", DIAG_ZERO_VISION as u8, self.diag_digest.len());
        eprintln!("[qwen_drive] prompt_tail ids={:?}", &token_ids[token_ids.len().saturating_sub(16)..]);
        for line in self.diag_digest.drain(..) {
            eprintln!("{line}");
        }
        // TEMP-DIAG (implement_r1): exit summary with the first 8 generated ids; revert in the acceptance-bound revision.
        eprintln!("[qwen_drive] decode_exit steps={} eos={} first_ids={:?}", generated.len(), diag_eos, &generated[..generated.len().min(8)]);
        Ok(generated)
    }

    fn require_expert(&self) -> Result<&super::device_weights::ExpertDeviceWeights> {
        self.weights.expert.as_ref().ok_or_else(|| {
            Error::Other(
                "qwen_drive: no planning expert loaded; pass a planner directory at load".into(),
            )
        })
    }

    /// Direct planning: prefill the closed-empty-assistant prompt and run the
    /// flow-matching sampler against its scene caches.
    pub fn plan_direct(
        &mut self,
        token_ids: &[u32],
        pixel_values: &Tensor,
        grid_thw: &[[u32; 3]],
        cond: &ExpertConditioning,
        noise: &[f32],
        num_steps: Option<usize>,
    ) -> Result<Vec<f32>> {
        self.reset_state()?;
        self.prefill_impl(token_ids, Some((pixel_values, grid_thw)))?;
        let anchor = self.last_position;
        let scene = self.scene_caches()?;
        let weights = self.require_expert()?;
        expert::plan(
            &self.config,
            weights,
            self.ctx(),
            &ExpertPlan {
                scene: &scene,
                scene_len: self.cache_len,
                anchor,
                cond,
                noise,
                num_steps: num_steps.unwrap_or(self.config.num_inference_steps),
            },
        )
    }

    /// Reasoning planning: greedy assistant turn with min/max token bounds,
    /// cache completion to the trained turn ending, then the sampler. Returns
    /// `(generated_token_ids, normalized_trajectory)`.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_reasoning(
        &mut self,
        token_ids: &[u32],
        pixel_values: &Tensor,
        grid_thw: &[[u32; 3]],
        max_new_tokens: usize,
        min_new_tokens: usize,
        terminator_ids: &[u32],
        im_end_id: u32,
        newline_ids: &[u32],
        cond: &ExpertConditioning,
        noise: &[f32],
        num_steps: Option<usize>,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        self.reset_state()?;
        let mut sampler = self.backend.create_token_sampler(TokenSamplingSpec {
            vocab_size: self.config.text.vocab_size,
            max_sequence_len: token_ids.len() + max_new_tokens + 1,
        })?;
        sampler.begin(TokenSamplingInit {
            prompt_token_ids: token_ids,
            params: &TokenSamplingParams::greedy(),
            rng: RngKey::default(),
        })?;
        let mut logits = self.prefill_impl(token_ids, Some((pixel_values, grid_thw)))?;
        let prompt_anchor = self.last_position;
        let eos_dev = upload_u32(self.ctx(), terminator_ids)?;
        let mut generated: Vec<u32> = Vec::new();
        for step in 0..max_new_tokens {
            self.decode_step = Some(step);
            if step < min_new_tokens {
                let row = logits.shape().dims()[0] - 1;
                la::suppress_logits(self.ctx(), &logits, row, &eos_dev)?;
            }
            let sample = sampler.sample(NextTokenLogits::last(&logits, self.config.text.vocab_size)?)?;
            generated.push(sample.token_id);
            if terminator_ids.contains(&sample.token_id) || step + 1 == max_new_tokens {
                break;
            }
            logits = self.forward_tokens(&[sample.token_id])?;
        }
        // Close the turn in the cache exactly like the reference.
        let mut content = generated.clone();
        for (position, token) in generated.iter().enumerate() {
            if terminator_ids.contains(token) {
                content.truncate(position);
                break;
            }
        }
        let mut closed_turn = content;
        closed_turn.push(im_end_id);
        closed_turn.extend_from_slice(newline_ids);
        let already_cached = generated.len().saturating_sub(1);
        if already_cached < closed_turn.len() {
            let pending = closed_turn[already_cached..].to_vec();
            self.forward_tokens(&pending)?;
        }
        let anchor = prompt_anchor + closed_turn.len() as i64;
        let scene = self.scene_caches()?;
        let weights = self.require_expert()?;
        let trajectory = expert::plan(
            &self.config,
            weights,
            self.ctx(),
            &ExpertPlan {
                scene: &scene,
                scene_len: self.cache_len,
                anchor,
                cond,
                noise,
                num_steps: num_steps.unwrap_or(self.config.num_inference_steps),
            },
        )?;
        Ok((generated, trajectory))
    }
}

impl LlmTrait for QwenDriveModel {
    fn load(
        _config: ModelConfig,
        _weights: HashMap<String, Tensor>,
        _device: Device,
    ) -> Result<Self>
    where
        Self: Sized,
    {
        Err(Error::Other(
            "QwenDriveModel::load(ModelConfig) is not supported; use \
             QwenDriveModel::load(dir, planner, device) or the registry loader"
                .into(),
        ))
    }

    /// Token-level forward over the internal hybrid cache. `start_pos` is
    /// advisory: positions derive from the model-owned cache length plus the
    /// multimodal rope delta, matching the shared generation loop's usage.
    fn forward(&mut self, token_ids: &[u32], _start_pos: u32) -> Result<Tensor> {
        self.forward_tokens(token_ids)
    }

    fn backend(&self) -> &dyn Backend {
        &*self.backend
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::VISION
    }

    fn prefill(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        self.reset_state()?;
        match input.image {
            Some(image) => self.prefill_impl(input.token_ids, Some((image.pixel_values, image.grid_thw))),
            None => self.prefill_impl(input.token_ids, None),
        }
    }

    fn reset(&mut self) {
        let _ = self.reset_state();
    }

    fn vocab_size(&self) -> usize {
        self.config.text.vocab_size
    }
}
