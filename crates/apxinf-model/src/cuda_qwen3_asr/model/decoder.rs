//! qwen3-asr text decoder (Qwen3 dense, 28 layers) with explicit KV cache,
//! BF16 on apxinf-cuda-new.
//!
//! Layout facts (checkpoint-validated): hidden 2048, GQA 16 q / 8 kv heads,
//! head_dim 128 (full-width rope, theta 1e6), MLP 2048->12288 fused
//! gate+up -> SwiGLU -> 6144 -> 2048, no attention or MLP bias, per-head
//! q_norm/k_norm RMSNorm before rope, tied embeddings.

use apxinf_core::{DType, Error, Result, Shape, Tensor};
use apxinf_cuda_new::ops::{
    self, Affine, ArgmaxArgs, AttentionArgs, BiasOrder, FusedAddNormArgs, GemmArgs,
    KvCacheDecodeMeta, NormArgs, NormKind, NormNumerics,
};
use apxinf_cuda_new::transfers::copy_cuda_to_cuda;
use apxinf_cuda_new::{CudaBuffer, CudaContext as Context};

use crate::cuda_qwen3_asr::config::*;
use crate::cuda_qwen3_asr::weights::{DTYPE, Weights};

fn alloc(ctx: &Context, dims: &[usize]) -> Result<Tensor> {
    ctx.allocate_output(Shape::new(dims.to_vec()), DTYPE)
}

/// Gemm without bias, into a caller-provided output buffer. Reused outputs
/// keep the a/output addresses step-invariant so the Execution cache hits;
/// the buffer is assumed pre-shaped `[M, N]` by the caller.
pub(crate) fn gemm_into(ctx: &Context, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let a_dims = a.shape().dims().to_vec();
    let b_dims = b.shape().dims().to_vec();
    if a_dims.len() != 2 || b_dims.len() != 2 || a_dims[1] != b_dims[0] {
        return Err(Error::Other(format!(
            "qwen3-asr text gemm shape mismatch: a={a_dims:?} b={b_dims:?}"
        )));
    }
    ops::gemm(ctx, GemmArgs::new(a, b, out))
        .map_err(|e| Error::Other(format!("text gemm a={a_dims:?} b={b_dims:?}: {e}")))?;
    Ok(())
}

/// Gemm against a lazily uploaded weight, writing into a caller-provided
/// output buffer. `a` is a decode-step `[1, K]` activation; the weight is
/// fetched by checkpoint key.
pub(crate) fn gemm_weight_into(
    ctx: &Context,
    weights: &mut Weights,
    a: &Tensor,
    key: &str,
    out: &mut Tensor,
) -> Result<()> {
    let b = weights.get(ctx, key)?;
    gemm_into(ctx, a, &b, out)
}

/// RMSNorm into a caller-provided output buffer (same shape as `input`).
/// See [`gemm_into`] for why the output is reused.
pub(crate) fn rms_norm_into(ctx: &Context, input: &Tensor, weight: &Tensor, out: &mut Tensor) -> Result<()> {
    let dims = input.shape().dims().to_vec();
    ops::norm(ctx, NormArgs::rms(input, weight, out, RMS_EPS))
        .map_err(|e| Error::Other(format!("rms_norm in={dims:?}: {e}")))?;
    Ok(())
}

/// Fused residual-add + RMS norm: one launch standing in for the adjacent
/// `add_into` + `rms_norm_into` pair (56 pairs per step — one at each
/// residual tail and one at each MLP tail feeding the next layer's entry
/// norm; layer 0 has no preceding add and keeps `rms_norm_into`).
///
/// Bit-exact against the separated pair: the addition folds
/// `f32(input) + f32(residual)` with a single bf16 round-trip through
/// `hidden`, and the reduction and multiply chain match the standalone
/// RMSNorm launch (same `<<<rows, 256>>>` layout and reduction tree).
pub(crate) fn add_rms_norm_into(
    ctx: &Context,
    residual: &Tensor,
    addend: &Tensor,
    weight: &Tensor,
    hidden: &mut Tensor,
    normalized: &mut Tensor,
) -> Result<()> {
    let dims = residual.shape().dims().to_vec();
    if dims != addend.shape().dims() {
        return Err(Error::Other(format!(
            "text add+rms shape mismatch {dims:?} vs {:?}",
            addend.shape().dims()
        )));
    }
    ops::fused_add_norm(
        ctx,
        FusedAddNormArgs {
            input: addend,
            bias: None,
            residual,
            gate: None,
            hidden,
            normalized: Some(normalized),
            kind: NormKind::Rms,
            affine: Affine::Weight {
                weight,
                bias: None,
                offset: false,
            },
            quantized: None,
            numerics: NormNumerics::Flexible,
            bias_order: BiasOrder::SingleRound,
            eps: RMS_EPS,
        },
    )
    .map_err(|e| Error::Other(format!("add_rms_norm in={dims:?}: {e}")))?;
    Ok(())
}

/// Device I32 `[1]` -> host value.
///
/// The readback is preceded by a device-wide barrier: the context stream is
/// non-blocking, so a legacy-default-stream `cudaMemcpy` after
/// `ctx.synchronize()` has no ordering against it and can observe a stale
/// value (the same hazard `tests.rs` documents for the mel readback).
pub(crate) fn id_to_host(ctx: &Context, tensor: &Tensor) -> Result<i32> {
    let _ = ctx;
    apxinf_cuda_new::transfers::synchronize_device()
        .map_err(|e| Error::Other(format!("id read sync: {e}")))?;
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    let mut bytes = [0u8; 4];
    buffer
        .copy_to_host(&mut bytes)
        .map_err(|e| Error::Other(format!("id read: {e}")))?;
    Ok(i32::from_le_bytes(bytes))
}

/// Diagnostic hook, free when disabled: with `APXINF_QWEN3ASR_DUMP_LOGITS`
/// set, every step's lm_head logits (`[1, VOCAB]` bf16) are appended verbatim
/// to that path (the caller empties the file first); with
/// `APXINF_QWEN3ASR_DUMP_HIDDEN` set, the same step's final normalized hidden
/// (`ws.final_norm`, `[1, TXT_HIDDEN]` bf16, 4096 B) is appended to its own
/// path, line for line with the logits file. Call it after the outer
/// `id_to_host` has synchronized the stream. It only reads back and appends:
/// no captured graph, no change to any numerical path. Both switches are
/// cached in a `OnceLock`, so the disabled case costs two atomic loads per
/// step.
pub(crate) fn dump_logits_step(_ctx: &Context, ws: &StepWorkspace) -> Result<()> {
    use std::sync::OnceLock;
    static LOGITS_PATH: OnceLock<Option<String>> = OnceLock::new();
    let logits_path: Option<String> = LOGITS_PATH
        .get_or_init(|| std::env::var("APXINF_QWEN3ASR_DUMP_LOGITS").ok())
        .clone();
    static HIDDEN_PATH: OnceLock<Option<String>> = OnceLock::new();
    let hidden_path: Option<String> = HIDDEN_PATH
        .get_or_init(|| std::env::var("APXINF_QWEN3ASR_DUMP_HIDDEN").ok())
        .clone();
    if logits_path.is_none() && hidden_path.is_none() {
        return Ok(());
    }
    // Device-wide barrier: the legacy memcpy below has no ordering against
    // the crate's non-blocking context stream (see tests.rs).
    apxinf_cuda_new::transfers::synchronize_device()
        .map_err(|e| Error::Other(format!("dump sync: {e}")))?;
    use std::io::Write;
    if let Some(path) = logits_path {
        let buffer = CudaBuffer::from_tensor(&ws.logits).map_err(Error::Cuda)?;
        let n: usize = ws.logits.shape().dims().iter().product::<usize>() * 2;
        let mut bytes = vec![0u8; n];
        buffer
            .copy_to_host(&mut bytes)
            .map_err(|e| Error::Other(format!("logits read: {e}")))?;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| Error::Other(format!("dump logits open: {e}")))?;
        f.write_all(&bytes)
            .map_err(|e| Error::Other(format!("dump logits write: {e}")))?;
    }
    if let Some(path) = hidden_path {
        let buffer = CudaBuffer::from_tensor(&ws.final_norm).map_err(Error::Cuda)?;
        let n: usize = ws.final_norm.shape().dims().iter().product::<usize>() * 2;
        let mut bytes = vec![0u8; n];
        buffer
            .copy_to_host(&mut bytes)
            .map_err(|e| Error::Other(format!("hidden read: {e}")))?;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| Error::Other(format!("dump hidden open: {e}")))?;
        f.write_all(&bytes)
            .map_err(|e| Error::Other(format!("dump hidden write: {e}")))?;
    }
    Ok(())
}


/// Elementwise `a + b` into a caller-provided output buffer, as a single
/// graph node. The same aliasing discipline as before applies: the output
/// must never share storage with either input — in the workspaces the target
/// is always the *other* rotating hidden slot, distinct from the residual
/// (the current slot, held via a shallow clone) and from the projected
/// activation (its own workspace buffer). Bit-identical to the previous
/// two-node form (exact copy of `a`, then `out += b`).
pub(crate) fn add_into(ctx: &Context, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let dims = a.shape().dims().to_vec();
    if dims != b.shape().dims() {
        return Err(Error::Other(format!(
            "text add shape mismatch {dims:?} vs {:?}",
            b.shape().dims()
        )));
    }
    ops::elementwise_add(ctx, a, b, out).map_err(|e| Error::Other(format!("text add: {e}")))?;
    Ok(())
}

/// Deep copy (fresh device allocation) — `Tensor::clone` shares storage, so
/// stage captures of tensors mutated in place later must be deep. The
/// persistent prefill/encode workspaces rewrite every buffer in place, so
/// *all* stage captures taken from them go through here.
pub(crate) fn deep_clone(ctx: &Context, tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims().to_vec();
    let out = alloc(ctx, &dims)?;
    copy_cuda_to_cuda(ctx, tensor, &out).map_err(|e| Error::Other(format!("deep_clone: {e}")))?;
    Ok(out)
}

/// Row-range view `[count, cols]` of a contiguous BF16 `[rows, cols]` tensor.
pub(crate) fn view_rows(tensor: &Tensor, start: usize, count: usize, cols: usize) -> Result<Tensor> {
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    let view = buffer
        .view(start * cols * 2, count * cols * 2)
        .map_err(|e| Error::Other(format!("decoder view_rows: {e}")))?;
    view.as_tensor(Shape::new(vec![count, cols]), DTYPE)
        .map_err(|e| Error::Other(format!("decoder view_rows as_tensor: {e}")))
}

/// Per-layer KV cache rows `[KV_CAPACITY, kv_heads * head_dim]`.
pub struct KvCaches {
    pub(crate) keys: Vec<Tensor>,
    pub(crate) values: Vec<Tensor>,
}

impl KvCaches {
    pub fn alloc(ctx: &Context) -> Result<Self> {
        let mut keys = Vec::with_capacity(TXT_LAYERS);
        let mut values = Vec::with_capacity(TXT_LAYERS);
        for _ in 0..TXT_LAYERS {
            keys.push(alloc(ctx, &[KV_CAPACITY, TXT_KV_HEADS * TXT_HEAD_DIM])?);
            values.push(alloc(ctx, &[KV_CAPACITY, TXT_KV_HEADS * TXT_HEAD_DIM])?);
        }
        Ok(Self { keys, values })
    }

    /// `[1, len, kv_heads, head_dim]` view of rows `[start, start+len)`.
    fn view(&self, tensor: &Tensor, start: usize, len: usize) -> Result<Tensor> {
        let rows = view_rows(tensor, start, len, TXT_KV_HEADS * TXT_HEAD_DIM)?;
        rows.reshape(Shape::new(vec![1, len, TXT_KV_HEADS, TXT_HEAD_DIM]))
    }

    pub(crate) fn key_view(&self, layer: usize, start: usize, len: usize) -> Result<Tensor> {
        self.view(&self.keys[layer], start, len)
    }

    pub(crate) fn value_view(&self, layer: usize, start: usize, len: usize) -> Result<Tensor> {
        self.view(&self.values[layer], start, len)
    }
}

/// Persistent per-step scratch buffers for the decode (m=1) path.
///
/// Decode runs every operator once per generated token, and the Execution
/// cache keys native prepares on the bare addresses of the spec's
/// a/b/output tensors. Freshly allocated intermediates churn those
/// addresses every step, so the cache misses and each of the
/// 28 layers x 7 gemms re-prepares — the dominant decode cost. These
/// buffers are allocated once and written in place every step, making the
/// a/output addresses step-invariant; address reproducibility is precisely
/// what makes the cache hit.
///
/// `hidden` is a rotating double buffer. Each residual is a shallow clone
/// sharing the current slot's storage, and the residual add writes the
/// *other* slot — so the add output never aliases either of its inputs
/// (see the `add_into` discipline above), and every layer hands the next
/// layer a stable hidden address too.
pub(crate) struct StepWorkspace {
    /// Rotating hidden double buffer; residual = hidden[cur] shares storage.
    pub(crate) hidden: [Tensor; 2],
    /// input_layernorm output `[1, TXT_HIDDEN]` (a of the q/k/v gemms).
    pub(crate) normed: Tensor,
    /// q projection output `[1, TXT_HIDDEN]` (per-head norm + rope run on
    /// `q_normed` downstream; this buffer itself is never mutated).
    pub(crate) q: Tensor,
    /// k projection output `[1, TXT_KV_HEADS * TXT_HEAD_DIM]`.
    pub(crate) k: Tensor,
    /// v projection output `[1, TXT_KV_HEADS * TXT_HEAD_DIM]`; a distinct
    /// buffer from `k` because both are live simultaneously (cache write).
    pub(crate) v: Tensor,
    /// q after per-head norm `[TXT_HEADS, TXT_HEAD_DIM]` — a fixed buffer so
    /// its address is step-stable for the captured decode graph.
    pub(crate) q_normed: Tensor,
    /// k after per-head norm `[TXT_KV_HEADS, TXT_HEAD_DIM]` (same reason).
    pub(crate) k_normed: Tensor,
    /// Attention output `[1, TXT_HIDDEN]` (a of the o gemm).
    pub(crate) attn_out: Tensor,
    /// o projection output `[1, TXT_HIDDEN]`.
    pub(crate) attn_proj: Tensor,
    /// post_attention_layernorm output `[1, TXT_HIDDEN]` (a of gate_up).
    pub(crate) normed2: Tensor,
    /// gate_up output `[1, 2 * TXT_INTER]`.
    pub(crate) fused: Tensor,
    /// swiglu output `[1, TXT_INTER]` (a of the down gemm).
    pub(crate) inner: Tensor,
    /// down projection output `[1, TXT_HIDDEN]`.
    pub(crate) outer: Tensor,
    /// Final norm output `[1, TXT_HIDDEN]` (a of the lm_head gemm).
    pub(crate) final_norm: Tensor,
    /// Logits `[1, VOCAB]`.
    pub(crate) logits: Tensor,
    /// Argmax index `[1]`, DType::I32.
    pub(crate) index: Tensor,
    /// Next token id `[1]` I32, persistent device tensor. The embedding
    /// gather reads it directly, so a captured graph stays valid while the
    /// host rewrites its contents between replays.
    pub(crate) token: Tensor,
    /// Current position `[1]` I32, persistent device tensor. Read by the
    /// RoPE kernels and by the KV-cache row scatter at kernel runtime.
    pub(crate) positions: Tensor,
    /// Fixed-address decode metadata: the attention kernels read the runtime
    /// KV length and query position from it during graph replay.
    pub(crate) decode_meta: KvCacheDecodeMeta,
}

impl StepWorkspace {
    pub fn alloc(ctx: &Context) -> Result<Self> {
        let kv = TXT_KV_HEADS * TXT_HEAD_DIM;
        Ok(Self {
            hidden: [alloc(ctx, &[1, TXT_HIDDEN])?, alloc(ctx, &[1, TXT_HIDDEN])?],
            normed: alloc(ctx, &[1, TXT_HIDDEN])?,
            q: alloc(ctx, &[1, TXT_HIDDEN])?,
            k: alloc(ctx, &[1, kv])?,
            v: alloc(ctx, &[1, kv])?,
            q_normed: alloc(ctx, &[TXT_HEADS, TXT_HEAD_DIM])?,
            k_normed: alloc(ctx, &[TXT_KV_HEADS, TXT_HEAD_DIM])?,
            attn_out: alloc(ctx, &[1, TXT_HIDDEN])?,
            attn_proj: alloc(ctx, &[1, TXT_HIDDEN])?,
            normed2: alloc(ctx, &[1, TXT_HIDDEN])?,
            fused: alloc(ctx, &[1, 2 * TXT_INTER])?,
            inner: alloc(ctx, &[1, TXT_INTER])?,
            outer: alloc(ctx, &[1, TXT_HIDDEN])?,
            final_norm: alloc(ctx, &[1, TXT_HIDDEN])?,
            logits: alloc(ctx, &[1, VOCAB])?,
            // I32, unlike every other buffer (BF16): `alloc` pins DTYPE.
            index: ctx.allocate_output(Shape::new(vec![1]), DType::I32)?,
            token: ctx.allocate_output(Shape::new(vec![1]), DType::I32)?,
            positions: ctx.allocate_output(Shape::new(vec![1]), DType::I32)?,
            decode_meta: KvCacheDecodeMeta::new(ctx.device_id(), KV_CAPACITY)
                .map_err(|e| Error::Other(format!("decode meta alloc: {e}")))?,
        })
    }

    /// Rewrite the persistent token tensor from the host (synchronous 4-byte
    /// H2D; called between graph replays / eager steps, never during capture).
    pub(crate) fn set_token(&mut self, token: i32) -> Result<()> {
        let buffer = CudaBuffer::from_tensor(&self.token).map_err(Error::Cuda)?;
        buffer
            .copy_from_host(&token.to_le_bytes())
            .map_err(|e| Error::Other(format!("token upload: {e}")))?;
        Ok(())
    }

    /// Rewrite the persistent position tensor from the host (same contract).
    pub(crate) fn set_position(&mut self, position: usize) -> Result<()> {
        let buffer = CudaBuffer::from_tensor(&self.positions).map_err(Error::Cuda)?;
        buffer
            .copy_from_host(&(position as i32).to_le_bytes())
            .map_err(|e| Error::Other(format!("position upload: {e}")))?;
        Ok(())
    }
}

/// Persistent workspace for the prefill (m=T) path.
///
/// A steady-state transcription profile showed roughly a thousand
/// `cudaMalloc`/`cudaFree` calls per transcription in the prefill segment
/// (one fresh allocation per intermediate per layer), one of the main costs of
/// the prefill and encode segments. Decode already solved the same problem with
/// [`StepWorkspace`]: fixed output addresses drop the allocations and make the
/// execution cache — keyed on the raw a/b/out addresses — recur between
/// transcriptions, so nothing is prepared twice. This workspace applies that
/// pattern to prefill: every intermediate goes into a persistent buffer reused
/// across transcriptions, and the numerical path stays operator-for-operator
/// identical to the fresh-allocation version (only the output buffer moves).
///
/// The shape follows the prompt length T and the model constants; `tokens` is
/// the shape signature. A different audio length changes n_audio and with it T,
/// and the caller drops and rebuilds the whole workspace, so any input up to
/// 30 s stays correct. In steady state (repeated transcriptions of
/// equal-length audio) nothing is rebuilt and nothing is allocated fresh.
///
/// `hidden` is a rotating double buffer with the same discipline as
/// [`StepWorkspace`]: the residual takes a shallow clone of the current slot
/// (shared storage) and the residual add writes the other slot, so an add
/// output never aliases either input (see `add_into`); `hidden[0]` doubles as
/// the embedding gather output, including the audio splice rows.
pub(crate) struct PrefillWorkspace {
    /// Shape signature: the prompt length T.
    pub(crate) tokens: usize,
    /// Rotating hidden double buffer `[T, TXT_HIDDEN]`; `hidden[0]` holds the embed output.
    pub(crate) hidden: [Tensor; 2],
    /// input_layernorm output `[T, TXT_HIDDEN]`, the a operand of the q/k/v GEMMs.
    pub(crate) normed: Tensor,
    /// q projection output `[T, TXT_HIDDEN]`, before per-head norm and rope.
    pub(crate) q: Tensor,
    /// k projection output `[T, TXT_KV_HEADS * TXT_HEAD_DIM]`.
    pub(crate) k: Tensor,
    /// v projection output `[T, TXT_KV_HEADS * TXT_HEAD_DIM]`; kept apart from
    /// k because both are live while the cache rows are written.
    pub(crate) v: Tensor,
    /// Per-head normalized q, `[T * TXT_HEADS, TXT_HEAD_DIM]`: the rank-2 shape
    /// rms_norm writes, with the `[T, H, D]` head layout taken as a view at the
    /// use site.
    pub(crate) q_normed: Tensor,
    /// Per-head normalized k, `[T * TXT_KV_HEADS, TXT_HEAD_DIM]`.
    pub(crate) k_normed: Tensor,
    /// Attention output `[T, TXT_HIDDEN]`, the a operand of the o projection.
    pub(crate) attn_out: Tensor,
    /// o projection output `[T, TXT_HIDDEN]`.
    pub(crate) attn_proj: Tensor,
    /// post_attention_layernorm output `[T, TXT_HIDDEN]`, the a operand of gate_up.
    pub(crate) normed2: Tensor,
    /// Fused gate_up output `[T, 2 * TXT_INTER]`.
    pub(crate) fused: Tensor,
    /// SwiGLU output `[T, TXT_INTER]`, the a operand of the down projection.
    pub(crate) inner: Tensor,
    /// down projection output `[T, TXT_HIDDEN]`.
    pub(crate) outer: Tensor,
    /// Final norm output `[T, TXT_HIDDEN]`; lm_head reads only its last row.
    pub(crate) final_norm: Tensor,
    /// lm_head logits `[1, VOCAB]`; prefill computes logits for the last row only.
    pub(crate) logits: Tensor,
    /// argmax output, `[1]` I32.
    pub(crate) index: Tensor,
    /// This prompt's token ids, `[T]` I32: a persistent device buffer the host
    /// rewrites each transcription, which saves the per-transcription upload
    /// and keeps the gather input address stable.
    pub(crate) ids: Tensor,
    /// Positions, `[T]` I32: same, read by RoPE; the entry point rewrites them
    /// as `[kv_offset, kv_offset + T)`.
    pub(crate) positions: Tensor,
}

/// Build or rebuild the persistent prefill workspace for `tokens` rows.
///
/// Must be called outside any execution session: workspace buffers are
/// caller-owned and live across transcriptions, while `allocate_output` inside
/// a session sub-allocates from that session's deterministic arena (see the
/// model's `StageSession`).
pub(crate) fn ensure_prefill_workspace(
    ctx: &Context,
    ws: &mut Option<PrefillWorkspace>,
    tokens: usize,
) -> Result<()> {
    if ws.as_ref().map_or(true, |w| w.tokens != tokens) {
        *ws = Some(PrefillWorkspace::alloc(ctx, tokens)?);
    }
    Ok(())
}

impl PrefillWorkspace {
    pub fn alloc(ctx: &Context, tokens: usize) -> Result<Self> {
        let kv = TXT_KV_HEADS * TXT_HEAD_DIM;
        Ok(Self {
            tokens,
            hidden: [
                alloc(ctx, &[tokens, TXT_HIDDEN])?,
                alloc(ctx, &[tokens, TXT_HIDDEN])?,
            ],
            normed: alloc(ctx, &[tokens, TXT_HIDDEN])?,
            q: alloc(ctx, &[tokens, TXT_HIDDEN])?,
            k: alloc(ctx, &[tokens, kv])?,
            v: alloc(ctx, &[tokens, kv])?,
            q_normed: alloc(ctx, &[tokens * TXT_HEADS, TXT_HEAD_DIM])?,
            k_normed: alloc(ctx, &[tokens * TXT_KV_HEADS, TXT_HEAD_DIM])?,
            attn_out: alloc(ctx, &[tokens, TXT_HIDDEN])?,
            attn_proj: alloc(ctx, &[tokens, TXT_HIDDEN])?,
            normed2: alloc(ctx, &[tokens, TXT_HIDDEN])?,
            fused: alloc(ctx, &[tokens, 2 * TXT_INTER])?,
            inner: alloc(ctx, &[tokens, TXT_INTER])?,
            outer: alloc(ctx, &[tokens, TXT_HIDDEN])?,
            final_norm: alloc(ctx, &[tokens, TXT_HIDDEN])?,
            logits: alloc(ctx, &[1, VOCAB])?,
            // I32: `alloc` pins BF16.
            index: ctx.allocate_output(Shape::new(vec![1]), DType::I32)?,
            ids: ctx.allocate_output(Shape::new(vec![tokens]), DType::I32)?,
            positions: ctx.allocate_output(Shape::new(vec![tokens]), DType::I32)?,
        })
    }

    /// Host rewrite of the ids buffer (a synchronous H2D copy, called between
    /// transcriptions only, never during capture).
    pub(crate) fn set_ids(&mut self, ids: &[i32]) -> Result<()> {
        if ids.len() != self.tokens {
            return Err(Error::Other(format!(
                "prefill ids len {} does not match workspace signature {}",
                ids.len(),
                self.tokens
            )));
        }
        let bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buffer = CudaBuffer::from_tensor(&self.ids).map_err(Error::Cuda)?;
        buffer
            .copy_from_host(&bytes)
            .map_err(|e| Error::Other(format!("prefill ids upload: {e}")))?;
        Ok(())
    }

    /// Host rewrite of the positions buffer to `[kv_offset, kv_offset + tokens)`.
    pub(crate) fn set_positions(&mut self, kv_offset: usize, tokens: usize) -> Result<()> {
        if tokens != self.tokens {
            return Err(Error::Other(format!(
                "prefill positions len {tokens} does not match workspace signature {}",
                self.tokens
            )));
        }
        let values: Vec<i32> = (kv_offset as i32..(kv_offset + tokens) as i32).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buffer = CudaBuffer::from_tensor(&self.positions).map_err(Error::Cuda)?;
        buffer
            .copy_from_host(&bytes)
            .map_err(|e| Error::Other(format!("prefill positions upload: {e}")))?;
        Ok(())
    }
}

/// Intermediate tensors captured for prefill parity tests.
///
/// Stage captures always go through [`deep_clone`]: prefill workspace buffers
/// are overwritten in place by later layers and by the next transcription, so a
/// shallow clone would follow those writes (unlike the fresh-allocation era,
/// where a cloned tensor was never written again).
pub struct DecoderStages {
    /// Hidden after layer 0, `[T, 2048]`.
    pub layer0: Tensor,
    /// Hidden after the last layer, pre final-norm, `[T, 2048]`.
    pub layer27: Tensor,
    /// Final-norm output, `[T, 2048]`.
    pub final_norm: Tensor,
    // Layer-0 internals (parity debugging, valid only for prefill calls).
    /// rms_norm(embed) output.
    pub normed0: Tensor,
    /// q/k/v projection outputs.
    pub q0: Tensor,
    pub k0: Tensor,
    pub v0: Tensor,
    /// q/k after per-head norm + rope (row-major `[T, H, D]` order).
    pub qrot0: Tensor,
    pub krot0: Tensor,
    /// Attention output (pre o-projection), `[T, 2048]`.
    pub attn0: Tensor,
    /// o-projection output (pre residual add), `[T, 2048]`.
    pub o_proj0: Tensor,
    /// post_attention_layernorm output.
    pub normed2_0: Tensor,
    /// Fused gate+up GEMM output, `[T, 12288]`.
    pub gate_up0: Tensor,
    /// SwiGLU output, `[T, 6144]`.
    pub inner0: Tensor,
}

/// Stateless Qwen3 text decoder; all mutable state lives in [`KvCaches`]
/// (prefill intermediates live in the caller's [`PrefillWorkspace`]).
pub struct Decoder;

impl Decoder {
    pub fn new() -> Self {
        Self
    }

    /// Prefill `ids` (length T) with `audio_embeds` `[n_audio, 2048]` copied
    /// over the token rows `[audio_offset, audio_offset + n_audio)`.
    /// Returns the first greedily-selected next token.
    ///
    /// Every intermediate goes into `prefill_ws`'s persistent buffers (see
    /// [`PrefillWorkspace`]): in steady state (equal prompt lengths) the
    /// per-transcription cudaMalloc count drops to zero and the bound addresses
    /// recur in the execution cache; the caller drops and rebuilds the
    /// workspace when the signature T changes. The KV cache write path is
    /// unchanged.
    pub(crate) fn prefill(
        &self,
        ctx: &Context,
        weights: &mut Weights,
        caches: &mut KvCaches,
        prefill_ws: &mut Option<PrefillWorkspace>,
        ids: &[i32],
        audio_embeds: &Tensor,
        audio_offset: usize,
        n_audio: usize,
    ) -> Result<i32> {
        let tokens = ids.len();
        if audio_offset + n_audio > tokens {
            return Err(Error::Other(format!(
                "audio region [{audio_offset}, {}+{n_audio}) overruns the prompt length {tokens}",
                audio_offset
            )));
        }
        if audio_embeds.shape().dims() != [n_audio, TXT_HIDDEN] {
            return Err(Error::Other(format!(
                "audio embeds shape {:?} does not match [{n_audio}, {TXT_HIDDEN}]",
                audio_embeds.shape().dims()
            )));
        }

        // A different shape signature (prompt length T) drops and rebuilds the
        // workspace, which keeps any input up to 30 s correct; equal lengths
        // rebuild nothing and keep every address.
        if prefill_ws.as_ref().map_or(true, |w| w.tokens != tokens) {
            *prefill_ws = Some(PrefillWorkspace::alloc(ctx, tokens)?);
        }
        let ws = prefill_ws.as_mut().expect("allocated above");

        self.embed_into(ctx, weights, ids, ws)?;
        // Splice the projected audio embeddings over the placeholder rows.
        let dest = view_rows(&ws.hidden[0], audio_offset, n_audio, TXT_HIDDEN)?;
        copy_cuda_to_cuda(ctx, audio_embeds, &dest)
            .map_err(|e| Error::Other(format!("audio splice: {e}")))?;

        self.layers_staged(ctx, weights, caches, ws, 0, true, None)?;
        // Last row -> logits -> argmax, all into persistent buffers (prefill
        // computes logits for the last row's view only).
        self.logits_argmax_into(
            ctx,
            weights,
            &ws.final_norm,
            tokens - 1,
            &mut ws.logits,
            &ws.index,
        )
    }

    /// The embedding gather writes into the persistent `ws.hidden[0]`, and the
    /// prompt ids are rewritten into the persistent `ws.ids` (no
    /// per-transcription allocation, stable gather input address).
    pub(crate) fn embed_into(
        &self,
        ctx: &Context,
        weights: &mut Weights,
        ids: &[i32],
        ws: &mut PrefillWorkspace,
    ) -> Result<()> {
        let table = weights.get(ctx, "text.embed.w")?;
        ws.set_ids(ids)?;
        ops::embedding_gather(ctx, &table, &ws.ids, &ws.hidden[0])
            .map_err(|e| Error::Other("embedding_gather: ".to_owned() + &e.to_string()))?;
        Ok(())
    }

    /// Run the 28 transformer layers over the persistent prefill workspace.
    /// `kv_offset` is the cache row the new tokens start at; `causal` selects
    /// the prefill attention mask. The entry point rewrites `ws.positions` to
    /// `[kv_offset, kv_offset + T)`; hidden rotates through the `ws.hidden`
    /// double buffer under [`StepWorkspace`]'s discipline. The KV cache write
    /// path is unchanged.
    ///
    /// The `stages` capture path stays available: captures always deep-clone,
    /// because workspace buffers are overwritten in place and a shallow clone
    /// would follow those writes. Only prefill calls this; the decode loops
    /// live in `model_runner::decode`.
    pub(crate) fn layers_staged(
        &self,
        ctx: &Context,
        weights: &mut Weights,
        caches: &mut KvCaches,
        ws: &mut PrefillWorkspace,
        kv_offset: usize,
        causal: bool,
        mut stages: Option<&mut DecoderStages>,
    ) -> Result<()> {
        let tokens = ws.tokens;
        let kv_len = kv_offset + tokens;
        if kv_len > KV_CAPACITY {
            return Err(Error::Other(format!(
                "kv cache overflow: {kv_len} rows exceed capacity {KV_CAPACITY}"
            )));
        }
        ws.set_positions(kv_offset, tokens)?;
        let mut cur = 0usize;
        for layer in 0..TXT_LAYERS {
            let p = format!("text.layers.{layer}");
            // The residual is a shallow clone of the current slot (shared
            // storage); this layer's add writes the other slot, which keeps the
            // snapshot safe - the same rotation as the decode steps.
            let residual = ws.hidden[cur].clone();
            let iln_w = weights.get(ctx, &format!("{p}.input_layernorm.w"))?;
            rms_norm_into(ctx, &ws.hidden[cur], &iln_w, &mut ws.normed)?;
            if layer == 0 {
                if let Some(st) = stages.as_deref_mut() {
                    st.normed0 = deep_clone(ctx, &ws.normed)?;
                }
            }

            // Q/K/V projections (no bias), all into persistent buffers.
            let qw = weights.get(ctx, &format!("{p}.self_attn.q.w"))?;
            let kw = weights.get(ctx, &format!("{p}.self_attn.k.w"))?;
            let vw = weights.get(ctx, &format!("{p}.self_attn.v.w"))?;
            gemm_into(ctx, &ws.normed, &qw, &mut ws.q)?; // [T, 2048]
            gemm_into(ctx, &ws.normed, &kw, &mut ws.k)?; // [T, 1024]
            gemm_into(ctx, &ws.normed, &vw, &mut ws.v)?; // [T, 1024]
            if layer == 0 {
                if let Some(st) = stages.as_deref_mut() {
                    st.q0 = deep_clone(ctx, &ws.q)?;
                    st.k0 = deep_clone(ctx, &ws.k)?;
                    st.v0 = deep_clone(ctx, &ws.v)?;
                }
            }

            // Per-head q_norm / k_norm. Qwen3's RMSNorm multiplies the weight
            // directly (plain-weight semantics), so the rank-2 RMSNorm over
            // the flattened heads applies here. The numerical path matches
            // the fresh-allocation version: rank-2
            // q/k views -> rms_norm into ws.q_normed/k_normed -> reshape to
            // [T, H, D].
            let q_flat = ws
                .q
                .reshape(Shape::new(vec![tokens * TXT_HEADS, TXT_HEAD_DIM]))?;
            let k_flat = ws
                .k
                .reshape(Shape::new(vec![tokens * TXT_KV_HEADS, TXT_HEAD_DIM]))?;
            let q_norm_w = weights.get(ctx, &format!("{p}.self_attn.q_norm.w"))?;
            let k_norm_w = weights.get(ctx, &format!("{p}.self_attn.k_norm.w"))?;
            // Fused per-head RMSNorm + full-width rope: bit-identical to the
            // rms_norm_into + partial_rope pair it replaces (norm.cuh
            // plain-weight semantics; grid dim3(tokens, heads) covers T > 1).
            ops::qk_norm_rope(
                ctx,
                &q_flat,
                &q_norm_w,
                &mut ws.q_normed,
                &ws.positions,
                TXT_HEADS,
                TXT_HEAD_DIM,
                TXT_HEAD_DIM,
                ROPE_THETA,
                RMS_EPS,
            )
            .map_err(|e| Error::Other(format!("layer {layer} q norm+rope: {e}")))?;
            ops::qk_norm_rope(
                ctx,
                &k_flat,
                &k_norm_w,
                &mut ws.k_normed,
                &ws.positions,
                TXT_KV_HEADS,
                TXT_HEAD_DIM,
                TXT_HEAD_DIM,
                ROPE_THETA,
                RMS_EPS,
            )
            .map_err(|e| Error::Other(format!("layer {layer} k norm+rope: {e}")))?;

            // Heads views are taken AFTER the fused kernel has written
            // q_normed/k_normed: a view built before the write could hand the
            // deep_clone/attention/cache stage stale pre-norm data.
            let q_heads = ws
                .q_normed
                .reshape(Shape::new(vec![tokens, TXT_HEADS, TXT_HEAD_DIM]))?;
            let k_heads = ws
                .k_normed
                .reshape(Shape::new(vec![tokens, TXT_KV_HEADS, TXT_HEAD_DIM]))?;
            if layer == 0 {
                if let Some(st) = stages.as_deref_mut() {
                    st.qrot0 = deep_clone(ctx, &q_heads)?;
                    st.krot0 = deep_clone(ctx, &k_heads)?;
                }
            }

            // Write K/V into the cache rows [kv_offset, kv_offset + T).
            let k_src = k_heads.reshape(Shape::new(vec![1, tokens, TXT_KV_HEADS, TXT_HEAD_DIM]))?;
            let v_src = {
                let v_rows =
                    ws.v.reshape(Shape::new(vec![tokens, TXT_KV_HEADS, TXT_HEAD_DIM]))?;
                v_rows.reshape(Shape::new(vec![1, tokens, TXT_KV_HEADS, TXT_HEAD_DIM]))?
            };
            let k_dst = caches.key_view(layer, kv_offset, tokens)?;
            let v_dst = caches.value_view(layer, kv_offset, tokens)?;
            copy_cuda_to_cuda(ctx, &k_src, &k_dst)
                .map_err(|e| Error::Other(format!("layer {layer} k cache write: {e}")))?;
            copy_cuda_to_cuda(ctx, &v_src, &v_dst)
                .map_err(|e| Error::Other(format!("layer {layer} v cache write: {e}")))?;

            // Attention over the whole cached range, writing ws.attn_out.
            let k_cache = caches.key_view(layer, 0, kv_len)?;
            let v_cache = caches.value_view(layer, 0, kv_len)?;
            let q_attn = q_heads.reshape(Shape::new(vec![1, tokens, TXT_HEADS, TXT_HEAD_DIM]))?;
            {
                let mut out_heads = ws
                    .attn_out
                    .reshape(Shape::new(vec![1, tokens, TXT_HEADS, TXT_HEAD_DIM]))?;
                let mut args = AttentionArgs::new(&q_attn, &k_cache, &v_cache, &mut out_heads);
                if causal && tokens > 1 {
                    args = args.causal();
                }
                ops::attention(ctx, args).map_err(|e| {
                    Error::Other(format!("layer {layer} text attention (kv_len {kv_len}): {e}"))
                })?;
            }
            if layer == 0 {
                if let Some(st) = stages.as_deref_mut() {
                    st.attn0 = deep_clone(ctx, &ws.attn_out)?;
                }
            }
            let ow = weights.get(ctx, &format!("{p}.self_attn.o.w"))?;
            gemm_into(ctx, &ws.attn_out, &ow, &mut ws.attn_proj)?;
            if layer == 0 {
                if let Some(st) = stages.as_deref_mut() {
                    st.o_proj0 = deep_clone(ctx, &ws.attn_proj)?;
                }
            }
            add_into(ctx, &residual, &ws.attn_proj, &mut ws.hidden[(cur + 1) % 2])?;
            cur = (cur + 1) % 2;

            // MLP: fused gate+up -> SwiGLU -> down, all persistent buffers.
            let residual2 = ws.hidden[cur].clone();
            let pln_w = weights.get(ctx, &format!("{p}.post_attention_layernorm.w"))?;
            rms_norm_into(ctx, &ws.hidden[cur], &pln_w, &mut ws.normed2)?;
            if layer == 0 {
                if let Some(st) = stages.as_deref_mut() {
                    st.normed2_0 = deep_clone(ctx, &ws.normed2)?;
                }
            }
            let guw = weights.get(ctx, &format!("{p}.mlp.gate_up.w"))?;
            let dw = weights.get(ctx, &format!("{p}.mlp.down.w"))?;
            gemm_into(ctx, &ws.normed2, &guw, &mut ws.fused)?; // [T, 12288]
            if layer == 0 {
                if let Some(st) = stages.as_deref_mut() {
                    st.gate_up0 = deep_clone(ctx, &ws.fused)?;
                }
            }
            ops::swiglu_into(ctx, &ws.fused, &ws.inner)
                .map_err(|e| Error::Other(format!("layer {layer} swiglu: {e}")))?;
            if layer == 0 {
                if let Some(st) = stages.as_deref_mut() {
                    st.inner0 = deep_clone(ctx, &ws.inner)?;
                }
            }
            gemm_into(ctx, &ws.inner, &dw, &mut ws.outer)?;
            add_into(ctx, &residual2, &ws.outer, &mut ws.hidden[(cur + 1) % 2])?;
            cur = (cur + 1) % 2;
            if let Some(st) = stages.as_deref_mut() {
                if layer == 0 {
                    st.layer0 = deep_clone(ctx, &ws.hidden[cur])?;
                }
                if layer == TXT_LAYERS - 1 {
                    st.layer27 = deep_clone(ctx, &ws.hidden[cur])?;
                }
            }
        }

        let norm_w = weights.get(ctx, "text.norm.w")?;
        rms_norm_into(ctx, &ws.hidden[cur], &norm_w, &mut ws.final_norm)?;
        if let Some(st) = stages.as_deref_mut() {
            st.final_norm = deep_clone(ctx, &ws.final_norm)?;
        }
        Ok(())
    }

    /// Row `row` of `final_hidden` -> lm_head logits + greedy argmax, both
    /// written into caller-provided workspace buffers. Shared by the prefill
    /// hot path and the parity tests.
    pub(crate) fn logits_argmax_into(
        &self,
        ctx: &Context,
        weights: &mut Weights,
        final_hidden: &Tensor,
        row: usize,
        logits: &mut Tensor,
        index: &Tensor,
    ) -> Result<i32> {
        let last = view_rows(final_hidden, row, 1, TXT_HIDDEN)?;
        let lm_head = weights.get(ctx, "text.lm_head.w")?; // [2048, 151936]
        gemm_into(ctx, &last, &lm_head, logits)
            .map_err(|e| Error::Other(format!("lm_head gemm: {e}")))?;
        ops::argmax(ctx, ArgmaxArgs::new(logits, index))
            .map_err(|e| Error::Other(format!("argmax: {e}")))?;
        id_to_host(ctx, index)
    }
}
