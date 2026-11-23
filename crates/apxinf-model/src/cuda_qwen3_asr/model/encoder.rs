//! qwen3-asr audio tower + multimodal projector: BF16 on apxinf-cuda-new.
//!
//! Pipeline (HF `Qwen3ASREncoder.forward` semantics):
//! mel [3000,128] -> chunked conv2d stack (time as H, mel as W; weights
//! pre-swapped by the converter) -> conv_out Linear (column-permuted weight)
//! -> +sinusoidal pos -> valid-position packing -> 24 pre-norm layers with
//! windowed non-causal attention (windows of 104 positions) -> ln_post ->
//! projector (linear_1 + gelu + linear_2) -> [N, 2048].
//!
//! Every device-side intermediate of encode goes into [`EncodeWorkspace`]'s
//! persistent buffers, reused across transcriptions: steady-state
//! `cudaMalloc` counts drop to zero, and the bound addresses recur in the
//! execution cache of the relaxed `ExecutionSession` on every transcription.
//! The host-side valid-position packing logic is untouched; only the upload
//! target changed to a persistent buffer.

use apxinf_core::{DType, Error, Result, Shape, Tensor};
use apxinf_cuda_new::ops::{
    self, AttentionArgs, Conv2dArgs, GemmArgs, GemmBiasArgs, NormArgs,
    PointwiseActivation, PointwiseArgs, PointwiseSemantic,
};
use apxinf_cuda_new::transfers::copy_cuda_to_cuda;
use apxinf_cuda_new::{CudaBuffer, CudaContext as Context};

use crate::cuda_qwen3_asr::config::*;
use crate::cuda_qwen3_asr::weights::{DTYPE, Weights};

/// Packed valid positions per chunk (len N_CHUNKS).
fn chunk_post_cnn_lens(valid_frames: usize) -> Vec<usize> {
    (0..N_CHUNKS)
        .map(|i| {
            let start = i * CHUNK_FRAMES;
            let frames = valid_frames.saturating_sub(start).min(CHUNK_FRAMES);
            post_cnn_len(frames)
        })
        .collect()
}

fn alloc(ctx: &Context, dims: &[usize]) -> Result<Tensor> {
    ctx.allocate_output(Shape::new(dims.to_vec()), DTYPE)
}

/// GEMM (optional bias) into a caller-provided output buffer. Persistent
/// outputs keep the a/out addresses unchanged between transcriptions, which
/// hits the execution cache and drops the `cudaMalloc` calls; the caller (the
/// workspace's signature) guarantees the buffer shape is `[M, N]`.
fn gemm_bias_into(
    ctx: &Context,
    a: &Tensor,
    b: &Tensor,
    bias: Option<&Tensor>,
    out: &mut Tensor,
) -> Result<()> {
    let a_dims = a.shape().dims().to_vec();
    let b_dims = b.shape().dims().to_vec();
    if a_dims.len() != 2 || b_dims.len() != 2 || a_dims[1] != b_dims[0] {
        return Err(Error::Other(format!(
            "qwen3-asr gemm shape mismatch: a={a_dims:?} b={b_dims:?}"
        )));
    }
    match bias {
        Some(bias) => ops::gemm_bias(
            ctx,
            GemmBiasArgs {
                gemm: GemmArgs::new(a, b, out),
                bias,
            },
        )
        .map_err(|e| Error::Other(format!("gemm_bias a={a_dims:?} b={b_dims:?}: {e}")))?,
        None => ops::gemm(ctx, GemmArgs::new(a, b, out))
            .map_err(|e| Error::Other(format!("gemm a={a_dims:?} b={b_dims:?}: {e}")))?,
    }
    Ok(())
}

/// LayerNorm into a caller-provided output buffer (same shape as `input`).
/// See [`gemm_bias_into`] for why the outputs are persistent.
fn layer_norm_into(
    ctx: &Context,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    out: &mut Tensor,
) -> Result<()> {
    let dims = input.shape().dims().to_vec();
    ops::norm(ctx, NormArgs::layer(input, weight, bias, out, LN_EPS))
        .map_err(|e| Error::Other(format!("layer_norm in={dims:?}: {e}")))?;
    Ok(())
}

/// Exact (erf) GELU into a caller-provided output buffer: the input is
/// flattened along its last dimension into a zero-copy `[rows, cols]` rank-2
/// view, and `out` must already be a rank-2 buffer of the same shape (the
/// workspace's gelu buffers are allocated in exactly that flattened shape;
/// take a 4-D view at the use site when one is needed).
fn gelu_exact_into(ctx: &Context, input: &Tensor, out: &mut Tensor) -> Result<()> {
    let dims = input.shape().dims().to_vec();
    let cols = *dims
        .last()
        .ok_or_else(|| Error::Other("gelu needs a rank".into()))?;
    let rows = input.numel() / cols;
    let flat = input.reshape(Shape::new(vec![rows, cols]))?;
    let mut args = PointwiseArgs::new(PointwiseSemantic::BiasActivation, &flat, out);
    args.activation = PointwiseActivation::GeluExact;
    ops::pointwise(ctx, args)
        .map_err(|e| Error::Other(format!("gelu_exact in={dims:?}: {e}")))?;
    Ok(())
}

fn read_bf16(ctx: &Context, tensor: &Tensor) -> Result<Vec<f32>> {
    if tensor.dtype() != DType::BF16 {
        return Err(Error::Other("encoder readback expects BF16".into()));
    }
    // Device-wide barrier: a legacy-default-stream memcpy has no ordering
    // against the crate's non-blocking context stream (see tests.rs).
    let _ = ctx;
    apxinf_cuda_new::transfers::synchronize_device()
        .map_err(|e| Error::Other(format!("readback sync: {e}")))?;
    let mut bytes = vec![0u8; tensor.numel() * 2];
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .copy_to_host(&mut bytes)
        .map_err(|e| Error::Other(format!("readback: {e}")))?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}

/// Row-range view `[count, cols]` of a contiguous BF16 `[rows, cols]` tensor.
fn view_rows(tensor: &Tensor, start: usize, count: usize, cols: usize) -> Result<Tensor> {
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    let view = buffer
        .view(start * cols * 2, count * cols * 2)
        .map_err(|e| Error::Other(format!("view_rows: {e}")))?;
    view.as_tensor(Shape::new(vec![count, cols]), DTYPE)
        .map_err(|e| Error::Other(format!("view_rows as_tensor: {e}")))
}

/// Deep copy (a fresh device allocation): `Tensor::clone` shares storage and
/// workspace buffers are overwritten in place, so capturing a stage tensor out
/// of the workspace must deep-copy.
fn deep_clone(ctx: &Context, tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims().to_vec();
    let out = alloc(ctx, &dims)?;
    copy_cuda_to_cuda(ctx, tensor, &out).map_err(|e| Error::Other(format!("deep_clone: {e}")))?;
    Ok(out)
}

/// Elementwise `a + b` into a caller-provided output buffer, with the same
/// aliasing discipline as the decoder: `out` never shares storage with either
/// input. In the workspace `out` is always the hidden slot on the other side of
/// the rotation, which aliases neither the residual (a shallow clone of the
/// current slot) nor the projected activations (independent buffers).
fn add_into(ctx: &Context, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let dims = a.shape().dims().to_vec();
    if dims != b.shape().dims() {
        return Err(Error::Other(format!(
            "add shape mismatch {dims:?} vs {:?}",
            b.shape().dims()
        )));
    }
    copy_cuda_to_cuda(ctx, a, out).map_err(|e| Error::Other(format!("add copy: {e}")))?;
    // add_into: accumulator += addend, in place.
    ops::add_into(ctx, b, out).map_err(|e| Error::Other(format!("add: {e}")))?;
    Ok(())
}

/// Intermediate tensors captured for stage parity tests.
/// Captures always go through `deep_clone`: workspace buffers are overwritten
/// in place.
pub struct EncoderStages {
    /// Conv3 output, flattened `[390, 16*480]` (freq-major columns).
    pub conv3_out: Tensor,
    /// conv_out GEMM output (before packing), `[390, 1024]`.
    pub conv_out: Tensor,
    /// Packed conv_out + positional embedding, `[N, 1024]`.
    pub packed: Tensor,
    /// ln_post output, `[N, 1024]`.
    pub ln_post: Tensor,
}

pub struct Encoded {
    /// Projector output, `[N, 2048]` BF16. This points into the workspace's
    /// persistent buffer: prefill consumes it before the next encode overwrites
    /// it.
    pub projected: Tensor,
    /// Packed valid positions.
    pub n: usize,
}

/// Persistent workspace for the audio tower (encode) path.
///
/// A steady-state transcription profile showed roughly a thousand
/// `cudaMalloc` calls per transcription in the prefill and encode segments
/// (one fresh allocation per intermediate per layer). Decode already solved
/// the same problem with its `StepWorkspace`; this workspace applies that
/// pattern to encode: the conv stack, `conv_out`, every layer's
/// norm/q/k/v/attention/FFN, ln_post and the projector all write into
/// persistent buffers reused across transcriptions, so steady-state
/// `cudaMalloc` counts drop to zero and the a/out addresses recur in the
/// execution cache of the relaxed session.
///
/// The conv stack's shapes follow only from the 30 s full-window mel (short
/// audio is zero-padded to the full window on the host), so the
/// conv/gelu/conv_out buffers are constant; the n-dependent buffers are
/// rebuilt per signature. The shape signature is `(valid_frames, n)`, where n
/// is the packed row count; the two together determine the window layout
/// (`attention_windows`). A mismatch (audio of a different length) drops and
/// rebuilds the workspace, so any input up to 30 s stays correct.
///
/// `hidden` follows the same rotating double-buffer discipline as the
/// decoder's `StepWorkspace`: the residual is a shallow clone of the current
/// slot and the residual add writes the other; `hidden[0]` doubles as the
/// upload target for the host packing (layer 0's input, i.e. the packed
/// rows).
pub(crate) struct EncodeWorkspace {
    /// Shape signature `(valid_frames, n)`.
    sig: (usize, usize),
    /// conv1 output `[30, 50, 64, 480]`.
    conv1: Tensor,
    /// gelu after conv1 (flattened `[96000, 480]`; take the 4-D view at the
    /// use site).
    gelu1: Tensor,
    /// conv2 output `[30, 25, 32, 480]`.
    conv2: Tensor,
    /// gelu after conv2 (flattened `[12000, 480]`).
    gelu2: Tensor,
    /// conv3 output `[30, 13, 16, 480]`.
    conv3: Tensor,
    /// gelu after conv3 (flattened `[6240, 480]`, i.e. the row-major
    /// `[390, 7680]`).
    gelu3: Tensor,
    /// conv_out GEMM output `[390, 1024]`.
    conv_out: Tensor,
    /// Rotating hidden double buffer `[n, ENC_DMODEL]`; `hidden[0]` doubles as
    /// the packed upload target.
    hidden: [Tensor; 2],
    /// Each layer's self_attn_layer_norm output `[n, ENC_DMODEL]`.
    normed: Tensor,
    /// q projection output `[n, ENC_DMODEL]`.
    q: Tensor,
    /// k projection output `[n, ENC_DMODEL]`.
    k: Tensor,
    /// v projection output `[n, ENC_DMODEL]`; all three projections are live
    /// at once during the windowed attention, each in its own buffer.
    v: Tensor,
    /// Windowed attention output `[n, ENC_DMODEL]` (each window view writes
    /// its row range directly; the windows together cover all n rows).
    attn_out: Tensor,
    /// o projection output `[n, ENC_DMODEL]`.
    attn_projected: Tensor,
    /// Each layer's final_layer_norm output `[n, ENC_DMODEL]`.
    normed2: Tensor,
    /// fc1 output `[n, ENC_FF]`.
    inner: Tensor,
    /// gelu after fc1 `[n, ENC_FF]`.
    inner_gelu: Tensor,
    /// fc2 output `[n, ENC_DMODEL]`.
    outer: Tensor,
    /// ln_post output `[n, ENC_DMODEL]`.
    ln_post_out: Tensor,
    /// projector linear_1 output `[n, ENC_DMODEL]` (1024 -> 1024).
    proj1: Tensor,
    /// projector gelu output `[n, ENC_DMODEL]`.
    proj1_gelu: Tensor,
    /// projector linear_2 output `[n, OUTPUT_DIM]`.
    projected: Tensor,
}

/// Audio token count for `valid_frames`: the sum of the per-chunk packed
/// lengths. The prompt length is a pure function of this.
pub(crate) fn audio_token_count(valid_frames: usize) -> usize {
    chunk_post_cnn_lens(valid_frames).iter().sum()
}

/// Build or rebuild the persistent audio-tower workspace for `valid_frames`.
///
/// Must be called outside any [`ExecutionSession`]: workspace buffers are
/// caller-owned and live across transcriptions, while `allocate_output` inside
/// a session sub-allocates from the session's deterministic arena, which is
/// reserved for operator-internal scratch (see the model's `StageSession`).
pub(crate) fn ensure_workspace(
    ctx: &Context,
    ws: &mut Option<EncodeWorkspace>,
    valid_frames: usize,
) -> Result<()> {
    let n = audio_token_count(valid_frames);
    if ws.as_ref().map_or(true, |w| !w.matches(valid_frames, n)) {
        *ws = Some(EncodeWorkspace::alloc(ctx, valid_frames, n)?);
    }
    Ok(())
}

impl EncodeWorkspace {
    pub fn alloc(ctx: &Context, valid_frames: usize, n: usize) -> Result<Self> {
        // Conv stack shape derivation (matching the per-stage computation in
        // encode): k=3, s=2, p=1.
        let c1_h = (CHUNK_FRAMES + 2 - CONV_K) / 2 + 1; // 50
        let c1_w = (MEL_BINS + 2 - CONV_K) / 2 + 1; // 64
        let c2_h = (c1_h + 2 - CONV_K) / 2 + 1; // 25
        let c2_w = (c1_w + 2 - CONV_K) / 2 + 1; // 32
        let c3_h = (c2_h + 2 - CONV_K) / 2 + 1; // 13 == POST_CNN_TIME
        let c3_w = (c2_w + 2 - CONV_K) / 2 + 1; // 16 == FREQ_BINS
        debug_assert_eq!(c3_h, POST_CNN_TIME);
        debug_assert_eq!(c3_w, FREQ_BINS);
        let rows1 = N_CHUNKS * c1_h * c1_w;
        let rows2 = N_CHUNKS * c2_h * c2_w;
        let rows3 = N_CHUNKS * c3_h * c3_w;
        Ok(Self {
            sig: (valid_frames, n),
            conv1: alloc(ctx, &[N_CHUNKS, c1_h, c1_w, CONV_CH])?,
            gelu1: alloc(ctx, &[rows1, CONV_CH])?,
            conv2: alloc(ctx, &[N_CHUNKS, c2_h, c2_w, CONV_CH])?,
            gelu2: alloc(ctx, &[rows2, CONV_CH])?,
            conv3: alloc(ctx, &[N_CHUNKS, c3_h, c3_w, CONV_CH])?,
            gelu3: alloc(ctx, &[rows3, CONV_CH])?,
            conv_out: alloc(ctx, &[N_CHUNKS * POST_CNN_TIME, ENC_DMODEL])?,
            hidden: [
                alloc(ctx, &[n, ENC_DMODEL])?,
                alloc(ctx, &[n, ENC_DMODEL])?,
            ],
            normed: alloc(ctx, &[n, ENC_DMODEL])?,
            q: alloc(ctx, &[n, ENC_DMODEL])?,
            k: alloc(ctx, &[n, ENC_DMODEL])?,
            v: alloc(ctx, &[n, ENC_DMODEL])?,
            attn_out: alloc(ctx, &[n, ENC_DMODEL])?,
            attn_projected: alloc(ctx, &[n, ENC_DMODEL])?,
            normed2: alloc(ctx, &[n, ENC_DMODEL])?,
            inner: alloc(ctx, &[n, ENC_FF])?,
            inner_gelu: alloc(ctx, &[n, ENC_FF])?,
            outer: alloc(ctx, &[n, ENC_DMODEL])?,
            ln_post_out: alloc(ctx, &[n, ENC_DMODEL])?,
            proj1: alloc(ctx, &[n, ENC_DMODEL])?,
            proj1_gelu: alloc(ctx, &[n, ENC_DMODEL])?,
            projected: alloc(ctx, &[n, OUTPUT_DIM])?,
        })
    }

    /// Whether the shape signature matches (reuse) or not (drop and rebuild).
    fn matches(&self, valid_frames: usize, n: usize) -> bool {
        self.sig == (valid_frames, n)
    }
}

/// Run the full audio tower + projector. `mel` is `[3000, 128]` BF16 with
/// padding frames zeroed; `valid_frames` counts the live frames.
///
/// `ws` is the model-level persistent workspace; this function builds or
/// rebuilds the `Option` lazily per signature.
pub fn encode(
    ctx: &Context,
    weights: &mut Weights,
    ws: &mut Option<EncodeWorkspace>,
    mel: &Tensor,
    valid_frames: usize,
    mut stages: Option<&mut EncoderStages>,
) -> Result<Encoded> {
    if mel.shape().dims() != [FRAMES, MEL_BINS] || mel.dtype() != DTYPE {
        return Err(Error::Other(format!(
            "encode expects mel [{FRAMES}, {MEL_BINS}] {DTYPE}, got {:?} {:?}",
            mel.shape().dims(),
            mel.dtype()
        )));
    }

    // --- The host packing inputs (n / window layout) are computed on the host
    // first.
    let chunk_lens = chunk_post_cnn_lens(valid_frames);
    let n: usize = chunk_lens.iter().sum();

    // A shape-signature mismatch drops and rebuilds the workspace; equal
    // steady-state lengths rebuild nothing and keep every address.
    if ws.as_ref().map_or(true, |w| !w.matches(valid_frames, n)) {
        *ws = Some(EncodeWorkspace::alloc(ctx, valid_frames, n)?);
    }
    let ws = ws.as_mut().expect("allocated above");

    // --- Conv stack. Chunks are the batch axis; time is H, mel is W.
    // [3000,128] -> [30,100,128,1] is a zero-copy reinterpretation because
    // (chunk, t, m) row-major == (t*128 + m) element order per chunk.
    // Each conv writes its persistent conv buffer and each gelu its persistent
    // flattened buffer (numerical path unchanged).
    let in0 = mel.reshape(Shape::new(vec![N_CHUNKS, CHUNK_FRAMES, MEL_BINS, 1]))?;
    {
        let w = weights.get(ctx, "audio.conv2d1.w")?;
        let b = weights.get(ctx, "audio.conv2d1.b")?;
        let dims = in0.shape().dims().to_vec();
        ops::conv2d(
            ctx,
            Conv2dArgs::new(&in0, &w, &mut ws.conv1)
                .with_bias(&b)
                .with_stride(2, 2)
                .with_padding(1, 1),
        )
        .map_err(|e| Error::Other(format!("conv2d1 in={dims:?}: {e}")))?;
    }
    gelu_exact_into(ctx, &ws.conv1, &mut ws.gelu1)?;
    // 50 = (100+2-3)/2+1, 64 = (128+2-3)/2+1 (k=3 s=2 p=1).
    let in1 = ws
        .gelu1
        .reshape(Shape::new(vec![N_CHUNKS, 50, 64, CONV_CH]))?;
    {
        let w = weights.get(ctx, "audio.conv2d2.w")?;
        let b = weights.get(ctx, "audio.conv2d2.b")?;
        let dims = in1.shape().dims().to_vec();
        ops::conv2d(
            ctx,
            Conv2dArgs::new(&in1, &w, &mut ws.conv2)
                .with_bias(&b)
                .with_stride(2, 2)
                .with_padding(1, 1),
        )
        .map_err(|e| Error::Other(format!("conv2d2 in={dims:?}: {e}")))?;
    }
    gelu_exact_into(ctx, &ws.conv2, &mut ws.gelu2)?;
    // 25 = (50+2-3)/2+1, 32 = (64+2-3)/2+1.
    let in2 = ws
        .gelu2
        .reshape(Shape::new(vec![N_CHUNKS, 25, 32, CONV_CH]))?;
    {
        let w = weights.get(ctx, "audio.conv2d3.w")?;
        let b = weights.get(ctx, "audio.conv2d3.b")?;
        let dims = in2.shape().dims().to_vec();
        ops::conv2d(
            ctx,
            Conv2dArgs::new(&in2, &w, &mut ws.conv3)
                .with_bias(&b)
                .with_stride(2, 2)
                .with_padding(1, 1),
        )
        .map_err(|e| Error::Other(format!("conv2d3 in={dims:?}: {e}")))?;
    }
    gelu_exact_into(ctx, &ws.conv3, &mut ws.gelu3)?;
    // [30, 13, 16, 480] -> [390, 7680] (freq-major columns; the conv_out
    // weight was row-permuted at conversion to match).
    let flat = ws
        .gelu3
        .reshape(Shape::new(vec![N_CHUNKS * POST_CNN_TIME, CONV_OUT_IN]))?;
    if let Some(stages) = stages.as_deref_mut() {
        stages.conv3_out = deep_clone(ctx, &flat)?;
    }
    let conv_out_w = weights.get(ctx, "audio.conv_out.w")?;
    gemm_bias_into(ctx, &flat, &conv_out_w, None, &mut ws.conv_out)?; // [390, 1024]
    if let Some(stages) = stages.as_deref_mut() {
        stages.conv_out = deep_clone(ctx, &ws.conv_out)?;
    }

    // --- Host packing: select valid rows, add the sinusoidal embedding.
    // (The host-side logic is unchanged: read back conv_out and pos_emb,
    // assemble on the host.)
    let pos_emb = weights.get(ctx, "audio.pos_emb")?; // [13, 1024] BF16
    let pos_host = read_bf16(ctx, &pos_emb)?;
    let conv_host = read_bf16(ctx, &ws.conv_out)?; // [390, 1024] f32
    let mut packed = Vec::with_capacity(n * ENC_DMODEL);
    for (chunk, &len) in chunk_lens.iter().enumerate() {
        for t in 0..len {
            let src = (chunk * POST_CNN_TIME + t) * ENC_DMODEL;
            let pos = t * ENC_DMODEL;
            for c in 0..ENC_DMODEL {
                packed.push(conv_host[src + c] + pos_host[pos + c]);
            }
        }
    }
    // The upload target is now the persistent hidden[0] buffer (layer 0's
    // input is the packed rows). The bytes are identical to the original
    // upload_bf16's to_cuda copy; only the per-transcription device allocation
    // is gone.
    let bytes: Vec<u8> = packed
        .iter()
        .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
        .collect();
    CudaBuffer::from_tensor(&ws.hidden[0])
        .map_err(Error::Cuda)?
        .copy_from_host(&bytes)
        .map_err(|e| Error::Other(format!("packed upload: {e}")))?;
    if let Some(stages) = stages.as_deref_mut() {
        stages.packed = deep_clone(ctx, &ws.hidden[0])?;
    }

    // --- 24 windowed pre-norm layers (all writing persistent buffers).
    let windows = attention_windows(n, valid_frames);
    for layer in 0..ENC_LAYERS {
        encoder_layer(ctx, weights, layer, ws, &windows)?;
    }

    // --- ln_post.
    let ln_w = weights.get(ctx, "audio.ln_post.w")?;
    let ln_b = weights.get(ctx, "audio.ln_post.b")?;
    layer_norm_into(ctx, &ws.hidden[0], &ln_w, &ln_b, &mut ws.ln_post_out)?;
    if let Some(stages) = stages.as_deref_mut() {
        stages.ln_post = deep_clone(ctx, &ws.ln_post_out)?;
    }

    // --- Projector: linear_1 + gelu + linear_2.
    let p1_w = weights.get(ctx, "proj.linear_1.w")?;
    let p1_b = weights.get(ctx, "proj.linear_1.b")?;
    let p2_w = weights.get(ctx, "proj.linear_2.w")?;
    let p2_b = weights.get(ctx, "proj.linear_2.b")?;
    gemm_bias_into(ctx, &ws.ln_post_out, &p1_w, Some(&p1_b), &mut ws.proj1)?;
    gelu_exact_into(ctx, &ws.proj1, &mut ws.proj1_gelu)?;
    gemm_bias_into(ctx, &ws.proj1_gelu, &p2_w, Some(&p2_b), &mut ws.projected)?;

    Ok(Encoded {
        projected: ws.projected.clone(),
        n,
    })
}

/// One pre-norm encoder layer over the packed sequence with windowed
/// non-causal attention. Input and output are both `ws.hidden[0]` (the
/// attention block's intermediate rotates to `ws.hidden[1]`); every
/// intermediate goes into a persistent buffer.
fn encoder_layer(
    ctx: &Context,
    weights: &mut Weights,
    layer: usize,
    ws: &mut EncodeWorkspace,
    windows: &[usize],
) -> Result<()> {
    let p = format!("audio.layers.{layer}");

    // Self-attention block.
    // The residual is a shallow clone of the current slot (hidden[0]); the add
    // writes the other slot (hidden[1]).
    let residual = ws.hidden[0].clone();
    let saln_w = weights.get(ctx, &format!("{p}.self_attn_layer_norm.w"))?;
    let saln_b = weights.get(ctx, &format!("{p}.self_attn_layer_norm.b"))?;
    layer_norm_into(ctx, &ws.hidden[0], &saln_w, &saln_b, &mut ws.normed)?;

    let qw = weights.get(ctx, &format!("{p}.self_attn.q.w"))?;
    let qb = weights.get(ctx, &format!("{p}.self_attn.q.b"))?;
    let kw = weights.get(ctx, &format!("{p}.self_attn.k.w"))?;
    let kb = weights.get(ctx, &format!("{p}.self_attn.k.b"))?;
    let vw = weights.get(ctx, &format!("{p}.self_attn.v.w"))?;
    let vb = weights.get(ctx, &format!("{p}.self_attn.v.b"))?;
    gemm_bias_into(ctx, &ws.normed, &qw, Some(&qb), &mut ws.q)?;
    gemm_bias_into(ctx, &ws.normed, &kw, Some(&kb), &mut ws.k)?;
    gemm_bias_into(ctx, &ws.normed, &vw, Some(&vb), &mut ws.v)?;

    // Windowed non-causal attention: block-diagonal over the packed rows.
    // Each window is a contiguous row range; per-window attention over
    // `[1, len, heads, head_dim]` views reproduces the HF varlen kernel.
    // The window views are fixed-layout slices; out writes directly into the
    // matching window of ws.attn_out.
    let mut start = 0usize;
    for &len in windows {
        let q_win = window_heads(&ws.q, start, len, ENC_HEADS)?;
        let k_win = window_heads(&ws.k, start, len, ENC_HEADS)?;
        let v_win = window_heads(&ws.v, start, len, ENC_HEADS)?;
        let mut out_win = window_heads(&ws.attn_out, start, len, ENC_HEADS)?;
        ops::attention(ctx, AttentionArgs::new(&q_win, &k_win, &v_win, &mut out_win)).map_err(
            |e| Error::Other(format!("encoder layer {layer} window attn (len {len}): {e}")),
        )?;
        start += len;
    }

    let o_w = weights.get(ctx, &format!("{p}.self_attn.o.w"))?;
    let o_b = weights.get(ctx, &format!("{p}.self_attn.o.b"))?;
    gemm_bias_into(ctx, &ws.attn_out, &o_w, Some(&o_b), &mut ws.attn_projected)?;
    add_into(ctx, &residual, &ws.attn_projected, &mut ws.hidden[1])?;

    // FFN block. Input is the attention block's output (hidden[1]); the output
    // rotates back into hidden[0].
    let residual2 = ws.hidden[1].clone();
    let fln_w = weights.get(ctx, &format!("{p}.final_layer_norm.w"))?;
    let fln_b = weights.get(ctx, &format!("{p}.final_layer_norm.b"))?;
    layer_norm_into(ctx, &ws.hidden[1], &fln_w, &fln_b, &mut ws.normed2)?;
    let fc1_w = weights.get(ctx, &format!("{p}.fc1.w"))?;
    let fc1_b = weights.get(ctx, &format!("{p}.fc1.b"))?;
    let fc2_w = weights.get(ctx, &format!("{p}.fc2.w"))?;
    let fc2_b = weights.get(ctx, &format!("{p}.fc2.b"))?;
    gemm_bias_into(ctx, &ws.normed2, &fc1_w, Some(&fc1_b), &mut ws.inner)?;
    gelu_exact_into(ctx, &ws.inner, &mut ws.inner_gelu)?;
    gemm_bias_into(ctx, &ws.inner_gelu, &fc2_w, Some(&fc2_b), &mut ws.outer)?;
    add_into(ctx, &residual2, &ws.outer, &mut ws.hidden[0])?;
    Ok(())
}

/// `[1, len, heads, head_dim]` view of rows `[start, start+len)` of a
/// `[tokens, heads*head_dim]` BF16 tensor.
fn window_heads(tensor: &Tensor, start: usize, len: usize, heads: usize) -> Result<Tensor> {
    let cols = heads * ENC_HEAD_DIM;
    let rows = view_rows(tensor, start, len, cols)?;
    rows.reshape(Shape::new(vec![1, len, heads, ENC_HEAD_DIM]))
}
