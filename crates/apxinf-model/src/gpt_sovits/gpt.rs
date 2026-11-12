//! GPT (Text2Semantic) decoder for GPT-SoVITS V5.

use std::cell::RefCell;

use apxinf_core::{philox4x32_10, Error, Result, Shape, Tensor};
use apxinf_cuda_new::ops::{
    self as cuda_ops, AttentionMask, KvCacheAttentionArgs, KvCacheDecodeMeta,
};
use apxinf_cuda_new::{transfers, CudaBuffer, CudaContext as Context};

use super::config::{GPT_AUDIO_VOCAB, GPT_BERT_DIM, GPT_EOS, GPT_HEADS, GPT_HIDDEN, GPT_LAYERS};
use super::ops;
use super::weights::{fp16_tensor, ids_tensor, Linear, Weights};

const MAX_POSITIONS: usize = 4000;
const MIN_GENERATED_BEFORE_EOS: usize = 11;
const DECODE_WORKSPACE_BYTES: usize = 64 * 1024 * 1024;
// Prefill keeps one whole eager traversal of intermediates in the session
// arena. Measured peak is ~51 MB; the rest is headroom for longer prompts.
const PREFILL_WORKSPACE_BYTES: usize = 256 * 1024 * 1024;

struct Layer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ff1: Linear,
    ff2: Linear,
    ln1_w: Tensor,
    ln1_b: Tensor,
    ln2_w: Tensor,
    ln2_b: Tensor,
}

struct LayerCache {
    key: Tensor,
    value: Tensor,
}

struct DecodeCache {
    layers: Vec<LayerCache>,
    text_len: usize,
    audio_len: usize,
    capacity: usize,
}

// Keep the executable before its referenced storage so it is destroyed first.
// One latest capacity is retained; requests never retain another request's KV values.
struct DecodeResources {
    graph: Option<apxinf_cuda_new::CapturedGraph>,
    graph_logits: Option<Tensor>,
    session: cuda_ops::ExecutionSession,
    meta: KvCacheDecodeMeta,
    input: Tensor,
    cache: DecodeCache,
    embed: DecodeEmbedScratch,
}

/// Fixed buffers for the one-token decode embedding.
///
/// `embed_audio` allocates five tensors per call: two 4-byte id vectors, two
/// 2 KB gathered rows and the 2 KB sum, and the decode loop dropped all five
/// as soon as they had been copied into `decode_input`. Each free synchronizes
/// the device, so the step paid five allocator round trips and five drains for
/// 6 KB of scratch. Keeping them for the life of the decode resources removes
/// the allocation entirely and writes the sum straight into the graph input.
struct DecodeEmbedScratch {
    ids: Tensor,
    positions: Tensor,
    table_row: Tensor,
    position_row: Tensor,
}

impl DecodeEmbedScratch {
    fn new(ctx: &Context) -> Result<Self> {
        Ok(Self {
            ids: transfers::to_cuda(&Tensor::from_i32(vec![1], &[0])?, ctx.device_id())?,
            positions: transfers::to_cuda(&Tensor::from_i32(vec![1], &[0])?, ctx.device_id())?,
            table_row: ops::alloc(ctx, &[1, GPT_HIDDEN])?,
            position_row: ops::alloc(ctx, &[1, GPT_HIDDEN])?,
        })
    }
}

// Prepared prefill operators are keyed by operand address, so the two
// embedded inputs are staged into fixed storage and the whole traversal runs
// inside one session: the first pass prepares and caches native resources,
// every later pass with the same lengths replays the prepared sequence
// without allocating or synchronizing per operator.
struct PrefillResources {
    session: cuda_ops::ExecutionSession,
    text: Tensor,
    audio: Tensor,
    text_len: usize,
    audio_len: usize,
    capacity: usize,
    prepared: bool,
}

pub struct Gpt {
    decode_resources: RefCell<Option<DecodeResources>>,
    prefill_resources: RefCell<Option<PrefillResources>>,
    text_emb: Tensor,
    audio_emb: Tensor,
    bert_proj: Linear,
    text_pe: Tensor,
    audio_pe: Tensor,
    layers: Vec<Layer>,
    out: Tensor,
}

fn sine_pe(positions: usize, dim: usize, alpha: f32) -> Vec<f32> {
    let mut values = vec![0.0f32; positions * dim];
    for position in 0..positions {
        for pair in 0..dim / 2 {
            let div = (-(10000.0f64).ln() / dim as f64 * (2 * pair) as f64).exp() as f32;
            let argument = position as f32 * div;
            values[position * dim + 2 * pair] = argument.sin() * alpha;
            values[position * dim + 2 * pair + 1] = argument.cos() * alpha;
        }
    }
    values
}

fn cuda_view(tensor: &Tensor, byte_offset: usize, dims: &[usize]) -> Result<Tensor> {
    let bytes = dims
        .iter()
        .try_fold(tensor.dtype().size_in_bytes(), |bytes, dim| {
            bytes.checked_mul(*dim)
        })
        .ok_or_else(|| Error::Other("GPT CUDA view size overflow".into()))?;
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .view(byte_offset, bytes)
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(dims.to_vec()), tensor.dtype())
        .map_err(Error::Cuda)
}

fn cache_rows(cache: &Tensor, row_offset: usize, rows: usize) -> Result<Tensor> {
    let head_dim = GPT_HIDDEN / GPT_HEADS;
    let row_bytes = GPT_HIDDEN * cache.dtype().size_in_bytes();
    cuda_view(
        cache,
        row_offset
            .checked_mul(row_bytes)
            .ok_or_else(|| Error::Other("GPT cache offset overflow".into()))?,
        &[1, rows, GPT_HEADS, head_dim],
    )
}

fn logits_prefix(logits: &Tensor, vocab_size: usize) -> Result<Tensor> {
    cuda_view(logits, 0, &[1, vocab_size])
}

fn fp16(value: f32) -> f32 {
    half::f16::from_f32(value).to_f32()
}

/// The reference sampler advances the Philox counter once per decode step, so
/// step `n` draws from counter offset `n`.
fn philox_counter_offset(step: usize) -> u64 {
    step as u64
}

fn counter_exponential_fp16(seed: u64, counter_offset: u64, element: usize) -> f32 {
    let words = philox4x32_10(
        [
            counter_offset as u32,
            (counter_offset >> 32) as u32,
            element as u32,
            (element >> 32) as u32,
        ],
        [seed as u32, (seed >> 32) as u32],
    );
    let uniform =
        (f64::from(words[0]) * 2.328_306_436_538_696_3e-10 + 1.164_153_218_269_348_1e-10) as f32;
    let exponential = if uniform >= 1.0 - f32::EPSILON / 2.0 {
        f32::EPSILON / 2.0
    } else {
        -uniform.ln()
    };
    fp16(exponential)
}

fn exponential_race_argmax(probabilities: &[f32], exponentials: &[f32]) -> Result<u32> {
    if probabilities.is_empty() || probabilities.len() != exponentials.len() {
        return Err(Error::Other(format!(
            "GPT exponential race length mismatch: probabilities={}, exponentials={}",
            probabilities.len(),
            exponentials.len()
        )));
    }
    let mut sampled = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (index, (&probability, &noise)) in probabilities.iter().zip(exponentials).enumerate() {
        let score = fp16(probability / noise);
        if score > best {
            best = score;
            sampled = index;
        }
    }
    Ok(sampled as u32)
}

// The filtering order and dtype boundaries follow the reference logits_to_probs.
// The official CUDA oracle over all 437 GPT calls resolves to
// curand_init(seed, subsequence=element, offset=step) for these one-grid-stride
// tensors. PyTorch's four-word generator advance is already represented by the
// four words consumed from each Philox counter; multiplying the curand offset
// by four again is incorrect.
fn logits_to_probs_fp16(
    mut values: Vec<f32>,
    history: &[u32],
    top_k: usize,
    top_p: f32,
    temperature: f32,
    repetition_penalty: f32,
) -> Result<(Vec<f32>, u32)> {
    let vocab = values.len();
    if vocab == 0 {
        return Err(Error::Other("GPT sampling received empty logits".into()));
    }
    let mut seen = vec![false; vocab];
    for &token in history {
        let token = token as usize;
        if token < vocab && !seen[token] {
            let score = values[token];
            values[token] = if score < 0.0 {
                fp16(score * repetition_penalty)
            } else {
                fp16(score / repetition_penalty)
            };
            seen[token] = true;
        }
    }
    let adjusted_argmax = values
        .iter()
        .enumerate()
        .reduce(|best, candidate| {
            if candidate.1 > best.1 {
                candidate
            } else {
                best
            }
        })
        .map(|(index, _)| index as u32)
        .ok_or_else(|| Error::Other("GPT adjusted argmax had no value".into()))?;

    if top_p < 1.0 {
        let mut order = (0..vocab).collect::<Vec<_>>();
        order.sort_unstable_by(|&left, &right| values[right].total_cmp(&values[left]));
        let maximum = values[order[0]];
        let exponentials = order
            .iter()
            .map(|&index| (values[index] - maximum).exp())
            .collect::<Vec<_>>();
        let sum = exponentials.iter().sum::<f32>();
        let mut cumulative = 0.0f32;
        for (rank, (&index, exponential)) in order.iter().zip(&exponentials).enumerate() {
            // PyTorch stores softmax and cumsum outputs in the logits dtype.
            // The scan accumulates FP16 probabilities in FP32 and rounds only
            // the published prefix; its scalar comparison is also FP16.
            cumulative += fp16(exponential / sum);
            if rank != 0 && fp16(cumulative) > fp16(top_p) {
                values[index] = f32::NEG_INFINITY;
            }
        }
    }

    let temperature = temperature.max(1.0e-5);
    for value in &mut values {
        *value = fp16(*value / temperature);
    }
    if top_k != 0 && top_k < vocab {
        let mut sorted = values.clone();
        sorted.sort_unstable_by(|left, right| right.total_cmp(left));
        let pivot = sorted[top_k - 1];
        for value in &mut values {
            if *value < pivot {
                *value = f32::NEG_INFINITY;
            }
        }
    }

    let maximum = values
        .iter()
        .copied()
        .max_by(f32::total_cmp)
        .ok_or_else(|| Error::Other("GPT filtered logits had no value".into()))?;
    if !maximum.is_finite() {
        return Err(Error::Other("GPT filtered logits are all invalid".into()));
    }
    let exponentials = values
        .iter()
        .map(|value| {
            if value.is_finite() {
                (value - maximum).exp()
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let sum = exponentials.iter().sum::<f32>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(Error::Other("GPT probability mass is invalid".into()));
    }
    let probabilities = exponentials
        .into_iter()
        .map(|value| fp16(value / sum))
        .collect();
    Ok((probabilities, adjusted_argmax))
}

#[allow(clippy::too_many_arguments)]
fn sample_fp16_exponential_race(
    logits: &Tensor,
    history: &[u32],
    seed: u64,
    step: usize,
    top_k: usize,
    top_p: f32,
    temperature: f32,
    repetition_penalty: f32,
) -> Result<(u32, u32)> {
    let values = transfers::to_cpu(logits)?.to_f32_vec()?;
    let (probabilities, adjusted_argmax) = logits_to_probs_fp16(
        values,
        history,
        top_k,
        top_p,
        temperature,
        repetition_penalty,
    )?;
    let exponentials = (0..probabilities.len())
        .map(|index| counter_exponential_fp16(seed, philox_counter_offset(step), index))
        .collect::<Vec<_>>();
    let sampled = exponential_race_argmax(&probabilities, &exponentials)?;
    Ok((sampled, adjusted_argmax))
}

impl Gpt {
    pub fn load(weights: &mut Weights, ctx: &Context) -> Result<Self> {
        let text_emb = weights.get(ctx, "gpt.text_emb")?;
        let audio_emb = weights.get(ctx, "gpt.audio_emb")?;
        let bert_proj = Linear::load(weights, ctx, "gpt.bert_proj")?;
        let text_alpha = weights.host("gpt.text_alpha")?.to_f32_vec()?[0];
        let audio_alpha = weights.host("gpt.audio_alpha")?.to_f32_vec()?[0];
        let text_pe = fp16_tensor(
            ctx,
            &[MAX_POSITIONS, GPT_HIDDEN],
            &sine_pe(MAX_POSITIONS, GPT_HIDDEN, text_alpha),
        )?;
        let audio_pe = fp16_tensor(
            ctx,
            &[MAX_POSITIONS, GPT_HIDDEN],
            &sine_pe(MAX_POSITIONS, GPT_HIDDEN, audio_alpha),
        )?;
        let mut layers = Vec::with_capacity(GPT_LAYERS);
        for index in 0..GPT_LAYERS {
            let prefix = format!("gpt.layers.{index}");
            let q = Linear::load(weights, ctx, &format!("{prefix}.q"))?;
            let k = Linear::load(weights, ctx, &format!("{prefix}.k"))?;
            let v = Linear::load(weights, ctx, &format!("{prefix}.v"))?;
            layers.push(Layer {
                q,
                k,
                v,
                o: Linear::load(weights, ctx, &format!("{prefix}.o"))?,
                ff1: Linear::load(weights, ctx, &format!("{prefix}.ff1"))?,
                ff2: Linear::load(weights, ctx, &format!("{prefix}.ff2"))?,
                ln1_w: weights.get(ctx, &format!("{prefix}.ln1.w"))?,
                ln1_b: weights.get(ctx, &format!("{prefix}.ln1.b"))?,
                ln2_w: weights.get(ctx, &format!("{prefix}.ln2.w"))?,
                ln2_b: weights.get(ctx, &format!("{prefix}.ln2.b"))?,
            });
        }
        let out = weights.get(ctx, "gpt.out.w")?;
        Ok(Self {
            text_emb,
            audio_emb,
            bert_proj,
            text_pe,
            audio_pe,
            layers,
            out,
            decode_resources: RefCell::new(None),
            prefill_resources: RefCell::new(None),
        })
    }

    fn embed_text(&self, ctx: &Context, phones: &[u32], bert: &Tensor) -> Result<Tensor> {
        // The device lookups turn out-of-range ids into zero rows; reject them
        // here so a bad request fails instead of synthesizing from padding.
        if phones.len() > MAX_POSITIONS {
            return Err(Error::Other(format!(
                "GPT text length {} exceeds {MAX_POSITIONS} positions",
                phones.len()
            )));
        }
        let vocab = self.text_emb.shape().dims()[0];
        if let Some(&id) = phones.iter().find(|&&id| id as usize >= vocab) {
            return Err(Error::Other(format!(
                "GPT phone id {id} is outside the {vocab}-entry vocabulary"
            )));
        }
        let ids = ids_tensor(ctx, phones)?;
        let mut text = ops::embedding(ctx, &self.text_emb, &ids)?;
        let bert_dim = bert.shape().dims()[1];
        if bert_dim != GPT_BERT_DIM {
            return Err(Error::Other("GPT BERT feature width mismatch".into()));
        }
        let projected = self.bert_proj.apply(ctx, bert).map_err(|error| {
            Error::Other(format!(
                "GPT BERT projection input={:?} weight={:?}: {error}",
                bert.shape().dims(),
                self.bert_proj.w.shape().dims()
            ))
        })?;
        text = ops::residual_add(ctx, &text, &projected)
            .map_err(|error| Error::Other(format!("GPT text+BERT residual: {error}")))?;
        let positions: Vec<u32> = (0..phones.len() as u32).collect();
        let position_ids = ids_tensor(ctx, &positions)?;
        let position = ops::embedding(ctx, &self.text_pe, &position_ids)?;
        ops::residual_add(ctx, &text, &position)
    }

    fn embed_audio(&self, ctx: &Context, tokens: &[u32], position_start: usize) -> Result<Tensor> {
        let ids = ids_tensor(ctx, tokens)?;
        let audio = ops::embedding(ctx, &self.audio_emb, &ids)?;
        let end = position_start
            .checked_add(tokens.len())
            .ok_or_else(|| Error::Other("GPT audio position overflow".into()))?;
        if end > MAX_POSITIONS {
            return Err(Error::Other(format!(
                "GPT audio position {end} exceeds {MAX_POSITIONS}"
            )));
        }
        let positions = (position_start..end)
            .map(|position| position as u32)
            .collect::<Vec<_>>();
        let position_ids = ids_tensor(ctx, &positions)?;
        let position = ops::embedding(ctx, &self.audio_pe, &position_ids)?;
        ops::residual_add(ctx, &audio, &position)
    }

    /// One-token embedding for the decode loop, written into `destination`.
    fn embed_audio_step(
        &self,
        ctx: &Context,
        token: u32,
        position_start: usize,
        destination: &Tensor,
        scratch: &mut DecodeEmbedScratch,
    ) -> Result<()> {
        if position_start >= MAX_POSITIONS {
            return Err(Error::Other(format!(
                "GPT audio position {position_start} exceeds {MAX_POSITIONS}"
            )));
        }
        let position = position_start as i32;
        let ids = Tensor::from_i32(vec![1], &[token as i32])?;
        transfers::copy_cpu_to_cuda(&ids, &scratch.ids)?;
        let positions = Tensor::from_i32(vec![1], &[position])?;
        transfers::copy_cpu_to_cuda(&positions, &scratch.positions)?;
        cuda_ops::embedding_gather(ctx, &self.audio_emb, &scratch.ids, &scratch.table_row)
            .map_err(|error| Error::Other(format!("GPT audio embedding: {error}")))?;
        cuda_ops::embedding_gather(
            ctx,
            &self.audio_pe,
            &scratch.positions,
            &scratch.position_row,
        )
        .map_err(|error| Error::Other(format!("GPT audio position embedding: {error}")))?;
        ops::residual_add_into(ctx, &scratch.table_row, &scratch.position_row, destination)
    }

    fn allocate_cache(&self, ctx: &Context, capacity: usize) -> Result<Vec<LayerCache>> {
        let head_dim = GPT_HIDDEN / GPT_HEADS;
        (0..self.layers.len())
            .map(|_| {
                Ok(LayerCache {
                    key: ops::alloc(ctx, &[1, capacity, GPT_HEADS, head_dim])?,
                    value: ops::alloc(ctx, &[1, capacity, GPT_HEADS, head_dim])?,
                })
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn layer_segment(
        &self,
        ctx: &Context,
        layer: &Layer,
        hidden: &Tensor,
        cache: &LayerCache,
        cache_offset: usize,
        valid_key_tokens: usize,
        query_start: usize,
        mask: AttentionMask,
        decode_meta: Option<&KvCacheDecodeMeta>,
    ) -> Result<Tensor> {
        let len = hidden.shape().dims()[0];
        let head_dim = GPT_HIDDEN / GPT_HEADS;
        // Separate q/k/v GEMMs. Although the official model computes one
        // fused F.linear followed by chunk(3, dim=-1), our CUTLASS
        // [T,512]x[512,1536] GEMM rounds differently from official cuBLAS
        // and broke token-exact teacher forcing (step 18). The separate
        // [512,512] GEMMs reproduce official sampling bit-exactly.
        let q = layer
            .q
            .apply(ctx, hidden)
            .map_err(|error| {
                Error::Other(format!(
                    "GPT q projection hidden={:?}: {error}",
                    hidden.shape().dims()
                ))
            })?
            .reshape(Shape::new(vec![1, len, GPT_HEADS, head_dim]))?;
        let k = layer
            .k
            .apply(ctx, hidden)
            .map_err(|error| {
                Error::Other(format!(
                    "GPT k projection hidden={:?}: {error}",
                    hidden.shape().dims()
                ))
            })?
            .reshape(Shape::new(vec![1, len, GPT_HEADS, head_dim]))?;
        let v = layer
            .v
            .apply(ctx, hidden)
            .map_err(|error| {
                Error::Other(format!(
                    "GPT v projection hidden={:?}: {error}",
                    hidden.shape().dims()
                ))
            })?
            .reshape(Shape::new(vec![1, len, GPT_HEADS, head_dim]))?;
        if let Some(meta) = decode_meta {
            cuda_ops::write_kv_cache_token(ctx, &k, &cache.key, meta)?;
            cuda_ops::write_kv_cache_token(ctx, &v, &cache.value, meta)?;
        } else {
            let key_slot = cache_rows(&cache.key, cache_offset, len)?;
            let value_slot = cache_rows(&cache.value, cache_offset, len)?;
            transfers::copy_cuda_to_cuda(ctx, &k, &key_slot)?;
            transfers::copy_cuda_to_cuda(ctx, &v, &value_slot)?;
        }

        let (key, value) = if decode_meta.is_some() {
            (cache.key.clone(), cache.value.clone())
        } else {
            (
                cache_rows(&cache.key, 0, valid_key_tokens)?,
                cache_rows(&cache.value, 0, valid_key_tokens)?,
            )
        };
        let mut attended = ops::alloc(ctx, &[1, len, GPT_HEADS, head_dim])?;
        let mut args = KvCacheAttentionArgs::new(&q, &key, &value, &mut attended);
        if let Some(decode_meta) = decode_meta {
            args = args.with_decode_meta(decode_meta);
            // L3 publishes these runtime values into the mapped metadata and
            // keys the prepared execution by capacity, not by the live length.
            args.valid_key_tokens = valid_key_tokens;
            args.query_start = query_start;
        } else {
            args.valid_key_tokens = valid_key_tokens;
            args.query_start = query_start;
        }
        args.mask = mask;
        cuda_ops::kv_cache_attention(ctx, args).map_err(|error| {
            Error::Other(format!(
                "GPT KV attention q={:?} k={:?} valid={valid_key_tokens} query_start={query_start} mask={mask:?}: {error}",
                q.shape().dims(),
                key.shape().dims()
            ))
        })?;

        let attended = attended.reshape(Shape::new(vec![len, GPT_HIDDEN]))?;
        let projected = layer.o.apply(ctx, &attended).map_err(|error| {
            Error::Other(format!(
                "GPT output projection attended={:?}: {error}",
                attended.shape().dims()
            ))
        })?;
        let summed = ops::residual_add(ctx, &projected, hidden)?;
        let normed = ops::layer_norm(ctx, &summed, &layer.ln1_w, &layer.ln1_b, 1e-5)?;
        let ff = layer.ff1.apply(ctx, &normed).map_err(|error| {
            Error::Other(format!(
                "GPT ff1 normed={:?}: {error}",
                normed.shape().dims()
            ))
        })?;
        let ff = ops::relu(ctx, &ff)?;
        let ff = layer.ff2.apply(ctx, &ff).map_err(|error| {
            Error::Other(format!(
                "GPT ff2 activation={:?}: {error}",
                ff.shape().dims()
            ))
        })?;
        let summed = ops::residual_add(ctx, &ff, &normed)?;
        ops::layer_norm(ctx, &summed, &layer.ln2_w, &layer.ln2_b, 1e-5)
    }

    fn output_logits(&self, ctx: &Context, hidden: &Tensor) -> Result<Tensor> {
        let len = hidden.shape().dims()[0];
        let last = ops::slice_rows(ctx, hidden, len - 1, 1)?;
        ops::linear(ctx, &last, &self.out, None).map_err(|error| {
            Error::Other(format!(
                "GPT output head last={:?} weight={:?}: {error}",
                last.shape().dims(),
                self.out.shape().dims()
            ))
        })
    }

    /// Prefill inside a reusable execution session. Native operator resources
    /// are prepared once per (text_len, audio_len, cache capacity) instead of
    /// once per operator per request; a replay pass allocates from the session
    /// arena and launches the prepared executions directly.
    fn prefill_session_cached(
        &self,
        ctx: &Context,
        text: &Tensor,
        audio: &Tensor,
        cache: &mut DecodeCache,
    ) -> Result<Tensor> {
        let text_len = text.shape().dims()[0];
        let audio_len = audio.shape().dims()[0];
        let mut slot = self.prefill_resources.borrow_mut();
        let reusable = matches!(slot.as_ref(), Some(resources)
            if resources.capacity == cache.capacity
                && resources.text_len == text_len
                && resources.audio_len == audio_len);
        if !reusable {
            *slot = Some(PrefillResources {
                session: cuda_ops::ExecutionSession::with_capacity_pinning_prepared_buffers(
                    PREFILL_WORKSPACE_BYTES,
                    ctx.device_id(),
                )?,
                text: ops::alloc(ctx, &[text_len, GPT_HIDDEN])?,
                audio: ops::alloc(ctx, &[audio_len, GPT_HIDDEN])?,
                text_len,
                audio_len,
                capacity: cache.capacity,
                prepared: false,
            });
        }
        // Either the slot was reusable (so already `Some`) or it was just
        // filled above, so this borrow cannot fail.
        let resources = slot.as_mut().expect("prefill resources");
        transfers::copy_cuda_to_cuda(ctx, text, &resources.text)?;
        transfers::copy_cuda_to_cuda(ctx, audio, &resources.audio)?;
        let staged_text = resources.text.clone();
        let staged_audio = resources.audio.clone();
        let session = resources.session.clone();
        let prepared = resources.prepared;
        let saved = (cache.text_len, cache.audio_len);
        let mut replay_error: Option<Error> = None;
        if prepared {
            match cuda_ops::with_session(&session, || {
                self.prefill_cached(ctx, staged_text.clone(), staged_audio.clone(), cache)
            }) {
                Ok(logits) => return Ok(logits),
                Err(error) => {
                    // The address-keyed cache no longer matches this operand
                    // set; fall back to a full prepare pass instead of failing.
                    replay_error = Some(error);
                    cache.text_len = saved.0;
                    cache.audio_len = saved.1;
                }
            }
        }
        let logits = cuda_ops::prepare_with_session(&session, || {
            self.prefill_cached(ctx, staged_text.clone(), staged_audio.clone(), cache)
        })
        .map_err(|error| match replay_error {
            Some(previous) => Error::Other(format!(
                "GPT prefill replay failed ({previous}) and re-prepare failed ({error})"
            )),
            None => error,
        })?;
        if let Some(resources) = slot.as_mut() {
            resources.prepared = true;
        }
        Ok(logits)
    }

    fn prefill_cached(
        &self,
        ctx: &Context,
        text: Tensor,
        audio: Tensor,
        cache: &mut DecodeCache,
    ) -> Result<Tensor> {
        let capacity = cache.capacity;
        let text_len = text.shape().dims()[0];
        let audio_len = audio.shape().dims()[0];
        if text_len == 0 || audio_len == 0 || text_len + audio_len > capacity {
            return Err(Error::Other("invalid GPT prefill lengths".into()));
        }
        let mut text_hidden = text;
        let mut audio_hidden = audio;
        for (layer_index, (layer, cache)) in self.layers.iter().zip(&cache.layers).enumerate() {
            text_hidden = self
                .layer_segment(
                    ctx,
                    layer,
                    &text_hidden,
                    cache,
                    0,
                    text_len,
                    0,
                    AttentionMask::None,
                    None,
                )
                .map_err(|error| {
                    Error::Other(format!("GPT text prefill layer {layer_index}: {error}"))
                })?;
            audio_hidden = self
                .layer_segment(
                    ctx,
                    layer,
                    &audio_hidden,
                    cache,
                    text_len,
                    text_len + audio_len,
                    text_len,
                    AttentionMask::Causal,
                    None,
                )
                .map_err(|error| {
                    Error::Other(format!("GPT audio prefill layer {layer_index}: {error}"))
                })?;
        }
        let logits = self.output_logits(ctx, &audio_hidden)?;
        // Prefix rows were all overwritten above. Old tail rows are excluded
        // by the live valid length and overwritten before being attended.
        cache.text_len = text_len;
        cache.audio_len = audio_len;
        Ok(logits)
    }

    fn decode_cached(
        &self,
        ctx: &Context,
        input_hidden: &Tensor,
        cache: &mut DecodeCache,
        decode_meta: &KvCacheDecodeMeta,
    ) -> Result<Tensor> {
        let cache_position = cache
            .text_len
            .checked_add(cache.audio_len)
            .ok_or_else(|| Error::Other("GPT cache position overflow".into()))?;
        if cache_position >= cache.capacity {
            return Err(Error::Other("GPT cache capacity exceeded".into()));
        }
        let mut hidden = input_hidden.clone();
        for (layer_index, (layer, layer_cache)) in self.layers.iter().zip(&cache.layers).enumerate()
        {
            hidden = self
                .layer_segment(
                    ctx,
                    layer,
                    &hidden,
                    layer_cache,
                    cache_position,
                    cache_position + 1,
                    cache_position,
                    AttentionMask::Causal,
                    Some(decode_meta),
                )
                .map_err(|error| {
                    Error::Other(format!("GPT decode layer {layer_index}: {error}"))
                })?;
        }
        cache.audio_len += 1;
        self.output_logits(ctx, &hidden)
    }

    /// Generate semantic tokens. Text positions attend bidirectionally; audio
    /// positions attend all text and causal audio, matching the upstream mask.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        ctx: &Context,
        phones: &[u32],
        bert: &Tensor,
        prompt: &[u32],
        seed: u64,
        top_k: usize,
        top_p: f32,
        temperature: f32,
        repetition_penalty: f32,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>> {
        if prompt.is_empty() {
            return Err(Error::Other("GPT semantic prompt is empty".into()));
        }
        let text = self
            .embed_text(ctx, phones, bert)
            .map_err(|error| Error::Other(format!("GPT embed text: {error}")))?;
        let audio = self
            .embed_audio(ctx, prompt, 0)
            .map_err(|error| Error::Other(format!("GPT embed prompt audio: {error}")))?;
        let capacity = phones
            .len()
            .checked_add(prompt.len())
            .and_then(|value| value.checked_add(max_new_tokens))
            .ok_or_else(|| Error::Other("GPT cache capacity overflow".into()))?;
        let saved = self.decode_resources.borrow_mut().take();
        let resources = match saved {
            Some(resources) if resources.cache.capacity == capacity => {
                // Mapped metadata and graph storage may only be rebound after
                // the previous invocation has completed.
                ctx.synchronize().map_err(Error::Cuda)?;
                resources
            }
            _ => DecodeResources {
                graph: None,
                graph_logits: None,
                session: cuda_ops::ExecutionSession::with_capacity_pinning_prepared_buffers(
                    DECODE_WORKSPACE_BYTES,
                    ctx.device_id(),
                )?,
                meta: KvCacheDecodeMeta::new(ctx.device_id(), capacity)?,
                input: ops::alloc(ctx, &[1, GPT_HIDDEN])?,
                embed: DecodeEmbedScratch::new(ctx)?,
                cache: DecodeCache {
                    layers: self.allocate_cache(ctx, capacity)?,
                    text_len: 0,
                    audio_len: 0,
                    capacity,
                },
            },
        };
        let DecodeResources {
            graph: mut decode_graph,
            mut graph_logits,
            session: decode_session,
            meta: decode_meta,
            input: decode_input,
            mut cache,
            embed: mut embed_scratch,
        } = resources;
        let _gpt_prefill_range = crate::profiling::trace::range("tts.gpt.prefill");
        let mut logits = self
            .prefill_session_cached(ctx, &text, &audio, &mut cache)
            .map_err(|error| Error::Other(format!("GPT cached prefill: {error}")))?;
        drop(_gpt_prefill_range);
        let mut audio_tokens = prompt.to_vec();

        for index in 0..max_new_tokens {
            let _gpt_step_range = crate::profiling::trace::range("tts.gpt.step");
            let active_vocab = if index < MIN_GENERATED_BEFORE_EOS {
                GPT_AUDIO_VOCAB - 1
            } else {
                GPT_AUDIO_VOCAB
            };
            let active_logits = logits_prefix(&logits, active_vocab)
                .map_err(|error| Error::Other(format!("GPT logits view step {index}: {error}")))?;
            let _gpt_sample_range = crate::profiling::trace::range("tts.gpt.sample");
            let (sample, argmax) = sample_fp16_exponential_race(
                &active_logits,
                &audio_tokens,
                seed,
                index,
                top_k,
                top_p,
                temperature,
                repetition_penalty,
            )
            .map_err(|error| Error::Other(format!("GPT sample step {index}: {error}")))?;
            drop(_gpt_sample_range);
            if sample == GPT_EOS || argmax == GPT_EOS {
                break;
            }
            audio_tokens.push(sample);
            if index + 1 < max_new_tokens {
                let cache_position = cache
                    .text_len
                    .checked_add(cache.audio_len)
                    .ok_or_else(|| Error::Other("GPT cache position overflow".into()))?;
                decode_meta.update(cache_position + 1, cache_position)?;
                self.embed_audio_step(
                    ctx,
                    sample,
                    cache.audio_len,
                    &decode_input,
                    &mut embed_scratch,
                )?;
                if let Some(graph) = &decode_graph {
                    let _gpt_replay_range = crate::profiling::trace::range("tts.gpt.replay");
                    graph.replay().map_err(|error| {
                        Error::Other(format!("GPT decode replay step {index}: {error}"))
                    })?;
                    drop(_gpt_replay_range);
                    cache.audio_len += 1;
                    // `graph` and `graph_logits` are only ever assigned
                    // together, so holding the graph implies its output tensor.
                    logits = graph_logits.as_ref().expect("prepared GPT logits").clone();
                } else {
                    let original_audio_len = cache.audio_len;
                    logits = cuda_ops::prepare_with_session(&decode_session, || {
                        self.decode_cached(ctx, &decode_input, &mut cache, &decode_meta)
                    })
                    .map_err(|error| {
                        Error::Other(format!("GPT prepare decode step {index}: {error}"))
                    })?;
                    ctx.synchronize().map_err(Error::Cuda)?;
                    // Capture traverses the same resources without executing the
                    // nodes. Restore the host position so the captured traversal
                    // uses the prepared semantic; device writes read live metadata.
                    cache.audio_len = original_audio_len;
                    let graph = apxinf_cuda_new::capture(ctx, || {
                        cuda_ops::with_session(&decode_session, || {
                            self.decode_cached(ctx, &decode_input, &mut cache, &decode_meta)
                        })?;
                        Ok(())
                    })
                    .map_err(|error| {
                        Error::Other(format!("GPT capture decode step {index}: {error}"))
                    })?;
                    graph_logits = Some(logits.clone());
                    decode_graph = Some(graph);
                }
            }
        }
        let generated = audio_tokens[prompt.len()..].to_vec();
        *self.decode_resources.borrow_mut() = Some(DecodeResources {
            graph: decode_graph,
            graph_logits,
            session: decode_session,
            meta: decode_meta,
            input: decode_input,
            cache,
            embed: embed_scratch,
        });
        Ok(generated)
    }
}

#[cfg(test)]
mod sampling_tests {
    use super::*;

    #[test]
    fn fp16_sampling_probabilities_match_reference_boundaries() {
        // Independent official AR/models/utils.py::logits_to_probs output from
        // PyTorch 2.11 FP16. This exercises filtering and rounding, not RNG.
        let cases: &[(&str, &[f32], &[u32], usize, f32, f32, f32, &[f32], u32)] = &[
            (
                "ties",
                &[2.0, 2.0, -1.0, -4.0],
                &[],
                0usize,
                1.0f32,
                1.0f32,
                1.0f32,
                &[
                    0.4873046875,
                    0.4873046875,
                    0.024261474609375,
                    0.0012073516845703125,
                ],
                0u32,
            ),
            (
                "repeated_penalty",
                &[3.0, 1.0, -2.0, -4.0, 0.5],
                &[0, 0, 2],
                3usize,
                0.97f32,
                0.7f32,
                1.35f32,
                &[0.8515625, 0.1484375, 0.0, 0.0, 0.0],
                0u32,
            ),
            (
                "nucleus_fp16",
                &[2.0, 1.0, 0.0, -1.0, -2.0],
                &[],
                0usize,
                0.871f32,
                1.0f32,
                1.0f32,
                &[0.73095703125, 0.26904296875, 0.0, 0.0, 0.0],
                0u32,
            ),
            (
                "topk_ties",
                &[3.0, 2.0, 2.0, -3.0],
                &[],
                2usize,
                1.0f32,
                0.83f32,
                1.0f32,
                &[0.62548828125, 0.1873779296875, 0.1873779296875, 0.0],
                0u32,
            ),
            (
                "minimum_one",
                &[5.0, 3.0, 2.0, 0.0],
                &[],
                0usize,
                0.01f32,
                1.0f32,
                1.0f32,
                &[1.0, 0.0, 0.0, 0.0],
                0u32,
            ),
            (
                "nucleus_threshold_rounding",
                &[2.0, 1.0, 0.0, -1.0, -2.0],
                &[],
                0usize,
                0.8701071875f32,
                1.0f32,
                1.0f32,
                &[0.73095703125, 0.26904296875, 0.0, 0.0, 0.0],
                0u32,
            ),
        ];
        for &(name, logits, history, top_k, top_p, temperature, repetition, expected, argmax) in
            cases
        {
            let (actual, actual_argmax) = logits_to_probs_fp16(
                logits.to_vec(),
                history,
                top_k,
                top_p,
                temperature,
                repetition,
            )
            .unwrap();
            assert_eq!(actual, expected, "{name}: reference FP16 probabilities");
            assert_eq!(actual_argmax, argmax, "{name}: first-index adjusted argmax");
        }
    }
}
