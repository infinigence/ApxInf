//! GPT (Text2Semantic) decoder for GPT-SoVITS V5.

use std::cell::RefCell;
#[cfg(test)]
use std::io::Write;

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
// PyTorch 2.11 launches its CUDA universal kernel with 256 threads per block.
// On Orin's current scheduler this kernel receives five blocks for the GPT
// 1024/1025-element tensors, so every element is assigned in one launch step.
const TORCH_CUDA_EXPONENTIAL_ACTIVE_THREADS: usize = 5 * 256;

struct Layer {
    #[allow(dead_code)]
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
}

pub struct Gpt {
    decode_resources: RefCell<Option<DecodeResources>>,
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

fn torch_cuda_exponential_counter_offset(step: usize) -> u64 {
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

#[cfg(test)]
struct SamplingAlignment {
    directory: std::path::PathBuf,
    run: String,
    teacher_tokens: Vec<u32>,
}

#[cfg(test)]
impl SamplingAlignment {
    fn from_env() -> Result<Option<Self>> {
        let Some(directory) = std::env::var_os("APXINF_GPT_SAMPLING_ALIGNMENT_DIR") else {
            return Ok(None);
        };
        let directory = std::path::PathBuf::from(directory);
        let run = std::env::var("APXINF_GPT_SAMPLING_ALIGNMENT_RUN")
            .unwrap_or_else(|_| "default".to_string());
        let teacher_tokens = std::fs::read(directory.join(format!("{run}-teacher-tokens.u32")))
            .map_err(|error| Error::Other(format!("GPT alignment teacher tokens: {error}")))?
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte token")))
            .collect();
        Ok(Some(Self {
            directory,
            run,
            teacher_tokens,
        }))
    }

    fn teacher_token(&self, step: usize) -> Option<u32> {
        self.teacher_tokens.get(step).copied()
    }

    fn exponential_draw(&self, step: usize, expected_len: usize) -> Result<Vec<f32>> {
        let path = self
            .directory
            .join(format!("{}-exponential-{step:04}.f16", self.run));
        let bytes = std::fs::read(&path).map_err(|error| {
            Error::Other(format!(
                "GPT alignment random stream {}: {error}",
                path.display()
            ))
        })?;
        if bytes.len() != expected_len * 2 {
            return Err(Error::Other(format!(
                "GPT alignment random stream {} has {} bytes, expected {}",
                path.display(),
                bytes.len(),
                expected_len * 2
            )));
        }
        Ok(bytes
            .chunks_exact(2)
            .map(|value| half::f16::from_bits(u16::from_le_bytes([value[0], value[1]])).to_f32())
            .collect())
    }

    fn write_logits(&self, step: usize, values: &[f32]) -> Result<()> {
        let path = self
            .directory
            .join(format!("{}-native-logits-{step:04}.f16", self.run));
        let mut bytes = Vec::with_capacity(values.len() * 2);
        for value in values {
            bytes.extend_from_slice(&half::f16::from_f32(*value).to_bits().to_le_bytes());
        }
        std::fs::write(&path, bytes).map_err(|error| {
            Error::Other(format!(
                "GPT alignment native logits {}: {error}",
                path.display()
            ))
        })
    }

    fn record_selection(
        &self,
        step: usize,
        sampled: u32,
        argmax: u32,
        expected: Option<u32>,
    ) -> Result<()> {
        let path = self
            .directory
            .join(format!("{}-native-selections.csv", self.run));
        let first_step = step == 0;
        let mut options = std::fs::OpenOptions::new();
        options.create(true).write(true);
        if first_step {
            options.truncate(true);
        } else {
            options.append(true);
        }
        let mut file = options.open(&path).map_err(|error| {
            Error::Other(format!(
                "GPT alignment selection log {}: {error}",
                path.display()
            ))
        })?;
        if first_step {
            writeln!(file, "step,sampled,argmax,expected")
                .map_err(|error| Error::Other(format!("GPT alignment header: {error}")))?;
        }
        writeln!(
            file,
            "{step},{sampled},{argmax},{}",
            expected
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string())
        )
        .map_err(|error| Error::Other(format!("GPT alignment selection: {error}")))
    }
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
    #[cfg(test)] alignment: Option<&SamplingAlignment>,
) -> Result<(u32, u32)> {
    let values = transfers::to_cpu(logits)?.to_f32_vec()?;
    #[cfg(test)]
    if let Some(alignment) = alignment {
        alignment.write_logits(step, &values)?;
    }
    let (probabilities, adjusted_argmax) = logits_to_probs_fp16(
        values,
        history,
        top_k,
        top_p,
        temperature,
        repetition_penalty,
    )?;
    #[cfg(test)]
    let exponentials = match alignment {
        Some(alignment) => alignment.exponential_draw(step, probabilities.len())?,
        None => (0..probabilities.len())
            .map(|index| {
                counter_exponential_fp16(seed, torch_cuda_exponential_counter_offset(step), index)
            })
            .collect(),
    };
    #[cfg(not(test))]
    let exponentials = (0..probabilities.len())
        .map(|index| {
            counter_exponential_fp16(seed, torch_cuda_exponential_counter_offset(step), index)
        })
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
        })
    }

    fn embed_text(&self, ctx: &Context, phones: &[u32], bert: &Tensor) -> Result<Tensor> {
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
        } = resources;
        let _gpt_prefill_range = crate::profiling::trace::range("tts.gpt.prefill");
        let mut logits = self
            .prefill_cached(ctx, text, audio, &mut cache)
            .map_err(|error| Error::Other(format!("GPT cached prefill: {error}")))?;
        drop(_gpt_prefill_range);
        let mut audio_tokens = prompt.to_vec();
        #[cfg(test)]
        let sampling_alignment = SamplingAlignment::from_env()?;

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
                #[cfg(test)]
                sampling_alignment.as_ref(),
            )
            .map_err(|error| Error::Other(format!("GPT sample step {index}: {error}")))?;
            drop(_gpt_sample_range);
            #[cfg(test)]
            let teacher = sampling_alignment
                .as_ref()
                .and_then(|alignment| alignment.teacher_token(index));
            #[cfg(not(test))]
            let teacher: Option<u32> = None;
            #[cfg(test)]
            if let Some(alignment) = sampling_alignment.as_ref() {
                alignment.record_selection(index, sample, argmax, teacher)?;
            }
            // Teacher forcing keeps a diagnostic run aligned after the first
            // native token mismatch. It is unavailable to production builds.
            if teacher.is_none() && (sample == GPT_EOS || argmax == GPT_EOS) {
                break;
            }
            let next_token = teacher.unwrap_or(sample);
            audio_tokens.push(next_token);
            if index + 1 < max_new_tokens {
                let cache_position = cache
                    .text_len
                    .checked_add(cache.audio_len)
                    .ok_or_else(|| Error::Other("GPT cache position overflow".into()))?;
                decode_meta.update(cache_position + 1, cache_position)?;
                let embedded = self.embed_audio(ctx, &[next_token], cache.audio_len)?;
                transfers::copy_cuda_to_cuda(ctx, &embedded, &decode_input)?;
                if let Some(graph) = &decode_graph {
                    let _gpt_replay_range = crate::profiling::trace::range("tts.gpt.replay");
                    graph.replay().map_err(|error| {
                        Error::Other(format!("GPT decode replay step {index}: {error}"))
                    })?;
                    drop(_gpt_replay_range);
                    cache.audio_len += 1;
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
        });
        Ok(generated)
    }
}

#[cfg(test)]
mod capture_tests {
    use super::*;
    use std::path::Path;
    fn u32_file(path: &Path) -> Vec<u32> {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
    // Frozen independent PyTorch inputs and tokens are supplied privately.
    #[test]
    #[ignore = "requires APXINF_GPT_CAPTURE_DIR and APXINF_GPT_CHECKPOINT"]
    fn cached_decode_matches_captured_tokens() {
        let root = std::path::PathBuf::from(std::env::var("APXINF_GPT_CAPTURE_DIR").unwrap());
        let checkpoint = std::path::PathBuf::from(std::env::var("APXINF_GPT_CHECKPOINT").unwrap());
        let ctx = Context::new(0).unwrap();
        let mut weights = Weights::load(&checkpoint).unwrap();
        let gpt = Gpt::load(&mut weights, &ctx).unwrap();
        for pair in ["pair1", "pair1", "pair2", "pair2", "pair1"] {
            let phones = u32_file(&root.join(format!("{pair}-phones.i32")));
            let prompt = u32_file(&root.join(format!("{pair}-prompt.i32")));
            let expected = u32_file(&root.join(format!("{pair}-target-tokens.i32")));
            let bytes = std::fs::read(root.join(format!("{pair}-bert-rows.f32"))).unwrap();
            let values = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            let bert = fp16_tensor(&ctx, &[phones.len(), GPT_BERT_DIM], &values).unwrap();
            let start = std::time::Instant::now();
            let actual = gpt
                .generate(&ctx, &phones, &bert, &prompt, 1234, 5, 1.0, 1.0, 1.35, 500)
                .unwrap();
            let mismatch = actual.iter().zip(&expected).position(|(a, b)| a != b);
            println!(
                "{pair} seconds={:.6} actual={} expected={} mismatch={mismatch:?}",
                start.elapsed().as_secs_f64(),
                actual.len(),
                expected.len()
            );
            assert!(
                mismatch.is_none() && actual.len() == expected.len(),
                "captured token stream differs"
            );
            // Change content while retaining exactly the same capacity. The
            // reused executable must match a fresh allocation and prefill.
            let mut changed_phones = phones.clone();
            changed_phones[0] = (changed_phones[0] + 1) % 100;
            let changed = gpt
                .generate(
                    &ctx,
                    &changed_phones,
                    &bert,
                    &prompt,
                    4321,
                    5,
                    1.,
                    1.,
                    1.35,
                    500,
                )
                .unwrap();
            let saved = gpt.decode_resources.borrow_mut().take();
            let fresh = gpt
                .generate(
                    &ctx,
                    &changed_phones,
                    &bert,
                    &prompt,
                    4321,
                    5,
                    1.,
                    1.,
                    1.35,
                    500,
                )
                .unwrap();
            assert_eq!(
                changed, fresh,
                "reused decode resources retained stale request state"
            );
            drop(saved);
            // Equal total capacity does not imply equal text/audio partition.
            // A shorter text prefix with one extra prompt code must also rebind.
            let shorter_phones = &phones[..phones.len() - 1];
            let shorter_bert = ops::slice_rows(&ctx, &bert, 0, shorter_phones.len()).unwrap();
            let mut longer_prompt = prompt.clone();
            longer_prompt.push(prompt[0]);
            let changed_partition = gpt
                .generate(
                    &ctx,
                    shorter_phones,
                    &shorter_bert,
                    &longer_prompt,
                    1234,
                    5,
                    1.,
                    1.,
                    1.35,
                    500,
                )
                .unwrap();
            let saved = gpt.decode_resources.borrow_mut().take();
            let fresh_partition = gpt
                .generate(
                    &ctx,
                    shorter_phones,
                    &shorter_bert,
                    &longer_prompt,
                    1234,
                    5,
                    1.,
                    1.,
                    1.35,
                    500,
                )
                .unwrap();
            assert_eq!(
                changed_partition, fresh_partition,
                "reused graph baked in a previous text/audio partition"
            );
            drop(saved);
        }
    }

    #[test]
    #[ignore = "requires APXINF_GPT_SAMPLING_ALIGNMENT_DIR, APXINF_GPT_CAPTURE_DIR and APXINF_GPT_CHECKPOINT"]
    fn teacher_forcing_matches_official_sampling_oracle() {
        let alignment_root =
            std::path::PathBuf::from(std::env::var("APXINF_GPT_SAMPLING_ALIGNMENT_DIR").unwrap());
        let capture_root =
            std::path::PathBuf::from(std::env::var("APXINF_GPT_CAPTURE_DIR").unwrap());
        let run = std::env::var("APXINF_GPT_SAMPLING_ALIGNMENT_RUN").unwrap();
        let pair = std::env::var("APXINF_GPT_CAPTURE_PAIR").unwrap_or_else(|_| "pair2".to_string());
        let checkpoint = std::path::PathBuf::from(std::env::var("APXINF_GPT_CHECKPOINT").unwrap());
        let phones = u32_file(&capture_root.join(format!("{pair}-phones.i32")));
        let prompt = u32_file(&capture_root.join(format!("{pair}-prompt.i32")));
        let teacher = u32_file(&alignment_root.join(format!("{run}-teacher-tokens.u32")));
        let bytes = std::fs::read(capture_root.join(format!("{pair}-bert-rows.f32"))).unwrap();
        let values = bytes
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
            .collect::<Vec<_>>();

        let ctx = Context::new(0).unwrap();
        let mut weights = Weights::load(&checkpoint).unwrap();
        let gpt = Gpt::load(&mut weights, &ctx).unwrap();
        let bert = fp16_tensor(&ctx, &[phones.len(), GPT_BERT_DIM], &values).unwrap();
        let generated = gpt
            .generate(
                &ctx,
                &phones,
                &bert,
                &prompt,
                1234,
                5,
                1.0,
                1.0,
                1.35,
                teacher.len() + 1,
            )
            .unwrap();

        let selections =
            std::fs::read_to_string(alignment_root.join(format!("{run}-native-selections.csv")))
                .unwrap();
        let rows = selections
            .lines()
            .skip(1)
            .map(|line| {
                line.split(',')
                    .take(3)
                    .map(|value| value.parse::<u32>().unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            rows.len(),
            teacher.len() + 1,
            "missing native sampling step"
        );
        let first_mismatch = rows
            .iter()
            .zip(&teacher)
            .position(|(row, &expected)| row[1] != expected);
        println!(
            "{pair} teacher-forced sampled={} teacher={} generated={} first_mismatch={first_mismatch:?}",
            rows.len(),
            teacher.len(),
            generated.len()
        );
        assert_eq!(first_mismatch, None, "teacher-forced sampled token differs");
    }
}

#[cfg(test)]
mod sampling_tests {
    use super::*;

    #[test]
    #[ignore = "requires APXINF_GPT_SAMPLING_ALIGNMENT_DIR and APXINF_GPT_SAMPLING_ALIGNMENT_RUN"]
    fn torch_cuda_philox_matches_official_exponential_streams() {
        let directory =
            std::path::PathBuf::from(std::env::var("APXINF_GPT_SAMPLING_ALIGNMENT_DIR").unwrap());
        let run = std::env::var("APXINF_GPT_SAMPLING_ALIGNMENT_RUN").unwrap();
        let teacher_bytes =
            std::fs::read(directory.join(format!("{run}-teacher-tokens.u32"))).unwrap();
        let expected_steps = teacher_bytes.len() / 4 + 1;
        let mut values = 0usize;
        let mut exact = 0usize;
        let mut max_abs = 0.0f32;
        let mut farthest_ulp = 0u32;
        let mut worst = (0usize, 0usize, 0u16, 0u16);
        let mut steps = 0usize;
        while let Ok(bytes) =
            std::fs::read(directory.join(format!("{run}-exponential-{steps:04}.f16")))
        {
            assert_eq!(bytes.len() % 2, 0, "odd FP16 stream length at step {steps}");
            assert!(
                bytes.len() / 2 <= TORCH_CUDA_EXPONENTIAL_ACTIVE_THREADS,
                "Philox mapping only covers one grid-stride launch step"
            );
            for (element, encoded) in bytes.chunks_exact(2).enumerate() {
                let expected_bits = u16::from_le_bytes([encoded[0], encoded[1]]);
                let expected = half::f16::from_bits(expected_bits).to_f32();
                let actual = counter_exponential_fp16(
                    1234,
                    torch_cuda_exponential_counter_offset(steps),
                    element,
                );
                let actual_bits = half::f16::from_f32(actual).to_bits();
                let distance = u16::abs_diff(actual_bits, expected_bits);
                if distance == 0 {
                    exact += 1;
                }
                let absolute = (actual - expected).abs();
                if absolute > max_abs {
                    max_abs = absolute;
                }
                if u32::from(distance) > farthest_ulp {
                    farthest_ulp = u32::from(distance);
                    worst = (steps, element, expected_bits, actual_bits);
                }
                assert!(
                    distance <= 1,
                    "step={steps} element={element} expected={expected_bits:#06x} actual={actual_bits:#06x}"
                );
                values += 1;
            }
            steps += 1;
        }
        println!(
            "official Philox streams={steps} values={values} exact={exact} ulp_mismatches={} max_abs={max_abs} farthest_ulp={farthest_ulp} worst={worst:?}",
            values - exact
        );
        assert_eq!(steps, expected_steps);
        assert!(exact * 1000 > values * 999, "exact-value rate below 99.9%");
    }

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
