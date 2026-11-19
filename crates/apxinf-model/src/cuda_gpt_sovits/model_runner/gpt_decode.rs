//! Decode-side execution strategy for the semantic decoder.

use apxinf_core::{DType, Error, Result, Shape, Tensor};
use apxinf_cuda_new::ops::{self as cuda_ops, KvCacheDecodeMeta};
use apxinf_cuda_new::{transfers, CudaContext as Context};

use crate::cuda_gpt_sovits::config::{GPT_AUDIO_VOCAB, GPT_EOS, GPT_HIDDEN};
use crate::cuda_gpt_sovits::model::{
    device_sample_fp16_exponential_race, logits_prefix, sample_fp16_exponential_race, DecodeCache,
    Gpt, DECODE_WORKSPACE_BYTES, DEVICE_SAMPLING_MAX_TOP_K, MIN_GENERATED_BEFORE_EOS,
};
use crate::cuda_gpt_sovits::ops;

// Keep the executable before its referenced storage so it is destroyed first.
// One latest capacity is retained; requests never retain another request's KV values.
pub(crate) struct DecodeResources {
    graph: Option<apxinf_cuda_new::CapturedGraph>,
    graph_logits: Option<Tensor>,
    session: cuda_ops::ExecutionSession,
    meta: KvCacheDecodeMeta,
    input: Tensor,
    cache: DecodeCache,
    embed: DecodeEmbedScratch,
    /// Repetition-penalty flags for the device sampler: `seen[token] != 0`
    /// marks scores the penalty has already been applied to, matching the
    /// host sampler's first-occurrence dedup over the token history.
    seen: Tensor,
    /// Device sampler output words: `[sampled, adjusted_argmax, status, 0]`.
    sample_out: Tensor,
}

/// Fixed buffers for the one-token decode embedding.
///
/// `embed_audio` allocates five tensors per call: two 4-byte id vectors, two
/// 2 KB gathered rows and the 2 KB sum, and the decode loop dropped all five
/// as soon as they had been copied into `decode_input`. Each free synchronizes
/// the device, so the step paid five allocator round trips and five drains for
/// 6 KB of scratch. Keeping them for the life of the decode resources removes
/// the allocation entirely and writes the sum straight into the graph input.
pub(crate) struct DecodeEmbedScratch {
    pub(crate) ids: Tensor,
    pub(crate) positions: Tensor,
    pub(crate) table_row: Tensor,
    pub(crate) position_row: Tensor,
}

impl DecodeEmbedScratch {
    pub(crate) fn new(ctx: &Context) -> Result<Self> {
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
pub(crate) struct PrefillResources {
    pub(crate) session: cuda_ops::ExecutionSession,
    pub(crate) text: Tensor,
    pub(crate) audio: Tensor,
    pub(crate) text_len: usize,
    pub(crate) audio_len: usize,
    pub(crate) capacity: usize,
    pub(crate) prepared: bool,
}

impl Gpt {
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
                seen: ctx.allocate_output(Shape::new(vec![GPT_AUDIO_VOCAB]), DType::I8)?,
                sample_out: ctx.allocate_output(Shape::new(vec![4]), DType::I32)?,
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
            seen: decode_seen,
            sample_out: decode_sample_out,
        } = resources;
        let _gpt_prefill_range = crate::profiling::trace::range("tts.gpt.prefill");
        let mut logits = self
            .prefill_session_cached(ctx, &text, &audio, &mut cache)
            .map_err(|error| Error::Other(format!("GPT cached prefill: {error}")))?;
        drop(_gpt_prefill_range);
        let mut audio_tokens = prompt.to_vec();

        // The device sampler reproduces the host pipeline bit for bit inside
        // its envelope (top_p >= 1.0, bounded top_k); outside it the decode
        // loop keeps the reference host sampler.
        let device_sampling = top_p >= 1.0
            && top_k <= u32::MAX as usize
            && (top_k == 0 || top_k >= GPT_AUDIO_VOCAB || top_k <= DEVICE_SAMPLING_MAX_TOP_K);
        if device_sampling {
            // The flags vector below defines the entire buffer state (zeros
            // except the prompt entries), and the synchronous copy lands
            // before any later kernel launch. Do NOT prepend a stream-level
            // fill_zero: on the non-blocking context stream it can trail the
            // default-stream copy — on the second and later calls it queues
            // behind the prefill replay and wipes the prompt flags, silently
            // dropping the repetition penalty.
            // First-occurrence dedup of the prompt history, matching the host
            // sampler's seen marking; generated tokens are flagged by the
            // kernel itself as they are sampled.
            let mut flags = vec![0i8; GPT_AUDIO_VOCAB];
            for &token in prompt {
                let token = token as usize;
                if token < GPT_AUDIO_VOCAB {
                    flags[token] = 1;
                }
            }
            let host = Tensor::from_i8(Shape::new(vec![GPT_AUDIO_VOCAB]), &flags)?;
            transfers::copy_cpu_to_cuda(&host, &decode_seen)
                .map_err(|error| Error::Other(format!("GPT sampler flag upload: {error}")))?;
        }

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
            let (sample, argmax) = if device_sampling {
                match device_sample_fp16_exponential_race(
                    ctx,
                    &active_logits,
                    &decode_seen,
                    &decode_sample_out,
                    seed,
                    index,
                    top_k,
                    top_p,
                    temperature,
                    repetition_penalty,
                )
                .map_err(|error| Error::Other(format!("GPT device sample step {index}: {error}")))?
                {
                    Some(sampled) => sampled,
                    None => sample_fp16_exponential_race(
                        &active_logits,
                        &audio_tokens,
                        seed,
                        index,
                        top_k,
                        top_p,
                        temperature,
                        repetition_penalty,
                    )
                    .map_err(|error| Error::Other(format!("GPT sample step {index}: {error}")))?,
                }
            } else {
                sample_fp16_exponential_race(
                    &active_logits,
                    &audio_tokens,
                    seed,
                    index,
                    top_k,
                    top_p,
                    temperature,
                    repetition_penalty,
                )
                .map_err(|error| Error::Other(format!("GPT sample step {index}: {error}")))?
            };
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
            seen: decode_seen,
            sample_out: decode_sample_out,
        });
        Ok(generated)
    }
}
