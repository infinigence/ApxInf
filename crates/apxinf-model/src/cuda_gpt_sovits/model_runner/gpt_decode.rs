//! Decode-side execution strategy for the semantic decoder.

use apxinf_core::{DType, Error, Result, Shape, Tensor};
use apxinf_cuda_new::ops::{self as cuda_ops, KvCacheDecodeMeta};
use apxinf_cuda_new::{transfers, CudaContext as Context};

use crate::cuda_gpt_sovits::config::{GPT_AUDIO_VOCAB, GPT_EOS, GPT_HIDDEN};
use crate::cuda_gpt_sovits::model::{
    device_sample_fp16_exponential_race, logits_prefix, sample_fp16_exponential_race, DecodeCache,
    DecodeEmbedScratch, Gpt, DECODE_WORKSPACE_BYTES, DEVICE_SAMPLING_MAX_TOP_K,
    MIN_GENERATED_BEFORE_EOS,
};
use crate::cuda_gpt_sovits::ops;

// Prefill keeps one whole eager traversal of intermediates in the session
// arena. Measured peak is ~51 MB; the rest is headroom for longer prompts.
const PREFILL_WORKSPACE_BYTES: usize = 256 * 1024 * 1024;

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

/// Long-lived decode/prefill execution state owned by the runner side. The
/// model exposes the compute interfaces; how long their scratch, sessions and
/// captured graphs live is a runner decision.
pub(crate) struct GptExecutionState {
    pub(crate) decode: Option<DecodeResources>,
    pub(crate) prefill: Option<PrefillResources>,
}

impl GptExecutionState {
    pub(crate) fn new() -> Self {
        Self {
            decode: None,
            prefill: None,
        }
    }
}

impl Default for GptExecutionState {
    fn default() -> Self {
        Self::new()
    }
}

/// Prefill inside a reusable execution session. Native operator resources
/// are prepared once per (text_len, audio_len, cache capacity) instead of
/// once per operator per request; a replay pass allocates from the session
/// arena and launches the prepared executions directly.
fn prefill_session_cached(
    gpt: &Gpt,
    ctx: &Context,
    slot: &mut Option<PrefillResources>,
    text: &Tensor,
    audio: &Tensor,
    cache: &mut DecodeCache,
) -> Result<Tensor> {
    let text_len = text.shape().dims()[0];
    let audio_len = audio.shape().dims()[0];
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
            gpt.prefill_cached(ctx, staged_text.clone(), staged_audio.clone(), cache)
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
        gpt.prefill_cached(ctx, staged_text.clone(), staged_audio.clone(), cache)
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

/// Generate semantic tokens. Text positions attend bidirectionally; audio
/// positions attend all text and causal audio, matching the upstream mask.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_semantic(
    gpt: &Gpt,
    ctx: &Context,
    state: &mut GptExecutionState,
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
    let text = gpt
        .embed_text(ctx, phones, bert)
        .map_err(|error| Error::Other(format!("GPT embed text: {error}")))?;
    let audio = gpt
        .embed_audio(ctx, prompt, 0)
        .map_err(|error| Error::Other(format!("GPT embed prompt audio: {error}")))?;
    let capacity = phones
        .len()
        .checked_add(prompt.len())
        .and_then(|value| value.checked_add(max_new_tokens))
        .ok_or_else(|| Error::Other("GPT cache capacity overflow".into()))?;
    let saved = state.decode.take();
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
                layers: gpt.allocate_cache(ctx, capacity)?,
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
    let mut prefill_slot = state.prefill.take();
    let prefill_result =
        prefill_session_cached(gpt, ctx, &mut prefill_slot, &text, &audio, &mut cache);
    state.prefill = prefill_slot;
    let mut logits =
        prefill_result.map_err(|error| Error::Other(format!("GPT cached prefill: {error}")))?;
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
            gpt.embed_audio_step(
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
                    gpt.decode_cached(ctx, &decode_input, &mut cache, &decode_meta)
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
                        gpt.decode_cached(ctx, &decode_input, &mut cache, &decode_meta)
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
    state.decode = Some(DecodeResources {
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
