//! Common LLM trait for all model implementations.

use std::collections::HashMap;

use apxinf_core::{Device, Error, Result, Tensor};
use apxinf_loader::ModelConfig;

use crate::profiling::GenerationProfile;

/// Processor output for one or more images in a generation prompt.
///
/// `pixel_values` is deliberately borrowed: creating a text-only request does
/// not allocate, clone a tensor, or alter the decode hot path. Models define
/// the exact tensor layout they accept. `grid_thw` contains one entry per
/// image represented by the (possibly concatenated) tensor.
#[derive(Clone, Copy, Debug)]
pub struct ImageInput<'a> {
    pub pixel_values: &'a Tensor,
    pub grid_thw: &'a [[u32; 3]],
}

impl<'a> ImageInput<'a> {
    pub const fn new(pixel_values: &'a Tensor, grid_thw: &'a [[u32; 3]]) -> Self {
        Self {
            pixel_values,
            grid_thw,
        }
    }
}

/// Unified prompt input for text and vision-language generation.
///
/// Media is attached to the prompt and consumed during prefill. Autoregressive
/// decode continues to use token-only [`LlmTrait::forward`], so text and VLM
/// models share the same generation loop without a modality check per token.
#[derive(Clone, Copy, Debug)]
pub struct LlmInput<'a> {
    pub token_ids: &'a [u32],
    pub image: Option<ImageInput<'a>>,
}

impl<'a> LlmInput<'a> {
    pub const fn text(token_ids: &'a [u32]) -> Self {
        Self {
            token_ids,
            image: None,
        }
    }

    pub const fn with_image(token_ids: &'a [u32], image: ImageInput<'a>) -> Self {
        Self {
            token_ids,
            image: Some(image),
        }
    }
}

/// Input modalities accepted by an LLM implementation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LlmCapabilities {
    pub image: bool,
}

impl LlmCapabilities {
    pub const TEXT_ONLY: Self = Self { image: false };
    pub const VISION: Self = Self { image: true };
}

/// Result metadata for an exact greedy draft-block decode.
///
/// `consumed_draft` is the number of draft positions evaluated. The verified
/// output contains one ordinary greedy token for each consumed position.
/// `accepted_prefix` is the number of leading draft tokens equal to those
/// outputs. A mismatch, when present, is the final consumed position; otherwise
/// the entire draft was accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftBlockResult {
    pub consumed_draft: usize,
    pub accepted_prefix: usize,
}


/// Common interface for all LLM implementations.
pub trait LlmTrait {
    /// Load model weights and configure for the given device.
    fn load(config: ModelConfig, weights: HashMap<String, Tensor>, device: Device) -> Result<Self>
    where
        Self: Sized;

    /// Token-level forward pass.
    /// Returns logits of shape `[seq_len, vocab_size]`.
    fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor>;

    /// Modalities accepted by [`Self::prefill`]. Text is always supported.
    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::TEXT_ONLY
    }

    /// Process a complete prompt and return its logits.
    ///
    /// Text-only models inherit this implementation. It rejects image input
    /// explicitly instead of silently ignoring it. Vision-language models
    /// override this one request-level hook to encode and merge image features.
    fn prefill(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        if input.image.is_some() {
            return Err(Error::Other(
                "this model does not support image input".into(),
            ));
        }
        self.forward(input.token_ids, 0)
    }

    /// Optional greedy prefill fast path returning only the first token id.
    /// Implementations must use the same logits and strict argmax semantics as
    /// [`Self::prefill`].
    fn prefill_token(&mut self, _input: LlmInput<'_>) -> Option<Result<u32>> {
        None
    }


    /// Reset state for a new generation.
    fn reset(&mut self);

    /// Optional hook called once before prefill, with the prompt length and
    /// the number of tokens that will be generated. Models with a CUDA
    /// decode graph use it to pre-capture every bucket they'll hit so the
    /// per-token TPOT stays at pure graph-replay cost. Default: no-op.
    fn prewarm_decode(&mut self, _prompt_len: usize, _max_new_tokens: usize) {}

    /// Greedy-decode one token directly to its id, skipping the full-logits
    /// D2H + CPU argmax. Returns `None` if the model has no GPU-argmax fast
    /// path (caller falls back to `forward` + `argmax_last_row`).
    fn decode_token(&mut self, _token: u32, _pos: u32) -> Option<Result<u32>> {
        None
    }

    /// Optionally verify a proposed block with the model's exact serial greedy
    /// decode path.
    ///
    /// `current_token` has already been emitted and is decoded at `start_pos`.
    /// Implementations append the resulting greedy tokens to `verified` and
    /// leave model state exactly as the same sequence of [`Self::decode_token`]
    /// calls. They stop after the first mismatch or after consuming the whole
    /// draft. Returning `None` means unsupported and must not mutate state.
    ///
    /// For `Some`, `verified.len() == consumed_draft`. A mismatch has
    /// `accepted_prefix + 1 == consumed_draft`; a full match has
    /// `accepted_prefix == consumed_draft == draft.len()`.
    fn decode_draft_block(
        &mut self,
        _current_token: u32,
        _start_pos: u32,
        _draft: &[u32],
        _verified: &mut Vec<u32>,
    ) -> Option<Result<DraftBlockResult>> {
        None
    }

    /// Vocabulary size (used by default generate_streaming for argmax).
    fn vocab_size(&self) -> usize;

    /// Ergonomic, statically typed streaming entrypoint.
    fn generate_streaming(
        &mut self,
        input: LlmInput<'_>,
        max_new_tokens: usize,
        on_token: impl FnMut(u32),
        eos_token_id: Option<u32>,
    ) -> Result<(Vec<u32>, GenerationProfile)>
    where
        Self: Sized,
    {
        generate_streaming(self, input, max_new_tokens, on_token, eos_token_id)
    }

    /// Generate into reusable caller-owned storage.
    fn generate_streaming_into(
        &mut self,
        input: LlmInput<'_>,
        max_new_tokens: usize,
        generated: &mut Vec<u32>,
        on_token: impl FnMut(u32),
        eos_token_id: Option<u32>,
    ) -> Result<GenerationProfile>
    where
        Self: Sized,
    {
        generate_streaming_into(self, input, max_new_tokens, generated, on_token, eos_token_id)
    }

    /// Object-safe entry used by `AutoModel`.
    fn generate_streaming_dyn(
        &mut self,
        input: LlmInput<'_>,
        max_new_tokens: usize,
        on_token: &mut dyn FnMut(u32),
        eos_token_id: Option<u32>,
    ) -> Result<(Vec<u32>, GenerationProfile)> {
        generate_streaming(self, input, max_new_tokens, on_token, eos_token_id)
    }

    /// Object-safe reusable-storage entry used by `AutoModel` and services.
    fn generate_streaming_into_dyn(
        &mut self,
        input: LlmInput<'_>,
        max_new_tokens: usize,
        generated: &mut Vec<u32>,
        on_token: &mut dyn FnMut(u32),
        eos_token_id: Option<u32>,
    ) -> Result<GenerationProfile> {
        generate_streaming_into(self, input, max_new_tokens, generated, on_token, eos_token_id)
    }
}


/// Run the shared greedy loop and return owned output storage.
pub fn generate_streaming<M, F>(
    model: &mut M,
    input: LlmInput<'_>,
    max_new_tokens: usize,
    on_token: F,
    eos_token_id: Option<u32>,
) -> Result<(Vec<u32>, GenerationProfile)>
where
    M: LlmTrait + ?Sized,
    F: FnMut(u32),
{
    let mut generated = Vec::with_capacity(max_new_tokens);
    let profile = generate_streaming_into(
        model, input, max_new_tokens, &mut generated, on_token, eos_token_id,
    )?;
    Ok((generated, profile))
}

/// Run the canonical greedy loop into reusable caller-owned output storage.
pub fn generate_streaming_into<M, F>(
    model: &mut M,
    input: LlmInput<'_>,
    max_new_tokens: usize,
    generated: &mut Vec<u32>,
    mut on_token: F,
    eos_token_id: Option<u32>,
) -> Result<GenerationProfile>
where
    M: LlmTrait + ?Sized,
    F: FnMut(u32),
{
    let prompt_tokens = input.token_ids;
    if prompt_tokens.is_empty() {
        return Err(Error::Other("generate_streaming: empty prompt".into()));
    }
    if input.image.is_some() && !model.capabilities().image {
        return Err(Error::Other(
            "this model does not support image input".into(),
        ));
    }

    generated.clear();
    generated.reserve(max_new_tokens);

    let prompt_lookup = std::env::var("APXINF_PROMPT_LOOKUP").as_deref() == Ok("1");
    if prompt_lookup && max_new_tokens == 0 {
        let mut profile = GenerationProfile::new();
        profile.finalize(prompt_tokens.len(), 0);
        return Ok(profile);
    }

    let mut profile = GenerationProfile::new();
    let vocab_size = model.vocab_size();

    model.reset();
    model.prewarm_decode(prompt_tokens.len(), max_new_tokens);
    let next_token = match model.prefill_token(input) {
        Some(result) => result?,
        None => {
            let logits = model.prefill(input)?;
            argmax_last_row(&logits, prompt_tokens.len(), vocab_size)?
        }
    };
    profile.record_first_token();

    generated.push(next_token);
    on_token(next_token);

    if eos_token_id == Some(next_token) {
        profile.finalize(prompt_tokens.len(), generated.len());
        return Ok(profile);
    }

    let prompt_len = prompt_tokens.len();
    let mut current_token = next_token;
    let perf = std::env::var_os("APXINF_PERF").is_some();
    let mut t_fwd = std::time::Duration::ZERO;
    let mut t_am = std::time::Duration::ZERO;
    let mut t_cb = std::time::Duration::ZERO;
    let (lookup_ngram, lookup_block) = if prompt_lookup {
        (
            prompt_lookup_env_usize("APXINF_PROMPT_LOOKUP_NGRAM", 8),
            prompt_lookup_env_usize("APXINF_PROMPT_LOOKUP_BLOCK", 8).min(8),
        )
    } else {
        (0, 0)
    };
    let scratch_capacity = lookup_block.min(max_new_tokens.saturating_sub(1));
    let mut draft = Vec::with_capacity(scratch_capacity);
    let mut verified = Vec::with_capacity(scratch_capacity);

    while generated.len() < max_new_tokens {
        let pos = (prompt_len + generated.len() - 1) as u32;
        let remaining = max_new_tokens - generated.len();
        let mut used_block = false;

        if prompt_lookup && lookup_ngram != 0 && lookup_block != 0 {
            fill_prompt_lookup_draft(
                prompt_tokens,
                generated,
                lookup_ngram,
                lookup_block.min(remaining),
                eos_token_id,
                &mut draft,
            );
            if !draft.is_empty() {
                verified.clear();
                if let Some(result) = model.decode_draft_block(
                    current_token,
                    pos,
                    &draft,
                    &mut verified,
                ) {
                    let result = result?;
                    validate_draft_block_result(&draft, &verified, result)?;
                    for &token in &verified {
                        generated.push(token);
                        current_token = token;
                        if perf {
                            let t0 = std::time::Instant::now();
                            on_token(token);
                            t_cb += t0.elapsed();
                        } else {
                            on_token(token);
                        }
                        if eos_token_id == Some(token) {
                            break;
                        }
                    }
                    used_block = true;
                }
            }
        }

        if eos_token_id == Some(current_token) {
            break;
        }
        if used_block {
            continue;
        }

        match model.decode_token(current_token, pos) {
            Some(result) => current_token = result?,
            None if perf => {
                let t0 = std::time::Instant::now();
                let logits = model.forward(std::slice::from_ref(&current_token), pos)?;
                t_fwd += t0.elapsed();
                let t1 = std::time::Instant::now();
                current_token = argmax_last_row(&logits, 1, vocab_size)?;
                t_am += t1.elapsed();
            }
            None => {
                let logits = model.forward(std::slice::from_ref(&current_token), pos)?;
                current_token = argmax_last_row(&logits, 1, vocab_size)?;
            }
        }
        generated.push(current_token);
        if perf {
            let t0 = std::time::Instant::now();
            on_token(current_token);
            t_cb += t0.elapsed();
        } else {
            on_token(current_token);
        }
        if eos_token_id == Some(current_token) {
            break;
        }
    }
    if perf {
        let n = generated.len().saturating_sub(1).max(1) as f32;
        eprintln!(
            "[loop] fwd={:.2}ms am={:.3}ms cb={:.3}ms (per-tok)",
            t_fwd.as_secs_f32() * 1000.0 / n,
            t_am.as_secs_f32() * 1000.0 / n,
            t_cb.as_secs_f32() * 1000.0 / n
        );
    }

    profile.finalize(prompt_len, generated.len());
    Ok(profile)
}
fn prompt_lookup_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn history_token(prompt: &[u32], generated: &[u32], index: usize) -> u32 {
    if index < prompt.len() {
        prompt[index]
    } else {
        generated[index - prompt.len()]
    }
}

fn fill_prompt_lookup_draft(
    prompt: &[u32],
    generated: &[u32],
    max_ngram: usize,
    max_draft: usize,
    eos_token_id: Option<u32>,
    draft: &mut Vec<u32>,
) {
    draft.clear();
    let history_len = prompt.len() + generated.len();
    let largest_ngram = max_ngram.min(history_len.saturating_sub(1));

    for ngram in (1..=largest_ngram).rev() {
        let suffix_start = history_len - ngram;
        for candidate_start in 0..suffix_start {
            if candidate_start + ngram >= history_len {
                continue;
            }
            let matches = (0..ngram).all(|offset| {
                history_token(prompt, generated, candidate_start + offset)
                    == history_token(prompt, generated, suffix_start + offset)
            });
            if !matches {
                continue;
            }

            let available = history_len - (candidate_start + ngram);
            for offset in 0..max_draft.min(available) {
                let token = history_token(prompt, generated, candidate_start + ngram + offset);
                draft.push(token);
                if eos_token_id == Some(token) {
                    break;
                }
            }
            return;
        }
    }
}

fn validate_draft_block_result(
    draft: &[u32],
    verified: &[u32],
    result: DraftBlockResult,
) -> Result<()> {
    let shape_valid = result.consumed_draft != 0
        && result.consumed_draft <= draft.len()
        && verified.len() == result.consumed_draft
        && result.accepted_prefix <= result.consumed_draft;
    let prefix_valid = shape_valid
        && verified[..result.accepted_prefix] == draft[..result.accepted_prefix];
    let completion_valid = shape_valid
        && if result.accepted_prefix == result.consumed_draft {
            result.consumed_draft == draft.len()
        } else {
            result.accepted_prefix + 1 == result.consumed_draft
                && verified[result.accepted_prefix] != draft[result.accepted_prefix]
        };
    if !prefix_valid || !completion_valid {
        return Err(Error::Other(
            "decode_draft_block returned inconsistent metadata".into(),
        ));
    }
    Ok(())
}

fn argmax_last_row(logits: &Tensor, seq_len: usize, vocab_size: usize) -> Result<u32> {
    #[cfg(feature = "cuda")]
    let logits = if logits.device().is_gpu() {
        std::borrow::Cow::Owned(apxinf_cuda::transfers::to_cpu(logits)?)
    } else {
        std::borrow::Cow::Borrowed(logits)
    };
    #[cfg(not(feature = "cuda"))]
    let logits = std::borrow::Cow::Borrowed(logits);
    // The GPU fast path returns only the final row ([1, vocab]); the CPU path
    // returns [seq, vocab]. Trust the tensor's own row count.
    let rows = logits.shape().dims().first().copied().unwrap_or(1);
    let _ = seq_len;
    let last_row_offset = rows.saturating_sub(1) * vocab_size;
    // Fast path: scan bf16 directly (the decode graph returns a bf16 row).
    // Manual loop with `>` beats the iterator + partial_cmp (no NaN handling
    // overhead; logits don't contain NaN in practice).
    if let Ok(data) = logits.as_bf16() {
        let row = &data[last_row_offset..last_row_offset + vocab_size];
        let mut best = half::bf16::from_f32(f32::NEG_INFINITY);
        let mut best_i: u32 = 0;
        for (i, &v) in row.iter().enumerate() {
            if v > best {
                best = v;
                best_i = i as u32;
            }
        }
        return Ok(best_i);
    }
    // Fallback: f32 path for non-bf16 tensors (prefill, CPU models).
    let data = logits.to_f32_vec()?;
    let row = &data[last_row_offset..last_row_offset + vocab_size];
    let mut best = f32::NEG_INFINITY;
    let mut best_i: u32 = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > best {
            best = v;
            best_i = i as u32;
        }
    }
    Ok(best_i)
}
