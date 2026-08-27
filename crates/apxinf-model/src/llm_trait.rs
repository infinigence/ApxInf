//! Common LLM trait for all model implementations.

use std::collections::HashMap;

use apxinf_core::{
    Backend, Device, Error, NextTokenLogits, Result, Tensor, TokenSamplingInit,
    TokenSamplingParams, TokenSamplingSpec,
};
use apxinf_loader::ModelConfig;

use crate::generation_config::{GenerationOptions, ResolvedGenerationOptions};
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

/// Processor output for one or more audio clips in a generation prompt.
///
/// Features use the model-owned `[frames, mel_bins]` layout. The mask is the
/// processor-produced frame mask, `feature_lengths` identifies the valid
/// frames in each group, and `token_counts` maps each group to the expanded
/// audio-placeholder run in `token_ids`. The model validates all four views
/// before executing the audio tower.
#[derive(Clone, Copy, Debug)]
pub struct AudioInput<'a> {
    pub input_features: &'a Tensor,
    pub attention_mask: &'a Tensor,
    pub feature_lengths: &'a [u32],
    pub token_counts: &'a [u32],
}

impl<'a> AudioInput<'a> {
    pub const fn new(
        input_features: &'a Tensor,
        attention_mask: &'a Tensor,
        feature_lengths: &'a [u32],
        token_counts: &'a [u32],
    ) -> Self {
        Self {
            input_features,
            attention_mask,
            feature_lengths,
            token_counts,
        }
    }
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
    pub audio: Option<AudioInput<'a>>,
}

impl<'a> LlmInput<'a> {
    pub const fn text(token_ids: &'a [u32]) -> Self {
        Self {
            token_ids,
            image: None,
            audio: None,
        }
    }

    pub const fn with_image(token_ids: &'a [u32], image: ImageInput<'a>) -> Self {
        Self {
            token_ids,
            image: Some(image),
            audio: None,
        }
    }

    pub const fn with_audio(token_ids: &'a [u32], audio: AudioInput<'a>) -> Self {
        Self {
            token_ids,
            image: None,
            audio: Some(audio),
        }
    }

    pub const fn with_media(
        token_ids: &'a [u32],
        image: Option<ImageInput<'a>>,
        audio: Option<AudioInput<'a>>,
    ) -> Self {
        Self {
            token_ids,
            image,
            audio,
        }
    }
}

/// Input modalities accepted by an LLM implementation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LlmCapabilities {
    pub image: bool,
    pub audio: bool,
}

impl LlmCapabilities {
    pub const TEXT_ONLY: Self = Self {
        image: false,
        audio: false,
    };
    pub const VISION: Self = Self {
        image: true,
        audio: false,
    };
    pub const OMNI: Self = Self {
        image: true,
        audio: true,
    };
}

/// Complete prompt plus generation policy.
#[derive(Clone, Copy, Debug)]
pub struct GenerationRequest<'a> {
    pub input: LlmInput<'a>,
    pub options: &'a GenerationOptions,
}

/// One generated token and its optional post-filter log-probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeneratedToken {
    pub token_id: u32,
    pub logprob: Option<f32>,
}

/// Generated tokens together with host-side timing metrics.
pub struct GenerationOutput {
    pub tokens: Vec<GeneratedToken>,
    pub profile: GenerationProfile,
}

impl GenerationOutput {
    pub fn token_ids(&self) -> Vec<u32> {
        self.tokens.iter().map(|token| token.token_id).collect()
    }
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

    /// Backend that owns model tensors and creates the model-neutral sampler.
    fn backend(&self) -> &dyn Backend;

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
        if input.audio.is_some() {
            return Err(Error::Other(
                "this model does not support audio input".into(),
            ));
        }
        self.forward(input.token_ids, 0)
    }

    /// Reset state for a new generation.
    fn reset(&mut self);

    /// Optional hook called once before prefill, with the prompt length and
    /// the number of tokens that will be generated. Models with a CUDA
    /// decode graph use it to pre-capture every bucket they'll hit so the
    /// per-token TPOT stays at pure graph-replay cost. Default: no-op.
    fn prewarm_decode(&mut self, _prompt_len: usize, _max_new_tokens: usize) {}

    /// Optional hard context capacity. The shared loop rejects combined
    /// prompt+completion overflow before prewarm or cache mutation.
    fn max_context_len(&self) -> Option<usize> {
        None
    }

    /// Optional per-request generation limit owned by a deployed model.
    fn max_new_tokens_limit(&self) -> Option<usize> {
        None
    }

    /// Greedy-decode one token directly to its id, skipping the full-logits
    /// D2H + CPU argmax. Returns `None` if the model has no GPU-argmax fast
    /// path (caller falls back to `forward` + `argmax_last_row`).
    fn decode_token(&mut self, _token: u32, _pos: u32) -> Option<Result<u32>> {
        None
    }

    /// Vocabulary size used to validate logits and allocate sampler state.
    fn vocab_size(&self) -> usize;

    /// Options-based streaming entry point with GPU sampling support.
    fn generate_streaming_with_options(
        &mut self,
        request: GenerationRequest<'_>,
        on_token: impl FnMut(GeneratedToken),
    ) -> Result<GenerationOutput>
    where
        Self: Sized,
    {
        generate_streaming_with_options(self, request, on_token)
    }

    /// Ergonomic, statically typed streaming entrypoint. Models that replace
    /// the shared greedy algorithm should override `generate_streaming_dyn`
    /// so the same behavior is visible through `AutoModel`.
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

    /// Object-safe options-based entry used by [`crate::LoadedModel`].
    fn generate_streaming_with_options_dyn(
        &mut self,
        request: GenerationRequest<'_>,
        on_token: &mut dyn FnMut(GeneratedToken),
    ) -> Result<GenerationOutput> {
        generate_streaming_with_options(self, request, on_token)
    }

    /// Object-safe entry used by `AutoModel`. The vtable dispatch happens
    /// once for the complete request; the concrete model then owns the whole
    /// prefill/decode loop rather than paying model dispatch per token.
    fn generate_streaming_dyn(
        &mut self,
        input: LlmInput<'_>,
        max_new_tokens: usize,
        on_token: &mut dyn FnMut(u32),
        eos_token_id: Option<u32>,
    ) -> Result<(Vec<u32>, GenerationProfile)> {
        generate_streaming(self, input, max_new_tokens, on_token, eos_token_id)
    }
}

/// Run the shared sampling-aware generation loop. Model implementations only
/// produce device-resident logits; this driver owns EOS handling and invokes
/// the sampler created by the model's backend.
pub fn generate_streaming_with_options<M, F>(
    model: &mut M,
    request: GenerationRequest<'_>,
    on_token: F,
) -> Result<GenerationOutput>
where
    M: LlmTrait + ?Sized,
    F: FnMut(GeneratedToken),
{
    let options = request.options.resolve()?;
    generate_streaming_with_resolved_options(model, request.input, &options, on_token)
}

fn generate_streaming_with_resolved_options<M, F>(
    model: &mut M,
    input: LlmInput<'_>,
    options: &ResolvedGenerationOptions,
    mut on_token: F,
) -> Result<GenerationOutput>
where
    M: LlmTrait + ?Sized,
    F: FnMut(GeneratedToken),
{
    let prompt_tokens = input.token_ids;
    validate_generation_limits(
        prompt_tokens.len(),
        options.max_new_tokens,
        model.max_new_tokens_limit(),
        model.max_context_len(),
    )?;
    if input.image.is_some() && !model.capabilities().image {
        return Err(Error::Other(
            "this model does not support image input".into(),
        ));
    }
    if input.audio.is_some() && !model.capabilities().audio {
        return Err(Error::Other(
            "this model does not support audio input".into(),
        ));
    }
    let mut profile = GenerationProfile::new();
    if options.max_new_tokens == 0 {
        profile.finalize(prompt_tokens.len(), 0);
        return Ok(GenerationOutput {
            tokens: Vec::new(),
            profile,
        });
    }

    let capacity = prompt_tokens
        .len()
        .checked_add(options.max_new_tokens)
        .ok_or_else(|| Error::Other("generation sequence length overflow".into()))?;
    let spec = TokenSamplingSpec {
        vocab_size: model.vocab_size(),
        max_sequence_len: capacity,
    };
    let mut sampler = model.backend().create_token_sampler(spec)?;
    sampler.begin(TokenSamplingInit {
        prompt_token_ids: prompt_tokens,
        params: &options.sampling,
        rng: options.rng,
    })?;

    model.reset();
    model.prewarm_decode(prompt_tokens.len(), options.max_new_tokens);

    let mut generated = Vec::with_capacity(options.max_new_tokens);
    let logits = model.prefill(input)?;
    let first = sampler.sample(NextTokenLogits::last(&logits, spec.vocab_size)?)?;
    profile.record_first_token();
    let mut current = GeneratedToken {
        token_id: first.token_id,
        logprob: first.logprob,
    };
    generated.push(current);
    on_token(current);
    let use_decode_token = options.sampling == TokenSamplingParams::greedy();

    for index in 1..options.max_new_tokens {
        if options.eos_token_ids.contains(&current.token_id) {
            break;
        }
        let position = prompt_tokens
            .len()
            .checked_add(index - 1)
            .and_then(|position| u32::try_from(position).ok())
            .ok_or_else(|| Error::Other("generation position exceeds u32".into()))?;
        if use_decode_token {
            if let Some(result) = model.decode_token(current.token_id, position) {
                current = GeneratedToken {
                    token_id: result?,
                    logprob: None,
                };
                generated.push(current);
                on_token(current);
                continue;
            }
        }
        let logits = model.forward(&[current.token_id], position)?;
        let sample = sampler.sample(NextTokenLogits::last(&logits, spec.vocab_size)?)?;
        current = GeneratedToken {
            token_id: sample.token_id,
            logprob: sample.logprob,
        };
        generated.push(current);
        on_token(current);
    }

    profile.finalize(prompt_tokens.len(), generated.len());
    Ok(GenerationOutput {
        tokens: generated,
        profile,
    })
}

/// Run the shared greedy generation loop for a concrete model or
/// `dyn LlmTrait`. Most callers use [`LlmTrait::generate_streaming`] or
/// [`crate::LoadedModel::generate_streaming`]; this function contains the one
/// canonical generation algorithm.
pub fn generate_streaming<M, F>(
    model: &mut M,
    input: LlmInput<'_>,
    max_new_tokens: usize,
    mut on_token: F,
    eos_token_id: Option<u32>,
) -> Result<(Vec<u32>, GenerationProfile)>
where
    M: LlmTrait + ?Sized,
    F: FnMut(u32),
{
    let options = GenerationOptions::greedy(max_new_tokens, eos_token_id);
    let output = generate_streaming_with_options(
        model,
        GenerationRequest {
            input,
            options: &options,
        },
        |token| on_token(token.token_id),
    )?;
    Ok((output.token_ids(), output.profile))
}

/// Validate the one canonical autoregressive capacity contract.
///
/// Prompt and completion limits are not independent allocation promises: a
/// request must satisfy both the completion cap and `prompt + completion <=
/// context`. The checked addition makes oversized input fail before model,
/// cache, or backend work.
pub fn validate_generation_limits(
    prompt_tokens: usize,
    max_new_tokens: usize,
    max_new_tokens_limit: Option<usize>,
    context_limit: Option<usize>,
) -> Result<()> {
    if prompt_tokens == 0 {
        return Err(Error::Other("generate_streaming: empty prompt".into()));
    }
    if let Some(limit) = max_new_tokens_limit {
        if max_new_tokens > limit {
            return Err(Error::Other(format!(
                "generate_streaming: max_new_tokens {max_new_tokens} exceeds model limit {limit}"
            )));
        }
    }
    if let Some(limit) = context_limit {
        let required = prompt_tokens.checked_add(max_new_tokens).ok_or_else(|| {
            Error::Other("generate_streaming: combined context length overflow".into())
        })?;
        if required > limit {
            return Err(Error::Other(format!(
                "generate_streaming: prompt {prompt_tokens} + completion {max_new_tokens} exceeds context {limit}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use apxinf_core::{Backend, CpuBackend, Device, Error, Result, Tensor};
    use apxinf_loader::ModelConfig;

    use super::{validate_generation_limits, LlmInput, LlmTrait};

    struct DecodeErrorModel {
        forward_calls: usize,
    }

    impl LlmTrait for DecodeErrorModel {
        fn load(
            _config: ModelConfig,
            _weights: HashMap<String, Tensor>,
            _device: Device,
        ) -> Result<Self> {
            Ok(Self { forward_calls: 0 })
        }

        fn forward(&mut self, _token_ids: &[u32], _start_pos: u32) -> Result<Tensor> {
            self.forward_calls += 1;
            assert_eq!(self.forward_calls, 1, "decode error fell back to forward");
            Tensor::from_f32(vec![1, 2], &[0.0, 1.0])
        }

        fn backend(&self) -> &dyn Backend {
            static BACKEND: CpuBackend = CpuBackend;
            &BACKEND
        }

        fn reset(&mut self) {}

        fn decode_token(&mut self, _token: u32, _pos: u32) -> Option<Result<u32>> {
            Some(Err(Error::Other("GPU token selection failed".into())))
        }

        fn vocab_size(&self) -> usize {
            2
        }
    }

    #[test]
    fn combined_context_contract_is_checked_and_fail_closed() {
        assert!(validate_generation_limits(32_640, 128, Some(128), Some(32_768)).is_ok());
        assert!(validate_generation_limits(32_767, 1, Some(128), Some(32_768)).is_ok());
        assert!(validate_generation_limits(32_768, 1, Some(128), Some(32_768)).is_err());
        assert!(validate_generation_limits(1, 129, Some(128), Some(32_768)).is_err());
        assert!(validate_generation_limits(usize::MAX, 1, Some(128), Some(usize::MAX)).is_err());
        assert!(validate_generation_limits(0, 1, Some(128), Some(32_768)).is_err());
        assert!(validate_generation_limits(1, 0, Some(128), Some(32_768)).is_ok());
    }

    #[test]
    fn decode_token_error_fails_closed_without_forward_fallback() {
        let mut model = DecodeErrorModel { forward_calls: 0 };
        let error = model
            .generate_streaming(LlmInput::text(&[1]), 2, |_| {}, None)
            .err()
            .expect("GPU token-selection error must fail generation");
        assert!(error.to_string().contains("GPU token selection failed"));
        assert_eq!(model.forward_calls, 1);
    }
}
