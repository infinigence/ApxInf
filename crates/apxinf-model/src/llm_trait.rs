//! Common LLM trait for all model implementations.

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use apxinf_core::{
    Backend, Device, Error, NextTokenLogits, Result, Tensor, TokenSampler, TokenSamplingInit,
    TokenSamplingParams, TokenSamplingSpec,
};
use apxinf_loader::ModelConfig;
use serde::Serialize;

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

/// Observable preparation state for text inference. This is separate from
/// the VLA preparation contract and does not select or prepare an execution path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TextPreparationState {
    /// The implementation does not expose explicit preparation. This is not
    /// evidence that resources have been prepared or compilation has succeeded.
    #[default]
    RuntimeManaged,
    Unprepared,
    /// All required preparation has succeeded for the reported profiles.
    Ready,
    /// Preparation or execution failed and the reported error requires recovery.
    Invalidated,
}

/// A compiled scope that the implementation has successfully prepared.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TextCompilationScope {
    LocalSubgraphs,
    DecoderBlock,
    DecodeStep,
    DraftProposal,
}

/// A snapshot of family-owned preparation, without owning runtime resources.
///
/// `compiled_scopes` lists only successfully prepared required scopes: execution
/// of their admitted profiles must fail rather than silently fall back to an
/// uncompiled path. A selected variant alone does not establish preparation.
/// Empty scopes do not imply that an implementation has no optimized operations.
/// The family defines the scope/profile relationship; these fields do not promise
/// the absence of driver JIT, retracing, host transfers, or temporary allocations.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TextPreparationStatus {
    pub state: TextPreparationState,
    pub implementation: &'static str,
    pub variant: Option<&'static str>,
    pub compiled_scopes: Vec<TextCompilationScope>,
    pub prepared_prompt_tokens: Option<usize>,
    /// The successfully admitted sequence lengths, in ascending order.
    pub prepared_sequence_lengths: Vec<usize>,
    pub kv_capacity: Option<usize>,
    pub max_decode_rows: Option<usize>,
    pub error: Option<String>,
}

impl Default for TextPreparationStatus {
    fn default() -> Self {
        Self {
            state: TextPreparationState::RuntimeManaged,
            implementation: "legacy",
            variant: None,
            compiled_scopes: Vec::new(),
            prepared_prompt_tokens: None,
            prepared_sequence_lengths: Vec::new(),
            kv_capacity: None,
            max_decode_rows: None,
            error: None,
        }
    }
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

/// A control boundary where a caller may stop generation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GenerationBoundary {
    /// No prompt work has started. This does not report consumed positions.
    PrefillStart,
    PrefillProgress {
        consumed_tokens: usize,
    },
    Token(GeneratedToken),
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

/// Cooperative cancellation. A cancelled handle stays cancelled; use a new
/// handle for a new request. An in-flight token read can publish one last token.
#[derive(Clone, Default)]
pub struct GenerationCancellation(Arc<AtomicBool>);

impl GenerationCancellation {
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Request-owned token production. The shared driver owns publication and stops.
#[doc(hidden)]
pub trait TokenSequence {
    /// May submit one successor before reading the current token.
    fn next(&mut self, lookahead: bool) -> Result<GeneratedToken>;
    /// Drain work and restore the state corresponding to the published prefix.
    fn finish(self: Box<Self>, published: usize) -> Result<()>;
    fn overlaps_callbacks(&self) -> bool {
        false
    }
}

enum GenerationSampler<'a> {
    Backend(Box<dyn TokenSampler>),
    Caller(&'a mut dyn TokenSampler),
}

impl GenerationSampler<'_> {
    fn sampler(&mut self) -> &mut dyn TokenSampler {
        match self {
            Self::Backend(sampler) => sampler.as_mut(),
            Self::Caller(sampler) => *sampler,
        }
    }
}

pub(crate) struct SyncTokenSequence<'a, M: LlmTrait + ?Sized> {
    model: &'a mut M,
    sampler: GenerationSampler<'a>,
    first: Option<Tensor>,
    previous: Option<GeneratedToken>,
    position: u32,
}

impl<'a, M: LlmTrait + ?Sized> SyncTokenSequence<'a, M> {
    pub(crate) fn new(
        model: &'a mut M,
        first: Tensor,
        sampler: Box<dyn TokenSampler>,
        position: u32,
    ) -> Self {
        Self {
            model,
            sampler: GenerationSampler::Backend(sampler),
            first: Some(first),
            previous: None,
            position,
        }
    }
    fn with_caller(
        model: &'a mut M,
        first: Tensor,
        sampler: &'a mut dyn TokenSampler,
        position: u32,
    ) -> Self {
        Self { model, first: Some(first), sampler: GenerationSampler::Caller(sampler), previous: None, position }
    }

}

impl<M: LlmTrait + ?Sized> TokenSequence for SyncTokenSequence<'_, M> {
    fn next(&mut self, _lookahead: bool) -> Result<GeneratedToken> {
        let logits = if let Some(first) = self.first.take() {
            first
        } else {
            let logits = self
                .model
                .forward(&[self.previous.unwrap().token_id], self.position)?;
            self.position += 1;
            logits
        };
        let sample = self
            .sampler
            .sampler()
            .sample(NextTokenLogits::last(&logits, self.model.vocab_size())?)?;
        let token = GeneratedToken {
            token_id: sample.token_id,
            logprob: sample.logprob,
        };
        self.previous = Some(token);
        Ok(token)
    }
    fn finish(self: Box<Self>, _published: usize) -> Result<()> {
        Ok(())
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

    /// Inspect family-owned preparation without performing compilation, changing
    /// state, or selecting a runtime. Legacy implementations report
    /// [`TextPreparationState::RuntimeManaged`], which is not an explicit Ready.
    fn preparation_status(&self) -> TextPreparationStatus {
        TextPreparationStatus::default()
    }

    /// Process a complete prompt and return its logits.
    /// A generation-oriented implementation may return only the final row
    /// `[1, vocab_size]`, after consuming every prompt position into its state.
    /// Call [`Self::forward`] when logits for every input position are needed.
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

    /// Process a prompt with control before work and after evaluated progress.
    /// A false callback result stops prefill without logits for the sampler.
    /// The caller must still release partial request state after a stop or error.
    fn prefill_controlled(
        &mut self,
        input: LlmInput<'_>,
        boundary: &mut dyn FnMut(GenerationBoundary) -> Result<bool>,
    ) -> Result<Option<Tensor>> {
        if !boundary(GenerationBoundary::PrefillStart)? {
            return Ok(None);
        }
        let logits = self.prefill(input)?;
        if !boundary(GenerationBoundary::PrefillProgress {
            consumed_tokens: input.token_ids.len(),
        })? {
            return Ok(None);
        }
        Ok(Some(logits))
    }

    /// Reset state for a new generation.
    fn reset(&mut self);

    /// Release request-owned state after checked device settlement.
    fn release_request_state(&mut self) -> Result<()> {
        Err(Error::Contract("request state release is unsupported"))
    }

    /// Release request arrays while retaining one bounded model-owned preparation entry.
    /// Implementations must reject this operation unless they separate these lifetimes.
    fn release_request_state_retaining_preparation(&mut self) -> Result<()> {
        Err(Error::Contract("prepared-code retention is unsupported"))
    }

    /// Run the shared loop with control before prefill and at completed execution boundaries.
    fn generate_streaming_controlled_dyn(
        &mut self,
        request: GenerationRequest<'_>,
        boundary: &mut dyn FnMut(GenerationBoundary) -> Result<bool>,
    ) -> Result<GenerationOutput> {
        let options = request.options.resolve()?;
        generate_streaming_with_resolved_options(self, request.input, &options, boundary)
    }

    /// Optional hook called once before prefill, with the prompt length and
    /// the number of tokens that will be generated. Models with a CUDA
    /// decode graph use it to pre-capture every bucket they'll hit so the
    /// per-token TPOT stays at pure graph-replay cost. Default: no-op.
    fn prewarm_decode(&mut self, _prompt_len: usize, _max_new_tokens: usize) {}

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

    /// Request-owned producer; the default preserves synchronous sampling.
    #[doc(hidden)]
    fn token_sequence<'a>(
        &'a mut self,
        first: Tensor,
        sampler: Box<dyn TokenSampler>,
        _params: &TokenSamplingParams,
        prompt_end: u32,
    ) -> Result<Box<dyn TokenSequence + 'a>> {
        Ok(Box::new(SyncTokenSequence::new(
            self, first, sampler, prompt_end,
        )))
    }

    /// Cooperative cancellation through the same shared publication loop.
    fn generate_streaming_cancellable_dyn(
        &mut self,
        request: GenerationRequest<'_>,
        cancel: &GenerationCancellation,
        on_token: &mut dyn FnMut(GeneratedToken),
    ) -> Result<GenerationOutput> {
        let options = request.options.resolve()?;
        if cancel.is_cancelled() {
            return Ok(empty_generation_output(request.input.token_ids.len(), GenerationProfile::new()));
        }
        generate_streaming_with_resolved_options(self, request.input, &options, |event| {
            if cancel.is_cancelled() { return Ok(false); }
            if let GenerationBoundary::Token(token) = event { on_token(token); }
            Ok(!cancel.is_cancelled())
        })
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
    mut on_token: F,
) -> Result<GenerationOutput>
where
    M: LlmTrait + ?Sized,
    F: FnMut(GeneratedToken),
{
    let options = request.options.resolve()?;
    generate_streaming_with_resolved_options(model, request.input, &options, |boundary| {
        if let GenerationBoundary::Token(token) = boundary {
            on_token(token);
        }
        Ok(true)
    })
}

fn generate_streaming_with_resolved_options<M, F>(
    model: &mut M,
    input: LlmInput<'_>,
    options: &ResolvedGenerationOptions,
    boundary: F,
) -> Result<GenerationOutput>
where
    M: LlmTrait + ?Sized,
    F: FnMut(GenerationBoundary) -> Result<bool>,
{
    validate_generation_input(model, input)?;
    let profile = GenerationProfile::new();
    if options.max_new_tokens == 0 {
        return Ok(empty_generation_output(input.token_ids.len(), profile));
    }
    let spec = generation_sampling_spec(model, input, options)?;
    let sampler = model.backend().create_token_sampler(spec)?;
    generate_streaming_with_resolved_sampler(
        model,
        input,
        options,
        GenerationSampler::Backend(sampler),
        profile,
        boundary,
    )
}

/// Run controlled generation with a caller-selected sampler.
/// Check its vocabulary and capacity before model mutation.
/// The caller must supply a sampler that accepts this model's logits.
pub fn generate_streaming_controlled_with_sampler<M, F>(
    model: &mut M,
    request: GenerationRequest<'_>,
    sampler: &mut dyn TokenSampler,
    boundary: F,
) -> Result<GenerationOutput>
where
    M: LlmTrait + ?Sized,
    F: FnMut(GenerationBoundary) -> Result<bool>,
{
    let options = request.options.resolve()?;
    validate_generation_input(model, request.input)?;
    let profile = GenerationProfile::new();
    let required = generation_sampling_spec(model, request.input, &options)?;
    let spec = sampler.spec();
    spec.validate()?;
    if spec.vocab_size != required.vocab_size {
        return Err(Error::Other(format!(
            "sampler vocabulary size must equal {}, got {}",
            required.vocab_size, spec.vocab_size
        )));
    }
    if spec.max_sequence_len < required.max_sequence_len {
        return Err(Error::Other(format!(
            "sampler capacity must be at least {}, got {}",
            required.max_sequence_len, spec.max_sequence_len
        )));
    }
    if options.max_new_tokens == 0 {
        return Ok(empty_generation_output(
            request.input.token_ids.len(),
            profile,
        ));
    }
    generate_streaming_with_resolved_sampler(
        model,
        request.input,
        &options,
        GenerationSampler::Caller(sampler),
        profile,
        boundary,
    )
}

fn validate_generation_input<M: LlmTrait + ?Sized>(model: &M, input: LlmInput<'_>) -> Result<()> {
    if input.token_ids.is_empty() {
        return Err(Error::Other("generate_streaming: empty prompt".into()));
    }
    if input.image.is_some() && !model.capabilities().image {
        return Err(Error::Other(
            "this model does not support image input".into(),
        ));
    }
    Ok(())
}

fn generation_sampling_spec<M: LlmTrait + ?Sized>(
    model: &M,
    input: LlmInput<'_>,
    options: &ResolvedGenerationOptions,
) -> Result<TokenSamplingSpec> {
    let capacity = input
        .token_ids
        .len()
        .checked_add(options.max_new_tokens)
        .ok_or_else(|| Error::Other("generation sequence length overflow".into()))?;
    Ok(TokenSamplingSpec {
        vocab_size: model.vocab_size(),
        max_sequence_len: capacity,
    })
}

fn empty_generation_output(prompt_len: usize, mut profile: GenerationProfile) -> GenerationOutput {
    profile.finalize(prompt_len, 0);
    GenerationOutput {
        tokens: Vec::new(),
        profile,
    }
}

fn generate_streaming_with_resolved_sampler<M, F>(
    model: &mut M,
    input: LlmInput<'_>,
    options: &ResolvedGenerationOptions,
    mut sampler: GenerationSampler<'_>,
    mut profile: GenerationProfile,
    mut boundary: F,
) -> Result<GenerationOutput>
where
    M: LlmTrait + ?Sized,
    F: FnMut(GenerationBoundary) -> Result<bool>,
{
    let prompt_tokens = input.token_ids;
    let prompt_end = u32::try_from(prompt_tokens.len())
        .map_err(|_| Error::Other("generation position exceeds u32".into()))?;
    u32::try_from(prompt_tokens.len().checked_add(options.max_new_tokens)
        .ok_or_else(|| Error::Other("generation sequence length overflow".into()))?)
        .map_err(|_| Error::Other("generation position exceeds u32".into()))?;
    sampler.sampler().begin(TokenSamplingInit {
        prompt_token_ids: prompt_tokens,
        params: &options.sampling,
        rng: options.rng,
    })?;
    if !boundary(GenerationBoundary::PrefillStart)? {
        return Ok(empty_generation_output(prompt_tokens.len(), profile));
    }
    let preparation_started = std::time::Instant::now();
    model.reset();
    model.prewarm_decode(prompt_tokens.len(), options.max_new_tokens);
    profile.record_preparation(preparation_started.elapsed());
    let Some(logits) = model.prefill_controlled(input, &mut |event| {
        if matches!(event, GenerationBoundary::PrefillStart) { Ok(true) } else { boundary(event) }
    })? else {
        return Ok(empty_generation_output(prompt_tokens.len(), profile));
    };
    let mut sequence: Box<dyn TokenSequence + '_> = match sampler {
        GenerationSampler::Backend(sampler) => model.token_sequence(logits, sampler, &options.sampling, prompt_end)?,
        GenerationSampler::Caller(sampler) => Box::new(SyncTokenSequence::with_caller(model, logits, sampler, prompt_end)),
    };
    let overlaps = sequence.overlaps_callbacks();
    let mut generated = Vec::with_capacity(options.max_new_tokens);
    let run = (|| -> Result<()> {
        for index in 0..options.max_new_tokens {
            let token = sequence.next(index + 1 < options.max_new_tokens)?;
            if index == 0 { profile.record_first_token(); }
            generated.push(token);
            let keep_going = if overlaps { boundary(GenerationBoundary::Token(token))? }
                else { profile.run_unprofiled(|| boundary(GenerationBoundary::Token(token)))? };
            if !keep_going || options.eos_token_ids.contains(&token.token_id) { break; }
        }
        Ok(())
    })();
    let cleanup = sequence.finish(generated.len());
    match (run, cleanup) {
        (Err(primary), Err(cleanup)) => return Err(Error::Other(format!("{primary}; cleanup failed: {cleanup}"))),
        (Err(primary), _) => return Err(primary),
        (_, Err(cleanup)) => return Err(cleanup),
        _ => {}
    }
    profile.finalize(prompt_tokens.len(), generated.len());
    Ok(GenerationOutput { tokens: generated, profile })
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
