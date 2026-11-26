//! GPT-SoVITS V5 network semantics and the single synthesis orchestration.

use std::path::Path;
use std::sync::Arc;

use apxinf_core::{Error, Result, Tensor};
use apxinf_cuda_new::{transfers, CudaContext as Context};

use super::config::GptSovitsConfig;
use super::ops;
use super::weights::Weights;
use crate::tts_trait::{gpt_sovits_inputs, GeneratedAudio, TtsRequest, TtsSampling};

mod bert;
mod cfm;
pub(super) mod conditioning;
mod executor;
mod gpt;
mod hubert;
mod sovits;
mod vocoder;

pub(crate) use self::bert::Bert;
pub(crate) use self::cfm::{run_chunked, validate_sample_steps, Cfm, StepModulation};
pub(crate) use self::cfm::{ChunkArgs, MAX_TARGET_CHUNK, REFERENCE_FRAMES, TOTAL_CHUNK};
pub(crate) use self::executor::SynthesisExecutor;
pub(crate) use self::gpt::{
    device_sample_fp16_exponential_race, logits_prefix, sample_fp16_exponential_race, DecodeCache,
    DecodeEmbedScratch, Gpt, DECODE_WORKSPACE_BYTES, DEVICE_SAMPLING_MAX_TOP_K,
    MIN_GENERATED_BEFORE_EOS,
};
pub(crate) use self::hubert::Hubert;
pub(crate) use self::sovits::{repeat_row_ids, Sovits};
pub(crate) use self::vocoder::Vocoder;

/// Sampling defaults used when a [`TtsRequest`] leaves its knobs unset. They
/// mirror the official GPT-SoVITS V5 inference settings.
const DEFAULT_TOP_K: usize = 5;
const DEFAULT_TOP_P: f32 = 1.0;
const DEFAULT_TEMPERATURE: f32 = 1.0;
const DEFAULT_REPETITION_PENALTY: f32 = 1.35;
const DEFAULT_MAX_SECONDS: f32 = 30.0;

/// Resolve a request's sampling knobs: explicit values win, otherwise the
/// family defaults above.
pub(super) fn sampling_of(request: &TtsRequest<'_>) -> (usize, f32, f32, f32, f32) {
    let s: &TtsSampling<'_> = &request.sampling;
    (
        s.top_k.unwrap_or(DEFAULT_TOP_K),
        s.top_p.unwrap_or(DEFAULT_TOP_P),
        s.temperature.unwrap_or(DEFAULT_TEMPERATURE),
        s.repetition_penalty.unwrap_or(DEFAULT_REPETITION_PENALTY),
        s.max_seconds.unwrap_or(DEFAULT_MAX_SECONDS),
    )
}

pub(super) const SPEC_MIN: f32 = -12.0;
pub(super) const SPEC_MAX: f32 = 2.0;
pub(crate) const OUTPUT_SAMPLE_RATE: u32 = 48000;
/// Semantic tokens per second and the hard cap on one request's generation.
const SEMANTIC_TOKENS_PER_SECOND: f32 = 25.0;
const MAX_SEMANTIC_TOKENS: usize = 1500;

/// Semantic-token budget for `max_seconds` (at least one second's worth).
pub(super) fn max_semantic_tokens(max_seconds: f32) -> usize {
    ((max_seconds.max(1.0) * SEMANTIC_TOKENS_PER_SECOND) as usize).min(MAX_SEMANTIC_TOKENS)
}

/// Wrap vocoder output. `Gpt::generate` stops before pushing EOS, so it
/// returns fewer than `max_new_tokens` tokens exactly when EOS was sampled;
/// a full budget means the length limit cut generation short.
pub(super) fn mono_audio(
    samples: Vec<f32>,
    generated_tokens: usize,
    max_new_tokens: usize,
) -> GeneratedAudio {
    GeneratedAudio {
        sample_rate: OUTPUT_SAMPLE_RATE,
        channels: 1,
        samples,
        complete: generated_tokens < max_new_tokens,
    }
}

/// Start offset of the reference-feature window paired with the prompt mel.
///
/// Official `synthesize_v5_mel` first truncates both streams to
/// `min(mel_frames, feature_frames)`, then keeps their last 500 frames. The
/// public request only carries the already-capped prompt mel, so the original
/// mel length is reconstructed from the spectrogram (`2 * spec_frames`, whose
/// hop is twice the mel hop). A zero offset still needs an explicit slice:
/// short prompts must use the head of the feature stream, not its tail.
pub(super) fn reference_feature_window(
    decoded_frames: usize,
    reference_spec_frames: usize,
    reference_frames: usize,
) -> Option<usize> {
    let limit = reference_spec_frames.saturating_mul(2).min(decoded_frames);
    let offset = limit.saturating_sub(reference_frames);
    (offset + reference_frames <= decoded_frames).then_some(offset)
}

// Upstream evaluates (x + 1) / 2 * 14 - 12 in FP16. The addition before
// scaling and the subtraction after scaling each round independently.
// For FP16 x, dividing round(x + 1) by two is exact, so the two scales can
// combine to seven while preserving those rounding boundaries.
pub(super) fn denormalize_mel(
    ctx: &Context,
    input: &Tensor,
    one: &Tensor,
    scale: &Tensor,
    zero: &Tensor,
    minimum: &Tensor,
) -> Result<Tensor> {
    let shifted = ops::add_bias(ctx, input, one)?;
    let scaled = ops::affine(ctx, &shifted, scale, zero)?;
    ops::add_bias(ctx, &scaled, minimum)
}

pub(super) fn owned_tensor(ctx: &Context, source: &Tensor) -> Result<Tensor> {
    let value = ctx.allocate_output(source.shape().clone(), source.dtype())?;
    transfers::copy_cuda_to_cuda(ctx, source, &value)?;
    Ok(value)
}

pub struct GptSovitsModel {
    pub(super) ctx: Arc<Context>,
    pub(crate) bert: Arc<Bert>,
    pub(crate) hubert: Arc<Hubert>,
    pub(crate) gpt: Arc<Gpt>,
    pub(crate) sovits: Arc<Sovits>,
    pub(crate) cfm: Arc<Cfm>,
    pub(crate) vocoder: Arc<Vocoder>,
    pub(crate) config: GptSovitsConfig,
    mel_denorm: MelDenormConstants,
}

/// The four mel-denormalization affine constants (see `denormalize_mel`),
/// uploaded once at load and shared by every synthesis request.
struct MelDenormConstants {
    one: Tensor,
    scale: Tensor,
    zero: Tensor,
    shift: Tensor,
}

impl GptSovitsModel {
    pub fn load_with_autotune(directory: &Path, device_id: usize, autotune: bool) -> Result<Self> {
        let config = GptSovitsConfig::from_dir(directory)?;
        let ctx = Context::new_with_autotune(device_id, autotune).map_err(Error::Cuda)?;
        let mut weights = Weights::load(directory)?;
        let bert = Bert::load(&mut weights, &ctx)?;
        let hubert = Hubert::load(&mut weights, &ctx)?;
        let gpt = Gpt::load(&mut weights, &ctx)?;
        let sovits = Sovits::load(&mut weights, &ctx)?;
        let cfm = Cfm::load(&mut weights, &ctx)?;
        let vocoder = Vocoder::load(&mut weights, &ctx)?;
        let mel_denorm = MelDenormConstants {
            one: super::weights::fp16_tensor(&ctx, &[100], &[1.; 100])?,
            scale: super::weights::fp16_tensor(
                &ctx,
                &[100],
                &vec![(SPEC_MAX - SPEC_MIN) / 2.0; 100],
            )?,
            zero: super::weights::fp16_tensor(&ctx, &[100], &[0.; 100])?,
            shift: super::weights::fp16_tensor(&ctx, &[100], &vec![SPEC_MIN; 100])?,
        };
        Ok(Self {
            ctx: Arc::new(ctx),
            bert: Arc::new(bert),
            hubert: Arc::new(hubert),
            gpt: Arc::new(gpt),
            sovits: Arc::new(sovits),
            cfm: Arc::new(cfm),
            vocoder: Arc::new(vocoder),
            config,
            mel_denorm,
        })
    }

    fn codes_from_vec(&self, codes: &[u32]) -> Result<Tensor> {
        let values: Vec<i32> = codes.iter().map(|value| *value as i32).collect();
        let host = Tensor::from_i32(vec![values.len()], &values)?;
        transfers::to_cuda(&host, self.ctx.device_id())
    }

    /// The single network orchestration: conditioning, semantic decoding,
    /// acoustic feature synthesis, mel denormalization and vocoding, in the
    /// official order and with the official alignment rules. How each
    /// capturable stage executes — eager operators or prepared graph replay —
    /// is the executor's decision; the math and its ordering live here once.
    pub(crate) fn synthesize_with(
        &self,
        request: &TtsRequest<'_>,
        exec: &dyn SynthesisExecutor,
    ) -> Result<GeneratedAudio> {
        let inputs = gpt_sovits_inputs(request)?;
        let (top_k, top_p, temperature, repetition_penalty, max_seconds) = sampling_of(request);
        if !inputs.speed_factor.is_finite() || inputs.speed_factor <= 0.0 {
            return Err(Error::Other(format!(
                "speed_factor must be finite and positive, got {}",
                inputs.speed_factor
            )));
        }
        if inputs.phone_ids.is_empty() || inputs.reference_phone_ids.is_empty() {
            return Err(Error::Other(
                "tts request requires target and reference text".into(),
            ));
        }
        if inputs.reference_audio_16k.len() < 400 {
            return Err(Error::Other("reference audio is too short".into()));
        }
        let reference_frames = inputs.reference_mel.len() / 100;
        let reference_spec_frames = inputs.reference_spec.len() / 704;
        if reference_frames == 0 || reference_spec_frames == 0 {
            return Err(Error::Other("reference mel/spec is empty".into()));
        }
        if inputs.cfg_rate > 1e-5 {
            return Err(Error::Other(
                "gpt_sovits v5turbo does not support classifier-free guidance".into(),
            ));
        }
        validate_sample_steps(inputs.sample_steps)?;
        // NVTX segment timeline for nsys. The graph-replay executor is a black
        // box, so the four stages are only separable with host-side marks;
        // without the `nvtx` feature each range is a zero-cost no-op.
        let _tts_reference_range = crate::profiling::trace::range("tts.reference");
        let reference = exec.reference_features(inputs)?;
        drop(_tts_reference_range);
        let reference_mel = reference.mel.clone();
        let reference_bert = reference.bert.clone();
        let _tts_bert_range = crate::profiling::trace::range("tts.bert");
        let target_bert = exec
            .target_bert(inputs)
            .map_err(|error| Error::Other(format!("target bert: {error}")))?;
        drop(_tts_bert_range);
        let bert = ops::concat_rows(&self.ctx, &reference_bert, &target_bert)
            .map_err(|error| Error::Other(format!("bert concat: {error}")))?;
        let mut phones = inputs.reference_phone_ids.to_vec();
        phones.extend_from_slice(inputs.phone_ids);

        let prompt = &reference.prompt;
        let _tts_gpt_range = crate::profiling::trace::range("tts.gpt");
        let max_new_tokens = max_semantic_tokens(max_seconds);
        let generated = exec
            .generate_semantic(
                &phones,
                &bert,
                prompt,
                request.seed,
                top_k,
                top_p,
                temperature,
                repetition_penalty,
                max_new_tokens,
            )
            .map_err(|error| Error::Other(format!("gpt: {error}")))?;
        drop(_tts_gpt_range);
        if generated.is_empty() {
            return Err(Error::Other("semantic decoder produced no tokens".into()));
        }
        if let Some(expected) = exec.expected_target_frames() {
            let generated_frames = generated
                .len()
                .checked_mul(4)
                .ok_or_else(|| Error::Other("prepared target length overflow".into()))?;
            if generated_frames != expected {
                return Err(Error::Other(format!(
                    "prepared target frame mismatch: graph expects {expected}, GPT produced {generated_frames} frames from {} semantic tokens",
                    generated.len()
                )));
            }
        }
        let target_codes = self.codes_from_vec(&generated)?;

        let ge = reference.style.clone();
        let reference_features = reference.features.clone();
        let _tts_sovits_range = crate::profiling::trace::range("tts.sovits");
        let target_features = exec
            .target_features(&target_codes, inputs.phone_ids, &ge, inputs.speed_factor)
            .map_err(|error| Error::Other(format!("target decode: {error}")))?;
        drop(_tts_sovits_range);
        if let Some(expected) = exec.expected_target_frames() {
            if target_features.shape().dims() != [expected, 512] {
                return Err(Error::Other(format!(
                    "prepared target frame mismatch: graph expects {expected}, GPT produced {} frames from {} semantic tokens",
                    target_features.shape().dims()[0],
                    generated.len()
                )));
            }
        }
        // The tts request carries the final prompt mel (typically 500
        // frames), while reference features are decoded at the pre-truncation
        // rate derived from the full reference spectrogram. PyTorch first
        // truncates to min(2 * spec_frames, feature_frames), then keeps the
        // final prompt-mel window. Reproduce that two-stage alignment here.
        let decoded_frames = reference_features.shape().dims()[0];
        let reference_features =
            match reference_feature_window(decoded_frames, reference_spec_frames, reference_frames)
            {
                Some(offset) => {
                    ops::slice_rows(&self.ctx, &reference_features, offset, reference_frames)?
                }
                None => reference_features,
            };
        let reference_frames = reference_frames
            .min(reference_features.shape().dims()[0])
            .min(REFERENCE_FRAMES);
        let original_mel = ops::slice_rows(
            &self.ctx,
            &reference_mel,
            reference_mel.shape().dims()[0] - reference_frames,
            reference_frames,
        )?;
        let original_features = ops::slice_rows(
            &self.ctx,
            &reference_features,
            reference_features.shape().dims()[0] - reference_frames,
            reference_frames,
        )?;
        let mel = run_chunked(
            &self.ctx,
            &original_mel,
            &original_features,
            &target_features,
            reference_frames,
            |args| {
                let _tts_cfm_range = crate::profiling::trace::range("tts.cfm.chunk");
                let seed = request.seed.wrapping_add(args.chunk_index as u64);
                let generated = exec.cfm_chunk(args, inputs.sample_steps, seed)?;
                drop(_tts_cfm_range);
                Ok(generated)
            },
        )
        .map_err(|error| Error::Other(format!("cfm: {error}")))?;
        let constants = &self.mel_denorm;
        let mel = denormalize_mel(
            &self.ctx,
            &mel,
            &constants.one,
            &constants.scale,
            &constants.zero,
            &constants.shift,
        )?;
        let _tts_vocoder_range = crate::profiling::trace::range("tts.vocoder");
        let samples = exec.vocode(&mel)?;
        drop(_tts_vocoder_range);
        // No implicit trailing pause. The public pipeline appends
        // `fragment_interval` silence after every text fragment, matching the
        // official `audio_postprocess`.
        Ok(mono_audio(samples, generated.len(), max_new_tokens))
    }
}

#[cfg(test)]
mod reference_window_tests {
    use super::reference_feature_window;

    #[test]
    fn short_prompt_uses_feature_head_not_tail() {
        // A 4.95 s reference: 552 decoded feature rows, 245 spectrogram
        // frames, 492 prompt mel frames. The window must start at zero.
        assert_eq!(reference_feature_window(552, 245, 492), Some(0));
    }

    #[test]
    fn long_prompt_uses_last_500_frames_of_estimated_mel() {
        // A 6.09 s reference with a 500-frame capped prompt mel.
        assert_eq!(reference_feature_window(668, 302, 500), Some(104));
    }

    #[test]
    fn window_is_clamped_to_decoded_features() {
        assert_eq!(reference_feature_window(300, 302, 300), Some(0));
        assert_eq!(reference_feature_window(100, 10, 200), None);
    }
}

#[cfg(test)]
mod output_tests {
    use super::{max_semantic_tokens, mono_audio, MAX_SEMANTIC_TOKENS, OUTPUT_SAMPLE_RATE};

    #[test]
    fn semantic_budget_matches_seconds_and_cap() {
        assert_eq!(max_semantic_tokens(0.2), 25);
        assert_eq!(max_semantic_tokens(30.0), 750);
        assert_eq!(max_semantic_tokens(600.0), MAX_SEMANTIC_TOKENS);
    }

    #[test]
    fn audio_is_mono_and_reports_length_limit() {
        let audio = mono_audio(vec![0.0; 8], 12, 750);
        assert_eq!(audio.sample_rate, OUTPUT_SAMPLE_RATE);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.samples.len(), 8);
        assert!(audio.complete, "stopping before the budget means EOS");
        assert!(!mono_audio(vec![0.0; 8], 750, 750).complete);
    }
}
