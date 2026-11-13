//! Model-neutral TTS synthesis API.
//!
//! One contract covers every TTS family; each loads as
//! [`crate::LoadedModel::Tts`] and returns [`GeneratedAudio`]. Families
//! differ in who owns the text front-end, so [`TtsRequest`] carries the
//! shared core (text, seed), optional sampling knobs ([`TtsSampling`]) and
//! one typed per-family extension ([`TtsExtension`]) that hands over
//! whatever else the family needs:
//!
//! - A runtime that owns its text front-end (tokenization inside the native
//!   code) takes the shared core alone; see [`TtsExtension::None`].
//! - A conditioned runtime leaves text normalization, G2P and BERT
//!   tokenization to the policy layer and receives the front-end IDs and
//!   reference-audio features through its extension; see
//!   [`TtsExtension::GptSovits`].
//!
//! Knobs left as `None` fall back to the family's own defaults. Capabilities
//! beyond `synthesize` are optional and fail explicitly by default instead
//! of implying they are active: fixed output capabilities
//! ([`TtsTrait::contract`]), conditioning reuse ([`TtsCacheConfig`]) and
//! fixed-profile preparation ([`TtsSpec`]). The seed fully determines the
//! family's sampling stream.

use apxinf_core::{Error, Result};

/// Fixed capabilities of a loaded TTS runtime. Frame rates describe the
/// family's acoustic and semantic stages; a family without such a stage
/// reports zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TtsContract {
    pub sample_rate: u32,
    pub mel_frames_per_second: usize,
    pub semantic_frames_per_second: usize,
}

/// Optional request knobs shared across TTS families. A `None` knob means
/// the family applies its own default.
#[derive(Clone, Copy, Debug, Default)]
pub struct TtsSampling<'a> {
    /// Speaker identity for families with a voice catalog.
    pub voice: Option<&'a str>,
    /// Output language for families that need it spelled out.
    pub language: Option<&'a str>,
    /// Style instruction; `None` or empty means none.
    pub instruction: Option<&'a str>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub temperature: Option<f32>,
    pub repetition_penalty: Option<f32>,
    /// Generation length limit; reaching it yields
    /// `GeneratedAudio::complete == false`.
    pub max_seconds: Option<f32>,
}

/// One complete TTS request: the shared core, optional sampling knobs and
/// one family extension.
///
/// A family documents which extension it accepts and rejects others before
/// any device work. A conditioned family ignores the shared text — the
/// policy layer already turned the text into the extension's IDs.
#[derive(Clone, Copy, Debug)]
pub struct TtsRequest<'a> {
    /// Target text for families that own their text front-end; unused by
    /// conditioned families.
    pub text: &'a str,
    /// Fully determines the family's sampling stream.
    pub seed: u64,
    /// Optional knobs; `None` fields fall back to family defaults.
    pub sampling: TtsSampling<'a>,
    /// Family-specific typed extension; see [`TtsExtension`].
    pub extension: TtsExtension<'a>,
}

impl Default for TtsRequest<'_> {
    fn default() -> Self {
        Self {
            text: "",
            seed: 0,
            sampling: TtsSampling::default(),
            extension: TtsExtension::None,
        }
    }
}

/// Typed, per-family extension of a [`TtsRequest`]. One variant per family;
/// a runtime rejects extensions it cannot serve before any device work.
#[derive(Clone, Copy, Debug)]
pub enum TtsExtension<'a> {
    /// No extension: the runtime owns its text front-end and takes the
    /// shared core alone.
    None,
    /// GPT-SoVITS V5: policy-layer front-end IDs and reference features;
    /// see [`GptSovitsInputs`].
    GptSovits(GptSovitsInputs<'a>),
}

/// Policy-layer front-end output and family-only knobs for GPT-SoVITS V5.
///
/// Text normalization, G2P and BERT tokenization are Python front-end code;
/// everything neural stays native. Sampling knobs shared across TTS families
/// live in [`TtsSampling`]; only the fields unique to this family remain
/// here. Field layout follows the family's canonical inputs.
#[derive(Clone, Copy, Debug)]
pub struct GptSovitsInputs<'a> {
    /// Target text phone IDs.
    pub phone_ids: &'a [u32],
    /// Target BERT token IDs, including `[CLS]` and `[SEP]`.
    pub bert_token_ids: &'a [u32],
    /// Number of phone positions each stripped BERT token expands to.
    pub word2ph: &'a [u32],
    /// Reference text phone IDs.
    pub reference_phone_ids: &'a [u32],
    /// Reference BERT token IDs, including `[CLS]` and `[SEP]`.
    pub reference_bert_token_ids: &'a [u32],
    /// Expansion of the reference BERT tokens to phone positions.
    pub reference_word2ph: &'a [u32],
    /// 16 kHz mono reference waveform.
    pub reference_audio_16k: &'a [f32],
    /// Normalized reference mel `[frames, 100]`.
    pub reference_mel: &'a [f32],
    /// Unnormalized reference spectrogram `[frames, 704]`.
    pub reference_spec: &'a [f32],
    pub sample_steps: usize,
    pub cfg_rate: f32,
    /// Playback-speed factor. `1.0` is the model's native rate. The native
    /// runtime applies this to the SoVITS text-encoder latent, matching the
    /// official `TextEncoder.forward(..., speed)` path; it is not a waveform
    /// resample.
    pub speed_factor: f32,
}

/// Synthesized audio owned by the caller: interleaved samples in `[-1, 1]`,
/// `channels` per frame. Later requests never overwrite it. Every sample is
/// valid: the frame count is `samples.len() / channels`.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratedAudio {
    pub sample_rate: u32,
    pub channels: usize,
    pub samples: Vec<f32>,
    /// False when generation stopped at the family's length limit instead of
    /// its end-of-speech token.
    pub complete: bool,
}

/// Fixed request profile for explicit preparation.
///
/// Every field which changes allocation, kernel dispatch, or graph topology is
/// part of the profile. Runtime values such as token IDs, audio samples, seed,
/// and sampling temperature may change between replays. Field layout follows
/// the first prepared family, GPT-SoVITS V5; families with different profile
/// needs extend the spec rather than bypassing preparation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TtsSpec {
    pub phone_count: usize,
    pub bert_token_count: usize,
    pub reference_phone_count: usize,
    pub reference_bert_token_count: usize,
    pub reference_audio_samples: usize,
    pub reference_frames: usize,
    pub reference_spec_frames: usize,
    /// Number of 100 Hz target acoustic frames admitted by the prepared graph.
    /// GPT-SoVITS V5 produces four acoustic frames per semantic token.
    pub target_frames: usize,
    pub sample_steps: usize,
}

/// A prepared, fixed-shape TTS plan.
pub trait PreparedTts {
    fn spec(&self) -> &TtsSpec;
    fn run(&self, request: &TtsRequest<'_>) -> Result<GeneratedAudio>;
}

/// Instance-local conditioning cache policy. Generation tokens and PCM are never cached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TtsCacheConfig {
    pub prompt: bool,
    pub text: bool,
    pub max_text_entries: usize,
}

impl Default for TtsCacheConfig {
    fn default() -> Self {
        Self {
            prompt: true,
            text: true,
            max_text_entries: 32,
        }
    }
}

/// Observable conditioning-cache state; byte counts include retained tensors,
/// not weights or prepared graph workspaces. `supported == false` means the
/// runtime has no cache and every other field is zero.
#[derive(Clone, Copy, Debug, Default)]
pub struct TtsCacheInfo {
    pub supported: bool,
    pub prompt_enabled: bool,
    pub text_enabled: bool,
    pub prompt_entries: usize,
    pub text_entries: usize,
    pub max_text_entries: usize,
    pub prompt_hits: u64,
    pub prompt_misses: u64,
    pub text_hits: u64,
    pub text_misses: u64,
    pub host_bytes: usize,
    pub device_bytes: usize,
    pub text_device_budget_bytes: usize,
}

/// One TTS contract for every TTS family, owned by
/// [`crate::LoadedModel::Tts`].
///
/// The implementation owns its weights, request state (prompt embeddings,
/// caches, random keys), and prepared resources. Capabilities a family does
/// not have fail explicitly by default rather than implying they are active.
pub trait TtsTrait {
    /// Fixed output capabilities of this runtime.
    fn contract(&self) -> Result<TtsContract> {
        Err(contract_unsupported())
    }

    fn synthesize(&mut self, request: &TtsRequest<'_>) -> Result<GeneratedAudio>;

    /// Configure instance-local conditioning reuse.
    fn configure_cache(&self, _config: TtsCacheConfig) -> Result<()> {
        Err(cache_unsupported())
    }

    /// Drop cached conditioning, optionally resetting hit/miss counters.
    fn clear_cache(&self, _reset_stats: bool) -> Result<()> {
        Err(cache_unsupported())
    }

    fn cache_info(&self) -> TtsCacheInfo {
        TtsCacheInfo::default()
    }

    /// Prepare a fixed profile.
    fn prepare(&self, _spec: &TtsSpec) -> Result<Box<dyn PreparedTts>> {
        Err(Error::Other(
            "explicit TTS preparation is not supported by this runtime".into(),
        ))
    }
}

fn contract_unsupported() -> Error {
    Error::Other("output capabilities are not reported by this TTS runtime".into())
}

fn cache_unsupported() -> Error {
    Error::Other("conditioning cache is not supported by this TTS runtime".into())
}

/// Extract the GPT-SoVITS extension, rejecting everything else before any
/// device work.
pub fn gpt_sovits_inputs<'a>(request: &'a TtsRequest<'a>) -> Result<&'a GptSovitsInputs<'a>> {
    match &request.extension {
        TtsExtension::GptSovits(inputs) => Ok(inputs),
        TtsExtension::None => Err(Error::Other(
            "this TTS runtime is conditioned and requires GptSovits request inputs".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Bare;

    impl TtsTrait for Bare {
        fn synthesize(&mut self, _request: &TtsRequest<'_>) -> Result<GeneratedAudio> {
            Err(Error::Other("unused".into()))
        }
    }

    #[test]
    fn optional_capabilities_fail_explicitly_by_default() {
        let runtime = Bare;
        assert!(runtime.contract().is_err());
        assert!(runtime.configure_cache(TtsCacheConfig::default()).is_err());
        assert!(runtime.clear_cache(true).is_err());
        assert!(!runtime.cache_info().supported);
        let spec = TtsSpec {
            phone_count: 1,
            bert_token_count: 3,
            reference_phone_count: 1,
            reference_bert_token_count: 3,
            reference_audio_samples: 400,
            reference_frames: 1,
            reference_spec_frames: 1,
            target_frames: 4,
            sample_steps: 1,
        };
        assert!(runtime.prepare(&spec).is_err());
        let request = TtsRequest {
            text: "",
            seed: 0,
            sampling: TtsSampling::default(),
            extension: TtsExtension::None,
        };
        assert!(gpt_sovits_inputs(&request).is_err());
    }
}
