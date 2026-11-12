//! Common model interface for text-to-speech synthesis.
//!
//! The contract is deliberately narrower than a general ASR/TTS framework: it
//! carries the canonical model inputs a family needs and returns mono PCM. Text
//! normalization, G2P and BERT tokenization belong to the policy/input layer;
//! the native runtime owns every neural computation.

use apxinf_core::Result;

/// Fixed capabilities of a loaded speech runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TtsContract {
    pub sample_rate: u32,
    pub mel_frames_per_second: usize,
    pub semantic_frames_per_second: usize,
}

/// One synthesis request in the family's canonical input representation.
#[derive(Clone, Copy, Debug)]
pub struct TtsRequest<'a> {
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
    pub seed: u64,
    pub top_k: usize,
    pub top_p: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub sample_steps: usize,
    pub cfg_rate: f32,
    pub max_seconds: f32,
    /// Playback-speed factor. `1.0` is the model's native rate. The native
    /// runtime applies this to the SoVITS text-encoder latent, matching the
    /// official `TextEncoder.forward(..., speed)` path; it is not a waveform
    /// resample.
    pub speed_factor: f32,
}

/// Mono PCM plus its output sample rate and valid length.
#[derive(Clone, Debug)]
pub struct PcmAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub valid_length: usize,
}

/// Fixed request profile for explicit preparation.
///
/// Every field which changes allocation, kernel dispatch, or graph topology is
/// part of the profile. Runtime values such as token IDs, audio samples, seed,
/// and sampling temperature may change between replays.
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

/// A prepared, fixed-shape speech plan.
pub trait PreparedTts {
    fn spec(&self) -> &TtsSpec;
    fn run(&self, request: &TtsRequest<'_>) -> Result<PcmAudio>;
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
/// not weights or prepared graph workspaces.
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

/// Common interface for TTS implementations, owned by `LoadedModel::Tts`.
///
/// Like `LlmTrait`, this contract exposes model capabilities and inference.
/// The implementation owns request state, caches, and prepared resources;
/// text normalization and G2P remain in the policy layer.
pub trait TtsTrait {
    fn contract(&self) -> TtsContract;

    fn synthesize(&self, request: &TtsRequest<'_>) -> Result<PcmAudio>;

    /// Configure instance-local conditioning reuse. Unsupported families reject
    /// this explicitly rather than implying that a cache is active.
    fn configure_cache(&self, _config: TtsCacheConfig) -> Result<()> {
        Err(apxinf_core::Error::Other(
            "conditioning cache is not supported by this TTS implementation".into(),
        ))
    }

    fn clear_cache(&self, _reset_stats: bool) {}

    fn cache_info(&self) -> TtsCacheInfo {
        TtsCacheInfo::default()
    }

    /// Prepare a fixed profile. Implementations that have not adopted explicit
    /// preparation fail clearly instead of pretending readiness.
    fn prepare(&self, _spec: &TtsSpec) -> Result<Box<dyn PreparedTts>> {
        Err(apxinf_core::Error::Other(
            "explicit speech preparation is not supported by this runtime".into(),
        ))
    }
}
