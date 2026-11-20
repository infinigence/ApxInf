//! Common model interface for automatic speech recognition.
//!
//! The contract is deliberately narrower than a general ASR framework: it
//! carries the canonical model inputs a family needs and returns generated
//! token ids plus stop metadata. Text decoding (tokenizer inverse) belongs to
//! the policy layer; the native runtime owns every neural computation from
//! waveform upload to token ids.

use apxinf_core::Result;

/// Fixed capabilities of a loaded ASR runtime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AsrContract {
    /// Sample rate the runtime's canonical waveform input uses.
    pub sample_rate: u32,
    /// Mel bins the acoustic frontend produces (checkpoint-driven).
    pub num_mel_bins: usize,
    /// Maximum waveform samples accepted by one request.
    pub max_audio_samples: usize,
    /// Maximum decoder positions the model can address.
    pub max_decoder_positions: usize,
    /// Whether the checkpoint is multilingual (accepts a language).
    pub multilingual: bool,
    /// Language codes accepted by [`AsrRequest::language`], when known.
    pub languages: Vec<String>,
}

/// One transcription request in the family's canonical input representation.
///
/// The waveform must already be the runtime's canonical representation
/// (16 kHz mono F32 in `[-1, 1]` for the Whisper family); resampling and
/// channel downmix belong to the policy layer. Shorter inputs than the
/// contract's window are zero-padded by the runtime; longer inputs are
/// rejected explicitly rather than silently truncated.
#[derive(Clone, Copy, Debug)]
pub struct AsrRequest<'a> {
    /// Canonical mono waveform.
    pub waveform: &'a [f32],
    /// Explicit language (e.g. `"zh"`). `None` performs language auto-detection
    /// when the family supports it; families without detection reject it.
    pub language: Option<&'a str>,
    /// `"transcribe"` or `"translate"`; `None` defaults to transcribe.
    pub task: Option<&'a str>,
    /// Complete decoder prefix token ids, including the decoder start token,
    /// language and task tokens as applicable. The policy layer constructs it.
    pub prompt_token_ids: &'a [u32],
    /// Upper bound on generated tokens (the prefix is not counted).
    pub max_new_tokens: usize,
}

/// Why generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AsrStopReason {
    /// The model produced its end-of-sequence token.
    Eos,
    /// `max_new_tokens` generated tokens were produced.
    MaxTokens,
    /// The model's decoder position limit was reached.
    PositionCap,
}

/// Transcription result: generated token ids plus stop metadata.
///
/// `token_ids` excludes the request's prompt prefix and includes the
/// end-of-sequence token when [`AsrStopReason::Eos`] stopped generation.
/// Text decoding from these ids belongs to the policy layer.
#[derive(Clone, Debug)]
pub struct AsrTranscription {
    pub token_ids: Vec<u32>,
    pub stopped_reason: AsrStopReason,
    /// Number of valid generated tokens (== `token_ids.len()`).
    pub valid_tokens: usize,
}

/// Fixed request profile for explicit preparation.
///
/// Every field which changes allocation, kernel dispatch, or graph topology is
/// part of the profile. Runtime values such as waveform samples and prompt
/// token ids may change between replays within the prepared capacities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AsrSpec {
    /// Waveform capacity; the Whisper family admits exactly its 30 s window.
    pub audio_samples: usize,
    /// Prompt prefix length the prepared path admits.
    pub prompt_tokens: usize,
    /// Maximum generated tokens per replay (fixes decode loop trip counts).
    pub max_new_tokens: usize,
}

/// A prepared, fixed-shape transcription plan.
pub trait PreparedAsr {
    fn spec(&self) -> &AsrSpec;
    fn run(&self, request: &AsrRequest<'_>) -> Result<AsrTranscription>;
}

/// Common interface for ASR implementations, owned by `LoadedModel::Asr`.
///
/// Like `LlmTrait` and `TtsTrait`, this contract exposes model capabilities
/// and inference. The implementation owns request state and prepared
/// resources; tokenizer loading and text decoding remain in the policy layer.
pub trait AsrTrait {
    fn contract(&self) -> AsrContract;

    /// Eager transcription. Implementations keep all model computation on the
    /// declared device after the waveform upload.
    fn transcribe(&self, request: &AsrRequest<'_>) -> Result<AsrTranscription>;

    /// Prepare a fixed profile. Implementations that have not adopted explicit
    /// preparation fail clearly instead of pretending readiness.
    fn prepare(&self, _spec: &AsrSpec) -> Result<Box<dyn PreparedAsr>> {
        Err(apxinf_core::Error::Other(
            "explicit ASR preparation is not supported by this runtime".into(),
        ))
    }
}
