//! Model-neutral text-to-speech API.

use apxinf_core::Result;

/// One complete text-to-speech request. Families document the voices,
/// languages and instructions they support and reject others before any
/// device work. The seed fully determines the family's sampling stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpeechRequest<'a> {
    pub text: &'a str,
    pub voice: &'a str,
    pub language: &'a str,
    /// Style instruction; empty means none.
    pub instruction: &'a str,
    pub seed: u64,
}

/// Synthesized audio owned by the caller: interleaved samples in `[-1, 1]`,
/// `channels` per frame. Later requests never overwrite it.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratedAudio {
    pub sample_rate: u32,
    pub channels: usize,
    pub samples: Vec<f32>,
    /// False when generation stopped at the family's length limit instead of
    /// its end-of-speech token.
    pub complete: bool,
}

/// Text-to-speech runtime. Implementations own their weights and preparation;
/// request state (prompt embeddings, caches, random keys) lives for one request.
pub trait SpeechRuntime {
    fn synthesize(&mut self, request: &SpeechRequest<'_>) -> Result<GeneratedAudio>;
}
