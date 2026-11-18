//! Model-neutral text-to-video-and-audio API.

use apxinf_core::Result;

/// One complete text-to-video request. Families document the sizes, frame counts and step counts they support and reject others before
/// any device work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoRequest<'a> {
    pub prompt: &'a str,
    pub width: usize,
    pub height: usize,
    /// Frames asked for; a family may snap it onto the grid its model was trained on and reports the result in [`GeneratedVideo::frames`].
    pub frames: usize,
    pub seed: u64,
    pub steps: usize,
}

/// A generated clip owned by the caller. `rgb` is row-major RGB8 frames `[frames, height, width, 3]`; `audio` holds interleaved samples in
/// `[-1, 1]` with `audio_channels` per sample frame, empty for a silent family. Later requests never overwrite it.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratedVideo {
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub frame_rate: u32,
    pub rgb: Vec<u8>,
    pub audio_sample_rate: u32,
    pub audio_channels: usize,
    pub audio: Vec<f32>,
}

/// Text-to-video runtime. Implementations own their weights and preparation; request state (prompt embeddings, latents, noise) lives for one
/// request.
pub trait VideoRuntime {
    fn generate(&mut self, request: &VideoRequest<'_>) -> Result<GeneratedVideo>;
}
