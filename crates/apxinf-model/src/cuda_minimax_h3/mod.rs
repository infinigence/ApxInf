//! MiniMax-H3 text-to-video-and-audio on the cuda-new backend: INT8 ConvRot DiT over a packed text, audio and video sequence, the
//! streamed Qwen3-VL text encoder and the video and audio VAE decoders.
#[cfg(feature = "cuda")]
mod backend;
#[cfg(feature = "cuda")]
mod config;
#[cfg(feature = "cuda")]
pub(crate) mod load;
#[cfg(feature = "cuda")]
mod model;
#[cfg(feature = "cuda")]
mod model_runner;
mod package;
#[cfg(all(test, feature = "cuda"))]
mod tests;
#[cfg(feature = "cuda")]
mod weights;

pub use package::{MANIFEST, SCHEMA};

/// Whether `dir` is a MiniMax-H3 package (`AutoModel` detection).
pub(crate) use package::is_package;

/// Frame rate, bounds and step limits a request is validated against (see `config.rs`).
#[cfg(feature = "cuda")]
pub use config::{MAX_FRAMES, MAX_SIDE, MAX_STEPS, MIN_FRAMES, MIN_SIDE};

#[cfg(feature = "cuda")]
pub(super) fn register_builtin() {
    crate::registry::register("cuda_minimax_h3", load::load_registered);
    crate::registry::register("minimax-h3", load::load_registered);
}
