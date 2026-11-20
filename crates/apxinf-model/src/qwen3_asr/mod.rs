//! qwen3-asr-1.7B native runtime: log-mel frontend, audio tower, projector,
//! and Qwen3 text decoder, all BF16 on apxinf-cuda-new.

pub mod config;
pub(crate) mod decode_graph;
pub(crate) mod decoder;
pub(crate) mod encoder;
pub(crate) mod frontend;
pub(crate) mod weights;
pub mod load;
mod model;
#[cfg(test)]
mod tests;

pub use model::{Qwen3AsrModel, SUPPORTED_LANGUAGES};
