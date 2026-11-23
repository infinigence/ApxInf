//! qwen3-asr-1.7B native runtime: log-mel frontend, audio tower, projector,
//! and Qwen3 text decoder, all BF16 on apxinf-cuda-new.

pub mod config;
pub mod load;
mod model;
mod model_runner;
#[cfg(test)]
pub(crate) mod tests;
pub(crate) mod weights;

pub use model::Qwen3AsrModel;
