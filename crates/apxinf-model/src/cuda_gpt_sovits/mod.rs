//! GPT-SoVITS V5 text-to-speech family.

mod bert;
mod cache;
mod cfm;
pub mod config;
mod gpt;
mod hubert;
pub mod load;
mod model;
mod ops;
mod sovits;
mod vocoder;
mod weights;

pub use model::GptSovitsModel;
