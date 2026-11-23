//! GPT-SoVITS V5 text-to-speech family.

pub mod config;
pub mod load;
mod model;
mod model_runner;
mod ops;
mod weights;

pub use model_runner::GptSovitsTts;
