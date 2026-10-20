//! Decision 3.0 model family.
//!
//! The maintained implementation is intentionally independent of the
//! Qwen-Drive and Qwen3.8 model-family modules. They are useful architectural
//! references, but d3 owns its checkpoint contract, forward semantics and
//! execution lifecycle.

mod config;
mod runtime;
mod weights;

#[cfg(feature = "cuda")]
mod backend;
#[cfg(feature = "cuda")]
pub(crate) mod load;
#[cfg(feature = "cuda")]
mod model;
#[cfg(feature = "cuda")]
mod runner;
#[cfg(feature = "cuda")]
mod vision;

pub use config::{D3Config, D3DecisionConfig, D3PackageConfig, D3TextConfig, D3VisionConfig};
pub use runtime::{
    DecisionInput, DecisionOutput, DecisionRuntime, DecisionScore, D3_DEFAULT_TEXT_PROFILE_LIMIT,
    D3_DEFAULT_TEXT_PROFILE_TOKEN_BUDGET,
};
pub use weights::{D3CheckpointWeights, D3ReadoutWeights};
