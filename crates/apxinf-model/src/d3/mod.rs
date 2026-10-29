//! Decision 3.0 model family.
//!
//! The maintained implementation is intentionally independent of the
//! Qwen-Drive and Qwen3.8 model-family modules. They are useful architectural
//! references, but d3 owns its checkpoint contract, forward semantics and
//! execution lifecycle.

mod config;
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

pub use crate::decision::{DecisionInput, DecisionOutput, DecisionRuntime, DecisionScore};
pub use config::{D3Config, D3DecisionConfig, D3PackageConfig, D3TextConfig, D3VisionConfig};
pub use weights::{D3CheckpointWeights, D3ReadoutWeights};

/// Safe defaults for the d3 text CUDA Graph profile cache. The full public
/// Decision Index contains many distinct prompt lengths; leaving this cache
/// unbounded can exhaust a 24 GiB GPU before the run completes.
pub const D3_DEFAULT_TEXT_PROFILE_LIMIT: usize = 12;
pub const D3_DEFAULT_TEXT_PROFILE_TOKEN_BUDGET: usize = 30_000;
