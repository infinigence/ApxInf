//! GR00T-owned implementation of the truncated Cosmos/Qwen3-VL backbone.
//!
//! The released GR00T N1.7 checkpoint stores the selected backbone weights
//! under `backbone.model.*`.  This module intentionally owns the architecture
//! code needed to execute those weights: model-family code must not import the
//! private implementation from another model-family directory.

pub mod config;
pub mod vision;
pub mod vision_weights;
pub mod weights;

pub use config::Qwen3VLConfig;
pub use vision_weights::Qwen3VLVisionWeights;
#[cfg(feature = "cuda")]
pub(crate) use weights::transfer_weights as transfer_text_weights;
pub use weights::Qwen3VLTextWeights;
