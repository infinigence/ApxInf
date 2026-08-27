//! Qwen3-VL-2B-Instruct model implementation.
//!
//! Text stack + vision tower today; multimodal wiring in Phase 5. All CUDA
//! and bf16-first.

pub mod config;
#[cfg(feature = "cuda")]
pub mod decode_graph;
pub mod general;
pub mod vision;
pub mod vision_weights;
pub mod weights;

pub use config::Qwen3VLConfig;
#[cfg(feature = "cuda")]
pub use decode_graph::{
    Qwen3VLDecodeGraph, Qwen3VLDecodeGraphConfig, Qwen3VLDecodeGraphWeights,
    Qwen3VLDecodeLayerWeights,
};
pub use general::GeneralQwen3VL;
pub use vision::VisionOutput;
pub use vision_weights::Qwen3VLVisionWeights;
pub use weights::Qwen3VLTextWeights;
