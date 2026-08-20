//! Qwen3-VL-2B-Instruct model implementation.
//!
//! Text stack + vision tower today; multimodal wiring in Phase 5. All CUDA
//! and bf16-first — see `doc/20260619-qwen3vl/plan.md`.

pub mod config;
pub mod weights;
pub mod vision_weights;
pub mod vision;
pub mod general;
#[cfg(feature = "cuda")]
pub mod decode_graph;

pub use config::Qwen3VLConfig;
pub use weights::Qwen3VLTextWeights;
pub use vision_weights::Qwen3VLVisionWeights;
pub use vision::VisionOutput;
pub use general::GeneralQwen3VL;
#[cfg(feature = "cuda")]
pub use decode_graph::{Qwen3VLDecodeGraph, Qwen3VLDecodeGraphConfig,
                       Qwen3VLDecodeGraphWeights, Qwen3VLDecodeLayerWeights};
