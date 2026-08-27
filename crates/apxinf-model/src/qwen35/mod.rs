//! Qwen3.5/Qwen3.8 hybrid multimodal model contracts.
//!
//! The execution kernels are introduced incrementally. This module first
//! owns the strict Hugging Face config and compressed-tensors checkpoint
//! contract so unsupported inputs cannot fall through to the Llama runtime.

mod checkpoint;
mod config;
#[cfg(feature = "cuda")]
mod decode;
mod multimodal;
#[cfg(feature = "cuda")]
mod vision;

pub use checkpoint::Qwen35CheckpointReport;
pub use config::{
    Qwen35Config, Qwen35LayerType, Qwen35QuantizationConfig, Qwen35TextConfig, Qwen35VisionConfig,
};
#[cfg(feature = "cuda")]
pub use decode::{
    load_embedding_row, HybridUnit, HybridUnitMode, Qwen35KvCacheMode, Qwen35LmHead,
    Qwen35PrefillMode,
};
pub use multimodal::{compute_mrope_positions, Qwen35MropePositions};
#[cfg(feature = "cuda")]
pub use vision::Qwen35VisionEncoder;
