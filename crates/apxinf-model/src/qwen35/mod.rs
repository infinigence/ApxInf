//! Qwen3.5 text model type support.
//!
//! This module recognizes the `qwen3_5` Hugging Face model family used by the
//! Qwen3.8-27B target.  It parses the hybrid linear-attention/full-attention
//! config and validates compressed-tensors W4A16 weight groups.  The execution
//! kernels are intentionally not hidden behind this type: `GeneralQwen35`
//! returns an explicit unsupported-runtime error until the linear-attention and
//! packed INT4 GEMM paths are implemented.

#[cfg(feature = "cuda")]
pub mod cuda;
pub mod config;
pub mod general;
pub mod weights;
pub use config::{LayerKind, Qwen35Config, Qwen35TextConfig};
pub use general::GeneralQwen35;
pub use weights::{CompressedLinearWeight, Qwen35WeightsManifest};
