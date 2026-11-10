//! Llama model architecture.
//!
//! Contains: weight structs, legacy full-stack model, modern `dyn Backend`
//! general model, and the allocation-free decode workspace + graph capture.

#[cfg(feature = "cuda")]
pub mod decode_graph;
pub mod general;
pub mod model;
pub mod weights;

#[cfg(feature = "cuda")]
pub use decode_graph::{DecodeGraph, DecodeGraphConfig, DecodeGraphWeights, DecodeLayerWeights};
pub use general::GeneralLlama;
pub use model::{KVCache, LlamaModel};
pub use weights::{LlamaWeights, TransformerLayer};
