//! Llama model architecture.
//!
//! Contains: weight structs, legacy full-stack model, modern `dyn Backend`
//! general model, and the allocation-free decode workspace + graph capture.

pub mod weights;
pub mod model;
pub mod general;
#[cfg(feature = "cuda")]
pub mod decode_graph;

pub use weights::{LlamaWeights, TransformerLayer};
pub use model::{LlamaModel, KVCache};
pub use general::GeneralLlama;
#[cfg(feature = "cuda")]
pub use decode_graph::{DecodeGraph, DecodeGraphConfig, DecodeGraphWeights, DecodeLayerWeights};
