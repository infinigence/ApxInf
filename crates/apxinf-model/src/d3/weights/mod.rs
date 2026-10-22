//! Official d3 checkpoint mapping and physical BF16 layouts.

mod host;

pub use host::{D3CheckpointWeights, D3ReadoutWeights};

#[cfg(feature = "cuda")]
pub(crate) mod bf16;
