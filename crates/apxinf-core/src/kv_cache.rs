//! Backend-agnostic KV cache interface.

use crate::{Result, Tensor};

/// KV cache trait for transformer attention.
///
/// Each backend implements this with its own buffer type internally.
/// Object-safe.
pub trait KvCache {
    /// Append new K/V data for a layer.
    /// k, v: [append_len, n_kv_heads, head_dim]
    fn append(&mut self, layer_idx: usize, k: &Tensor, v: &Tensor, append_len: usize) -> Result<()>;

    /// Advance the sequence position by n tokens.
    fn advance(&mut self, n: usize);

    /// Current sequence length in the cache.
    fn seq_len(&self) -> usize;

    /// Reset the cache for a new generation.
    fn clear(&mut self) -> Result<()>;

    /// Number of layers.
    fn n_layers(&self) -> usize;

    /// Allow backends to downcast to their concrete type.
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}
