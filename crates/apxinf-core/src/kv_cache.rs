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

/// CPU KV cache using nested Vecs.
/// Layout: [n_layers, n_kv_heads, max_seq_len, head_dim]
pub struct CpuKVCache {
    k_cache: Vec<Vec<Vec<Vec<f32>>>>,
    v_cache: Vec<Vec<Vec<Vec<f32>>>>,
    seq_len: usize,
    max_seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl CpuKVCache {
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        let k_cache = (0..n_layers)
            .map(|_| {
                (0..n_kv_heads)
                    .map(|_| (0..max_seq_len).map(|_| vec![0.0f32; head_dim]).collect())
                    .collect()
            })
            .collect();

        let v_cache = (0..n_layers)
            .map(|_| {
                (0..n_kv_heads)
                    .map(|_| (0..max_seq_len).map(|_| vec![0.0f32; head_dim]).collect())
                    .collect()
            })
            .collect();

        Self { k_cache, v_cache, seq_len: 0, max_seq_len, n_kv_heads, head_dim }
    }

    /// Get K and V for a layer up to current position.
    /// Returns (K, V) where each is [n_kv_heads, seq_len, head_dim]
    pub fn get_kv(&self, layer_idx: usize) -> (&[Vec<Vec<f32>>], &[Vec<Vec<f32>>]) {
        (&self.k_cache[layer_idx], &self.v_cache[layer_idx])
    }
}

impl KvCache for CpuKVCache {
    fn append(&mut self, layer_idx: usize, k: &Tensor, v: &Tensor, append_len: usize) -> Result<()> {
        let k_data = k.as_f32()?;
        let v_data = v.as_f32()?;
        for s in 0..append_len {
            let pos = self.seq_len + s;
            for h in 0..self.n_kv_heads {
                for d in 0..self.head_dim {
                    self.k_cache[layer_idx][h][pos][d] =
                        k_data[s * self.n_kv_heads * self.head_dim + h * self.head_dim + d];
                    self.v_cache[layer_idx][h][pos][d] =
                        v_data[s * self.n_kv_heads * self.head_dim + h * self.head_dim + d];
                }
            }
        }
        Ok(())
    }

    fn advance(&mut self, n: usize) {
        self.seq_len += n;
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn clear(&mut self) -> Result<()> {
        self.seq_len = 0;
        for layer in &mut self.k_cache {
            for head in layer {
                for pos in head {
                    for v in pos { *v = 0.0; }
                }
            }
        }
        for layer in &mut self.v_cache {
            for head in layer {
                for pos in head {
                    for v in pos { *v = 0.0; }
                }
            }
        }
        Ok(())
    }

    fn n_layers(&self) -> usize {
        self.k_cache.len()
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}
