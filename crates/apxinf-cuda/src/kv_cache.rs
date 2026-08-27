//! CUDA GPU-backed KV cache for transformer attention.

use apxinf_core::Error;
use apxinf_core::KvCache;

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::kernels;

/// KV cache stored in CUDA buffers for GPU-native attention.
///
/// Buffer layout: `[n_kv_heads, max_seq_len, head_dim]` per layer.
pub struct CudaKVCache {
    /// Per-layer K cache buffers.
    k_buffers: Vec<CudaBuffer>,
    /// Per-layer V cache buffers.
    v_buffers: Vec<CudaBuffer>,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    seq_len: usize,
}

impl CudaKVCache {
    /// Create a new CUDA KV cache with zeroed buffers.
    pub fn new(
        device_id: usize,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<Self, Error> {
        let layer_bytes = n_kv_heads * max_seq_len * head_dim * std::mem::size_of::<f32>();

        let k_buffers = (0..n_layers)
            .map(|_| CudaBuffer::alloc_zeros(layer_bytes, device_id).map_err(Error::Cuda))
            .collect::<Result<Vec<_>, _>>()?;

        let v_buffers = (0..n_layers)
            .map(|_| CudaBuffer::alloc_zeros(layer_bytes, device_id).map_err(Error::Cuda))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            k_buffers,
            v_buffers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            seq_len: 0,
        })
    }

    /// Append K/V data for multiple positions using the GPU append kernel.
    pub fn append(
        &self,
        ctx: &CudaContext,
        layer_idx: usize,
        k_new: &apxinf_core::Tensor,
        v_new: &apxinf_core::Tensor,
        append_len: usize,
    ) -> Result<(), Error> {
        kernels::cache::append(
            ctx,
            &self.k_buffers[layer_idx],
            k_new,
            self.n_kv_heads,
            self.head_dim,
            self.max_seq_len,
            self.seq_len,
            append_len,
        )?;
        kernels::cache::append(
            ctx,
            &self.v_buffers[layer_idx],
            v_new,
            self.n_kv_heads,
            self.head_dim,
            self.max_seq_len,
            self.seq_len,
            append_len,
        )?;
        Ok(())
    }

    /// Get the K cache buffer for a layer.
    pub fn k_buffer(&self, layer_idx: usize) -> &CudaBuffer {
        &self.k_buffers[layer_idx]
    }

    /// Get the V cache buffer for a layer.
    pub fn v_buffer(&self, layer_idx: usize) -> &CudaBuffer {
        &self.v_buffers[layer_idx]
    }

    /// Current sequence length (number of cached positions).
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }
}

impl KvCache for CudaKVCache {
    fn append(
        &mut self,
        layer_idx: usize,
        k: &apxinf_core::Tensor,
        v: &apxinf_core::Tensor,
        append_len: usize,
    ) -> apxinf_core::Result<()> {
        // We need a CudaContext for the kernel call. For the KvCache trait impl,
        // we skip the context and rely on the backend's sdpa methods to call
        // the non-trait append directly. This trait impl is a placeholder.
        let _ = (layer_idx, k, v, append_len);
        Err(Error::Other(
            "use CudaKVCache::append(ctx, ...) directly or via CudaBackend::sdpa_*".into(),
        ))
    }

    fn advance(&mut self, n: usize) {
        self.seq_len += n;
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn clear(&mut self) -> apxinf_core::Result<()> {
        self.seq_len = 0;
        for buffer in &self.k_buffers {
            buffer.memset(0).map_err(Error::Cuda)?;
        }
        for buffer in &self.v_buffers {
            buffer.memset(0).map_err(Error::Cuda)?;
        }
        Ok(())
    }

    fn n_layers(&self) -> usize {
        self.k_buffers.len()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
