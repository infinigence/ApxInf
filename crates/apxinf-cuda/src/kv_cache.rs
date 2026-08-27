//! CUDA GPU-backed KV cache for transformer attention.

use std::sync::Mutex;

use apxinf_core::{DType, Device, Error, KvCache};

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
    device_id: usize,
    /// The first appended tensor fixes the interpretation of a dynamically
    /// typed cache. BF16-specialized caches set this at construction time.
    dtype: Mutex<Option<DType>>,
    fixed_dtype: Option<DType>,
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
        Self::new_with_dtype(device_id, n_layers, n_kv_heads, head_dim, max_seq_len, None)
    }

    /// Allocate a BF16 cache. Qwen3.5 must use this constructor: its key and
    /// value activations remain BF16 and no F32 cache replica is permitted.
    pub fn new_bf16(
        device_id: usize,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<Self, Error> {
        Self::new_with_dtype(
            device_id,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            Some(DType::BF16),
        )
    }

    fn new_with_dtype(
        device_id: usize,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        fixed_dtype: Option<DType>,
    ) -> Result<Self, Error> {
        if n_layers == 0 || n_kv_heads == 0 || head_dim == 0 || max_seq_len == 0 {
            return Err(Error::Other(
                "CUDA KV cache dimensions must be non-zero".into(),
            ));
        }
        let element_bytes = fixed_dtype.unwrap_or(DType::F32).size_in_bytes();
        let layer_bytes = n_kv_heads
            .checked_mul(max_seq_len)
            .and_then(|value| value.checked_mul(head_dim))
            .and_then(|value| value.checked_mul(element_bytes))
            .ok_or_else(|| Error::Other("CUDA KV cache size overflow".into()))?;

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
            device_id,
            dtype: Mutex::new(fixed_dtype),
            fixed_dtype,
        })
    }

    /// Overwrite this cache's pages with `source`'s contents (stream-ordered
    /// copies; caller synchronizes once per session). Requires identical
    /// geometry; the recycled-session fast path that avoids reallocating
    /// every layer's K/V pages on a prefix-cache fork.
    pub fn copy_from(&mut self, source: &Self, ctx: &crate::CudaContext) -> Result<(), Error> {
        if self.k_buffers.len() != source.k_buffers.len()
            || self.n_kv_heads != source.n_kv_heads
            || self.head_dim != source.head_dim
            || self.max_seq_len != source.max_seq_len
            || self.device_id != source.device_id
        {
            return Err(Error::Other(
                "KV cache refill requires identical geometry".into(),
            ));
        }
        let stream = ctx.stream();
        for (destination, origin) in self.k_buffers.iter().zip(&source.k_buffers) {
            destination
                .copy_from_device_async(origin, stream)
                .map_err(Error::Cuda)?;
        }
        for (destination, origin) in self.v_buffers.iter().zip(&source.v_buffers) {
            destination
                .copy_from_device_async(origin, stream)
                .map_err(Error::Cuda)?;
        }
        self.seq_len = source.seq_len;
        let source_dtype = *source
            .dtype
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *self
            .dtype
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = source_dtype;
        Ok(())
    }

    /// Deep-clone this cache (stream-ordered device-to-device copies of
    /// every layer's K and V pages plus the logical length). Used by the
    /// prefix cache to fork a prompt's KV state; the pages are bit-identical.
    /// Asynchronous on the context stream — the caller synchronizes once
    /// after cloning the whole session.
    pub fn deep_clone(&self, ctx: &crate::CudaContext) -> Result<Self, Error> {
        let stream = ctx.stream();
        let clone_buffers = |buffers: &Vec<CudaBuffer>| -> Result<Vec<CudaBuffer>, Error> {
            buffers
                .iter()
                .map(|buffer| buffer.try_clone_async(stream).map_err(Error::Cuda))
                .collect()
        };
        let dtype_value = *self
            .dtype
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(Self {
            k_buffers: clone_buffers(&self.k_buffers)?,
            v_buffers: clone_buffers(&self.v_buffers)?,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            max_seq_len: self.max_seq_len,
            seq_len: self.seq_len,
            device_id: self.device_id,
            dtype: Mutex::new(dtype_value),
            fixed_dtype: self.fixed_dtype,
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
        if layer_idx >= self.k_buffers.len() {
            return Err(Error::Other(format!(
                "KV layer index {layer_idx} exceeds {} layers",
                self.k_buffers.len()
            )));
        }
        let expected = Device::Cuda(self.device_id);
        let expected_shape = [append_len, self.n_kv_heads, self.head_dim];
        if append_len == 0
            || k_new.device() != expected
            || v_new.device() != expected
            || k_new.dtype() != v_new.dtype()
            || k_new.shape().dims() != expected_shape
            || v_new.shape().dims() != expected_shape
        {
            return Err(Error::Other(format!(
                "KV append expects CUDA{} {} {:?}, got K {} {:?} on {} and V {} {:?} on {}",
                self.device_id,
                expected_shape
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join("x"),
                DType::BF16,
                k_new.dtype(),
                k_new.shape().dims(),
                k_new.device(),
                v_new.dtype(),
                v_new.shape().dims(),
                v_new.device()
            )));
        }
        if !matches!(k_new.dtype(), DType::BF16 | DType::F32) {
            return Err(Error::Other(format!(
                "KV append does not support {}",
                k_new.dtype()
            )));
        }
        let mut dtype = self
            .dtype
            .lock()
            .map_err(|_| Error::Other("CUDA KV cache dtype lock is poisoned".into()))?;
        if let Some(dtype) = *dtype {
            if dtype != k_new.dtype() {
                return Err(Error::Other(format!(
                    "KV cache dtype is {dtype}, cannot append {}",
                    k_new.dtype()
                )));
            }
        }
        if self.seq_len.checked_add(append_len).is_none()
            || self.seq_len + append_len > self.max_seq_len
        {
            return Err(Error::Other(format!(
                "KV append would exceed max sequence length {}",
                self.max_seq_len
            )));
        }
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
        // Set this only after both launches have accepted the same contract.
        *dtype = Some(k_new.dtype());
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

    pub fn dtype(&self) -> Result<Option<DType>, Error> {
        self.dtype
            .lock()
            .map(|dtype| *dtype)
            .map_err(|_| Error::Other("CUDA KV cache dtype lock is poisoned".into()))
    }

    /// Dtype represented by the allocated cache buffers. An untyped cache is
    /// allocated as F32 for compatibility with the generic backend contract.
    pub fn storage_dtype(&self) -> DType {
        self.fixed_dtype.unwrap_or(DType::F32)
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
        *self
            .dtype
            .get_mut()
            .map_err(|_| Error::Other("CUDA KV cache dtype lock is poisoned".into()))? =
            self.fixed_dtype;
        let element_bytes = self.fixed_dtype.unwrap_or(DType::F32).size_in_bytes();
        let layer_bytes = self.n_kv_heads * self.max_seq_len * self.head_dim * element_bytes;
        for buf in &mut self.k_buffers {
            *buf = CudaBuffer::alloc_zeros(layer_bytes, self.device_id).map_err(Error::Cuda)?;
        }
        for buf in &mut self.v_buffers {
            *buf = CudaBuffer::alloc_zeros(layer_bytes, self.device_id).map_err(Error::Cuda)?;
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
