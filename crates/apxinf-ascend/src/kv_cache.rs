//! Ascend NPU KV cache for transformer attention.
//!
//! Buffer layout: `[n_kv_heads, max_seq_len, head_dim]` per layer.
//! Matches APXInf CUDA KV cache layout for consistency.

use std::any::Any;

use apxinf_core::{Error, KvCache, Result};

use crate::buffer::AscendBuffer;
use crate::ffi::acl;

/// KV cache stored in Ascend device buffers.
///
/// Each layer has its own K and V buffer of shape
/// `[n_kv_heads, max_seq_len, head_dim]`.
pub struct AscendKVCache {
    k_buffers: Vec<AscendBuffer>,
    v_buffers: Vec<AscendBuffer>,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    seq_len: usize,
}

impl AscendKVCache {
    /// Create a new KV cache with zeroed device buffers.
    pub fn new(
        device_id: usize,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        let layer_bytes = n_kv_heads * max_seq_len * head_dim * std::mem::size_of::<f32>();

        let mut k_buffers = Vec::with_capacity(n_layers);
        let mut v_buffers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k_buffers.push(
                AscendBuffer::malloc(layer_bytes, device_id)
                    .map_err(|e| Error::Other(format!("KV cache k alloc: {e}")))?,
            );
            v_buffers.push(
                AscendBuffer::malloc(layer_bytes, device_id)
                    .map_err(|e| Error::Other(format!("KV cache v alloc: {e}")))?,
            );
        }

        Ok(Self {
            k_buffers,
            v_buffers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            seq_len: 0,
        })
    }

    /// Get the K cache device pointer for a layer.
    pub fn k_ptr(&self, layer_idx: usize) -> usize {
        self.k_buffers[layer_idx].raw_ptr() as usize
    }

    /// Get the V cache device pointer for a layer.
    pub fn v_ptr(&self, layer_idx: usize) -> usize {
        self.v_buffers[layer_idx].raw_ptr() as usize
    }

    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Copy new K/V data into the cache at the current seq_len position.
    ///
    /// k_new/v_new: `[append_len, n_kv_heads, head_dim]` on device.
    /// The cache layout is `[n_kv_heads, max_seq_len, head_dim]`, so we
    /// copy row-by-row with D2D memcpy (stride conversion).
    pub fn append_d2d(
        &mut self,
        layer_idx: usize,
        k_new_ptr: usize,
        v_new_ptr: usize,
        append_len: usize,
    ) -> Result<()> {
        let elem_bytes = std::mem::size_of::<f32>();
        let head_dim_bytes = self.head_dim * elem_bytes;
        let src_row_bytes = self.n_kv_heads * head_dim_bytes;
        // Cache stride: [n_kv_heads, max_seq_len, head_dim]
        let cache_row_stride = self.max_seq_len * head_dim_bytes;

        for s in 0..append_len {
            let dst_offset = self.seq_len + s;
            for h in 0..self.n_kv_heads {
                // Source: k_new[s, h, :] → offset = s * src_row_bytes + h * head_dim_bytes
                let src_offset = s * src_row_bytes + h * head_dim_bytes;
                // Dest: k_cache[h, dst_offset, :] → offset = h * cache_row_stride + dst_offset * head_dim_bytes
                let dst_offset_bytes = h * cache_row_stride + dst_offset * head_dim_bytes;

                let ret = unsafe {
                    acl::aclrtMemcpy(
                        (self.k_buffers[layer_idx].raw_ptr() as usize + dst_offset_bytes)
                            as *mut std::ffi::c_void,
                        head_dim_bytes,
                        (k_new_ptr + src_offset) as *const std::ffi::c_void,
                        head_dim_bytes,
                        acl::aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE,
                    )
                };
                acl::check(ret, "kv_append k D2D").map_err(|e| Error::Other(e))?;

                let ret = unsafe {
                    acl::aclrtMemcpy(
                        (self.v_buffers[layer_idx].raw_ptr() as usize + dst_offset_bytes)
                            as *mut std::ffi::c_void,
                        head_dim_bytes,
                        (v_new_ptr + src_offset) as *const std::ffi::c_void,
                        head_dim_bytes,
                        acl::aclrtMemcpyKind::ACL_MEMCPY_DEVICE_TO_DEVICE,
                    )
                };
                acl::check(ret, "kv_append v D2D").map_err(|e| Error::Other(e))?;
            }
        }

        Ok(())
    }
}

impl KvCache for AscendKVCache {
    fn append(
        &mut self,
        _layer_idx: usize,
        _k: &apxinf_core::Tensor,
        _v: &apxinf_core::Tensor,
        _append_len: usize,
    ) -> Result<()> {
        // The trait-level append doesn't have access to device pointers.
        // Use AscendBackend::kv_append which calls append_d2d directly.
        Err(Error::Other(
            "use AscendBackend::kv_append instead of KvCache::append".into(),
        ))
    }

    fn advance(&mut self, n: usize) {
        self.seq_len += n;
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn clear(&mut self) -> Result<()> {
        // Zero all buffers without freeing (preserves addresses for graph capture)
        let total_bytes =
            self.n_kv_heads * self.max_seq_len * self.head_dim * std::mem::size_of::<f32>();
        for buf in &self.k_buffers {
            let ret = unsafe {
                acl::aclrtMemset(
                    buf.raw_ptr(),
                    total_bytes,
                    0,
                    total_bytes,
                )
            };
            acl::check(ret, "kv_cache clear k memset").map_err(|e| Error::Other(e))?;
        }
        for buf in &self.v_buffers {
            let ret = unsafe {
                acl::aclrtMemset(
                    buf.raw_ptr(),
                    total_bytes,
                    0,
                    total_bytes,
                )
            };
            acl::check(ret, "kv_cache clear v memset").map_err(|e| Error::Other(e))?;
        }
        self.seq_len = 0;
        Ok(())
    }

    fn n_layers(&self) -> usize {
        self.k_buffers.len()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
