//! KV cache and the attention that reads it.

use std::any::Any;
use std::sync::Arc;

use apxinf_core::{DType, Error, KvCache, Result, Tensor};

use crate::ffi::{self, check};
use crate::ops::{as_i32, dtype_code};
use crate::runtime::{HipBuffer, HipContext};

/// Per-layer K and V storage, token-major: `[max_seq_len, n_kv_heads, head_dim]`.
///
/// Token-major rather than apxinf-cuda's head-major layout because the model
/// appends `[append_len, n_kv_heads, head_dim]` blocks, which then land as one
/// contiguous copy instead of needing a scatter kernel. Attention reads keys
/// with a stride of `n_kv_heads * head_dim`, which costs it nothing. The layout
/// is private to this backend; nothing outside reads it.
///
/// Allocation is deferred to the first append. `Backend::create_kv_cache`
/// cannot return an error and does not say which dtype will be stored, and the
/// first append knows both. Once allocated, the buffers live as long as the
/// cache: `clear` resets positions but keeps every address, so device graphs
/// captured against them stay valid (the property apxinf-cuda had to restore in
/// its own cache, #89).
pub struct HipKVCache {
    ctx: Arc<HipContext>,
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    seq_len: usize,
    storage: Option<Storage>,
    /// Per layer, positions `[0, written)` hold data appended since the last
    /// `clear`. `append` writes at `seq_len` without advancing it (the model
    /// advances once after every layer), so this — not `seq_len` — is how much
    /// attention may read.
    written: Vec<usize>,
}

struct Storage {
    dtype: DType,
    k: Vec<HipBuffer>,
    v: Vec<HipBuffer>,
}

impl HipKVCache {
    pub(crate) fn new(
        ctx: Arc<HipContext>,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        Self {
            ctx,
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            seq_len: 0,
            storage: None,
            written: vec![0; n_layers],
        }
    }

    fn row_elements(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    fn ensure_storage(&mut self, dtype: DType) -> Result<&Storage> {
        if self.storage.is_none() {
            dtype_code("KV cache", dtype)?;
            let bytes = self
                .max_seq_len
                .checked_mul(self.row_elements())
                .and_then(|n| n.checked_mul(dtype.size_in_bytes()))
                .ok_or_else(|| Error::Other("apxinf-hip: KV cache size overflows".into()))?;
            self.ctx.bind()?;
            let mut k = Vec::with_capacity(self.n_layers);
            let mut v = Vec::with_capacity(self.n_layers);
            for _ in 0..self.n_layers {
                k.push(self.ctx.alloc(bytes)?);
                v.push(self.ctx.alloc(bytes)?);
            }
            self.storage = Some(Storage { dtype, k, v });
        }
        let storage = self.storage.as_ref().unwrap();
        if storage.dtype != dtype {
            return Err(Error::DTypeMismatch { expected: storage.dtype, got: dtype });
        }
        Ok(storage)
    }

    /// Causal attention for `q: [q_len, n_heads, head_dim]` against the first
    /// `kv_len` cached positions of `layer`. Query row `i` sees keys
    /// `[0, min(kv_len, kv_offset + i + 1))`. Returns `[q_len, n_heads * head_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention(
        &self,
        q: &Tensor,
        layer: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        kv_offset: usize,
    ) -> Result<Tensor> {
        if n_kv_heads != self.n_kv_heads || head_dim != self.head_dim {
            return Err(Error::Other(format!(
                "apxinf-hip: attention asked for {n_kv_heads} KV heads of {head_dim}, \
                 cache holds {} of {}",
                self.n_kv_heads, self.head_dim
            )));
        }
        if n_kv_heads == 0 || !n_heads.is_multiple_of(n_kv_heads) {
            return Err(Error::Other(format!(
                "apxinf-hip: {n_heads} query heads cannot share {n_kv_heads} KV heads"
            )));
        }
        let storage = self.storage.as_ref().ok_or_else(|| {
            Error::Other("apxinf-hip: attention read a KV cache nothing was appended to".into())
        })?;
        let written = *self.written.get(layer).ok_or_else(|| {
            Error::Other(format!("apxinf-hip: layer {layer} out of range for {} layers", self.n_layers))
        })?;
        // Reading past what was appended would return stale or uninitialized
        // memory with no error; refuse instead.
        if kv_len == 0 || kv_len > written {
            return Err(Error::Other(format!(
                "apxinf-hip: attention over {kv_len} positions, layer {layer} holds {written}"
            )));
        }
        let dtype = dtype_code("attention", q.dtype())?;
        if q.dtype() != storage.dtype {
            return Err(Error::DTypeMismatch { expected: storage.dtype, got: q.dtype() });
        }
        let row = n_heads * head_dim;
        if row == 0 || !q.numel().is_multiple_of(row) {
            return Err(Error::Other(format!(
                "apxinf-hip: attention query {} is not [q_len, {n_heads}, {head_dim}]",
                q.shape()
            )));
        }
        let q_len = q.numel() / row;

        let pq = self.ctx.ptr(q)?;
        self.ctx.bind()?;
        let (out, po) = self.ctx.empty(vec![q_len, row], q.dtype())?;
        check("attention", unsafe {
            ffi::apxinf_hip_attention(
                dtype,
                pq,
                storage.k[layer].ptr(),
                storage.v[layer].ptr(),
                po,
                as_i32("attention q_len", q_len)?,
                as_i32("attention n_heads", n_heads)?,
                as_i32("attention n_kv_heads", n_kv_heads)?,
                as_i32("attention head_dim", head_dim)?,
                as_i32("attention kv_len", kv_len)?,
                as_i32("attention kv_offset", kv_offset)?,
                1.0 / (head_dim as f32).sqrt(),
                self.ctx.stream(),
            )
        })?;
        Ok(out)
    }
}

impl KvCache for HipKVCache {
    /// Write `k`, `v` (`[append_len, n_kv_heads, head_dim]`) at positions
    /// `[seq_len, seq_len + append_len)` of `layer`. Does not advance.
    ///
    /// Works through the trait alone — apxinf-cuda's cache only accepts appends
    /// through its backend — because the cache holds its own context.
    fn append(&mut self, layer: usize, k: &Tensor, v: &Tensor, append_len: usize) -> Result<()> {
        if layer >= self.n_layers {
            return Err(Error::Other(format!(
                "apxinf-hip: layer {layer} out of range for {} layers",
                self.n_layers
            )));
        }
        if k.dtype() != v.dtype() {
            return Err(Error::DTypeMismatch { expected: k.dtype(), got: v.dtype() });
        }
        let elements = append_len * self.row_elements();
        for t in [k, v] {
            if t.numel() != elements {
                return Err(Error::ShapeMismatch {
                    expected: format!("[{append_len}, {}, {}]", self.n_kv_heads, self.head_dim),
                    got: t.shape().to_string(),
                });
            }
        }
        let end = self.seq_len + append_len;
        if end > self.max_seq_len {
            return Err(Error::Other(format!(
                "apxinf-hip: KV cache holds {} positions, append would reach {end}",
                self.max_seq_len
            )));
        }
        let (pk, pv) = (self.ctx.ptr(k)?, self.ctx.ptr(v)?);
        let element = k.dtype().size_in_bytes();
        let offset = self.seq_len * self.row_elements() * element;
        let bytes = elements * element;
        let storage = self.ensure_storage(k.dtype())?;
        // SAFETY: `end <= max_seq_len`, so `offset + bytes` stays inside the
        // layer buffer allocated for `max_seq_len` positions.
        let (dk, dv) = unsafe {
            (
                storage.k[layer].ptr().cast::<u8>().add(offset).cast(),
                storage.v[layer].ptr().cast::<u8>().add(offset).cast(),
            )
        };
        self.ctx.copy_on_device(dk, pk, bytes)?;
        self.ctx.copy_on_device(dv, pv, bytes)?;
        self.written[layer] = end;
        Ok(())
    }

    fn advance(&mut self, n: usize) {
        self.seq_len += n;
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Forget every position but keep the allocations, so their addresses stay
    /// stable across requests.
    fn clear(&mut self) -> Result<()> {
        self.seq_len = 0;
        self.written.iter_mut().for_each(|w| *w = 0);
        Ok(())
    }

    fn n_layers(&self) -> usize {
        self.n_layers
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
