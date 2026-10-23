//! CUDA GPU-backed KV cache for transformer attention.

use apxinf_core::{Error, KvCache, Result, Tensor};
use std::any::Any;

use crate::{ops, CudaBuffer, CudaContext};

/// KV cache stored in CUDA buffers.
///
/// Buffer layout: `[n_kv_heads, max_seq_len, head_dim]` per layer. Storage is
/// sized at four bytes per element, matching the legacy backend contract: the
/// same stable allocation can hold either F32 or BF16 cache data and remains
/// valid across captured graph replays.
pub struct CudaKvCache {
    k_buffers: Vec<CudaBuffer>,
    v_buffers: Vec<CudaBuffer>,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    seq_len: usize,
}

impl CudaKvCache {
    pub(crate) fn new(
        device_id: usize,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        let layer_bytes = n_kv_heads
            .checked_mul(max_seq_len)
            .and_then(|value| value.checked_mul(head_dim))
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| Error::Other("KV cache allocation size overflow".into()))?;

        let k_buffers = (0..n_layers)
            .map(|_| CudaBuffer::alloc_zeros(layer_bytes, device_id).map_err(Error::Cuda))
            .collect::<Result<Vec<_>>>()?;
        let v_buffers = (0..n_layers)
            .map(|_| CudaBuffer::alloc_zeros(layer_bytes, device_id).map_err(Error::Cuda))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            k_buffers,
            v_buffers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            seq_len: 0,
        })
    }

    /// Append K/V data for multiple positions using the legacy head-major
    /// cache contract.
    pub fn append(
        &self,
        ctx: &CudaContext,
        layer_idx: usize,
        k_new: &Tensor,
        v_new: &Tensor,
        append_len: usize,
    ) -> Result<()> {
        ops::cache::append(
            ctx,
            &self.k_buffers[layer_idx],
            k_new,
            self.n_kv_heads,
            self.head_dim,
            self.max_seq_len,
            self.seq_len,
            append_len,
        )?;
        ops::cache::append(
            ctx,
            &self.v_buffers[layer_idx],
            v_new,
            self.n_kv_heads,
            self.head_dim,
            self.max_seq_len,
            self.seq_len,
            append_len,
        )
    }

    pub fn k_buffer(&self, layer_idx: usize) -> &CudaBuffer {
        &self.k_buffers[layer_idx]
    }

    pub fn v_buffer(&self, layer_idx: usize) -> &CudaBuffer {
        &self.v_buffers[layer_idx]
    }
}

impl KvCache for CudaKvCache {
    fn append(
        &mut self,
        layer_idx: usize,
        k: &Tensor,
        v: &Tensor,
        append_len: usize,
    ) -> Result<()> {
        // The portable trait has no stream/context argument. Keep the legacy
        // contract: Backend::kv_append owns the CUDA launch context.
        let _ = (layer_idx, k, v, append_len);
        Err(Error::Other(
            "use CudaKvCache::append(ctx, ...) through CudaNewBackend::kv_append".into(),
        ))
    }

    fn advance(&mut self, n: usize) {
        self.seq_len += n;
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn clear(&mut self) -> Result<()> {
        // Captured decode graphs retain these addresses. Zero in place rather
        // than replacing the allocations.
        for buffer in self.k_buffers.iter().chain(&self.v_buffers) {
            buffer.zero().map_err(Error::Cuda)?;
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

#[cfg(test)]
mod tests {
    use apxinf_core::{Shape, Tensor};
    use half::bf16;

    use super::*;

    #[test]
    fn append_preserves_legacy_head_major_layout_for_bf16_and_f32() {
        let _guard = crate::CUDA_TEST_LOCK.lock().unwrap();
        let ctx = CudaContext::new(0).unwrap();

        let bf16_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].map(bf16::from_f32);
        let bf16_host = Tensor::from_bf16(Shape::new(vec![2, 2, 2]), &bf16_values).unwrap();
        let bf16_input = crate::transfers::to_cuda(&bf16_host, 0).unwrap();
        let bf16_cache = CudaKvCache::new(0, 1, 2, 2, 3).unwrap();
        bf16_cache
            .append(&ctx, 0, &bf16_input, &bf16_input, 2)
            .unwrap();
        ctx.synchronize().unwrap();
        let mut bf16_bytes = vec![0; bf16_cache.k_buffer(0).len()];
        bf16_cache
            .k_buffer(0)
            .copy_to_host(&mut bf16_bytes)
            .unwrap();
        let actual_bf16 = bf16_bytes[..24]
            .chunks_exact(2)
            .map(|bytes| bf16::from_bits(u16::from_ne_bytes([bytes[0], bytes[1]])).to_f32())
            .collect::<Vec<_>>();
        assert_eq!(
            actual_bf16,
            vec![1.0, 2.0, 5.0, 6.0, 0.0, 0.0, 3.0, 4.0, 7.0, 8.0, 0.0, 0.0]
        );

        let f32_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let f32_host = Tensor::from_f32(Shape::new(vec![2, 2, 2]), &f32_values).unwrap();
        let f32_input = crate::transfers::to_cuda(&f32_host, 0).unwrap();
        let f32_cache = CudaKvCache::new(0, 1, 2, 2, 3).unwrap();
        f32_cache
            .append(&ctx, 0, &f32_input, &f32_input, 2)
            .unwrap();
        ctx.synchronize().unwrap();
        let mut f32_bytes = vec![0; f32_cache.k_buffer(0).len()];
        f32_cache.k_buffer(0).copy_to_host(&mut f32_bytes).unwrap();
        let actual_f32 = f32_bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect::<Vec<_>>();
        assert_eq!(
            actual_f32,
            vec![1.0, 2.0, 5.0, 6.0, 0.0, 0.0, 3.0, 4.0, 7.0, 8.0, 0.0, 0.0]
        );

        let decode_host =
            Tensor::from_f32(Shape::new(vec![2, 2]), &[9.0, 10.0, 11.0, 12.0]).unwrap();
        let decode_input = crate::transfers::to_cuda(&decode_host, 0).unwrap();
        let decode_input = CudaBuffer::from_tensor(&decode_input).unwrap();
        let decode_cache = CudaKvCache::new(0, 1, 2, 2, 3).unwrap();
        let position = crate::HostMappedBuffer::alloc(4, 0).unwrap();
        position.write_u32(1).unwrap();
        ops::cache::append_at(
            &ctx,
            apxinf_core::DType::F32,
            decode_cache.k_buffer(0),
            &decode_input,
            2,
            2,
            3,
            position.address(),
        )
        .unwrap();
        ctx.synchronize().unwrap();
        let mut decode_bytes = vec![0; decode_cache.k_buffer(0).len()];
        decode_cache
            .k_buffer(0)
            .copy_to_host(&mut decode_bytes)
            .unwrap();
        let actual_decode = decode_bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect::<Vec<_>>();
        assert_eq!(
            actual_decode,
            vec![0.0, 0.0, 9.0, 10.0, 0.0, 0.0, 0.0, 0.0, 11.0, 12.0, 0.0, 0.0]
        );
    }
}
