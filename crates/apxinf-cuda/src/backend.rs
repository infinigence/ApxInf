//! CUDA backend implementing the Backend trait.

use apxinf_core::{Backend, Device, Error, Graph, KvCache, Result, Tensor};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::cublas::CublasHandle;
use crate::kernels;
use crate::transfers;
use crate::CudaKVCache;

struct CudaGraph {
    graph: crate::graph::CapturedGraph,
}

impl Graph for CudaGraph {
    fn replay(&self) -> Result<()> {
        self.graph.replay().map_err(Error::Cuda)
    }
}

/// CUDA backend — all ops execute on GPU via cuBLAS + custom kernels.
///
/// Implements the portable `Backend` trait. Also provides CUDA-specific
/// extension methods via `CudaBackend` directly.
pub struct CudaBackend {
    ctx: CudaContext,
}

impl CudaBackend {
    /// Create a CUDA backend for the given device.
    pub fn new(device_id: usize) -> Result<Self> {
        let ctx = CudaContext::new(device_id).map_err(Error::Cuda)?;
        eprintln!(
            "CUDA {}: {} (compute {}.{}, {}, {} SMs)",
            device_id,
            ctx.caps().device_name,
            ctx.caps().compute_major,
            ctx.caps().compute_minor,
            ctx.caps().arch_family,
            ctx.caps().multiprocessor_count,
        );
        Ok(Self { ctx })
    }

    /// Access the CUDA context.
    pub fn context(&self) -> &CudaContext {
        &self.ctx
    }

    /// Access the cuBLAS handle.
    pub fn cublas(&self) -> &CublasHandle {
        self.ctx.cublas()
    }

    // ── CUDA-specific extensions (not in Backend trait) ──────────────

    /// Get the device ID.
    pub fn device_id(&self) -> usize {
        self.ctx.device_id()
    }

    /// Begin a relaxed stream capture for decode graphs which call vendor
    /// libraries with internal thread-local state.
    pub fn begin_capture_relaxed(&self) -> Result<()> {
        crate::graph::begin(&self.ctx, crate::graph::CaptureMode::Relaxed).map_err(Error::Cuda)
    }
}

impl Backend for CudaBackend {
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        kernels::norm::rms(&self.ctx, input, weight, eps)
    }

    fn silu(&self, x: &Tensor) -> Result<Tensor> {
        kernels::activation::silu(&self.ctx, x)
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        kernels::elementwise::add(&self.ctx, a, b)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        kernels::elementwise::mul(&self.ctx, a, b)
    }

    fn scale(&self, input: &Tensor, factor: f32) -> Result<Tensor> {
        kernels::elementwise::scale(&self.ctx, input, factor)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        kernels::gemm::matmul(&self.ctx, a, b)
    }

    fn rope(
        &self,
        input: &Tensor,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        pos_offset: u32,
    ) -> Result<Tensor> {
        kernels::rope::apply_batched(&self.ctx, input, n_heads, head_dim, theta, pos_offset)
    }

    fn rope_mrope(
        &self,
        input: &Tensor,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        sections: [usize; 3],
        pos_ids: &[u32],
    ) -> Result<Tensor> {
        let dims = input.shape().dims();
        let seq_len = if dims.len() == 2 { 1 } else { dims[0] };
        if pos_ids.len() != seq_len * 3 {
            return Err(Error::Other(format!(
                "rope_mrope: pos_ids len {} != seq_len {} * 3",
                pos_ids.len(),
                seq_len
            )));
        }
        let ids_bytes: Vec<u8> = pos_ids.iter().flat_map(|&v| v.to_ne_bytes()).collect();
        let ids_buf =
            CudaBuffer::alloc(ids_bytes.len(), self.ctx.device_id()).map_err(Error::Cuda)?;
        ids_buf.copy_from_host(&ids_bytes).map_err(Error::Cuda)?;
        kernels::rope::apply_mrope(
            &self.ctx, input, n_heads, head_dim, theta, sections, &ids_buf,
        )
    }

    fn layer_norm(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        kernels::norm::layer(&self.ctx, input, weight, bias, eps)
    }

    fn gelu_tanh(&self, input: &Tensor) -> Result<Tensor> {
        kernels::activation::gelu_tanh(&self.ctx, input)
    }

    fn add_bias(&self, input: &Tensor, bias: &Tensor) -> Result<Tensor> {
        kernels::elementwise::add_bias(&self.ctx, input, bias)
    }

    fn rope_vision_2d(
        &self,
        input: &Tensor,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        pos_ids: &[u32],
    ) -> Result<Tensor> {
        let dims = input.shape().dims();
        let seq_len = if dims.len() == 2 { 1 } else { dims[0] };
        if pos_ids.len() != seq_len * 2 {
            return Err(Error::Other(format!(
                "rope_vision_2d: pos_ids len {} != seq_len {} * 2",
                pos_ids.len(),
                seq_len
            )));
        }
        let ids_bytes: Vec<u8> = pos_ids.iter().flat_map(|&v| v.to_ne_bytes()).collect();
        let ids_buf =
            CudaBuffer::alloc(ids_bytes.len(), self.ctx.device_id()).map_err(Error::Cuda)?;
        ids_buf.copy_from_host(&ids_bytes).map_err(Error::Cuda)?;
        kernels::rope::apply_vision_2d(&self.ctx, input, n_heads, head_dim, theta, &ids_buf)
    }

    fn vision_sdpa(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        kernels::attention::vision(&self.ctx, q, k, v, seq_len, n_heads, head_dim)
    }

    fn concat_2d(&self, tensors: &[&Tensor]) -> Result<Tensor> {
        use apxinf_core::Shape;
        if tensors.is_empty() {
            return Err(Error::Other("concat_2d: empty input".into()));
        }
        let device_id = self.ctx.device_id();
        let dtype = tensors[0].dtype();
        let elem = dtype.size_in_bytes();
        let dims0 = tensors[0].shape().dims();
        if dims0.len() != 2 {
            return Err(Error::Other(format!(
                "concat_2d: expected 2D, got {}D",
                dims0.len()
            )));
        }
        let rows = dims0[0];
        let total_cols: usize = tensors.iter().map(|t| t.shape().dims()[1]).sum();
        for t in tensors {
            let d = t.shape().dims();
            if d.len() != 2 || d[0] != rows || t.dtype() != dtype {
                return Err(Error::Other("concat_2d: shape/dtype mismatch".into()));
            }
        }
        let out_bytes = rows * total_cols * elem;
        let out_buf = crate::workspace::output_buffer(&self.ctx, out_bytes)?;
        let dst_pitch = total_cols * elem;
        let mut col_offset = 0usize;
        for t in tensors {
            let cols = t.shape().dims()[1];
            let width = cols * elem;
            let spitch = cols * elem;
            crate::transfers::copy_tensor_2d_to_buffer(
                &self.ctx,
                t,
                &out_buf,
                col_offset * elem,
                dst_pitch,
                spitch,
                width,
                rows,
            )?;
            col_offset += cols;
        }
        Ok(out_buf.into_tensor(Shape::new(vec![rows, total_cols]), dtype))
    }

    fn embedding(&self, table: &Tensor, ids: &[u32]) -> Result<Tensor> {
        let device_id = self.ctx.device_id();
        let ids_bytes: Vec<u8> = ids.iter().flat_map(|&v| v.to_ne_bytes()).collect();
        let ids_buf = CudaBuffer::alloc(ids_bytes.len(), device_id).map_err(Error::Cuda)?;
        ids_buf.copy_from_host(&ids_bytes).map_err(Error::Cuda)?;

        kernels::embedding::lookup(&self.ctx, table, &ids_buf, ids.len())
    }

    fn sdpa_decode(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        max_seq_len: usize,
    ) -> Result<Tensor> {
        // For decode (seq_len=1), the new token is at position kv_len-1
        // and must attend to all kv_len positions (including itself).
        // attention_softmax kernel computes valid_cols = seq_pos + kv_offset + 1.
        // With seq_pos=0, we need kv_offset = kv_len - 1.
        let kv_offset = (kv_len - 1) as u32;
        kernels::attention::sdpa(
            &self.ctx,
            q,
            kv,
            layer_idx,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_len,
            max_seq_len,
            kv_offset,
        )
    }

    fn sdpa_prefill(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        max_seq_len: usize,
    ) -> Result<Tensor> {
        let seq_len = q.shape().dims()[0];
        let kv_offset = kv_len - seq_len;
        kernels::attention::sdpa(
            &self.ctx,
            q,
            kv,
            layer_idx,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_len,
            max_seq_len,
            kv_offset as u32,
        )
    }

    fn create_kv_cache(
        &self,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Box<dyn KvCache> {
        Box::new(
            CudaKVCache::new(
                self.ctx.device_id(),
                n_layers,
                n_kv_heads,
                head_dim,
                max_seq_len,
            )
            .unwrap(),
        )
    }

    fn kv_append(
        &self,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        k: &Tensor,
        v: &Tensor,
        append_len: usize,
    ) -> Result<()> {
        let cache = kv
            .as_any()
            .downcast_ref::<CudaKVCache>()
            .ok_or_else(|| Error::Other("expected CudaKVCache".into()))?;
        cache.append(&self.ctx, layer_idx, k, v, append_len)
    }

    fn synchronize(&self) -> Result<()> {
        self.ctx.synchronize().map_err(Error::Cuda)
    }

    fn begin_capture(&self) -> Result<()> {
        // PI/VLA capture is driven entirely by this calling thread.
        // Thread-local mode preserves the captured work while avoiding
        // unrelated CUDA activity in other service/test threads from
        // invalidating the capture.
        crate::graph::begin(&self.ctx, crate::graph::CaptureMode::ThreadLocal).map_err(Error::Cuda)
    }

    fn end_capture(&self) -> Result<Box<dyn Graph>> {
        Ok(Box::new(CudaGraph {
            graph: crate::graph::end(&self.ctx).map_err(Error::Cuda)?,
        }))
    }

    fn device(&self) -> Device {
        Device::Cuda(self.ctx.device_id())
    }

    fn to_device(&self, tensor: &Tensor) -> Result<Tensor> {
        transfers::to_cuda(tensor, self.ctx.device_id())
    }

    fn to_cpu(&self, tensor: &Tensor) -> Result<Tensor> {
        transfers::to_cpu(tensor)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod graph_tests {
    use super::*;

    #[test]
    fn backend_graph_capture_replays_preallocated_work() {
        let backend = CudaBackend::new(0).unwrap();
        let buffer = CudaBuffer::alloc_zeros(64, 0).unwrap();
        backend.begin_capture().unwrap();
        crate::graph::captured_memset(backend.context(), &buffer, 0x5a).unwrap();
        let graph = backend.end_capture().unwrap();
        graph.replay().unwrap();
        backend.synchronize().unwrap();
        let mut output = vec![0u8; 64];
        buffer.copy_to_host(&mut output).unwrap();
        assert!(output.iter().all(|value| *value == 0x5a));
    }
}
