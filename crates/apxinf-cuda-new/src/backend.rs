//! Portable [`apxinf_core::Backend`] over the cuda-new runtime.
//!
//! Models written against `dyn Backend` — llama and qwen3-vl — compose the
//! model-neutral trait methods into an architecture. This adapter lets those
//! models run on cuda-new without a legacy backend, which is what lets the LLM
//! and VLM families finish their migration off `apxinf-cuda`.
//!
//! The trait is object-safe; [`CudaNewBackend`] is the concrete type the model
//! registry hands back through `Arc<dyn Backend>`. Numeric parity with the
//! legacy backend is a contract: every operator delegates to a cuda-new op
//! whose arithmetic matches the kernel it replaces.

use std::any::Any;

use apxinf_core::{
    Backend, DType, Device, Error, Graph, KvCache, NormalGenerator, Result, Shape, Tensor,
    TokenSampler, TokenSamplingSpec,
};

use crate::{ops, CudaContext, CudaKvCache};

/// The cuda-new backend returned to `dyn Backend` consumers.
pub struct CudaNewBackend {
    ctx: std::sync::Arc<CudaContext>,
}

// The context owns a native runtime pointer and a CUDA stream; both are valid
// across threads for the enqueue-only operations this trait exposes, matching
// the legacy backend's `unsafe impl Send/Sync` on its stream and cuBLAS handle.
unsafe impl Send for CudaNewBackend {}
unsafe impl Sync for CudaNewBackend {}

impl CudaNewBackend {
    /// Construct a backend for a CUDA device.
    ///
    /// This preserves the constructor contract exposed by the legacy CUDA
    /// backend so model code does not need to learn how the runtime owns its
    /// context.
    pub fn new(device_id: usize) -> Result<Self> {
        let ctx = CudaContext::new(device_id).map_err(Error::Cuda)?;
        Ok(Self::from_context(std::sync::Arc::new(ctx)))
    }

    /// Construct a backend around an existing context.
    ///
    /// This is reserved for execution paths that intentionally share one
    /// stream between direct cuda-new operators and portable `Backend` calls.
    pub fn from_context(ctx: std::sync::Arc<CudaContext>) -> Self {
        Self { ctx }
    }

    pub fn context(&self) -> &CudaContext {
        &self.ctx
    }

    /// Shared context handle, for models that own a `CudaContext` themselves
    /// and sample through this backend (see qwen38).
    pub fn shared_context(&self) -> &std::sync::Arc<CudaContext> {
        &self.ctx
    }

    /// Device ordinal this backend targets.
    pub fn device_id(&self) -> usize {
        self.ctx.device_id()
    }

    /// Copy a host tensor onto this backend's device.
    pub fn to_device(&self, tensor: &Tensor) -> Result<Tensor> {
        crate::transfers::to_cuda(tensor, self.ctx.device_id())
    }

    /// Copy a device tensor back to the host.
    pub fn to_cpu(&self, tensor: &Tensor) -> Result<Tensor> {
        crate::transfers::to_cpu(tensor)
    }

    /// Block until all queued work completes.
    pub fn synchronize(&self) -> Result<()> {
        self.ctx.synchronize().map_err(Error::Cuda)
    }

    /// Capture `operation` into a CUDA graph and return both. A failing
    /// operation aborts the capture before propagating the error.
    pub fn capture_graph<T>(
        &self,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<(Box<dyn Graph>, T)> {
        struct CaptureScope<'a> {
            backend: &'a CudaNewBackend,
            active: bool,
        }
        impl Drop for CaptureScope<'_> {
            fn drop(&mut self) {
                if self.active {
                    crate::graph::abort(self.backend.context());
                }
            }
        }
        use apxinf_core::Backend as _;
        self.begin_capture()?;
        let mut scope = CaptureScope {
            backend: self,
            active: true,
        };
        let output = operation()?;
        scope.active = false;
        Ok((self.end_capture()?, output))
    }
}

impl Backend for CudaNewBackend {
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        ops::norm::rms_bf16(&self.ctx, input, weight, eps)
    }

    fn silu(&self, input: &Tensor) -> Result<Tensor> {
        ops::activation::silu(&self.ctx, input)
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Dtype-dispatching like the legacy backend: smolvla's Xavier FP16
        // path adds residuals through `Backend::add`. BF16 keeps the exact
        // cuda-new operator route it had before.
        ops::elementwise::add(&self.ctx, a, b)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let dims = a.shape().clone();
        let output = self.ctx.allocate_output(dims, DType::BF16)?;
        ops::elementwise_mul(&self.ctx, a, b, &output)?;
        Ok(output)
    }

    fn scale(&self, input: &Tensor, factor: f32) -> Result<Tensor> {
        ops::elementwise::scale(&self.ctx, input, factor)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        ops::gemm::matmul(&self.ctx, a, b)
    }

    fn rope(
        &self,
        input: &Tensor,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        pos_offset: u32,
    ) -> Result<Tensor> {
        ops::rope_apply_batched(&self.ctx, input, n_heads, head_dim, theta, pos_offset)
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
        ops::rope_apply_mrope(
            &self.ctx, input, n_heads, head_dim, theta, sections, pos_ids,
        )
    }

    fn layer_norm(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        let dims = input.shape().clone();
        let mut output = self.ctx.allocate_output(dims, DType::BF16)?;
        let args = ops::LayerNormArgs::new(input, weight, bias, &mut output, eps);
        ops::layer_norm(&self.ctx, args)?;
        Ok(output)
    }

    fn gelu_tanh(&self, input: &Tensor) -> Result<Tensor> {
        ops::activation::gelu_tanh(&self.ctx, input)
    }

    fn add_bias(&self, input: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let dims = input.shape().clone();
        let output = self.ctx.allocate_output(dims, DType::BF16)?;
        ops::elementwise_add_bias(&self.ctx, input, bias, &output)?;
        Ok(output)
    }

    fn rope_vision_2d(
        &self,
        input: &Tensor,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        pos_ids: &[u32],
    ) -> Result<Tensor> {
        ops::rope_apply_vision_2d(&self.ctx, input, n_heads, head_dim, theta, pos_ids)
    }

    fn concat_2d(&self, tensors: &[&Tensor]) -> Result<Tensor> {
        use apxinf_core::Shape;
        if tensors.is_empty() {
            return Err(Error::Other("concat_2d: empty input".into()));
        }
        let dtype = tensors[0].dtype();
        let dims = tensors[0].shape().dims();
        if dims.len() != 2 {
            return Err(Error::Other(format!(
                "concat_2d: expected 2D, got {}D",
                dims.len()
            )));
        }
        let rows = dims[0];
        let total_cols = tensors.iter().try_fold(0usize, |total, tensor| {
            let dims = tensor.shape().dims();
            if dims.len() != 2 || dims[0] != rows || tensor.dtype() != dtype {
                return Err(Error::Other("concat_2d: shape/dtype mismatch".into()));
            }
            total
                .checked_add(dims[1])
                .ok_or_else(|| Error::Other("concat_2d: column count overflow".into()))
        })?;
        let elem = dtype.size_in_bytes();
        let dst_pitch = total_cols
            .checked_mul(elem)
            .ok_or_else(|| Error::Other("concat_2d: destination pitch overflow".into()))?;
        let bytes = rows
            .checked_mul(dst_pitch)
            .ok_or_else(|| Error::Other("concat_2d: output size overflow".into()))?;
        let output = crate::workspace::output_buffer(&self.ctx, bytes)?;
        let mut col_offset = 0usize;
        for tensor in tensors {
            let cols = tensor.shape().dims()[1];
            let width = cols
                .checked_mul(elem)
                .ok_or_else(|| Error::Other("concat_2d: source pitch overflow".into()))?;
            crate::transfers::copy_tensor_2d_to_buffer(
                &self.ctx,
                tensor,
                &output,
                col_offset * elem,
                dst_pitch,
                width,
                width,
                rows,
            )?;
            col_offset += cols;
        }
        Ok(output.into_tensor(Shape::new(vec![rows, total_cols]), dtype))
    }

    fn vision_sdpa(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        _seq_len: usize,
        _n_heads: usize,
        _head_dim: usize,
    ) -> Result<Tensor> {
        // [seq, heads, dim] -> [1, seq, heads, dim] dense non-causal attention.
        let q4 = q.reshape(vec![
            1,
            q.shape().dims()[0],
            q.shape().dims()[1],
            q.shape().dims()[2],
        ])?;
        let k4 = k.reshape(vec![
            1,
            k.shape().dims()[0],
            k.shape().dims()[1],
            k.shape().dims()[2],
        ])?;
        let v4 = v.reshape(vec![
            1,
            v.shape().dims()[0],
            v.shape().dims()[1],
            v.shape().dims()[2],
        ])?;
        let mut out = self.ctx.allocate_output(q4.shape().clone(), DType::BF16)?;
        ops::attention(&self.ctx, ops::AttentionArgs::new(&q4, &k4, &v4, &mut out))?;
        let dims = out.shape().dims().to_vec();
        Ok(out.reshape(vec![dims[1], dims[2] * dims[3]])?)
    }

    fn embedding(&self, table: &Tensor, ids: &[u32]) -> Result<Tensor> {
        let width = table.shape().dims()[1];
        let output = self
            .ctx
            .allocate_output(Shape::new(vec![ids.len(), width]), DType::BF16)?;
        let ids_i32: Vec<u8> = ids
            .iter()
            .flat_map(|&id| (id as i32).to_ne_bytes())
            .collect();
        let ids_buffer =
            crate::CudaBuffer::alloc(ids_i32.len(), self.ctx.device_id()).map_err(Error::Cuda)?;
        ids_buffer.copy_from_host(&ids_i32).map_err(Error::Cuda)?;
        let ids_tensor = ids_buffer
            .as_tensor(Shape::new(vec![ids.len()]), DType::I32)
            .map_err(Error::Cuda)?;
        ops::embedding_gather(&self.ctx, table, &ids_tensor, &output)?;
        Ok(output)
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
        self.sdpa(
            q,
            kv,
            layer_idx,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_len,
            max_seq_len,
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
        self.sdpa(
            q,
            kv,
            layer_idx,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_len,
            max_seq_len,
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
            CudaKvCache::new(
                self.ctx.device_id(),
                n_layers,
                n_kv_heads,
                head_dim,
                max_seq_len,
            )
            .expect("KV cache allocation"),
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
            .downcast_ref::<CudaKvCache>()
            .ok_or_else(|| Error::Other("expected a cuda-new KV cache".into()))?;
        cache.append(&self.ctx, layer_idx, k, v, append_len)
    }

    fn synchronize(&self) -> Result<()> {
        self.ctx.synchronize().map_err(Error::Cuda)
    }

    fn begin_capture(&self) -> Result<()> {
        crate::graph::begin(&self.ctx).map_err(Error::Cuda)
    }

    fn end_capture(&self) -> Result<Box<dyn Graph>> {
        let graph = crate::graph::end(&self.ctx).map_err(Error::Cuda)?;
        Ok(Box::new(graph))
    }

    fn device(&self) -> Device {
        Device::Cuda(self.ctx.device_id())
    }

    fn to_device(&self, tensor: &Tensor) -> Result<Tensor> {
        crate::transfers::to_cuda(tensor, self.ctx.device_id())
    }

    fn to_cpu(&self, tensor: &Tensor) -> Result<Tensor> {
        crate::transfers::to_cpu(tensor)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl CudaNewBackend {
    #[allow(clippy::too_many_arguments)]
    fn sdpa(
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
        if seq_len == 0 || kv_len < seq_len {
            return Err(Error::Other(
                "attention kv_len is smaller than the query".into(),
            ));
        }
        ops::attention::sdpa(
            &self.ctx,
            q,
            kv,
            layer_idx,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_len,
            max_seq_len,
            (kv_len - seq_len) as u32,
        )
    }
}

impl apxinf_core::SamplingBackend for CudaNewBackend {
    fn create_token_sampler(&self, spec: TokenSamplingSpec) -> Result<Box<dyn TokenSampler>> {
        crate::sampling::create_token_sampler(&self.ctx, spec)
    }

    fn create_normal_generator(&self, output: Tensor) -> Result<Box<dyn NormalGenerator>> {
        crate::sampling::create_normal_generator(&self.ctx, output)
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use apxinf_core::{Backend, CpuBackend, Shape, Tensor};
    use half::bf16;

    use super::*;

    #[test]
    fn portable_f32_sdpa_matches_cpu_with_head_major_cache() {
        let _guard = crate::CUDA_TEST_LOCK.lock().unwrap();
        let backend = CudaNewBackend::new(0).unwrap();
        let cpu = CpuBackend;
        let q = Tensor::from_f32(
            Shape::new(vec![2, 2, 2]),
            &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0],
        )
        .unwrap();
        let k = Tensor::from_f32(Shape::new(vec![2, 1, 2]), &[1.0, 0.0, 0.0, 1.0]).unwrap();
        let v = Tensor::from_f32(Shape::new(vec![2, 1, 2]), &[2.0, 4.0, 6.0, 8.0]).unwrap();

        let mut cpu_cache = cpu.create_kv_cache(1, 1, 2, 4);
        cpu.kv_append(&mut *cpu_cache, 0, &k, &v, 2).unwrap();
        let expected = cpu
            .sdpa_prefill(&q, &mut *cpu_cache, 0, 2, 1, 2, 2, 4)
            .unwrap()
            .to_f32_vec()
            .unwrap();

        let q_gpu = backend.to_device(&q).unwrap();
        let k_gpu = backend.to_device(&k).unwrap();
        let v_gpu = backend.to_device(&v).unwrap();
        let mut gpu_cache = backend.create_kv_cache(1, 1, 2, 4);
        backend
            .kv_append(&mut *gpu_cache, 0, &k_gpu, &v_gpu, 2)
            .unwrap();
        let output = backend
            .sdpa_prefill(&q_gpu, &mut *gpu_cache, 0, 2, 1, 2, 2, 4)
            .unwrap();
        backend.synchronize().unwrap();
        let actual = backend.to_cpu(&output).unwrap().to_f32_vec().unwrap();
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error < 1e-5, "F32 SDPA max error: {max_error}");
    }

    #[test]
    fn concat_2d_concatenates_columns() {
        let _guard = crate::CUDA_TEST_LOCK.lock().unwrap();
        let backend = CudaNewBackend::new(0).unwrap();
        let left = Tensor::from_bf16(
            Shape::new(vec![2, 2]),
            &[
                bf16::from_f32(1.0),
                bf16::from_f32(2.0),
                bf16::from_f32(3.0),
                bf16::from_f32(4.0),
            ],
        )
        .unwrap();
        let right = Tensor::from_bf16(
            Shape::new(vec![2, 1]),
            &[bf16::from_f32(5.0), bf16::from_f32(6.0)],
        )
        .unwrap();
        let left = backend.to_device(&left).unwrap();
        let right = backend.to_device(&right).unwrap();

        let output = backend.concat_2d(&[&left, &right]).unwrap();
        backend.synchronize().unwrap();
        let output = backend.to_cpu(&output).unwrap();

        assert_eq!(output.shape().dims(), [2, 3]);
        let actual: Vec<f32> = output
            .as_bf16()
            .unwrap()
            .iter()
            .map(|value| value.to_f32())
            .collect();
        assert_eq!(actual, [1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn capture_graph_aborts_after_panic() {
        let _guard = crate::CUDA_TEST_LOCK.lock().unwrap();
        let backend = CudaNewBackend::new(0).unwrap();
        let panicked = catch_unwind(AssertUnwindSafe(|| {
            let _ = backend.capture_graph::<()>(|| panic!("capture failure"));
        }));
        assert!(panicked.is_err());

        let buffer = crate::CudaBuffer::alloc(16, backend.device_id()).unwrap();
        let (graph, ()) = backend
            .capture_graph(|| {
                buffer
                    .zero_async(backend.context().stream())
                    .map_err(Error::Cuda)
            })
            .unwrap();
        graph.replay().unwrap();
        backend.synchronize().unwrap();
    }
}
