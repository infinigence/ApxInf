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

use crate::ops;
use crate::CudaContext;

/// A `[1, capacity, n_kv_heads, head_dim]` BF16 KV cache for one model.
///
/// The layout is the view [`crate::ops::kv_cache_attention`] consumes, so no
/// relayout is needed per step. Append writes through
/// [`crate::ops::cache_append`]; the cache never exposes its raw order, which
/// keeps the storage detail private to this file.
pub struct CudaKvCache {
    ctx: std::sync::Arc<CudaContext>,
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    seq_len: usize,
}

impl CudaKvCache {
    fn new(
        ctx: std::sync::Arc<CudaContext>,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        let shape = Shape::new(vec![1, max_seq_len, n_kv_heads, head_dim]);
        let mut keys = Vec::with_capacity(n_layers);
        let mut values = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            keys.push(ctx.allocate_output(shape.clone(), DType::BF16)?);
            values.push(ctx.allocate_output(shape.clone(), DType::BF16)?);
        }
        Ok(Self {
            ctx,
            keys,
            values,
            n_kv_heads,
            head_dim,
            max_seq_len,
            seq_len: 0,
        })
    }

    /// Key tensor for one layer, `[1, capacity, kv_heads, head_dim]`.
    pub fn key(&self, layer_idx: usize) -> &Tensor {
        &self.keys[layer_idx]
    }

    /// Value tensor for one layer, `[1, capacity, kv_heads, head_dim]`.
    pub fn value(&self, layer_idx: usize) -> &Tensor {
        &self.values[layer_idx]
    }

    fn append_impl(&mut self, layer_idx: usize, k: &Tensor, v: &Tensor, append_len: usize) -> Result<()> {
        ops::cache_append(
            &self.ctx,
            &self.keys[layer_idx],
            k,
            self.n_kv_heads,
            self.head_dim,
            self.max_seq_len,
            self.seq_len,
            append_len,
        )?;
        ops::cache_append(
            &self.ctx,
            &self.values[layer_idx],
            v,
            self.n_kv_heads,
            self.head_dim,
            self.max_seq_len,
            self.seq_len,
            append_len,
        )
    }
}

impl KvCache for CudaKvCache {
    fn append(&mut self, layer_idx: usize, k: &Tensor, v: &Tensor, append_len: usize) -> Result<()> {
        self.append_impl(layer_idx, k, v, append_len)
    }

    fn advance(&mut self, n: usize) {
        self.seq_len += n;
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn clear(&mut self) -> Result<()> {
        // The decode graph retains these addresses across requests, so the
        // buffers are zeroed in place rather than reallocated.
        for tensor in self.keys.iter().chain(self.values.iter()) {
            let buffer = crate::CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
            buffer.zero().map_err(Error::Cuda)?;
        }
        self.seq_len = 0;
        Ok(())
    }

    fn n_layers(&self) -> usize {
        self.keys.len()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

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
    pub fn new(ctx: std::sync::Arc<CudaContext>) -> Self {
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
        use apxinf_core::Backend as _;
        self.begin_capture()?;
        let output = match operation() {
            Ok(output) => output,
            Err(error) => {
                let _ = self.end_capture();
                return Err(error);
            }
        };
        Ok((self.end_capture()?, output))
    }
}

fn rank2(tensor: &Tensor, what: &str) -> Result<[usize; 2]> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!("{what} requires a rank-2 tensor")));
    }
    Ok([dims[0], dims[1]])
}

impl Backend for CudaNewBackend {
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let [rows, width] = rank2(input, "rms_norm")?;
        let output = self.ctx.allocate_output(Shape::new(vec![rows, width]), DType::BF16)?;
        ops::mlp::rms_norm(&self.ctx, input, weight, &output, eps)?;
        Ok(output)
    }

    fn silu(&self, input: &Tensor) -> Result<Tensor> {
        let dims = input.shape().clone();
        let output = self.ctx.allocate_output(dims, DType::BF16)?;
        ops::elementwise_activation(&self.ctx, input, &output, ops::ElementwiseActivation::Silu)?;
        Ok(output)
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let dims = a.shape().clone();
        let output = self.ctx.allocate_output(dims, DType::BF16)?;
        ops::elementwise_add(&self.ctx, a, b, &output)?;
        Ok(output)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let dims = a.shape().clone();
        let output = self.ctx.allocate_output(dims, DType::BF16)?;
        ops::elementwise_mul(&self.ctx, a, b, &output)?;
        Ok(output)
    }

    fn scale(&self, input: &Tensor, factor: f32) -> Result<Tensor> {
        let dims = input.shape().clone();
        let output = self.ctx.allocate_output(dims, DType::BF16)?;
        ops::elementwise_scale(&self.ctx, input, &output, factor)?;
        Ok(output)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let [m, k] = rank2(a, "matmul a")?;
        let [k2, n] = rank2(b, "matmul b")?;
        if k != k2 {
            return Err(Error::MatmulDimMismatch { m, k1: k, k2, n });
        }
        let mut output = self.ctx.allocate_output(Shape::new(vec![m, n]), DType::BF16)?;
        let args = ops::GemmArgs::new(a, b, &mut output);
        ops::gemm(&self.ctx, args)?;
        Ok(output)
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
        ops::rope_apply_mrope(&self.ctx, input, n_heads, head_dim, theta, sections, pos_ids)
    }

    fn layer_norm(&self, input: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Tensor> {
        let dims = input.shape().clone();
        let mut output = self.ctx.allocate_output(dims, DType::BF16)?;
        let args = ops::LayerNormArgs::new(input, weight, bias, &mut output, eps);
        ops::layer_norm(&self.ctx, args)?;
        Ok(output)
    }

    fn gelu_tanh(&self, input: &Tensor) -> Result<Tensor> {
        let dims = input.shape().clone();
        let output = self.ctx.allocate_output(dims, DType::BF16)?;
        ops::elementwise_activation(
            &self.ctx,
            input,
            &output,
            ops::ElementwiseActivation::GeluTanh,
        )?;
        Ok(output)
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
        if tensors.is_empty() {
            return Err(Error::Other("concat_2d: empty input".into()));
        }
        let mut result = tensors[0].clone();
        for tensor in &tensors[1..] {
            result = ops::concat_rows(&self.ctx, &result, tensor)?;
        }
        Ok(result)
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
        let q4 = q.reshape(vec![1, q.shape().dims()[0], q.shape().dims()[1], q.shape().dims()[2]])?;
        let k4 = k.reshape(vec![1, k.shape().dims()[0], k.shape().dims()[1], k.shape().dims()[2]])?;
        let v4 = v.reshape(vec![1, v.shape().dims()[0], v.shape().dims()[1], v.shape().dims()[2]])?;
        let mut out = self
            .ctx
            .allocate_output(q4.shape().clone(), DType::BF16)?;
        ops::attention(
            &self.ctx,
            ops::AttentionArgs::new(&q4, &k4, &v4, &mut out),
        )?;
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
        let ids_buffer = crate::CudaBuffer::alloc(ids_i32.len(), self.ctx.device_id())
            .map_err(Error::Cuda)?;
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
        _n_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        _max_seq_len: usize,
    ) -> Result<Tensor> {
        self.sdpa(q, kv, layer_idx, n_heads, head_dim, kv_len)
    }

    fn sdpa_prefill(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        n_heads: usize,
        _n_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        _max_seq_len: usize,
    ) -> Result<Tensor> {
        self.sdpa(q, kv, layer_idx, n_heads, head_dim, kv_len)
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
                self.ctx.clone(),
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
        kv.append(layer_idx, k, v, append_len)
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
        head_dim: usize,
        kv_len: usize,
    ) -> Result<Tensor> {
        let cache = kv
            .as_any()
            .downcast_ref::<CudaKvCache>()
            .ok_or_else(|| Error::Other("expected a cuda-new KV cache".into()))?;
        let q_dims = q.shape().dims().to_vec();
        if q_dims.len() != 3 || q_dims[1] != n_heads || q_dims[2] != head_dim {
            return Err(Error::Other(format!(
                "attention query must be [seq, {n_heads}, {head_dim}], got {q_dims:?}"
            )));
        }
        let seq_len = q_dims[0];
        if seq_len == 0 || kv_len < seq_len {
            return Err(Error::Other("attention kv_len is smaller than the query".into()));
        }
        let q4 = q.reshape(vec![1, seq_len, n_heads, head_dim])?;
        let key = cache.key(layer_idx);
        let value = cache.value(layer_idx);
        let mut out = self.ctx.allocate_output(q4.shape().clone(), DType::BF16)?;
        // Causal mask: query token i occupies cache position query_start + i,
        // so both fresh prefill (query_start 0) and decode (query_start
        // kv_len-1) use the same causal contract.
        let mut args = ops::KvCacheAttentionArgs::new(&q4, key, value, &mut out);
        args.valid_key_tokens = kv_len;
        args.query_start = kv_len - seq_len;
        args.mask = ops::AttentionMask::Causal;
        args.scale = 1.0 / (head_dim as f32).sqrt();
        ops::kv_cache_attention(&self.ctx, args)?;
        Ok(out.reshape(vec![seq_len, n_heads * head_dim])?)
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
