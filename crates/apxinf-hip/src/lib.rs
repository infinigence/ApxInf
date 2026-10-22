//! AMD ROCm/HIP backend.
//!
//! This crate implements `apxinf_core::Backend` for AMD GPUs, beside
//! `apxinf-cuda` and `apxinf-cuda-new` rather than sharing code with either.
//! The two CUDA crates are mid-restructure; coupling to them now would mean
//! building against a moving target, so this backend depends only on
//! `apxinf-core` and the trait as it already exists.
//!
//! It is a different mechanism from the `experiment/hip-bringup` compatibility
//! build, which hipifies the CUDA adapters behind a C shim that preserves the
//! CUDA ABI and still reports `cuda:0`. That build stays as it is; this crate
//! is a real backend with its own device identity, `Device::Hip`.
//!
//! # What is implemented
//!
//! Every method `Backend` requires, plus sampling: the elementwise operators,
//! RMSNorm, half-split RoPE, embedding, GEMM (hipBLAS, F32 accumulation),
//! cached causal attention, a KV cache, transfers and synchronization. F32 and
//! BF16 storage, F32 arithmetic, one rounding at the store — the numerics of
//! apxinf-cuda, which models already run against. Device code lives in
//! `kernels/apxinf_hip.hip`; see `build.rs` for how it is built and what
//! happens on a host without ROCm.
//!
//! Not yet: the seven defaulted methods only Qwen3VL calls (`layer_norm`,
//! `gelu_tanh`, `add_bias`, `rope_mrope`, `rope_vision_2d`, `vision_sdpa`,
//! `concat_2d`) and graph capture. Nothing is fused or tuned; this milestone is
//! about correctness, and the performance work comes after a model runs.
//!
//! Reachable models are those that compose through `dyn Backend` — Qwen3VL and
//! llama. PI0.5, pi0-fast, GR00T and Qwen-Drive downcast to a concrete CUDA
//! backend and call its kernels directly, so they cannot reach any non-CUDA
//! backend until they are composed through the trait.

mod ffi;
mod kv_cache;
mod ops;
mod runtime;
mod sampling;

use std::any::Any;
use std::sync::Arc;

use apxinf_core::{
    Backend, Device, Error, Graph, KvCache, NormalGenerator, Result, SamplingBackend, Tensor,
    TokenSampler, TokenSamplingSpec,
};

pub use kv_cache::HipKVCache;
pub use runtime::{HipContext, HipDeviceCaps};

/// A HIP device as an `apxinf_core::Backend`.
pub struct HipBackend {
    ctx: Arc<HipContext>,
}

impl HipBackend {
    /// Open device `hip:{device_id}`. Fails if the crate was built without
    /// ROCm, the device does not exist, or its architecture differs from the
    /// one the kernels were compiled for.
    pub fn new(device_id: usize) -> Result<Self> {
        Ok(Self { ctx: Arc::new(HipContext::new(device_id)?) })
    }

    pub fn device_id(&self) -> usize {
        self.ctx.device_id()
    }

    pub fn caps(&self) -> &HipDeviceCaps {
        self.ctx.caps()
    }

    pub fn context(&self) -> &Arc<HipContext> {
        &self.ctx
    }

    fn cache<'a>(&self, kv: &'a mut dyn KvCache) -> Result<&'a mut HipKVCache> {
        kv.as_any_mut()
            .downcast_mut::<HipKVCache>()
            .ok_or_else(|| Error::Other("apxinf-hip: expected a KV cache created by HipBackend".into()))
    }
}

impl SamplingBackend for HipBackend {
    fn create_token_sampler(&self, spec: TokenSamplingSpec) -> Result<Box<dyn TokenSampler>> {
        Ok(Box::new(sampling::HipTokenSampler::new(Arc::clone(&self.ctx), spec)?))
    }

    fn create_normal_generator(&self, output: Tensor) -> Result<Box<dyn NormalGenerator>> {
        Ok(Box::new(sampling::HipNormalGenerator::new(Arc::clone(&self.ctx), output)?))
    }
}

impl Backend for HipBackend {
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        ops::rms_norm(&self.ctx, input, weight, eps)
    }

    fn silu(&self, input: &Tensor) -> Result<Tensor> {
        ops::silu(&self.ctx, input)
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        ops::binary(&self.ctx, "add", 0, a, b)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        ops::binary(&self.ctx, "mul", 1, a, b)
    }

    fn scale(&self, input: &Tensor, factor: f32) -> Result<Tensor> {
        ops::scale(&self.ctx, input, factor)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        ops::matmul(&self.ctx, a, b)
    }

    fn rope(
        &self,
        input: &Tensor,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        pos_offset: u32,
    ) -> Result<Tensor> {
        ops::rope(&self.ctx, input, n_heads, head_dim, theta, pos_offset)
    }

    fn embedding(&self, table: &Tensor, ids: &[u32]) -> Result<Tensor> {
        ops::embedding(&self.ctx, table, ids)
    }

    /// One query token sees every cached position.
    #[allow(clippy::too_many_arguments)]
    fn sdpa_decode(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        _max_seq_len: usize,
    ) -> Result<Tensor> {
        let kv_offset = kv_len.checked_sub(1).ok_or_else(|| {
            Error::Other("apxinf-hip: decode attention over an empty cache".into())
        })?;
        self.cache(kv)?
            .attention(q, layer_idx, n_heads, n_kv_heads, head_dim, kv_len, kv_offset)
    }

    /// Query row `i` of `q_len` sees positions `[0, kv_len - q_len + i]`.
    #[allow(clippy::too_many_arguments)]
    fn sdpa_prefill(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        _max_seq_len: usize,
    ) -> Result<Tensor> {
        let q_len = q.shape().dims().first().copied().unwrap_or(0);
        let kv_offset = kv_len.checked_sub(q_len).ok_or_else(|| {
            Error::Other(format!(
                "apxinf-hip: prefill of {q_len} tokens cannot attend over {kv_len} positions"
            ))
        })?;
        self.cache(kv)?
            .attention(q, layer_idx, n_heads, n_kv_heads, head_dim, kv_len, kv_offset)
    }

    /// The cache allocates on its first append, once it knows the dtype; that
    /// is also where an allocation failure can be reported, since this method
    /// cannot return one.
    fn create_kv_cache(
        &self,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Box<dyn KvCache> {
        Box::new(HipKVCache::new(
            Arc::clone(&self.ctx),
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
        ))
    }

    fn kv_append(
        &self,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        k: &Tensor,
        v: &Tensor,
        append_len: usize,
    ) -> Result<()> {
        self.cache(kv)?.append(layer_idx, k, v, append_len)
    }

    fn synchronize(&self) -> Result<()> {
        self.ctx.synchronize()
    }

    fn begin_capture(&self) -> Result<()> {
        Err(Error::Other(
            "apxinf-hip: graph capture is not implemented yet; run eagerly".into(),
        ))
    }

    fn end_capture(&self) -> Result<Box<dyn Graph>> {
        Err(Error::Other(
            "apxinf-hip: graph capture is not implemented yet; run eagerly".into(),
        ))
    }

    fn device(&self) -> Device {
        self.ctx.device()
    }

    fn to_device(&self, tensor: &Tensor) -> Result<Tensor> {
        self.ctx.to_device(tensor)
    }

    fn to_cpu(&self, tensor: &Tensor) -> Result<Tensor> {
        self.ctx.to_cpu(tensor)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Device behaviour is tested on hardware in `tests/ops.rs`; this covers the
/// one thing a host without ROCm can check.
#[cfg(all(test, not(apxinf_hip_runtime)))]
mod tests {
    use super::*;

    /// Without ROCm the backend must refuse to start and say why, rather than
    /// fail to link or hand out a backend that errors on every call.
    #[test]
    fn without_rocm_the_backend_reports_why_it_cannot_start() {
        let message = match HipBackend::new(0) {
            Ok(_) => panic!("a stub build must not construct a backend"),
            Err(error) => error.to_string(),
        };
        assert!(message.contains("built without ROCm"), "unexpected: {message}");
    }
}
