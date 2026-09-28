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
//! # Status
//!
//! Skeleton. Every operator reports which operator is missing, so running a
//! model against it names the next thing to implement rather than failing with
//! a generic message. The runtime, memory and kernels arrive next; the model
//! path that reaches them is already wired, so that work is purely additive.
//!
//! Reachable models are those that compose through `dyn Backend` — Qwen3VL and
//! llama. PI0.5, pi0-fast, GR00T and Qwen-Drive downcast to the concrete
//! `CudaBackend` and call its kernels directly, so they cannot reach any
//! non-CUDA backend until they are composed through the trait.

use std::any::Any;

use apxinf_core::{
    Backend, Device, Error, Graph, KvCache, NormalGenerator, Result, SamplingBackend, Tensor,
    TokenSampler, TokenSamplingSpec,
};

/// Report a not-yet-implemented operator by name.
///
/// `apxinf_core::Error` has no dedicated variant for this on the current trait,
/// and adding one would change a shared type for every backend. The operator
/// name is what a caller needs, so it goes in the message.
fn pending<T>(op: &'static str) -> Result<T> {
    Err(Error::Other(format!(
        "apxinf-hip: `{op}` is not implemented yet"
    )))
}

/// A HIP device.
///
/// Construction does not yet touch the HIP runtime, so it cannot report a
/// missing device or a bad ordinal. Once the runtime lands, `new` validates the
/// ordinal and queries capabilities, and this note goes away.
pub struct HipBackend {
    device_id: usize,
}

impl HipBackend {
    pub fn new(device_id: usize) -> Result<Self> {
        Ok(Self { device_id })
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }
}

impl SamplingBackend for HipBackend {
    fn create_token_sampler(&self, _spec: TokenSamplingSpec) -> Result<Box<dyn TokenSampler>> {
        pending("create_token_sampler")
    }

    fn create_normal_generator(&self, _output: Tensor) -> Result<Box<dyn NormalGenerator>> {
        pending("create_normal_generator")
    }
}

impl Backend for HipBackend {
    fn rms_norm(&self, _input: &Tensor, _weight: &Tensor, _eps: f32) -> Result<Tensor> {
        pending("rms_norm")
    }

    fn silu(&self, _input: &Tensor) -> Result<Tensor> {
        pending("silu")
    }

    fn add(&self, _a: &Tensor, _b: &Tensor) -> Result<Tensor> {
        pending("add")
    }

    fn mul(&self, _a: &Tensor, _b: &Tensor) -> Result<Tensor> {
        pending("mul")
    }

    fn scale(&self, _input: &Tensor, _factor: f32) -> Result<Tensor> {
        pending("scale")
    }

    fn matmul(&self, _a: &Tensor, _b: &Tensor) -> Result<Tensor> {
        pending("matmul")
    }

    fn rope(
        &self,
        _input: &Tensor,
        _n_heads: usize,
        _head_dim: usize,
        _theta: f32,
        _pos_offset: u32,
    ) -> Result<Tensor> {
        pending("rope")
    }

    fn embedding(&self, _table: &Tensor, _ids: &[u32]) -> Result<Tensor> {
        pending("embedding")
    }

    #[allow(clippy::too_many_arguments)]
    fn sdpa_decode(
        &self,
        _q: &Tensor,
        _kv: &mut dyn KvCache,
        _layer_idx: usize,
        _n_heads: usize,
        _n_kv_heads: usize,
        _head_dim: usize,
        _kv_len: usize,
        _max_seq_len: usize,
    ) -> Result<Tensor> {
        pending("sdpa_decode")
    }

    #[allow(clippy::too_many_arguments)]
    fn sdpa_prefill(
        &self,
        _q: &Tensor,
        _kv: &mut dyn KvCache,
        _layer_idx: usize,
        _n_heads: usize,
        _n_kv_heads: usize,
        _head_dim: usize,
        _kv_len: usize,
        _max_seq_len: usize,
    ) -> Result<Tensor> {
        pending("sdpa_prefill")
    }

    /// Returns a placeholder cache.
    ///
    /// The trait returns `Box<dyn KvCache>` rather than a `Result`, so this
    /// cannot report the missing implementation here. [`PendingKvCache`] fails
    /// on first use instead, which is the earliest point an error can be
    /// returned.
    fn create_kv_cache(
        &self,
        _n_layers: usize,
        _n_kv_heads: usize,
        _head_dim: usize,
        _max_seq_len: usize,
    ) -> Box<dyn KvCache> {
        Box::new(PendingKvCache)
    }

    fn kv_append(
        &self,
        _kv: &mut dyn KvCache,
        _layer_idx: usize,
        _k: &Tensor,
        _v: &Tensor,
        _append_len: usize,
    ) -> Result<()> {
        pending("kv_append")
    }

    fn synchronize(&self) -> Result<()> {
        pending("synchronize")
    }

    fn begin_capture(&self) -> Result<()> {
        pending("begin_capture")
    }

    fn end_capture(&self) -> Result<Box<dyn Graph>> {
        pending("end_capture")
    }

    fn device(&self) -> Device {
        Device::Hip(self.device_id)
    }

    fn to_device(&self, _tensor: &Tensor) -> Result<Tensor> {
        pending("to_device")
    }

    fn to_cpu(&self, _tensor: &Tensor) -> Result<Tensor> {
        pending("to_cpu")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Placeholder KV cache, so `create_kv_cache` has something to hand back until
/// a real one exists. Every fallible method reports the missing backing store;
/// the infallible ones describe an empty cache.
struct PendingKvCache;

impl KvCache for PendingKvCache {
    fn append(
        &mut self,
        _layer_idx: usize,
        _k: &Tensor,
        _v: &Tensor,
        _append_len: usize,
    ) -> Result<()> {
        pending("KvCache::append")
    }

    fn advance(&mut self, _n: usize) {}

    fn seq_len(&self) -> usize {
        0
    }

    fn clear(&mut self) -> Result<()> {
        pending("KvCache::clear")
    }

    fn n_layers(&self) -> usize {
        0
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
    use super::*;

    /// The backend must be object-safe and report its own device: model code
    /// holds `Arc<dyn Backend>` and routes tensors by what `device()` returns.
    #[test]
    fn reports_its_own_hip_device_through_the_trait() {
        let backend: Box<dyn Backend> = Box::new(HipBackend::new(1).unwrap());
        assert_eq!(backend.device(), Device::Hip(1));
        assert!(backend.device().is_gpu());
    }

    /// A missing operator names itself, so a model run points at the next thing
    /// to implement rather than reporting a generic failure.
    #[test]
    fn unimplemented_operators_name_themselves() {
        let backend = HipBackend::new(0).unwrap();
        let x = Tensor::zeros(vec![2, 2], apxinf_core::DType::F32);
        let message = backend.silu(&x).unwrap_err().to_string();
        assert!(message.contains("silu"), "unexpected message: {message}");
        assert!(message.contains("apxinf-hip"), "unexpected message: {message}");
    }

    /// `create_kv_cache` cannot fail in the trait, so the failure has to land on
    /// first use rather than being swallowed.
    #[test]
    fn placeholder_kv_cache_fails_on_use_not_on_creation() {
        let backend = HipBackend::new(0).unwrap();
        let mut cache = backend.create_kv_cache(2, 1, 4, 8);
        let t = Tensor::zeros(vec![1, 1, 4], apxinf_core::DType::F32);
        assert!(cache.append(0, &t, &t, 1).is_err());
        assert!(cache.clear().is_err());
    }
}
