//! `LlmTrait` integration and `AutoModel` factory for Qwen3-MoE AWQ INT4.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use apxinf_core::{Backend, Device, Error, Result, Tensor};

use super::config::Qwen3MoeConfig;
use super::runtime::Qwen3MoeRuntime;
use super::weights::Qwen3MoeWeights;
use crate::accelerator::cuda::{downcast_arc, RuntimeBackend as CudaBackend};
use crate::auto::{LoadOptions, LoadedModel};
use crate::llm_trait::LlmTrait;

/// Default KV capacity. Qwen3-30B-A3B advertises 262k positions; the cache is
/// allocated eagerly, so keep the default modest (8k = ~0.8 GB at BF16) and
/// let callers raise it through [`Qwen3Moe::load_with_max_seq_len`].
pub const DEFAULT_MAX_SEQ_LEN: usize = 8192;

pub struct Qwen3Moe {
    backend: Arc<dyn Backend>,
    cuda: Arc<CudaBackend>,
    runtime: Qwen3MoeRuntime,
}

impl Qwen3Moe {
    /// Load an AutoAWQ checkpoint directory onto the CUDA backend.
    pub fn load(model_dir: &Path, backend: Arc<dyn Backend>) -> Result<Self> {
        Self::load_with_max_seq_len(model_dir, backend, DEFAULT_MAX_SEQ_LEN)
    }

    pub fn load_with_max_seq_len(
        model_dir: &Path,
        backend: Arc<dyn Backend>,
        max_seq_len: usize,
    ) -> Result<Self> {
        let cuda = downcast_arc(Arc::clone(&backend))
            .ok_or_else(|| Error::Other("qwen3moe requires the CUDA backend".into()))?;
        let config = Qwen3MoeConfig::from_json_file(&model_dir.join("config.json"))?;
        let max_seq_len = max_seq_len.min(config.max_position_embeddings).max(16);
        let started = std::time::Instant::now();
        let reserve = Qwen3MoeRuntime::pool_reserve_bytes(&config, max_seq_len);
        let weights = Qwen3MoeWeights::load(&config, model_dir, cuda.device_id(), reserve)?;
        eprintln!(
            "[apxinf] qwen3moe: loaded {:.2} GiB of packed weights in {:.1}s",
            weights.device_bytes as f64 / (1u64 << 30) as f64,
            started.elapsed().as_secs_f64()
        );
        let runtime = Qwen3MoeRuntime::new(&cuda, config, weights, max_seq_len)?;
        Ok(Self {
            backend,
            cuda,
            runtime,
        })
    }

    pub fn config(&self) -> &Qwen3MoeConfig {
        self.runtime.config()
    }

    pub fn runtime_mut(&mut self) -> &mut Qwen3MoeRuntime {
        &mut self.runtime
    }

    pub fn cuda_backend(&self) -> &CudaBackend {
        &self.cuda
    }

    /// Disable CUDA-graph decode (eager launches every step).
    pub fn set_use_graphs(&mut self, enabled: bool) {
        self.runtime.set_use_graphs(enabled);
    }
}

impl LlmTrait for Qwen3Moe {
    fn load(
        _config: apxinf_loader::ModelConfig,
        _weights: HashMap<String, Tensor>,
        _device: Device,
    ) -> Result<Self>
    where
        Self: Sized,
    {
        Err(Error::Other(
            "Qwen3Moe::load(ModelConfig) is not supported; use Qwen3Moe::load(dir, backend)".into(),
        ))
    }

    /// Prefill from an empty cache, or decode token by token. Logits are
    /// returned for the last token only (`[1, vocab]`), which is what the
    /// shared generation driver consumes.
    fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Err(Error::Other("qwen3moe forward: empty token_ids".into()));
        }
        let kv_len = self.runtime.kv_len();
        if start_pos as usize != kv_len {
            return Err(Error::Other(format!(
                "qwen3moe forward: start_pos {start_pos} does not match cached length {kv_len}"
            )));
        }
        if kv_len == 0 && token_ids.len() > 1 {
            let _range = crate::profiling::trace::range("prefill");
            return self.runtime.prefill(&self.cuda, token_ids);
        }
        let _range = crate::profiling::trace::range("decode");
        let mut logits = None;
        for &token in token_ids {
            logits = Some(self.runtime.decode(&self.cuda, token)?);
        }
        Ok(logits.expect("at least one token"))
    }

    fn backend(&self) -> &dyn Backend {
        &*self.backend
    }

    fn reset(&mut self) {
        self.runtime.reset();
    }

    fn vocab_size(&self) -> usize {
        self.runtime.config().vocab_size
    }
}

/// Registry factory for `model_type = "qwen3_moe"`.
pub fn load_qwen3moe(
    path: &Path,
    _device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    let model_dir = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    };
    if let Some(dtype) = options.text_weight_dtype {
        if dtype != apxinf_core::DType::BF16 {
            return Err(Error::Other(format!(
                "qwen3moe runs BF16 activations over AWQ INT4 weights; --dtype {dtype} is not supported"
            )));
        }
    }
    let model = Qwen3Moe::load(model_dir, backend)?;
    Ok(LoadedModel::text(Box::new(model)))
}
