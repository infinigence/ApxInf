//! Public Qwen3.8 text runtime wired through the common `LlmTrait` path.
//!
//! The resident HTTP server remains an application adapter.  This type owns
//! the checkpoint, decoder state, and sampling-facing logits so `AutoModel`
//! can load the same native implementation without going through that server.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use apxinf_core::{Backend, Device, Error, Result, Tensor};
use apxinf_cuda::CudaBackend;
use apxinf_loader::safetensors;
use apxinf_loader::ModelConfig;

use crate::accelerator::cuda::downcast_arc;
use crate::llm_trait::{LlmCapabilities, LlmInput, LlmTrait};

use super::decode::{HybridUnit, HybridUnitMode, Qwen35LmHead};
use super::Qwen35Config;

const HIDDEN: usize = 5120;
const VOCAB: usize = 248_320;
const PREFILL_TILE: usize = 8;

/// Native Qwen3.8-27B AWQ runtime exposed through the shared text interface.
///
/// The implementation deliberately exposes text only in this first public
/// loader.  The existing application server owns the optional Python image
/// processor; image support will be added to this same `prefill` seam once the
/// processor is a maintained policy input rather than a server-only helper.
pub struct Qwen35Model {
    backend: Arc<dyn Backend>,
    cuda: Arc<CudaBackend>,
    embedding: Tensor,
    decoder: HybridUnit,
    lm_head: Qwen35LmHead,
    max_seq_len: usize,
}

impl Qwen35Model {
    /// Construct from the strict Qwen3.8 checkpoint contract.
    pub fn from_path(
        path: &Path,
        backend: Arc<dyn Backend>,
        max_seq_len: Option<usize>,
    ) -> Result<Self> {
        let cuda = downcast_arc(backend.clone())
            .ok_or_else(|| Error::Other("Qwen3.8 native runtime requires a CUDA backend".into()))?;
        if cuda.device_id() != 0 || cuda.context().caps().sm != 89 {
            return Err(Error::Other("Qwen3.8 runtime requires CUDA0/SM89".into()));
        }
        let model_dir = if path.is_dir() {
            path
        } else {
            path.parent().unwrap_or_else(|| Path::new("."))
        };
        let config = Qwen35Config::from_json_file(&model_dir.join("config.json"))?;
        let manifest = safetensors::inspect_path(model_dir).map_err(Error::Other)?;
        let embedding_entry = manifest
            .tensor("model.language_model.embed_tokens.weight")
            .ok_or_else(|| Error::Other("missing Qwen3.8 embedding table".into()))?;
        if embedding_entry.dtype != apxinf_core::DType::BF16
            || embedding_entry.shape != [VOCAB, HIDDEN]
        {
            return Err(Error::Other(format!(
                "Qwen3.8 embedding table must be BF16 [{VOCAB},{HIDDEN}], got {} {:?}",
                embedding_entry.dtype, embedding_entry.shape
            )));
        }
        let embedding = safetensors::load_manifest_tensor(embedding_entry).map_err(Error::Other)?;
        let max_seq_len = max_seq_len
            .unwrap_or(config.text.max_position_embeddings)
            .min(config.text.max_position_embeddings);
        if max_seq_len == 0 {
            return Err(Error::Other(
                "Qwen3.8 max sequence length must be non-zero".into(),
            ));
        }
        let decoder = HybridUnit::load_all_with_prefill_mode(
            &manifest,
            cuda.context(),
            max_seq_len,
            super::decode::Qwen35PrefillMode::M8,
        )?;
        let lm_head = Qwen35LmHead::load(&manifest, cuda.context())?;
        Ok(Self {
            backend,
            cuda,
            embedding,
            decoder,
            lm_head,
            max_seq_len,
        })
    }

    fn embedding_tokens(&self, tokens: &[u32]) -> Result<Tensor> {
        if tokens.is_empty() {
            return Err(Error::Other("Qwen3.8 embedding request is empty".into()));
        }
        let table = self.embedding.as_bf16()?;
        let mut values = Vec::with_capacity(tokens.len() * HIDDEN);
        for &token in tokens {
            let token = token as usize;
            if token >= VOCAB {
                return Err(Error::Other(format!(
                    "Qwen3.8 token id {token} exceeds vocabulary"
                )));
            }
            let start = token * HIDDEN;
            values.extend_from_slice(&table[start..start + HIDDEN]);
        }
        Tensor::from_bf16(vec![tokens.len(), HIDDEN], &values)
    }

    fn prefill_text(&mut self, tokens: &[u32]) -> Result<Tensor> {
        if tokens.is_empty() || tokens.len() > self.max_seq_len {
            return Err(Error::Other(format!(
                "Qwen3.8 prompt length {} exceeds capacity {}",
                tokens.len(),
                self.max_seq_len
            )));
        }
        let first = self.embedding_tokens(&tokens[..1])?;
        self.decoder
            .reset_text_request(self.cuda.context(), &first)?;
        let tiled = tokens.len() / PREFILL_TILE * PREFILL_TILE;
        for position in (0..tiled).step_by(PREFILL_TILE) {
            let input = self.embedding_tokens(&tokens[position..position + PREFILL_TILE])?;
            self.decoder
                .set_prefill8_input(self.cuda.context(), &input)?;
            self.decoder
                .forward_prefill8(self.cuda.context(), position, false)?;
        }
        for (offset, &token) in tokens[tiled..].iter().enumerate() {
            let position = tiled + offset;
            if position > 0 || tiled > 0 {
                let input = self.embedding_tokens(std::slice::from_ref(&token))?;
                self.decoder.set_token_input(self.cuda.context(), &input)?;
            }
            let bucket = (position + 1).next_power_of_two().min(self.max_seq_len);
            self.decoder.forward(
                self.cuda.context(),
                HybridUnitMode::ModelOptimized,
                bucket,
                position as u32,
                false,
            )?;
        }
        if tiled == tokens.len() && tiled > 0 {
            self.decoder.commit_prefill8_last(self.cuda.context())?;
        }
        self.lm_head
            .forward(self.cuda.context(), self.decoder.normalized_output())?;
        self.cuda.synchronize()?;
        Ok(self.lm_head.logits().clone())
    }
}

impl LlmTrait for Qwen35Model {
    fn load(
        _config: ModelConfig,
        _weights: HashMap<String, Tensor>,
        _device: Device,
    ) -> Result<Self> {
        Err(Error::Other(
            "Qwen3.8 must be loaded from its SafeTensors directory through AutoModel".into(),
        ))
    }

    fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        if token_ids.len() != 1 {
            return Err(Error::Other(
                "Qwen3.8 decode forward accepts exactly one token".into(),
            ));
        }
        let input = self.embedding_tokens(token_ids)?;
        self.decoder.set_token_input(self.cuda.context(), &input)?;
        let position = start_pos as usize;
        if position >= self.max_seq_len {
            return Err(Error::Other(
                "Qwen3.8 decode position exceeds capacity".into(),
            ));
        }
        let bucket = (position + 1).next_power_of_two().min(self.max_seq_len);
        self.decoder.forward(
            self.cuda.context(),
            HybridUnitMode::ModelOptimized,
            bucket,
            start_pos,
            false,
        )?;
        self.lm_head
            .forward(self.cuda.context(), self.decoder.normalized_output())?;
        Ok(self.lm_head.logits().clone())
    }

    fn backend(&self) -> &dyn Backend {
        &*self.backend
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::TEXT_ONLY
    }

    fn prefill(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        if input.image.is_some() {
            return Err(Error::Other(
                "Qwen3.8 public AutoModel path currently supports text only".into(),
            ));
        }
        self.prefill_text(input.token_ids)
    }

    fn reset(&mut self) {
        // Request-local GDN/KV state is reset by `prefill_text` before every
        // public generation request.  Keeping reset side-effect free avoids a
        // second host/device synchronization in the shared generation driver.
    }

    fn vocab_size(&self) -> usize {
        VOCAB
    }
}
