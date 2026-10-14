//! The [`LlmTrait`] wrapper: state, execution policy, and the generation
//! surface over the split modules: [`super::weights`], [`super::model`], [`super::model_runner`].

use std::collections::HashMap;
use std::sync::Arc;

use apxinf_core::{Backend, DType, Device, Error, Result, Shape, Tensor};
use apxinf_cuda_new::{ops, CudaBuffer, CudaContext, PreparedPhase};
use apxinf_loader::ModelConfig;

use crate::accelerator::create_backend;
use crate::llm_trait::{LlmCapabilities, LlmInput, LlmTrait};

use super::config::{
    CHUNK, CONV_WIDTH, FULL_ATTENTION_INTERVAL, GDN_HEAD_DIM, GDN_V_HEADS, HEAD_DIM, KV_HEADS,
    LAYERS, MIN_KV_CAPACITY, QKV_WIDTH, VOCAB,
};
use super::vision_weights::VisionWeights;
use super::{backend, model, model_runner, vision, weights};

/// Qwen3.8-27B-NVFP4. Batch 1, text and optional image input, BF16 KV cache.
///
/// Execution policy is fixed: batched FlashInfer-GDN prefill, CUDA-graph
/// decode (one phase per layer -- MLP only for a full-attention layer, mixer
/// plus MLP for a GDN layer) and split-KV FA2 decode attention -- the
/// validated configuration. The kernel harness keeps its own switches for A/B
/// comparison.
pub struct Qwen38 {
    ctx: CudaContext,
    /// Existing model-neutral sampling backend. Vision and language compute
    /// remain on `ctx`; this field is used only by `LlmTrait` token sampling.
    backend: Arc<dyn Backend>,
    model: weights::Model,
    /// Some text-only checkpoints intentionally omit the visual tower.
    vision: Option<VisionWeights>,
    vision_state: vision::VisionState,
    scratch: model::Scratch,
    prefill: Option<model::PrefillScratch>,
    /// The prepared prefill session, with the token count it was prepared for.
    /// Executions bind the GEMM shapes of that traversal, so a different count
    /// cannot replay them.
    prefill_session: Option<(usize, ops::ExecutionSession)>,
    gdn_states: Vec<model::GdnState>,
    kv_caches: Vec<model::KvCache>,
    decode_phases: Option<Vec<PreparedPhase>>,
    position: usize,
    /// Difference between linear cache positions and compressed image mRoPE.
    rope_delta: i64,
    kv_capacity: usize,
}

impl Qwen38 {
    /// Load from an already-read checkpoint tensor map.
    ///
    /// `kv_capacity` bounds prompt + generation length; caches are allocated
    /// once at this size.
    pub fn new(
        weights: HashMap<String, Tensor>,
        device: Device,
        kv_capacity: usize,
    ) -> Result<Self> {
        let Device::Cuda(ordinal) = device else {
            return Err(Error::Other("qwen38 requires a CUDA device".into()));
        };
        // The remaining kernel-path selectors are getenv() switches inside the
        // cuda-new native code. They are part of this model's validated
        // execution (the acceptance md5 was produced with exactly these), so
        // the module supplies them as defaults rather than depending on the
        // caller's environment. An explicit environment value still wins, for
        // kernel diagnostics — but that run leaves the validated
        // configuration. TODO(cuda-new): promote to per-call arguments.
        model_runner::apply_validated_kernel_defaults();

        let ctx = CudaContext::new(ordinal).map_err(Error::Cuda)?;
        let backend = create_backend(device)?;
        let vision = weights
            .contains_key("model.visual.patch_embed.proj.weight")
            .then(|| VisionWeights::load(&weights, &ctx))
            .transpose()?;
        let model = weights::load_model(&ctx, &weights, true);
        ctx.synchronize().map_err(Error::Cuda)?;
        drop(weights);

        let capacity = kv_capacity.next_power_of_two();
        let gdn_states: Vec<model::GdnState> = (0..LAYERS - LAYERS / FULL_ATTENTION_INTERVAL)
            .map(|_| model::GdnState::new(&ctx))
            .collect();
        let kv_caches: Vec<model::KvCache> = (0..LAYERS / FULL_ATTENTION_INTERVAL)
            .map(|_| model::KvCache::new(&ctx, capacity))
            .collect();
        let scratch = model::Scratch::new(&ctx);

        Ok(Self {
            ctx,
            backend,
            model,
            vision,
            vision_state: vision::VisionState::default(),
            scratch,
            prefill: None,
            prefill_session: None,
            gdn_states,
            kv_caches,
            decode_phases: None,
            position: 0,
            rope_delta: 0,
            kv_capacity: capacity,
        })
    }

    fn ensure_graphs(&mut self) {
        if self.decode_phases.is_none() {
            self.decode_phases = Some(model_runner::capture_decode_phases(
                &self.ctx,
                &self.model,
                &mut self.scratch,
                &mut self.gdn_states,
            ));
        }
    }

    fn ensure_prefill_scratch(&mut self, tokens: usize) {
        let needed = tokens.div_ceil(CHUNK) * CHUNK;
        let enough = self
            .prefill
            .as_ref()
            .map(|p| p.capacity() >= needed)
            .unwrap_or(false);
        if !enough {
            self.prefill = Some(model::PrefillScratch::new(&self.ctx, needed));
            // Executions bind the scratch addresses, so a new scratch means
            // the prepared session no longer matches.
            self.prefill_session = None;
        }
        // Addresses are not the only binding: the prepared GEMMs carry the row
        // count of the traversal that prepared them, and `needed` rounds up to
        // CHUNK, so two different token counts share one scratch. Replaying
        // under a different count misses the execution cache, which a
        // non-preparation traversal cannot refill.
        if self
            .prefill_session
            .as_ref()
            .is_some_and(|(prepared, _)| *prepared != tokens)
        {
            self.prefill_session = None;
        }
    }

    /// Logits of the position decoded last, as a `[1, VOCAB]` BF16 tensor.
    fn logits_view(&self) -> Tensor {
        CudaBuffer::from_tensor(self.scratch.logits())
            .unwrap()
            .as_tensor(Shape::new(vec![1, VOCAB]), DType::BF16)
            .unwrap()
    }

    fn run_prefill(
        &mut self,
        token_ids: &[u32],
        positions: Option<&[[u32; 3]]>,
        prepared_embeddings: Option<&Tensor>,
    ) -> Result<Tensor> {
        validate_kv_capacity(0, token_ids.len(), self.kv_capacity)?;
        self.ensure_prefill_scratch(token_ids.len());
        self.ensure_graphs();
        let prefill = self.prefill.as_mut().unwrap();
        match positions {
            Some(positions) => {
                if positions.len() != token_ids.len() {
                    return Err(Error::Other(format!(
                        "qwen38 received {} mRoPE positions for {} tokens",
                        positions.len(),
                        token_ids.len()
                    )));
                }
                model_runner::write_tokens_with_positions(&self.ctx, prefill, token_ids, positions);
            }
            None => model_runner::write_tokens(&self.ctx, prefill, token_ids),
        }

        if let Some(source) = prepared_embeddings {
            if source.dtype() != DType::BF16
                || source.shape().dims() != [token_ids.len(), super::config::HIDDEN]
            {
                return Err(Error::Other(format!(
                    "qwen38 prepared embeddings must be BF16 [{}, {}], got {} {:?}",
                    token_ids.len(),
                    super::config::HIDDEN,
                    source.dtype(),
                    source.shape().dims()
                )));
            }
            let destination = prefill.hidden_rows(token_ids.len());
            ops::copy_tensor(&self.ctx, source, &destination)?;
        }

        let count = token_ids.len();
        let gather_embeddings = prepared_embeddings.is_none();
        let forward = |ctx: &CudaContext,
                       model: &weights::Model,
                       prefill: &mut model::PrefillScratch,
                       gdn_states: &mut Vec<model::GdnState>,
                       kv_caches: &mut Vec<model::KvCache>,
                       scratch: &model::Scratch| {
            model::prefill_step(
                ctx,
                model,
                prefill,
                gdn_states,
                kv_caches,
                count,
                gather_embeddings,
            );
            model::prefill_logits(ctx, model, prefill, scratch, count);
            Ok(())
        };
        if let Some((_, session)) = self.prefill_session.as_ref() {
            ops::with_session(session, || {
                forward(
                    &self.ctx,
                    &self.model,
                    prefill,
                    &mut self.gdn_states,
                    &mut self.kv_caches,
                    &self.scratch,
                )
            })?;
        } else {
            let session =
                ops::ExecutionSession::with_capacity(64 * 1024 * 1024, self.ctx.device_id())?;
            ops::prepare_with_session(&session, || {
                forward(
                    &self.ctx,
                    &self.model,
                    prefill,
                    &mut self.gdn_states,
                    &mut self.kv_caches,
                    &self.scratch,
                )
            })?;
            model_runner::report_session("prefill", &session);
            self.prefill_session = Some((count, session));
        }
        self.ctx.synchronize().map_err(Error::Cuda)?;
        self.position = token_ids.len();
        Ok(self.logits_view())
    }

    fn prefill_with_image(
        &mut self,
        token_ids: &[u32],
        pixel_values: &Tensor,
        grid_thw: &[[u32; 3]],
    ) -> Result<Tensor> {
        // Reject oversized multimodal prompts before running the expensive
        // vision tower. `run_prefill` repeats this guard so every prefill
        // caller shares the cache-boundary invariant.
        validate_kv_capacity(0, token_ids.len(), self.kv_capacity)?;
        if token_ids.len() <= 1 {
            return Err(Error::Other(
                "qwen38 image prefill requires a complete multimodal prompt".into(),
            ));
        }
        let (positions, embeddings, rope_delta) = {
            let vision_weights = self.vision.as_ref().ok_or_else(|| {
                Error::Other(
                    "qwen38 checkpoint does not contain model.visual weights required for image input"
                        .into(),
                )
            })?;
            let positions = vision::rope_positions(token_ids, grid_thw)?;
            let pixels = vision::upload_pixels(&self.ctx, pixel_values)?;
            let visual = vision::forward(
                vision_weights,
                &self.vision_state,
                &self.ctx,
                &pixels,
                grid_thw,
            )?;
            let embeddings =
                vision::embed_and_replace(&self.ctx, &self.model.embedding, token_ids, &visual)?;
            let max_position = positions
                .iter()
                .flat_map(|position| position.iter())
                .copied()
                .max()
                .unwrap_or(0) as i64;
            let rope_delta = max_position + 1 - token_ids.len() as i64;
            (positions, embeddings, rope_delta)
        };
        self.rope_delta = rope_delta;
        self.run_prefill(token_ids, Some(&positions), Some(&embeddings))
    }
}

impl LlmTrait for Qwen38 {
    fn load(config: ModelConfig, weights: HashMap<String, Tensor>, device: Device) -> Result<Self> {
        let capacity = config.max_seq_len.max(MIN_KV_CAPACITY);
        Self::new(weights, device, capacity)
    }

    /// `start_pos == 0` with several tokens runs the batched prefill; a single
    /// token continues decoding at `start_pos` through the captured graphs.
    /// Returns `[1, VOCAB]` logits for the last position — the shared
    /// generation loop samples with `NextTokenLogits::last`, which is exactly
    /// this row.
    fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Err(Error::Other("forward: empty token_ids".into()));
        }
        let start = start_pos as usize;
        validate_kv_capacity(start, token_ids.len(), self.kv_capacity)?;

        if start == 0 && token_ids.len() > 1 {
            self.rope_delta = 0;
            return self.run_prefill(token_ids, None, None);
        }

        // Token-by-token: the harness decode path, graphs captured lazily on
        // the first step.
        self.ensure_graphs();
        for (offset, &token) in token_ids.iter().enumerate() {
            let position = start + offset;
            model_runner::set_token(&self.scratch, token as i32);
            let mrope_position = position as i64 + self.rope_delta;
            if mrope_position < 0 || mrope_position > u32::MAX as i64 {
                return Err(Error::Other(format!(
                    "qwen38 mRoPE position {mrope_position} is out of range"
                )));
            }
            model_runner::set_mrope_position(&self.scratch, [mrope_position as u32; 3]);
            model_runner::decode_step_with_phases(
                &self.ctx,
                &self.model,
                &mut self.scratch,
                &mut self.gdn_states,
                &mut self.kv_caches,
                position,
                self.decode_phases.as_ref().unwrap(),
            );
        }
        self.ctx.synchronize().map_err(Error::Cuda)?;
        self.position = start + token_ids.len();
        Ok(self.logits_view())
    }

    fn backend(&self) -> &dyn Backend {
        &*self.backend
    }

    fn capabilities(&self) -> LlmCapabilities {
        if self.vision.is_some() {
            LlmCapabilities::VISION
        } else {
            LlmCapabilities::TEXT_ONLY
        }
    }

    fn prefill(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        match input.image {
            Some(image) => {
                self.prefill_with_image(input.token_ids, image.pixel_values, image.grid_thw)
            }
            None => self.forward(input.token_ids, 0),
        }
    }

    fn reset(&mut self) {
        model_runner::reset_state(&self.ctx, &mut self.gdn_states, &mut self.kv_caches);
        self.position = 0;
        self.rope_delta = 0;
    }

    fn vocab_size(&self) -> usize {
        VOCAB
    }
}

fn validate_kv_capacity(start: usize, len: usize, capacity: usize) -> Result<()> {
    let end = start
        .checked_add(len)
        .ok_or_else(|| Error::Other("qwen38 sequence length overflow".into()))?;
    if end > capacity {
        return Err(Error::Other(format!(
            "sequence {end} exceeds the KV capacity {capacity} fixed at load"
        )));
    }
    Ok(())
}

// Constructors the wrapper needs for state it owns per generation session.
impl model::GdnState {
    pub(crate) fn new(ctx: &CudaContext) -> Self {
        model::GdnState {
            recurrent: backend::zeros(
                ctx,
                vec![GDN_V_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM],
                DType::F32,
            ),
            conv_window: backend::zeros(ctx, vec![QKV_WIDTH, CONV_WIDTH], DType::F32),
        }
    }
}

impl model::KvCache {
    pub(crate) fn new(ctx: &CudaContext, capacity: usize) -> Self {
        model::KvCache {
            keys: backend::zeros(ctx, vec![1, capacity, KV_HEADS, HEAD_DIM], DType::BF16),
            values: backend::zeros(ctx, vec![1, capacity, KV_HEADS, HEAD_DIM], DType::BF16),
        }
    }
}

// Unused-import silencer for non-graph builds of ops.
#[allow(unused_imports)]
use ops as _ops_used;

#[cfg(test)]
mod tests {
    use super::validate_kv_capacity;

    #[test]
    fn kv_capacity_accepts_boundary_and_rejects_oversized_prefill() {
        assert!(validate_kv_capacity(0, 8192, 8192).is_ok());
        let error = validate_kv_capacity(0, 8193, 8192).unwrap_err();
        assert!(error
            .to_string()
            .contains("sequence 8193 exceeds the KV capacity 8192"));
    }

    #[test]
    fn kv_capacity_accounts_for_decode_start() {
        assert!(validate_kv_capacity(8191, 1, 8192).is_ok());
        assert!(validate_kv_capacity(8192, 1, 8192).is_err());
        assert!(validate_kv_capacity(usize::MAX, 1, usize::MAX).is_err());
    }
}
