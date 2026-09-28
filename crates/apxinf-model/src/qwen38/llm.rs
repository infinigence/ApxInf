//! The [`LlmTrait`] wrapper: state, execution policy, and the generation
//! surface over the split modules: [`super::weights`], [`super::model`], [`super::model_runner`].

use std::collections::HashMap;
use std::sync::Arc;

use apxinf_core::{Backend, Device, DType, Error, Result, Shape, Tensor};
use apxinf_cuda_new::{ops, CapturedGraph, CudaBuffer, CudaContext};
use apxinf_loader::ModelConfig;

use crate::accelerator::create_backend;
use crate::llm_trait::LlmTrait;

use super::config::{CHUNK, CONV_WIDTH, FULL_ATTENTION_INTERVAL, GDN_HEAD_DIM,
    GDN_V_HEADS, HEAD_DIM, KV_HEADS, LAYERS, QKV_WIDTH, VOCAB};
use super::{backend, model, model_runner, weights};

/// Qwen3.8-27B-NVFP4. Batch 1, text only, BF16 KV cache.
///
/// Execution policy is fixed: batched FlashInfer-GDN prefill, CUDA-graph
/// decode (per-layer GDN graphs + per-layer MLP graphs) and split-KV FA2
/// decode attention -- the validated configuration. The kernel harness keeps
/// its own switches for A/B comparison.
pub struct Qwen38 {
    ctx: CudaContext,
    backend: Arc<dyn Backend>,
    model: weights::Model,
    scratch: model::Scratch,
    prefill: Option<model::PrefillScratch>,
    prefill_session: Option<ops::ExecutionSession>,
    gdn_states: Vec<model::GdnState>,
    kv_caches: Vec<model::KvCache>,
    mlp_graphs: Option<Vec<CapturedGraph>>,
    gdn_graphs: Option<Vec<Option<CapturedGraph>>>,
    position: usize,
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
        for (key, value) in [
            // Tiled GDN prefill conv path, tile 64 (report 05/23).
            ("APXINF_GDN_PREFILL_TILED", "1"),
            ("APXINF_GDN_PREFILL_TILE", "64"),
            // cuBLASLt native-NVFP4 candidates for the block-scaled GEMMs
            // (report 06).
            ("APXINF_GEMM_CUBLASLT_NVFP4", "1"),
            // 16-byte vectorized elementwise kernels (report 08).
            ("APXINF_QWEN38_VECTOR_ELEMENTWISE", "1"),
            // Paired-channel conv decode kernel (report 23).
            ("APXINF_GDN_CONV_PAIR", "1"),
            // The shared-activation FP8 GEMV variant lost its A/B (report 25).
            ("APXINF_FP8_GEMV_SHARED", "0"),
            // Native fp8x2 pair quantization, part of the D3 contract
            // (reports 30/62).
            ("APXINF_FP8_NATIVE_PAIR", "1"),
            // Five-warp-group FlashInfer prepare reduction (report 11).
            ("APXINF_GDN_PREPARE_PARALLEL", "5"),
            // Single split for the split-KV FA2 decode kernel (report 47).
            ("APXINF_FA2_DECODE_SPLITS", "1"),
        ] {
            if std::env::var_os(key).is_none() {
                std::env::set_var(key, value);
            }
        }

        let ctx = CudaContext::new(ordinal).map_err(Error::Cuda)?;
        let backend = create_backend(device)?;
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
            scratch,
            prefill: None,
            prefill_session: None,
            gdn_states,
            kv_caches,
            mlp_graphs: None,
            gdn_graphs: None,
            position: 0,
            kv_capacity: capacity,
        })
    }

    fn ensure_graphs(&mut self) {
        if self.mlp_graphs.is_none() {
            self.mlp_graphs = Some(model_runner::capture_mlp_graphs(&self.ctx, &self.model, &mut self.scratch));
        }
        if self.gdn_graphs.is_none() {
            self.gdn_graphs = Some(model_runner::capture_gdn_graphs(
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
    }

    /// Logits of the position decoded last, as a `[1, VOCAB]` BF16 tensor.
    fn logits_view(&self) -> Tensor {
        CudaBuffer::from_tensor(self.scratch.logits())
            .unwrap()
            .as_tensor(Shape::new(vec![1, VOCAB]), DType::BF16)
            .unwrap()
    }
}

impl LlmTrait for Qwen38 {
    fn load(config: ModelConfig, weights: HashMap<String, Tensor>, device: Device) -> Result<Self> {
        let capacity = config.max_seq_len.max(4096);
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
        if start + token_ids.len() > self.kv_capacity {
            return Err(Error::Other(format!(
                "sequence {} exceeds the KV capacity {} fixed at load",
                start + token_ids.len(),
                self.kv_capacity
            )));
        }

        if start == 0 && token_ids.len() > 1 {
            self.ensure_prefill_scratch(token_ids.len());
            self.ensure_graphs();
            let prefill = self.prefill.as_mut().unwrap();
            model_runner::write_tokens(&self.ctx, prefill, token_ids);
            // First pass on a fresh scratch prepares (tunes, allocates) inside
            // a session arena; every later prefill replays those executions
            // without allocation. Same-address rebinding is what makes the
            // replay valid: scratch, weights, states and caches are stable
            // between calls, and a scratch reallocation clears the session.
            let count = token_ids.len();
            let forward = |ctx: &CudaContext,
                           model: &weights::Model,
                           prefill: &mut model::PrefillScratch,
                           gdn_states: &mut Vec<model::GdnState>,
                           kv_caches: &mut Vec<model::KvCache>,
                           scratch: &model::Scratch| {
                model::prefill_step(ctx, model, prefill, gdn_states, kv_caches, count);
                model::prefill_logits(ctx, model, prefill, scratch, count);
                Ok(())
            };
            if let Some(session) = self.prefill_session.as_ref() {
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
                self.prefill_session = Some(session);
            }
            self.ctx.synchronize().map_err(Error::Cuda)?;
            self.position = token_ids.len();
            return Ok(self.logits_view());
        }

        // Token-by-token: the harness decode path, graphs captured lazily on
        // the first step.
        self.ensure_graphs();
        for (offset, &token) in token_ids.iter().enumerate() {
            let position = start + offset;
            model_runner::set_token(&self.scratch, token as i32);
            model_runner::set_position(&self.scratch, position);
            model_runner::decode_step_with_mlp_graphs(
                &self.ctx,
                &self.model,
                &mut self.scratch,
                &mut self.gdn_states,
                &mut self.kv_caches,
                position,
                self.mlp_graphs.as_ref().unwrap(),
                self.gdn_graphs.as_deref(),
            );
        }
        self.ctx.synchronize().map_err(Error::Cuda)?;
        self.position = start + token_ids.len();
        Ok(self.logits_view())
    }

    fn backend(&self) -> &dyn Backend {
        &*self.backend
    }

    fn reset(&mut self) {
        model_runner::reset_state(&self.ctx, &mut self.gdn_states, &mut self.kv_caches);
        self.position = 0;
    }

    fn vocab_size(&self) -> usize {
        VOCAB
    }
}

// Constructors the wrapper needs for state it owns per generation session.
impl model::GdnState {
    pub(crate) fn new(ctx: &CudaContext) -> Self {
        model::GdnState {
            recurrent: backend::zeros(ctx, vec![GDN_V_HEADS, GDN_HEAD_DIM, GDN_HEAD_DIM], DType::F32),
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
