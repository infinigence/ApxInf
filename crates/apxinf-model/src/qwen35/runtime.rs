use super::config::Qwen35ModelConfig;

const PREFILL_CHUNK_TOKENS: usize = 512;
const MULTIMODAL_PREFILL_CHUNK_TOKENS: usize = 256;
const MAX_PREFILL_CHUNK_TOKENS: usize = 1024;

/// Tokens per bounded prefill block. Larger blocks amortize the per-block W4
/// dequantization and give cuBLAS a larger M, at the cost of a proportionally
/// larger attention score workspace (`heads x chunk x max_model_len x 4 B`).
/// Measured per-token GDN layer cost on the pinned checkpoint: 94.7 us at 64
/// rows, 40.9 us at 256, 27.5 us at 512. At `max_model_len=32768` the
/// attention workspace is 768 MB at chunk 256 and 1536 MB at chunk 512, both
/// inside the measured headroom (4606 MiB free with weights resident).
/// Admission charges this per request via `request_state_bytes`, so an
/// over-large chunk fails closed at startup rather than mid-request.
///
/// With the multimodal tower resident (~915 MiB BF16), free memory after
/// weights drops to ~3.6 GiB and the chunk-512 request estimate (~4.0 GiB)
/// no longer fits, so the default halves to 256 (~3.2 GiB estimate) in that
/// configuration; the text-only default is unchanged. Override with
/// `APXINF_Q35_PREFILL_CHUNK`; clamped to `1..=MAX_PREFILL_CHUNK_TOKENS` so
/// the workspace estimate stays bounded.
pub fn prefill_chunk_tokens() -> usize {
    let default = if multimodal_enabled() {
        MULTIMODAL_PREFILL_CHUNK_TOKENS
    } else {
        PREFILL_CHUNK_TOKENS
    };
    std::env::var("APXINF_Q35_PREFILL_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .map(|value| value.min(MAX_PREFILL_CHUNK_TOKENS))
        .unwrap_or(default)
}

fn prefill_ranges(prompt_tokens: usize) -> impl Iterator<Item = std::ops::Range<usize>> {
    let chunk = prefill_chunk_tokens();
    (0..prompt_tokens)
        .step_by(chunk)
        .map(move |start| start..prompt_tokens.min(start + chunk))
}

fn request_capacity(
    prompt_tokens: usize,
    max_new_tokens: usize,
    max_model_len: usize,
) -> Result<usize, String> {
    if prompt_tokens == 0 || max_new_tokens == 0 {
        return Err("Qwen3.5 request requires a non-empty prompt and positive budget".into());
    }
    let capacity = prompt_tokens
        .checked_add(max_new_tokens)
        .ok_or_else(|| "request token budget overflow".to_string())?;
    if capacity > max_model_len {
        return Err(format!(
            "request budget {capacity} exceeds max_model_len {max_model_len}"
        ));
    }
    Ok(capacity)
}

/// Bytes reserved by one request's mutable CUDA state and peak prefill
/// workspace. GDN keeps current, scratch, and backup buffers so a failed
/// speculative launch cannot discard the previous rollback handle. Attention
/// prefill is bounded to one 64-token block, so its score tensor scales with
/// the retained KV length rather than the full prompt squared.
pub fn request_state_bytes(
    config: &Qwen35ModelConfig,
    max_model_len: usize,
) -> Result<usize, String> {
    if max_model_len == 0 {
        return Err("max_model_len must be non-zero".into());
    }
    let gdn_channels = config
        .linear_key_heads
        .checked_mul(config.linear_head_dim)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| {
            value.checked_add(
                config
                    .linear_value_heads
                    .checked_mul(config.linear_head_dim)?,
            )
        })
        .ok_or_else(|| "GDN convolution dimension overflow".to_string())?;
    let conv_bytes = gdn_channels
        .checked_mul(config.linear_conv_kernel_dim)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| "GDN convolution byte estimate overflow".to_string())?;
    let recurrent_bytes = config
        .linear_value_heads
        .checked_mul(config.linear_head_dim)
        .and_then(|value| value.checked_mul(config.linear_head_dim))
        .and_then(|value| value.checked_mul(4))
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| "GDN recurrent byte estimate overflow".to_string())?;
    let gdn_per_layer = conv_bytes
        .checked_add(recurrent_bytes)
        .ok_or_else(|| "GDN byte estimate overflow".to_string())?;
    let attention_per_layer = config
        .full_attention_kv_heads
        .checked_mul(max_model_len)
        .and_then(|value| value.checked_mul(config.full_attention_head_dim))
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| "attention KV byte estimate overflow".to_string())?;
    // The GDN sequence path always works in fixed 64-token chunks internally,
    // independent of the prefill block size.
    let chunk = 64usize;
    let qk = chunk
        .checked_mul(config.linear_head_dim)
        .ok_or_else(|| "GDN prefill workspace overflow".to_string())?;
    let values = chunk
        .checked_mul(config.linear_head_dim)
        .ok_or_else(|| "GDN prefill workspace overflow".to_string())?;
    let matrix = chunk
        .checked_mul(chunk)
        .ok_or_else(|| "GDN prefill workspace overflow".to_string())?;
    let workspace_floats = qk
        .checked_mul(2)
        .and_then(|value| value.checked_add(values))
        .and_then(|value| value.checked_add(chunk.checked_mul(2)?))
        .and_then(|value| value.checked_add(matrix))
        .and_then(|value| value.checked_add(values))
        .and_then(|value| value.checked_add(qk))
        .and_then(|value| value.checked_add(values))
        .ok_or_else(|| "GDN prefill workspace overflow".to_string())?;
    let gdn_prefill_workspace = config
        .linear_value_heads
        .checked_mul(workspace_floats)
        .and_then(|value| value.checked_mul(4))
        .ok_or_else(|| "GDN prefill workspace byte estimate overflow".to_string())?;
    let attention_chunk = prefill_chunk_tokens().min(max_model_len);
    let attention_prefill_workspace = config
        .full_attention_heads
        .checked_mul(attention_chunk)
        .and_then(|value| value.checked_mul(max_model_len))
        .and_then(|value| value.checked_mul(2 * 2))
        .ok_or_else(|| "attention prefill workspace byte estimate overflow".to_string())?;
    config
        .gdn_layer_count()
        .checked_mul(gdn_per_layer)
        .and_then(|value| {
            value.checked_add(
                config
                    .full_attention_layer_count()
                    .checked_mul(attention_per_layer)?,
            )
        })
        .and_then(|value| value.checked_add(gdn_prefill_workspace))
        .and_then(|value| value.checked_add(attention_prefill_workspace))
        .ok_or_else(|| "request state byte estimate overflow".to_string())
}

/// Per-request resident bytes for a request of `capacity = prompt +
/// max_new_tokens` tokens: its full-attention KV pages plus the GDN
/// convolution/recurrent state. Excludes the transient prefill workspaces,
/// which the concurrent runtime sizes once (prefill is serialized on the
/// worker) and subtracts from the admission budget up front. The serial
/// runtime keeps using the conservative `request_state_bytes` instead.
pub fn request_resident_bytes(
    config: &Qwen35ModelConfig,
    capacity: usize,
) -> Result<usize, String> {
    if capacity == 0 {
        return Err("request capacity must be non-zero".into());
    }
    let kv_bytes = config
        .full_attention_layer_count()
        .checked_mul(2)
        .and_then(|value| value.checked_mul(config.full_attention_kv_heads))
        .and_then(|value| value.checked_mul(config.full_attention_head_dim))
        .and_then(|value| value.checked_mul(capacity))
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| "KV byte estimate overflow".to_string())?;
    let gdn_channels = config
        .linear_key_heads
        .checked_mul(config.linear_head_dim)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| {
            value.checked_add(
                config
                    .linear_value_heads
                    .checked_mul(config.linear_head_dim)?,
            )
        })
        .ok_or_else(|| "GDN convolution dimension overflow".to_string())?;
    let conv_bytes = gdn_channels
        .checked_mul(config.linear_conv_kernel_dim)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| "GDN convolution byte estimate overflow".to_string())?;
    let recurrent_bytes = config
        .linear_value_heads
        .checked_mul(config.linear_head_dim)
        .and_then(|value| value.checked_mul(config.linear_head_dim))
        .and_then(|value| value.checked_mul(4))
        .and_then(|value| value.checked_mul(3))
        .ok_or_else(|| "GDN recurrent byte estimate overflow".to_string())?;
    config
        .gdn_layer_count()
        .checked_mul(
            conv_bytes
                .checked_add(recurrent_bytes)
                .ok_or_else(|| "GDN byte estimate overflow".to_string())?,
        )
        .and_then(|value| value.checked_add(kv_bytes))
        .ok_or_else(|| "request resident byte estimate overflow".to_string())
}

/// Whether the multimodal capability is enabled for this process. Off by
/// default: the vision tower is not loaded, `/v1/chat/completions` fails
/// closed, and the text configuration is byte-identical to the text-only
/// build. Flip with `APXINF_ENABLE_MULTIMODAL=1` only after the image suite
/// passes.
pub fn multimodal_enabled() -> bool {
    std::env::var("APXINF_ENABLE_MULTIMODAL").is_ok_and(|value| value == "1")
}

#[cfg(feature = "cuda")]
mod cuda_runtime {
    use std::sync::Arc;

    use apxinf_core::{Backend, Shape, Tensor};
    use apxinf_cuda::CudaBackend;

    use super::super::config::{LayerType, Qwen35ModelConfig};
    use super::super::cuda::{
        upload_standard_norm_payload, Qwen35Bf16Projection, Qwen35CudaFullAttentionLayer,
        Qwen35CudaFullAttentionState, Qwen35CudaGdnLayer, Qwen35CudaGdnState,
    };
    use super::super::loader::Qwen35CheckpointInventory;
    use super::super::model::greedy_argmax;
    use super::super::vision::Qwen35VisionTower;
    use crate::runtime::MultimodalPayload;

    enum Layer {
        Gdn(Qwen35CudaGdnLayer),
        Full(Qwen35CudaFullAttentionLayer),
    }

    enum State {
        Gdn(Qwen35CudaGdnState),
        Full(Qwen35CudaFullAttentionState),
    }

    /// Image-embedding rows to write over the text embeddings of the
    /// `<|image_pad|>` positions during prefill. Rows are host BF16 bytes in
    /// prompt order (`positions[i]` receives row `i`).
    struct ImageScatter {
        positions: Vec<usize>,
        row_bytes: Vec<u8>,
        row_stride: usize,
    }

    /// Resident CUDA weights for the complete Qwen3.5 text stack.
    pub struct Qwen35CudaModel {
        backend: CudaBackend,
        config: Qwen35ModelConfig,
        max_model_len: usize,
        embedding: Tensor,
        final_norm: Tensor,
        lm_head: Qwen35Bf16Projection,
        layers: Vec<Layer>,
        /// Loaded only when `APXINF_ENABLE_MULTIMODAL=1` (~880 MiB BF16).
        vision: Option<Qwen35VisionTower>,
    }

    impl Qwen35CudaModel {
        pub fn from_inventory(
            inventory: &Qwen35CheckpointInventory,
            device_id: usize,
            max_model_len: usize,
        ) -> Result<Arc<Self>, String> {
            let backend = CudaBackend::new(device_id).map_err(|error| error.to_string())?;
            Self::from_backend(inventory, backend, max_model_len)
        }

        pub fn from_inventory_attested(
            inventory: &Qwen35CheckpointInventory,
            device_id: usize,
            expected_uuid: &str,
            max_model_len: usize,
        ) -> Result<Arc<Self>, String> {
            let backend = CudaBackend::new_attested(device_id, expected_uuid)
                .map_err(|error| error.to_string())?;
            Self::from_backend(inventory, backend, max_model_len)
        }

        fn from_backend(
            inventory: &Qwen35CheckpointInventory,
            backend: CudaBackend,
            max_model_len: usize,
        ) -> Result<Arc<Self>, String> {
            if max_model_len == 0 || max_model_len > inventory.config.max_position_embeddings {
                return Err(format!(
                    "max_model_len {max_model_len} is outside checkpoint range [1, {}]",
                    inventory.config.max_position_embeddings
                ));
            }
            let device_id = backend.device_id();
            let ctx = backend.context();
            let config = inventory.config.clone();

            let embedding_payload = inventory
                .read_bf16_tensor_payload(
                    "model.language_model.embed_tokens.weight",
                    &[config.vocab_size, config.hidden_size],
                )
                .map_err(|error| error.to_string())?;
            let embedding_host = Tensor::from_bf16(
                Shape::new(embedding_payload.shape.clone()),
                &embedding_payload.values,
            )
            .map_err(|error| format!("create embedding tensor: {error}"))?;
            let embedding = apxinf_cuda::transfers::to_cuda(&embedding_host, device_id)
                .map_err(|error| format!("upload embedding tensor: {error}"))?;

            let final_norm_payload = inventory
                .read_bf16_tensor_payload("model.language_model.norm.weight", &[config.hidden_size])
                .map_err(|error| error.to_string())?;
            let final_norm = upload_standard_norm_payload(
                ctx,
                &final_norm_payload,
                config.hidden_size,
                "final norm",
            )?;
            let lm_payload = inventory
                .read_bf16_tensor_payload(
                    "lm_head.weight",
                    &[config.vocab_size, config.hidden_size],
                )
                .map_err(|error| error.to_string())?;
            let lm_head = Qwen35Bf16Projection::from_payload(ctx, &lm_payload)?;

            let mut layers = Vec::with_capacity(config.num_hidden_layers);
            for layer_index in 0..config.num_hidden_layers {
                let layer = match config.layer_types[layer_index] {
                    LayerType::Gdn => Layer::Gdn(Qwen35CudaGdnLayer::from_inventory(
                        ctx,
                        inventory,
                        layer_index,
                    )?),
                    LayerType::FullAttention => Layer::Full(
                        Qwen35CudaFullAttentionLayer::from_inventory(ctx, inventory, layer_index)?,
                    ),
                };
                layers.push(layer);
            }
            let vision = if super::multimodal_enabled() {
                eprintln!("qwen35: loading vision tower (multimodal enabled)");
                Some(Qwen35VisionTower::from_inventory(&backend, inventory)?)
            } else {
                None
            };
            Ok(Arc::new(Self {
                backend,
                config,
                max_model_len,
                embedding,
                final_norm,
                lm_head,
                layers,
                vision,
            }))
        }

        pub fn device_id(&self) -> usize {
            self.backend.device_id()
        }

        pub fn config(&self) -> &Qwen35ModelConfig {
            &self.config
        }

        pub fn max_model_len(&self) -> usize {
            self.max_model_len
        }

        pub fn memory_info(&self) -> Result<apxinf_cuda::CudaMemoryInfo, String> {
            self.backend.context().memory_info()
        }

        pub fn open(
            self: &Arc<Self>,
            input_ids: &[u32],
            max_new_tokens: usize,
        ) -> Result<Qwen35CudaSession, String> {
            self.open_with_cancel(
                input_ids,
                max_new_tokens,
                &crate::runtime::CancellationToken::new(),
            )
        }

        /// Open a session whose prefill can be aborted between bounded
        /// 64-token blocks. A long prompt otherwise pins the GPU worker (and
        /// the single capacity slot) for the full prefill even after the
        /// client has disconnected.
        pub fn open_with_cancel(
            self: &Arc<Self>,
            input_ids: &[u32],
            max_new_tokens: usize,
            cancel: &crate::runtime::CancellationToken,
        ) -> Result<Qwen35CudaSession, String> {
            self.open_with_cancel_multimodal(input_ids, max_new_tokens, cancel, None)
        }

        /// Open a session, optionally scattering a vision embedding over the
        /// `<|image_pad|>` positions. The vision tower forward runs here, on
        /// the caller's (GPU worker) thread, so image work is serialized with
        /// text work on the single device.
        pub fn open_with_cancel_multimodal(
            self: &Arc<Self>,
            input_ids: &[u32],
            max_new_tokens: usize,
            cancel: &crate::runtime::CancellationToken,
            multimodal: Option<&MultimodalPayload>,
        ) -> Result<Qwen35CudaSession, String> {
            let request_capacity =
                super::request_capacity(input_ids.len(), max_new_tokens, self.max_model_len)?;
            if let Some(token_id) = input_ids
                .iter()
                .copied()
                .find(|token| (*token as usize) >= self.config.vocab_size)
            {
                return Err(format!("token id {token_id} is outside model vocabulary"));
            }
            let image = match multimodal {
                None => None,
                Some(payload) => Some(self.prepare_image_scatter(input_ids, payload)?),
            };
            let states = self.new_states(request_capacity)?;
            let mut session = Qwen35CudaSession {
                model: Arc::clone(self),
                states,
                position: 0,
                emitted: 0,
                max_new_tokens,
                pending: None,
                last_input: None,
                last_hidden: None,
                #[cfg(test)]
                forward_token_calls: 0,
                #[cfg(test)]
                prefill_calls: 0,
            };
            session.prefill(input_ids, cancel, image.as_ref())?;
            Ok(session)
        }

        /// Run the vision tower and pair each merged embedding row with the
        /// prompt position of the corresponding `<|image_pad|>` token.
        fn prepare_image_scatter(
            &self,
            input_ids: &[u32],
            payload: &MultimodalPayload,
        ) -> Result<ImageScatter, String> {
            let vision = self.vision.as_ref().ok_or_else(|| {
                "multimodal request received but the vision tower is not loaded \
                 (APXINF_ENABLE_MULTIMODAL is off)"
                    .to_string()
            })?;
            let merged = vision.forward(&self.backend, &payload.pixel_values, payload.grid)?;
            let dims = merged.shape().dims().to_vec();
            if dims.len() != 2 || dims[1] != self.config.hidden_size {
                return Err(format!(
                    "vision embedding shape {dims:?} does not match hidden size {}",
                    self.config.hidden_size
                ));
            }
            let positions: Vec<usize> = input_ids
                .iter()
                .enumerate()
                .filter(|(_, token)| **token == self.config.image_token_id)
                .map(|(index, _)| index)
                .collect();
            if positions.len() != dims[0] {
                return Err(format!(
                    "prompt has {} image tokens but the vision tower produced {} rows",
                    positions.len(),
                    dims[0]
                ));
            }
            let host = self
                .backend
                .to_cpu(&merged)
                .map_err(|error| format!("vision embedding download: {error}"))?;
            let values = host
                .as_bf16()
                .map_err(|error| format!("vision embedding dtype: {error}"))?;
            let row_stride = dims[1] * std::mem::size_of::<half::bf16>();
            let mut row_bytes = Vec::with_capacity(values.len() * 2);
            for value in values.iter() {
                row_bytes.extend_from_slice(&value.to_le_bytes());
            }
            Ok(ImageScatter {
                positions,
                row_bytes,
                row_stride,
            })
        }

        fn new_states(&self, request_capacity: usize) -> Result<Vec<State>, String> {
            self.layers
                .iter()
                .map(|layer| match layer {
                    Layer::Gdn(layer) => Ok(State::Gdn(Qwen35CudaGdnState::new(
                        &self.backend,
                        layer.dimensions(),
                    )?)),
                    Layer::Full(_) => Ok(State::Full(Qwen35CudaFullAttentionState::new(
                        &self.backend,
                        request_capacity,
                    )?)),
                })
                .collect()
        }
    }

    pub struct Qwen35CudaSession {
        model: Arc<Qwen35CudaModel>,
        states: Vec<State>,
        position: usize,
        emitted: usize,
        max_new_tokens: usize,
        pending: Option<u32>,
        last_input: Option<u32>,
        last_hidden: Option<Tensor>,
        #[cfg(test)]
        forward_token_calls: usize,
        #[cfg(test)]
        prefill_calls: usize,
    }

    /// Advance several independent sessions by one decode step as a single
    /// batched forward. Each session must be "drained": its previous token
    /// already consumed (`pending == None`), a last input available, and
    /// budget remaining — the scheduler guarantees this. After the call every
    /// session holds its next token in `pending`, exactly as if
    /// `next_token`'s internal `forward_token` had run serially; the batched
    /// layer kernels are per-row bit-identical to the serial path.
    ///
    /// On error the affected sessions are left in an indeterminate state and
    /// must all be dropped by the caller.
    pub fn decode_step_batch(sessions: &mut [&mut Qwen35CudaSession]) -> Result<(), String> {
        let batch = sessions.len();
        if batch == 0 {
            return Err("batched decode requires at least one session".into());
        }
        let model = Arc::clone(&sessions[0].model);
        let mut tokens = Vec::with_capacity(batch);
        for (index, session) in sessions.iter().enumerate() {
            if !Arc::ptr_eq(&session.model, &model) {
                return Err(format!("session {index} belongs to a different model"));
            }
            if session.pending.is_some() {
                return Err(format!("session {index} has an unconsumed pending token"));
            }
            if session.emitted >= session.max_new_tokens {
                return Err(format!("session {index} has exhausted its budget"));
            }
            if session.position >= model.max_model_len {
                return Err(format!("session {index} exceeds max_model_len"));
            }
            tokens.push(
                session
                    .last_input
                    .ok_or_else(|| format!("session {index} has no previous input"))?,
            );
        }

        // A batch of one gains nothing from the batched plumbing (pointer
        // array uploads, row stacks) and pays ~10% TPOT for it (measured on
        // the 16K single-request cell). Run the serial step instead; the
        // numerics are identical either way.
        if batch == 1 {
            let session = &mut *sessions[0];
            session.forward_token(tokens[0])?;
            let hidden = session
                .last_hidden
                .as_ref()
                .ok_or_else(|| "decode produced no hidden state".to_string())?;
            let logits = session.logits_from_hidden(hidden)?;
            session.pending = Some(
                greedy_argmax(&logits, model.config.vocab_size)
                    .map_err(|error| error.to_string())?,
            );
            return Ok(());
        }

        let debug_timing = std::env::var("APXINF_Q35_BATCH_DEBUG").is_ok_and(|v| v == "1");
        let step_start = std::time::Instant::now();
        let mut gdn_total = std::time::Duration::ZERO;
        let mut attention_total = std::time::Duration::ZERO;
        let mut hidden = model
            .backend
            .embedding(&model.embedding, &tokens)
            .map_err(|error| format!("batched embedding failed: {error}"))?;
        for (layer_index, layer) in model.layers.iter().enumerate() {
            let layer_start = std::time::Instant::now();
            hidden = match layer {
                Layer::Gdn(layer) => {
                    let mut states: Vec<&mut Qwen35CudaGdnState> = Vec::with_capacity(batch);
                    for session in sessions.iter_mut() {
                        match &mut session.states[layer_index] {
                            State::Gdn(state) => states.push(state),
                            State::Full(_) => {
                                return Err(format!(
                                    "layer {layer_index} state type mismatch in batch"
                                ))
                            }
                        }
                    }
                    layer
                        .decode_token_batch(&model.backend, &hidden, &mut states)
                        .map_err(|error| {
                            format!("GDN layer {layer_index} batched decode failed: {error}")
                        })?
                }
                Layer::Full(layer) => {
                    let positions: Vec<usize> =
                        sessions.iter().map(|session| session.position).collect();
                    let mut states: Vec<&mut Qwen35CudaFullAttentionState> =
                        Vec::with_capacity(batch);
                    for session in sessions.iter_mut() {
                        match &mut session.states[layer_index] {
                            State::Full(state) => states.push(state),
                            State::Gdn(_) => {
                                return Err(format!(
                                    "layer {layer_index} state type mismatch in batch"
                                ))
                            }
                        }
                    }
                    layer
                        .decode_token_batch(&model.backend, &hidden, &positions, &mut states)
                        .map_err(|error| {
                            format!(
                                "full-attention layer {layer_index} batched decode failed: {error}"
                            )
                        })?
                }
            };
            if debug_timing {
                match layer {
                    Layer::Gdn(_) => gdn_total += layer_start.elapsed(),
                    Layer::Full(_) => attention_total += layer_start.elapsed(),
                }
            }
        }
        apxinf_cuda::kernels::qwen35_gdn::drain_deferred_status(
            model.backend.context(),
            "batched decode step",
        )
        .map_err(|error| error.to_string())?;

        // Batched LM head: one final RMSNorm and one [M, vocab] cuBLAS GEMM
        // instead of M serial GEMV projections. The GEMM's per-row
        // accumulation order differs from the serial GEMV (the usual cuBLAS
        // reassociation), which can flip only exactly-tied logits — the same
        // accepted trade as the prefill GEMM path.
        let logits_start = std::time::Instant::now();
        let normalized = model
            .backend
            .rms_norm(&hidden, &model.final_norm, model.config.rms_norm_eps)
            .map_err(|error| format!("batched final RMSNorm failed: {error}"))?;
        let logits_gpu = model
            .lm_head
            .project(model.backend.context(), &normalized)
            .map_err(|error| format!("batched LM head failed: {error}"))?;
        let logits_cpu = model
            .backend
            .to_cpu(&logits_gpu)
            .map_err(|error| format!("batched logits copy failed: {error}"))?;
        let logits = logits_cpu
            .to_f32_vec()
            .map_err(|error| format!("batched logits decode failed: {error}"))?;
        let vocab = model.config.vocab_size;
        for (index, session) in sessions.iter_mut().enumerate() {
            let row = row_slice(&model.backend, &hidden, index)?;
            session.pending = Some(
                greedy_argmax(&logits[index * vocab..(index + 1) * vocab], vocab)
                    .map_err(|error| error.to_string())?,
            );
            session.position += 1;
            session.last_hidden = Some(row);
            #[cfg(test)]
            {
                session.forward_token_calls += 1;
            }
        }
        if debug_timing {
            eprintln!(
                "[batch step] batch={batch} total={:.1}ms gdn={:.1}ms attn={:.1}ms logits={:.1}ms",
                step_start.elapsed().as_secs_f64() * 1e3,
                gdn_total.as_secs_f64() * 1e3,
                attention_total.as_secs_f64() * 1e3,
                logits_start.elapsed().as_secs_f64() * 1e3,
            );
        }
        Ok(())
    }

    impl Qwen35CudaSession {
        fn prefill(
            &mut self,
            input_ids: &[u32],
            cancel: &crate::runtime::CancellationToken,
            image: Option<&ImageScatter>,
        ) -> Result<(), String> {
            if input_ids.is_empty() {
                return Err("prefill requires at least one token".into());
            }
            if self.position != 0 {
                return Err("prefill session must start at position zero".into());
            }
            let mut final_row = None;
            for range in super::prefill_ranges(input_ids.len()) {
                if cancel.is_cancelled() {
                    return Err(format!(
                        "prefill cancelled by client disconnect at token {} of {}",
                        range.start,
                        input_ids.len()
                    ));
                }
                let position = range.start;
                let chunk_ids = &input_ids[range.clone()];
                let mut hidden = self
                    .model
                    .backend
                    .embedding(&self.model.embedding, chunk_ids)
                    .map_err(|error| format!("embedding prefill failed: {error}"))?;
                if let Some(image) = image {
                    scatter_image_rows(&hidden, image, range.clone())?;
                }
                if std::env::var_os("APXINF_DEBUG_HIDDEN_DIR").is_some() {
                    for row in 0..chunk_ids.len() {
                        let row_hidden = row_slice(&self.model.backend, &hidden, row)?;
                        debug_capture_hidden(
                            &self.model.backend,
                            &row_hidden,
                            "embedding",
                            position + row,
                        )?;
                    }
                }
                for (index, layer) in self.model.layers.iter().enumerate() {
                    hidden = match (layer, &mut self.states[index]) {
                        (Layer::Gdn(layer), State::Gdn(state)) => {
                            if state.position() != position {
                                return Err(format!(
                                    "GDN layer {index} position {} does not match prefill block {position}",
                                    state.position()
                                ));
                            }
                            layer
                                .prefill(&self.model.backend, &hidden, state)
                                .map_err(|error| {
                                    format!("GDN layer {index} prefill failed: {error}")
                                })?
                        }
                        (Layer::Full(layer), State::Full(state)) => layer
                            .prefill(&self.model.backend, &hidden, position, state)
                            .map_err(|error| {
                                format!("full-attention layer {index} prefill failed: {error}")
                            })?,
                        _ => return Err(format!("layer {index} and state type disagree")),
                    };
                    if std::env::var_os("APXINF_DEBUG_HIDDEN_DIR").is_some() {
                        for row in 0..chunk_ids.len() {
                            let row_hidden = row_slice(&self.model.backend, &hidden, row)?;
                            debug_capture_hidden(
                                &self.model.backend,
                                &row_hidden,
                                &format!("layer-{index:03}"),
                                position + row,
                            )?;
                        }
                    }
                }
                // In deferred-status mode this is the once-per-block
                // synchronize that surfaces any latched non-finite flag from
                // the 64 layers above; in eager mode it is a no-op.
                apxinf_cuda::kernels::qwen35_gdn::drain_deferred_status(
                    self.model.backend.context(),
                    "GDN/attention prefill block",
                )
                .map_err(|error| error.to_string())?;
                if range.end == input_ids.len() {
                    final_row = Some(row_slice(
                        &self.model.backend,
                        &hidden,
                        chunk_ids.len() - 1,
                    )?);
                }
                #[cfg(test)]
                {
                    self.prefill_calls += 1;
                }
            }
            let final_row = final_row.ok_or_else(|| "prefill produced no final row".to_string())?;
            let logits = self.logits_from_hidden(&final_row)?;
            self.pending = Some(
                greedy_argmax(&logits, self.model.config.vocab_size)
                    .map_err(|error| error.to_string())?,
            );
            self.position = input_ids.len();
            self.last_input = input_ids.last().copied();
            self.last_hidden = Some(final_row);
            Ok(())
        }

        /// Fork this session into an independent copy with bit-identical
        /// device state (KV pages, GDN convolution/recurrent state, pending
        /// token, hidden row). Greedy decoding is deterministic, so a fork
        /// taken right after prefill reproduces the original's exact output
        /// sequence — this is the prefix-cache fast path for repeated
        /// prompts. Cost is one device-to-device copy of the request state
        /// (~0.5 GiB, ~1-2 ms).
        pub fn fork(&self) -> Result<Qwen35CudaSession, String> {
            let backend = &self.model.backend;
            let states = self
                .states
                .iter()
                .map(|state| match state {
                    State::Gdn(state) => state.deep_clone(backend).map(State::Gdn),
                    State::Full(state) => state.deep_clone(backend).map(State::Full),
                })
                .collect::<Result<Vec<_>, String>>()?;
            let last_hidden = match &self.last_hidden {
                None => None,
                Some(hidden) => {
                    let dims = hidden.shape().dims();
                    if dims.len() != 2 || dims[0] != 1 {
                        return Err(format!("fork expects a [1, H] hidden row, got {dims:?}"));
                    }
                    Some(
                        apxinf_cuda::kernels::elementwise::slice_columns_bf16(
                            self.model.backend.context(),
                            hidden,
                            0,
                            dims[1],
                        )
                        .map_err(|error| format!("fork hidden copy: {error}"))?,
                    )
                }
            };
            // No explicit synchronize: every copy above is stream-ordered on
            // the context stream, and every future consumer of the cloned
            // state (batched kernels, further copies) runs on that same
            // stream, so ordering is guaranteed. A synchronize here would
            // stall admissions behind whichever batch step is in flight
            // (~200 ms measured), inflating the co-running requests' TPOT.
            Ok(Qwen35CudaSession {
                model: Arc::clone(&self.model),
                states,
                position: self.position,
                emitted: self.emitted,
                max_new_tokens: self.max_new_tokens,
                pending: self.pending,
                last_input: self.last_input,
                last_hidden,
                #[cfg(test)]
                forward_token_calls: 0,
                #[cfg(test)]
                prefill_calls: 0,
            })
        }

        /// Refill this session in place from `template`, reusing every device
        /// allocation (the recycled-session fast path: a fresh `fork` pays
        /// ~320 cudaMallocs, a refill only stream-ordered copies). Requires
        /// the same model and state geometry; budget and bookkeeping are
        /// copied so the refilled session is equivalent to a fresh fork.
        pub fn refill_from(&mut self, template: &Qwen35CudaSession) -> Result<(), String> {
            if !Arc::ptr_eq(&self.model, &template.model) {
                return Err("session refill requires the same model".into());
            }
            if self.states.len() != template.states.len() {
                return Err("session refill requires the same layer count".into());
            }
            let backend = &self.model.backend;
            for (destination, source) in self.states.iter_mut().zip(template.states.iter()) {
                match (destination, source) {
                    (State::Gdn(destination), State::Gdn(source)) => {
                        destination.copy_state_from(source, backend)?;
                    }
                    (State::Full(destination), State::Full(source)) => {
                        destination.copy_state_from(source, backend)?;
                    }
                    _ => return Err("session refill state type mismatch".into()),
                }
            }
            let last_hidden = match &template.last_hidden {
                None => None,
                Some(hidden) => {
                    let dims = hidden.shape().dims();
                    Some(
                        apxinf_cuda::kernels::elementwise::slice_columns_bf16(
                            backend.context(),
                            hidden,
                            0,
                            dims[1],
                        )
                        .map_err(|error| format!("refill hidden copy: {error}"))?,
                    )
                }
            };
            // Stream-ordered copies need no synchronize; see `fork`.
            self.position = template.position;
            self.emitted = template.emitted;
            self.max_new_tokens = template.max_new_tokens;
            self.pending = template.pending;
            self.last_input = template.last_input;
            self.last_hidden = last_hidden;
            Ok(())
        }

        /// Batch-mode consumption: pop the token produced by prefill or the
        /// last batched step and account for it, without ever running a
        /// serial forward. Returns `None` once the budget is exhausted.
        pub fn take_pending_token(&mut self) -> Option<u32> {
            if self.emitted >= self.max_new_tokens {
                return None;
            }
            let token = self.pending.take()?;
            self.last_input = Some(token);
            self.emitted += 1;
            Some(token)
        }

        /// True when the session owes more tokens and the next one has not
        /// been computed yet, i.e. it should join the next batched step.
        pub fn needs_batched_step(&self) -> bool {
            self.pending.is_none() && self.emitted < self.max_new_tokens
        }

        pub fn next_token(&mut self) -> Result<Option<u32>, String> {
            if self.emitted >= self.max_new_tokens {
                return Ok(None);
            }
            if self.pending.is_none() {
                let token = self
                    .last_input
                    .ok_or_else(|| "decode session has no previous input".to_string())?;
                self.forward_token(token)?;
                let hidden = self
                    .last_hidden
                    .as_ref()
                    .ok_or_else(|| "decode produced no hidden state".to_string())?;
                let logits = self.logits_from_hidden(hidden)?;
                self.pending = Some(
                    greedy_argmax(&logits, self.model.config.vocab_size)
                        .map_err(|error| error.to_string())?,
                );
            }
            let token = self.pending.take().unwrap();
            self.last_input = Some(token);
            self.emitted += 1;
            Ok(Some(token))
        }

        fn forward_token(&mut self, token: u32) -> Result<(), String> {
            #[cfg(test)]
            {
                self.forward_token_calls += 1;
            }
            if self.position >= self.model.max_model_len {
                return Err("decode position exceeds max_model_len".into());
            }
            let mut hidden = self
                .model
                .backend
                .embedding(&self.model.embedding, &[token])
                .map_err(|error| format!("embedding step failed: {error}"))?;
            debug_capture_hidden(&self.model.backend, &hidden, "embedding", self.position)?;
            for (index, layer) in self.model.layers.iter().enumerate() {
                hidden = match (layer, &mut self.states[index]) {
                    (Layer::Gdn(layer), State::Gdn(state)) => layer
                        .decode_token(&self.model.backend, &hidden, state)
                        .map_err(|error| format!("GDN layer {index} failed: {error}"))?,
                    (Layer::Full(layer), State::Full(state)) => layer
                        .decode_token(&self.model.backend, &hidden, self.position, state)
                        .map_err(|error| format!("full-attention layer {index} failed: {error}"))?,
                    _ => return Err(format!("layer {index} and state type disagree")),
                };
                debug_capture_hidden(
                    &self.model.backend,
                    &hidden,
                    &format!("layer-{index:03}"),
                    self.position,
                )?;
            }
            // Deferred-status drain: one synchronize per decoded token
            // instead of ~4 per GDN layer; no-op in eager mode.
            apxinf_cuda::kernels::qwen35_gdn::drain_deferred_status(
                self.model.backend.context(),
                "GDN/attention decode token",
            )
            .map_err(|error| error.to_string())?;
            self.position += 1;
            self.last_input = Some(token);
            self.last_hidden = Some(hidden);
            Ok(())
        }

        fn logits_from_hidden(&self, hidden: &Tensor) -> Result<Vec<f32>, String> {
            let normalized = self
                .model
                .backend
                .rms_norm(
                    hidden,
                    &self.model.final_norm,
                    self.model.config.rms_norm_eps,
                )
                .map_err(|error| format!("final RMSNorm failed: {error}"))?;
            let logits = self
                .model
                .lm_head
                .project(self.model.backend.context(), &normalized)
                .map_err(|error| format!("LM head projection failed: {error}"))?;
            let logits = self
                .model
                .backend
                .to_cpu(&logits)
                .map_err(|error| format!("copy logits to host failed: {error}"))?;
            let values = logits
                .to_f32_vec()
                .map_err(|error| format!("decode logits failed: {error}"))?;
            debug_capture_logits(&values, self.position);
            Ok(values)
        }
    }

    /// Overwrite the embedding rows of image-pad tokens that fall inside the
    /// current prefill chunk with their vision-embedding rows. The copy
    /// aliases the chunk's device memory through a bounds-checked view, so
    /// non-image rows are untouched (no download/re-upload round trip).
    fn scatter_image_rows(
        hidden: &Tensor,
        image: &ImageScatter,
        chunk: std::ops::Range<usize>,
    ) -> Result<(), String> {
        let dims = hidden.shape().dims();
        if dims.len() != 2 || dims[1] * std::mem::size_of::<half::bf16>() != image.row_stride {
            return Err(format!(
                "image scatter row width {} does not match hidden {:?}",
                image.row_stride, dims
            ));
        }
        let buffer = apxinf_cuda::CudaBuffer::from_tensor(hidden)
            .map_err(|error| format!("image scatter buffer: {error}"))?;
        for (row_index, position) in image.positions.iter().enumerate() {
            if !chunk.contains(position) {
                continue;
            }
            let row_in_chunk = position - chunk.start;
            let view = buffer
                .view(row_in_chunk * image.row_stride, image.row_stride)
                .map_err(|error| format!("image scatter view: {error}"))?;
            let source_start = row_index * image.row_stride;
            view.copy_from_host(&image.row_bytes[source_start..source_start + image.row_stride])
                .map_err(|error| format!("image scatter copy: {error}"))?;
        }
        Ok(())
    }

    fn row_slice(backend: &CudaBackend, hidden: &Tensor, row: usize) -> Result<Tensor, String> {
        let dims = hidden.shape().dims();
        if dims.len() != 2 || row >= dims[0] {
            return Err(format!(
                "hidden row slice requires row {row} in matrix, got {:?}",
                dims
            ));
        }
        let flattened = hidden
            .reshape(vec![1, dims[0] * dims[1]])
            .map_err(|error| format!("reshape hidden for row slice failed: {error}"))?;
        apxinf_cuda::kernels::elementwise::slice_columns_bf16(
            backend.context(),
            &flattened,
            row * dims[1],
            dims[1],
        )
        .map_err(|error| format!("copy hidden row {row} failed: {error}"))
    }

    fn debug_capture_logits(logits: &[f32], position: usize) {
        let Some(directory) = std::env::var_os("APXINF_DEBUG_LOGITS_DIR") else {
            return;
        };
        let mut top1 = 0usize;
        let mut top2 = 0usize;
        for (index, value) in logits.iter().enumerate() {
            if *value > logits[top1] {
                top2 = top1;
                top1 = index;
            } else if index != top1 && *value > logits[top2] {
                top2 = index;
            }
        }
        let margin = logits[top1] - logits[top2];
        let path = std::path::Path::new(&directory)
            .join(format!("service-logits-pos-{position:03}.f32.bin"));
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut bytes = Vec::with_capacity(logits.len() * std::mem::size_of::<f32>());
        for value in logits {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let _ = std::fs::write(&path, bytes);
        eprintln!(
            "qwen35 debug logits position {position}: top1={top1} top2={top2} \
             v1={:.6} v2={:.6} margin={:.6}",
            logits[top1], logits[top2], margin
        );
    }

    fn debug_capture_hidden(
        backend: &CudaBackend,
        hidden: &Tensor,
        label: &str,
        position: usize,
    ) -> Result<(), String> {
        let Some(directory) = std::env::var_os("APXINF_DEBUG_HIDDEN_DIR") else {
            return Ok(());
        };
        let selected_position = matches!(position, 7 | 35 | 83);
        let selected_layer = matches!(
            label,
            "embedding"
                | "layer-000"
                | "layer-003"
                | "layer-031"
                | "layer-032"
                | "layer-060"
                | "layer-063"
        );
        if !selected_position || !selected_layer {
            return Ok(());
        }
        let cpu = backend
            .to_cpu(hidden)
            .map_err(|error| format!("debug copy {label} position {position}: {error}"))?;
        let values = cpu
            .to_f32_vec()
            .map_err(|error| format!("debug decode {label} position {position}: {error}"))?;
        let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
        for value in &values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let path = std::path::Path::new(&directory)
            .join(format!("service-{label}-pos-{position:03}.f32.bin"));
        std::fs::create_dir_all(path.parent().unwrap_or(std::path::Path::new(".")))
            .map_err(|error| format!("debug create directory: {error}"))?;
        std::fs::write(&path, bytes)
            .map_err(|error| format!("debug write {}: {error}", path.display()))?;
        let (min, max, sum_abs) = values.iter().fold(
            (f32::INFINITY, f32::NEG_INFINITY, 0.0f64),
            |(min, max, sum_abs), value| {
                (
                    min.min(*value),
                    max.max(*value),
                    sum_abs + f64::from(value.abs()),
                )
            },
        );
        eprintln!(
            "qwen35 debug hidden {label} position {position}: min={min:.7} max={max:.7} mean_abs={:.7}",
            sum_abs / values.len().max(1) as f64
        );
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use apxinf_loader::QWEN35_MODEL_REVISION;

        #[test]
        #[ignore = "requires GPU and the pinned Qwen3.5 checkpoint"]
        fn open_prefills_prompt_in_bounded_chunks_and_keeps_final_row() {
            let checkpoint = std::env::var_os("APXINF_QWEN35_CHECKPOINT")
                .map(std::path::PathBuf::from)
                .expect("APXINF_QWEN35_CHECKPOINT must point to checkpoint");
            let device = std::env::var("APXINF_CUDA_DEVICE")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(0);
            let inventory =
                Qwen35CheckpointInventory::from_checkpoint_dir(&checkpoint, QWEN35_MODEL_REVISION)
                    .unwrap();
            let model = Qwen35CudaModel::from_inventory(&inventory, device, 80).unwrap();
            let prompt = vec![1; 65];
            let mut session = model.open(&prompt, 2).unwrap();

            assert_eq!(session.prefill_calls, 2);
            assert_eq!(session.forward_token_calls, 0);
            assert_eq!(session.position, prompt.len());
            assert_eq!(session.last_input, Some(*prompt.last().unwrap()));
            assert_eq!(
                session.last_hidden.as_ref().unwrap().shape().dims(),
                &[1, inventory.config.hidden_size]
            );
            for state in &session.states {
                match state {
                    State::Gdn(state) => assert_eq!(state.position(), prompt.len()),
                    State::Full(state) => {
                        assert_eq!(state.seq_len(), prompt.len());
                        assert_eq!(state.max_seq_len(), prompt.len() + 2);
                    }
                }
            }

            assert!(session.next_token().unwrap().is_some());
            assert_eq!(session.forward_token_calls, 0);
            assert_eq!(session.position, prompt.len());
            assert!(session.next_token().unwrap().is_some());
            assert_eq!(session.forward_token_calls, 1);
            assert_eq!(session.position, prompt.len() + 1);
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_runtime::{decode_step_batch, Qwen35CudaModel, Qwen35CudaSession};

#[cfg(test)]
mod tests {
    use super::super::config::Qwen35ModelConfig;

    /// Tests below read (and one mutates) the prefill-chunk environment
    /// variables; serialize them so parallel test threads cannot observe a
    /// mid-mutation value.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn env_guard() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[test]
    fn prefill_plan_bounds_every_block_to_the_configured_chunk() {
        let _guard = env_guard();
        let chunk = super::prefill_chunk_tokens();
        assert!(chunk > 0 && chunk <= super::MAX_PREFILL_CHUNK_TOKENS);
        // Contiguous, gapless cover of the prompt with every block bounded.
        let tokens = chunk * 2 + 3;
        let ranges = super::prefill_ranges(tokens).collect::<Vec<_>>();
        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0].start, 0);
        assert_eq!(ranges.last().unwrap().end, tokens);
        for window in ranges.windows(2) {
            assert_eq!(window[0].end, window[1].start);
        }
        assert!(ranges.iter().all(|range| range.len() <= chunk));
        // A prompt shorter than one chunk is a single block.
        assert_eq!(
            super::prefill_ranges(chunk - 1).collect::<Vec<_>>(),
            vec![0..chunk - 1]
        );
    }

    #[test]
    fn prefill_chunk_override_is_clamped_to_the_workspace_bound() {
        let _guard = env_guard();
        // Restore the environment before asserting so a failure cannot leak
        // the override into other cases.
        let restore = |previous: Option<String>| match previous {
            Some(value) => std::env::set_var("APXINF_Q35_PREFILL_CHUNK", value),
            None => std::env::remove_var("APXINF_Q35_PREFILL_CHUNK"),
        };
        let previous = std::env::var("APXINF_Q35_PREFILL_CHUNK").ok();
        let previous_multimodal = std::env::var("APXINF_ENABLE_MULTIMODAL").ok();

        std::env::set_var("APXINF_Q35_PREFILL_CHUNK", "128");
        let explicit = super::prefill_chunk_tokens();
        std::env::set_var("APXINF_Q35_PREFILL_CHUNK", "99999");
        let clamped = super::prefill_chunk_tokens();
        std::env::set_var("APXINF_Q35_PREFILL_CHUNK", "0");
        let zero_rejected = super::prefill_chunk_tokens();
        std::env::set_var("APXINF_Q35_PREFILL_CHUNK", "not-a-number");
        let garbage_rejected = super::prefill_chunk_tokens();
        // With the vision tower resident the chunk-512 workspace no longer
        // fits; the multimodal default must drop to 256 (still overridable).
        std::env::remove_var("APXINF_Q35_PREFILL_CHUNK");
        std::env::set_var("APXINF_ENABLE_MULTIMODAL", "1");
        let multimodal_default = super::prefill_chunk_tokens();
        std::env::remove_var("APXINF_ENABLE_MULTIMODAL");
        let text_default = super::prefill_chunk_tokens();
        match previous_multimodal {
            Some(value) => std::env::set_var("APXINF_ENABLE_MULTIMODAL", value),
            None => std::env::remove_var("APXINF_ENABLE_MULTIMODAL"),
        }
        restore(previous);

        assert_eq!(explicit, 128);
        assert_eq!(clamped, super::MAX_PREFILL_CHUNK_TOKENS);
        assert_eq!(zero_rejected, super::PREFILL_CHUNK_TOKENS);
        assert_eq!(garbage_rejected, super::PREFILL_CHUNK_TOKENS);
        assert_eq!(multimodal_default, super::MULTIMODAL_PREFILL_CHUNK_TOKENS);
        assert_eq!(text_default, super::PREFILL_CHUNK_TOKENS);
    }

    #[test]
    fn request_capacity_is_prompt_plus_generation_budget() {
        assert_eq!(super::request_capacity(8_192, 1, 32_768).unwrap(), 8_193);
        assert!(super::request_capacity(32_768, 1, 32_768).is_err());
        assert!(super::request_capacity(1, 0, 32_768).is_err());
    }

    #[test]
    fn request_state_estimate_accounts_for_all_gdn_and_attention_buffers() {
        let _guard = env_guard();
        let raw = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../fixtures/qwen35-metadata/config.json"),
        )
        .unwrap();
        let config = Qwen35ModelConfig::from_json_str(&raw).unwrap();
        let bytes = super::request_state_bytes(&config, 32_768).unwrap();
        let gdn_channels = config.linear_key_heads * config.linear_head_dim * 2
            + config.linear_value_heads * config.linear_head_dim;
        let gdn_per_layer = 3 * gdn_channels * config.linear_conv_kernel_dim * 2
            + 3 * config.linear_value_heads * config.linear_head_dim * config.linear_head_dim * 4;
        let attention_per_layer =
            2 * config.full_attention_kv_heads * 32_768 * config.full_attention_head_dim * 2;
        // The GDN scan chunk is fixed at 64 regardless of the prefill block
        // size; the attention score workspace scales with the block size.
        let chunk = 64usize;
        let qk = chunk * config.linear_head_dim;
        let values = chunk * config.linear_head_dim;
        let matrix = chunk * chunk;
        let workspace_floats = qk * 2 + values + chunk * 2 + matrix + values + qk + values;
        let attention_prefill_workspace =
            config.full_attention_heads * super::prefill_chunk_tokens() * 32_768 * 2 * 2;
        let expected = 48 * gdn_per_layer
            + 16 * attention_per_layer
            + config.linear_value_heads * workspace_floats * 4
            + attention_prefill_workspace;
        assert_eq!(bytes, expected);
    }
}
