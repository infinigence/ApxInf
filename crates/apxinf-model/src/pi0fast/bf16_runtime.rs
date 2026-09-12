//! Fixed-shape native-BF16 π0-FAST inference runtime.
//!
//! Execution is two explicit phases:
//!
//! 1. **Prefix**: the SigLIP tower + projector produce image embeddings, the
//!    prompt token ids are looked up and scaled, the two are concatenated, and
//!    the Gemma stack runs once with bidirectional attention. Each layer's
//!    K/V is parked in a cache sized for the prefix plus every decoding step.
//! 2. **Decode**: the LM head argmaxes the last position, the winning token is
//!    written to a device buffer, and it is fed straight back into the tied
//!    embedding lookup through the persistent cache. The token never leaves the
//!    device inside the loop, so this path is CUDA-graph eligible.
//!
//! The runtime returns the raw action tokens. FAST detokenization (BPE + DCT)
//! is action postprocessing and belongs to the Python policy layer.

use std::sync::{Arc, Mutex};

use super::backend::{kernels, Context, DeviceBuffer as CudaBuffer, RuntimeBackend};
use apxinf_core::{Error, Result, Tensor};
use kernels::{cache, elementwise, embedding, gemm, norm, sampling, GraphWorkspace};

use super::{
    language_layer_bf16, language_layer_cached_decode_bf16, vision_layer_bf16,
    vision_patch_embed_f32_bf16, Pi0FastConfig, StaticBf16Pi0FastWeights,
};

/// Conservative arena reservation for one `infer` traversal.
///
/// [`GraphWorkspace`] is a bump arena: every buffer `output_buffer` hands out
/// during a call stays live until the scope ends, so the capacity has to cover
/// the *sum* of the call's intermediates rather than their peak. The estimate
/// follows `infer_arena` op by op and is deliberately generous; an
/// under-estimate surfaces as a workspace-exhausted error naming the exact
/// requirement.
fn arena_bytes(config: &Pi0FastConfig, token_count: usize) -> usize {
    const BF16: usize = 2;
    const F32: usize = 4;
    const ALIGN: usize = 256;

    let patches = config.patch_tokens();
    let vw = config.vision_width;
    let vmlp = config.vision_mlp_dim;
    let lw = config.language.width;
    let lmlp = config.language.mlp_dim;
    let prefix = patches + token_count;
    let cache_rows = prefix + config.max_action_tokens;
    let kv_cols = config.language.num_kv_heads * config.language.head_dim;

    let mut total = 0usize;
    let mut add = |bytes: usize| total += bytes + ALIGN;

    // FP32 SigLIP patch embedding: projection, bias+position, BF16 output.
    add(patches * vw * F32);
    add(patches * vw * F32);
    add(patches * vw * BF16);

    for _ in 0..config.vision_depth {
        add(patches * vw * BF16);
        add(patches * 3 * vw * BF16);
        add(patches * 3 * vw * BF16);
        add(patches * vw * BF16);
        add(patches * vw * BF16);
        add(patches * vw * BF16);
        add(patches * vw * BF16);
        add(patches * vmlp * BF16);
        add(patches * vmlp * BF16);
        add(patches * vw * BF16);
        add(patches * vw * BF16);
    }
    add(patches * vw * BF16);
    add(patches * vw * BF16);
    add(patches * lw * BF16);
    add(patches * lw * BF16);

    add(token_count * lw * BF16);
    add(prefix * lw * BF16);
    for _ in 0..config.language.depth {
        add(prefix * lw * BF16);
        add(prefix * 3 * lw * BF16);
        add(prefix * 2 * lw * BF16);
        add(prefix * lw * BF16);
        add(prefix * lw * BF16);
        add(prefix * lw * BF16);
        add(prefix * lw * BF16);
        add(prefix * 2 * lmlp * BF16);
        add(prefix * lmlp * BF16);
        add(prefix * lw * BF16);
        add(prefix * lw * BF16);
    }
    // Prefix KV cache copy plus the last-row gather.
    add(2 * config.language.depth * cache_rows * kv_cols * BF16);
    add(lw * BF16 * 4);

    // Autoregressive steps: one token through every layer, plus lm_head logits.
    let per_step = config.language.depth * 7 * lw * BF16 + config.language.depth * 3 * lmlp * BF16;
    add(config.max_action_tokens * (per_step + config.vocab_size * BF16));

    total + (total / 4)
}

pub struct Pi0FastBf16Runtime {
    backend: Arc<RuntimeBackend>,
    config: Arc<Pi0FastConfig>,
    weights: Arc<StaticBf16Pi0FastWeights>,
    /// Persistent arena for the per-call intermediates. Without it every
    /// intermediate is a raw `cudaMalloc`/`cudaFree` pair.
    arena: Mutex<Option<GraphWorkspace>>,
}

impl Pi0FastBf16Runtime {
    pub fn new(
        backend: Arc<RuntimeBackend>,
        config: Arc<Pi0FastConfig>,
        weights: Arc<StaticBf16Pi0FastWeights>,
    ) -> Result<Self> {
        config.validate()?;
        if weights.vision_layers.len() != config.vision_depth
            || weights.language_layers.len() != config.language.depth
        {
            return Err(Error::Other(
                "π0-FAST BF16 device weight depth mismatch".into(),
            ));
        }
        Ok(Self {
            backend,
            config,
            weights,
            arena: Mutex::new(None),
        })
    }

    fn ctx(&self) -> &Context {
        self.backend.context()
    }

    /// Run `operation` inside the persistent arena, growing it when a call
    /// needs more room than the current one provides.
    fn with_arena<T>(
        &self,
        token_count: usize,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let capacity = arena_bytes(&self.config, token_count);
        let mut arena = self
            .arena
            .lock()
            .map_err(|_| Error::Other("π0-FAST arena mutex is poisoned".into()))?;
        if arena
            .as_ref()
            .is_none_or(|workspace| workspace.capacity() < capacity)
        {
            *arena = Some(GraphWorkspace::new(capacity, self.ctx().device_id())?);
        }
        kernels::with_workspace_eager(
            arena.as_ref().expect("arena is populated above"),
            operation,
        )
    }

    /// Bytes of the arena the last call used.
    pub fn arena_used_bytes(&self) -> usize {
        self.arena
            .lock()
            .ok()
            .and_then(|arena| arena.as_ref().map(|workspace| workspace.used()))
            .unwrap_or(0)
    }

    /// Longest token sequence the decode loop can address: the whole prompt
    /// (conversation prefix + BOS) plus every action token.
    pub fn max_sequence(&self, token_count: usize) -> usize {
        self.config.patch_tokens() + token_count + self.config.max_action_tokens
    }

    /// Run the vision tower and projector, returning `[views*patches, hidden]`.
    ///
    /// `patches` is the FP32 patch-major tensor PaliGemma's SigLIP tower
    /// consumes; the BF16 encoder input is produced inside the patch embedding.
    
    fn embed_images(&self, patches: &Tensor) -> Result<Tensor> {
        let config = &self.config;
        let weights = &self.weights;
        let patches_per_view = config.patches_per_view();
        let mut hidden = vision_patch_embed_f32_bf16(
            self.ctx(),
            &weights.patch_embedding,
            patches,
            patches_per_view,
        )?;
        for layer in &weights.vision_layers {
            hidden = vision_layer_bf16(
                self.ctx(),
                layer,
                &hidden,
                patches_per_view,
                config.vision_heads,
                config.vision_head_dim,
                config.layer_norm_eps,
            )?;
        }
        let hidden = norm::layer_bf16(
            self.ctx(),
            &hidden,
            &weights.vision_post_norm.weight,
            &weights.vision_post_norm.bias,
            config.layer_norm_eps,
        )?;
        let projected = gemm::bf16(self.ctx(), &hidden, &weights.multimodal_projector.weight)?;
        elementwise::bias_bf16(
            self.ctx(),
            &projected,
            weights.multimodal_projector.bias.as_ref(),
        )
    }

    /// Full observation-to-token inference. `token_count` is the prompt length,
    /// which already includes the BOS token the reference appends.
    pub fn infer(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        stop_token: Option<u32>,
    ) -> Result<Vec<u32>> {
        self.with_arena(token_count, || {
            self.infer_arena(patches, token_ids, token_count, stop_token)
        })
    }

    /// One full traversal. The caller has already bound the arena.
    fn infer_arena(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        stop_token: Option<u32>,
    ) -> Result<Vec<u32>> {
        let config = &self.config;
        if token_count == 0 || token_count > config.max_token_len + 1 {
            return Err(Error::Other(format!(
                "pi0fast prompt token count must be in 1..={}, got {token_count}",
                config.max_token_len + 1
            )));
        }
        let max_sequence = self.max_sequence(token_count);

        let image = self.embed_images(patches)?;
        let language = self.embed_tokens(token_ids, token_count)?;
        let prefix = elementwise::concat_rows_bf16(self.ctx(), &image, &language)?;

        let (hidden, keys, values) = self.decode_prefix(prefix, max_sequence)?;
        let prefix_len = token_count + config.patch_tokens();

        let step_bytes = std::mem::size_of::<u32>();
        let tokens =
            CudaBuffer::alloc(config.max_action_tokens * step_bytes, self.ctx().device_id())
                .map_err(Error::Cuda)?;
        let mut logits = self.lm_head(&hidden)?;

        // One generated token per step: argmax on device, feed the id straight
        // back into the tied embedding lookup, and only read the ids back once
        // the loop ends. `stop_token` (the `|` terminator) ends the stream one
        // step after it is emitted; the detokenizer truncates there regardless,
        // so the decoded chunk is unchanged and the remaining steps are skipped.
        let mut produced = config.max_action_tokens;
        for step in 0..config.max_action_tokens {
            let slot = tokens
                .view(step * step_bytes, step_bytes)
                .map_err(Error::Cuda)?;
            sampling::argmax_bf16_into(self.ctx(), &logits, &slot)?;
            if let Some(stop) = stop_token {
                let mut raw = [0u8; 4];
                slot.copy_to_host(&mut raw).map_err(Error::Cuda)?;
                if u32::from_ne_bytes(raw) == stop {
                    produced = step + 1;
                    break;
                }
            }
            if step + 1 == config.max_action_tokens {
                break;
            }
            let mut step_hidden = self.embed_tokens(&slot, 1)?;
            for (index, layer) in self.weights.language_layers.iter().enumerate() {
                step_hidden = language_layer_cached_decode_bf16(
                    self.ctx(),
                    config.language,
                    layer,
                    &step_hidden,
                    &keys[index],
                    &values[index],
                    prefix_len + step,
                    config.rms_norm_eps,
                    config.rope_theta,
                )?;
            }
            logits = self.lm_head(&step_hidden)?;
        }

        let mut bytes = vec![0u8; produced * step_bytes];
        tokens.copy_to_host(&mut bytes).map_err(Error::Cuda)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }

    fn decode_prefix(
        &self,
        prefix: Tensor,
        max_sequence: usize,
    ) -> Result<(Tensor, Vec<Tensor>, Vec<Tensor>)> {
        let config = &self.config;
        let weights = &self.weights;
        let mut hidden = prefix;
        let mut keys = Vec::with_capacity(weights.language_layers.len());
        let mut values = Vec::with_capacity(weights.language_layers.len());
        for layer in &weights.language_layers {
            let output = language_layer_bf16(
                self.ctx(),
                config.language,
                layer,
                &hidden,
                true,
                0,
                config.rms_norm_eps,
                config.rope_theta,
            )?;
            hidden = output.hidden;
            keys.push(cache::reserve_prefix_bf16(
                self.ctx(),
                &output.key,
                max_sequence,
            )?);
            values.push(cache::reserve_prefix_bf16(
                self.ctx(),
                &output.value,
                max_sequence,
            )?);
        }
        let rows = hidden.shape().dims()[0];
        let step_bytes = std::mem::size_of::<u32>();
        let index = CudaBuffer::alloc(step_bytes, self.ctx().device_id()).map_err(Error::Cuda)?;
        index
            .copy_from_host(&((rows - 1) as u32).to_ne_bytes())
            .map_err(Error::Cuda)?;
        let last = elementwise::gather_rows_bf16(self.ctx(), &hidden, &index, 1)?;
        Ok((last, keys, values))
    }

    fn lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        let normalized = norm::rms_bf16(
            self.ctx(),
            hidden,
            &self.weights.language_final_norm_scale,
            self.config.rms_norm_eps,
        )?;
        super::bf16_executor::decode_projection(self.ctx(), &normalized, &self.weights.lm_head.weight)
    }

    /// Look up `token_count` ids and scale by PaliGemma's `sqrt(width)`.
    ///
    /// LeRobot scales language and generated action-token embeddings but leaves
    /// image embeddings alone, so the factor belongs to the caller's token
    /// stream rather than to the lookup kernel. The text tower, the generated
    /// action token, and the autoregressive feedback loop all share this path.
    fn embed_tokens(&self, ids: &CudaBuffer, token_count: usize) -> Result<Tensor> {
        let width = self.config.language.width;
        let tensor = embedding::lookup(self.ctx(), &self.weights.token_embedding, ids, token_count)?;
        elementwise::scale(self.ctx(), &tensor, (width as f32).sqrt())
    }
}
