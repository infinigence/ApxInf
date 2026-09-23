//! Fixed-shape FP8 π0-FAST inference runtime.
//!
//! Identical structure to `bf16_runtime`: one prefix pass (SigLIP tower +
//! projector + prompt prefill, K/V parked in a cache) followed by an
//! autoregressive decode that keeps the token on device. Only the projection
//! dispatch differs — `gemm::fp8_bf16` with E4M3 weights and calibrated
//! activation scales.

use std::sync::{Arc, Mutex};

use super::backend::{kernels, Context, DeviceBuffer as CudaBuffer, RuntimeBackend};
use apxinf_core::{Error, Result, Tensor};
use kernels::{cache, elementwise, embedding, norm, sampling, GraphWorkspace};

use super::fp8_executor::fp8_projection;
use super::{
    language_layer_cached_decode_fp8, language_layer_fp8, vision_layer_fp8,
    Pi0FastConfig, Pi0FastFp8Scales, StaticFp8Pi0FastWeights,
};
use super::bf16_executor::vision_patch_embed_f32_bf16;

/// Conservative arena reservation for one FP8 `infer` traversal.
///
/// Same op-by-op accounting as the BF16 runtime, since every intermediate stays
/// BF16; the extra 25% over the BF16 estimate covers the E4M3 copies each
/// projection makes of its activation before the GEMM.
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
    add(2 * config.language.depth * cache_rows * kv_cols * BF16);
    add(lw * BF16 * 4);

    let per_step = config.language.depth * 7 * lw * BF16 + config.language.depth * 3 * lmlp * BF16;
    add(config.max_action_tokens * (per_step + config.action_head_width() * BF16));

    total + total / 2
}

pub struct Pi0FastFp8Runtime {
    backend: Arc<RuntimeBackend>,
    config: Arc<Pi0FastConfig>,
    weights: Arc<StaticFp8Pi0FastWeights>,
    scales: Arc<Pi0FastFp8Scales>,
    /// Maps the pruned LM-head columns back to global token ids; see
    /// [`super::action_head_remap`].
    lm_head_remap: CudaBuffer,
    arena: Mutex<Option<GraphWorkspace>>,
}

impl Pi0FastFp8Runtime {
    pub fn new(
        backend: Arc<RuntimeBackend>,
        config: Arc<Pi0FastConfig>,
        weights: Arc<StaticFp8Pi0FastWeights>,
        scales: Arc<Pi0FastFp8Scales>,
    ) -> Result<Self> {
        config.validate()?;
        if weights.vision_layers.len() != config.vision_depth
            || weights.language_layers.len() != config.language.depth
        {
            return Err(Error::Other(
                "π0-FAST FP8 device weight depth mismatch".into(),
            ));
        }
        if scales.vision_layers.len() != config.vision_depth
            || scales.language_layers.len() != config.language.depth
        {
            return Err(Error::Other(
                "π0-FAST FP8 activation scale depth mismatch".into(),
            ));
        }
        let lm_head_remap = super::action_head_remap(&config, backend.context())?;
        Ok(Self {
            backend,
            config,
            weights,
            scales,
            lm_head_remap,
            arena: Mutex::new(None),
        })
    }

    fn ctx(&self) -> &Context {
        self.backend.context()
    }

    fn with_arena<T>(
        &self,
        token_count: usize,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let capacity = arena_bytes(&self.config, token_count);
        let mut arena = self
            .arena
            .lock()
            .map_err(|_| Error::Other("π0-FAST FP8 arena mutex is poisoned".into()))?;
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

    pub fn max_sequence(&self, token_count: usize) -> usize {
        self.config.patch_tokens() + token_count + self.config.max_action_tokens
    }

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
        for (layer, scales) in weights.vision_layers.iter().zip(&self.scales.vision_layers) {
            hidden = vision_layer_fp8(
                self.ctx(),
                layer,
                *scales,
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
        let projected = fp8_projection(
            self.ctx(),
            &hidden,
            &weights.multimodal_projector,
            self.scales.multimodal_projector,
        )?;
        elementwise::bias_bf16(
            self.ctx(),
            &projected,
            weights.multimodal_projector.bias.as_ref(),
        )
    }

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
        let tokens = CudaBuffer::alloc(config.max_action_tokens * step_bytes, self.ctx().device_id())
            .map_err(Error::Cuda)?;
        let mut logits = self.lm_head(&hidden)?;

        let mut produced = config.max_action_tokens;
        for step in 0..config.max_action_tokens {
            let slot = tokens
                .view(step * step_bytes, step_bytes)
                .map_err(Error::Cuda)?;
            sampling::argmax_bf16_remapped_into(
                self.ctx(),
                &logits,
                &self.lm_head_remap,
                &slot,
            )?;
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
                step_hidden = language_layer_cached_decode_fp8(
                    self.ctx(),
                    config.language,
                    layer,
                    self.scales.language_layers[index],
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
        for (index, layer) in weights.language_layers.iter().enumerate() {
            let output = language_layer_fp8(
                self.ctx(),
                config.language,
                layer,
                self.scales.language_layers[index],
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
        fp8_projection(
            self.ctx(),
            &normalized,
            &self.weights.lm_head,
            self.scales.lm_head,
        )
    }

    fn embed_tokens(&self, ids: &CudaBuffer, token_count: usize) -> Result<Tensor> {
        let width = self.config.language.width;
        let tensor = embedding::lookup(self.ctx(), &self.weights.token_embedding, ids, token_count)?;
        elementwise::scale(self.ctx(), &tensor, (width as f32).sqrt())
    }
}
