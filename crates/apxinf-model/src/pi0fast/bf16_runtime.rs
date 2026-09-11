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

use std::sync::Arc;

use super::backend::{
    kernels, Context, DeviceAddress, DeviceBuffer as CudaBuffer, RuntimeBackend,
};
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use kernels::{cache, elementwise, embedding, gemm, norm, sampling};

use super::{
    language_layer_bf16, language_layer_cached_bf16, vision_layer_bf16,
    vision_patch_embed_f32_bf16, Pi0FastConfig, StaticBf16Pi0FastWeights,
};

pub struct Pi0FastBf16Runtime {
    backend: Arc<RuntimeBackend>,
    config: Arc<Pi0FastConfig>,
    weights: Arc<StaticBf16Pi0FastWeights>,
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
        })
    }

    fn ctx(&self) -> &Context {
        self.backend.context()
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
        let config = &self.config;
        if token_count == 0 || token_count > config.max_token_len + 1 {
            return Err(Error::Other(format!(
                "pi0fast prompt token count must be in 1..={}, got {token_count}",
                config.max_token_len + 1
            )));
        }
        let max_sequence = self.max_sequence(token_count);

        let image = self.embed_images(patches)?;
        let language = self.embed_tokens(token_ids.address(), token_count)?;
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
            let mut step_hidden = self.embed_tokens(slot.address(), 1)?;
            for (index, layer) in self.weights.language_layers.iter().enumerate() {
                step_hidden = language_layer_cached_bf16(
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
        gemm::bf16(self.ctx(), &normalized, &self.weights.lm_head.weight)
    }

    /// Look up `token_count` ids and scale by PaliGemma's `sqrt(width)`.
    ///
    /// LeRobot scales language and generated action-token embeddings but leaves
    /// image embeddings alone, so the factor belongs to the caller's token
    /// stream rather than to the lookup kernel. The text tower, the generated
    /// action token, and the autoregressive feedback loop all share this path.
    fn embed_tokens(&self, ids: DeviceAddress, token_count: usize) -> Result<Tensor> {
        let width = self.config.language.width;
        let table = CudaBuffer::from_tensor(&self.weights.token_embedding).map_err(Error::Cuda)?;
        let bytes = token_count * width * DType::BF16.size_in_bytes();
        let output = CudaBuffer::alloc(bytes, self.ctx().device_id()).map_err(Error::Cuda)?;
        embedding::lookup_into(
            self.ctx(),
            DType::BF16,
            &table,
            ids,
            &output,
            width,
            token_count,
        )?;
        let tensor = output
            .as_tensor(Shape::new(vec![token_count, width]), DType::BF16)
            .map_err(Error::Cuda)?;
        elementwise::scale(self.ctx(), &tensor, (width as f32).sqrt())
    }
}
