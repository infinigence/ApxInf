//! PI0.5 BF16 network computation, shared by eager and captured execution.
//!
//! This first migration slice retains the existing precision-specific Block
//! calls and solver exactly. It owns fixed configuration/weights and tensor
//! dataflow; it does not allocate a graph workspace, capture, replay, cache
//! prepared plans or bind host requests. Other precisions are not migrated yet.
//! Calibration is an explicit diagnostic traversal of this same computation.

use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Arc;

use super::backend::{kernels, Context, DeviceBuffer as CudaBuffer, RuntimeBackend};
use super::{
    action_layer_bf16, language_layer_bf16, vision_layer_bf16, vision_patch_embed_bf16,
    Bf16LinearWeights, Pi05CalibrationObserver, Pi05Config, StaticBf16Pi05Weights,
};
use apxinf_core::{Backend, DType, Error, Result, Tensor};
use kernels::{activation, cache, elementwise, embedding, gemm, norm};

pub struct Bf16PrefixKvCache {
    pub keys: Vec<Tensor>,
    pub values: Vec<Tensor>,
    pub tokens: usize,
}

pub(super) struct Bf16StepStyles {
    attention: Vec<Tensor>,
    mlp: Vec<Tensor>,
    final_norm: Tensor,
}

pub(super) struct Pi05Bf16Network {
    backend: Arc<RuntimeBackend>,
    config: Arc<Pi05Config>,
    weights: Arc<StaticBf16Pi05Weights>,
}

impl Pi05Bf16Network {
    pub fn new(
        backend: Arc<RuntimeBackend>,
        config: Arc<Pi05Config>,
        weights: Arc<StaticBf16Pi05Weights>,
    ) -> Result<Self> {
        config.validate()?;
        if weights.vision_layers.len() != config.vision_depth
            || weights.language_layers.len() != config.language.depth
            || weights.action_layers.len() != config.action_expert.depth
        {
            return Err(Error::Other(
                "π0.5 BF16 device weight depth mismatch".into(),
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

    pub fn encode_vision(&self, patches: &Tensor) -> Result<Tensor> {
        if patches.dtype() != DType::BF16 {
            return Err(Error::DTypeMismatch {
                expected: DType::BF16,
                got: patches.dtype(),
            });
        }
        let mut hidden = vision_patch_embed_bf16(
            self.ctx(),
            &self.weights.patch_embedding,
            &self.weights.position_embedding,
            patches,
            self.config.patches_per_view(),
        )?;
        for layer in &self.weights.vision_layers {
            hidden = vision_layer_bf16(
                self.ctx(),
                layer,
                &hidden,
                self.config.patches_per_view(),
                self.config.vision_heads,
                self.config.vision_head_dim,
                self.config.layer_norm_eps,
            )?;
        }
        let hidden = norm::layer_bf16(
            self.ctx(),
            &hidden,
            &self.weights.vision_post_norm.weight,
            &self.weights.vision_post_norm.bias,
            self.config.layer_norm_eps,
        )?;
        let projected = gemm::bf16(
            self.ctx(),
            &hidden,
            &self.weights.multimodal_projector.weight,
        )?;
        elementwise::bias_bf16(
            self.ctx(),
            &projected,
            self.weights.multimodal_projector.bias.as_ref(),
        )
    }

    pub fn embed_prefix(
        &self,
        vision_tokens: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
    ) -> Result<Tensor> {
        if token_count == 0 || token_count > self.config.max_token_len {
            return Err(Error::Other(format!(
                "π0.5 token count must be in 1..={}, got {token_count}",
                self.config.max_token_len
            )));
        }
        let language = embedding::lookup_bf16(
            self.ctx(),
            &self.weights.token_embedding,
            token_ids,
            token_count,
        )?;
        elementwise::concat_rows_bf16(self.ctx(), vision_tokens, &language)
    }

    pub fn prefix_forward(&self, prefix: &Tensor) -> Result<Bf16PrefixKvCache> {
        let mut hidden = prefix.clone();
        let mut keys = Vec::with_capacity(self.config.language.depth);
        let mut values = Vec::with_capacity(self.config.language.depth);
        for (index, layer) in self.weights.language_layers.iter().enumerate() {
            let output = language_layer_bf16(
                self.ctx(),
                self.config.language,
                layer,
                &hidden,
                index + 1 < self.config.language.depth,
                0,
                self.config.rms_norm_eps,
                self.config.rope_theta,
            )?;
            hidden = output.hidden;
            let cache_rows = prefix.shape().dims()[0] + self.config.action_horizon;
            keys.push(cache::reserve_prefix_bf16(
                self.ctx(),
                &output.key,
                cache_rows,
            )?);
            values.push(cache::reserve_prefix_bf16(
                self.ctx(),
                &output.value,
                cache_rows,
            )?);
        }
        Ok(Bf16PrefixKvCache {
            keys,
            values,
            tokens: prefix.shape().dims()[0],
        })
    }

    fn conditioning(&self, time_embedding: &Tensor) -> Result<Tensor> {
        let hidden = gemm::bf16(self.ctx(), time_embedding, &self.weights.time_mlp_in.weight)?;
        let hidden = activation::bias_silu_bf16(
            self.ctx(),
            &hidden,
            self.weights.time_mlp_in.bias.as_ref(),
        )?;
        let output = gemm::bf16(self.ctx(), &hidden, &self.weights.time_mlp_out.weight)?;
        activation::bias_silu_bf16(self.ctx(), &output, self.weights.time_mlp_out.bias.as_ref())
    }

    fn style(&self, conditioning: &Tensor, weights: &Bf16LinearWeights) -> Result<Tensor> {
        let projected = gemm::bf16(self.ctx(), conditioning, &weights.weight)?;
        let style = elementwise::bias_bf16(self.ctx(), &projected, weights.bias.as_ref())?;
        style.reshape(vec![style.numel()])
    }

    fn prepare_step_styles(&self, time_embedding: &Tensor) -> Result<Bf16StepStyles> {
        let conditioning = self.conditioning(time_embedding)?;
        let mut attention = Vec::with_capacity(self.config.action_expert.depth);
        let mut mlp = Vec::with_capacity(self.config.action_expert.depth);
        for layer in &self.weights.action_layers {
            attention.push(self.style(&conditioning, &layer.input_style)?);
            mlp.push(self.style(&conditioning, &layer.post_attention_style)?);
        }
        let final_norm = self.style(&conditioning, &self.weights.action_final_style)?;
        Ok(Bf16StepStyles {
            attention,
            mlp,
            final_norm,
        })
    }

    pub(super) fn prepare_all_styles(
        &self,
        time_embeddings: &[Tensor],
    ) -> Result<Vec<Bf16StepStyles>> {
        if time_embeddings.len() != self.config.num_flow_steps {
            return Err(Error::Other(format!(
                "π0.5 expected {} timestep embeddings, got {}",
                self.config.num_flow_steps,
                time_embeddings.len()
            )));
        }
        time_embeddings
            .iter()
            .map(|embedding| self.prepare_step_styles(embedding))
            .collect()
    }

    fn denoise_step_with_styles(
        &self,
        state: &Tensor,
        styles: &Bf16StepStyles,
        prefix: &Bf16PrefixKvCache,
        dt: f32,
    ) -> Result<Tensor> {
        if prefix.keys.len() != self.config.action_expert.depth
            || prefix.values.len() != self.config.action_expert.depth
            || styles.attention.len() != self.config.action_expert.depth
            || styles.mlp.len() != self.config.action_expert.depth
        {
            return Err(Error::Other("π0.5 BF16 prefix/style depth mismatch".into()));
        }
        let hidden = gemm::bf16(self.ctx(), state, &self.weights.action_in.weight)?;
        let mut hidden =
            elementwise::bias_bf16(self.ctx(), &hidden, self.weights.action_in.bias.as_ref())?;
        let mut attention_normalized = None;
        for index in 0..self.config.action_expert.depth {
            let layer = &self.weights.action_layers[index];
            let next_norm_style = if index + 1 < self.config.action_expert.depth {
                &styles.attention[index + 1]
            } else {
                &styles.final_norm
            };
            let output = action_layer_bf16(
                self.ctx(),
                self.config.action_expert,
                layer,
                &hidden,
                attention_normalized.as_ref(),
                &styles.attention[index],
                &styles.mlp[index],
                next_norm_style,
                &prefix.keys[index],
                &prefix.values[index],
                prefix.tokens,
                self.config.rms_norm_eps,
                self.config.rope_theta,
            )?;
            hidden = output.hidden;
            attention_normalized = Some(output.next_normalized);
        }
        let hidden = attention_normalized.ok_or_else(|| {
            Error::Other("π0.5 action expert must contain at least one layer".into())
        })?;
        let velocity = gemm::bf16(self.ctx(), &hidden, &self.weights.action_out.weight)?;
        let velocity =
            elementwise::bias_bf16(self.ctx(), &velocity, self.weights.action_out.bias.as_ref())?;
        elementwise::euler_update_bf16(self.ctx(), state, &velocity, dt)
    }

    pub fn denoise_step(
        &self,
        state: &Tensor,
        time_embedding: &Tensor,
        prefix: &Bf16PrefixKvCache,
        dt: f32,
    ) -> Result<Tensor> {
        let styles = self.prepare_step_styles(time_embedding)?;
        self.denoise_step_with_styles(state, &styles, prefix, dt)
    }

    fn denoise_all_steps_with_styles(
        &self,
        noise: &Tensor,
        styles: &[Bf16StepStyles],
        prefix: &Bf16PrefixKvCache,
    ) -> Result<Tensor> {
        if styles.len() != self.config.num_flow_steps {
            return Err(Error::Other(format!(
                "π0.5 expected {} precomputed style sets, got {}",
                self.config.num_flow_steps,
                styles.len()
            )));
        }
        let mut state = noise.clone();
        let dt = -self.config.flow_start_time / self.config.num_flow_steps as f32;
        for step_styles in styles {
            state = self.denoise_step_with_styles(&state, step_styles, prefix, dt)?;
        }
        Ok(state)
    }

    pub fn denoise_all_steps(
        &self,
        noise: &Tensor,
        time_embeddings: &[Tensor],
        prefix: &Bf16PrefixKvCache,
    ) -> Result<Tensor> {
        let styles = self.prepare_all_styles(time_embeddings)?;
        self.denoise_all_steps_with_styles(noise, &styles, prefix)
    }

    pub(super) fn infer_with_styles(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        styles: &[Bf16StepStyles],
    ) -> Result<Tensor> {
        let vision = self.encode_vision(patches)?;
        let prefix = self.embed_prefix(&vision, token_ids, token_count)?;
        let prefix = self.prefix_forward(&prefix)?;
        self.denoise_all_steps_with_styles(noise, styles, &prefix)
    }

    pub fn infer(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<Tensor> {
        let styles = self.prepare_all_styles(time_embeddings)?;
        self.infer_with_styles(patches, token_ids, token_count, noise, &styles)
    }

    pub fn calibrate(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<BTreeMap<String, f32>> {
        let observer = Rc::new(Pi05CalibrationObserver::new(
            Arc::clone(&self.backend),
            &self.config,
            &self.weights,
        )?);
        let _guard = kernels::gemm::install_bf16_observer(observer.clone())?;
        self.infer(patches, token_ids, token_count, noise, time_embeddings)?;
        self.backend.synchronize()?;
        observer.records()
    }
}
