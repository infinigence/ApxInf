//! Native-BF16 π0.5 transformer-layer computation.

use crate::pi05::backend::{kernels, Context};
use apxinf_core::{Error, Result, Tensor};
use kernels::{activation, attention, embedding, fused, gemm, norm, rope};

use crate::pi05::{
    Bf16DeviceActionLayer, Bf16DeviceLanguageLayer, Bf16DeviceVisionBlock, Bf16LinearWeights,
    GemmaVariantConfig,
};

pub struct Bf16LanguageLayerOutput {
    pub hidden: Tensor,
    pub key: Tensor,
    pub value: Tensor,
}

pub struct Bf16ActionLayerOutput {
    pub hidden: Tensor,
    pub next_normalized: Tensor,
}

#[allow(clippy::too_many_arguments)]
pub fn language_layer_bf16(
    ctx: &Context,
    config: GemmaVariantConfig,
    weights: &Bf16DeviceLanguageLayer,
    input: &Tensor,
    compute_tail: bool,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Bf16LanguageLayerOutput> {
    let normalized = norm::rms_bf16(ctx, input, &weights.input_norm_scale, rms_eps)?;
    let qkv = gemm::bf16(ctx, &normalized, &weights.qkv.weight)?;
    let qkv = rope::split_qkv_apply_bf16(
        ctx,
        &qkv,
        weights.qkv.bias.as_ref(),
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        rope_theta,
        position_offset,
    )?;
    let tokens = input.shape().dims()[0];
    if !compute_tail {
        return Ok(Bf16LanguageLayerOutput {
            hidden: input.clone(),
            key: qkv.key_2d(tokens, config.head_dim)?,
            value: qkv.value_2d(tokens, config.head_dim)?,
        });
    }
    let attention = attention::mqa_bf16(ctx, &qkv.q, &qkv.k, &qkv.v, tokens)?
        .reshape(vec![tokens, config.num_heads * config.head_dim])?;
    let projected = gemm::bf16(ctx, &attention, &weights.output.weight)?;
    let fused = fused::bias_residual_rms_bf16(
        ctx,
        &projected,
        weights.output.bias.as_ref(),
        input,
        &weights.post_attention_norm_scale,
        rms_eps,
    )?;
    let activated = gemm::bf16_geglu_fused(
        ctx,
        &fused.normalized,
        &weights.gate_up.weight,
        weights.gate_up.bf16_dual_geglu_interleaved,
        weights.gate_up.bf16_dual_geglu_auto_interleaved.as_ref(),
        weights.gate_up.bf16_sm89_geglu_interleaved.as_ref(),
    )?;
    let projected = gemm::bf16(ctx, &activated, &weights.down.weight)?;
    let hidden =
        fused::bias_residual_bf16(ctx, &projected, weights.down.bias.as_ref(), &fused.hidden)?;
    Ok(Bf16LanguageLayerOutput {
        hidden,
        key: qkv.key_2d(tokens, config.head_dim)?,
        value: qkv.value_2d(tokens, config.head_dim)?,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn action_layer_bf16(
    ctx: &Context,
    config: GemmaVariantConfig,
    weights: &Bf16DeviceActionLayer,
    input: &Tensor,
    attention_normalized: Option<&Tensor>,
    attention_modulation: &Tensor,
    mlp_modulation: &Tensor,
    next_norm_modulation: &Tensor,
    prefix_k: &Tensor,
    prefix_v: &Tensor,
    position_offset: usize,
    rms_eps: f32,
    rope_theta: f32,
) -> Result<Bf16ActionLayerOutput> {
    let normalized = match attention_normalized {
        Some(value) => value.clone(),
        None => norm::adaptive_rms_bf16(ctx, input, attention_modulation, rms_eps)?,
    };
    let qkv = gemm::bf16(ctx, &normalized, &weights.qkv.weight)?;
    let q = rope::apply_q_write_kv_bf16(
        ctx,
        &qkv,
        weights.qkv.bias.as_ref(),
        config.num_heads,
        config.num_kv_heads,
        config.head_dim,
        rope_theta,
        position_offset,
        prefix_k,
        prefix_v,
        position_offset,
    )?;
    let attention = attention::mqa_bf16(
        ctx,
        &q,
        prefix_k,
        prefix_v,
        position_offset + input.shape().dims()[0],
    )?
    .reshape(vec![
        input.shape().dims()[0],
        config.num_heads * config.head_dim,
    ])?;
    let projected = gemm::bf16(ctx, &attention, &weights.output.weight)?;
    let fused = fused::adaptive_gate_residual_rms_bf16(
        ctx,
        &projected,
        input,
        attention_modulation,
        mlp_modulation,
        rms_eps,
    )?;
    let activated = gemm::bf16_geglu_fused(
        ctx,
        &fused.normalized,
        &weights.gate_up.weight,
        weights.gate_up.bf16_dual_geglu_interleaved,
        weights.gate_up.bf16_dual_geglu_auto_interleaved.as_ref(),
        weights.gate_up.bf16_sm89_geglu_interleaved.as_ref(),
    )?;
    let projected = gemm::bf16(ctx, &activated, &weights.down.weight)?;
    let fused = fused::adaptive_gate_residual_rms_bf16(
        ctx,
        &projected,
        &fused.hidden,
        mlp_modulation,
        next_norm_modulation,
        rms_eps,
    )?;
    Ok(Bf16ActionLayerOutput {
        hidden: fused.hidden,
        next_normalized: fused.normalized,
    })
}

pub fn vision_patch_embed_bf16(
    ctx: &Context,
    weights: &Bf16LinearWeights,
    position_embedding: &Tensor,
    patches: &Tensor,
    patches_per_view: usize,
) -> Result<Tensor> {
    let projection = gemm::bf16(ctx, patches, &weights.weight)?;
    embedding::add_position_bf16(
        ctx,
        &projection,
        weights.bias.as_ref(),
        position_embedding,
        patches_per_view,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn vision_layer_bf16(
    ctx: &Context,
    weights: &Bf16DeviceVisionBlock,
    input: &Tensor,
    patches_per_view: usize,
    heads: usize,
    head_dim: usize,
    layer_norm_eps: f32,
) -> Result<Tensor> {
    let normalized = norm::layer_bf16(
        ctx,
        input,
        &weights.norm1.weight,
        &weights.norm1.bias,
        layer_norm_eps,
    )?;
    let qkv = gemm::bf16(ctx, &normalized, &weights.qkv.weight)?;
    let qkv =
        attention::split_qkv_bias_bf16(ctx, &qkv, weights.qkv.bias.as_ref(), heads, head_dim)?;
    let attention = attention::mha_bf16(ctx, &qkv.q, &qkv.k, &qkv.v, patches_per_view)?
        .reshape(vec![input.shape().dims()[0], heads * head_dim])?;
    let projection = gemm::bf16(ctx, &attention, &weights.output.weight)?;
    let fused = fused::bias_residual_layer_bf16(
        ctx,
        &projection,
        weights.output.bias.as_ref(),
        input,
        &weights.norm2.weight,
        &weights.norm2.bias,
        layer_norm_eps,
    )?;
    let activation = gemm::bf16(ctx, &fused.normalized, &weights.fc1.weight)?;
    let activation = activation::bias_gelu_bf16(ctx, &activation, weights.fc1.bias.as_ref())?;
    let projection = gemm::bf16(ctx, &activation, &weights.fc2.weight)?;
    fused::bias_residual_bf16(ctx, &projection, weights.fc2.bias.as_ref(), &fused.hidden)
}

trait QkvViews {
    fn key_2d(&self, tokens: usize, head_dim: usize) -> Result<Tensor>;
    fn value_2d(&self, tokens: usize, head_dim: usize) -> Result<Tensor>;
}

impl QkvViews for rope::QkvTensors {
    fn key_2d(&self, tokens: usize, head_dim: usize) -> Result<Tensor> {
        self.k.reshape(vec![tokens, head_dim])
    }

    fn value_2d(&self, tokens: usize, head_dim: usize) -> Result<Tensor> {
        self.v.reshape(vec![tokens, head_dim])
    }
}

// Precision-specific backbone operations share this file with their layers.
pub(in crate::pi05::model) mod backbone {
    use super::*;
    use crate::pi05::backend::{kernels, Context, DeviceBuffer as CudaBuffer, RuntimeBackend};
    use crate::pi05::weights::*;
    use crate::pi05::Pi05Config;
    use apxinf_core::{DType, Error, Result, Tensor};
    use kernels::{activation, cache, elementwise, embedding, gemm, norm};
    use std::sync::Arc;
    pub struct Bf16PrefixKvCache {
        pub keys: Vec<Tensor>,
        pub values: Vec<Tensor>,
        pub tokens: usize,
    }

    pub struct Bf16StepModulation {
        attention: Vec<Tensor>,
        mlp: Vec<Tensor>,
        final_norm: Tensor,
    }
    pub struct Bf16Blocks {
        pub(in crate::pi05::model) backend: Arc<RuntimeBackend>,
        pub(in crate::pi05::model) config: Arc<Pi05Config>,
        pub(in crate::pi05::model) weights: Arc<Bf16Weights>,
    }
    impl Bf16Blocks {
        pub fn new(
            backend: Arc<RuntimeBackend>,
            config: Arc<Pi05Config>,
            weights: Arc<Bf16Weights>,
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

        fn modulation(&self, conditioning: &Tensor, weights: &Bf16LinearWeights) -> Result<Tensor> {
            let projected = gemm::bf16(self.ctx(), conditioning, &weights.weight)?;
            let modulation = elementwise::bias_bf16(self.ctx(), &projected, weights.bias.as_ref())?;
            modulation.reshape(vec![modulation.numel()])
        }

        fn prepare_step_modulation(&self, time_embedding: &Tensor) -> Result<Bf16StepModulation> {
            let conditioning = self.conditioning(time_embedding)?;
            let mut attention = Vec::with_capacity(self.config.action_expert.depth);
            let mut mlp = Vec::with_capacity(self.config.action_expert.depth);
            for layer in &self.weights.action_layers {
                attention.push(self.modulation(&conditioning, &layer.input_modulation)?);
                mlp.push(self.modulation(&conditioning, &layer.post_attention_modulation)?);
            }
            let final_norm =
                self.modulation(&conditioning, &self.weights.action_final_modulation)?;
            Ok(Bf16StepModulation {
                attention,
                mlp,
                final_norm,
            })
        }

        pub(super) fn prepare_all_modulation(
            &self,
            time_embeddings: &[Tensor],
        ) -> Result<Vec<Bf16StepModulation>> {
            if time_embeddings.len() != self.config.num_flow_steps {
                return Err(Error::Other(format!(
                    "π0.5 expected {} timestep embeddings, got {}",
                    self.config.num_flow_steps,
                    time_embeddings.len()
                )));
            }
            time_embeddings
                .iter()
                .map(|embedding| self.prepare_step_modulation(embedding))
                .collect()
        }

        fn denoise_step_with_modulation(
            &self,
            state: &Tensor,
            modulation: &Bf16StepModulation,
            prefix: &Bf16PrefixKvCache,
            dt: f32,
        ) -> Result<Tensor> {
            if prefix.keys.len() != self.config.action_expert.depth
                || prefix.values.len() != self.config.action_expert.depth
                || modulation.attention.len() != self.config.action_expert.depth
                || modulation.mlp.len() != self.config.action_expert.depth
            {
                return Err(Error::Other(
                    "π0.5 BF16 prefix/modulation depth mismatch".into(),
                ));
            }
            let hidden = gemm::bf16(self.ctx(), state, &self.weights.action_in.weight)?;
            let mut hidden =
                elementwise::bias_bf16(self.ctx(), &hidden, self.weights.action_in.bias.as_ref())?;
            let mut attention_normalized = None;
            for index in 0..self.config.action_expert.depth {
                let layer = &self.weights.action_layers[index];
                let next_norm_modulation = if index + 1 < self.config.action_expert.depth {
                    &modulation.attention[index + 1]
                } else {
                    &modulation.final_norm
                };
                let output = action_layer_bf16(
                    self.ctx(),
                    self.config.action_expert,
                    layer,
                    &hidden,
                    attention_normalized.as_ref(),
                    &modulation.attention[index],
                    &modulation.mlp[index],
                    next_norm_modulation,
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
            let velocity = elementwise::bias_bf16(
                self.ctx(),
                &velocity,
                self.weights.action_out.bias.as_ref(),
            )?;
            elementwise::euler_update_bf16(self.ctx(), state, &velocity, dt)
        }

        pub fn denoise_step(
            &self,
            state: &Tensor,
            time_embedding: &Tensor,
            prefix: &Bf16PrefixKvCache,
            dt: f32,
        ) -> Result<Tensor> {
            let modulation = self.prepare_step_modulation(time_embedding)?;
            self.denoise_step_with_modulation(state, &modulation, prefix, dt)
        }
    }

    impl super::super::Blocks for Bf16Blocks {
        type Prefix = Bf16PrefixKvCache;
        type StepModulation = Bf16StepModulation;
        fn config(&self) -> &Pi05Config {
            &self.config
        }
        fn vision(&self, patches: &Tensor, native: bool) -> Result<Tensor> {
            let _ = native;
            self.encode_vision(patches)
        }
        fn embed_prefix(&self, vision: &Tensor, ids: &CudaBuffer, count: usize) -> Result<Tensor> {
            self.embed_prefix(vision, ids, count)
        }
        fn prefix(&self, input: &Tensor) -> Result<Self::Prefix> {
            self.prefix_forward(input)
        }
        fn prepare_modulation(&self, embeddings: &[Tensor]) -> Result<Vec<Self::StepModulation>> {
            self.prepare_all_modulation(embeddings)
        }
        fn eager_modulation(
            &self,
            embeddings: &[Tensor],
        ) -> Result<Option<Vec<Self::StepModulation>>> {
            self.prepare_all_modulation(embeddings).map(Some)
        }
        fn step(
            &self,
            state: &Tensor,
            embedding: &Tensor,
            prefix: &Self::Prefix,
            dt: f32,
        ) -> Result<Tensor> {
            self.denoise_step(state, embedding, prefix, dt)
        }
        fn step_with_modulation(
            &self,
            state: &Tensor,
            modulation: &Self::StepModulation,
            prefix: &Self::Prefix,
            dt: f32,
        ) -> Result<Tensor> {
            self.denoise_step_with_modulation(state, modulation, prefix, dt)
        }
    }
}

impl crate::pi05::model::PrepareBlocks for backbone::Bf16Blocks {
    fn backend(&self) -> &std::sync::Arc<crate::pi05::backend::RuntimeBackend> {
        &self.backend
    }
    fn workspace_requirements(
        &self,
        tokens: usize,
    ) -> apxinf_core::Result<crate::pi05::model::WorkspaceRequirements> {
        Ok(crate::pi05::model::WorkspaceRequirements {
            bytes: self.graph_workspace_bytes(tokens)?,
            fp8_scratch: None,
        })
    }
    fn raw_patch_dtype(&self) -> apxinf_core::DType {
        apxinf_core::DType::BF16
    }
    fn preprocess(
        &self,
        images: &crate::pi05::backend::DeviceBuffer,
        patches: &Tensor,
        layout: crate::pi05::Pi05ImageLayout,
    ) -> Result<()> {
        crate::pi05::backend::kernels::preprocess::rgb_u8_to_patches_bf16(
            self.backend.context(),
            images,
            patches,
            self.config.num_views,
            self.config.image_size,
            self.config.patch_size,
            layout,
        )
    }
}
impl backbone::Bf16Blocks {
    fn graph_workspace_bytes(&self, token_count: usize) -> Result<usize> {
        let mut bytes = self.config.cuda_graph_workspace_bytes_bf16(token_count)?;
        if self.backend.context().caps().arch_family == apxinf_cuda::CudaArchFamily::Sm80 {
            bytes = bytes
                .checked_add(self.splitkv_workspace_bytes(token_count)?)
                .ok_or_else(|| Error::Other("pi05 BF16 split-KV workspace overflow".into()))?;
        }
        Ok(bytes)
    }

    fn splitkv_workspace_bytes(&self, token_count: usize) -> Result<usize> {
        let patches = self.config.num_views * self.config.patches_per_view();
        let prefix = patches
            .checked_add(token_count)
            .ok_or_else(|| Error::Other("pi05 split-KV prefix length overflow".into()))?;
        let horizon = self.config.action_horizon;
        let action = self.config.action_expert;
        if action.num_heads <= action.num_kv_heads || action.head_dim != 256 || horizon > 64 {
            return Ok(0);
        }
        let key_tokens = prefix
            .checked_add(horizon)
            .ok_or_else(|| Error::Other("pi05 split-KV key length overflow".into()))?;
        let max_splits = key_tokens.div_ceil(64).min(128);
        let lse = max_splits
            .checked_mul(horizon)
            .and_then(|value| value.checked_mul(action.num_heads))
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| Error::Other("pi05 split-KV LSE workspace overflow".into()))?;
        let output = max_splits
            .checked_mul(horizon)
            .and_then(|value| value.checked_mul(action.num_heads))
            .and_then(|value| value.checked_mul(action.head_dim))
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| Error::Other("pi05 split-KV output workspace overflow".into()))?;
        lse.checked_add(output)
            .and_then(|value| value.checked_mul(self.config.action_expert.depth))
            .and_then(|value| value.checked_mul(self.config.num_flow_steps))
            .ok_or_else(|| Error::Other("pi05 split-KV workspace overflow".into()))
    }
}
