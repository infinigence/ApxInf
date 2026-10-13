use std::sync::Arc;
use std::collections::BTreeMap;
use std::f64::consts::TAU;

use apxinf_core::{Backend, DType, Device, Error, Result, Tensor};
use half::{bf16, f16};

use crate::accelerator::cuda::{
    downcast_arc, kernels, CudaEventTimer, Context, DeviceBuffer, RuntimeBackend,
};
use crate::vla::{ImageLayout, Observation, VisionObservation};

use super::{SmolVlaConfig, SmolVlaWeights};

pub struct SmolVlaModel {
    backend: Arc<RuntimeBackend>,
    config: Arc<SmolVlaConfig>,
    weights: Arc<SmolVlaWeights>,
    fp16_gemm: bool,
}

#[derive(Clone, Debug, Default)]
pub struct SmolVlaInferenceTiming {
    pub preprocess_ms: f64,
    pub prefix_embedding_ms: f64,
    pub vlm_ms: f64,
    pub action_expert_ms: f64,
    pub output_ms: f64,
}

impl SmolVlaInferenceTiming {
    pub fn as_map(&self) -> BTreeMap<String, f64> {
        let vlm_prefix_ms = self.prefix_embedding_ms + self.vlm_ms;
        BTreeMap::from([
            ("preprocess_ms".to_string(), self.preprocess_ms),
            ("prefix_embedding_ms".to_string(), self.prefix_embedding_ms),
            ("vlm_ms".to_string(), self.vlm_ms),
            ("vlm_prefix_ms".to_string(), vlm_prefix_ms),
            ("action_expert_ms".to_string(), self.action_expert_ms),
            ("output_ms".to_string(), self.output_ms),
        ])
    }
}

impl SmolVlaModel {
    pub fn new(
        backend: Arc<RuntimeBackend>,
        config: Arc<SmolVlaConfig>,
        weights: Arc<SmolVlaWeights>,
    ) -> Result<Self> {
        config.validate()?;
        if weights.vision_layers.len() != config.vision_depth
            || weights.text_layers.len() != config.language_depth
            || weights.expert_layers.len() != config.language_depth
        {
            return Err(Error::Other("SmolVLA device weight depth mismatch".into()));
        }
        let fp16_gemm = weights.patch_embedding.weight.dtype() == DType::F16;
        if fp16_gemm && weights.action_out.weight.dtype() != DType::F16 {
            return Err(Error::Other(
                "SmolVLA FP16 GEMM weights are incomplete".into(),
            ));
        }
        Ok(Self {
            backend,
            config,
            weights,
            fp16_gemm,
        })
    }

    pub(in crate::smolvla) fn backend(&self) -> &Arc<RuntimeBackend> {
        &self.backend
    }

    pub fn uses_fp16_gemm(&self) -> bool {
        self.fp16_gemm
    }

    pub(in crate::smolvla) fn dtype(&self) -> DType {
        if self.fp16_gemm {
            DType::F16
        } else {
            DType::BF16
        }
    }

    pub(in crate::smolvla) fn config(&self) -> &Arc<SmolVlaConfig> {
        &self.config
    }

    fn add_bias(&self, ctx: &Context, input: &Tensor, bias: &Tensor) -> Result<Tensor> {
        if self.fp16_gemm {
            kernels::elementwise::bias_f16(ctx, input, Some(bias))
        } else {
            kernels::elementwise::bias_bf16(ctx, input, Some(bias))
        }
    }

    fn gemm(&self, ctx: &Context, activation: &Tensor, weight: &Tensor) -> Result<Tensor> {
        if !self.fp16_gemm {
            return kernels::gemm::bf16(ctx, activation, weight);
        }
        if activation.dtype() != DType::F16 || weight.dtype() != DType::F16 {
            return Err(Error::Other(
                format!(
                    "SmolVLA FP16 GEMM expects FP16 activation and weight, got activation {} and weight {}",
                    activation.dtype(),
                    weight.dtype()
                ),
            ));
        }
        kernels::gemm::matmul(ctx, activation, weight)
    }

    pub fn infer(
        &self,
        observation: &Observation,
        noise: &Tensor,
        token_ids: &DeviceBuffer,
    ) -> Result<Tensor> {
        self.infer_with_timing(observation, noise, token_ids)
            .map(|(output, _)| output)
    }

    pub fn infer_with_timing(
        &self,
        observation: &Observation,
        noise: &Tensor,
        token_ids: &DeviceBuffer,
    ) -> Result<(Tensor, SmolVlaInferenceTiming)> {
        let ctx = self.backend.context();
        let preprocess_timer = CudaEventTimer::new()?;
        preprocess_timer.start(ctx)?;
        let patches = match &observation.vision {
            VisionObservation::Patches(patches) => {
                let expected = [
                    self.config.num_views * self.config.patches_per_view(),
                    3 * self.config.patch_size * self.config.patch_size,
                ];
                if patches.dtype() != self.dtype() || patches.shape().dims() != expected {
                    return Err(Error::Other("SmolVLA patch input mismatch".into()));
                }
                self.backend.to_device(patches)?
            }
            VisionObservation::RgbU8 { bytes, layout } => {
                let expected = self.config.num_views * 3 * self.config.image_size.pow(2);
                if bytes.len() != expected {
                    return Err(Error::Other(format!(
                        "SmolVLA expected {expected} image bytes, got {}",
                        bytes.len()
                    )));
                }
                let raw = DeviceBuffer::alloc(expected, ctx.device_id()).map_err(Error::Cuda)?;
                raw.copy_from_host(bytes).map_err(Error::Cuda)?;
                let patches = self.backend.to_device(&Tensor::zeros(
                    vec![
                        self.config.num_views * self.config.patches_per_view(),
                        3 * self.config.patch_size * self.config.patch_size,
                    ],
                    self.dtype(),
                ))?;
                (if self.fp16_gemm {
                    kernels::preprocess::rgb_u8_to_patches_f16
                } else {
                    kernels::preprocess::rgb_u8_to_patches_bf16
                })(
                    ctx,
                    &raw,
                    &patches,
                    self.config.num_views,
                    self.config.image_size,
                    self.config.patch_size,
                    kernel_layout(*layout),
                )?;
                patches
            }
        };
        preprocess_timer.stop(ctx)?;
        let token_count = observation.token_ids.len();
        if token_count == 0 || token_count > self.config.max_token_len {
            return Err(Error::Other(format!(
                "SmolVLA token count must be in 1..={}, got {token_count}",
                self.config.max_token_len
            )));
        }
        if noise.dtype() != self.dtype()
            || noise.shape().dims() != [self.config.action_horizon, self.config.max_action_dim]
        {
            return Err(Error::Other("SmolVLA initial latent shape mismatch".into()));
        }
        let noise = if noise.device() == Device::Cpu {
            self.backend.to_device(noise)?
        } else {
            noise.clone()
        };
        let prefix_timer = CudaEventTimer::new()?;
        prefix_timer.start(ctx)?;
        let state = self.backend.to_device(&self.prepare_state(observation)?)?;
        let prefix = self.embed_prefix(&patches, token_ids, token_count, &state)?;
        prefix_timer.stop(ctx)?;
        let vlm_timer = CudaEventTimer::new()?;
        vlm_timer.start(ctx)?;
        let prefix_states = self.encode_prefix(prefix)?;
        vlm_timer.stop(ctx)?;
        let mut state = noise;
        let dt = -1.0 / self.config.num_flow_steps as f32;
        let expert_timer = CudaEventTimer::new()?;
        expert_timer.start(ctx)?;
        let time_embeddings = self.time_embeddings()?;
        for step in 0..self.config.num_flow_steps {
            let velocity = self.denoise_step_with_embedding(
                &state,
                &time_embeddings[step],
                &prefix_states,
            )?;
            if step == 0 {
            }
            state = (if self.fp16_gemm {
                kernels::elementwise::euler_update_f16
            } else {
                kernels::elementwise::euler_update_bf16
            })(ctx, &state, &velocity, dt)?;
        }
        expert_timer.stop(ctx)?;
        let output_timer = CudaEventTimer::new()?;
        output_timer.start(ctx)?;
        let output = (if self.fp16_gemm {
            kernels::quantization::slice_columns_f16
        } else {
            kernels::quantization::slice_columns_bf16
        })(ctx, &state, self.config.action_dim)?;
        output_timer.stop(ctx)?;

        self.backend.synchronize()?;
        let timing = SmolVlaInferenceTiming {
            preprocess_ms: preprocess_timer.elapsed_ms()?,
            prefix_embedding_ms: prefix_timer.elapsed_ms()?,
            vlm_ms: vlm_timer.elapsed_ms()?,
            action_expert_ms: expert_timer.elapsed_ms()?,
            output_ms: output_timer.elapsed_ms()?,
        };
        Ok((output, timing))
    }

    pub(in crate::smolvla) fn time_embeddings(&self) -> Result<Vec<Tensor>> {
        let dt = -1.0 / self.config.num_flow_steps as f32;
        (0..self.config.num_flow_steps)
            .map(|step| self.time_embedding(1.0 + step as f32 * dt))
            .collect()
    }

    pub(in crate::smolvla) fn infer_fixed(
        &self,
        raw_images: Option<&DeviceBuffer>,
        patches: &Tensor,
        state: &Tensor,
        noise: &Tensor,
        token_ids: &DeviceBuffer,
        token_count: usize,
        layout: ImageLayout,
        time_embeddings: &[Tensor],
    ) -> Result<Tensor> {
        if let Some(raw_images) = raw_images {
            (if self.fp16_gemm {
                kernels::preprocess::rgb_u8_to_patches_f16
            } else {
                kernels::preprocess::rgb_u8_to_patches_bf16
            })(
                self.backend.context(),
                raw_images,
                patches,
                self.config.num_views,
                self.config.image_size,
                self.config.patch_size,
                kernel_layout(layout),
            )?;
        }
        self.infer_parts(
            patches,
            state,
            noise,
            token_ids,
            token_count,
            time_embeddings,
        )
    }

    fn infer_parts(
        &self,
        patches: &Tensor,
        state: &Tensor,
        noise: &Tensor,
        token_ids: &DeviceBuffer,
        token_count: usize,
        time_embeddings: &[Tensor],
    ) -> Result<Tensor> {
        if token_count == 0 || token_count > self.config.max_token_len {
            return Err(Error::Other(format!(
                "SmolVLA token count must be in 1..={}, got {token_count}",
                self.config.max_token_len
            )));
        }
        if noise.dtype() != self.dtype()
            || noise.shape().dims() != [self.config.action_horizon, self.config.max_action_dim]
        {
            return Err(Error::Other("SmolVLA initial latent shape mismatch".into()));
        }
        let ctx = self.backend.context();
        let prefix = self.embed_prefix(patches, token_ids, token_count, state)?;
        let prefix_states = self.encode_prefix(prefix)?;
        let mut state = noise.clone();
        let dt = -1.0 / self.config.num_flow_steps as f32;
        for embedding in time_embeddings {
            let velocity = self.denoise_step_with_embedding(&state, embedding, &prefix_states)?;
            state = (if self.fp16_gemm {
                kernels::elementwise::euler_update_f16
            } else {
                kernels::elementwise::euler_update_bf16
            })(ctx, &state, &velocity, dt)?;
        }
        (if self.fp16_gemm {
            kernels::quantization::slice_columns_f16
        } else {
            kernels::quantization::slice_columns_bf16
        })(ctx, &state, self.config.action_dim)
    }

    fn embed_prefix(
        &self,
        patches: &Tensor,
        token_ids: &DeviceBuffer,
        token_count: usize,
        state: &Tensor,
    ) -> Result<Tensor> {
        let ctx = self.backend.context();
        let vision = self.encode_vision(ctx, patches)?;
        let language = (if self.fp16_gemm {
            kernels::embedding::lookup_f16
        } else {
            kernels::embedding::lookup_bf16
        })(
            ctx,
            &self.weights.token_embedding,
            token_ids,
            token_count,
        )?;
        let state = self.embed_state(state)?;
        let concat_rows = if self.fp16_gemm {
            kernels::elementwise::concat_rows_f16
        } else {
            kernels::elementwise::concat_rows_bf16
        };
        let prefix = concat_rows(ctx, &vision, &language)?;
        concat_rows(ctx, &prefix, &state)
    }

    fn encode_vision(&self, ctx: &Context, patches: &Tensor) -> Result<Tensor> {
        let projection = self.gemm(
            ctx,
            patches,
            &self.weights.patch_embedding.weight,
        )?;
        let mut hidden = (if self.fp16_gemm {
            kernels::embedding::add_position_f16
        } else {
            kernels::embedding::add_position_bf16
        })(
            ctx,
            &projection,
            self.weights.patch_embedding.bias.as_ref(),
            &self.weights.position_embedding,
            self.config.patches_per_view(),
        )?;
        for layer in &self.weights.vision_layers {
            hidden = self.vision_layer(ctx, layer, &hidden)?;
        }
        let hidden = (if self.fp16_gemm {
            kernels::norm::layer_f16
        } else {
            kernels::norm::layer_bf16
        })(
            ctx,
            &hidden,
            &self.weights.vision_post_norm.weight,
            self.weights.vision_post_norm.bias.as_ref().unwrap(),
            self.config.layer_norm_eps,
        )?;
        let shuffled = (if self.fp16_gemm {
            kernels::preprocess::pixel_shuffle_4_f16
        } else {
            kernels::preprocess::pixel_shuffle_4_bf16
        })(
            ctx,
            &hidden,
            self.config.patches_per_view(),
        )?;
        let projected = self.gemm(ctx, &shuffled, &self.weights.connector.weight)?;
        kernels::elementwise::scale(
            ctx,
            &projected,
            (self.config.language_width as f32).sqrt(),
        )
    }

    fn vision_layer(
        &self,
        ctx: &Context,
        weights: &super::weights::VisionLayer,
        input: &Tensor,
    ) -> Result<Tensor> {
        let normalized = (if self.fp16_gemm {
            kernels::norm::layer_f16
        } else {
            kernels::norm::layer_bf16
        })(
            ctx,
            input,
            &weights.norm1.weight,
            weights.norm1.bias.as_ref().unwrap(),
            self.config.layer_norm_eps,
        )?;
        let qkv = self.gemm(ctx, &normalized, &weights.qkv.weight)?;
        let qkv = (if self.fp16_gemm {
            kernels::attention::split_qkv_bias_f16
        } else {
            kernels::attention::split_qkv_bias_bf16
        })(
            ctx,
            &qkv,
            weights.qkv.bias.as_ref(),
            self.config.vision_heads,
            self.config.vision_width / self.config.vision_heads,
        )?;
        let attention = if self.fp16_gemm {
            kernels::attention::vision_mha_f16(
                ctx,
                &qkv.q,
                &qkv.k,
                &qkv.v,
                self.config.patches_per_view(),
            )?
        } else {
            kernels::attention::vision_mha_bf16(
                ctx,
                &qkv.q,
                &qkv.k,
                &qkv.v,
                self.config.patches_per_view(),
            )?
        }
        .reshape(vec![input.shape().dims()[0], self.config.vision_width])?;
        let projected =
            self.gemm(ctx, &attention, &weights.attention_output.weight)?;
        let hidden = if self.fp16_gemm {
            let projected = self.add_bias(
                ctx,
                &projected,
                weights.attention_output.bias.as_ref().unwrap(),
            )?;
            self.backend.add(&projected, input)?
        } else {
            kernels::fused::bias_then_residual_bf16_packed4(
                ctx,
                &projected,
                Some(weights.attention_output.bias.as_ref().unwrap()),
                input,
            )?
        };
        let normalized = (if self.fp16_gemm {
            kernels::norm::layer_f16
        } else {
            kernels::norm::layer_bf16
        })(
            ctx,
            &hidden,
            &weights.norm2.weight,
            weights.norm2.bias.as_ref().unwrap(),
            self.config.layer_norm_eps,
        )?;
        let activated = self.gemm(ctx, &normalized, &weights.fc1.weight)?;
        let activated = (if self.fp16_gemm {
            kernels::activation::bias_gelu_f16
        } else {
            kernels::activation::bias_gelu_bf16
        })(
            ctx,
            &activated,
            weights.fc1.bias.as_ref(),
        )?;
        let projected = self.gemm(ctx, &activated, &weights.fc2.weight)?;
        if self.fp16_gemm {
            let projected =
                self.add_bias(ctx, &projected, weights.fc2.bias.as_ref().unwrap())?;
            self.backend.add(&projected, &hidden)
        } else {
            kernels::fused::bias_then_residual_bf16_packed4(
                ctx,
                &projected,
                Some(weights.fc2.bias.as_ref().unwrap()),
                &hidden,
            )
        }
    }

    pub(in crate::smolvla) fn prepare_state(&self, observation: &Observation) -> Result<Tensor> {
        let state = observation.state.as_ref().ok_or_else(|| {
            Error::Other("SmolVLA observation requires proprioceptive state".into())
        })?;
        if state.dtype() != DType::F32 && state.dtype() != DType::BF16 {
            return Err(Error::Other("SmolVLA state must be F32 or BF16".into()));
        }
        if state.device() != Device::Cpu {
            return Err(Error::Other("SmolVLA state must be a CPU tensor".into()));
        }
        let values = state.to_f32_vec()?;
        if values.len() > self.config.max_state_dim {
            return Err(Error::Other("SmolVLA state exceeds maximum dimension".into()));
        }
        let mut padded = vec![0.0; self.config.max_state_dim];
        padded[..values.len()].copy_from_slice(&values);
        let host = if self.fp16_gemm {
            Tensor::from_f16(
                vec![1, self.config.max_state_dim],
                &padded.into_iter().map(f16::from_f32).collect::<Vec<_>>(),
            )?
        } else {
            Tensor::from_bf16(
                vec![1, self.config.max_state_dim],
                &padded.into_iter().map(bf16::from_f32).collect::<Vec<_>>(),
            )?
        };
        Ok(host)
    }

    fn embed_state(&self, state: &Tensor) -> Result<Tensor> {
        if state.device() != Device::Cuda(self.backend.device_id())
            || state.dtype() != self.dtype()
            || state.shape().dims() != [1, self.config.max_state_dim]
        {
            return Err(Error::Other(
                "SmolVLA fixed state input shape or dtype mismatch".into(),
            ));
        }
        let projected = self.gemm(
            self.backend.context(),
            state,
            &self.weights.state_projection.weight,
        )?;
        self.add_bias(
            self.backend.context(),
            &projected,
            self.weights.state_projection.bias.as_ref().unwrap(),
        )
    }

    fn encode_prefix(&self, prefix: Tensor) -> Result<PrefixStates> {
        let ctx = self.backend.context();
        let mut hidden = prefix;
        let mut keys = Vec::with_capacity(self.config.language_depth);
        let mut values = Vec::with_capacity(self.config.language_depth);
        for layer in &self.weights.text_layers {
            let normalized = (if self.fp16_gemm {
                kernels::norm::rms_f16
            } else {
                kernels::norm::rms_bf16
            })(
                ctx,
                &hidden,
                &layer.input_norm,
                self.config.rms_norm_eps,
            )?;
            let qkv = self.gemm(ctx, &normalized, &layer.qkv.weight)?;
            let qkv = (if self.fp16_gemm {
                kernels::rope::split_qkv_apply_f16
            } else {
                kernels::rope::split_qkv_apply_bf16
            })(
                ctx,
                &qkv,
                layer.qkv.bias.as_ref(),
                self.config.language_heads,
                self.config.language_kv_heads,
                self.config.language_width / self.config.language_heads,
                self.config.rope_theta,
                0,
            )?;
            let tokens = hidden.shape().dims()[0];
            let kv_width = self.config.language_kv_heads
                * (self.config.language_width / self.config.language_heads);
            keys.push(qkv.k.clone().reshape(vec![tokens, kv_width])?);
            values.push(qkv.v.clone().reshape(vec![tokens, kv_width])?);
            let attention = (if self.fp16_gemm {
                kernels::attention::prefix_gqa_f16
            } else {
                kernels::attention::prefix_gqa_bf16
            })(
                ctx,
                &qkv.q,
                &qkv.k,
                &qkv.v,
                tokens - 1,
            )?
            .reshape(vec![tokens, self.config.language_width])?;
            if keys.len() == 1 {
            }
            let projected =
                self.gemm(ctx, &attention, &layer.attention_output.weight)?;
            hidden = self.backend.add(&projected, &hidden)?;
            let mlp_output =
                self.mlp(ctx, &hidden, &layer.post_norm, &layer.gate_up, &layer.down)?;
            if keys.len() == 1 {
            }
            hidden = mlp_output;
        }
        let hidden = (if self.fp16_gemm {
            kernels::norm::rms_f16
        } else {
            kernels::norm::rms_bf16
        })(
            ctx,
            &hidden,
            &self.weights.text_norm,
            self.config.rms_norm_eps,
        )?;
        let mut cross_keys = Vec::new();
        let mut cross_values = Vec::new();
        for (layer_index, layer) in self.weights.expert_layers.iter().enumerate() {
            let weights = match layer {
                super::weights::ExpertLayer::CrossAttention(weights) => weights,
                super::weights::ExpertLayer::SelfAttention(_) => continue,
            };
            let prefix_tokens = keys[layer_index].shape().dims()[0];
            let key = self.gemm(
                ctx,
                &keys[layer_index].reshape(vec![prefix_tokens, 320])?,
                &weights.key.weight,
            )?;
            let value = self.gemm(
                ctx,
                &values[layer_index].reshape(vec![prefix_tokens, 320])?,
                &weights.value.weight,
            )?;
            let kv_heads = self.config.language_kv_heads;
            cross_keys.push(key.reshape(vec![prefix_tokens, kv_heads, 64])?);
            cross_values.push(value.reshape(vec![prefix_tokens, kv_heads, 64])?);
        }
        Ok(PrefixStates {
            keys,
            values,
            cross_keys,
            cross_values,
            hidden,
        })
    }

    fn mlp(
        &self,
        ctx: &Context,
        input: &Tensor,
        norm: &Tensor,
        gate_up: &super::weights::Linear,
        down: &super::weights::Linear,
    ) -> Result<Tensor> {
        let normalized = (if self.fp16_gemm {
            kernels::norm::rms_f16
        } else {
            kernels::norm::rms_bf16
        })(ctx, input, norm, self.config.rms_norm_eps)?;
        let activated =
            self.gemm(ctx, &normalized, &gate_up.weight)?;
        let activated = (if self.fp16_gemm {
            kernels::activation::swiglu_f16
        } else {
            kernels::activation::swiglu_bf16
        })(ctx, &activated)?;
        let projected = self.gemm(ctx, &activated, &down.weight)?;
        self.backend.add(&projected, input)
    }

    fn denoise_step(
        &self,
        state: &Tensor,
        time: f32,
        prefix: &PrefixStates,
    ) -> Result<Tensor> {
        let embedding = self.time_embedding(time)?;
        self.denoise_step_with_embedding(state, &embedding, prefix)
    }

    fn denoise_step_with_embedding(
        &self,
        state: &Tensor,
        embedding: &Tensor,
        prefix: &PrefixStates,
    ) -> Result<Tensor> {
        let ctx = self.backend.context();
        let action = self.gemm(ctx, state, &self.weights.action_in.weight)?;
        let action = self.add_bias(
            ctx,
            &action,
            self.weights.action_in.bias.as_ref().unwrap(),
        )?;
        let action_time = (if self.fp16_gemm {
            kernels::elementwise::concat_columns_f16
        } else {
            kernels::elementwise::concat_columns_bf16
        })(
            ctx,
            &[&action, embedding],
        )?;
        let fused = self.gemm(
            ctx,
            &action_time,
            &self.weights.action_time_in.weight,
        )?;
        let fused = self.add_bias(
            ctx,
            &fused,
            self.weights.action_time_in.bias.as_ref().unwrap(),
        )?;
        let fused = if self.fp16_gemm {
            kernels::activation::bias_silu_f16(ctx, &fused, None)?
        } else {
            kernels::activation::silu(ctx, &fused)?
        };
        let mut hidden = self.gemm(ctx, &fused, &self.weights.action_time_out.weight)?;
        hidden = self.add_bias(
            ctx,
            &hidden,
            self.weights.action_time_out.bias.as_ref().unwrap(),
        )?;
        let mut cross_layer_index = 0;
        for (layer_index, layer) in self.weights.expert_layers.iter().enumerate() {
            hidden = match layer {
                super::weights::ExpertLayer::SelfAttention(weights) => self.expert_self_layer(
                    ctx,
                    weights,
                    &hidden,
                    &prefix.keys[layer_index],
                    &prefix.values[layer_index],
                )?,
                super::weights::ExpertLayer::CrossAttention(weights) => {
                    let key = &prefix.cross_keys[cross_layer_index];
                    let value = &prefix.cross_values[cross_layer_index];
                    cross_layer_index += 1;
                    self.expert_cross_layer(ctx, weights, &hidden, key, value)?
                }
            };
        }
        let hidden = (if self.fp16_gemm {
            kernels::norm::rms_f16
        } else {
            kernels::norm::rms_bf16
        })(
            ctx,
            &hidden,
            &self.weights.expert_norm,
            self.config.rms_norm_eps,
        )?;
        let velocity = self.gemm(ctx, &hidden, &self.weights.action_out.weight)?;
        self.add_bias(ctx, &velocity, self.weights.action_out.bias.as_ref().unwrap())
    }

    #[allow(clippy::too_many_arguments)]
    fn expert_self_layer(
        &self,
        ctx: &Context,
        weights: &super::weights::ExpertSelfLayer,
        input: &Tensor,
        prefix_key: &Tensor,
        prefix_value: &Tensor,
    ) -> Result<Tensor> {
        let normalized = (if self.fp16_gemm {
            kernels::norm::rms_f16
        } else {
            kernels::norm::rms_bf16
        })(
            ctx,
            input,
            &weights.input_norm,
            self.config.rms_norm_eps,
        )?;
        let qkv = self.gemm(ctx, &normalized, &weights.qkv.weight)?;
        let prefix_tokens = prefix_key.shape().dims()[0];
        let qkv = (if self.fp16_gemm {
            kernels::rope::split_qkv_apply_f16
        } else {
            kernels::rope::split_qkv_apply_bf16
        })(
            ctx,
            &qkv,
            weights.qkv.bias.as_ref(),
            self.config.language_heads,
            self.config.language_kv_heads,
            64,
            self.config.rope_theta,
            prefix_tokens,
        )?;
        let input_tokens = input.shape().dims()[0];
        let kv_width = self.config.language_kv_heads * 64;
        let concat_rows = if self.fp16_gemm {
            kernels::elementwise::concat_rows_f16
        } else {
            kernels::elementwise::concat_rows_bf16
        };
        let key = concat_rows(
            ctx,
            prefix_key,
            &qkv.k.reshape(vec![input_tokens, kv_width])?,
        )?;
        let value = concat_rows(
            ctx,
            prefix_value,
            &qkv.v.reshape(vec![input_tokens, kv_width])?,
        )?;
        let key = key.reshape(vec![
            key.shape().dims()[0],
            self.config.language_kv_heads,
            64,
        ])?;
        let value = value.reshape(vec![
            value.shape().dims()[0],
            self.config.language_kv_heads,
            64,
        ])?;
        let key_tokens = key.shape().dims()[0];
        let attention = if self.fp16_gemm {
            kernels::attention::suffix_causal_gqa_f16(
                ctx,
                &qkv.q,
                &key,
                &value,
                key_tokens,
                key_tokens - input_tokens,
            )?
        } else {
            kernels::attention::causal_gqa_bf16(
                ctx,
                &qkv.q,
                &key,
                &value,
                key_tokens,
            )?
        }
        .reshape(vec![input_tokens, self.config.language_width])?;
        let projected =
            self.gemm(ctx, &attention, &weights.attention_output.weight)?;
        let hidden = self.backend.add(&projected, input)?;
        self.mlp(ctx, &hidden, &weights.post_norm, &weights.gate_up, &weights.down)
    }

    fn expert_cross_layer(
        &self,
        ctx: &Context,
        weights: &super::weights::ExpertCrossLayer,
        input: &Tensor,
        key: &Tensor,
        value: &Tensor,
    ) -> Result<Tensor> {
        let normalized = (if self.fp16_gemm {
            kernels::norm::rms_f16
        } else {
            kernels::norm::rms_bf16
        })(
            ctx,
            input,
            &weights.input_norm,
            self.config.rms_norm_eps,
        )?;
        let query = self.gemm(ctx, &normalized, &weights.query.weight)?;
        let query = query.reshape(vec![
            input.shape().dims()[0],
            self.config.language_heads,
            64,
        ])?;
        let query = if self.fp16_gemm {
            kernels::rope::apply_half_split_f16(
                ctx,
                &query,
                self.config.language_heads,
                64,
                self.config.rope_theta,
                0,
            )?
        } else {
            kernels::rope::apply_batched(
                ctx,
                &query,
                self.config.language_heads,
                64,
                self.config.rope_theta,
                0,
            )?
        };
        let prefix_tokens = key.shape().dims()[0];
        let attention = (if self.fp16_gemm {
            kernels::attention::full_gqa_f16
        } else {
            kernels::attention::full_gqa_bf16
        })(
            ctx,
            &query,
            &key,
            &value,
            prefix_tokens,
        )?
        .reshape(vec![input.shape().dims()[0], self.config.language_width])?;
        let projected =
            self.gemm(ctx, &attention, &weights.attention_output.weight)?;
        let hidden = self.backend.add(&projected, input)?;
        self.mlp(ctx, &hidden, &weights.post_norm, &weights.gate_up, &weights.down)
    }

    fn time_embedding(&self, time: f32) -> Result<Tensor> {
        let base = sinusoidal_time_embedding(
            time,
            self.config.expert_width,
            self.config.time_min_period,
            self.config.time_max_period,
        );
        let mut values = Vec::with_capacity(
            self.config.action_horizon * self.config.expert_width,
        );
        for _ in 0..self.config.action_horizon {
            values.extend_from_slice(&base);
        }
        let host = if self.fp16_gemm {
            let row = values.iter().map(|value| f16::from_f32(*value)).collect::<Vec<_>>();
            Tensor::from_f16(
                vec![self.config.action_horizon, self.config.expert_width],
                &row,
            )?
        } else {
            let row = values.iter().map(|value| bf16::from_f32(*value)).collect::<Vec<_>>();
            Tensor::from_bf16(
                vec![self.config.action_horizon, self.config.expert_width],
                &row,
            )?
        };
        self.backend.to_device(&host)
    }
}

fn sinusoidal_time_embedding(
    time: f32,
    dimension: usize,
    min_period: f32,
    max_period: f32,
) -> Vec<f32> {
    assert_eq!(dimension % 2, 0);
    let half = dimension / 2;
    let mut sin = Vec::with_capacity(half);
    let mut cos = Vec::with_capacity(half);
    for index in 0..half {
        let fraction = if half == 1 {
            0.0
        } else {
            index as f64 / (half - 1) as f64
        };
        let period = min_period as f64 * (max_period as f64 / min_period as f64).powf(fraction);
        let phase = time as f64 * TAU / period;
        sin.push(phase.sin() as f32);
        cos.push(phase.cos() as f32);
    }
    sin.extend(cos);
    sin
}

pub struct PrefixStates {
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
    cross_keys: Vec<Tensor>,
    cross_values: Vec<Tensor>,
    hidden: Tensor,
}


fn kernel_layout(layout: crate::vla::ImageLayout) -> kernels::preprocess::ImageLayout {
    match layout {
        crate::vla::ImageLayout::Nhwc => kernels::preprocess::ImageLayout::Nhwc,
        crate::vla::ImageLayout::Nchw => kernels::preprocess::ImageLayout::Nchw,
    }
}

pub fn downcast_backend(backend: Arc<dyn Backend>) -> Result<Arc<RuntimeBackend>> {
    downcast_arc(backend)
        .ok_or_else(|| Error::Other("SmolVLA is only registered for CUDA".into()))
}
