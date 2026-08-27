//! Fixed-shape native-BF16 π0.5 inference runtime.

use std::sync::Arc;

use super::backend::{kernels, transfers, Context, DeviceBuffer as CudaBuffer, RuntimeBackend};
use apxinf_core::{Backend, DType, Error, Graph, Result, Tensor};
use apxinf_cuda::CudaArchFamily;
use half::bf16 as HalfBf16;
use kernels::{activation, cache, elementwise, embedding, gemm, norm, preprocess};

use super::{
    action_layer_bf16, language_layer_bf16, sinusoidal_time_embedding, vision_layer_bf16,
    vision_patch_embed_bf16, Bf16LinearWeights, Pi05Config, Pi05ImageLayout, StaticBf16Pi05Weights,
};

pub struct Bf16PrefixKvCache {
    pub keys: Vec<Tensor>,
    pub values: Vec<Tensor>,
    pub tokens: usize,
}

struct Bf16StepStyles {
    attention: Vec<Tensor>,
    mlp: Vec<Tensor>,
    final_norm: Tensor,
}

pub struct Pi05Bf16CapturedGraph {
    graph: Box<dyn Graph>,
    output: Tensor,
    patches: Tensor,
    raw_images: Option<CudaBuffer>,
    raw_image_layout: Option<Pi05ImageLayout>,
    noise: Tensor,
    _styles: Vec<Bf16StepStyles>,
    token_ids: CudaBuffer,
    token_count: usize,
    backend: Arc<RuntimeBackend>,
    _config: Arc<Pi05Config>,
    _weights: Arc<StaticBf16Pi05Weights>,
    workspace: kernels::GraphWorkspace,
}

impl Pi05Bf16CapturedGraph {
    pub fn replay(&self) -> Result<()> {
        self.graph.replay()
    }

    pub fn replay_and_synchronize(&self) -> Result<()> {
        self.graph.replay()?;
        self.backend.synchronize()
    }

    pub fn output(&self) -> &Tensor {
        &self.output
    }

    pub fn raw_image_layout(&self) -> Option<Pi05ImageLayout> {
        self.raw_image_layout
    }

    fn update_tokens(&self, token_ids: &[u32]) -> Result<()> {
        let bytes = token_ids
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        self.token_ids.copy_from_host(&bytes).map_err(Error::Cuda)
    }

    pub fn update_inputs(&self, patches: &Tensor, token_ids: &[u32], noise: &Tensor) -> Result<()> {
        self.update_inputs_without_noise(patches, token_ids)?;
        transfers::copy_cpu_to_cuda(noise, &self.noise)
    }

    pub fn update_inputs_without_noise(&self, patches: &Tensor, token_ids: &[u32]) -> Result<()> {
        if self.raw_images.is_some() {
            return Err(Error::Other(
                "π0.5 BF16 graph uses raw RGB input; call update_raw_image_inputs".into(),
            ));
        }
        if token_ids.len() != self.token_count {
            return Err(Error::Other(format!(
                "π0.5 BF16 graph expects {} token IDs, got {}",
                self.token_count,
                token_ids.len()
            )));
        }
        self.backend.synchronize()?;
        transfers::copy_cpu_to_cuda(patches, &self.patches)?;
        self.update_tokens(token_ids)
    }

    pub fn update_raw_image_inputs(
        &self,
        images: &[u8],
        token_ids: &[u32],
        noise: &Tensor,
    ) -> Result<()> {
        self.update_raw_image_inputs_without_noise(images, token_ids)?;
        transfers::copy_cpu_to_cuda(noise, &self.noise)
    }

    pub fn update_raw_image_inputs_without_noise(
        &self,
        images: &[u8],
        token_ids: &[u32],
    ) -> Result<()> {
        let raw_images = self.raw_images.as_ref().ok_or_else(|| {
            Error::Other("π0.5 BF16 graph uses patch input; call update_inputs".into())
        })?;
        if images.len() != raw_images.len() {
            return Err(Error::Other(format!(
                "π0.5 BF16 graph expects {} raw image bytes, got {}",
                raw_images.len(),
                images.len()
            )));
        }
        if token_ids.len() != self.token_count {
            return Err(Error::Other(format!(
                "π0.5 BF16 graph expects {} token IDs, got {}",
                self.token_count,
                token_ids.len()
            )));
        }
        self.backend.synchronize()?;
        raw_images.copy_from_host(images).map_err(Error::Cuda)?;
        self.update_tokens(token_ids)
    }

    pub fn workspace_bytes(&self) -> usize {
        self.workspace.capacity()
    }

    pub fn workspace_used_bytes(&self) -> usize {
        self.workspace.used()
    }
}

#[derive(Clone)]
pub struct Pi05Bf16CudaRuntime {
    backend: Arc<RuntimeBackend>,
    config: Arc<Pi05Config>,
    weights: Arc<StaticBf16Pi05Weights>,
}

impl Pi05Bf16CudaRuntime {
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

    fn graph_workspace_bytes(&self, token_count: usize) -> Result<usize> {
        let mut bytes = self.config.cuda_graph_workspace_bytes_bf16(token_count)?;
        if self.ctx().caps().arch_family == CudaArchFamily::Sm80 {
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

    fn prepare_all_styles(&self, time_embeddings: &[Tensor]) -> Result<Vec<Bf16StepStyles>> {
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
        let dt = -1.0 / self.config.num_flow_steps as f32;
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

    fn infer_with_styles(
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

    #[allow(clippy::too_many_arguments)]
    fn infer_captured_inputs(
        &self,
        patches: &Tensor,
        raw_images: Option<&CudaBuffer>,
        raw_image_layout: Option<Pi05ImageLayout>,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        styles: &[Bf16StepStyles],
    ) -> Result<Tensor> {
        match (raw_images, raw_image_layout) {
            (None, None) => self.infer_with_styles(patches, token_ids, token_count, noise, styles),
            (Some(images), Some(layout)) => {
                preprocess::rgb_u8_to_patches_bf16(
                    self.ctx(),
                    images,
                    patches,
                    self.config.num_views,
                    self.config.image_size,
                    self.config.patch_size,
                    layout,
                )?;
                self.infer_with_styles(patches, token_ids, token_count, noise, styles)
            }
            _ => Err(Error::Other(
                "π0.5 BF16 raw image capture state is inconsistent".into(),
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn capture_infer_impl(
        &self,
        patches: Tensor,
        raw_images: Option<CudaBuffer>,
        raw_image_layout: Option<Pi05ImageLayout>,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<Pi05Bf16CapturedGraph> {
        let backend = &self.backend;
        if raw_images.is_some() != raw_image_layout.is_some() {
            return Err(Error::Other(
                "π0.5 BF16 raw image capture state is inconsistent".into(),
            ));
        }
        let styles = self.prepare_all_styles(time_embeddings)?;
        backend.synchronize()?;
        let workspace = kernels::GraphWorkspace::new(
            self.graph_workspace_bytes(token_count)?,
            self.ctx().device_id(),
        )?;
        let eager_output = kernels::prepare_with_workspace(&workspace, || {
            self.infer_captured_inputs(
                &patches,
                raw_images.as_ref(),
                raw_image_layout,
                token_ids,
                token_count,
                noise,
                &styles,
            )
        })?;
        backend.synchronize()?;
        drop(eager_output);

        backend.begin_capture()?;
        let output = match kernels::with_workspace(&workspace, || {
            self.infer_captured_inputs(
                &patches,
                raw_images.as_ref(),
                raw_image_layout,
                token_ids,
                token_count,
                noise,
                &styles,
            )
        }) {
            Ok(output) => output,
            Err(error) => {
                let _ = backend.end_capture();
                return Err(error);
            }
        };
        let graph = backend.end_capture()?;
        Ok(Pi05Bf16CapturedGraph {
            graph,
            output,
            patches,
            raw_images,
            raw_image_layout,
            noise: noise.clone(),
            _styles: styles,
            token_ids: token_ids.clone(),
            token_count,
            backend: Arc::clone(&self.backend),
            _config: Arc::clone(&self.config),
            _weights: Arc::clone(&self.weights),
            workspace,
        })
    }

    pub fn capture_infer(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<Pi05Bf16CapturedGraph> {
        self.capture_infer_impl(
            patches.clone(),
            None,
            None,
            token_ids,
            token_count,
            noise,
            time_embeddings,
        )
    }

    pub fn capture_infer_rgb_u8(
        &self,
        layout: Pi05ImageLayout,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<Pi05Bf16CapturedGraph> {
        let backend = &self.backend;
        let raw_image_bytes =
            self.config.num_views * 3 * self.config.image_size * self.config.image_size;
        let raw_images = CudaBuffer::alloc_zeros(raw_image_bytes, self.ctx().device_id())
            .map_err(Error::Cuda)?;
        let patch_rows = self.config.num_views * self.config.patches_per_view();
        let patch_width = 3 * self.config.patch_size * self.config.patch_size;
        let patches =
            backend.to_device(&Tensor::zeros(vec![patch_rows, patch_width], DType::BF16))?;
        self.capture_infer_impl(
            patches,
            Some(raw_images),
            Some(layout),
            token_ids,
            token_count,
            noise,
            time_embeddings,
        )
    }
}

pub fn upload_time_embeddings_bf16(
    config: &Pi05Config,
    backend: &dyn Backend,
) -> Result<Vec<Tensor>> {
    (0..config.num_flow_steps)
        .map(|step| {
            let time = 1.0 - step as f32 / config.num_flow_steps as f32;
            let values = sinusoidal_time_embedding(
                time,
                config.action_expert.width,
                config.time_min_period,
                config.time_max_period,
            )
            .into_iter()
            .map(HalfBf16::from_f32)
            .collect::<Vec<_>>();
            backend.to_device(&Tensor::from_bf16(
                vec![1, config.action_expert.width],
                &values,
            )?)
        })
        .collect()
}
