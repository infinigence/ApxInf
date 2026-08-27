//! Static-shape π0.5 inference orchestration for CUDA.

use std::sync::Arc;

pub use super::backend::ImageLayout as Pi05ImageLayout;
use apxinf_core::{Backend, Error, Graph, Result, Tensor};
use half::f16;

use super::backend::{kernels, transfers, Context, DeviceBuffer as CudaBuffer, RuntimeBackend};
use kernels::{activation, cache, elementwise, embedding, gemm, norm, preprocess, quantization};

use super::StaticFp8Calibration;
use super::{
    action_layer, language_layer, sinusoidal_time_embedding, vision_layer, vision_patch_embed_fp8,
    vision_qkv_packed_from_env, Pi05Config, StaticFp8Pi05Weights, TransformerLayerScales,
    VisionLayerScales,
};

#[derive(Clone, Debug)]
pub struct Pi05ActivationScales {
    pub vision_patch_input: f32,
    pub vision_layers: Vec<VisionLayerScales>,
    pub vision_post_norm: f32,
    pub language_layers: Vec<TransformerLayerScales>,
    pub action_input: f32,
    pub time_input: f32,
    pub time_hidden: f32,
    pub conditioning: f32,
    pub action_layers: Vec<TransformerLayerScales>,
    pub action_final_norm: f32,
}

impl Pi05ActivationScales {
    /// Resolve every graph activation scale from a named calibration file.
    pub fn from_calibration(
        config: &Pi05Config,
        calibration: &StaticFp8Calibration,
    ) -> Result<Self> {
        let transformer_layer = |prefix: &str, index: usize| -> Result<TransformerLayerScales> {
            Ok(TransformerLayerScales {
                attention_norm: calibration
                    .scale(&format!("{prefix}.layers.{index}.attention_norm"))?,
                attention_output: calibration
                    .scale(&format!("{prefix}.layers.{index}.attention_output"))?,
                mlp_norm: calibration.scale(&format!("{prefix}.layers.{index}.mlp_norm"))?,
                mlp_activation: calibration
                    .scale(&format!("{prefix}.layers.{index}.mlp_activation"))?,
            })
        };
        let vision_layers = (0..config.vision_depth)
            .map(|index| {
                Ok(VisionLayerScales {
                    attention_norm: calibration
                        .scale(&format!("vision.layers.{index}.attention_norm"))?,
                    attention_output: calibration
                        .scale(&format!("vision.layers.{index}.attention_output"))?,
                    mlp_norm: calibration.scale(&format!("vision.layers.{index}.mlp_norm"))?,
                    mlp_activation: calibration
                        .scale(&format!("vision.layers.{index}.mlp_activation"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let language_layers = (0..config.language.depth)
            .map(|index| transformer_layer("language", index))
            .collect::<Result<Vec<_>>>()?;
        let action_layers = (0..config.action_expert.depth)
            .map(|index| transformer_layer("action", index))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            vision_patch_input: calibration.scale("vision.patch_input")?,
            vision_layers,
            vision_post_norm: calibration.scale("vision.post_norm")?,
            language_layers,
            action_input: calibration.scale("action.input")?,
            time_input: calibration.scale("time.input")?,
            time_hidden: calibration.scale("time.hidden")?,
            conditioning: calibration.scale("action.conditioning")?,
            action_layers,
            action_final_norm: calibration.scale("action.final_norm")?,
        })
    }

    /// Useful for kernel smoke tests. Production inference should load named,
    /// measured scales from `StaticFp8Calibration`.
    pub fn uniform(config: &Pi05Config, scale: f32) -> Result<Self> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(Error::Other(format!("invalid uniform FP8 scale {scale}")));
        }
        let transformer = TransformerLayerScales {
            attention_norm: scale,
            attention_output: scale,
            mlp_norm: scale,
            mlp_activation: scale,
        };
        let vision = VisionLayerScales {
            attention_norm: scale,
            attention_output: scale,
            mlp_norm: scale,
            mlp_activation: scale,
        };
        Ok(Self {
            vision_patch_input: scale,
            vision_layers: vec![vision; config.vision_depth],
            vision_post_norm: scale,
            language_layers: vec![transformer; config.language.depth],
            action_input: scale,
            time_input: scale,
            time_hidden: scale,
            conditioning: scale,
            action_layers: vec![transformer; config.action_expert.depth],
            action_final_norm: scale,
        })
    }

    fn validate(&self, config: &Pi05Config) -> Result<()> {
        if self.vision_layers.len() != config.vision_depth
            || self.language_layers.len() != config.language.depth
            || self.action_layers.len() != config.action_expert.depth
        {
            return Err(Error::Other(
                "π0.5 activation calibration depth mismatch".into(),
            ));
        }
        Ok(())
    }
}

pub struct PrefixKvCache {
    pub keys: Vec<Tensor>,
    pub values: Vec<Tensor>,
    pub tokens: usize,
}

struct Pi05StepStyles {
    attention: Vec<Tensor>,
    mlp: Vec<Tensor>,
    final_norm: Tensor,
}

/// A fixed-address, replayable full π0.5 inference graph.
///
/// Input tensor contents may be updated in place between replays, but their
/// addresses and `token_count` must remain unchanged. Shared owners keep the
/// runtime weights, backend, and all fixed-address inputs alive.
pub struct Pi05CapturedGraph {
    // Drop the executable before any memory it references.
    graph: Box<dyn Graph>,
    output: Tensor,
    patches: Tensor,
    raw_images: Option<CudaBuffer>,
    raw_image_layout: Option<Pi05ImageLayout>,
    noise: Tensor,
    _styles: Vec<Pi05StepStyles>,
    token_ids: CudaBuffer,
    token_count: usize,
    backend: Arc<RuntimeBackend>,
    _config: Arc<Pi05Config>,
    _weights: Arc<StaticFp8Pi05Weights>,
    _scales: Arc<Pi05ActivationScales>,
    workspace: kernels::GraphWorkspace,
}

impl Pi05CapturedGraph {
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

    /// Raw-image layout captured into the graph, or `None` for the legacy
    /// normalized FP16 patch input path.
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

    /// Replace captured inputs while preserving every device address.
    /// `patches` and `noise` must be CPU tensors with the captured shapes.
    pub fn update_inputs(&self, patches: &Tensor, token_ids: &[u32], noise: &Tensor) -> Result<()> {
        self.update_inputs_without_noise(patches, token_ids)?;
        transfers::copy_cpu_to_cuda(noise, &self.noise)
    }

    /// Replace non-random captured inputs while retaining the existing device
    /// latent. Used when a bound device generator fills `noise` in place.
    pub fn update_inputs_without_noise(&self, patches: &Tensor, token_ids: &[u32]) -> Result<()> {
        if self.raw_images.is_some() {
            return Err(Error::Other(
                "π0.5 graph was captured for raw RGB input; use update_raw_image_inputs".into(),
            ));
        }
        if token_ids.len() != self.token_count {
            return Err(Error::Other(format!(
                "π0.5 captured graph expects {} token IDs, got {}",
                self.token_count,
                token_ids.len()
            )));
        }
        // Do not overwrite an input while a preceding replay still reads it.
        self.backend.synchronize()?;
        transfers::copy_cpu_to_cuda(patches, &self.patches)?;
        self.update_tokens(token_ids)
    }

    /// Replace a captured raw RGB `uint8` batch and the shared prompt/noise
    /// inputs without changing any graph-visible device address.
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
            Error::Other("π0.5 graph was captured for FP16 patches; use update_inputs".into())
        })?;
        if images.len() != raw_images.len() {
            return Err(Error::Other(format!(
                "π0.5 captured graph expects {} raw image bytes, got {}",
                raw_images.len(),
                images.len()
            )));
        }
        if token_ids.len() != self.token_count {
            return Err(Error::Other(format!(
                "π0.5 captured graph expects {} token IDs, got {}",
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
pub struct Pi05CudaRuntime {
    backend: Arc<RuntimeBackend>,
    config: Arc<Pi05Config>,
    weights: Arc<StaticFp8Pi05Weights>,
    scales: Arc<Pi05ActivationScales>,
    packed_vision_qkv: bool,
}

impl Pi05CudaRuntime {
    pub fn new(
        backend: Arc<RuntimeBackend>,
        config: Arc<Pi05Config>,
        weights: Arc<StaticFp8Pi05Weights>,
        scales: Arc<Pi05ActivationScales>,
    ) -> Result<Self> {
        config.validate()?;
        scales.validate(&config)?;
        let packed_vision_qkv = vision_qkv_packed_from_env()?;
        if weights.vision_layers.len() != config.vision_depth
            || weights.language_layers.len() != config.language.depth
            || weights.action_layers.len() != config.action_expert.depth
        {
            return Err(Error::Other("π0.5 device weight depth mismatch".into()));
        }
        Ok(Self {
            backend,
            config,
            weights,
            scales,
            packed_vision_qkv,
        })
    }

    fn ctx(&self) -> &Context {
        self.backend.context()
    }

    fn encode_vision_fp8_patches(&self, patches: &Tensor) -> Result<Tensor> {
        let mut hidden = vision_patch_embed_fp8(
            self.ctx(),
            &self.weights.patch_embedding,
            &self.weights.position_embedding,
            patches,
            self.config.patches_per_view(),
            self.scales.vision_patch_input,
        )?;
        for (layer, scale) in self
            .weights
            .vision_layers
            .iter()
            .zip(&self.scales.vision_layers)
        {
            hidden = vision_layer(
                self.ctx(),
                layer,
                *scale,
                &hidden,
                self.config.patches_per_view(),
                self.config.vision_heads,
                self.config.vision_head_dim,
                self.packed_vision_qkv,
                self.config.layer_norm_eps,
            )?;
        }
        let hidden = norm::layer_quant_f16_e4m3(
            self.ctx(),
            &hidden,
            &self.weights.vision_post_norm.weight,
            &self.weights.vision_post_norm.bias,
            self.config.layer_norm_eps,
            self.scales.vision_post_norm,
        )?;
        let projected = gemm::fp8(
            self.ctx(),
            &hidden,
            self.scales.vision_post_norm,
            self.weights.multimodal_projector.as_kernel_view(),
        )?;
        elementwise::bias_f16(
            self.ctx(),
            &projected,
            self.weights.multimodal_projector.bias.as_ref(),
        )
    }

    /// Input patches are already normalized and flattened as
    /// `[views*patches_per_view, 3*patch_size*patch_size]` FP16.
    pub fn encode_vision(&self, patches: &Tensor) -> Result<Tensor> {
        let patches =
            quantization::quantize_f16_e4m3(self.ctx(), patches, self.scales.vision_patch_input)?;
        self.encode_vision_fp8_patches(&patches)
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
        let language = embedding::lookup_f16(
            self.ctx(),
            &self.weights.token_embedding,
            token_ids,
            token_count,
        )?;
        elementwise::concat_rows_f16(self.ctx(), vision_tokens, &language)
    }

    pub fn prefix_forward(&self, prefix: &Tensor) -> Result<PrefixKvCache> {
        let mut hidden = prefix.clone();
        let mut keys = Vec::with_capacity(self.config.language.depth);
        let mut values = Vec::with_capacity(self.config.language.depth);
        for (index, (layer, scale)) in self
            .weights
            .language_layers
            .iter()
            .zip(&self.scales.language_layers)
            .enumerate()
        {
            let output = language_layer(
                self.ctx(),
                self.config.language,
                layer,
                *scale,
                &hidden,
                index + 1 < self.config.language.depth,
                0,
                self.config.rms_norm_eps,
                self.config.rope_theta,
            )?;
            hidden = output.hidden;
            let cache_rows = prefix.shape().dims()[0] + self.config.action_horizon;
            keys.push(cache::reserve_prefix_f16(
                self.ctx(),
                &output.key,
                cache_rows,
            )?);
            values.push(cache::reserve_prefix_f16(
                self.ctx(),
                &output.value,
                cache_rows,
            )?);
        }
        Ok(PrefixKvCache {
            keys,
            values,
            tokens: prefix.shape().dims()[0],
        })
    }

    fn conditioning(&self, time_embedding: &Tensor) -> Result<Tensor> {
        let input =
            quantization::quantize_f16_e4m3(self.ctx(), time_embedding, self.scales.time_input)?;
        let hidden = gemm::fp8(
            self.ctx(),
            &input,
            self.scales.time_input,
            self.weights.time_mlp_in.as_kernel_view(),
        )?;
        let hidden = activation::bias_silu_quant_f16_e4m3(
            self.ctx(),
            &hidden,
            self.weights.time_mlp_in.bias.as_ref(),
            self.scales.time_hidden,
        )?;
        let output = gemm::fp8(
            self.ctx(),
            &hidden,
            self.scales.time_hidden,
            self.weights.time_mlp_out.as_kernel_view(),
        )?;
        activation::bias_silu_f16(self.ctx(), &output, self.weights.time_mlp_out.bias.as_ref())
    }

    fn style(&self, conditioning: &Tensor, weights: &super::Fp8LinearWeights) -> Result<Tensor> {
        let projected = gemm::fp8(
            self.ctx(),
            conditioning,
            self.scales.conditioning,
            weights.as_kernel_view(),
        )?;
        let style = elementwise::bias_f16(self.ctx(), &projected, weights.bias.as_ref())?;
        style.reshape(vec![style.numel()])
    }

    fn prepare_step_styles(&self, time_embedding: &Tensor) -> Result<Pi05StepStyles> {
        let conditioning = self.conditioning(time_embedding)?;
        let conditioning =
            quantization::quantize_f16_e4m3(self.ctx(), &conditioning, self.scales.conditioning)?;
        let mut attention = Vec::with_capacity(self.config.action_expert.depth);
        let mut mlp = Vec::with_capacity(self.config.action_expert.depth);
        for layer in &self.weights.action_layers {
            attention.push(self.style(&conditioning, &layer.input_style)?);
            mlp.push(self.style(&conditioning, &layer.post_attention_style)?);
        }
        let final_norm = self.style(&conditioning, &self.weights.action_final_style)?;
        Ok(Pi05StepStyles {
            attention,
            mlp,
            final_norm,
        })
    }

    fn prepare_all_styles(&self, time_embeddings: &[Tensor]) -> Result<Vec<Pi05StepStyles>> {
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

    pub fn denoise_step(
        &self,
        state: &Tensor,
        time_embedding: &Tensor,
        prefix: &PrefixKvCache,
        dt: f32,
    ) -> Result<Tensor> {
        let styles = self.prepare_step_styles(time_embedding)?;
        self.denoise_step_with_styles(state, &styles, prefix, dt)
    }

    fn denoise_step_with_styles(
        &self,
        state: &Tensor,
        styles: &Pi05StepStyles,
        prefix: &PrefixKvCache,
        dt: f32,
    ) -> Result<Tensor> {
        if prefix.keys.len() != self.config.action_expert.depth
            || prefix.values.len() != self.config.action_expert.depth
            || styles.attention.len() != self.config.action_expert.depth
            || styles.mlp.len() != self.config.action_expert.depth
        {
            return Err(Error::Other("π0.5 prefix KV/style depth mismatch".into()));
        }
        let state_fp8 =
            quantization::quantize_f16_e4m3(self.ctx(), state, self.scales.action_input)?;
        let hidden = gemm::fp8(
            self.ctx(),
            &state_fp8,
            self.scales.action_input,
            self.weights.action_in.as_kernel_view(),
        )?;
        let mut hidden =
            elementwise::bias_f16(self.ctx(), &hidden, self.weights.action_in.bias.as_ref())?;

        let mut attention_normalized = None;
        for index in 0..self.config.action_expert.depth {
            let layer = &self.weights.action_layers[index];
            let (next_norm_style, next_norm_scale) = if index + 1 < self.config.action_expert.depth
            {
                (
                    &styles.attention[index + 1],
                    self.scales.action_layers[index + 1].attention_norm,
                )
            } else {
                (&styles.final_norm, self.scales.action_final_norm)
            };
            let output = action_layer(
                self.ctx(),
                self.config.action_expert,
                layer,
                self.scales.action_layers[index],
                &hidden,
                attention_normalized.as_ref(),
                &styles.attention[index],
                &styles.mlp[index],
                next_norm_style,
                next_norm_scale,
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
        let velocity = gemm::fp8(
            self.ctx(),
            &hidden,
            self.scales.action_final_norm,
            self.weights.action_out.as_kernel_view(),
        )?;
        let velocity =
            elementwise::bias_f16(self.ctx(), &velocity, self.weights.action_out.bias.as_ref())?;
        elementwise::euler_update_f16(self.ctx(), state, &velocity, dt)
    }

    pub fn denoise_all_steps(
        &self,
        noise: &Tensor,
        time_embeddings: &[Tensor],
        prefix: &PrefixKvCache,
    ) -> Result<Tensor> {
        if time_embeddings.len() != self.config.num_flow_steps {
            return Err(Error::Other(format!(
                "π0.5 expected {} timestep embeddings, got {}",
                self.config.num_flow_steps,
                time_embeddings.len()
            )));
        }
        let mut state = noise.clone();
        let dt = -1.0 / self.config.num_flow_steps as f32;
        for embedding in time_embeddings {
            state = self.denoise_step(&state, embedding, prefix, dt)?;
        }
        Ok(state)
    }

    fn denoise_all_steps_with_styles(
        &self,
        noise: &Tensor,
        styles: &[Pi05StepStyles],
        prefix: &PrefixKvCache,
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

    pub fn infer(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<Tensor> {
        let vision = self.encode_vision(patches)?;
        let prefix = self.embed_prefix(&vision, token_ids, token_count)?;
        let prefix = self.prefix_forward(&prefix)?;
        self.denoise_all_steps(noise, time_embeddings, &prefix)
    }

    fn infer_with_styles(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        styles: &[Pi05StepStyles],
    ) -> Result<Tensor> {
        let vision = self.encode_vision(patches)?;
        let prefix = self.embed_prefix(&vision, token_ids, token_count)?;
        let prefix = self.prefix_forward(&prefix)?;
        self.denoise_all_steps_with_styles(noise, styles, &prefix)
    }

    fn infer_with_styles_fp8_patches(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        styles: &[Pi05StepStyles],
    ) -> Result<Tensor> {
        let vision = self.encode_vision_fp8_patches(patches)?;
        let prefix = self.embed_prefix(&vision, token_ids, token_count)?;
        let prefix = self.prefix_forward(&prefix)?;
        self.denoise_all_steps_with_styles(noise, styles, &prefix)
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
        styles: &[Pi05StepStyles],
    ) -> Result<Tensor> {
        match (raw_images, raw_image_layout) {
            (None, None) => self.infer_with_styles(patches, token_ids, token_count, noise, styles),
            (Some(images), Some(layout)) => {
                preprocess::rgb_u8_to_patches_e4m3(
                    self.ctx(),
                    images,
                    patches,
                    self.config.num_views,
                    self.config.image_size,
                    self.config.patch_size,
                    layout,
                    self.scales.vision_patch_input,
                )?;
                self.infer_with_styles_fp8_patches(patches, token_ids, token_count, noise, styles)
            }
            _ => Err(Error::Other(
                "π0.5 raw image buffer/layout capture state is inconsistent".into(),
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
    ) -> Result<Pi05CapturedGraph> {
        let backend = &self.backend;
        if raw_images.is_some() != raw_image_layout.is_some() {
            return Err(Error::Other(
                "π0.5 raw image buffer/layout capture state is inconsistent".into(),
            ));
        }
        // Timestep embeddings are constant for the reverse-flow schedule, so
        // all AdaRMS projections can be excluded from steady-state replay.
        let styles = self.prepare_all_styles(time_embeddings)?;
        backend.synchronize()?;
        let (max_activation_elements, max_weight_elements) =
            self.config.fp8_emulation_scratch_elements(token_count)?;
        let workspace = kernels::GraphWorkspace::new_fp8(
            self.config.cuda_graph_workspace_bytes(token_count)?,
            max_activation_elements,
            max_weight_elements,
            self.ctx().device_id(),
        )?;

        // Fail shape, calibration, and workspace checks before beginning a
        // stream capture, where recovery from a rejected launch is harder.
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
                // End (or invalidate) the active capture before returning so
                // the backend stream remains usable by the caller.
                let _ = backend.end_capture();
                return Err(error);
            }
        };
        let graph = backend.end_capture()?;
        Ok(Pi05CapturedGraph {
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
            _scales: Arc::clone(&self.scales),
            workspace,
        })
    }

    /// Validate once eagerly, then capture the complete fixed-shape inference
    /// schedule into a CUDA graph backed by a persistent device arena.
    pub fn capture_infer(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<Pi05CapturedGraph> {
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

    /// Capture full inference beginning with already resized RGB `uint8`
    /// images. The graph owns a stable raw-image device buffer, and its first
    /// node performs fused normalization, patchification, and E4M3
    /// quantization. Call `update_raw_image_inputs` before each replay.
    pub fn capture_infer_rgb_u8(
        &self,
        layout: Pi05ImageLayout,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<Pi05CapturedGraph> {
        let backend = &self.backend;
        let raw_image_bytes = self
            .config
            .num_views
            .checked_mul(3)
            .and_then(|value| value.checked_mul(self.config.image_size))
            .and_then(|value| value.checked_mul(self.config.image_size))
            .ok_or_else(|| Error::Other("π0.5 raw image size overflow".into()))?;
        let raw_images = CudaBuffer::alloc_zeros(raw_image_bytes, self.ctx().device_id())
            .map_err(Error::Cuda)?;
        let patch_rows = self.config.num_views * self.config.patches_per_view();
        let patch_width = 3 * self.config.patch_size * self.config.patch_size;
        let patches = backend.to_device(&Tensor::zeros(
            vec![patch_rows, patch_width],
            apxinf_core::DType::F8E4M3,
        ))?;
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

/// Precompute the fixed reverse-flow timesteps before CUDA graph capture.
pub fn upload_time_embeddings(config: &Pi05Config, backend: &dyn Backend) -> Result<Vec<Tensor>> {
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
            .map(f16::from_f32)
            .collect::<Vec<_>>();
            let tensor = Tensor::from_f16(vec![1, config.action_expert.width], &values)?;
            backend.to_device(&tensor)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_scales_match_model_depths() {
        let config = Pi05Config::thor_two_view();
        let scales = Pi05ActivationScales::uniform(&config, 0.01).unwrap();
        assert_eq!(scales.vision_layers.len(), 27);
        assert_eq!(scales.language_layers.len(), 18);
        assert_eq!(scales.action_layers.len(), 18);
    }

    #[test]
    fn named_calibration_reports_first_missing_scale() {
        let calibration = StaticFp8Calibration::from_json_str("{}").unwrap();
        let error =
            Pi05ActivationScales::from_calibration(&Pi05Config::thor_two_view(), &calibration)
                .unwrap_err();
        assert!(error.to_string().contains("vision.layers.0.attention_norm"));
    }
}
