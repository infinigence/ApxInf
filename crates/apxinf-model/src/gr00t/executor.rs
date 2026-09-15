//! Precision-parameterized CUDA execution engine for NVIDIA GR00T N1.7.
//!
//! The public model-neutral [`crate::VlaRuntime`] adapter lives in
//! `vla_runtime`; this module owns GR00T's device state and execution plan.

use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use apxinf_core::{Backend, DType, Device, Error, Graph, Result, Tensor};
use half::bf16;

use super::action_weights::{
    select_category_linear, select_category_mlp, Gr00tActionEncoderWeights, Gr00tActionHeadWeights,
    Gr00tAttentionWeights, Gr00tCategoryLinearWeights, Gr00tCategoryMlpWeights,
    Gr00tDitBlockWeights, Gr00tEmbodimentWeights, Gr00tFeedForwardWeights, Gr00tLayerNormWeights,
    Gr00tLinearWeights, Gr00tMlpWeights, Gr00tVlSelfAttentionBlockWeights,
};
use super::backbone::vision;
use super::backbone::vision_weights::transfer_vision_weights;
use super::backbone::{
    transfer_text_weights, Qwen3VLConfig, Qwen3VLTextWeights, Qwen3VLVisionWeights,
};
use super::backend::{kernels, transfers, DeviceBuffer, RuntimeBackend, TuningMode};
use super::device_weights::DeviceLinearWeights;
use super::geometry::{
    build_backbone_token_groups, dit_attention_source, Gr00tBackboneTokenGroups,
    Gr00tDitAttentionSource,
};
use super::math::{
    action_timestep_embedding, dit_timestep_projection, flow_schedule, Gr00tFlowStep,
};
use super::weights::Gr00tWeights;
use super::Gr00tConfig;

// The current stable-address bump arena needs cumulative allocation volume,
// not peak live bytes. Nsight measured 2.521 GiB for the representative
// three-view input; 4 GiB leaves alignment and small-shape variance headroom.
// This is intentionally a first graph-correctness bound, not the final memory
// plan. The maintained rationale lives in doc/gr00t-n1.7.md.
const INITIAL_GRAPH_WORKSPACE_BYTES: usize = 4usize << 30;
const USE_FUSED_FP8_RMS_NORM: bool = true;
const USE_FUSED_FP8_VISION_LAYER_NORM: bool = true;
const USE_FUSED_SELF_QKV: bool = true;
const USE_FUSED_ADAPTIVE_LAYER_NORM: bool = true;
const USE_STRIDED_FUSED_QKV_ATTENTION: bool = true;
const USE_FUSED_BF16_SILU_MUL: bool = true;
const USE_FUSED_W8A8_SILU_MUL_QUANT: bool = true;
const USE_FUSED_FP8_BIAS_GELU_QUANT: bool = true;
const USE_FUSED_FP8_SELF_QKV_BIAS: bool = true;
const USE_FUSED_ADAPTIVE_LAYER_NORM_QUANT: bool = true;
const USE_FUSED_FFN_BIAS_RESIDUAL: bool = true;

/// Precision-specific weight materialization and layer-composition policy.
///
/// The public loader selects one concrete implementation.  `Gr00tExecutor`
/// is then monomorphized for that precision, so model execution contains no
/// per-linear BF16/FP8/INT8 enum dispatch.
pub(super) trait Gr00tPrecisionExecution: Sized + 'static {
    type Dense: DeviceLinearWeights;
    type FeedForward: DeviceLinearWeights;
    type FusedQkv: DeviceLinearWeights;
    type Backbone: DeviceLinearWeights;

    const NAME: &'static str;
    const SUPPORTS_CALIBRATION: bool;

    fn transfer_dense(
        &self,
        weights: Gr00tLinearWeights,
        name: &str,
        backend: &RuntimeBackend,
    ) -> Result<Self::Dense>;

    fn transfer_feed_forward(
        &self,
        weights: Gr00tLinearWeights,
        name: &str,
        backend: &RuntimeBackend,
    ) -> Result<Self::FeedForward>;

    fn transfer_fused_qkv(
        &self,
        weights: Gr00tLinearWeights,
        name: &str,
        backend: &RuntimeBackend,
    ) -> Result<Option<Self::FusedQkv>>;

    fn transfer_backbone_linears(
        &self,
        text: &Qwen3VLTextWeights,
        vision: &Qwen3VLVisionWeights,
        backend: &RuntimeBackend,
    ) -> Result<BTreeMap<String, Self::Backbone>>;
}

/// Processor-owned, fully normalized input used by the device executor.
///
/// This remains private to the GR00T implementation. Public callers use the
/// model-neutral [`crate::VlaRequest`] contract through `vla_runtime`.
#[derive(Clone, Debug)]
pub(super) struct Gr00tObservation {
    pub pixel_values: Tensor,
    pub image_grid_thw: Vec<[u32; 3]>,
    pub token_ids: Vec<u32>,
    pub attention_mask: Vec<u8>,
    pub state: Tensor,
    pub embodiment_id: usize,
    pub noise: Tensor,
}

impl Gr00tObservation {
    fn inference_spec(&self, config: &Gr00tConfig) -> Result<Gr00tInferenceSpec> {
        self.validate(config)?;
        Ok(Gr00tInferenceSpec {
            batch_size: 1,
            token_count: self.token_ids.len(),
            image_grid_thw: self.image_grid_thw.clone(),
            state_history_length: config.state_history_length,
            state_dim: config.max_state_dim,
            action_horizon: config.action_horizon,
            action_dim: config.max_action_dim,
        })
    }

    fn validate(&self, config: &Gr00tConfig) -> Result<()> {
        config.validate()?;
        if self.token_ids.is_empty() {
            return Err(Error::Other(
                "GR00T observation requires at least one token".into(),
            ));
        }
        if self.token_ids.len() != self.attention_mask.len() {
            return Err(Error::Other(format!(
                "GR00T token/mask length mismatch: {} tokens, {} mask entries",
                self.token_ids.len(),
                self.attention_mask.len()
            )));
        }
        if let Some((index, value)) = self
            .attention_mask
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| *value > 1)
        {
            return Err(Error::Other(format!(
                "GR00T attention_mask[{index}] must be 0 or 1, got {value}"
            )));
        }
        if self.image_grid_thw.is_empty() {
            return Err(Error::Other(
                "GR00T observation requires at least one image grid".into(),
            ));
        }
        for (index, grid) in self.image_grid_thw.iter().enumerate() {
            if grid.contains(&0) {
                return Err(Error::Other(format!(
                    "GR00T image_grid_thw[{index}] must contain non-zero dimensions, got {grid:?}"
                )));
            }
        }
        if self.embodiment_id >= config.max_num_embodiments {
            return Err(Error::Other(format!(
                "GR00T embodiment_id {} is outside 0..{}",
                self.embodiment_id, config.max_num_embodiments
            )));
        }
        expect_bf16_shape(
            "state",
            &self.state,
            &[1, config.state_history_length, config.max_state_dim],
        )?;
        expect_bf16_shape(
            "noise",
            &self.noise,
            &[1, config.action_horizon, config.max_action_dim],
        )?;
        if self.pixel_values.dtype() != DType::BF16 {
            return Err(Error::Other(format!(
                "GR00T pixel_values must be BF16 after preprocessing, got {}",
                self.pixel_values.dtype()
            )));
        }
        if self.pixel_values.numel() == 0 {
            return Err(Error::Other("GR00T pixel_values must not be empty".into()));
        }
        Ok(())
    }
}

fn expect_bf16_shape(name: &str, tensor: &Tensor, expected: &[usize]) -> Result<()> {
    if tensor.dtype() != DType::BF16 || tensor.shape().dims() != expected {
        return Err(Error::Other(format!(
            "GR00T {name} must be BF16 {expected:?}, got {} {:?}",
            tensor.dtype(),
            tensor.shape().dims()
        )));
    }
    Ok(())
}

/// Shape-and-topology key used to decide whether a captured graph is reusable.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Gr00tInferenceSpec {
    batch_size: usize,
    token_count: usize,
    image_grid_thw: Vec<[u32; 3]>,
    state_history_length: usize,
    state_dim: usize,
    action_horizon: usize,
    action_dim: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Gr00tGraphKey {
    spec: Gr00tInferenceSpec,
    pixel_shape: Vec<usize>,
    image_positions: Vec<usize>,
    embodiment_id: usize,
}

impl Gr00tGraphKey {
    fn new(
        observation: &Gr00tObservation,
        config: &Gr00tConfig,
        backbone: &Qwen3VLConfig,
    ) -> Result<Self> {
        let image_positions = observation
            .token_ids
            .iter()
            .enumerate()
            .filter_map(|(index, token)| (*token == backbone.image_token_id).then_some(index))
            .collect();
        Ok(Self {
            spec: observation.inference_spec(config)?,
            pixel_shape: observation.pixel_values.shape().dims().to_vec(),
            image_positions,
            embodiment_id: observation.embodiment_id,
        })
    }
}

struct Gr00tGraphInputs {
    pixel_values: Tensor,
    state: Tensor,
    noise: Tensor,
    token_ids: DeviceBuffer,
}

struct Gr00tRowIndexCache {
    token_count: usize,
    image_rows: Vec<usize>,
    non_image_rows: Vec<usize>,
    image_indices: kernels::elementwise::PreparedRowIndices,
    non_image_indices: kernels::elementwise::PreparedRowIndices,
}

struct Gr00tCapturedGraph {
    graph: Box<dyn Graph>,
    _workspace: kernels::GraphWorkspace,
    output: Tensor,
    inputs: Gr00tGraphInputs,
}

struct Gr00tBackboneOutput {
    non_image: Tensor,
    image: Tensor,
}

impl Gr00tGraphInputs {
    fn new(observation: &Gr00tObservation, backend: &RuntimeBackend) -> Result<Self> {
        let pixel_values = backend.to_device(&observation.pixel_values)?;
        let state = backend.to_device(&observation.state)?;
        let noise = backend.to_device(&observation.noise)?;
        let token_bytes = observation
            .token_ids
            .iter()
            .flat_map(|token| token.to_ne_bytes())
            .collect::<Vec<_>>();
        let token_ids =
            DeviceBuffer::alloc(token_bytes.len(), backend.device_id()).map_err(Error::Cuda)?;
        token_ids
            .copy_from_host(&token_bytes)
            .map_err(Error::Cuda)?;
        Ok(Self {
            pixel_values,
            state,
            noise,
            token_ids,
        })
    }

    fn observation(&self, source: &Gr00tObservation) -> Gr00tObservation {
        Gr00tObservation {
            pixel_values: self.pixel_values.clone(),
            image_grid_thw: source.image_grid_thw.clone(),
            token_ids: source.token_ids.clone(),
            attention_mask: source.attention_mask.clone(),
            state: self.state.clone(),
            embodiment_id: source.embodiment_id,
            noise: self.noise.clone(),
        }
    }

    fn update(&self, observation: &Gr00tObservation, backend: &RuntimeBackend) -> Result<()> {
        backend.synchronize()?;
        transfers::copy_cpu_to_cuda(&observation.pixel_values, &self.pixel_values)?;
        transfers::copy_cpu_to_cuda(&observation.state, &self.state)?;
        transfers::copy_cpu_to_cuda(&observation.noise, &self.noise)?;
        let token_bytes = observation
            .token_ids
            .iter()
            .flat_map(|token| token.to_ne_bytes())
            .collect::<Vec<_>>();
        if token_bytes.len() != self.token_ids.len() {
            return Err(Error::Other(format!(
                "GR00T graph expects {} token bytes, got {}",
                self.token_ids.len(),
                token_bytes.len()
            )));
        }
        self.token_ids
            .copy_from_host(&token_bytes)
            .map_err(Error::Cuda)
    }
}

impl Gr00tCapturedGraph {
    fn run(
        &self,
        observation: &Gr00tObservation,
        backend: &RuntimeBackend,
        config: &Gr00tConfig,
    ) -> Result<Tensor> {
        self.inputs.update(observation, backend)?;
        self.graph.replay()?;
        backend.synchronize()?;
        let output = self
            .output
            .reshape(vec![1, config.action_horizon, config.max_action_dim])?;
        backend.to_cpu(&output)
    }
}

fn graph_input_supported(observation: &Gr00tObservation) -> bool {
    observation.pixel_values.device() == Device::Cpu
        && observation.state.device() == Device::Cpu
        && observation.noise.device() == Device::Cpu
}

pub(crate) struct Gr00tExecutor<E: Gr00tPrecisionExecution> {
    // Drop the captured graph before the weight/cache fields whose addresses
    // its nodes reference.
    captured_graph: Option<(Gr00tGraphKey, Gr00tCapturedGraph)>,
    config: Gr00tConfig,
    backbone_config: Qwen3VLConfig,
    backend: Arc<RuntimeBackend>,
    backbone_text: Qwen3VLTextWeights,
    backbone_linears: BTreeMap<String, E::Backbone>,
    backbone_vision: Qwen3VLVisionWeights,
    vision_position_cache: Option<(Vec<[u32; 3]>, vision::PreparedVisionPositions)>,
    backbone_token_cache: Option<(Vec<u32>, DeviceBuffer)>,
    backbone_position_cache: Option<(Vec<u32>, DeviceBuffer)>,
    backbone_row_index_cache: Option<Gr00tRowIndexCache>,
    category_weights: Gr00tCategoryWeights,
    selected_embodiment: Option<(usize, Gr00tDeviceEmbodimentWeights<E>)>,
    action: Gr00tDeviceActionWeights<E>,
    execution: E,
}

impl<E: Gr00tPrecisionExecution> Gr00tExecutor<E> {
    pub(crate) fn from_backend(
        config: Gr00tConfig,
        backbone_config: Qwen3VLConfig,
        weights: Gr00tWeights,
        execution: E,
        backend: Arc<RuntimeBackend>,
    ) -> Result<Self> {
        validate_runtime_support(&config)?;
        let Gr00tWeights {
            backbone_text,
            backbone_vision,
            action_head,
        } = weights;
        let backbone_linears =
            execution.transfer_backbone_linears(&backbone_text, &backbone_vision, &*backend)?;
        let backbone_text = transfer_text_weights(&backbone_text, &*backend)?;
        let backbone_vision = transfer_vision_weights(&backbone_vision, &*backend)?;
        let (category_weights, action) =
            Gr00tDeviceActionWeights::from_host(&config, action_head, &execution, &*backend)?;
        Ok(Self {
            config,
            captured_graph: None,
            backbone_config,
            backend,
            backbone_text,
            backbone_linears,
            backbone_vision,
            vision_position_cache: None,
            backbone_token_cache: None,
            backbone_position_cache: None,
            backbone_row_index_cache: None,
            category_weights,
            selected_embodiment: None,
            action,
            execution,
        })
    }

    pub fn config(&self) -> &Gr00tConfig {
        &self.config
    }

    pub(crate) fn max_token_len(&self) -> usize {
        self.backbone_config.text.max_position_embeddings.min(4096)
    }

    pub(crate) fn pixel_width(&self) -> usize {
        self.backbone_config.vision.in_channels
            * self.backbone_config.vision.temporal_patch_size
            * self.backbone_config.vision.patch_size
            * self.backbone_config.vision.patch_size
    }

    pub(crate) fn patch_size(&self) -> usize {
        self.backbone_config.vision.patch_size
    }

    /// Return whether the runtime currently owns a successfully captured graph.
    ///
    /// Production inference deliberately falls back to eager execution when
    /// capture is unavailable. Benchmarks use this signal to fail closed
    /// instead of accidentally reporting an eager run as CUDA Graph latency.
    pub fn has_captured_graph(&self) -> bool {
        self.captured_graph.is_some()
    }

    pub(crate) fn calibration_plan(&self) -> Vec<String> {
        super::calibration::fp8_consumers(&self.config, &self.backbone_config)
            .into_iter()
            .map(|consumer| format!("{consumer}.input"))
            .collect()
    }

    /// Collect the BF16 activation maxima consumed by the static-FP8 plan.
    /// Calibration is deliberately eager and may copy one tensor per site to
    /// the host; ordinary inference never installs the collector.
    pub(crate) fn calibration_amax(
        &mut self,
        observation: &Gr00tObservation,
    ) -> Result<BTreeMap<String, f32>> {
        if !E::SUPPORTS_CALIBRATION {
            return Err(Error::Other(format!(
                "GR00T activation calibration requires a BF16 runtime, got {}",
                E::NAME
            )));
        }
        self.validate_observation(observation)?;
        drop(self.captured_graph.take());
        self.prepare_fixed_inputs(observation, None)?;

        let expected = self.calibration_plan();
        let collector = Rc::new(super::calibration::Gr00tCalibrationCollector::new(
            Arc::clone(&self.backend),
        ));
        let guard = super::calibration::install(Rc::clone(&collector))?;
        self.replay_static_conditioning_for_calibration()?;
        let output = self.infer_device(observation)?;
        self.backend.synchronize()?;
        drop(output);
        drop(guard);
        collector.records(&expected)
    }

    fn replay_static_conditioning_for_calibration(&self) -> Result<()> {
        for projection in &self.action.dit_timestep_projections {
            let timestep = linear_silu(&*self.backend, projection, &self.action.timestep_input)?;
            let timestep = linear(&*self.backend, &timestep, &self.action.timestep_output)?;
            let activated = self.backend.silu(&timestep)?;
            for block in &self.action.dit_blocks {
                drop(linear(&*self.backend, &activated, &block.adaptive_norm)?);
            }
            drop(linear(
                &*self.backend,
                &activated,
                &self.action.output_modulation,
            )?);
        }
        Ok(())
    }

    /// Run one batch-one GR00T inference and return CPU BF16 actions shaped
    /// `[1, action_horizon, max_action_dim]`.
    pub fn infer(&mut self, observation: &Gr00tObservation) -> Result<Tensor> {
        let _inference_range = crate::profiling::trace::range("gr00t_infer");
        self.validate_observation(observation)?;

        if graph_input_supported(observation) {
            let key = Gr00tGraphKey::new(observation, &self.config, &self.backbone_config)?;
            if let Some((cached, graph)) = &self.captured_graph {
                if cached == &key {
                    if let Some((tokens, _)) = &mut self.backbone_token_cache {
                        *tokens = observation.token_ids.clone();
                    }
                    return graph.run(observation, &*self.backend, &self.config);
                }
            }

            // Captured nodes retain raw addresses into runtime-owned weights
            // and selected embodiment tensors. Destroy an obsolete graph
            // before replacing any of those resources.
            drop(self.captured_graph.take());
            if self.backend.context().tuning().mode() == TuningMode::AutoTune {
                // Online tuning requires the real request operands and is
                // intentionally disabled while sizing or capturing a graph.
                // Traverse once eagerly, just as the shared PI0.5 runtime
                // does, so every exact GEMM shape is persisted before graph
                // preparation freezes the selected plans.
                self.prepare_fixed_inputs(observation, None)?;
                let tuned_output = self.infer_device(observation)?;
                self.backend.synchronize()?;
                drop(tuned_output);
            }
            match self.capture_graph(observation) {
                Ok(graph) => {
                    let result = graph.run(observation, &*self.backend, &self.config);
                    self.captured_graph = Some((key, graph));
                    return result;
                }
                Err(error) => {
                    eprintln!(
                        "[apxinf] GR00T CUDA Graph capture unavailable, using eager: {error}"
                    );
                }
            }
        }

        // The eager path may replace device buffers retained by an older
        // graph (for example when a Rust caller switches to device-resident
        // inputs). Destroy that graph before mutating any runtime cache.
        drop(self.captured_graph.take());
        self.prepare_fixed_inputs(observation, None)?;
        let actions = self.infer_device(observation)?;
        self.finish_actions(actions)
    }

    fn validate_observation(&self, observation: &Gr00tObservation) -> Result<()> {
        observation.validate(&self.config)?;
        if observation.token_ids.len() > self.backbone_config.text.max_position_embeddings.min(4096)
        {
            return Err(Error::Other(format!(
                "GR00T token count {} exceeds prepared Qwen cache capacity {}",
                observation.token_ids.len(),
                self.backbone_config.text.max_position_embeddings.min(4096)
            )));
        }
        if let Some((index, token)) = observation
            .token_ids
            .iter()
            .copied()
            .enumerate()
            .find(|(_, token)| *token as usize >= self.backbone_config.text.vocab_size)
        {
            return Err(Error::Other(format!(
                "GR00T token_ids[{index}]={token} exceeds Qwen vocabulary size {}",
                self.backbone_config.text.vocab_size
            )));
        }
        if observation.attention_mask.iter().any(|value| *value != 1) {
            return Err(Error::Other(
                "GR00T batch-one CUDA backbone currently requires an unpadded attention mask"
                    .into(),
            ));
        }
        Ok(())
    }

    fn infer_device(&mut self, observation: &Gr00tObservation) -> Result<Tensor> {
        let backbone = self
            .infer_backbone_device(observation)
            .map_err(|error| Error::Other(format!("GR00T backbone execution failed: {error}")))?;
        self.infer_action_device(observation, &backbone)
            .map_err(|error| Error::Other(format!("GR00T action execution failed: {error}")))
    }

    fn infer_backbone_device(
        &mut self,
        observation: &Gr00tObservation,
    ) -> Result<Gr00tBackboneOutput> {
        let token_groups = build_backbone_token_groups(
            &observation.token_ids,
            &observation.attention_mask,
            self.backbone_config.image_token_id,
        )?;
        let backbone = self
            .forward_backbone(observation, &token_groups)
            .map_err(|error| Error::Other(format!("GR00T backbone model failed: {error}")))?;
        let backbone = self
            .forward_backbone_adapter(&backbone)
            .map_err(|error| Error::Other(format!("GR00T backbone adapter failed: {error}")))?;
        let row_indices = self
            .backbone_row_index_cache
            .as_ref()
            .ok_or_else(|| Error::Other("GR00T row indices were not prepared".into()))?;
        let non_image_backbone = kernels::elementwise::gather_rows_bf16_prepared(
            self.backend.context(),
            &backbone,
            &row_indices.non_image_indices,
        )
        .map_err(|error| Error::Other(format!("GR00T non-image gather failed: {error}")))?;
        let image_backbone = kernels::elementwise::gather_rows_bf16_prepared(
            self.backend.context(),
            &backbone,
            &row_indices.image_indices,
        )
        .map_err(|error| Error::Other(format!("GR00T image gather failed: {error}")))?;

        Ok(Gr00tBackboneOutput {
            non_image: non_image_backbone,
            image: image_backbone,
        })
    }

    fn infer_action_device(
        &mut self,
        observation: &Gr00tObservation,
        backbone: &Gr00tBackboneOutput,
    ) -> Result<Tensor> {
        self.prepare_embodiment(observation.embodiment_id)?;
        let selected = &self
            .selected_embodiment
            .as_ref()
            .ok_or_else(|| Error::Other("GR00T embodiment weights were not prepared".into()))?
            .1;
        let state = observation
            .state
            .reshape(vec![1, self.config.state_input_dim()?])?;
        let state = if state.device() == self.backend.device() {
            state
        } else {
            self.backend.to_device(&state)?
        };
        let state = linear_relu(&*self.backend, &state, &selected.state_encoder.input)
            .map_err(|error| Error::Other(format!("GR00T state encoder input failed: {error}")))?;
        let state = linear(&*self.backend, &state, &selected.state_encoder.output)
            .map_err(|error| Error::Other(format!("GR00T state encoder output failed: {error}")))?;

        let noise = observation
            .noise
            .reshape(vec![self.config.action_horizon, self.config.max_action_dim])?;
        let mut actions = if noise.device() == self.backend.device() {
            noise
        } else {
            self.backend.to_device(&noise)?
        };
        let schedule = flow_schedule(
            self.config.num_inference_timesteps,
            self.config.num_timestep_buckets,
        )?;
        let use_cross_attention_cache = true;
        let mut cross_attention = Vec::with_capacity(self.action.dit_blocks.len());
        for (layer_index, block) in self.action.dit_blocks.iter().enumerate() {
            let source = match dit_attention_source(&self.config, layer_index)? {
                Gr00tDitAttentionSource::FullBackbone => {
                    return Err(Error::Other(
                        "GR00T phase-one runtime requires AlternateVLDiT".into(),
                    ));
                }
                Gr00tDitAttentionSource::NonImageBackbone => Some(&backbone.non_image),
                Gr00tDitAttentionSource::ImageBackbone => Some(&backbone.image),
                Gr00tDitAttentionSource::StateActionSelf => None,
            };
            let prepared = if use_cross_attention_cache {
                source
                    .map(|source| {
                        prepare_attention_key_value(
                            &*self.backend,
                            source,
                            &block.attention,
                            self.config.diffusion.num_attention_heads,
                            self.config.diffusion.attention_head_dim,
                        )
                    })
                    .transpose()
                    .map_err(|error| {
                        Error::Other(format!(
                            "GR00T cross-attention layer {layer_index} preparation failed: {error}"
                        ))
                    })?
            } else {
                None
            };
            cross_attention.push(prepared);
        }
        for (step_index, step) in schedule.iter().enumerate() {
            actions = self
                .forward_flow_step(
                    &actions,
                    &state,
                    &backbone.non_image,
                    &backbone.image,
                    &cross_attention,
                    selected,
                    step_index,
                    *step,
                )
                .map_err(|error| {
                    Error::Other(format!("GR00T flow step {step_index} failed: {error}"))
                })?;
        }
        actions.reshape(vec![
            1,
            self.config.action_horizon,
            self.config.max_action_dim,
        ])
    }

    fn finish_actions(&self, actions: Tensor) -> Result<Tensor> {
        self.backend.synchronize()?;
        self.backend.to_cpu(&actions)
    }

    fn prepare_fixed_inputs(
        &mut self,
        observation: &Gr00tObservation,
        token_buffer: Option<DeviceBuffer>,
    ) -> Result<()> {
        self.prepare_embodiment(observation.embodiment_id)?;
        if let Some(buffer) = token_buffer {
            self.backbone_token_cache = Some((observation.token_ids.clone(), buffer));
        } else if !matches!(
            &self.backbone_token_cache,
            Some((tokens, _)) if tokens == &observation.token_ids
        ) {
            let bytes = observation
                .token_ids
                .iter()
                .flat_map(|token| token.to_ne_bytes())
                .collect::<Vec<_>>();
            let buffer = match &self.backbone_token_cache {
                Some((_, buffer)) if buffer.len() == bytes.len() => buffer.clone(),
                _ => DeviceBuffer::alloc(bytes.len(), self.backend.device_id())
                    .map_err(Error::Cuda)?,
            };
            buffer.copy_from_host(&bytes).map_err(Error::Cuda)?;
            self.backbone_token_cache = Some((observation.token_ids.clone(), buffer));
        }
        let token_groups = build_backbone_token_groups(
            &observation.token_ids,
            &observation.attention_mask,
            self.backbone_config.image_token_id,
        )?;
        let row_cache_matches = self.backbone_row_index_cache.as_ref().is_some_and(|cache| {
            cache.token_count == observation.token_ids.len()
                && cache.image_rows == token_groups.image
                && cache.non_image_rows == token_groups.non_image
        });
        if !row_cache_matches {
            let image_indices = kernels::elementwise::prepare_row_indices(
                self.backend.context(),
                &token_groups.image,
                observation.token_ids.len(),
            )?;
            let non_image_indices = kernels::elementwise::prepare_row_indices(
                self.backend.context(),
                &token_groups.non_image,
                observation.token_ids.len(),
            )?;
            self.backbone_row_index_cache = Some(Gr00tRowIndexCache {
                token_count: observation.token_ids.len(),
                image_rows: token_groups.image,
                non_image_rows: token_groups.non_image,
                image_indices,
                non_image_indices,
            });
        }
        Ok(())
    }

    fn capture_graph(&mut self, observation: &Gr00tObservation) -> Result<Gr00tCapturedGraph> {
        let inputs = Gr00tGraphInputs::new(observation, &*self.backend)?;
        self.prepare_fixed_inputs(observation, Some(inputs.token_ids.clone()))?;
        let device_observation = inputs.observation(observation);
        self.capture_whole_graph(inputs, device_observation)
    }

    fn capture_whole_graph(
        &mut self,
        inputs: Gr00tGraphInputs,
        device_observation: Gr00tObservation,
    ) -> Result<Gr00tCapturedGraph> {
        let workspace =
            kernels::GraphWorkspace::new(INITIAL_GRAPH_WORKSPACE_BYTES, self.backend.device_id())?;

        let preflight_start = Instant::now();
        let eager_output =
            kernels::prepare_with_workspace(&workspace, || self.infer_device(&device_observation))
                .map_err(|error| Error::Other(format!("whole-graph preflight failed: {error}")))?;
        self.backend.synchronize()?;
        let preflight_seconds = preflight_start.elapsed().as_secs_f64();
        drop(eager_output);

        let capture_start = Instant::now();
        self.backend.begin_capture()?;
        let output =
            match kernels::with_workspace(&workspace, || self.infer_device(&device_observation)) {
                Ok(output) => output,
                Err(error) => {
                    let _ = self.backend.end_capture();
                    return Err(Error::Other(format!(
                        "whole-graph capture body failed: {error}"
                    )));
                }
            };
        let graph = self
            .backend
            .end_capture()
            .map_err(|error| Error::Other(format!("whole-graph finalization failed: {error}")))?;
        let capture_seconds = capture_start.elapsed().as_secs_f64();
        eprintln!(
            "[apxinf] GR00T CUDA Graph captured with {} / {} workspace bytes; preflight {:.3}s, capture+instantiate {:.3}s",
            workspace.used(),
            workspace.capacity(),
            preflight_seconds,
            capture_seconds,
        );
        Ok(Gr00tCapturedGraph {
            graph,
            _workspace: workspace,
            output,
            inputs,
        })
    }

    fn prepare_embodiment(&mut self, embodiment_id: usize) -> Result<()> {
        if self
            .selected_embodiment
            .as_ref()
            .is_some_and(|(selected, _)| *selected == embodiment_id)
        {
            return Ok(());
        }
        let host = self.category_weights.select(embodiment_id)?;
        let device = transfer_embodiment(host, &self.execution, &*self.backend)?;
        self.selected_embodiment = Some((embodiment_id, device));
        Ok(())
    }

    fn forward_backbone(
        &mut self,
        observation: &Gr00tObservation,
        token_groups: &Gr00tBackboneTokenGroups,
    ) -> Result<Tensor> {
        let _backbone_range = crate::profiling::trace::range("gr00t_backbone");
        let pixels = if observation.pixel_values.device() == self.backend.device() {
            observation.pixel_values.clone()
        } else {
            self.backend.to_device(&observation.pixel_values)?
        };
        if !matches!(
            &self.vision_position_cache,
            Some((grid, _)) if grid == &observation.image_grid_thw
        ) {
            let positions = vision::prepare_positions(
                &self.backbone_config,
                &self.backbone_vision,
                &*self.backend,
                &observation.image_grid_thw,
            )?;
            self.vision_position_cache = Some((observation.image_grid_thw.clone(), positions));
        }
        let positions = &self
            .vision_position_cache
            .as_ref()
            .ok_or_else(|| Error::Other("GR00T vision positions were not prepared".into()))?
            .1;
        let vision_matmul = |name: &str, input: &Tensor, weight: &Tensor| {
            qwen_matmul(&*self.backend, &self.backbone_linears, name, input, weight)
        };
        let vision_norm_matmul = |name: &str,
                                  input: &Tensor,
                                  norm_weight: &Tensor,
                                  norm_bias: &Tensor,
                                  eps: f32,
                                  weight: &Tensor| {
            if USE_FUSED_FP8_VISION_LAYER_NORM {
                if let Some(linear) = self.backbone_linears.get(name) {
                    let scale = linear
                        .activation_scale()
                        .ok_or_else(|| Error::Other(format!("{name} is not an FP8 linear")))?;
                    let quantized = kernels::norm::layer_quant_bf16_e4m3(
                        self.backend.context(),
                        input,
                        norm_weight,
                        norm_bias,
                        eps,
                        scale,
                    )?;
                    return linear.forward_quantized_tensor(&quantized, &*self.backend);
                }
            }
            let normalized = layer_norm(&self.backend, input, norm_weight, norm_bias, eps)?;
            qwen_matmul(
                &*self.backend,
                &self.backbone_linears,
                name,
                &normalized,
                weight,
            )
        };
        let vision_bias_gelu = |_name: &str, input: &Tensor, bias: &Tensor| {
            kernels::activation::bias_gelu_bf16(self.backend.context(), input, Some(bias))
        };
        let vision_output = vision::forward_with_prepared_positions_and_matmul(
            &self.backbone_config,
            &self.backbone_vision,
            &*self.backend,
            &pixels,
            &observation.image_grid_thw,
            positions,
            &vision_matmul,
            Some(&vision_norm_matmul),
            Some(&vision_bias_gelu),
        )
        .map_err(|error| Error::Other(format!("GR00T vision forward failed: {error}")))?;
        let image_positions = &token_groups.image;
        if image_positions.len() != vision_output.primary.shape().dims()[0] {
            return Err(Error::Other(format!(
                "GR00T image token count {} does not match Qwen vision output rows {}",
                image_positions.len(),
                vision_output.primary.shape().dims()[0]
            )));
        }

        if !matches!(
            &self.backbone_token_cache,
            Some((tokens, _)) if tokens == &observation.token_ids
        ) {
            return Err(Error::Other(
                "GR00T token IDs must be prepared before device execution".into(),
            ));
        }
        let token_buffer = &self
            .backbone_token_cache
            .as_ref()
            .ok_or_else(|| Error::Other("GR00T token IDs were not prepared".into()))?
            .1;
        let embedded = kernels::embedding::lookup(
            self.backend.context(),
            &self.backbone_text.token_embedding,
            token_buffer,
            observation.token_ids.len(),
        )?;
        let image_indices = self
            .backbone_row_index_cache
            .as_ref()
            .ok_or_else(|| Error::Other("GR00T row indices were not prepared".into()))?
            .image_indices
            .clone();
        let mut hidden = scatter_rows(
            &*self.backend,
            &embedded,
            &vision_output.primary,
            false,
            &image_indices,
        )?;
        let position_ids = multimodal_rope_positions(
            &self.backbone_config,
            &observation.token_ids,
            &observation.image_grid_thw,
        )?;
        let position_buffer = match &self.backbone_position_cache {
            Some((cached, buffer)) if cached == &position_ids => buffer.clone(),
            _ => {
                let bytes = position_ids
                    .iter()
                    .flat_map(|position| position.to_ne_bytes())
                    .collect::<Vec<_>>();
                let buffer = DeviceBuffer::alloc(bytes.len(), self.backend.device_id())
                    .map_err(Error::Cuda)?;
                buffer.copy_from_host(&bytes).map_err(Error::Cuda)?;
                self.backbone_position_cache = Some((position_ids.clone(), buffer.clone()));
                buffer
            }
        };
        for layer in 0..self.backbone_config.text.n_layers {
            let _layer_range = crate::profiling::trace::range("gr00t_qwen_layer");
            hidden = forward_qwen_layer(
                &*self.backend,
                &self.backbone_config,
                &self.backbone_text,
                &self.backbone_linears,
                &hidden,
                layer,
                &position_buffer,
            )
            .map_err(|error| {
                Error::Other(format!("GR00T Qwen text layer {layer} failed: {error}"))
            })?;
            if let Some(deepstack) = vision_output.deepstack.get(layer) {
                hidden = scatter_rows(&*self.backend, &hidden, deepstack, true, &image_indices)?;
            }
        }
        // GR00T consumes Qwen3-VL's selected decoder-layer hidden state via
        // `outputs.hidden_states[-1]`. That value is captured before the
        // language model's final RMSNorm, even though `last_hidden_state`
        // itself is normalized. Applying the final norm here would therefore
        // change the checkpoint graph before GR00T's own VLLN adapter.
        Ok(hidden)
    }

    fn forward_backbone_adapter(&self, hidden: &Tensor) -> Result<Tensor> {
        let _adapter_range = crate::profiling::trace::range("gr00t_backbone_adapter");
        let mut hidden = match &self.action.backbone_layer_norm {
            Some(weights) => {
                layer_norm(&self.backend, hidden, &weights.weight, &weights.bias, 1e-5)?
            }
            None => hidden.clone(),
        };
        let config = self.config.vl_self_attention.as_ref();
        for block in &self.action.vl_self_attention {
            let _block_range = crate::profiling::trace::range("gr00t_vl_self_attention_block");
            let config = config.ok_or_else(|| {
                Error::Other("GR00T VL self-attention weights have no config".into())
            })?;
            hidden = forward_standard_block(
                &*self.backend,
                &hidden,
                block,
                config.num_attention_heads,
                config.attention_head_dim,
                1e-5,
            )?;
        }
        Ok(hidden)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_flow_step(
        &self,
        actions: &Tensor,
        state: &Tensor,
        non_image_backbone: &Tensor,
        image_backbone: &Tensor,
        cross_attention: &[Option<Gr00tPreparedKeyValue>],
        selected: &Gr00tDeviceEmbodimentWeights<E>,
        step_index: usize,
        step: Gr00tFlowStep,
    ) -> Result<Tensor> {
        let _flow_range = crate::profiling::trace::range("gr00t_flow_step");
        let mut action_features = linear(&*self.backend, actions, &selected.action_encoder.input)?;
        let time_features = self
            .action
            .action_time_embeddings
            .get(step_index)
            .ok_or_else(|| {
                Error::Other(format!("missing GR00T action time embedding {step_index}"))
            })?;
        action_features = kernels::elementwise::concat_columns_bf16(
            self.backend.context(),
            &[&action_features, time_features],
        )?;
        action_features = linear_silu(
            &*self.backend,
            &action_features,
            &selected.action_encoder.time,
        )?;
        action_features = linear(
            &*self.backend,
            &action_features,
            &selected.action_encoder.output,
        )?;
        if let Some(position) = &self.action.action_position_embedding {
            // NVIDIA adds positions to the 40 action tokens before prepending
            // the state token; the state token intentionally has no learned
            // positional embedding.
            action_features = self.backend.add(&action_features, position)?;
        }
        let mut hidden = kernels::elementwise::concat_rows_bf16(
            self.backend.context(),
            state,
            &action_features,
        )?;

        let use_static_conditioning = true;
        let dynamic_timestep = if use_static_conditioning {
            None
        } else {
            let timestep_projection = self
                .action
                .dit_timestep_projections
                .get(step_index)
                .ok_or_else(|| Error::Other(format!("missing GR00T DiT timestep {step_index}")))?;
            let timestep = linear_silu(
                &*self.backend,
                timestep_projection,
                &self.action.timestep_input,
            )?;
            let timestep = linear(&*self.backend, &timestep, &self.action.timestep_output)?;
            Some(self.backend.silu(&timestep)?)
        };
        let block_modulations = if use_static_conditioning {
            Some(self.action.dit_modulations.get(step_index).ok_or_else(|| {
                Error::Other(format!("missing GR00T DiT modulations {step_index}"))
            })?)
        } else {
            None
        };

        for (layer_index, block) in self.action.dit_blocks.iter().enumerate() {
            let _block_range = crate::profiling::trace::range("gr00t_dit_block");
            let dynamic_modulation;
            let modulation = if let Some(modulations) = block_modulations {
                modulations.get(layer_index).ok_or_else(|| {
                    Error::Other(format!(
                        "missing GR00T DiT modulation {step_index}/{layer_index}"
                    ))
                })?
            } else {
                dynamic_modulation = linear(
                    &*self.backend,
                    dynamic_timestep
                        .as_ref()
                        .ok_or_else(|| Error::Other("missing dynamic GR00T timestep".into()))?,
                    &block.adaptive_norm,
                )?
                .reshape(vec![2 * self.config.input_embedding_dim])?;
                &dynamic_modulation
            };
            let prepared_key_value = cross_attention.get(layer_index).ok_or_else(|| {
                Error::Other(format!(
                    "missing GR00T cross-attention cache slot {layer_index}"
                ))
            })?;
            let uses_fused_self_qkv = prepared_key_value.is_none()
                && block
                    .attention
                    .fused_qkv
                    .as_ref()
                    .is_some_and(DeviceLinearWeights::supports_fused_self_qkv);
            let consumes_quantized_query = !uses_fused_self_qkv;
            let fused_qkv_normalized = if uses_fused_self_qkv && USE_FUSED_ADAPTIVE_LAYER_NORM {
                block
                    .attention
                    .fused_qkv
                    .as_ref()
                    .expect("fused self QKV was checked")
                    .adaptive_layer_norm_quantized(
                        &hidden,
                        modulation,
                        self.config.diffusion.norm_eps,
                        &self.backend,
                    )?
            } else {
                None
            };
            let query_normalized = if fused_qkv_normalized.is_none()
                && USE_FUSED_ADAPTIVE_LAYER_NORM_QUANT
                && consumes_quantized_query
            {
                block.attention.query.adaptive_layer_norm_quantized(
                    &hidden,
                    modulation,
                    self.config.diffusion.norm_eps,
                    &self.backend,
                )?
            } else {
                None
            };
            let (normalized, normalized_quantized, normalized_fused_qkv) =
                if let Some((normalized, quantized)) = fused_qkv_normalized {
                    (normalized, None, Some(quantized))
                } else if let Some((normalized, quantized)) = query_normalized {
                    (normalized, Some(quantized), None)
                } else {
                    (
                        kernels::norm::adaptive_layer(
                            self.backend.context(),
                            &hidden,
                            modulation,
                            self.config.diffusion.norm_eps,
                        )?,
                        None,
                        None,
                    )
                };
            let source = match dit_attention_source(&self.config, layer_index)? {
                Gr00tDitAttentionSource::FullBackbone => {
                    return Err(Error::Other(
                        "GR00T phase-one runtime requires AlternateVLDiT".into(),
                    ));
                }
                Gr00tDitAttentionSource::NonImageBackbone => non_image_backbone,
                Gr00tDitAttentionSource::ImageBackbone => image_backbone,
                Gr00tDitAttentionSource::StateActionSelf => &normalized,
            };
            let attention = match prepared_key_value {
                Some(key_value) => forward_attention_with_prepared_key_value(
                    &*self.backend,
                    &normalized,
                    normalized_quantized.as_ref(),
                    key_value,
                    &block.attention,
                    self.config.diffusion.num_attention_heads,
                    self.config.diffusion.attention_head_dim,
                    true,
                )?,
                None => forward_attention(
                    &*self.backend,
                    &normalized,
                    normalized_quantized.as_ref(),
                    normalized_fused_qkv.as_ref(),
                    source,
                    &block.attention,
                    self.config.diffusion.num_attention_heads,
                    self.config.diffusion.attention_head_dim,
                    true,
                )?,
            };
            hidden = self.backend.add(&hidden, &attention)?;
            let normalized = layer_norm(
                &self.backend,
                &hidden,
                &self.action.dit_norm_weight,
                &self.action.dit_norm_bias,
                self.config.diffusion.norm_eps,
            )?;
            hidden = if USE_FUSED_FFN_BIAS_RESIDUAL
                && block.feed_forward.output.uses_quantized_output()
            {
                forward_feed_forward_residual(
                    &*self.backend,
                    &normalized,
                    &block.feed_forward,
                    &hidden,
                )?
            } else {
                let feed_forward =
                    forward_feed_forward(&*self.backend, &normalized, &block.feed_forward)?;
                self.backend.add(&hidden, &feed_forward)?
            };
        }

        let dynamic_output_modulation;
        let output_modulation = if use_static_conditioning {
            self.action
                .output_modulations
                .get(step_index)
                .ok_or_else(|| {
                    Error::Other(format!("missing GR00T output modulation {step_index}"))
                })?
        } else {
            dynamic_output_modulation = linear(
                &*self.backend,
                dynamic_timestep
                    .as_ref()
                    .ok_or_else(|| Error::Other("missing dynamic GR00T timestep".into()))?,
                &self.action.output_modulation,
            )?
            .reshape(vec![2 * self.config.input_embedding_dim])?;
            &dynamic_output_modulation
        };
        let hidden = kernels::norm::adaptive_layer(
            self.backend.context(),
            &hidden,
            output_modulation,
            1e-6,
        )?;
        let hidden = linear(&*self.backend, &hidden, &self.action.output_projection)?;
        let decoded = linear_relu(&*self.backend, &hidden, &selected.action_decoder.input)?;
        let decoded = linear(&*self.backend, &decoded, &selected.action_decoder.output)?;
        let velocity = kernels::elementwise::contiguous_rows(
            self.backend.context(),
            &decoded,
            1,
            self.config.action_horizon,
        )?;
        kernels::elementwise::euler_update_bf16(
            self.backend.context(),
            actions,
            &velocity,
            step.delta,
        )
    }
}

struct Gr00tCategoryWeights {
    state_encoder: Gr00tCategoryMlpWeights,
    action_encoder_input: Gr00tCategoryLinearWeights,
    action_encoder_time: Gr00tCategoryLinearWeights,
    action_encoder_output: Gr00tCategoryLinearWeights,
    action_decoder: Gr00tCategoryMlpWeights,
}

struct Gr00tDeviceMlpWeights<L: DeviceLinearWeights> {
    input: L,
    output: L,
}

struct Gr00tDeviceActionEncoderWeights<L: DeviceLinearWeights> {
    input: L,
    time: L,
    output: L,
}

struct Gr00tDeviceEmbodimentWeights<E: Gr00tPrecisionExecution> {
    state_encoder: Gr00tDeviceMlpWeights<E::Dense>,
    action_encoder: Gr00tDeviceActionEncoderWeights<E::Dense>,
    action_decoder: Gr00tDeviceMlpWeights<E::Dense>,
}

struct Gr00tDeviceAttentionWeights<E: Gr00tPrecisionExecution> {
    query: E::Dense,
    key: E::Dense,
    value: E::Dense,
    fused_qkv: Option<E::FusedQkv>,
    output: E::Dense,
}

struct Gr00tPreparedKeyValue {
    key: Tensor,
    value: Tensor,
}

struct Gr00tDeviceFeedForwardWeights<E: Gr00tPrecisionExecution> {
    input: E::FeedForward,
    output: E::FeedForward,
}

struct Gr00tDeviceDitBlockWeights<E: Gr00tPrecisionExecution> {
    adaptive_norm: E::Dense,
    attention: Gr00tDeviceAttentionWeights<E>,
    feed_forward: Gr00tDeviceFeedForwardWeights<E>,
}

struct Gr00tDeviceVlSelfAttentionBlockWeights<E: Gr00tPrecisionExecution> {
    attention_norm: Gr00tLayerNormWeights,
    attention: Gr00tDeviceAttentionWeights<E>,
    feed_forward_norm: Gr00tLayerNormWeights,
    feed_forward: Gr00tDeviceFeedForwardWeights<E>,
}

impl Gr00tCategoryWeights {
    fn select(&self, embodiment_id: usize) -> Result<Gr00tEmbodimentWeights> {
        Ok(Gr00tEmbodimentWeights {
            state_encoder: select_category_mlp(&self.state_encoder, embodiment_id)?,
            action_encoder: Gr00tActionEncoderWeights {
                input: select_category_linear(&self.action_encoder_input, embodiment_id)?,
                time: select_category_linear(&self.action_encoder_time, embodiment_id)?,
                output: select_category_linear(&self.action_encoder_output, embodiment_id)?,
            },
            action_decoder: select_category_mlp(&self.action_decoder, embodiment_id)?,
        })
    }
}

struct Gr00tDeviceActionWeights<E: Gr00tPrecisionExecution> {
    action_position_embedding: Option<Tensor>,
    backbone_layer_norm: Option<Gr00tLayerNormWeights>,
    vl_self_attention: Vec<Gr00tDeviceVlSelfAttentionBlockWeights<E>>,
    timestep_input: E::Dense,
    timestep_output: E::Dense,
    dit_blocks: Vec<Gr00tDeviceDitBlockWeights<E>>,
    output_modulation: E::Dense,
    output_projection: E::Dense,
    dit_norm_weight: Tensor,
    dit_norm_bias: Tensor,
    action_time_embeddings: Vec<Tensor>,
    dit_timestep_projections: Vec<Tensor>,
    dit_modulations: Vec<Vec<Tensor>>,
    output_modulations: Vec<Tensor>,
}

impl<E: Gr00tPrecisionExecution> Gr00tDeviceActionWeights<E> {
    fn from_host(
        config: &Gr00tConfig,
        weights: Gr00tActionHeadWeights,
        execution: &E,
        backend: &RuntimeBackend,
    ) -> Result<(Gr00tCategoryWeights, Self)> {
        let Gr00tActionHeadWeights {
            state_encoder,
            action_encoder_input,
            action_encoder_time,
            action_encoder_output,
            action_decoder,
            position_embedding,
            backbone_layer_norm,
            vl_self_attention,
            timestep_input,
            timestep_output,
            dit_blocks,
            output_modulation,
            output_projection,
        } = weights;
        let category = Gr00tCategoryWeights {
            state_encoder,
            action_encoder_input,
            action_encoder_time,
            action_encoder_output,
            action_decoder,
        };
        let position_embedding = position_embedding
            .map(|tensor| first_host_rows(&tensor, config.action_horizon))
            .transpose()?
            .map(|tensor| backend.to_device(&tensor))
            .transpose()?;
        let backbone_layer_norm = backbone_layer_norm
            .map(|weights| transfer_layer_norm(weights, backend))
            .transpose()?;
        let vl_self_attention = vl_self_attention
            .into_iter()
            .enumerate()
            .map(|(index, weights)| transfer_vl_block(weights, execution, index, backend))
            .collect::<Result<Vec<_>>>()?;
        let timestep_input = transfer_linear(
            timestep_input,
            execution,
            "action_head.timestep_input",
            backend,
        )?;
        let timestep_output = transfer_linear(
            timestep_output,
            execution,
            "action_head.timestep_output",
            backend,
        )?;
        let dit_blocks = dit_blocks
            .into_iter()
            .enumerate()
            .map(|(index, weights)| transfer_dit_block(weights, execution, index, backend))
            .collect::<Result<Vec<_>>>()?;
        let output_modulation = transfer_linear(
            output_modulation,
            execution,
            "action_head.output_modulation",
            backend,
        )?;
        let output_projection = transfer_linear(
            output_projection,
            execution,
            "action_head.output_projection",
            backend,
        )?;
        let dit_norm_weight = backend.to_device(&Tensor::from_bf16(
            vec![config.input_embedding_dim],
            &vec![bf16::ONE; config.input_embedding_dim],
        )?)?;
        let dit_norm_bias = backend.to_device(&Tensor::zeros(
            vec![config.input_embedding_dim],
            DType::BF16,
        ))?;

        let schedule = flow_schedule(config.num_inference_timesteps, config.num_timestep_buckets)?;
        let mut action_time_embeddings = Vec::with_capacity(schedule.len());
        let mut dit_timestep_projections = Vec::with_capacity(schedule.len());
        let mut dit_modulations = Vec::with_capacity(schedule.len());
        let mut output_modulations = Vec::with_capacity(schedule.len());
        for step in schedule {
            let embedding =
                action_timestep_embedding(step.discrete_timestep, config.input_embedding_dim)?;
            let repeated = (0..config.action_horizon)
                .flat_map(|_| embedding.iter().copied())
                .map(bf16::from_f32)
                .collect::<Vec<_>>();
            action_time_embeddings.push(backend.to_device(&Tensor::from_bf16(
                vec![config.action_horizon, config.input_embedding_dim],
                &repeated,
            )?)?);

            let projection = dit_timestep_projection(step.discrete_timestep, 256)?
                .into_iter()
                .map(bf16::from_f32)
                .collect::<Vec<_>>();
            let projection = backend.to_device(&Tensor::from_bf16(vec![1, 256], &projection)?)?;
            dit_timestep_projections.push(projection.clone());
            let timestep = linear_silu(backend, &projection, &timestep_input)?;
            let timestep = linear(backend, &timestep, &timestep_output)?;
            let activated_timestep = backend.silu(&timestep)?;
            dit_modulations.push(
                dit_blocks
                    .iter()
                    .map(|block| {
                        linear(backend, &activated_timestep, &block.adaptive_norm)
                            .and_then(|value| value.reshape(vec![2 * config.input_embedding_dim]))
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
            output_modulations.push(
                linear(backend, &activated_timestep, &output_modulation)?
                    .reshape(vec![2 * config.input_embedding_dim])?,
            );
        }

        Ok((
            category,
            Self {
                action_position_embedding: position_embedding,
                backbone_layer_norm,
                vl_self_attention,
                timestep_input,
                timestep_output,
                dit_blocks,
                output_modulation,
                output_projection,
                dit_norm_weight,
                dit_norm_bias,
                action_time_embeddings,
                dit_timestep_projections,
                dit_modulations,
                output_modulations,
            },
        ))
    }
}

fn transfer_embodiment<E: Gr00tPrecisionExecution>(
    weights: Gr00tEmbodimentWeights,
    execution: &E,
    backend: &RuntimeBackend,
) -> Result<Gr00tDeviceEmbodimentWeights<E>> {
    Ok(Gr00tDeviceEmbodimentWeights {
        state_encoder: transfer_mlp(
            weights.state_encoder,
            execution,
            "action_head.state_encoder",
            backend,
        )?,
        action_encoder: Gr00tDeviceActionEncoderWeights {
            input: transfer_linear(
                weights.action_encoder.input,
                execution,
                "action_head.action_encoder.input",
                backend,
            )?,
            time: transfer_linear(
                weights.action_encoder.time,
                execution,
                "action_head.action_encoder.time",
                backend,
            )?,
            output: transfer_linear(
                weights.action_encoder.output,
                execution,
                "action_head.action_encoder.output",
                backend,
            )?,
        },
        action_decoder: transfer_mlp(
            weights.action_decoder,
            execution,
            "action_head.action_decoder",
            backend,
        )?,
    })
}

fn transfer_mlp<E: Gr00tPrecisionExecution>(
    weights: Gr00tMlpWeights,
    execution: &E,
    name: &str,
    backend: &RuntimeBackend,
) -> Result<Gr00tDeviceMlpWeights<E::Dense>> {
    Ok(Gr00tDeviceMlpWeights {
        input: transfer_linear(weights.input, execution, &format!("{name}.input"), backend)?,
        output: transfer_linear(
            weights.output,
            execution,
            &format!("{name}.output"),
            backend,
        )?,
    })
}

fn transfer_linear<E: Gr00tPrecisionExecution>(
    weights: Gr00tLinearWeights,
    execution: &E,
    name: &str,
    backend: &RuntimeBackend,
) -> Result<E::Dense> {
    execution.transfer_dense(weights, name, backend)
}

fn transfer_layer_norm(
    weights: Gr00tLayerNormWeights,
    backend: &RuntimeBackend,
) -> Result<Gr00tLayerNormWeights> {
    Ok(Gr00tLayerNormWeights {
        weight: backend.to_device(&weights.weight)?,
        bias: backend.to_device(&weights.bias)?,
    })
}

fn transfer_attention<E: Gr00tPrecisionExecution>(
    weights: Gr00tAttentionWeights,
    execution: &E,
    name: &str,
    backend: &RuntimeBackend,
) -> Result<Gr00tDeviceAttentionWeights<E>> {
    let Gr00tAttentionWeights {
        query,
        key,
        value,
        output,
    } = weights;
    let shared_qkv_input = matches!(
        (
            query.weight.shape().dims(),
            key.weight.shape().dims(),
            value.weight.shape().dims(),
        ),
        ([query_rows, _], [key_rows, _], [value_rows, _])
            if query_rows == key_rows && key_rows == value_rows
    );
    let fused_qkv = if shared_qkv_input {
        let weight = concat_columns_bf16(&[&query.weight, &key.weight, &value.weight])?;
        let bias = concat_vectors_bf16(&[&query.bias, &key.bias, &value.bias])?;
        execution.transfer_fused_qkv(
            Gr00tLinearWeights { weight, bias },
            &format!("{name}.fused_qkv"),
            backend,
        )?
    } else {
        None
    };
    Ok(Gr00tDeviceAttentionWeights {
        query: transfer_linear(query, execution, &format!("{name}.query"), backend)?,
        key: transfer_linear(key, execution, &format!("{name}.key"), backend)?,
        value: transfer_linear(value, execution, &format!("{name}.value"), backend)?,
        fused_qkv,
        output: transfer_linear(output, execution, &format!("{name}.output"), backend)?,
    })
}

fn transfer_feed_forward<E: Gr00tPrecisionExecution>(
    weights: Gr00tFeedForwardWeights,
    execution: &E,
    name: &str,
    backend: &RuntimeBackend,
) -> Result<Gr00tDeviceFeedForwardWeights<E>> {
    Ok(Gr00tDeviceFeedForwardWeights {
        input: execution.transfer_feed_forward(weights.input, &format!("{name}.input"), backend)?,
        output: execution.transfer_feed_forward(
            weights.output,
            &format!("{name}.output"),
            backend,
        )?,
    })
}

fn transfer_dit_block<E: Gr00tPrecisionExecution>(
    weights: Gr00tDitBlockWeights,
    execution: &E,
    index: usize,
    backend: &RuntimeBackend,
) -> Result<Gr00tDeviceDitBlockWeights<E>> {
    let name = format!("action_head.dit_blocks.{index}");
    Ok(Gr00tDeviceDitBlockWeights {
        adaptive_norm: transfer_linear(
            weights.adaptive_norm,
            execution,
            &format!("{name}.adaptive_norm"),
            backend,
        )?,
        attention: transfer_attention(
            weights.attention,
            execution,
            &format!("{name}.attention"),
            backend,
        )?,
        feed_forward: transfer_feed_forward(
            weights.feed_forward,
            execution,
            &format!("{name}.feed_forward"),
            backend,
        )?,
    })
}

fn transfer_vl_block<E: Gr00tPrecisionExecution>(
    weights: Gr00tVlSelfAttentionBlockWeights,
    execution: &E,
    index: usize,
    backend: &RuntimeBackend,
) -> Result<Gr00tDeviceVlSelfAttentionBlockWeights<E>> {
    let name = format!("action_head.vl_self_attention.{index}");
    Ok(Gr00tDeviceVlSelfAttentionBlockWeights {
        attention_norm: transfer_layer_norm(weights.attention_norm, backend)?,
        attention: transfer_attention(
            weights.attention,
            execution,
            &format!("{name}.attention"),
            backend,
        )?,
        feed_forward_norm: transfer_layer_norm(weights.feed_forward_norm, backend)?,
        feed_forward: transfer_feed_forward(
            weights.feed_forward,
            execution,
            &format!("{name}.feed_forward"),
            backend,
        )?,
    })
}

fn linear<L: DeviceLinearWeights>(
    backend: &RuntimeBackend,
    input: &Tensor,
    weights: &L,
) -> Result<Tensor> {
    let output = weights.forward(input, backend)?;
    kernels::elementwise::bias_bf16(backend.context(), &output, weights.bias())
}

fn rms_norm(backend: &RuntimeBackend, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    kernels::norm::rms_bf16(backend.context(), input, weight, eps)
}

fn layer_norm(
    backend: &RuntimeBackend,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    kernels::norm::layer_bf16(backend.context(), input, weight, bias, eps)
}

fn linear_quantized<L: DeviceLinearWeights>(
    backend: &RuntimeBackend,
    input: &L::ReusableInput,
    weights: &L,
) -> Result<Tensor> {
    let output = weights.forward_reusable_quantized(input, backend)?;
    kernels::elementwise::bias_bf16(backend.context(), &output, weights.bias())
}

fn linear_relu<L: DeviceLinearWeights>(
    backend: &RuntimeBackend,
    input: &Tensor,
    weights: &L,
) -> Result<Tensor> {
    let output = weights.forward(input, backend)?;
    kernels::activation::bias_relu_bf16(backend.context(), &output, weights.bias())
}

fn linear_silu<L: DeviceLinearWeights>(
    backend: &RuntimeBackend,
    input: &Tensor,
    weights: &L,
) -> Result<Tensor> {
    let output = weights.forward(input, backend)?;
    kernels::activation::bias_silu_bf16(backend.context(), &output, weights.bias())
}

fn linear_gelu<L: DeviceLinearWeights>(
    backend: &RuntimeBackend,
    input: &Tensor,
    weights: &L,
) -> Result<Tensor> {
    let output = weights.forward(input, backend)?;
    kernels::activation::bias_gelu_bf16(backend.context(), &output, weights.bias())
}

fn forward_feed_forward<E: Gr00tPrecisionExecution>(
    backend: &RuntimeBackend,
    input: &Tensor,
    weights: &Gr00tDeviceFeedForwardWeights<E>,
) -> Result<Tensor> {
    let output = forward_feed_forward_projection(backend, input, weights)?;
    kernels::elementwise::bias_bf16(backend.context(), &output, weights.output.bias())
}

fn forward_feed_forward_projection<E: Gr00tPrecisionExecution>(
    backend: &RuntimeBackend,
    input: &Tensor,
    weights: &Gr00tDeviceFeedForwardWeights<E>,
) -> Result<Tensor> {
    let fused_quantized_bias_gelu = USE_FUSED_FP8_BIAS_GELU_QUANT
        && weights.input.activation_scale().is_some()
        && weights.output.activation_scale().is_some();
    if fused_quantized_bias_gelu {
        let hidden = weights.input.forward(input, backend)?;
        let bias = weights.input.bias().ok_or_else(|| {
            Error::Other("GR00T fused quantized bias GELU requires input bias".into())
        })?;
        let hidden = weights
            .output
            .quantize_bias_gelu_reusable_input(&hidden, bias, backend)?
            .ok_or_else(|| {
                Error::Other("GR00T FP8 bias-GELU fusion did not produce quantized input".into())
            })?;
        let output = weights
            .output
            .forward_reusable_quantized(&hidden, backend)?;
        return Ok(output);
    }
    let hidden = linear_gelu(backend, input, &weights.input)?;
    weights.output.forward(&hidden, backend)
}

fn forward_feed_forward_residual<E: Gr00tPrecisionExecution>(
    backend: &RuntimeBackend,
    input: &Tensor,
    weights: &Gr00tDeviceFeedForwardWeights<E>,
    residual: &Tensor,
) -> Result<Tensor> {
    let output = forward_feed_forward_projection(backend, input, weights)?;
    kernels::fused::bias_then_residual_bf16(
        backend.context(),
        &output,
        weights.output.bias(),
        residual,
    )
}

fn forward_attention<E: Gr00tPrecisionExecution>(
    backend: &RuntimeBackend,
    query_input: &Tensor,
    prequantized_query: Option<&<E::Dense as DeviceLinearWeights>::ReusableInput>,
    prequantized_fused_qkv: Option<&<E::FusedQkv as DeviceLinearWeights>::ReusableInput>,
    key_value_input: &Tensor,
    weights: &Gr00tDeviceAttentionWeights<E>,
    heads: usize,
    head_dim: usize,
    apply_output_bias: bool,
) -> Result<Tensor> {
    let query_len = query_input.shape().dims()[0];
    let key_value_len = key_value_input.shape().dims()[0];
    if std::ptr::eq(query_input, key_value_input) {
        if let Some(fused_qkv) = &weights.fused_qkv {
            if USE_FUSED_SELF_QKV
                && fused_qkv.supports_fused_self_qkv()
                && !super::calibration::is_active()
            {
                let owned_input;
                let quantized_input = if let Some(input) = prequantized_fused_qkv {
                    Some(input)
                } else {
                    owned_input = fused_qkv.quantize_reusable_input(query_input, backend)?;
                    owned_input.as_ref()
                };
                let qkv = if let Some(input) = quantized_input {
                    let projected = fused_qkv.forward_reusable_quantized(input, backend)?;
                    kernels::elementwise::bias_bf16(
                        backend.context(),
                        &projected,
                        fused_qkv.bias(),
                    )?
                } else {
                    linear(backend, query_input, fused_qkv)?
                };
                let output = if USE_STRIDED_FUSED_QKV_ATTENTION {
                    kernels::attention::noncausal_strided_qkv(
                        backend.context(),
                        &qkv,
                        heads,
                        head_dim,
                    )?
                } else {
                    let qkv = kernels::attention::split_qkv_bias_bf16(
                        backend.context(),
                        &qkv,
                        None,
                        heads,
                        head_dim,
                    )?;
                    kernels::attention::noncausal(
                        backend.context(),
                        &qkv.q,
                        &qkv.k,
                        &qkv.v,
                        heads,
                        head_dim,
                    )?
                };
                return if apply_output_bias {
                    linear(backend, &output, &weights.output)
                } else {
                    weights.output.forward(&output, backend)
                };
            }
        }
    }
    let query_scale = weights.query.activation_scale();
    let key_scale = weights.key.activation_scale();
    let value_scale = weights.value.activation_scale();
    let shared_all = std::ptr::eq(query_input, key_value_input)
        && query_scale.is_some()
        && query_scale == key_scale
        && key_scale == value_scale;
    let shared_key_value = key_scale.is_some() && key_scale == value_scale;
    let owned_query_quantized = if prequantized_query.is_none() && shared_all {
        weights
            .query
            .quantize_reusable_input(query_input, backend)?
    } else {
        None
    };
    let query_quantized = prequantized_query.or(owned_query_quantized.as_ref());
    let key_value_quantized = if shared_all {
        None
    } else if shared_key_value {
        weights
            .key
            .quantize_reusable_input(key_value_input, backend)?
    } else {
        None
    };
    if shared_all && USE_FUSED_FP8_SELF_QKV_BIAS {
        let input = query_quantized.ok_or_else(|| {
            Error::Other("fused FP8 self QKV bias requires shared quantized input".into())
        })?;
        let query = weights.query.forward_reusable_quantized(input, backend)?;
        let key = weights.key.forward_reusable_quantized(input, backend)?;
        let value = weights.value.forward_reusable_quantized(input, backend)?;
        let (query, key, value) =
            kernels::elementwise::bias_qkv_in_place_bf16(
                backend.context(),
                query,
                key,
                value,
                weights.query.bias().ok_or_else(|| {
                    Error::Other("fused FP8 self QKV query bias is missing".into())
                })?,
                weights
                    .key
                    .bias()
                    .ok_or_else(|| Error::Other("fused FP8 self QKV key bias is missing".into()))?,
                weights.value.bias().ok_or_else(|| {
                    Error::Other("fused FP8 self QKV value bias is missing".into())
                })?,
            )?;
        let query = query.reshape(vec![query_len, heads, head_dim])?;
        let key = key.reshape(vec![key_value_len, heads, head_dim])?;
        let value = value.reshape(vec![key_value_len, heads, head_dim])?;
        let output = kernels::attention::noncausal(
            backend.context(),
            &query,
            &key,
            &value,
            heads,
            head_dim,
        )?;
        return if apply_output_bias {
            linear(backend, &output, &weights.output)
        } else {
            weights.output.forward(&output, backend)
        };
    }
    let query = if let Some(input) = query_quantized {
        linear_quantized(backend, input, &weights.query)?
    } else {
        linear(backend, query_input, &weights.query)?
    }
    .reshape(vec![query_len, heads, head_dim])?;
    let key_source = if shared_all {
        query_quantized
    } else {
        key_value_quantized.as_ref()
    };
    let key = if let Some(input) = key_source {
        linear_quantized(backend, input, &weights.key)?
    } else {
        linear(backend, key_value_input, &weights.key)?
    }
    .reshape(vec![key_value_len, heads, head_dim])?;
    let value = if let Some(input) = key_source {
        linear_quantized(backend, input, &weights.value)?
    } else {
        linear(backend, key_value_input, &weights.value)?
    }
    .reshape(vec![key_value_len, heads, head_dim])?;
    let output =
        kernels::attention::noncausal(backend.context(), &query, &key, &value, heads, head_dim)?;
    if apply_output_bias {
        linear(backend, &output, &weights.output)
    } else {
        weights.output.forward(&output, backend)
    }
}

fn prepare_attention_key_value<E: Gr00tPrecisionExecution>(
    backend: &RuntimeBackend,
    key_value_input: &Tensor,
    weights: &Gr00tDeviceAttentionWeights<E>,
    heads: usize,
    head_dim: usize,
) -> Result<Gr00tPreparedKeyValue> {
    let key_value_len = key_value_input.shape().dims()[0];
    let key_scale = weights.key.activation_scale();
    let value_scale = weights.value.activation_scale();
    let shared_key_value = key_scale.is_some() && key_scale == value_scale;
    let key_value_quantized = if shared_key_value {
        weights
            .key
            .quantize_reusable_input(key_value_input, backend)?
    } else {
        None
    };
    let key = if let Some(input) = key_value_quantized.as_ref() {
        linear_quantized(backend, input, &weights.key)?
    } else {
        linear(backend, key_value_input, &weights.key)?
    }
    .reshape(vec![key_value_len, heads, head_dim])?;
    let value = if let Some(input) = key_value_quantized.as_ref() {
        linear_quantized(backend, input, &weights.value)?
    } else {
        linear(backend, key_value_input, &weights.value)?
    }
    .reshape(vec![key_value_len, heads, head_dim])?;
    Ok(Gr00tPreparedKeyValue { key, value })
}

fn forward_attention_with_prepared_key_value<E: Gr00tPrecisionExecution>(
    backend: &RuntimeBackend,
    query_input: &Tensor,
    prequantized_query: Option<&<E::Dense as DeviceLinearWeights>::ReusableInput>,
    key_value: &Gr00tPreparedKeyValue,
    weights: &Gr00tDeviceAttentionWeights<E>,
    heads: usize,
    head_dim: usize,
    apply_output_bias: bool,
) -> Result<Tensor> {
    let query_len = query_input.shape().dims()[0];
    let query = if let Some(input) = prequantized_query {
        linear_quantized(backend, input, &weights.query)?
    } else {
        linear(backend, query_input, &weights.query)?
    }
    .reshape(vec![query_len, heads, head_dim])?;
    let output = kernels::attention::noncausal(
        backend.context(),
        &query,
        &key_value.key,
        &key_value.value,
        heads,
        head_dim,
    )?;
    if apply_output_bias {
        linear(backend, &output, &weights.output)
    } else {
        weights.output.forward(&output, backend)
    }
}

fn forward_standard_block<E: Gr00tPrecisionExecution>(
    backend: &RuntimeBackend,
    hidden: &Tensor,
    weights: &Gr00tDeviceVlSelfAttentionBlockWeights<E>,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<Tensor> {
    let normalized = layer_norm(
        backend,
        hidden,
        &weights.attention_norm.weight,
        &weights.attention_norm.bias,
        eps,
    )?;
    let attention = forward_attention(
        backend,
        &normalized,
        None,
        None,
        &normalized,
        &weights.attention,
        heads,
        head_dim,
        true,
    )?;
    let hidden = backend.add(hidden, &attention)?;
    let normalized = layer_norm(
        backend,
        &hidden,
        &weights.feed_forward_norm.weight,
        &weights.feed_forward_norm.bias,
        eps,
    )?;
    if USE_FUSED_FFN_BIAS_RESIDUAL && weights.feed_forward.output.uses_quantized_output() {
        forward_feed_forward_residual(backend, &normalized, &weights.feed_forward, &hidden)
    } else {
        let feed_forward = forward_feed_forward(backend, &normalized, &weights.feed_forward)?;
        backend.add(&hidden, &feed_forward)
    }
}

fn concat_columns_bf16(tensors: &[&Tensor]) -> Result<Tensor> {
    let first = tensors
        .first()
        .ok_or_else(|| Error::Other("cannot concatenate an empty tensor list".into()))?;
    let [rows, _] = first.shape().dims() else {
        return Err(Error::Other(
            "column concatenation requires matrices".into(),
        ));
    };
    let rows = *rows;
    let mut total_cols = 0usize;
    let mut columns = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let [tensor_rows, cols] = tensor.shape().dims() else {
            return Err(Error::Other(
                "column concatenation requires matrices".into(),
            ));
        };
        if tensor.dtype() != DType::BF16 || *tensor_rows != rows {
            return Err(Error::Other(
                "column concatenation requires BF16 matrices with equal rows".into(),
            ));
        }
        total_cols += *cols;
        columns.push((tensor.as_bf16()?, *cols));
    }
    let mut values = Vec::with_capacity(rows * total_cols);
    for row in 0..rows {
        for (column, cols) in &columns {
            values.extend_from_slice(&column[row * cols..(row + 1) * cols]);
        }
    }
    Tensor::from_bf16(vec![rows, total_cols], &values)
}

fn concat_vectors_bf16(tensors: &[&Tensor]) -> Result<Tensor> {
    let mut values = Vec::new();
    for tensor in tensors {
        let [length] = tensor.shape().dims() else {
            return Err(Error::Other(
                "vector concatenation requires one-dimensional tensors".into(),
            ));
        };
        if tensor.dtype() != DType::BF16 {
            return Err(Error::Other(
                "vector concatenation requires BF16 tensors".into(),
            ));
        }
        values.reserve(*length);
        values.extend_from_slice(tensor.as_bf16()?);
    }
    Tensor::from_bf16(vec![values.len()], &values)
}

fn qwen_matmul<L: DeviceLinearWeights>(
    backend: &RuntimeBackend,
    linears: &BTreeMap<String, L>,
    name: &str,
    input: &Tensor,
    bf16_weight: &Tensor,
) -> Result<Tensor> {
    if let Some(weights) = linears.get(name) {
        if input.dtype() == DType::F8E4M3 {
            return weights.forward_quantized_tensor(input, backend);
        }
        return weights.forward(input, backend);
    }
    super::calibration::observe(name, input)?;
    kernels::gemm::bf16(backend.context(), input, bf16_weight)
}

fn forward_qwen_layer<L: DeviceLinearWeights>(
    backend: &RuntimeBackend,
    config: &Qwen3VLConfig,
    weights: &Qwen3VLTextWeights,
    linears: &BTreeMap<String, L>,
    input: &Tensor,
    layer_index: usize,
    position_ids: &DeviceBuffer,
) -> Result<Tensor> {
    let sequence_len = input.shape().dims()[0];
    let text = &config.text;
    let layer = &weights.layers[layer_index];
    let prefix = format!("backbone.text.layers.{layer_index}");
    let query_name = format!("{prefix}.query");
    let key_name = format!("{prefix}.key");
    let value_name = format!("{prefix}.value");
    let shared_qkv = linears
        .get(&query_name)
        .zip(linears.get(&key_name))
        .zip(linears.get(&value_name))
        .filter(|((query, key), value)| {
            query.activation_scale() == key.activation_scale()
                && key.activation_scale() == value.activation_scale()
        });
    let fused_qkv_norm = shared_qkv.is_some() && USE_FUSED_FP8_RMS_NORM;
    let normalized = if !fused_qkv_norm {
        Some(rms_norm(
            backend,
            input,
            &layer.attn_norm_weight,
            text.rms_norm_eps,
        )?)
    } else {
        None
    };
    let shared_qkv_input = shared_qkv
        .map(|((query, _), _)| {
            if fused_qkv_norm {
                kernels::norm::rms_quant_bf16_e4m3(
                    backend.context(),
                    input,
                    &layer.attn_norm_weight,
                    text.rms_norm_eps,
                    query.activation_scale().ok_or_else(|| {
                        Error::Other(format!("{query_name} is missing its FP8 activation scale"))
                    })?,
                )
            } else {
                query
                    .quantize_tensor_input(
                        normalized.as_ref().ok_or_else(|| {
                            Error::Other(format!("BF16 QKV input is unavailable for {prefix}"))
                        })?,
                        backend,
                    )?
                    .ok_or_else(|| Error::Other("shared QKV weights must be FP8".into()))
            }
        })
        .transpose()?;
    let query = if let (Some(((query, _), _)), Some(input)) = (shared_qkv, &shared_qkv_input) {
        query.forward_quantized_tensor(input, backend)?
    } else {
        qwen_matmul(
            backend,
            linears,
            &query_name,
            normalized.as_ref().ok_or_else(|| {
                Error::Other(format!("BF16 QKV input is unavailable for {prefix}"))
            })?,
            &layer.wq,
        )?
    };
    let key = if let (Some(((_, key), _)), Some(input)) = (shared_qkv, &shared_qkv_input) {
        key.forward_quantized_tensor(input, backend)?
    } else {
        qwen_matmul(
            backend,
            linears,
            &key_name,
            normalized.as_ref().ok_or_else(|| {
                Error::Other(format!("BF16 QKV input is unavailable for {prefix}"))
            })?,
            &layer.wk,
        )?
    };
    let value = if let (Some(((_, _), value)), Some(input)) = (shared_qkv, &shared_qkv_input) {
        value.forward_quantized_tensor(input, backend)?
    } else {
        qwen_matmul(
            backend,
            linears,
            &value_name,
            normalized.as_ref().ok_or_else(|| {
                Error::Other(format!("BF16 QKV input is unavailable for {prefix}"))
            })?,
            &layer.wv,
        )?
    };
    let query = query.reshape(vec![sequence_len * text.n_heads, text.head_dim])?;
    let key = key.reshape(vec![sequence_len * text.n_kv_heads, text.head_dim])?;
    let value = value.reshape(vec![sequence_len, text.n_kv_heads, text.head_dim])?;
    let query = rms_norm(backend, &query, &layer.q_norm_weight, text.rms_norm_eps)?
        .reshape(vec![sequence_len, text.n_heads, text.head_dim])?;
    let key = rms_norm(backend, &key, &layer.k_norm_weight, text.rms_norm_eps)?.reshape(vec![
        sequence_len,
        text.n_kv_heads,
        text.head_dim,
    ])?;
    let query = kernels::rope::apply_mrope(
        backend.context(),
        &query,
        text.n_heads,
        text.head_dim,
        text.rope_theta,
        text.mrope_section,
        position_ids,
    )?;
    let key = kernels::rope::apply_mrope(
        backend.context(),
        &key,
        text.n_kv_heads,
        text.head_dim,
        text.rope_theta,
        text.mrope_section,
        position_ids,
    )?;
    let attention =
        kernels::attention::causal_gqa_prefill_bf16(backend.context(), &query, &key, &value)?
            .ok_or_else(|| {
                Error::Other(
                    "GR00T Qwen prefill requires the in-tree BF16 causal GQA attention backend"
                        .into(),
                )
            })?;
    let attention = attention.reshape(vec![sequence_len, text.n_heads * text.head_dim])?;
    let attention = qwen_matmul(
        backend,
        linears,
        &format!("{prefix}.output"),
        &attention,
        &layer.wo,
    )?;
    let hidden = backend.add(input, &attention)?;
    let gate_name = format!("{prefix}.gate");
    let up_name = format!("{prefix}.up");
    let shared_gate_up = linears
        .get(&gate_name)
        .zip(linears.get(&up_name))
        .filter(|(gate, up)| gate.can_share_quantized_input_with(up));
    let fused_gate_up_norm = shared_gate_up
        .is_some_and(|(gate, _)| gate.activation_scale().is_some())
        && USE_FUSED_FP8_RMS_NORM;
    let normalized = if !fused_gate_up_norm {
        Some(rms_norm(
            backend,
            &hidden,
            &layer.ffn_norm_weight,
            text.rms_norm_eps,
        )?)
    } else {
        None
    };
    let shared_gate_up_input = shared_gate_up
        .map(|(gate, _)| {
            if fused_gate_up_norm {
                gate.rms_norm_quantized(
                    &hidden,
                    &layer.ffn_norm_weight,
                    text.rms_norm_eps,
                    backend,
                )?
                .ok_or_else(|| {
                    Error::Other(format!("{gate_name} did not produce FP8 normalized input"))
                })
            } else {
                gate.quantize_reusable_input(
                    normalized.as_ref().ok_or_else(|| {
                        Error::Other(format!("BF16 gate/up input is unavailable for {prefix}"))
                    })?,
                    backend,
                )?
                .ok_or_else(|| {
                    Error::Other(format!(
                        "{gate_name} did not produce reusable quantized input"
                    ))
                })
            }
        })
        .transpose()?;
    let gate = if let (Some((gate, _)), Some(input)) = (shared_gate_up, &shared_gate_up_input) {
        gate.forward_reusable_quantized(input, backend)?
    } else {
        qwen_matmul(
            backend,
            linears,
            &gate_name,
            normalized.as_ref().ok_or_else(|| {
                Error::Other(format!("BF16 gate input is unavailable for {prefix}"))
            })?,
            &layer.w_gate,
        )?
    };
    let up = if let (Some((_, up)), Some(input)) = (shared_gate_up, &shared_gate_up_input) {
        up.forward_reusable_quantized(input, backend)?
    } else {
        qwen_matmul(
            backend,
            linears,
            &up_name,
            normalized.as_ref().ok_or_else(|| {
                Error::Other(format!("BF16 up input is unavailable for {prefix}"))
            })?,
            &layer.w_up,
        )?
    };
    let down_name = format!("{prefix}.down");
    let feed_forward = if let Some(down) = linears
        .get(&down_name)
        .filter(|down| down.activation_scale().is_some())
    {
        let gated = kernels::activation::silu_mul_quant_bf16_e4m3(
            backend.context(),
            &gate,
            &up,
            down.activation_scale().ok_or_else(|| {
                Error::Other(format!("{down_name} is missing its FP8 activation scale"))
            })?,
        )?;
        down.forward_quantized_tensor(&gated, backend)?
    } else {
        if let Some(output) = linears
            .get(&down_name)
            .filter(|_| USE_FUSED_W8A8_SILU_MUL_QUANT)
            .map(|down| down.fused_silu_mul(&gate, &up, backend))
            .transpose()?
            .flatten()
        {
            output
        } else {
            let gated = if USE_FUSED_BF16_SILU_MUL {
                kernels::activation::silu_mul_bf16(backend.context(), &gate, &up)?
            } else {
                let gate = backend.silu(&gate)?;
                backend.mul(&gate, &up)?
            };
            qwen_matmul(backend, linears, &down_name, &gated, &layer.w_down)?
        }
    };
    backend.add(&hidden, &feed_forward)
}

fn multimodal_rope_positions(
    config: &Qwen3VLConfig,
    token_ids: &[u32],
    image_grid_thw: &[[u32; 3]],
) -> Result<Vec<u32>> {
    let merge = u32::try_from(config.vision.spatial_merge_size)
        .map_err(|_| Error::Other("Qwen3-VL spatial merge size overflow".into()))?;
    let mut positions = Vec::with_capacity(token_ids.len() * 3);
    let mut start = 0usize;
    let mut image_index = 0usize;
    let mut next_position = 0u32;
    loop {
        let image_start =
            (start..token_ids.len()).find(|index| token_ids[*index] == config.image_token_id);
        let Some(image_start) = image_start else {
            for _ in start..token_ids.len() {
                positions.extend_from_slice(&[next_position; 3]);
                next_position = next_position
                    .checked_add(1)
                    .ok_or_else(|| Error::Other("Qwen3-VL position overflow".into()))?;
            }
            break;
        };
        for _ in start..image_start {
            positions.extend_from_slice(&[next_position; 3]);
            next_position = next_position
                .checked_add(1)
                .ok_or_else(|| Error::Other("Qwen3-VL position overflow".into()))?;
        }
        let [temporal, height, width] = *image_grid_thw.get(image_index).ok_or_else(|| {
            Error::Other(format!("missing Qwen3-VL grid for image {image_index}"))
        })?;
        if merge == 0 || height % merge != 0 || width % merge != 0 {
            return Err(Error::Other(format!(
                "Qwen3-VL grid [{temporal}, {height}, {width}] is incompatible with merge {merge}"
            )));
        }
        image_index += 1;
        let merged_height = height / merge;
        let merged_width = width / merge;
        for time in 0..temporal {
            for row in 0..merged_height {
                for column in 0..merged_width {
                    positions.extend_from_slice(&[
                        next_position + time,
                        next_position + row,
                        next_position + column,
                    ]);
                }
            }
        }
        let image_tokens_u32 = temporal
            .checked_mul(merged_height)
            .and_then(|value| value.checked_mul(merged_width))
            .ok_or_else(|| Error::Other("Qwen3-VL image token count overflow".into()))?;
        let image_tokens = usize::try_from(image_tokens_u32)
            .map_err(|_| Error::Other("Qwen3-VL image token count overflow".into()))?;
        let image_end = image_start
            .checked_add(image_tokens)
            .ok_or_else(|| Error::Other("Qwen3-VL image token range overflow".into()))?;
        let valid_placeholder_run = match token_ids.get(image_start..image_end) {
            Some(tokens) => tokens.iter().all(|token| *token == config.image_token_id),
            None => false,
        };
        if !valid_placeholder_run {
            return Err(Error::Other(
                "Qwen3-VL image placeholder run does not match image_grid_thw".into(),
            ));
        }
        let image_position_span = temporal
            .saturating_sub(1)
            .max(merged_height.saturating_sub(1))
            .max(merged_width.saturating_sub(1))
            .checked_add(1)
            .ok_or_else(|| Error::Other("Qwen3-VL image position span overflow".into()))?;
        next_position = next_position
            .checked_add(image_position_span)
            .ok_or_else(|| Error::Other("Qwen3-VL position overflow".into()))?;
        start = image_end;
    }
    if image_index != image_grid_thw.len() || positions.len() != token_ids.len() * 3 {
        return Err(Error::Other(format!(
            "Qwen3-VL token/grid mismatch: consumed {image_index}/{} grids and produced {}/{} positions",
            image_grid_thw.len(),
            positions.len(),
            token_ids.len() * 3
        )));
    }
    Ok(positions)
}

fn scatter_rows(
    backend: &RuntimeBackend,
    destination: &Tensor,
    source: &Tensor,
    add: bool,
    indices: &kernels::elementwise::PreparedRowIndices,
) -> Result<Tensor> {
    kernels::elementwise::scatter_rows_bf16_prepared(
        backend.context(),
        destination,
        indices,
        source,
        add,
    )
}

fn first_host_rows(tensor: &Tensor, rows: usize) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if tensor.device() != Device::Cpu
        || tensor.dtype() != DType::BF16
        || dims.len() != 2
        || rows == 0
        || rows > dims[0]
    {
        return Err(Error::Other(format!(
            "GR00T position embedding cannot select {rows} rows from {} {dims:?} on {}",
            tensor.dtype(),
            tensor.device()
        )));
    }
    let columns = dims[1];
    Tensor::from_bf16(vec![rows, columns], &tensor.as_bf16()?[..rows * columns])
}

fn validate_runtime_support(config: &Gr00tConfig) -> Result<()> {
    if !config.use_alternate_vl_dit
        || !config.diffusion.interleave_self_attention
        || config.diffusion.positional_embeddings.is_some()
        || !config.diffusion.attention_bias
        || config.diffusion.upcast_attention
        || config.diffusion.norm_elementwise_affine
    {
        return Err(Error::Other(
            "GR00T CUDA phase one supports the released AlternateVLDiT BF16 layout only".into(),
        ));
    }
    if let Some(vl) = &config.vl_self_attention {
        if vl.positional_embeddings.is_some() || !vl.attention_bias || vl.upcast_attention {
            return Err(Error::Other(
                "GR00T CUDA phase one does not support VL positional embeddings, bias-free attention, or attention upcasting"
                    .into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qwen_config() -> Qwen3VLConfig {
        Qwen3VLConfig::from_json_str(
            r#"{
                "image_token_id": 99,
                "text_config": {
                    "rope_scaling": {"mrope_section": [24, 20, 20]}
                },
                "vision_config": {"spatial_merge_size": 2}
            }"#,
        )
        .unwrap()
    }

    fn graph_observation(config: &Gr00tConfig) -> Gr00tObservation {
        Gr00tObservation {
            pixel_values: Tensor::zeros(vec![4, 1536], DType::BF16),
            image_grid_thw: vec![[1, 2, 2]],
            token_ids: vec![10, 99, 11],
            attention_mask: vec![1, 1, 1],
            state: Tensor::zeros(
                vec![1, config.state_history_length, config.max_state_dim],
                DType::BF16,
            ),
            embodiment_id: 0,
            noise: Tensor::zeros(
                vec![1, config.action_horizon, config.max_action_dim],
                DType::BF16,
            ),
        }
    }

    #[test]
    fn multimodal_positions_match_qwen3vl_for_multiple_images() {
        let image = 99;
        let tokens = [10, image, image, image, image, 11, image, 12];
        let positions =
            multimodal_rope_positions(&qwen_config(), &tokens, &[[1, 4, 4], [1, 2, 2]]).unwrap();
        assert_eq!(
            positions,
            vec![0, 0, 0, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5,]
        );
    }

    #[test]
    fn multimodal_positions_reject_grid_placeholder_mismatch() {
        assert!(multimodal_rope_positions(&qwen_config(), &[10, 99], &[[1, 4, 4]]).is_err());
    }

    #[test]
    fn graph_key_tracks_topology_but_allows_text_value_updates() {
        let config = Gr00tConfig::default();
        let backbone = qwen_config();
        let observation = graph_observation(&config);
        let key = Gr00tGraphKey::new(&observation, &config, &backbone).unwrap();

        let mut changed_text = observation.clone();
        changed_text.token_ids[0] = 42;
        assert_eq!(
            key,
            Gr00tGraphKey::new(&changed_text, &config, &backbone).unwrap()
        );

        let mut changed_layout = observation.clone();
        changed_layout.token_ids.swap(0, 1);
        assert_ne!(
            key,
            Gr00tGraphKey::new(&changed_layout, &config, &backbone).unwrap()
        );

        let mut changed_grid = observation.clone();
        changed_grid.image_grid_thw = vec![[1, 4, 4]];
        assert_ne!(
            key,
            Gr00tGraphKey::new(&changed_grid, &config, &backbone).unwrap()
        );

        let mut changed_embodiment = observation;
        changed_embodiment.embodiment_id = 1;
        assert_ne!(
            key,
            Gr00tGraphKey::new(&changed_embodiment, &config, &backbone).unwrap()
        );
    }
}
