//! CUDA BF16 runtime for NVIDIA GR00T N1.7.
//!
//! The runtime is intentionally independent from the existing PI0.5-oriented
//! `VlaRuntime` trait. It accepts fully preprocessed GR00T input and preserves
//! the released checkpoint's four-step flow-matching semantics.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, LazyLock};
use std::time::Instant;

use apxinf_core::{Backend, DType, Device, Error, Graph, KvCache, Result, Tensor};
use half::bf16;

use crate::accelerator::create_backend;
use crate::accelerator::cuda::{
    downcast_arc, kernels, transfers, tuning, DeviceBuffer, KvCache as DeviceKvCache,
    RuntimeBackend,
};
use crate::qwen3vl::general::transfer_weights as transfer_text_weights;
use crate::qwen3vl::vision;
use crate::qwen3vl::vision_weights::transfer_vision_weights;
use crate::qwen3vl::{Qwen3VLConfig, Qwen3VLTextWeights, Qwen3VLVisionWeights};
use crate::ModelPrecision;

use super::fp8::{
    Gr00tDeviceLinearWeights, Gr00tFp8Calibration, Gr00tFp8Collector, Gr00tQuantizedLinearInput,
};
use super::weights::{select_category_linear, select_category_mlp};
use super::{
    action_timestep_embedding, build_backbone_token_groups, dit_attention_source,
    dit_timestep_projection, flow_schedule, Gr00tActionEncoderWeights, Gr00tActionHeadWeights,
    Gr00tAttentionWeights, Gr00tCategoryLinearWeights, Gr00tCategoryMlpWeights, Gr00tConfig,
    Gr00tDitAttentionSource, Gr00tDitBlockWeights, Gr00tEmbodimentWeights, Gr00tFeedForwardWeights,
    Gr00tFlowStep, Gr00tInferenceSpec, Gr00tLayerNormWeights, Gr00tLinearWeights, Gr00tLoadOptions,
    Gr00tMlpWeights, Gr00tObservation, Gr00tVlSelfAttentionBlockWeights, Gr00tWeights,
};

// The current stable-address bump arena needs cumulative allocation volume,
// not peak live bytes. Nsight measured 2.521 GiB for the representative
// three-view input; 4 GiB leaves alignment and small-shape variance headroom.
// This is intentionally a first graph-correctness bound, not the final memory
// plan. See doc/gr00t-n1.7/optimization-journal.md.
const INITIAL_GRAPH_WORKSPACE_BYTES: usize = 4usize << 30;
static USE_FUSED_FP8_RMS_NORM: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("APXINF_GR00T_DISABLE_FUSED_FP8_NORM").is_none());
static USE_FUSED_FP8_VISION_LAYER_NORM: LazyLock<bool> = LazyLock::new(|| {
    std::env::var_os("APXINF_GR00T_DISABLE_FUSED_FP8_VISION_LAYER_NORM").is_none()
});
static USE_FUSED_FP8_VISION_FFN_HANDOFF: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("APXINF_GR00T_FUSED_FP8_VISION_FFN_HANDOFF").is_ok_and(|value| value == "1")
});
static USE_FUSED_FP8_QWEN_QKV: LazyLock<bool> = LazyLock::new(|| {
    std::env::var_os("APXINF_GR00T_FUSED_QWEN_LINEAR").is_some()
        || std::env::var_os("APXINF_GR00T_FUSED_QWEN_QKV").is_some()
});
static USE_FUSED_FP8_QWEN_GATE_UP: LazyLock<bool> = LazyLock::new(|| {
    std::env::var_os("APXINF_GR00T_FUSED_QWEN_LINEAR").is_some()
        || std::env::var_os("APXINF_GR00T_FUSED_QWEN_GATE_UP").is_some()
});
static USE_FUSED_BF16_SELF_QKV: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("APXINF_GR00T_BF16_FUSED_SELF_QKV").map_or(true, |value| value != "0")
});
static USE_FUSED_W8A8_SELF_QKV: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("APXINF_GR00T_W8A8_FUSED_SELF_QKV").map_or(true, |value| value != "0")
});
static USE_FUSED_W8A8_ADAPTIVE_LAYER_NORM: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("APXINF_GR00T_W8A8_FUSED_ADAPTIVE_LAYER_NORM").map_or(true, |value| value != "0")
});
static USE_STRIDED_FUSED_QKV_ATTENTION: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("APXINF_GR00T_STRIDED_FUSED_QKV_ATTENTION").map_or(true, |value| value != "0")
});
static USE_FUSED_BF16_SILU_MUL: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("APXINF_GR00T_FUSED_SILU_MUL").map_or(true, |value| value != "0")
});
static USE_FUSED_W8A8_SILU_MUL_QUANT: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("APXINF_GR00T_W8A8_FUSED_SILU_MUL_QUANT").map_or(true, |value| value != "0")
});
static USE_FUSED_FP8_BIAS_GELU_QUANT: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("APXINF_GR00T_FP8_FUSED_BIAS_GELU_QUANT").map_or(true, |value| value != "0")
});
static USE_FUSED_FP8_SELF_QKV_BIAS: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("APXINF_GR00T_FP8_FUSED_SELF_QKV_BIAS").map_or(true, |value| value != "0")
});
static USE_FUSED_FP8_LINEAR_BIAS: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("APXINF_GR00T_FP8_FUSED_LINEAR_BIAS").is_ok_and(|value| value == "1")
});
static USE_FUSED_ADAPTIVE_LAYER_NORM_QUANT: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("APXINF_GR00T_FUSED_ADAPTIVE_LAYER_NORM_QUANT").map_or(true, |value| value != "0")
});
static USE_FUSED_FFN_BIAS_RESIDUAL: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("APXINF_GR00T_FUSED_FFN_BIAS_RESIDUAL").map_or(true, |value| value != "0")
});
static USE_PRECOMPUTED_QWEN_MROPE: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("APXINF_GR00T_PRECOMPUTED_QWEN_MROPE").is_ok_and(|value| value == "1")
});

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
    graphs: Gr00tCapturedGraphs,
    output: Tensor,
    inputs: Gr00tGraphInputs,
}

enum Gr00tCapturedGraphs {
    Whole {
        graph: Box<dyn Graph>,
        _workspace: kernels::GraphWorkspace,
    },
    Split {
        backbone: Box<dyn Graph>,
        action_head: Box<dyn Graph>,
        synchronize_between: bool,
        _backbone_workspace: kernels::GraphWorkspace,
        _action_workspace: kernels::GraphWorkspace,
    },
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
        match &self.graphs {
            Gr00tCapturedGraphs::Whole { graph, .. } => graph.replay()?,
            Gr00tCapturedGraphs::Split {
                backbone,
                action_head,
                synchronize_between,
                ..
            } => {
                backbone.replay()?;
                if *synchronize_between {
                    backend.synchronize()?;
                }
                action_head.replay()?;
            }
        }
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

pub struct Gr00tVlaRuntime {
    // Drop the captured graph before the weight/cache fields whose addresses
    // its nodes reference.
    captured_graph: Option<(Gr00tGraphKey, Gr00tCapturedGraph)>,
    config: Gr00tConfig,
    backbone_config: Qwen3VLConfig,
    backend: Arc<RuntimeBackend>,
    backbone_text: Qwen3VLTextWeights,
    backbone_fp8_linears: BTreeMap<String, Gr00tDeviceLinearWeights>,
    backbone_vision: Qwen3VLVisionWeights,
    backbone_cache: Box<dyn KvCache>,
    vision_position_cache: Option<(Vec<[u32; 3]>, vision::PreparedVisionPositions)>,
    backbone_token_cache: Option<(Vec<u32>, DeviceBuffer)>,
    backbone_position_cache: Option<(Vec<u32>, DeviceBuffer)>,
    backbone_row_index_cache: Option<Gr00tRowIndexCache>,
    category_weights: Gr00tCategoryWeights,
    selected_embodiment: Option<(usize, Gr00tDeviceEmbodimentWeights)>,
    action: Gr00tDeviceActionWeights,
    w8a8: bool,
    fp8_calibration: Option<Gr00tFp8Calibration>,
    fp8_collector: Option<Gr00tFp8Collector>,
    tuning_records: usize,
}

impl Gr00tVlaRuntime {
    pub fn from_dir(
        checkpoint_path: &Path,
        options: Gr00tLoadOptions,
        device: Device,
    ) -> Result<Self> {
        if !matches!(
            options.precision,
            ModelPrecision::Auto
                | ModelPrecision::Bf16
                | ModelPrecision::Fp8
                | ModelPrecision::W8A8
        ) {
            return Err(Error::Other(format!(
                "GR00T N1.7 supports Auto/BF16/FP8/W8A8 precision, got {:?}",
                options.precision
            )));
        }
        let device_id = match device {
            Device::Cuda(device_id) => device_id,
            other => {
                return Err(Error::Other(format!(
                    "GR00T N1.7 BF16 runtime currently requires CUDA, got {other}"
                )));
            }
        };
        let config = match options.config {
            Some(config) => {
                config.validate()?;
                config
            }
            None => Gr00tConfig::from_json_file(&checkpoint_path.join("config.json"))?,
        };
        let fp8_calibration = if options.precision == ModelPrecision::Fp8 {
            let path = options.fp8_calibration_path.as_deref().ok_or_else(|| {
                Error::Other(
                    "GR00T FP8 requires Gr00tLoadOptions.fp8_calibration_path pointing to a validated calibration JSON"
                        .into(),
                )
            })?;
            let calibration = Gr00tFp8Calibration::from_json_file(path)?;
            calibration.validate_checkpoint(&checkpoint_path.join("config.json"))?;
            eprintln!(
                "[apxinf] GR00T FP8 calibration fixture manifest SHA-256: {}",
                calibration.fixture_manifest_sha256()
            );
            Some(calibration)
        } else {
            None
        };
        let fp8_collector = if options.precision == ModelPrecision::Fp8 {
            None
        } else {
            Gr00tFp8Collector::from_env(checkpoint_path)?
        };
        validate_runtime_support(&config)?;
        let backbone_path = options.backbone_path.ok_or_else(|| {
            Error::Other(
                "GR00T loading requires Gr00tLoadOptions.backbone_path pointing to Cosmos-Reason2-2B"
                    .into(),
            )
        })?;
        let (backbone_config, weights) =
            Gr00tWeights::from_safetensors(&config, &backbone_path, checkpoint_path)?;

        let backend = create_backend(Device::Cuda(device_id))?;
        let backend = downcast_arc(backend)
            .ok_or_else(|| Error::Other("GR00T CUDA backend downcast failed".into()))?;
        let tuning_records = if let Some(path) = options.tuning_path.as_deref() {
            let database = tuning::TuningDb::from_json_file(path)?;
            kernels::gemm::install_tuning_db(backend.context(), &database)?;
            let records = backend
                .context()
                .tuning()
                .snapshot()?
                .gemm_records()
                .count();
            if records == 0 {
                return Err(Error::Other(format!(
                    "GR00T tactic database {} installed no compatible records",
                    path.display()
                )));
            }
            records
        } else {
            0
        };
        let Gr00tWeights {
            backbone_text,
            backbone_vision,
            action_head,
        } = weights;
        let backbone_fp8_linears = transfer_qwen_quantized_linears(
            &backbone_text,
            &backbone_vision,
            fp8_calibration.as_ref(),
            options.precision == ModelPrecision::W8A8,
            &*backend,
        )?;
        let backbone_text = transfer_text_weights(&backbone_text, &*backend)?;
        let backbone_vision = transfer_vision_weights(&backbone_vision, &*backend)?;
        let (category_weights, action) = Gr00tDeviceActionWeights::from_host(
            &config,
            action_head,
            fp8_calibration.as_ref(),
            fp8_collector.as_ref(),
            options.precision == ModelPrecision::W8A8,
            &*backend,
        )?;
        let backbone_cache = backend.create_kv_cache(
            backbone_config.text.n_layers,
            backbone_config.text.n_kv_heads,
            backbone_config.text.head_dim,
            backbone_config.text.max_position_embeddings.min(4096),
        );

        Ok(Self {
            config,
            captured_graph: None,
            backbone_config,
            backend,
            backbone_text,
            backbone_fp8_linears,
            backbone_vision,
            backbone_cache,
            vision_position_cache: None,
            backbone_token_cache: None,
            backbone_position_cache: None,
            backbone_row_index_cache: None,
            category_weights,
            selected_embodiment: None,
            action,
            w8a8: options.precision == ModelPrecision::W8A8,
            fp8_calibration,
            fp8_collector,
            tuning_records,
        })
    }

    pub fn config(&self) -> &Gr00tConfig {
        &self.config
    }

    /// Compatible GEMM tactic records installed in this runtime's CUDA
    /// context. This is exposed for benchmark provenance checks.
    pub fn tuning_record_count(&self) -> usize {
        self.tuning_records
    }

    pub fn inference_spec(&self, observation: &Gr00tObservation) -> Result<Gr00tInferenceSpec> {
        observation.inference_spec(&self.config)
    }

    /// Return whether the runtime currently owns a successfully captured graph.
    ///
    /// Production inference deliberately falls back to eager execution when
    /// capture is unavailable. Benchmarks use this signal to fail closed
    /// instead of accidentally reporting an eager run as CUDA Graph latency.
    pub fn has_captured_graph(&self) -> bool {
        self.captured_graph.is_some()
    }

    /// Run one batch-one GR00T inference and return CPU BF16 actions shaped
    /// `[1, action_horizon, max_action_dim]`.
    pub fn infer(&mut self, observation: &Gr00tObservation) -> Result<Tensor> {
        let _inference_range = crate::profiling::trace::range("gr00t_infer");
        self.validate_observation(observation)?;

        if self.fp8_collector.is_none() && graph_input_supported(observation) {
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
        let actions = self.finish_actions(actions)?;
        if let Some(collector) = &self.fp8_collector {
            collector.save()?;
        }
        Ok(actions)
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
        let backbone = self.infer_backbone_device(observation)?;
        self.infer_action_device(observation, &backbone)
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
        let backbone = self.forward_backbone(observation, &token_groups)?;
        let backbone = self.forward_backbone_adapter(&backbone)?;
        let row_indices = self
            .backbone_row_index_cache
            .as_ref()
            .ok_or_else(|| Error::Other("GR00T row indices were not prepared".into()))?;
        let non_image_backbone = kernels::elementwise::gather_rows_bf16_prepared(
            self.backend.context(),
            &backbone,
            &row_indices.non_image_indices,
        )?;
        let image_backbone = kernels::elementwise::gather_rows_bf16_prepared(
            self.backend.context(),
            &backbone,
            &row_indices.image_indices,
        )?;

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
        let state = linear_relu(&*self.backend, &state, &selected.state_encoder.input)?;
        let state = linear(&*self.backend, &state, &selected.state_encoder.output)?;

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
        let use_cross_attention_cache =
            std::env::var("APXINF_GR00T_CROSS_KV_CACHE").map_or(true, |value| value != "0");
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
            cross_attention.push(if use_cross_attention_cache {
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
                    .transpose()?
            } else {
                None
            });
        }
        for (step_index, step) in schedule.iter().enumerate() {
            actions = self.forward_flow_step(
                &actions,
                &state,
                &backbone.non_image,
                &backbone.image,
                &cross_attention,
                selected,
                step_index,
                *step,
            )?;
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
        let graph_mode =
            std::env::var("APXINF_GR00T_GRAPH_STRUCTURE").unwrap_or_else(|_| "whole".into());
        if graph_mode == "whole" {
            return self.capture_whole_graph(inputs, observation, device_observation);
        }
        let synchronize_between = match graph_mode.as_str() {
            "split-sync" => true,
            "split-async" => false,
            other => {
                return Err(Error::Other(format!(
                    "invalid APXINF_GR00T_GRAPH_STRUCTURE {other:?}; expected whole, split-sync or split-async"
                )))
            }
        };
        self.capture_split_graphs(inputs, observation, device_observation, synchronize_between)
    }

    fn capture_whole_graph(
        &mut self,
        inputs: Gr00tGraphInputs,
        _observation: &Gr00tObservation,
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
            graphs: Gr00tCapturedGraphs::Whole {
                graph,
                _workspace: workspace,
            },
            output,
            inputs,
        })
    }

    fn capture_split_graphs(
        &mut self,
        inputs: Gr00tGraphInputs,
        _observation: &Gr00tObservation,
        device_observation: Gr00tObservation,
        synchronize_between: bool,
    ) -> Result<Gr00tCapturedGraph> {
        let backbone_workspace =
            kernels::GraphWorkspace::new(INITIAL_GRAPH_WORKSPACE_BYTES, self.backend.device_id())?;
        let action_workspace =
            kernels::GraphWorkspace::new(INITIAL_GRAPH_WORKSPACE_BYTES, self.backend.device_id())?;

        let preflight_start = Instant::now();
        let backbone_output = kernels::prepare_with_workspace(&backbone_workspace, || {
            self.infer_backbone_device(&device_observation)
        })
        .map_err(|error| Error::Other(format!("backbone-graph preflight failed: {error}")))?;
        let eager_output = kernels::prepare_with_workspace(&action_workspace, || {
            self.infer_action_device(&device_observation, &backbone_output)
        })
        .map_err(|error| Error::Other(format!("action-graph preflight failed: {error}")))?;
        self.backend.synchronize()?;
        let preflight_seconds = preflight_start.elapsed().as_secs_f64();
        drop(eager_output);

        let capture_start = Instant::now();
        self.backend.begin_capture()?;
        let backbone_output = match kernels::with_workspace(&backbone_workspace, || {
            self.infer_backbone_device(&device_observation)
        }) {
            Ok(output) => output,
            Err(error) => {
                let _ = self.backend.end_capture();
                return Err(Error::Other(format!(
                    "backbone-graph capture body failed: {error}"
                )));
            }
        };
        let backbone_graph = self.backend.end_capture().map_err(|error| {
            Error::Other(format!("backbone-graph finalization failed: {error}"))
        })?;

        self.backend.begin_capture()?;
        let output = match kernels::with_workspace(&action_workspace, || {
            self.infer_action_device(&device_observation, &backbone_output)
        }) {
            Ok(output) => output,
            Err(error) => {
                let _ = self.backend.end_capture();
                return Err(Error::Other(format!(
                    "action-graph capture body failed: {error}"
                )));
            }
        };
        let action_head_graph = self
            .backend
            .end_capture()
            .map_err(|error| Error::Other(format!("action-graph finalization failed: {error}")))?;
        let capture_seconds = capture_start.elapsed().as_secs_f64();
        eprintln!(
            "[apxinf] GR00T split CUDA Graphs captured with backbone {} / {} and action {} / {} workspace bytes; preflight {:.3}s, capture+instantiate {:.3}s, synchronize_between={}",
            backbone_workspace.used(),
            backbone_workspace.capacity(),
            action_workspace.used(),
            action_workspace.capacity(),
            preflight_seconds,
            capture_seconds,
            synchronize_between,
        );
        Ok(Gr00tCapturedGraph {
            graphs: Gr00tCapturedGraphs::Split {
                backbone: backbone_graph,
                action_head: action_head_graph,
                synchronize_between,
                _backbone_workspace: backbone_workspace,
                _action_workspace: action_workspace,
            },
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
        let device = transfer_embodiment(
            host,
            self.fp8_calibration.as_ref(),
            self.fp8_collector.as_ref(),
            self.w8a8,
            &*self.backend,
        )?;
        self.selected_embodiment = Some((embodiment_id, device));
        Ok(())
    }

    fn forward_backbone(
        &mut self,
        observation: &Gr00tObservation,
        token_groups: &super::Gr00tBackboneTokenGroups,
    ) -> Result<Tensor> {
        let _backbone_range = crate::profiling::trace::range("gr00t_backbone");
        // GR00T performs a complete fixed-shape prefill on every inference,
        // so every K/V row that attention may read is overwritten. Rewind the
        // logical length while retaining stable cache addresses for graph
        // capture; keep the general KvCache::clear semantics unchanged for
        // other model families.
        self.backbone_cache
            .as_any_mut()
            .downcast_mut::<DeviceKvCache>()
            .ok_or_else(|| Error::Other("GR00T expected a CUDA KV cache".into()))?
            .rewind();
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
            qwen_matmul(
                &*self.backend,
                &self.backbone_fp8_linears,
                self.fp8_collector.as_ref(),
                name,
                input,
                weight,
            )
        };
        let vision_norm_matmul = |name: &str,
                                  input: &Tensor,
                                  norm_weight: &Tensor,
                                  norm_bias: &Tensor,
                                  eps: f32,
                                  weight: &Tensor| {
            if *USE_FUSED_FP8_VISION_LAYER_NORM {
                if let Some(linear) = self.backbone_fp8_linears.get(name) {
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
                    return linear.forward_quantized(&quantized, &*self.backend);
                }
            }
            let normalized = self
                .backend
                .layer_norm(input, norm_weight, norm_bias, eps)?;
            qwen_matmul(
                &*self.backend,
                &self.backbone_fp8_linears,
                self.fp8_collector.as_ref(),
                name,
                &normalized,
                weight,
            )
        };
        let vision_bias_gelu = |name: &str, input: &Tensor, bias: &Tensor| {
            let next_name = name
                .strip_suffix(".fc1")
                .map(|prefix| format!("{prefix}.fc2"));
            if *USE_FUSED_FP8_VISION_FFN_HANDOFF {
                if let Some(next) = next_name
                    .as_ref()
                    .and_then(|next_name| self.backbone_fp8_linears.get(next_name))
                    .filter(|next| next.is_fp8())
                {
                    return kernels::activation::bias_gelu_quant_bf16_e4m3(
                        self.backend.context(),
                        input,
                        bias,
                        next.activation_scale().ok_or_else(|| {
                            Error::Other(format!(
                                "{next_name:?} is missing its FP8 activation scale"
                            ))
                        })?,
                    );
                }
            }
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
        let mrope_table = if *USE_PRECOMPUTED_QWEN_MROPE {
            Some(kernels::rope::prepare_mrope_cos_sin(
                self.backend.context(),
                observation.token_ids.len(),
                self.backbone_config.text.head_dim,
                self.backbone_config.text.rope_theta,
                self.backbone_config.text.mrope_section,
                &position_buffer,
            )?)
        } else {
            None
        };
        let mut used_kv_cache = false;
        for layer in 0..self.backbone_config.text.n_layers {
            let _layer_range = crate::profiling::trace::range("gr00t_qwen_layer");
            let (next_hidden, layer_used_kv_cache) = forward_qwen_layer(
                &*self.backend,
                &self.backbone_config,
                &self.backbone_text,
                &self.backbone_fp8_linears,
                self.fp8_collector.as_ref(),
                &mut *self.backbone_cache,
                &hidden,
                layer,
                &position_buffer,
                mrope_table.as_ref(),
            )
            .map_err(|error| {
                Error::Other(format!("GR00T Qwen text layer {layer} failed: {error}"))
            })?;
            hidden = next_hidden;
            used_kv_cache |= layer_used_kv_cache;
            if let Some(deepstack) = vision_output.deepstack.get(layer) {
                hidden = scatter_rows(&*self.backend, &hidden, deepstack, true, &image_indices)?;
            }
        }
        if used_kv_cache {
            // During CUDA Graph capture, the fallback K/V append and attention
            // kernels record the zero-based offsets computed after `rewind()`.
            // Replay executes those fixed kernel arguments and overwrites every
            // valid cache row, so this host-only logical length does not need to
            // be rewound between replays. A later eager run or graph recapture
            // calls `rewind()` above before consulting it again.
            self.backbone_cache.advance(observation.token_ids.len());
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
                self.backend
                    .layer_norm(hidden, &weights.weight, &weights.bias, 1e-5)?
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
        selected: &Gr00tDeviceEmbodimentWeights,
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
        action_features = self.backend.concat_2d(&[&action_features, time_features])?;
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

        let use_static_conditioning =
            std::env::var("APXINF_GR00T_STATIC_CONDITIONING").map_or(true, |value| value != "0");
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
                && block.attention.fused_qkv.as_ref().is_some_and(|weights| {
                    (weights.is_w8a8() && *USE_FUSED_W8A8_SELF_QKV)
                        || (!weights.is_w8a8() && *USE_FUSED_BF16_SELF_QKV)
                });
            let uses_fused_w8a8_self_qkv = uses_fused_self_qkv
                && block
                    .attention
                    .fused_qkv
                    .as_ref()
                    .is_some_and(Gr00tDeviceLinearWeights::is_w8a8);
            let consumes_quantized_query = !uses_fused_self_qkv;
            let (normalized, normalized_quantized, normalized_low_precision) =
                if uses_fused_w8a8_self_qkv && *USE_FUSED_W8A8_ADAPTIVE_LAYER_NORM {
                    let (normalized, quantized) =
                        kernels::gemm::adaptive_layer_norm_quantize_w8a8_activation(
                            self.backend.context(),
                            &hidden,
                            modulation,
                            self.config.diffusion.norm_eps,
                        )?;
                    (
                        normalized,
                        None,
                        Some(Gr00tQuantizedLinearInput::W8A8(quantized)),
                    )
                } else if *USE_FUSED_ADAPTIVE_LAYER_NORM_QUANT && consumes_quantized_query {
                    if let Some(scale) = block.attention.query.activation_scale() {
                        let (normalized, quantized) =
                            kernels::norm::adaptive_layer_quant_bf16_e4m3(
                                self.backend.context(),
                                &hidden,
                                modulation,
                                self.config.diffusion.norm_eps,
                                scale,
                            )?;
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
                    }
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
                    normalized_low_precision.as_ref(),
                    source,
                    &block.attention,
                    self.config.diffusion.num_attention_heads,
                    self.config.diffusion.attention_head_dim,
                    true,
                )?,
            };
            hidden = self.backend.add(&hidden, &attention)?;
            let normalized = self.backend.layer_norm(
                &hidden,
                &self.action.dit_norm_weight,
                &self.action.dit_norm_bias,
                self.config.diffusion.norm_eps,
            )?;
            hidden = if *USE_FUSED_FFN_BIAS_RESIDUAL && block.feed_forward.output.is_quantized() {
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

struct Gr00tDeviceMlpWeights {
    input: Gr00tDeviceLinearWeights,
    output: Gr00tDeviceLinearWeights,
}

struct Gr00tDeviceActionEncoderWeights {
    input: Gr00tDeviceLinearWeights,
    time: Gr00tDeviceLinearWeights,
    output: Gr00tDeviceLinearWeights,
}

struct Gr00tDeviceEmbodimentWeights {
    state_encoder: Gr00tDeviceMlpWeights,
    action_encoder: Gr00tDeviceActionEncoderWeights,
    action_decoder: Gr00tDeviceMlpWeights,
}

struct Gr00tDeviceAttentionWeights {
    query: Gr00tDeviceLinearWeights,
    key: Gr00tDeviceLinearWeights,
    value: Gr00tDeviceLinearWeights,
    fused_qkv: Option<Gr00tDeviceLinearWeights>,
    output: Gr00tDeviceLinearWeights,
}

struct Gr00tPreparedKeyValue {
    key: Tensor,
    value: Tensor,
}

struct Gr00tDeviceFeedForwardWeights {
    input: Gr00tDeviceLinearWeights,
    output: Gr00tDeviceLinearWeights,
}

struct Gr00tDeviceDitBlockWeights {
    adaptive_norm: Gr00tDeviceLinearWeights,
    attention: Gr00tDeviceAttentionWeights,
    feed_forward: Gr00tDeviceFeedForwardWeights,
}

struct Gr00tDeviceVlSelfAttentionBlockWeights {
    attention_norm: Gr00tLayerNormWeights,
    attention: Gr00tDeviceAttentionWeights,
    feed_forward_norm: Gr00tLayerNormWeights,
    feed_forward: Gr00tDeviceFeedForwardWeights,
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

struct Gr00tDeviceActionWeights {
    action_position_embedding: Option<Tensor>,
    backbone_layer_norm: Option<Gr00tLayerNormWeights>,
    vl_self_attention: Vec<Gr00tDeviceVlSelfAttentionBlockWeights>,
    timestep_input: Gr00tDeviceLinearWeights,
    timestep_output: Gr00tDeviceLinearWeights,
    dit_blocks: Vec<Gr00tDeviceDitBlockWeights>,
    output_modulation: Gr00tDeviceLinearWeights,
    output_projection: Gr00tDeviceLinearWeights,
    dit_norm_weight: Tensor,
    dit_norm_bias: Tensor,
    action_time_embeddings: Vec<Tensor>,
    dit_timestep_projections: Vec<Tensor>,
    dit_modulations: Vec<Vec<Tensor>>,
    output_modulations: Vec<Tensor>,
}

impl Gr00tDeviceActionWeights {
    fn from_host(
        config: &Gr00tConfig,
        weights: Gr00tActionHeadWeights,
        calibration: Option<&Gr00tFp8Calibration>,
        collector: Option<&Gr00tFp8Collector>,
        w8a8: bool,
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
            .map(|(index, weights)| {
                transfer_vl_block(weights, calibration, collector, w8a8, index, backend)
            })
            .collect::<Result<Vec<_>>>()?;
        let timestep_input = transfer_linear(
            timestep_input,
            calibration,
            collector,
            w8a8,
            "action_head.timestep_input",
            backend,
        )?;
        let timestep_output = transfer_linear(
            timestep_output,
            calibration,
            collector,
            w8a8,
            "action_head.timestep_output",
            backend,
        )?;
        let dit_blocks = dit_blocks
            .into_iter()
            .enumerate()
            .map(|(index, weights)| {
                transfer_dit_block(weights, calibration, collector, w8a8, index, backend)
            })
            .collect::<Result<Vec<_>>>()?;
        let output_modulation = transfer_linear(
            output_modulation,
            calibration,
            collector,
            w8a8,
            "action_head.output_modulation",
            backend,
        )?;
        let output_projection = transfer_linear(
            output_projection,
            calibration,
            collector,
            w8a8,
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

fn transfer_embodiment(
    weights: Gr00tEmbodimentWeights,
    calibration: Option<&Gr00tFp8Calibration>,
    collector: Option<&Gr00tFp8Collector>,
    w8a8: bool,
    backend: &RuntimeBackend,
) -> Result<Gr00tDeviceEmbodimentWeights> {
    Ok(Gr00tDeviceEmbodimentWeights {
        state_encoder: transfer_mlp(
            weights.state_encoder,
            calibration,
            collector,
            w8a8,
            "action_head.state_encoder",
            backend,
        )?,
        action_encoder: Gr00tDeviceActionEncoderWeights {
            input: transfer_linear(
                weights.action_encoder.input,
                calibration,
                collector,
                w8a8,
                "action_head.action_encoder.input",
                backend,
            )?,
            time: transfer_linear(
                weights.action_encoder.time,
                calibration,
                collector,
                w8a8,
                "action_head.action_encoder.time",
                backend,
            )?,
            output: transfer_linear(
                weights.action_encoder.output,
                calibration,
                collector,
                w8a8,
                "action_head.action_encoder.output",
                backend,
            )?,
        },
        action_decoder: transfer_mlp(
            weights.action_decoder,
            calibration,
            collector,
            w8a8,
            "action_head.action_decoder",
            backend,
        )?,
    })
}

fn transfer_mlp(
    weights: Gr00tMlpWeights,
    calibration: Option<&Gr00tFp8Calibration>,
    collector: Option<&Gr00tFp8Collector>,
    w8a8: bool,
    name: &str,
    backend: &RuntimeBackend,
) -> Result<Gr00tDeviceMlpWeights> {
    Ok(Gr00tDeviceMlpWeights {
        input: transfer_linear(
            weights.input,
            calibration,
            collector,
            w8a8,
            &format!("{name}.input"),
            backend,
        )?,
        output: transfer_linear(
            weights.output,
            calibration,
            collector,
            w8a8,
            &format!("{name}.output"),
            backend,
        )?,
    })
}

fn transfer_linear(
    weights: Gr00tLinearWeights,
    calibration: Option<&Gr00tFp8Calibration>,
    collector: Option<&Gr00tFp8Collector>,
    w8a8: bool,
    name: &str,
    backend: &RuntimeBackend,
) -> Result<Gr00tDeviceLinearWeights> {
    match calibration {
        Some(calibration) => {
            Gr00tDeviceLinearWeights::fp8(weights, calibration.scale(name)?, backend)
        }
        None if w8a8 && name.contains("feed_forward") => {
            Gr00tDeviceLinearWeights::w8a8(weights, backend)
        }
        None => Gr00tDeviceLinearWeights::bf16(weights, name, collector, backend),
    }
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

fn transfer_attention(
    weights: Gr00tAttentionWeights,
    calibration: Option<&Gr00tFp8Calibration>,
    collector: Option<&Gr00tFp8Collector>,
    w8a8: bool,
    name: &str,
    backend: &RuntimeBackend,
) -> Result<Gr00tDeviceAttentionWeights> {
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
    let fused_qkv = if calibration.is_none()
        && collector.is_none()
        && w8a8
        && *USE_FUSED_W8A8_SELF_QKV
        && shared_qkv_input
        && query.weight.shape().dims() == [1536, 1536]
    {
        let weight = concat_columns_bf16(&[&query.weight, &key.weight, &value.weight])?;
        let bias = concat_vectors_bf16(&[&query.bias, &key.bias, &value.bias])?;
        Some(Gr00tDeviceLinearWeights::w8a8(
            Gr00tLinearWeights { weight, bias },
            backend,
        )?)
    } else if calibration.is_none() && collector.is_none() && shared_qkv_input {
        let weight = concat_columns_bf16(&[&query.weight, &key.weight, &value.weight])?;
        let bias = concat_vectors_bf16(&[&query.bias, &key.bias, &value.bias])?;
        Some(Gr00tDeviceLinearWeights::bf16(
            Gr00tLinearWeights { weight, bias },
            &format!("{name}.qkv_fused"),
            None,
            backend,
        )?)
    } else {
        None
    };
    Ok(Gr00tDeviceAttentionWeights {
        query: transfer_linear(
            query,
            calibration,
            collector,
            w8a8,
            &format!("{name}.query"),
            backend,
        )?,
        key: transfer_linear(
            key,
            calibration,
            collector,
            w8a8,
            &format!("{name}.key"),
            backend,
        )?,
        value: transfer_linear(
            value,
            calibration,
            collector,
            w8a8,
            &format!("{name}.value"),
            backend,
        )?,
        fused_qkv,
        output: transfer_linear(
            output,
            calibration,
            collector,
            w8a8,
            &format!("{name}.output"),
            backend,
        )?,
    })
}

fn transfer_feed_forward(
    weights: Gr00tFeedForwardWeights,
    calibration: Option<&Gr00tFp8Calibration>,
    collector: Option<&Gr00tFp8Collector>,
    w8a8: bool,
    name: &str,
    backend: &RuntimeBackend,
) -> Result<Gr00tDeviceFeedForwardWeights> {
    Ok(Gr00tDeviceFeedForwardWeights {
        input: transfer_linear(
            weights.input,
            calibration,
            collector,
            w8a8,
            &format!("{name}.input"),
            backend,
        )?,
        output: transfer_linear(
            weights.output,
            calibration,
            collector,
            w8a8,
            &format!("{name}.output"),
            backend,
        )?,
    })
}

fn transfer_dit_block(
    weights: Gr00tDitBlockWeights,
    calibration: Option<&Gr00tFp8Calibration>,
    collector: Option<&Gr00tFp8Collector>,
    w8a8: bool,
    index: usize,
    backend: &RuntimeBackend,
) -> Result<Gr00tDeviceDitBlockWeights> {
    let name = format!("action_head.dit_blocks.{index}");
    Ok(Gr00tDeviceDitBlockWeights {
        adaptive_norm: transfer_linear(
            weights.adaptive_norm,
            calibration,
            collector,
            w8a8,
            &format!("{name}.adaptive_norm"),
            backend,
        )?,
        attention: transfer_attention(
            weights.attention,
            calibration,
            collector,
            w8a8,
            &format!("{name}.attention"),
            backend,
        )?,
        feed_forward: transfer_feed_forward(
            weights.feed_forward,
            calibration,
            collector,
            w8a8,
            &format!("{name}.feed_forward"),
            backend,
        )?,
    })
}

fn transfer_vl_block(
    weights: Gr00tVlSelfAttentionBlockWeights,
    calibration: Option<&Gr00tFp8Calibration>,
    collector: Option<&Gr00tFp8Collector>,
    w8a8: bool,
    index: usize,
    backend: &RuntimeBackend,
) -> Result<Gr00tDeviceVlSelfAttentionBlockWeights> {
    let name = format!("action_head.vl_self_attention.{index}");
    Ok(Gr00tDeviceVlSelfAttentionBlockWeights {
        attention_norm: transfer_layer_norm(weights.attention_norm, backend)?,
        attention: transfer_attention(
            weights.attention,
            calibration,
            collector,
            w8a8,
            &format!("{name}.attention"),
            backend,
        )?,
        feed_forward_norm: transfer_layer_norm(weights.feed_forward_norm, backend)?,
        feed_forward: transfer_feed_forward(
            weights.feed_forward,
            calibration,
            collector,
            w8a8,
            &format!("{name}.feed_forward"),
            backend,
        )?,
    })
}

fn linear(
    backend: &RuntimeBackend,
    input: &Tensor,
    weights: &Gr00tDeviceLinearWeights,
) -> Result<Tensor> {
    if *USE_FUSED_FP8_LINEAR_BIAS && weights.is_fp8() {
        return weights.forward_fp8_bias(input, backend);
    }
    let output = weights.forward(input, backend)?;
    kernels::elementwise::bias_bf16(backend.context(), &output, weights.bias())
}

fn linear_quantized(
    backend: &RuntimeBackend,
    input: &Tensor,
    weights: &Gr00tDeviceLinearWeights,
) -> Result<Tensor> {
    if *USE_FUSED_FP8_LINEAR_BIAS && weights.is_fp8() {
        return weights.forward_fp8_quantized_bias(input, backend);
    }
    let output = weights.forward_quantized(input, backend)?;
    kernels::elementwise::bias_bf16(backend.context(), &output, weights.bias())
}

fn linear_relu(
    backend: &RuntimeBackend,
    input: &Tensor,
    weights: &Gr00tDeviceLinearWeights,
) -> Result<Tensor> {
    let output = weights.forward(input, backend)?;
    kernels::activation::bias_relu_bf16(backend.context(), &output, weights.bias())
}

fn linear_silu(
    backend: &RuntimeBackend,
    input: &Tensor,
    weights: &Gr00tDeviceLinearWeights,
) -> Result<Tensor> {
    let output = weights.forward(input, backend)?;
    kernels::activation::bias_silu_bf16(backend.context(), &output, weights.bias())
}

fn linear_gelu(
    backend: &RuntimeBackend,
    input: &Tensor,
    weights: &Gr00tDeviceLinearWeights,
) -> Result<Tensor> {
    let output = weights.forward(input, backend)?;
    kernels::activation::bias_gelu_bf16(backend.context(), &output, weights.bias())
}

fn forward_feed_forward(
    backend: &RuntimeBackend,
    input: &Tensor,
    weights: &Gr00tDeviceFeedForwardWeights,
) -> Result<Tensor> {
    let output = forward_feed_forward_projection(backend, input, weights)?;
    kernels::elementwise::bias_bf16(backend.context(), &output, weights.output.bias())
}

fn forward_feed_forward_projection(
    backend: &RuntimeBackend,
    input: &Tensor,
    weights: &Gr00tDeviceFeedForwardWeights,
) -> Result<Tensor> {
    let fused_quantized_bias_gelu =
        *USE_FUSED_FP8_BIAS_GELU_QUANT && weights.input.is_fp8() && weights.output.is_fp8();
    if fused_quantized_bias_gelu {
        let hidden = weights.input.forward(input, backend)?;
        let bias = weights.input.bias().ok_or_else(|| {
            Error::Other("GR00T fused quantized bias GELU requires input bias".into())
        })?;
        let hidden = weights
            .output
            .quantize_bias_gelu_reusable_input(&hidden, bias, backend)?;
        let output = weights
            .output
            .forward_reusable_quantized(&hidden, backend)?;
        return Ok(output);
    }
    let hidden = linear_gelu(backend, input, &weights.input)?;
    weights.output.forward(&hidden, backend)
}

fn forward_feed_forward_residual(
    backend: &RuntimeBackend,
    input: &Tensor,
    weights: &Gr00tDeviceFeedForwardWeights,
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

fn forward_attention(
    backend: &RuntimeBackend,
    query_input: &Tensor,
    prequantized_query: Option<&Tensor>,
    prequantized_low_precision: Option<&Gr00tQuantizedLinearInput>,
    key_value_input: &Tensor,
    weights: &Gr00tDeviceAttentionWeights,
    heads: usize,
    head_dim: usize,
    apply_output_bias: bool,
) -> Result<Tensor> {
    let query_len = query_input.shape().dims()[0];
    let key_value_len = key_value_input.shape().dims()[0];
    if std::ptr::eq(query_input, key_value_input) {
        if let Some(fused_qkv) = &weights.fused_qkv {
            let enabled = (fused_qkv.is_w8a8() && *USE_FUSED_W8A8_SELF_QKV)
                || (!fused_qkv.is_w8a8() && *USE_FUSED_BF16_SELF_QKV);
            if enabled {
                let qkv = if fused_qkv.is_w8a8() {
                    let owned_input;
                    let input = if let Some(input) = prequantized_low_precision {
                        input
                    } else {
                        owned_input = fused_qkv.quantize_reusable_input(query_input, backend)?;
                        &owned_input
                    };
                    let projected = fused_qkv.forward_reusable_quantized(input, backend)?;
                    kernels::elementwise::bias_bf16(
                        backend.context(),
                        &projected,
                        fused_qkv.bias(),
                    )?
                } else {
                    linear(backend, query_input, fused_qkv)?
                };
                let output = if *USE_STRIDED_FUSED_QKV_ATTENTION {
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
        weights.query.quantize_input(query_input, backend)?
    } else {
        None
    };
    let query_quantized = prequantized_query.or(owned_query_quantized.as_ref());
    let key_value_quantized = if shared_all {
        None
    } else if shared_key_value {
        weights.key.quantize_input(key_value_input, backend)?
    } else {
        None
    };
    if shared_all && *USE_FUSED_FP8_SELF_QKV_BIAS {
        let input = query_quantized.ok_or_else(|| {
            Error::Other("fused FP8 self QKV bias requires shared quantized input".into())
        })?;
        let query = weights.query.forward_quantized(input, backend)?;
        let key = weights.key.forward_quantized(input, backend)?;
        let value = weights.value.forward_quantized(input, backend)?;
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

fn prepare_attention_key_value(
    backend: &RuntimeBackend,
    key_value_input: &Tensor,
    weights: &Gr00tDeviceAttentionWeights,
    heads: usize,
    head_dim: usize,
) -> Result<Gr00tPreparedKeyValue> {
    let key_value_len = key_value_input.shape().dims()[0];
    let key_scale = weights.key.activation_scale();
    let value_scale = weights.value.activation_scale();
    let shared_key_value = key_scale.is_some() && key_scale == value_scale;
    let key_value_quantized = if shared_key_value {
        weights.key.quantize_input(key_value_input, backend)?
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

fn forward_attention_with_prepared_key_value(
    backend: &RuntimeBackend,
    query_input: &Tensor,
    prequantized_query: Option<&Tensor>,
    key_value: &Gr00tPreparedKeyValue,
    weights: &Gr00tDeviceAttentionWeights,
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

fn forward_standard_block(
    backend: &RuntimeBackend,
    hidden: &Tensor,
    weights: &Gr00tDeviceVlSelfAttentionBlockWeights,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<Tensor> {
    let normalized = backend.layer_norm(
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
    let normalized = backend.layer_norm(
        &hidden,
        &weights.feed_forward_norm.weight,
        &weights.feed_forward_norm.bias,
        eps,
    )?;
    if *USE_FUSED_FFN_BIAS_RESIDUAL && weights.feed_forward.output.is_quantized() {
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

fn transfer_qwen_quantized_linears(
    text_weights: &Qwen3VLTextWeights,
    vision_weights: &Qwen3VLVisionWeights,
    calibration: Option<&Gr00tFp8Calibration>,
    w8a8: bool,
    backend: &RuntimeBackend,
) -> Result<BTreeMap<String, Gr00tDeviceLinearWeights>> {
    if w8a8 {
        let mut output = BTreeMap::new();
        for (index, layer) in text_weights.layers.iter().enumerate() {
            let prefix = format!("backbone.text.layers.{index}");
            for (projection, weight) in [
                ("gate", &layer.w_gate),
                ("up", &layer.w_up),
                ("down", &layer.w_down),
            ] {
                output.insert(
                    format!("{prefix}.{projection}"),
                    Gr00tDeviceLinearWeights::w8a8_matrix(weight, backend)?,
                );
            }
        }
        return Ok(output);
    }
    let Some(calibration) = calibration else {
        return Ok(BTreeMap::new());
    };
    let mut output = BTreeMap::new();
    for (index, layer) in text_weights.layers.iter().enumerate() {
        let projections = [
            ("query", &layer.wq),
            ("key", &layer.wk),
            ("value", &layer.wv),
            ("output", &layer.wo),
            ("gate", &layer.w_gate),
            ("up", &layer.w_up),
            ("down", &layer.w_down),
        ];
        for (projection, weight) in projections {
            let name = format!("backbone.text.layers.{index}.{projection}");
            output.insert(
                name.clone(),
                Gr00tDeviceLinearWeights::fp8_matrix(weight, calibration.scale(&name)?, backend)?,
            );
        }
        let prefix = format!("backbone.text.layers.{index}");
        let query_scale = calibration.scale(&format!("{prefix}.query"))?;
        let key_scale = calibration.scale(&format!("{prefix}.key"))?;
        let value_scale = calibration.scale(&format!("{prefix}.value"))?;
        if query_scale == key_scale && key_scale == value_scale {
            let qkv = concat_columns_bf16(&[&layer.wq, &layer.wk, &layer.wv])?;
            output.insert(
                format!("{prefix}.qkv_fused"),
                Gr00tDeviceLinearWeights::fp8_matrix(&qkv, query_scale, backend)?,
            );
        }
        let gate_scale = calibration.scale(&format!("{prefix}.gate"))?;
        let up_scale = calibration.scale(&format!("{prefix}.up"))?;
        if gate_scale == up_scale {
            let gate_up = concat_columns_bf16(&[&layer.w_gate, &layer.w_up])?;
            output.insert(
                format!("{prefix}.gate_up_fused"),
                Gr00tDeviceLinearWeights::fp8_matrix(&gate_up, gate_scale, backend)?,
            );
        }
    }
    let mut insert = |name: String, weight: &Tensor| -> Result<()> {
        output.insert(
            name.clone(),
            Gr00tDeviceLinearWeights::fp8_matrix(weight, calibration.scale(&name)?, backend)?,
        );
        Ok(())
    };
    insert(
        "backbone.vision.patch_embed".into(),
        &vision_weights.patch_embed_weight,
    )?;
    for (index, block) in vision_weights.blocks.iter().enumerate() {
        insert(format!("backbone.vision.blocks.{index}.qkv"), &block.qkv_w)?;
        insert(
            format!("backbone.vision.blocks.{index}.output"),
            &block.proj_w,
        )?;
        insert(format!("backbone.vision.blocks.{index}.fc1"), &block.fc1_w)?;
        insert(format!("backbone.vision.blocks.{index}.fc2"), &block.fc2_w)?;
    }
    insert(
        "backbone.vision.merger.fc1".into(),
        &vision_weights.merger.fc1_w,
    )?;
    insert(
        "backbone.vision.merger.fc2".into(),
        &vision_weights.merger.fc2_w,
    )?;
    for (index, merger) in vision_weights.deepstack_mergers.iter().enumerate() {
        insert(
            format!("backbone.vision.deepstack.{index}.fc1"),
            &merger.fc1_w,
        )?;
        insert(
            format!("backbone.vision.deepstack.{index}.fc2"),
            &merger.fc2_w,
        )?;
    }
    Ok(output)
}

fn qwen_matmul(
    backend: &RuntimeBackend,
    fp8_linears: &BTreeMap<String, Gr00tDeviceLinearWeights>,
    collector: Option<&Gr00tFp8Collector>,
    name: &str,
    input: &Tensor,
    bf16_weight: &Tensor,
) -> Result<Tensor> {
    if let Some(weights) = fp8_linears.get(name) {
        if input.dtype() == DType::F8E4M3 {
            return weights.forward_quantized(input, backend);
        }
        return weights.forward(input, backend);
    }
    if let Some(collector) = collector {
        collector.observe(name, input, backend)?;
    }
    backend.matmul(input, bf16_weight)
}

fn forward_qwen_layer(
    backend: &RuntimeBackend,
    config: &Qwen3VLConfig,
    weights: &Qwen3VLTextWeights,
    fp8_linears: &BTreeMap<String, Gr00tDeviceLinearWeights>,
    collector: Option<&Gr00tFp8Collector>,
    cache: &mut dyn KvCache,
    input: &Tensor,
    layer_index: usize,
    position_ids: &DeviceBuffer,
    mrope_table: Option<&DeviceBuffer>,
) -> Result<(Tensor, bool)> {
    let sequence_len = input.shape().dims()[0];
    let text = &config.text;
    let layer = &weights.layers[layer_index];
    let prefix = format!("backbone.text.layers.{layer_index}");
    let query_name = format!("{prefix}.query");
    let key_name = format!("{prefix}.key");
    let value_name = format!("{prefix}.value");
    let fused_qkv_name = format!("{prefix}.qkv_fused");
    let shared_qkv = fp8_linears
        .get(&query_name)
        .zip(fp8_linears.get(&key_name))
        .zip(fp8_linears.get(&value_name))
        .filter(|((query, key), value)| {
            query.activation_scale() == key.activation_scale()
                && key.activation_scale() == value.activation_scale()
        });
    let fused_qkv_norm = shared_qkv.is_some() && *USE_FUSED_FP8_RMS_NORM;
    let normalized = if !fused_qkv_norm {
        Some(backend.rms_norm(input, &layer.attn_norm_weight, text.rms_norm_eps)?)
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
                    .quantize_input(
                        normalized.as_ref().ok_or_else(|| {
                            Error::Other(format!("BF16 QKV input is unavailable for {prefix}"))
                        })?,
                        backend,
                    )?
                    .ok_or_else(|| Error::Other("shared QKV weights must be FP8".into()))
            }
        })
        .transpose()?;
    let (query, key, value) = if *USE_FUSED_FP8_QWEN_QKV {
        if let (Some(fused), Some(input)) = (fp8_linears.get(&fused_qkv_name), &shared_qkv_input) {
            let qkv = fused.forward_quantized(input, backend)?;
            let split = kernels::attention::split_qkv_gqa_bf16(
                backend.context(),
                &qkv,
                text.n_heads * text.head_dim,
                text.n_kv_heads * text.head_dim,
            )?;
            (split.q, split.k, split.v)
        } else {
            return Err(Error::Other(format!(
                "fused Qwen QKV is unavailable for {prefix}"
            )));
        }
    } else {
        let query = if let (Some(((query, _), _)), Some(input)) = (shared_qkv, &shared_qkv_input) {
            query.forward_quantized(input, backend)?
        } else {
            qwen_matmul(
                backend,
                fp8_linears,
                collector,
                &query_name,
                normalized.as_ref().ok_or_else(|| {
                    Error::Other(format!("BF16 QKV input is unavailable for {prefix}"))
                })?,
                &layer.wq,
            )?
        };
        let key = if let (Some(((_, key), _)), Some(input)) = (shared_qkv, &shared_qkv_input) {
            key.forward_quantized(input, backend)?
        } else {
            qwen_matmul(
                backend,
                fp8_linears,
                collector,
                &key_name,
                normalized.as_ref().ok_or_else(|| {
                    Error::Other(format!("BF16 QKV input is unavailable for {prefix}"))
                })?,
                &layer.wk,
            )?
        };
        let value = if let (Some(((_, _), value)), Some(input)) = (shared_qkv, &shared_qkv_input) {
            value.forward_quantized(input, backend)?
        } else {
            qwen_matmul(
                backend,
                fp8_linears,
                collector,
                &value_name,
                normalized.as_ref().ok_or_else(|| {
                    Error::Other(format!("BF16 QKV input is unavailable for {prefix}"))
                })?,
                &layer.wv,
            )?
        };
        (query, key, value)
    };
    let query = query.reshape(vec![sequence_len * text.n_heads, text.head_dim])?;
    let key = key.reshape(vec![sequence_len * text.n_kv_heads, text.head_dim])?;
    let value = value.reshape(vec![sequence_len, text.n_kv_heads, text.head_dim])?;
    let query = backend
        .rms_norm(&query, &layer.q_norm_weight, text.rms_norm_eps)?
        .reshape(vec![sequence_len, text.n_heads, text.head_dim])?;
    let key = backend
        .rms_norm(&key, &layer.k_norm_weight, text.rms_norm_eps)?
        .reshape(vec![sequence_len, text.n_kv_heads, text.head_dim])?;
    let query = if let Some(table) = mrope_table {
        kernels::rope::apply_mrope_precomputed(
            backend.context(),
            &query,
            text.n_heads,
            text.head_dim,
            table,
        )?
    } else {
        kernels::rope::apply_mrope(
            backend.context(),
            &query,
            text.n_heads,
            text.head_dim,
            text.rope_theta,
            text.mrope_section,
            position_ids,
        )?
    };
    let key = if let Some(table) = mrope_table {
        kernels::rope::apply_mrope_precomputed(
            backend.context(),
            &key,
            text.n_kv_heads,
            text.head_dim,
            table,
        )?
    } else {
        kernels::rope::apply_mrope(
            backend.context(),
            &key,
            text.n_kv_heads,
            text.head_dim,
            text.rope_theta,
            text.mrope_section,
            position_ids,
        )?
    };
    let (attention, used_kv_cache) =
        match kernels::attention::causal_gqa_prefill_bf16(backend.context(), &query, &key, &value)?
        {
            Some(attention) => (attention, false),
            None => {
                backend.kv_append(cache, layer_index, &key, &value, sequence_len)?;
                (
                    backend.sdpa_prefill(
                        &query,
                        cache,
                        layer_index,
                        text.n_heads,
                        text.n_kv_heads,
                        text.head_dim,
                        sequence_len,
                        text.max_position_embeddings.min(4096),
                    )?,
                    true,
                )
            }
        };
    let attention = attention.reshape(vec![sequence_len, text.n_heads * text.head_dim])?;
    let attention = qwen_matmul(
        backend,
        fp8_linears,
        collector,
        &format!("{prefix}.output"),
        &attention,
        &layer.wo,
    )?;
    let hidden = backend.add(input, &attention)?;
    let gate_name = format!("{prefix}.gate");
    let up_name = format!("{prefix}.up");
    let shared_gate_up = fp8_linears
        .get(&gate_name)
        .zip(fp8_linears.get(&up_name))
        .filter(|(gate, up)| gate.can_share_quantized_input_with(up));
    let fused_gate_up_norm = shared_gate_up
        .is_some_and(|(gate, _)| gate.activation_scale().is_some())
        && *USE_FUSED_FP8_RMS_NORM;
    let use_fused_fp8_gate_up = shared_gate_up
        .is_some_and(|(gate, _)| gate.activation_scale().is_some())
        && *USE_FUSED_FP8_QWEN_GATE_UP;
    let normalized = if !fused_gate_up_norm {
        Some(backend.rms_norm(&hidden, &layer.ffn_norm_weight, text.rms_norm_eps)?)
    } else {
        None
    };
    let shared_gate_up_input = shared_gate_up
        .map(|(gate, _)| {
            if fused_gate_up_norm {
                kernels::norm::rms_quant_bf16_e4m3(
                    backend.context(),
                    &hidden,
                    &layer.ffn_norm_weight,
                    text.rms_norm_eps,
                    gate.activation_scale().ok_or_else(|| {
                        Error::Other(format!("{gate_name} is missing its FP8 activation scale"))
                    })?,
                )
                .map(Gr00tQuantizedLinearInput::Fp8)
            } else {
                gate.quantize_reusable_input(
                    normalized.as_ref().ok_or_else(|| {
                        Error::Other(format!("BF16 gate/up input is unavailable for {prefix}"))
                    })?,
                    backend,
                )
            }
        })
        .transpose()?;
    let separate_gate_up = if use_fused_fp8_gate_up {
        None
    } else {
        let gate = if let (Some((gate, _)), Some(input)) = (shared_gate_up, &shared_gate_up_input) {
            gate.forward_reusable_quantized(input, backend)?
        } else {
            qwen_matmul(
                backend,
                fp8_linears,
                collector,
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
                fp8_linears,
                collector,
                &up_name,
                normalized.as_ref().ok_or_else(|| {
                    Error::Other(format!("BF16 up input is unavailable for {prefix}"))
                })?,
                &layer.w_up,
            )?
        };
        Some((gate, up))
    };
    let down_name = format!("{prefix}.down");
    let feed_forward = if let Some(down) = fp8_linears.get(&down_name).filter(|down| down.is_fp8())
    {
        let gated = if use_fused_fp8_gate_up {
            let fused = fp8_linears
                .get(&format!("{prefix}.gate_up_fused"))
                .ok_or_else(|| {
                    Error::Other(format!("fused Qwen gate/up is unavailable for {prefix}"))
                })?;
            let Gr00tQuantizedLinearInput::Fp8(shared_input) =
                shared_gate_up_input.as_ref().ok_or_else(|| {
                    Error::Other(format!(
                        "shared Qwen gate/up input is unavailable for {prefix}"
                    ))
                })?
            else {
                return Err(Error::Other(format!(
                    "fused Qwen gate/up requires an FP8 input for {prefix}"
                )));
            };
            let packed = fused.forward_quantized(shared_input, backend)?;
            kernels::activation::packed_gate_up_quant_bf16_e4m3(
                backend.context(),
                &packed,
                down.activation_scale().ok_or_else(|| {
                    Error::Other(format!("{down_name} is missing its FP8 activation scale"))
                })?,
            )?
        } else {
            let (gate_projection, up) = separate_gate_up.as_ref().ok_or_else(|| {
                Error::Other(format!(
                    "separate Qwen gate/up projections are unavailable for {prefix}"
                ))
            })?;
            kernels::activation::silu_mul_quant_bf16_e4m3(
                backend.context(),
                gate_projection,
                up,
                down.activation_scale().ok_or_else(|| {
                    Error::Other(format!("{down_name} is missing its FP8 activation scale"))
                })?,
            )?
        };
        down.forward_quantized(&gated, backend)?
    } else {
        let (gate_projection, up) = separate_gate_up.as_ref().ok_or_else(|| {
            Error::Other(format!(
                "BF16 Qwen gate/up projections are unavailable for {prefix}"
            ))
        })?;
        if let Some(down) = fp8_linears
            .get(&down_name)
            .filter(|down| down.is_w8a8() && *USE_FUSED_W8A8_SILU_MUL_QUANT)
        {
            down.forward_w8a8_silu_mul(gate_projection, up, backend)?
        } else {
            let gated = if *USE_FUSED_BF16_SILU_MUL {
                kernels::activation::silu_mul_bf16(backend.context(), gate_projection, up)?
            } else {
                let gate = backend.silu(gate_projection)?;
                backend.mul(&gate, up)?
            };
            qwen_matmul(
                backend,
                fp8_linears,
                collector,
                &down_name,
                &gated,
                &layer.w_down,
            )?
        }
    };
    Ok((backend.add(&hidden, &feed_forward)?, used_kv_cache))
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
