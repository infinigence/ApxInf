use std::cell::RefCell;
use std::collections::BTreeMap;
use std::env;
use std::sync::Arc;
use std::time::Instant;

use apxinf_core::{Backend, Error, Graph, NormalGenerator, Result, SamplingBackend, Tensor};
use half::{bf16, f16};

use crate::accelerator::cuda::{kernels, transfers, DeviceBuffer};
use crate::vla::{
    Action, ExecutionMode, ExecutionPolicy, ImageLayout, InferenceSpec, InitialLatent,
    Observation, PreparationStatus, PreparedInference, VisionObservation, VlaContract,
    VlaRequest, VlaRuntime,
};

use super::{SmolVlaConfig, SmolVlaModel};

const GRAPH_WORKSPACE_CAPACITIES_BYTES: [usize; 2] = [64 << 20, 96 << 20];

pub struct SmolVlaModelRunner {
    model: Arc<SmolVlaModel>,
    config: Arc<SmolVlaConfig>,
    prepared: RefCell<Option<SmolVlaPreparedPlan>>,
}

impl SmolVlaModelRunner {
    pub fn new(model: Arc<SmolVlaModel>, config: Arc<SmolVlaConfig>) -> Self {
        Self {
            model,
            config,
            prepared: RefCell::new(None),
        }
    }

    fn padded_noise_host(&self, latent: &Tensor) -> Result<Tensor> {
        let shape = latent.shape().dims();
        if latent.device() != apxinf_core::Device::Cpu
            || shape.len() != 2
            || shape[0] != self.config.action_horizon
            || shape[1] > self.config.max_action_dim
        {
            return Err(Error::Other(format!(
                "SmolVLA provided latent must be CPU [{}, 0..={}]",
                self.config.action_horizon, self.config.max_action_dim
            )));
        }
        let source_width = shape[1];
        let values = latent.to_f32_vec()?;
        let mut padded = vec![0.0; self.config.action_horizon * self.config.max_action_dim];
        for (row, source) in values.chunks_exact(source_width).enumerate() {
            let target = &mut padded
                [row * self.config.max_action_dim..(row + 1) * self.config.max_action_dim];
            target[..source_width].copy_from_slice(source);
        }
        if self.model.uses_fp16_gemm() {
            Tensor::from_f16(
                vec![self.config.action_horizon, self.config.max_action_dim],
                &padded.into_iter().map(f16::from_f32).collect::<Vec<_>>(),
            )
        } else {
            Tensor::from_bf16(
                vec![self.config.action_horizon, self.config.max_action_dim],
                &padded.into_iter().map(bf16::from_f32).collect::<Vec<_>>(),
            )
        }
    }

    fn noise(&self, latent: InitialLatent<'_>) -> Result<Tensor> {
        match latent {
            InitialLatent::Provided(tensor) => {
                let host = self.padded_noise_host(tensor)?;
                self.model.backend().to_device(&host)
            }
            InitialLatent::Generate { rng } => {
                let host = Tensor::zeros(
                    vec![self.config.action_horizon, self.config.max_action_dim],
                    self.model.dtype(),
                );
                let output = self.model.backend().to_device(&host)?;
                let mut generator = self.model.backend().create_normal_generator(output)?;
                generator.generate(rng)?;
                Ok(generator.output().clone())
            }
        }
    }

    fn infer_profiled_eager(
        &self,
        request: &VlaRequest<'_>,
    ) -> Result<(Action, BTreeMap<String, f64>)> {
        let observation = request.observation;
        observation.validate()?;
        let token_bytes = observation
            .token_ids
            .iter()
            .flat_map(|token| u32::to_ne_bytes(*token))
            .collect::<Vec<_>>();
        let token_ids = DeviceBuffer::alloc(
            token_bytes.len(),
            self.model.backend().context().device_id(),
        )
        .map_err(Error::Cuda)?;
        token_ids
            .copy_from_host(&token_bytes)
            .map_err(Error::Cuda)?;
        let noise = self.noise(request.initial_latent)?;
        let (tensor, timing) = self
            .model
            .infer_with_timing(observation, &noise, &token_ids)?;
        Ok((Action::new(tensor), timing.as_map()))
    }

    fn build_prepared_plan(
        &self,
        spec: &InferenceSpec,
        policy: ExecutionPolicy,
    ) -> Result<SmolVlaPreparedPlan> {
        spec.validate()?;
        if spec.token_count == 0 || spec.token_count > self.config.max_token_len {
            return Err(Error::Other(format!(
                "SmolVLA token count must be in 1..={}, got {}",
                spec.token_count, self.config.max_token_len
            )));
        }
        match policy {
            ExecutionPolicy::Eager => Ok(SmolVlaPreparedPlan {
                spec: *spec,
                strategy: PreparedStrategy::Eager,
                model: Arc::clone(&self.model),
                config: Arc::clone(&self.config),
                fallback_reason: None,
            }),
            ExecutionPolicy::RequireGraph => {
                let graph = SmolVlaCapturedGraph::new(Arc::clone(&self.model), spec)?;
                Ok(SmolVlaPreparedPlan {
                    spec: *spec,
                    strategy: PreparedStrategy::Graph(graph),
                    model: Arc::clone(&self.model),
                    config: Arc::clone(&self.config),
                    fallback_reason: None,
                })
            }
            ExecutionPolicy::PreferGraph => {
                match SmolVlaCapturedGraph::new(Arc::clone(&self.model), spec) {
                    Ok(graph) => Ok(SmolVlaPreparedPlan {
                        spec: *spec,
                        strategy: PreparedStrategy::Graph(graph),
                        model: Arc::clone(&self.model),
                        config: Arc::clone(&self.config),
                        fallback_reason: None,
                    }),
                    Err(error) => {
                        let reason = error.to_string();
                        eprintln!(
                            "[apxinf] SmolVLA graph capture unavailable, using eager: {reason}"
                        );
                        Ok(SmolVlaPreparedPlan {
                            spec: *spec,
                            strategy: PreparedStrategy::Eager,
                            model: Arc::clone(&self.model),
                            config: Arc::clone(&self.config),
                            fallback_reason: Some(reason),
                        })
                    }
                }
            }
        }
    }

    fn ensure_prepared_plan(&self, request: &VlaRequest<'_>) -> Result<()> {
        let observation = request.observation;
        observation.validate()?;
        let spec = observation.inference_spec();
        let needs_prepare = self
            .prepared
            .borrow()
            .as_ref()
            .is_none_or(|plan| plan.spec != spec);
        if needs_prepare {
            let policy = if env::var_os("APXINF_SMOLVLA_NO_GRAPH").is_some() {
                ExecutionPolicy::Eager
            } else {
                ExecutionPolicy::PreferGraph
            };
            let plan = self.build_prepared_plan(&spec, policy)?;
            *self.prepared.borrow_mut() = Some(plan);
        }
        Ok(())
    }

    fn infer_profiled(
        &self,
        request: &VlaRequest<'_>,
    ) -> Result<(Action, BTreeMap<String, f64>)> {
        self.ensure_prepared_plan(request)?;
        let uses_graph = self
            .prepared
            .borrow()
            .as_ref()
            .is_some_and(|plan| matches!(plan.strategy, PreparedStrategy::Graph(_)));
        if uses_graph {
            let prepared = self.prepared.borrow();
            let plan = prepared
                .as_ref()
                .expect("prepared plan was just checked");
            plan.run_profiled(request)
        } else {
            self.infer_profiled_eager(request)
        }
    }
}

struct SmolVlaGraphInputs {
    raw_images: Option<DeviceBuffer>,
    patches: Tensor,
    state: Tensor,
    noise: Tensor,
    token_ids: DeviceBuffer,
}

struct SmolVlaCapturedGraph {
    graph: Box<dyn Graph>,
    output: Tensor,
    inputs: SmolVlaGraphInputs,
    token_count: usize,
    layout: ImageLayout,
    _time_embeddings: Vec<Tensor>,
    model: Arc<SmolVlaModel>,
    normal_generator: RefCell<Box<dyn NormalGenerator>>,
    _workspace: kernels::GraphWorkspace,
}

impl SmolVlaCapturedGraph {
    fn new(model: Arc<SmolVlaModel>, spec: &InferenceSpec) -> Result<Self> {
        let backend = model.backend();
        let device_id = backend.context().device_id();
        let model_config = model.config();
        let patch_rows = model_config
            .num_views
            .checked_mul(model_config.patches_per_view())
            .ok_or_else(|| Error::Other("SmolVLA patch rows overflow".into()))?;
        let patch_width = 3 * model_config.patch_size * model_config.patch_size;
        let image_bytes = model_config
            .num_views
            .checked_mul(3)
            .and_then(|bytes| bytes.checked_mul(model_config.image_size))
            .and_then(|bytes| bytes.checked_mul(model_config.image_size))
            .ok_or_else(|| Error::Other("SmolVLA image byte count overflow".into()))?;
        let token_bytes = spec
            .token_count
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| Error::Other("SmolVLA token byte count overflow".into()))?;

        let raw_images = match spec.image_layout {
            Some(_) => {
                Some(DeviceBuffer::alloc_zeros(image_bytes, device_id).map_err(Error::Cuda)?)
            }
            None => None,
        };
        let patches = backend.to_device(&Tensor::zeros(
            vec![patch_rows, patch_width],
            model.dtype(),
        ))?;
        let state = backend.to_device(&Tensor::zeros(
            vec![1, model_config.max_state_dim],
            model.dtype(),
        ))?;
        let noise = backend.to_device(&Tensor::zeros(
            vec![
                model_config.action_horizon,
                model_config.max_action_dim,
            ],
            model.dtype(),
        ))?;
        let token_ids = DeviceBuffer::alloc_zeros(token_bytes, device_id).map_err(Error::Cuda)?;
        let normal_generator = backend.create_normal_generator(noise.clone())?;
        let time_embeddings = model.time_embeddings()?;
        let layout = spec.image_layout.unwrap_or(ImageLayout::Nhwc);

        let mut last_workspace_error = None;
        for capacity in GRAPH_WORKSPACE_CAPACITIES_BYTES {
            let workspace =
                kernels::GraphWorkspace::new(capacity, device_id)?;
            let preflight = kernels::prepare_with_workspace(&workspace, || {
                model.infer_fixed(
                    raw_images.as_ref(),
                    &patches,
                    &state,
                    &noise,
                    &token_ids,
                    spec.token_count,
                    layout,
                    &time_embeddings,
                )
            });
            let expected_output = match preflight {
                Ok(output) => output,
                Err(error) => {
                    let message = error.to_string();
                    if !message.contains("workspace exhausted") {
                        return Err(Error::Other(format!(
                            "SmolVLA graph preflight failed: {message}"
                        )));
                    }
                    last_workspace_error = Some(message);
                    continue;
                }
            };
            backend.synchronize()?;
            let expected_host = backend.to_cpu(&expected_output)?;
            drop(expected_output);

            let (graph, output) = backend.capture_graph(|| {
                kernels::with_workspace(&workspace, || {
                    model.infer_fixed(
                        raw_images.as_ref(),
                        &patches,
                        &state,
                        &noise,
                        &token_ids,
                        spec.token_count,
                        layout,
                        &time_embeddings,
                    )
                })
            })?;
            graph.replay()?;
            backend.synchronize()?;
            let replayed_output = backend.to_cpu(&output)?;
            if replayed_output.storage().as_cpu() != expected_host.storage().as_cpu() {
                return Err(Error::Other(
                    "SmolVLA CUDA graph replay does not match eager preflight".into(),
                ));
            }
            eprintln!(
                "[apxinf] SmolVLA CUDA Graph captured with {}/{} workspace bytes",
                workspace.peak_used(),
                workspace.capacity()
            );

            return Ok(Self {
                graph,
                output,
                inputs: SmolVlaGraphInputs {
                    raw_images,
                    patches,
                    state,
                    noise,
                    token_ids,
                },
                token_count: spec.token_count,
                layout,
                _time_embeddings: time_embeddings,
                model,
                normal_generator: RefCell::new(normal_generator),
                _workspace: workspace,
            });
        }
        Err(Error::Other(format!(
            "SmolVLA graph workspace exhausted at {} bytes: {}",
            GRAPH_WORKSPACE_CAPACITIES_BYTES
                .last()
                .copied()
                .unwrap_or_default(),
            last_workspace_error.unwrap_or_else(|| "unknown error".into())
        )))
    }

    fn update(
        &self,
        observation: &Observation,
        provided_noise: Option<&Tensor>,
    ) -> Result<()> {
        if observation.token_ids.len() != self.token_count {
            return Err(Error::Other(format!(
                "SmolVLA graph expects {} token IDs, got {}",
                self.token_count,
                observation.token_ids.len()
            )));
        }
        self.model.backend().synchronize()?;
        match &observation.vision {
            VisionObservation::Patches(patches) => {
                if self.inputs.raw_images.is_some() {
                    return Err(Error::Other(
                        "SmolVLA graph uses raw RGB input; patches cannot replace it".into(),
                    ));
                }
                transfers::copy_cpu_to_cuda(patches, &self.inputs.patches)?;
            }
            VisionObservation::RgbU8 { bytes, layout } => {
                let raw_images = self.inputs.raw_images.as_ref().ok_or_else(|| {
                    Error::Other("SmolVLA graph uses patch input; RGB cannot replace it".into())
                })?;
                if bytes.len() != raw_images.len() {
                    return Err(Error::Other(format!(
                        "SmolVLA graph expects {} image bytes, got {}",
                        raw_images.len(),
                        bytes.len()
                    )));
                }
                if *layout != self.layout {
                    return Err(Error::Other(
                        "SmolVLA graph image layout does not match prepared input".into(),
                    ));
                }
                raw_images.copy_from_host(bytes).map_err(Error::Cuda)?;
            }
        }
        let token_bytes = observation
            .token_ids
            .iter()
            .flat_map(|token| u32::to_ne_bytes(*token))
            .collect::<Vec<_>>();
        self.inputs
            .token_ids
            .copy_from_host(&token_bytes)
            .map_err(Error::Cuda)?;
        let state = self.model.prepare_state(observation)?;
        transfers::copy_cpu_to_cuda(&state, &self.inputs.state)?;
        if let Some(noise) = provided_noise {
            let host = {
                let runner = SmolVlaModelRunner::new(
                    Arc::clone(&self.model),
                    Arc::clone(self.model.config()),
                );
                runner.padded_noise_host(noise)?
            };
            transfers::copy_cpu_to_cuda(&host, &self.inputs.noise)?;
        }
        Ok(())
    }

    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        let provided_noise = match request.initial_latent {
            InitialLatent::Provided(noise) => Some(noise),
            InitialLatent::Generate { rng } => {
                self.update(request.observation, None)?;
                self.normal_generator.borrow_mut().generate(rng)?;
                None
            }
        };
        if let Some(noise) = provided_noise {
            self.update(request.observation, Some(noise))?;
        }
        self.graph.replay()?;
        Ok(Action::new(self.output.clone()))
    }

    fn run_profiled(&self, request: &VlaRequest<'_>) -> Result<(Action, BTreeMap<String, f64>)> {
        let started = Instant::now();
        let action = self.run(request)?;
        self.model.backend().synchronize()?;
        let graph_ms = started.elapsed().as_secs_f64() * 1000.0;
        Ok((action, BTreeMap::from([("graph_ms".to_string(), graph_ms)])))
    }
}

enum PreparedStrategy {
    Graph(SmolVlaCapturedGraph),
    Eager,
}

struct SmolVlaPreparedPlan {
    spec: InferenceSpec,
    strategy: PreparedStrategy,
    model: Arc<SmolVlaModel>,
    config: Arc<SmolVlaConfig>,
    fallback_reason: Option<String>,
}

impl SmolVlaPreparedPlan {
    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        if !self.spec.matches(request.observation) {
            return Err(Error::Other(
                "SmolVLA request does not match prepared inference spec".into(),
            ));
        }
        match &self.strategy {
            PreparedStrategy::Graph(graph) => graph.run(request),
            PreparedStrategy::Eager => {
                let runner = SmolVlaModelRunner {
                    model: Arc::clone(&self.model),
                    config: Arc::clone(&self.config),
                    prepared: RefCell::new(None),
                };
                runner.infer_profiled_eager(request).map(|(action, _)| action)
            }
        }
    }

    fn run_profiled(&self, request: &VlaRequest<'_>) -> Result<(Action, BTreeMap<String, f64>)> {
        if !self.spec.matches(request.observation) {
            return Err(Error::Other(
                "SmolVLA request does not match prepared inference spec".into(),
            ));
        }
        match &self.strategy {
            PreparedStrategy::Graph(graph) => graph.run_profiled(request),
            PreparedStrategy::Eager => {
                let runner = SmolVlaModelRunner {
                    model: Arc::clone(&self.model),
                    config: Arc::clone(&self.config),
                    prepared: RefCell::new(None),
                };
                runner.infer_profiled_eager(request)
            }
        }
    }
}

impl VlaRuntime for SmolVlaModelRunner {
    fn model_variant(&self) -> Option<&'static str> {
        Some(if self.model.uses_fp16_gemm() {
            "fp16"
        } else {
            "bf16"
        })
    }

    fn contract(&self) -> VlaContract {
        VlaContract {
            action_shape: [self.config.action_horizon, self.config.action_dim],
            patch_shape: [
                self.config.num_views * self.config.patches_per_view(),
                3 * self.config.patch_size * self.config.patch_size,
            ],
            max_token_len: self.config.max_token_len,
            num_views: self.config.num_views,
            image_size: self.config.image_size,
            patch_size: self.config.patch_size,
            accepts_rgb_u8: true,
        }
    }

    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        self.infer_profiled(request).map(|(action, _)| action)
    }

    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        self.prepare_with_policy(spec, ExecutionPolicy::PreferGraph)
    }

    fn prepare_with_policy(
        &self,
        spec: &InferenceSpec,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        let plan = self.build_prepared_plan(spec, policy)?;
        Ok(Box::new(SmolVlaPreparedInference { plan }))
    }

    fn prepare_for(
        &self,
        sample: &VlaRequest<'_>,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        sample.observation.validate()?;
        self.prepare_with_policy(&sample.observation.inference_spec(), policy)
    }

    fn clear_prepared(&self) -> Result<()> {
        self.model.backend().synchronize()?;
        *self.prepared.borrow_mut() = None;
        Ok(())
    }

    fn execution_mode(&self) -> &'static str {
        if self
            .prepared
            .borrow()
            .as_ref()
            .is_some_and(|plan| matches!(plan.strategy, PreparedStrategy::Graph(_)))
        {
            "graph"
        } else {
            "eager"
        }
    }

    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        let action = self.infer(request)?;
        self.model
            .backend()
            .to_cpu(action.tensor())?
            .to_f32_vec()
    }

    fn infer_host_f32_profiled(
        &self,
        request: &VlaRequest<'_>,
    ) -> Result<(Vec<f32>, BTreeMap<String, f64>)> {
        let (action, profile) = self.infer_profiled(request)?;
        let values = self
            .model
            .backend()
            .to_cpu(action.tensor())?
            .to_f32_vec()?;
        Ok((values, profile))
    }
}

struct SmolVlaPreparedInference {
    plan: SmolVlaPreparedPlan,
}

impl PreparedInference for SmolVlaPreparedInference {
    fn spec(&self) -> &InferenceSpec {
        &self.plan.spec
    }

    fn status(&self) -> PreparationStatus {
        PreparationStatus::Ready {
            mode: match self.plan.strategy {
                PreparedStrategy::Graph(_) => ExecutionMode::Graph,
                PreparedStrategy::Eager => ExecutionMode::Eager,
            },
            fallback_reason: self.plan.fallback_reason.clone(),
        }
    }

    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        self.plan.run(request)
    }
}
