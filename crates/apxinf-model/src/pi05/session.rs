//! PI0.5 session: preparation, bounded plan cache, and per-call input binding.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{
    Backend, DType, Device, Error, NormalGenerator, Result, SamplingBackend, Tensor,
};
use half::{bf16, f16};

use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};
use crate::vla::{
    Action, ExecutionMode, ExecutionPolicy, ImageLayout, InferenceSpec, InitialLatent, Observation,
    PreparationStatus, PreparedInference, VisionObservation, VlaRequest, VlaRuntime,
};

use super::backend::{
    kernels, transfers, tuning, DeviceBuffer, ImageLayout as KernelImageLayout, RuntimeBackend,
};
use super::{
    checkpoint_identity, upload_time_embeddings, upload_time_embeddings_bf16,
    upload_time_embeddings_int8, Pi05ActivationScales, Pi05Bf16CapturedGraph, Pi05Bf16CudaRuntime,
    Pi05CapturedGraph, Pi05Config, Pi05CudaRuntime, Pi05Int8CapturedGraph, Pi05Int8CudaRuntime,
    Pi05Weights, StaticBf16Pi05Weights, StaticFp8Calibration, StaticFp8Pi05Weights,
    StaticInt8Pi05Weights,
};

#[derive(Clone)]
enum RuntimeVariant {
    Fp8 {
        runtime: Pi05CudaRuntime,
        time_embeddings: Arc<Vec<Tensor>>,
        vision_scale: f32,
    },
    Bf16 {
        runtime: Pi05Bf16CudaRuntime,
        time_embeddings: Arc<Vec<Tensor>>,
    },
    W8A8 {
        runtime: Pi05Int8CudaRuntime,
        time_embeddings: Arc<Vec<Tensor>>,
    },
}

impl RuntimeVariant {
    fn input_dtype(&self) -> DType {
        match self {
            Self::Fp8 { .. } => DType::F16,
            Self::Bf16 { .. } | Self::W8A8 { .. } => DType::BF16,
        }
    }

    fn captured_patch_dtype(&self, raw_rgb: bool) -> DType {
        match (self, raw_rgb) {
            (Self::Fp8 { .. }, true) => DType::F8E4M3,
            _ => self.input_dtype(),
        }
    }

    fn infer(
        &self,
        patches: &Tensor,
        token_ids: &DeviceBuffer,
        token_count: usize,
        noise: &Tensor,
        prequantized_fp8_patches: bool,
    ) -> Result<Tensor> {
        match self {
            Self::Fp8 {
                runtime,
                time_embeddings,
                ..
            } if prequantized_fp8_patches => {
                runtime.infer_fp8_patches(patches, token_ids, token_count, noise, time_embeddings)
            }
            Self::Fp8 {
                runtime,
                time_embeddings,
                ..
            } => runtime.infer(patches, token_ids, token_count, noise, time_embeddings),
            Self::Bf16 {
                runtime,
                time_embeddings,
            } => runtime.infer(patches, token_ids, token_count, noise, time_embeddings),
            Self::W8A8 {
                runtime,
                time_embeddings,
            } => runtime.infer(patches, token_ids, token_count, noise, time_embeddings),
        }
    }

    fn capture(
        &self,
        spec: &InferenceSpec,
        patches: &Tensor,
        token_ids: &DeviceBuffer,
        noise: &Tensor,
    ) -> Result<GraphVariant> {
        let layout = spec.image_layout.map(kernel_image_layout);
        match (self, layout) {
            (
                Self::Fp8 {
                    runtime,
                    time_embeddings,
                    ..
                },
                None,
            ) => Ok(GraphVariant::Fp8(runtime.capture_infer(
                patches,
                token_ids,
                spec.token_count,
                noise,
                time_embeddings,
            )?)),
            (
                Self::Fp8 {
                    runtime,
                    time_embeddings,
                    ..
                },
                Some(layout),
            ) => Ok(GraphVariant::Fp8(runtime.capture_infer_rgb_u8(
                layout,
                token_ids,
                spec.token_count,
                noise,
                time_embeddings,
            )?)),
            (
                Self::Bf16 {
                    runtime,
                    time_embeddings,
                },
                None,
            ) => Ok(GraphVariant::Bf16(runtime.capture_infer(
                patches,
                token_ids,
                spec.token_count,
                noise,
                time_embeddings,
            )?)),
            (
                Self::Bf16 {
                    runtime,
                    time_embeddings,
                },
                Some(layout),
            ) => Ok(GraphVariant::Bf16(runtime.capture_infer_rgb_u8(
                layout,
                token_ids,
                spec.token_count,
                noise,
                time_embeddings,
            )?)),
            (
                Self::W8A8 {
                    runtime,
                    time_embeddings,
                },
                None,
            ) => Ok(GraphVariant::W8A8(runtime.capture_infer(
                patches,
                token_ids,
                spec.token_count,
                noise,
                time_embeddings,
            )?)),
            (
                Self::W8A8 {
                    runtime,
                    time_embeddings,
                },
                Some(layout),
            ) => Ok(GraphVariant::W8A8(runtime.capture_infer_rgb_u8(
                layout,
                token_ids,
                spec.token_count,
                noise,
                time_embeddings,
            )?)),
        }
    }
}

enum GraphVariant {
    Fp8(Pi05CapturedGraph),
    Bf16(Pi05Bf16CapturedGraph),
    W8A8(Pi05Int8CapturedGraph),
}

impl GraphVariant {
    fn update(
        &self,
        observation: &Observation,
        noise: &Tensor,
        patches: Option<&Tensor>,
    ) -> Result<()> {
        match (&observation.vision, self) {
            (VisionObservation::Patches(_), Self::Fp8(graph)) => graph.update_inputs(
                patches.expect("validated patches"),
                &observation.token_ids,
                noise,
            ),
            (VisionObservation::Patches(_), Self::Bf16(graph)) => graph.update_inputs(
                patches.expect("validated patches"),
                &observation.token_ids,
                noise,
            ),
            (VisionObservation::Patches(_), Self::W8A8(graph)) => graph.update_inputs(
                patches.expect("validated patches"),
                &observation.token_ids,
                noise,
            ),
            (VisionObservation::RgbU8 { bytes, .. }, Self::Fp8(graph)) => {
                graph.update_raw_image_inputs(bytes, &observation.token_ids, noise)
            }
            (VisionObservation::RgbU8 { bytes, .. }, Self::Bf16(graph)) => {
                graph.update_raw_image_inputs(bytes, &observation.token_ids, noise)
            }
            (VisionObservation::RgbU8 { bytes, .. }, Self::W8A8(graph)) => {
                graph.update_raw_image_inputs(bytes, &observation.token_ids, noise)
            }
        }
    }

    fn replay(&self) -> Result<Tensor> {
        match self {
            Self::Fp8(graph) => {
                graph.replay()?;
                Ok(graph.output().clone())
            }
            Self::Bf16(graph) => {
                graph.replay()?;
                Ok(graph.output().clone())
            }
            Self::W8A8(graph) => {
                graph.replay()?;
                Ok(graph.output().clone())
            }
        }
    }

    fn update_without_noise(
        &self,
        observation: &Observation,
        patches: Option<&Tensor>,
    ) -> Result<()> {
        match (&observation.vision, self) {
            (VisionObservation::Patches(_), Self::Fp8(graph)) => graph.update_inputs_without_noise(
                patches.expect("validated patches"),
                &observation.token_ids,
            ),
            (VisionObservation::Patches(_), Self::Bf16(graph)) => graph
                .update_inputs_without_noise(
                    patches.expect("validated patches"),
                    &observation.token_ids,
                ),
            (VisionObservation::Patches(_), Self::W8A8(graph)) => graph
                .update_inputs_without_noise(
                    patches.expect("validated patches"),
                    &observation.token_ids,
                ),
            (VisionObservation::RgbU8 { bytes, .. }, Self::Fp8(graph)) => {
                graph.update_raw_image_inputs_without_noise(bytes, &observation.token_ids)
            }
            (VisionObservation::RgbU8 { bytes, .. }, Self::Bf16(graph)) => {
                graph.update_raw_image_inputs_without_noise(bytes, &observation.token_ids)
            }
            (VisionObservation::RgbU8 { bytes, .. }, Self::W8A8(graph)) => {
                graph.update_raw_image_inputs_without_noise(bytes, &observation.token_ids)
            }
        }
    }
}

struct EagerInputs {
    patches: Tensor,
    raw_images: Option<DeviceBuffer>,
    noise: Tensor,
    token_ids: DeviceBuffer,
}

struct PreparedBuffers {
    patches: Tensor,
    noise: Tensor,
    token_ids: DeviceBuffer,
    normal_generator: Box<dyn NormalGenerator>,
}

enum ExecStrategy {
    Graph(GraphVariant),
    Eager(EagerInputs),
}

/// Owning prepared PI0.5 inference plan. Runs neither capture nor autotune.
/// Graph actions alias reusable device output: copy to host/device storage before
/// the next run if a stable result is needed. Runs are serialized on this session.
/// Each call fully binds input and RNG key; there is no implicit episode counter.
pub struct Pi05PreparedInference {
    spec: InferenceSpec,
    backend: Arc<RuntimeBackend>,
    config: Arc<Pi05Config>,
    runtime: RuntimeVariant,
    strategy: ExecStrategy,
    normal_generator: RefCell<Box<dyn NormalGenerator>>,
    tuning_generation: u64,
    fallback_reason: Option<String>,
}

impl Pi05PreparedInference {
    fn update_eager_inputs(&self, inputs: &EagerInputs, request: &VlaRequest<'_>) -> Result<()> {
        let observation = request.observation;
        self.backend.synchronize()?;
        let patches = normalize_tensor(
            match &observation.vision {
                VisionObservation::Patches(patches) => Some(patches),
                VisionObservation::RgbU8 { bytes, layout } => {
                    let raw = inputs
                        .raw_images
                        .as_ref()
                        .expect("prepared raw image input");
                    if bytes.len() != raw.len() {
                        return Err(Error::Other(format!(
                            "PI0.5 expected {} raw image bytes, got {}",
                            raw.len(),
                            bytes.len()
                        )));
                    }
                    raw.copy_from_host(bytes).map_err(Error::Cuda)?;
                    self.preprocess_rgb(raw, &inputs.patches, *layout)?;
                    None
                }
            },
            self.runtime.input_dtype(),
            patch_shape(&self.config),
            "patches",
        )?;
        if let Some(patches) = patches.as_ref() {
            transfers::copy_cpu_to_cuda(patches, &inputs.patches)?;
        }
        match request.initial_latent {
            InitialLatent::Provided(latent) => {
                let noise = normalize_tensor(
                    Some(latent),
                    self.runtime.input_dtype(),
                    noise_shape(&self.config),
                    "initial latent",
                )?
                .expect("provided latent is present");
                transfers::copy_cpu_to_cuda(&noise, &inputs.noise)?;
            }
            InitialLatent::Generate { rng } => {
                self.normal_generator.borrow_mut().generate(rng)?;
            }
        }
        copy_token_ids(&inputs.token_ids, &observation.token_ids)?;
        Ok(())
    }

    fn run_eager(&self, inputs: &EagerInputs, request: &VlaRequest<'_>) -> Result<Action> {
        let observation = request.observation;
        self.update_eager_inputs(inputs, request)?;
        Ok(Action::new(self.runtime.infer(
            &inputs.patches,
            &inputs.token_ids,
            self.spec.token_count,
            &inputs.noise,
            matches!(&observation.vision, VisionObservation::RgbU8 { .. }),
        )?))
    }

    fn calibrate_eager(
        &self,
        inputs: &EagerInputs,
        request: &VlaRequest<'_>,
    ) -> Result<BTreeMap<String, f32>> {
        self.update_eager_inputs(inputs, request)?;
        match &self.runtime {
            RuntimeVariant::Bf16 {
                runtime,
                time_embeddings,
            } => runtime.calibrate(
                &inputs.patches,
                &inputs.token_ids,
                self.spec.token_count,
                &inputs.noise,
                time_embeddings,
            ),
            _ => Err(Error::Other(
                "PI0.5 activation calibration requires a BF16 model".into(),
            )),
        }
    }

    fn preprocess_rgb(
        &self,
        images: &DeviceBuffer,
        patches: &Tensor,
        layout: ImageLayout,
    ) -> Result<()> {
        let cuda = &*self.backend;
        let layout = kernel_image_layout(layout);
        match &self.runtime {
            RuntimeVariant::Fp8 { vision_scale, .. } => {
                kernels::preprocess::rgb_u8_to_patches_e4m3(
                    cuda.context(),
                    images,
                    patches,
                    self.config.num_views,
                    self.config.image_size,
                    self.config.patch_size,
                    layout,
                    *vision_scale,
                )
            }
            RuntimeVariant::Bf16 { .. } => kernels::preprocess::rgb_u8_to_patches_bf16(
                cuda.context(),
                images,
                patches,
                self.config.num_views,
                self.config.image_size,
                self.config.patch_size,
                layout,
            ),
            RuntimeVariant::W8A8 { .. } => kernels::preprocess::rgb_u8_to_patches_bf16(
                cuda.context(),
                images,
                patches,
                self.config.num_views,
                self.config.image_size,
                self.config.patch_size,
                layout,
            ),
        }
    }
}

impl PreparedInference for Pi05PreparedInference {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }

    fn status(&self) -> PreparationStatus {
        if self.backend.context().tuning().generation() != self.tuning_generation {
            return PreparationStatus::Invalidated;
        }
        PreparationStatus::Ready {
            mode: match self.strategy {
                ExecStrategy::Graph(_) => ExecutionMode::Graph,
                ExecStrategy::Eager(_) => ExecutionMode::Eager,
            },
            fallback_reason: self.fallback_reason.clone(),
        }
    }

    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        tuning::without_autotune(|| self.run_impl(request))
    }
}

impl Pi05PreparedInference {
    fn run_impl(&self, request: &VlaRequest<'_>) -> Result<Action> {
        if self.backend.context().tuning().generation() != self.tuning_generation {
            return Err(Error::Other(
                "prepared PI0.5 plan is stale after a tactic update; prepare it again".into(),
            ));
        }
        let observation = request.observation;
        observation.validate()?;
        if !self.spec.matches(observation) {
            return Err(Error::Other(format!(
                "prepared PI0.5 spec {:?} does not match observation {:?}",
                self.spec,
                observation.inference_spec()
            )));
        }
        let patches = match &observation.vision {
            VisionObservation::Patches(tensor) => normalize_tensor(
                Some(tensor),
                self.runtime.input_dtype(),
                patch_shape(&self.config),
                "patches",
            )?,
            VisionObservation::RgbU8 { bytes, .. } => {
                validate_image_bytes(&self.config, bytes)?;
                None
            }
        };
        match &self.strategy {
            ExecStrategy::Graph(graph) => {
                match request.initial_latent {
                    InitialLatent::Provided(latent) => {
                        let noise = normalize_tensor(
                            Some(latent),
                            self.runtime.input_dtype(),
                            noise_shape(&self.config),
                            "initial latent",
                        )?
                        .expect("provided latent is present");
                        graph.update(observation, &noise, patches.as_ref())?;
                    }
                    InitialLatent::Generate { rng } => {
                        graph.update_without_noise(observation, patches.as_ref())?;
                        self.normal_generator.borrow_mut().generate(rng)?;
                    }
                }
                Ok(Action::new(graph.replay()?))
            }
            ExecStrategy::Eager(inputs) => self.run_eager(inputs, request),
        }
    }
}

/// PI0.5 runtime whose cached prepared plan owns all graph-visible resources.
///
/// A graph workspace can reserve multiple GiB, so the implicit `infer` path
/// retains only the most recently used shape. Callers that need more than one
/// simultaneously prepared shape can own those plans explicitly via `prepare`.
pub struct Pi05Session {
    backend: Arc<RuntimeBackend>,
    config: Arc<Pi05Config>,
    runtime: RuntimeVariant,
    prepared: RefCell<Option<(InferenceSpec, Rc<Pi05PreparedInference>)>>,
}

// Keep policy selection testable without inducing a real GPU capture failure.
fn select_graph<G>(
    policy: ExecutionPolicy,
    capture: impl FnOnce() -> Result<G>,
) -> Result<(Option<G>, Option<String>)> {
    match policy {
        ExecutionPolicy::Eager => Ok((None, None)),
        ExecutionPolicy::RequireGraph => capture().map(|graph| (Some(graph), None)),
        ExecutionPolicy::PreferGraph => match capture() {
            Ok(graph) => Ok((Some(graph), None)),
            Err(error) => Ok((None, Some(error.to_string()))),
        },
    }
}

fn cached_or_build<K, V>(
    cache: &mut Option<(K, Rc<V>)>,
    key: K,
    build: impl FnOnce() -> Result<V>,
) -> Result<Rc<V>>
where
    K: Copy + PartialEq,
{
    if let Some((cached_key, value)) = cache.as_ref() {
        if *cached_key == key {
            return Ok(Rc::clone(value));
        }
    }

    // Drop the previous value before building its replacement. PI0.5 graph
    // workspaces can reserve multiple GiB, so a transient 2x peak can OOM.
    drop(cache.take());
    let value = Rc::new(build()?);
    *cache = Some((key, Rc::clone(&value)));
    Ok(value)
}

impl Pi05Session {
    fn allocate_prepared_buffers(&self, spec: &InferenceSpec) -> Result<PreparedBuffers> {
        spec.validate()?;
        if spec.token_count > self.config.max_token_len {
            return Err(Error::Other(format!(
                "PI0.5 token count {} exceeds maximum {}",
                spec.token_count, self.config.max_token_len
            )));
        }
        let cuda = &*self.backend;
        let dtype = self.runtime.input_dtype();
        let raw_rgb = spec.image_layout.is_some();
        let patches = self.backend.to_device(&Tensor::zeros(
            patch_shape(&self.config),
            self.runtime.captured_patch_dtype(raw_rgb),
        ))?;
        let noise = self
            .backend
            .to_device(&Tensor::zeros(noise_shape(&self.config), dtype))?;
        let normal_generator = self.backend.create_normal_generator(noise.clone())?;
        let token_ids = DeviceBuffer::alloc_zeros(spec.token_count * 4, cuda.device_id())
            .map_err(Error::Cuda)?;
        Ok(PreparedBuffers {
            patches,
            noise,
            token_ids,
            normal_generator,
        })
    }

    fn build_eager(&self, spec: &InferenceSpec) -> Result<Pi05PreparedInference> {
        let PreparedBuffers {
            patches,
            noise,
            token_ids,
            normal_generator,
        } = self.allocate_prepared_buffers(spec)?;
        let cuda = &*self.backend;
        let raw_images = spec
            .image_layout
            .is_some()
            .then(|| DeviceBuffer::alloc_zeros(image_bytes(&self.config), cuda.device_id()))
            .transpose()
            .map_err(Error::Cuda)?;
        Ok(Pi05PreparedInference {
            spec: *spec,
            backend: Arc::clone(&self.backend),
            config: Arc::clone(&self.config),
            runtime: self.runtime.clone(),
            strategy: ExecStrategy::Eager(EagerInputs {
                patches,
                raw_images,
                noise,
                token_ids,
            }),
            normal_generator: RefCell::new(normal_generator),
            tuning_generation: cuda.context().tuning().generation(),
            fallback_reason: None,
        })
    }

    fn build_prepared(
        &self,
        spec: &InferenceSpec,
        policy: ExecutionPolicy,
    ) -> Result<Pi05PreparedInference> {
        if policy == ExecutionPolicy::Eager {
            return self.build_eager(spec);
        }
        let PreparedBuffers {
            patches,
            noise,
            token_ids,
            normal_generator,
        } = self.allocate_prepared_buffers(spec)?;
        let cuda = &*self.backend;
        let raw_rgb = spec.image_layout.is_some();

        let (graph, fallback_reason) = select_graph(policy, || {
            tuning::without_autotune(|| self.runtime.capture(spec, &patches, &token_ids, &noise))
        })?;
        if let Some(reason) = &fallback_reason {
            eprintln!("[apxinf] PI0.5 graph capture unavailable, using eager: {reason}");
        }
        let strategy = match graph {
            Some(graph) => ExecStrategy::Graph(graph),
            None => {
                let raw_images = if raw_rgb {
                    Some(
                        DeviceBuffer::alloc_zeros(image_bytes(&self.config), cuda.device_id())
                            .map_err(Error::Cuda)?,
                    )
                } else {
                    None
                };
                ExecStrategy::Eager(EagerInputs {
                    patches,
                    raw_images,
                    noise,
                    token_ids,
                })
            }
        };
        Ok(Pi05PreparedInference {
            spec: *spec,
            backend: Arc::clone(&self.backend),
            config: Arc::clone(&self.config),
            runtime: self.runtime.clone(),
            strategy,
            fallback_reason,
            normal_generator: RefCell::new(normal_generator),
            tuning_generation: cuda.context().tuning().generation(),
        })
    }

    fn tune_sample(&self, request: &VlaRequest<'_>) -> Result<()> {
        if self.backend.context().tuning().mode() == tuning::TuningMode::AutoTune {
            let eager = self.build_eager(&request.observation.inference_spec())?;
            let output = eager.run_impl(request)?;
            self.backend.synchronize()?;
            drop(output);
        }
        Ok(())
    }
}

impl VlaRuntime for Pi05Session {
    fn contract(&self) -> crate::VlaContract {
        crate::VlaContract {
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
        let observation = request.observation;
        observation.validate()?;
        let spec = observation.inference_spec();
        let generation = self.backend.context().tuning().generation();
        let needs_prepare =
            self.prepared
                .borrow()
                .as_ref()
                .map_or(true, |(cached_spec, prepared)| {
                    *cached_spec != spec || prepared.tuning_generation != generation
                });
        if needs_prepare {
            // Release the old implicit plan before temporary tuning allocations.
            // Explicitly retained plans are caller-owned and remain alive.
            self.clear_prepared()?;
            self.tune_sample(request)?;
        }
        let prepared = {
            let mut cache = self.prepared.borrow_mut();
            if cache.as_ref().is_some_and(|(cached_spec, prepared)| {
                *cached_spec != spec
                    || prepared.tuning_generation != self.backend.context().tuning().generation()
            }) {
                drop(cache.take());
            }
            cached_or_build(&mut cache, spec, || {
                self.build_prepared(&spec, ExecutionPolicy::PreferGraph)
            })?
        };
        prepared.run(request)
    }

    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        self.prepare_with_policy(spec, ExecutionPolicy::PreferGraph)
    }

    fn prepare_with_policy(
        &self,
        spec: &InferenceSpec,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        Ok(Box::new(self.build_prepared(spec, policy)?))
    }

    fn prepare_for(
        &self,
        sample: &VlaRequest<'_>,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        sample.observation.validate()?;
        self.tune_sample(sample)?;
        self.prepare_with_policy(&sample.observation.inference_spec(), policy)
    }

    fn execution_mode(&self) -> &'static str {
        match self
            .prepared
            .borrow()
            .as_ref()
            .map(|(_, plan)| plan.status())
        {
            None => "unprepared",
            Some(PreparationStatus::Ready {
                mode: ExecutionMode::Graph,
                ..
            }) => "graph",
            Some(PreparationStatus::Ready {
                mode: ExecutionMode::Eager,
                ..
            }) => "eager",
            Some(PreparationStatus::Invalidated) => "invalidated",
            Some(PreparationStatus::RuntimeManaged) => "runtime-managed",
        }
    }

    fn clear_prepared(&self) -> Result<()> {
        self.backend.synchronize()?;
        drop(self.prepared.borrow_mut().take());
        Ok(())
    }

    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        let action = self.infer(request)?;
        self.backend.to_cpu(action.tensor())?.to_f32_vec()
    }

    fn calibration_amax(&self, request: &VlaRequest<'_>) -> Result<BTreeMap<String, f32>> {
        request.observation.validate()?;
        let prepared = self.build_eager(&request.observation.inference_spec())?;
        let ExecStrategy::Eager(inputs) = &prepared.strategy else {
            unreachable!("calibration plan is always eager")
        };
        prepared.calibrate_eager(inputs, request)
    }

    fn calibration_plan(&self) -> Result<Vec<String>> {
        Ok(super::Pi05CalibrationPlan::for_config(&self.config)
            .sites()
            .to_vec())
    }
}

pub(super) fn load_registered(
    path: &Path,
    _device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    let backend = crate::accelerator::cuda::downcast_arc(backend)
        .ok_or_else(|| Error::Other("PI0.5 is only registered for CUDA".into()))?;
    let cuda = &*backend;
    let root = artifact_root(path);
    let config_path = root.join("config.json");
    let config = Arc::new(if let Some(cfg) = options.config.clone() {
        cfg
    } else if config_path.is_file() {
        Pi05Config::from_json_file(&config_path)?
    } else {
        Pi05Config::default()
    });
    let synthetic = options.synthetic;
    let host_weights = match synthetic {
        Some(synthetic) => Pi05Weights::synthetic(&config, synthetic.seed)?,
        None => Pi05Weights::from_safetensors(&config, path)?,
    };
    // Synthetic (checkpoint-free) loads must not pick up stray calibration/tuning
    // files from the working directory; only honor explicitly passed paths.
    let calibration_path = options.calibration_path.clone().or_else(|| {
        (synthetic.is_none())
            .then(|| existing(root.join("calibration.json")))
            .flatten()
    });
    let precision = resolve_precision(
        options.precision,
        cuda.context().caps().sm,
        calibration_path.is_some(),
    );

    let runtime = match precision {
        ModelPrecision::Fp8 => {
            let scales = if let Some(scale) = options.uniform_fp8_scale {
                Arc::new(Pi05ActivationScales::uniform(&config, scale)?)
            } else {
                let calibration_path = calibration_path.ok_or_else(|| {
                    Error::Other(
                        "FP8 PI0.5 requires LoadOptions.calibration_path or calibration.json"
                            .into(),
                    )
                })?;
                let checkpoint = checkpoint_identity(path)?;
                let calibration =
                    StaticFp8Calibration::from_json_file(&calibration_path, &config, &checkpoint)?;
                Arc::new(Pi05ActivationScales::from_calibration(
                    &config,
                    &calibration,
                )?)
            };
            let weights = Arc::new(StaticFp8Pi05Weights::from_host(
                &host_weights,
                &*backend,
                config.language_dual_geglu_shape_possible(),
            )?);
            let time_embeddings = Arc::new(upload_time_embeddings(&config, &*backend)?);
            let vision_scale = scales.vision_patch_input;
            RuntimeVariant::Fp8 {
                runtime: Pi05CudaRuntime::new(
                    Arc::clone(&backend),
                    Arc::clone(&config),
                    weights,
                    scales,
                )?,
                time_embeddings,
                vision_scale,
            }
        }
        ModelPrecision::Bf16 => {
            let weights = Arc::new(StaticBf16Pi05Weights::from_host(
                &host_weights,
                &*backend,
                config.language_dual_geglu_shape_possible(),
            )?);
            let time_embeddings = Arc::new(upload_time_embeddings_bf16(&config, &*backend)?);
            RuntimeVariant::Bf16 {
                runtime: Pi05Bf16CudaRuntime::new(
                    Arc::clone(&backend),
                    Arc::clone(&config),
                    weights,
                )?,
                time_embeddings,
            }
        }
        ModelPrecision::W8A8 => {
            let weights = Arc::new(StaticInt8Pi05Weights::from_host(&host_weights, cuda)?);
            let time_embeddings = Arc::new(upload_time_embeddings_int8(&config, &*backend)?);
            RuntimeVariant::W8A8 {
                runtime: Pi05Int8CudaRuntime::new(
                    Arc::clone(&backend),
                    Arc::clone(&config),
                    weights,
                )?,
                time_embeddings,
            }
        }
        ModelPrecision::Auto => unreachable!("automatic precision was resolved"),
    };

    Ok(LoadedModel::Vla(Box::new(Pi05Session {
        backend,
        config,
        runtime,
        prepared: RefCell::new(None),
    })))
}

fn resolve_precision(
    requested: ModelPrecision,
    sm: u32,
    has_fp8_calibration: bool,
) -> ModelPrecision {
    match requested {
        ModelPrecision::Auto if sm >= 100 && has_fp8_calibration => ModelPrecision::Fp8,
        ModelPrecision::Auto if (80..100).contains(&sm) => ModelPrecision::W8A8,
        ModelPrecision::Auto => ModelPrecision::Bf16,
        explicit => explicit,
    }
}

fn artifact_root(path: &Path) -> &Path {
    if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    }
}

fn existing(path: PathBuf) -> Option<PathBuf> {
    path.is_file().then_some(path)
}

fn kernel_image_layout(layout: ImageLayout) -> KernelImageLayout {
    match layout {
        ImageLayout::Nhwc => KernelImageLayout::Nhwc,
        ImageLayout::Nchw => KernelImageLayout::Nchw,
    }
}

fn patch_shape(config: &Pi05Config) -> Vec<usize> {
    vec![
        config.num_views * config.patches_per_view(),
        3 * config.patch_size * config.patch_size,
    ]
}

fn noise_shape(config: &Pi05Config) -> Vec<usize> {
    vec![config.action_horizon, config.action_dim]
}

fn image_bytes(config: &Pi05Config) -> usize {
    config.num_views * 3 * config.image_size * config.image_size
}

fn validate_image_bytes(config: &Pi05Config, bytes: &[u8]) -> Result<()> {
    let expected = image_bytes(config);
    if bytes.len() != expected {
        return Err(Error::Other(format!(
            "PI0.5 expected {expected} raw image bytes, got {}",
            bytes.len()
        )));
    }
    Ok(())
}

fn copy_token_ids(buffer: &DeviceBuffer, token_ids: &[u32]) -> Result<()> {
    let bytes = token_ids
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    buffer.copy_from_host(&bytes).map_err(Error::Cuda)
}

fn normalize_tensor(
    tensor: Option<&Tensor>,
    dtype: DType,
    shape: Vec<usize>,
    label: &str,
) -> Result<Option<Tensor>> {
    let Some(tensor) = tensor else {
        return Ok(None);
    };
    if tensor.device() != Device::Cpu {
        return Err(Error::Other(format!("PI0.5 {label} must be a CPU tensor")));
    }
    if tensor.shape().dims() != shape {
        return Err(Error::Other(format!(
            "PI0.5 {label} shape {:?} does not match {:?}",
            tensor.shape().dims(),
            shape
        )));
    }
    if tensor.dtype() == dtype {
        return Ok(Some(tensor.clone()));
    }
    let values = tensor.to_f32_vec()?;
    let converted = match dtype {
        DType::F16 => Tensor::from_f16(
            shape,
            &values.into_iter().map(f16::from_f32).collect::<Vec<_>>(),
        )?,
        DType::BF16 => Tensor::from_bf16(
            shape,
            &values.into_iter().map(bf16::from_f32).collect::<Vec<_>>(),
        )?,
        _ => {
            return Err(Error::Other(format!(
                "PI0.5 cannot normalize {label} to {dtype}"
            )))
        }
    };
    Ok(Some(converted))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_policy_does_not_hide_capture_failure() {
        let failure = || Err::<(), _>(Error::Other("capture fixture failure".into()));
        assert!(select_graph(ExecutionPolicy::RequireGraph, failure).is_err());
        let (graph, reason) = select_graph(ExecutionPolicy::PreferGraph, failure).unwrap();
        assert!(graph.is_none());
        assert!(reason.unwrap().contains("capture fixture failure"));
        let eager = select_graph::<()>(ExecutionPolicy::Eager, || panic!("eager must not capture"));
        assert_eq!(eager.unwrap(), (None, None));
        assert_eq!(
            select_graph(ExecutionPolicy::RequireGraph, || Ok(7)).unwrap(),
            (Some(7), None)
        );
    }

    #[test]
    fn most_recent_cache_reuses_and_drops_before_replacement() {
        struct DropSpy(Rc<std::cell::Cell<usize>>);

        impl Drop for DropSpy {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let drops = Rc::new(std::cell::Cell::new(0));
        let mut cache = None;
        let first = cached_or_build(&mut cache, 1u8, || Ok(DropSpy(Rc::clone(&drops)))).unwrap();
        let reused =
            cached_or_build(&mut cache, 1u8, || panic!("must reuse cached value")).unwrap();
        assert!(Rc::ptr_eq(&first, &reused));

        drop(first);
        drop(reused);
        let replacement = cached_or_build(&mut cache, 2u8, || {
            assert_eq!(
                drops.get(),
                1,
                "old value must drop before replacement build"
            );
            Ok(DropSpy(Rc::clone(&drops)))
        })
        .unwrap();
        assert_eq!(cache.as_ref().map(|(key, _)| *key), Some(2));

        drop(replacement);
        drop(cache);
        assert_eq!(drops.get(), 2);
    }

    #[test]
    fn auto_precision_matches_thor_and_orin_policy() {
        assert_eq!(
            resolve_precision(ModelPrecision::Auto, 110, true),
            ModelPrecision::Fp8
        );
        assert_eq!(
            resolve_precision(ModelPrecision::Auto, 110, false),
            ModelPrecision::Bf16
        );
        assert_eq!(
            resolve_precision(ModelPrecision::Auto, 87, true),
            ModelPrecision::W8A8
        );
    }
}
