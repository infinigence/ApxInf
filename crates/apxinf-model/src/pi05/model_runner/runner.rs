//! PI0.5 runner: preparation, bounded plan cache, and per-call input binding.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{DType, Device, Error, NormalGenerator, Result, Tensor};
use half::{bf16, f16};

use crate::vla::{
    Action, ExecutionMode, ExecutionPolicy, ImageLayout, InferenceSpec, InitialLatent, Observation,
    PreparationStatus, PreparedInference, VisionObservation, VlaRequest, VlaRuntime,
};

use super::prepare::CapturedGraph;
use crate::pi05::backend::{
    self, ops, transfers, Context, DeviceBuffer, ExecutionSession,
    ImageLayout as KernelImageLayout,
};
use crate::pi05::model::ModelVariant;
use crate::pi05::Pi05Config;

impl CapturedGraph {
    fn update(
        &self,
        observation: &Observation,
        noise: &Tensor,
        patches: Option<&Tensor>,
    ) -> Result<()> {
        match &observation.vision {
            VisionObservation::Patches(_) => self.update_inputs(
                patches.expect("validated patches"),
                &observation.token_ids,
                noise,
            ),
            VisionObservation::RgbU8 { bytes, .. } => {
                self.update_raw_image_inputs(bytes, &observation.token_ids, noise)
            }
        }
    }
    fn replay_output(&self) -> Result<Tensor> {
        self.replay()?;
        Ok(self.output().clone())
    }
    fn update_without_noise(
        &self,
        observation: &Observation,
        patches: Option<&Tensor>,
    ) -> Result<()> {
        match &observation.vision {
            VisionObservation::Patches(_) => self.update_inputs_without_noise(
                patches.expect("validated patches"),
                &observation.token_ids,
            ),
            VisionObservation::RgbU8 { bytes, .. } => {
                self.update_raw_image_inputs_without_noise(bytes, &observation.token_ids)
            }
        }
    }
}

struct EagerInputs {
    patches: Tensor,
    raw_images: Option<DeviceBuffer>,
    noise: Tensor,
    token_ids: DeviceBuffer,
    session: ExecutionSession,
}

struct PreparedBuffers {
    patches: Tensor,
    noise: Tensor,
    token_ids: DeviceBuffer,
    normal_generator: Box<dyn NormalGenerator>,
}

enum ExecStrategy {
    Graph(CapturedGraph),
    Eager(EagerInputs),
}

/// Run the fixed eager traversal prefix shared by preparation, inference, and
/// calibration. Keeping RGB preprocessing here is important: an
/// `ExecutionSession` keys native executions by both operator and binding, so
/// every traversal after preparation must execute this prefix in the same
/// order.
fn eager_traversal<T>(
    model: &ModelVariant,
    spec: &InferenceSpec,
    inputs: &EagerInputs,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if let (Some(raw), Some(layout)) = (&inputs.raw_images, spec.image_layout) {
        model.preprocess_rgb(raw, &inputs.patches, kernel_image_layout(layout))?;
    }
    operation()
}

/// Owning prepared PI0.5 inference plan. Runs neither capture nor autotune.
/// Graph actions alias reusable device output: copy to host/device storage before
/// the next run if a stable result is needed. Runs are serialized on this runner.
/// Each call fully binds input and RNG key; there is no implicit episode counter.
pub struct Pi05PreparedInference {
    spec: InferenceSpec,
    backend: Arc<Context>,
    config: Arc<Pi05Config>,
    model: ModelVariant,
    strategy: ExecStrategy,
    normal_generator: RefCell<Box<dyn NormalGenerator>>,
    fallback_reason: Option<String>,
}

impl Pi05PreparedInference {
    fn update_eager_inputs(&self, inputs: &EagerInputs, request: &VlaRequest<'_>) -> Result<()> {
        let observation = request.observation;
        backend::synchronize(&self.backend)?;
        let patches = normalize_tensor(
            match &observation.vision {
                VisionObservation::Patches(patches) => Some(patches),
                VisionObservation::RgbU8 { bytes, .. } => {
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
                    None
                }
            },
            self.model.input_dtype(),
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
                    self.model.input_dtype(),
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
        let raw_rgb = matches!(&observation.vision, VisionObservation::RgbU8 { .. });
        Ok(Action::new(ops::with_session(&inputs.session, || {
            eager_traversal(&self.model, &self.spec, inputs, || {
                self.model.infer(
                    &inputs.patches,
                    &inputs.token_ids,
                    self.spec.token_count,
                    &inputs.noise,
                    raw_rgb,
                )
            })
        })?))
    }

    fn calibrate_eager(
        &self,
        inputs: &EagerInputs,
        request: &VlaRequest<'_>,
    ) -> Result<BTreeMap<String, f32>> {
        self.update_eager_inputs(inputs, request)?;
        ops::with_session(&inputs.session, || {
            eager_traversal(&self.model, &self.spec, inputs, || {
                self.model.calibrate(
                    &inputs.patches,
                    &inputs.token_ids,
                    self.spec.token_count,
                    &inputs.noise,
                )
            })
        })
    }
}

impl PreparedInference for Pi05PreparedInference {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }

    fn status(&self) -> PreparationStatus {
        PreparationStatus::Ready {
            mode: match self.strategy {
                ExecStrategy::Graph(_) => ExecutionMode::Graph,
                ExecStrategy::Eager(_) => ExecutionMode::Eager,
            },
            fallback_reason: self.fallback_reason.clone(),
        }
    }

    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        self.run_impl(request)
    }
}

impl Pi05PreparedInference {
    fn run_impl(&self, request: &VlaRequest<'_>) -> Result<Action> {
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
                self.model.input_dtype(),
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
                            self.model.input_dtype(),
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
                Ok(Action::new(graph.replay_output()?))
            }
            ExecStrategy::Eager(inputs) => self.run_eager(inputs, request),
        }
    }
}

/// PI0.5 model whose cached prepared plan owns all graph-visible resources.
///
/// A graph workspace can reserve multiple GiB, so the implicit `infer` path
/// retains only the most recently used shape. Callers that need more than one
/// simultaneously prepared shape can own those plans explicitly via `prepare`.
pub struct Pi05ModelRunner {
    backend: Arc<Context>,
    config: Arc<Pi05Config>,
    model: ModelVariant,
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

impl Pi05ModelRunner {
    pub(in crate::pi05) fn new(
        backend: Arc<Context>,
        config: Arc<Pi05Config>,
        model: ModelVariant,
    ) -> Self {
        Self {
            backend,
            config,
            model,
            prepared: RefCell::new(None),
        }
    }

    #[cfg(test)]
    pub(in crate::pi05) fn l3_policy_snapshot(&self) -> crate::pi05::model::L3PolicySnapshot {
        self.model.l3_policy_snapshot()
    }

    fn allocate_prepared_buffers(&self, spec: &InferenceSpec) -> Result<PreparedBuffers> {
        spec.validate()?;
        if spec.token_count > self.config.max_token_len {
            return Err(Error::Other(format!(
                "PI0.5 token count {} exceeds maximum {}",
                spec.token_count, self.config.max_token_len
            )));
        }
        let cuda = &*self.backend;
        let dtype = self.model.input_dtype();
        let raw_rgb = spec.image_layout.is_some();
        let patches = backend::to_device(&self.backend, &Tensor::zeros(
            patch_shape(&self.config),
            self.model.captured_patch_dtype(raw_rgb),
        ))?;
        let noise = backend::to_device(
            &self.backend,
            &Tensor::zeros(noise_shape(&self.config), dtype),
        )?;
        let normal_generator = backend::create_normal_generator(&self.backend, noise.clone())?;
        let token_ids = DeviceBuffer::alloc_zeros(spec.token_count * 4, cuda.device_id())
            .map_err(Error::Cuda)?;
        Ok(PreparedBuffers {
            patches,
            noise,
            token_ids,
            normal_generator,
        })
    }

    fn prepare_eager_inputs(
        &self,
        spec: &InferenceSpec,
        patches: Tensor,
        noise: Tensor,
        token_ids: DeviceBuffer,
        raw_images: Option<DeviceBuffer>,
    ) -> Result<EagerInputs> {
        let session = ExecutionSession::with_capacity(
            self.model.workspace_requirements(spec.token_count)?.bytes,
            self.backend.device_id(),
        )?;
        let inputs = EagerInputs {
            patches,
            raw_images,
            noise,
            token_ids,
            session,
        };
        let raw_rgb = inputs.raw_images.is_some();
        let output = ops::prepare_with_session(&inputs.session, || {
            eager_traversal(&self.model, spec, &inputs, || {
                self.model.infer(
                    &inputs.patches,
                    &inputs.token_ids,
                    spec.token_count,
                    &inputs.noise,
                    raw_rgb,
                )
            })
        })?;
        backend::synchronize(&self.backend)?;
        drop(output);
        Ok(inputs)
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
        let inputs = self.prepare_eager_inputs(spec, patches, noise, token_ids, raw_images)?;
        Ok(Pi05PreparedInference {
            spec: *spec,
            backend: Arc::clone(&self.backend),
            config: Arc::clone(&self.config),
            model: self.model.clone(),
            strategy: ExecStrategy::Eager(inputs),
            normal_generator: RefCell::new(normal_generator),
            fallback_reason: None,
        })
    }

    fn build_prepared(
        &self,
        spec: &InferenceSpec,
        policy: ExecutionPolicy,
    ) -> Result<Pi05PreparedInference> {
        self.build_prepared_using(spec, policy, |patches, tokens, noise| {
            super::prepare::capture_loaded(&self.model, spec, patches, tokens, noise)
        })
    }

    fn build_prepared_using(
        &self,
        spec: &InferenceSpec,
        policy: ExecutionPolicy,
        capture: impl FnOnce(&Tensor, &DeviceBuffer, &Tensor) -> Result<CapturedGraph>,
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

        let (graph, fallback_reason) = select_graph(policy, || capture(&patches, &token_ids, &noise))?;
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
                ExecStrategy::Eager(
                    self.prepare_eager_inputs(spec, patches, noise, token_ids, raw_images)?,
                )
            }
        };
        Ok(Pi05PreparedInference {
            spec: *spec,
            backend: Arc::clone(&self.backend),
            config: Arc::clone(&self.config),
            model: self.model.clone(),
            strategy,
            fallback_reason,
            normal_generator: RefCell::new(normal_generator),
        })
    }
}

impl VlaRuntime for Pi05ModelRunner {
    fn model_variant(&self) -> Option<&'static str> {
        Some(self.model.name())
    }

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
        let needs_prepare = self
            .prepared
            .borrow()
            .as_ref()
            .map_or(true, |(cached_spec, _)| *cached_spec != spec);
        if needs_prepare {
            // Release the old implicit plan before preparing a replacement.
            // Explicitly retained plans are caller-owned and remain alive.
            self.clear_prepared()?;
        }
        let prepared = {
            let mut cache = self.prepared.borrow_mut();
            if cache
                .as_ref()
                .is_some_and(|(cached_spec, _)| *cached_spec != spec)
            {
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
        backend::synchronize(&self.backend)?;
        drop(self.prepared.borrow_mut().take());
        Ok(())
    }

    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        let action = self.infer(request)?;
        backend::to_cpu(action.tensor())?.to_f32_vec()
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
        Ok(crate::pi05::Pi05CalibrationPlan::for_config(&self.config)
            .sites()
            .to_vec())
    }
}

pub(super) fn kernel_image_layout(layout: ImageLayout) -> KernelImageLayout {
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
    use crate::pi05::load::load_model_runner;
    use crate::LoadOptions;
    use std::path::Path;

    #[test]
    #[ignore = "requires CUDA and APXINF_PI05_TEST_CHECKPOINT"]
    fn native_preparation_failure_falls_back_and_retained_plan_runs() {
        let path =
            std::env::var("APXINF_PI05_TEST_CHECKPOINT").expect("fixed real checkpoint required");
        let backend = Arc::new(Context::new(0).unwrap());
        let runner = load_model_runner(
            Path::new(&path),
            backend.clone(),
            &LoadOptions {
                model_variant: Some("bf16".into()),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        let observation = Observation {
            vision: VisionObservation::Patches(Tensor::zeros(
                patch_shape(&runner.config),
                DType::BF16,
            )),
            token_ids: vec![0; 10],
            state: None,
            action_mask: None,
        };
        let noise = Tensor::zeros(noise_shape(&runner.config), DType::BF16);
        let request = VlaRequest::provided(&observation, &noise);
        let spec = observation.inference_spec();
        let failing_capture = |_: &Tensor, _: &DeviceBuffer, _: &Tensor| -> Result<CapturedGraph> {
            crate::pi05::backend::capture(&backend, || backend::synchronize(&backend))?;
            panic!("CUDA must reject synchronization during stream capture");
        };
        assert!(runner
            .build_prepared_using(&spec, ExecutionPolicy::RequireGraph, failing_capture)
            .is_err());
        let fallback = runner
            .build_prepared_using(&spec, ExecutionPolicy::PreferGraph, failing_capture)
            .unwrap();
        assert!(matches!(
            fallback.status(),
            PreparationStatus::Ready {
                mode: ExecutionMode::Eager,
                fallback_reason: Some(_)
            }
        ));
        let expected = backend::to_cpu(fallback.run(&request).unwrap().tensor())
            .unwrap()
            .to_f32_vec()
            .unwrap();
        drop(fallback);
        let recovered = runner
            .prepare_with_policy(&spec, ExecutionPolicy::RequireGraph)
            .unwrap();
        drop(runner);
        drop(backend);
        let actual = transfers::to_cpu(recovered.run(&request).unwrap().tensor())
            .unwrap()
            .to_f32_vec()
            .unwrap();
        assert_eq!(expected, actual);
    }

    #[test]
    #[ignore = "requires CUDA and APXINF_PI05_TEST_CHECKPOINT"]
    fn native_prepare_for_reuses_prepared_execution() {
        let path =
            std::env::var("APXINF_PI05_TEST_CHECKPOINT").expect("fixed real checkpoint required");
        let backend = Arc::new(Context::new(0).unwrap());
        let runner = load_model_runner(
            Path::new(&path),
            backend.clone(),
            &LoadOptions {
                model_variant: Some("bf16".into()),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        assert_eq!(runner.model_variant(), Some("bf16"));
        let observation = Observation {
            vision: VisionObservation::Patches(Tensor::zeros(
                patch_shape(&runner.config),
                DType::BF16,
            )),
            token_ids: vec![0; 10],
            state: None,
            action_mask: None,
        };
        let noise = Tensor::zeros(noise_shape(&runner.config), DType::BF16);
        let request = VlaRequest::provided(&observation, &noise);
        for policy in [ExecutionPolicy::Eager, ExecutionPolicy::RequireGraph] {
            let prepared = runner.prepare_for(&request, policy).unwrap();
            let first = backend::to_cpu(prepared.run(&request).unwrap().tensor())
                .unwrap()
                .to_f32_vec()
                .unwrap();
            let second = backend::to_cpu(prepared.run(&request).unwrap().tensor())
                .unwrap()
                .to_f32_vec()
                .unwrap();
            backend::synchronize(&backend).unwrap();
            assert_eq!(first, second);
            assert!(second.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    #[ignore = "requires CUDA and APXINF_PI05_TEST_CHECKPOINT"]
    fn rgb_calibration_reuses_prepared_eager_traversal() {
        let path =
            std::env::var("APXINF_PI05_TEST_CHECKPOINT").expect("fixed real checkpoint required");
        let backend = Arc::new(Context::new(0).unwrap());
        let runner = load_model_runner(
            Path::new(&path),
            backend,
            &LoadOptions {
                model_variant: Some("bf16".into()),
                ..LoadOptions::default()
            },
        )
        .unwrap();
        let observation = Observation {
            vision: VisionObservation::RgbU8 {
                bytes: vec![0; image_bytes(&runner.config)],
                layout: ImageLayout::Nhwc,
            },
            token_ids: vec![0; 10],
            state: None,
            action_mask: None,
        };
        let noise = Tensor::zeros(noise_shape(&runner.config), DType::F32);
        let records = runner
            .calibration_amax(&VlaRequest::provided(&observation, &noise))
            .unwrap();
        let expected = crate::pi05::Pi05CalibrationPlan::for_config(&runner.config)
            .sites()
            .len();
        assert_eq!(expected, 256);
        assert_eq!(records.len(), expected);
        assert!(records.values().all(|amax| amax.is_finite()));
    }

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
}
