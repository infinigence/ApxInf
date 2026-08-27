//! Owning `VlaRuntime` adapter for the native BF16 DM05 implementation.

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{
    Backend, DType, Device, Error, NormalGenerator, Result, SamplingBackend, Tensor,
};
use half::bf16;

use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};
use crate::vla::{
    Action, ImageLayout, InferenceSpec, InitialLatent, PreparedInference, VisionObservation,
    VlaDimensions, VlaRequest, VlaRuntime,
};

use super::backend::{
    kernels, transfers, DeviceBuffer, ImageLayout as KernelImageLayout, RuntimeBackend,
};
use super::runtime::Dm05CapturedGraph;
use super::{DeviceDm05Weights, Dm05Bf16Runtime, Dm05Config, Dm05PreparedShape, Dm05Weights};

struct EagerInputs {
    patches: Tensor,
    raw_images: Option<DeviceBuffer>,
    noise: Tensor,
    token_ids: DeviceBuffer,
}

enum ExecStrategy {
    /// DM05 capture is deferred until the first request because image-token
    /// run positions, not merely token count, determine graph topology.
    PendingCapture(Rc<EagerInputs>),
    Graph(Dm05CapturedGraph),
    EagerFallback {
        inputs: Rc<EagerInputs>,
        reason: String,
    },
}

pub struct Dm05PreparedInference {
    spec: InferenceSpec,
    backend: Arc<RuntimeBackend>,
    config: Arc<Dm05Config>,
    runtime: Dm05Bf16Runtime,
    shape: Dm05PreparedShape,
    strategy: RefCell<ExecStrategy>,
    normal_generator: RefCell<Box<dyn NormalGenerator>>,
}

impl Dm05PreparedInference {
    fn preprocess_rgb(
        &self,
        images: &DeviceBuffer,
        patches: &Tensor,
        layout: ImageLayout,
    ) -> Result<()> {
        kernels::preprocess::rgb_u8_to_patches_bf16(
            self.backend.context(),
            images,
            patches,
            self.config.num_views,
            self.config.vision.image_size,
            self.config.vision.patch_size,
            kernel_image_layout(layout),
        )
    }

    fn update_eager_inputs(&self, inputs: &EagerInputs, request: &VlaRequest<'_>) -> Result<()> {
        self.backend.synchronize()?;
        let observation = request.observation;
        match &observation.vision {
            VisionObservation::Patches(patches) => {
                let patches = normalize_bf16_tensor(
                    patches,
                    patch_shape(&self.config),
                    "preprocessed patches",
                )?;
                transfers::copy_cpu_to_cuda(&patches, &inputs.patches)?;
            }
            VisionObservation::RgbU8 { bytes, layout } => {
                validate_image_bytes(&self.config, bytes)?;
                let raw = inputs
                    .raw_images
                    .as_ref()
                    .ok_or_else(|| Error::Other("DM05 prepared plan expects patch input".into()))?;
                raw.copy_from_host(bytes).map_err(Error::Cuda)?;
                // This makes the same loaded inputs immediately usable if
                // graph capture fails after the preparation attempt.
                self.preprocess_rgb(raw, &inputs.patches, *layout)?;
            }
        }
        match request.initial_latent {
            InitialLatent::Provided(latent) => {
                let noise =
                    normalize_bf16_tensor(latent, noise_shape(&self.config), "initial latent")?;
                transfers::copy_cpu_to_cuda(&noise, &inputs.noise)?;
            }
            InitialLatent::Generate { rng } => {
                self.normal_generator.borrow_mut().generate(rng)?;
            }
        }
        copy_token_ids(&inputs.token_ids, &observation.token_ids)
    }

    fn run_loaded_eager(&self, inputs: &EagerInputs, request: &VlaRequest<'_>) -> Result<Action> {
        let observation = request.observation;
        Ok(Action::new(self.runtime.infer(
            &inputs.patches,
            &inputs.token_ids,
            &observation.token_ids,
            &inputs.noise,
            &self.shape,
        )?))
    }

    fn run_eager(&self, inputs: &EagerInputs, request: &VlaRequest<'_>) -> Result<Action> {
        self.update_eager_inputs(inputs, request)?;
        self.run_loaded_eager(inputs, request)
    }

    fn capture_loaded(
        &self,
        inputs: &EagerInputs,
        request: &VlaRequest<'_>,
    ) -> Result<Dm05CapturedGraph> {
        let token_ids = &request.observation.token_ids;
        match self.spec.image_layout {
            None => self.runtime.capture_infer(
                &inputs.patches,
                &inputs.token_ids,
                token_ids,
                &inputs.noise,
                &self.shape,
            ),
            Some(layout) => self.runtime.capture_infer_rgb_u8(
                kernel_image_layout(layout),
                inputs.raw_images.as_ref().ok_or_else(|| {
                    Error::Other("DM05 raw RGB capture is missing its input buffer".into())
                })?,
                &inputs.patches,
                &inputs.token_ids,
                token_ids,
                &inputs.noise,
                &self.shape,
            ),
        }
    }

    fn run_graph(&self, graph: &Dm05CapturedGraph, request: &VlaRequest<'_>) -> Result<Action> {
        let observation = request.observation;
        match (&observation.vision, request.initial_latent) {
            (VisionObservation::Patches(patches), InitialLatent::Provided(latent)) => {
                let patches = normalize_bf16_tensor(
                    patches,
                    patch_shape(&self.config),
                    "preprocessed patches",
                )?;
                let noise =
                    normalize_bf16_tensor(latent, noise_shape(&self.config), "initial latent")?;
                graph.update_inputs(&patches, &observation.token_ids, &noise)?;
            }
            (VisionObservation::Patches(patches), InitialLatent::Generate { rng }) => {
                let patches = normalize_bf16_tensor(
                    patches,
                    patch_shape(&self.config),
                    "preprocessed patches",
                )?;
                graph.update_inputs_without_noise(&patches, &observation.token_ids)?;
                self.normal_generator.borrow_mut().generate(rng)?;
            }
            (VisionObservation::RgbU8 { bytes, .. }, InitialLatent::Provided(latent)) => {
                validate_image_bytes(&self.config, bytes)?;
                let noise =
                    normalize_bf16_tensor(latent, noise_shape(&self.config), "initial latent")?;
                graph.update_raw_image_inputs(bytes, &observation.token_ids, &noise)?;
            }
            (VisionObservation::RgbU8 { bytes, .. }, InitialLatent::Generate { rng }) => {
                validate_image_bytes(&self.config, bytes)?;
                graph.update_raw_image_inputs_without_noise(bytes, &observation.token_ids)?;
                self.normal_generator.borrow_mut().generate(rng)?;
            }
        }
        graph.replay()?;
        Ok(Action::new(graph.snapshot_output()?))
    }

    /// Observable execution metadata for qualification and service logs.
    pub fn execution_strategy(&self) -> &'static str {
        match &*self.strategy.borrow() {
            ExecStrategy::PendingCapture(_) => "pending-cuda-graph-capture",
            ExecStrategy::Graph(_) => "cuda-graph",
            ExecStrategy::EagerFallback { .. } => "eager-fallback",
        }
    }

    pub fn eager_fallback_reason(&self) -> Option<String> {
        match &*self.strategy.borrow() {
            ExecStrategy::EagerFallback { reason, .. } => Some(reason.clone()),
            _ => None,
        }
    }
}

impl PreparedInference for Dm05PreparedInference {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }

    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        let observation = request.observation;
        observation.validate()?;
        self.config.validate_prefix_tokens(&observation.token_ids)?;
        if !self.spec.matches(observation) {
            return Err(Error::Other(format!(
                "prepared DM05 spec {:?} does not match observation {:?}",
                self.spec,
                observation.inference_spec()
            )));
        }
        let mut strategy = self.strategy.borrow_mut();

        // A same-length prompt may move an image run. Such a request is valid,
        // but replaying the old graph would silently use stale row views and
        // segment launch shapes. Preserve correctness by dropping the large
        // graph workspace before switching this prepared plan to eager.
        if let ExecStrategy::Graph(graph) = &*strategy {
            if !graph.matches_prefix_layout(&observation.token_ids)? {
                let (patches, raw_images, noise, token_ids) = graph.cloned_inputs();
                let reason = format!(
                    "image-token layout changed for fixed token count {}",
                    self.spec.token_count
                );
                eprintln!("[apxinf] DM05 CUDA graph invalidated, using eager: {reason}");
                *strategy = ExecStrategy::EagerFallback {
                    inputs: Rc::new(EagerInputs {
                        patches,
                        raw_images,
                        noise,
                        token_ids,
                    }),
                    reason,
                };
            }
        }

        match &*strategy {
            ExecStrategy::Graph(graph) => self.run_graph(graph, request),
            ExecStrategy::EagerFallback { inputs, .. } => self.run_eager(inputs, request),
            ExecStrategy::PendingCapture(inputs) => {
                let inputs = Rc::clone(inputs);
                self.update_eager_inputs(&inputs, request)?;
                match self.capture_loaded(&inputs, request) {
                    Ok(graph) => {
                        graph.replay()?;
                        let action = Action::new(graph.snapshot_output()?);
                        *strategy = ExecStrategy::Graph(graph);
                        Ok(action)
                    }
                    Err(error) => {
                        let reason = error.to_string();
                        eprintln!(
                            "[apxinf] DM05 CUDA graph capture unavailable, using eager: {reason}"
                        );
                        let action = self.run_loaded_eager(&inputs, request)?;
                        *strategy = ExecStrategy::EagerFallback { inputs, reason };
                        Ok(action)
                    }
                }
            }
        }
    }
}

pub struct Dm05VlaRuntime {
    backend: Arc<RuntimeBackend>,
    config: Arc<Dm05Config>,
    runtime: Dm05Bf16Runtime,
    prepared: RefCell<Option<(InferenceSpec, Rc<Dm05PreparedInference>)>>,
}

impl Dm05VlaRuntime {
    fn build_prepared(&self, spec: &InferenceSpec) -> Result<Dm05PreparedInference> {
        spec.validate()?;
        if spec.token_count > self.config.max_prefix_len {
            return Err(Error::Other(format!(
                "DM05 token count {} exceeds maximum {}",
                spec.token_count, self.config.max_prefix_len
            )));
        }
        let device = self.backend.context().device_id();
        let patches = self
            .backend
            .to_device(&Tensor::zeros(patch_shape(&self.config), DType::BF16))?;
        let noise = self
            .backend
            .to_device(&Tensor::zeros(noise_shape(&self.config), DType::BF16))?;
        let normal_generator = self.backend.create_normal_generator(noise.clone())?;
        let token_ids =
            DeviceBuffer::alloc_zeros(spec.token_count * 4, device).map_err(Error::Cuda)?;
        let raw_images = spec
            .image_layout
            .map(|_| DeviceBuffer::alloc_zeros(image_bytes(&self.config), device))
            .transpose()
            .map_err(Error::Cuda)?;
        let inputs = Rc::new(EagerInputs {
            patches,
            raw_images,
            noise,
            token_ids,
        });
        Ok(Dm05PreparedInference {
            spec: *spec,
            backend: Arc::clone(&self.backend),
            config: Arc::clone(&self.config),
            runtime: self.runtime.clone(),
            shape: self.runtime.prepare_shape(spec.token_count)?,
            strategy: RefCell::new(ExecStrategy::PendingCapture(inputs)),
            normal_generator: RefCell::new(normal_generator),
        })
    }
}

impl VlaRuntime for Dm05VlaRuntime {
    fn dimensions(&self) -> VlaDimensions {
        VlaDimensions {
            action_horizon: self.config.action_horizon,
            action_dim: self.config.action_dim,
            num_views: self.config.num_views,
            image_size: self.config.vision.image_size,
            patch_size: self.config.vision.patch_size,
            max_token_len: self.config.max_prefix_len,
        }
    }

    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        request.observation.validate()?;
        self.config
            .validate_prefix_tokens(&request.observation.token_ids)?;
        let spec = request.observation.inference_spec();
        let prepared = {
            let mut cache = self.prepared.borrow_mut();
            cached_or_build(&mut cache, spec, || self.build_prepared(&spec))?
        };
        prepared.run(request)
    }

    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        Ok(Box::new(self.build_prepared(spec)?))
    }

    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        let action = self.infer(request)?;
        self.backend.to_cpu(action.tensor())?.to_f32_vec()
    }
}

pub(super) fn load_registered(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if !matches!(device, Device::Cuda(_)) {
        return Err(Error::Other("DM05 native runtime requires CUDA".into()));
    }
    let backend = crate::accelerator::cuda::downcast_arc(backend)
        .ok_or_else(|| Error::Other("DM05 is only registered for CUDA".into()))?;
    if !matches!(
        options.precision,
        ModelPrecision::Auto | ModelPrecision::Bf16
    ) {
        return Err(Error::Other(
            "DM05 native runtime supports BF16 precision only".into(),
        ));
    }
    if options.config.is_some()
        || options.synthetic.is_some()
        || options.calibration_path.is_some()
        || options.tuning_path.is_some()
        || options.uniform_fp8_scale.is_some()
        || options.text_weight_dtype.is_some()
    {
        return Err(Error::Other(
            "DM05 native loader does not accept PI0.5 config, synthetic, calibration, tuning, or text-dtype options"
                .into(),
        ));
    }
    if let Some(num_views) = options.num_views {
        if num_views != Dm05Config::SUPPORTED_NUM_VIEWS {
            return Err(Error::Other(format!(
                "DM05 LIBERO requires exactly {} views, got {num_views}",
                Dm05Config::SUPPORTED_NUM_VIEWS
            )));
        }
    }
    let action_horizon = options.action_horizon.ok_or_else(|| {
        Error::Other("DM05 LIBERO load requires explicit action_horizon=10".into())
    })?;
    if action_horizon != 10 {
        return Err(Error::Other(format!(
            "DM05 LIBERO load requires action_horizon=10, got {action_horizon}"
        )));
    }
    let root = artifact_root(path);
    let config = Arc::new(
        Dm05Config::from_json_file(&root.join("config.json"))?
            .with_action_horizon(action_horizon)?,
    );
    let host_weights = Dm05Weights::from_safetensors(&config, path)?;
    let device_weights = Arc::new(DeviceDm05Weights::from_host(&host_weights, &*backend)?);
    let runtime = Dm05Bf16Runtime::new(Arc::clone(&backend), Arc::clone(&config), device_weights)?;
    Ok(LoadedModel::Vla(Box::new(Dm05VlaRuntime {
        backend,
        config,
        runtime,
        prepared: RefCell::new(None),
    })))
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
    drop(cache.take());
    let value = Rc::new(build()?);
    *cache = Some((key, Rc::clone(&value)));
    Ok(value)
}

fn artifact_root(path: &Path) -> &Path {
    if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    }
}

fn kernel_image_layout(layout: ImageLayout) -> KernelImageLayout {
    match layout {
        ImageLayout::Nhwc => KernelImageLayout::Nhwc,
        ImageLayout::Nchw => KernelImageLayout::Nchw,
    }
}

fn patch_shape(config: &Dm05Config) -> Vec<usize> {
    vec![
        config.num_views * config.patches_per_view(),
        3 * config.vision.patch_size * config.vision.patch_size,
    ]
}

fn noise_shape(config: &Dm05Config) -> Vec<usize> {
    vec![config.action_horizon, config.action_dim]
}

fn image_bytes(config: &Dm05Config) -> usize {
    config.num_views * 3 * config.vision.image_size * config.vision.image_size
}

fn validate_image_bytes(config: &Dm05Config, bytes: &[u8]) -> Result<()> {
    let expected = image_bytes(config);
    if bytes.len() != expected {
        return Err(Error::Other(format!(
            "DM05 expected {expected} raw image bytes, got {}",
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

fn normalize_bf16_tensor(tensor: &Tensor, shape: Vec<usize>, label: &str) -> Result<Tensor> {
    if tensor.device() != Device::Cpu {
        return Err(Error::Other(format!("DM05 {label} must be a CPU tensor")));
    }
    if tensor.shape().dims() != shape {
        return Err(Error::Other(format!(
            "DM05 {label} shape {:?} does not match {shape:?}",
            tensor.shape().dims()
        )));
    }
    if tensor.dtype() == DType::BF16 {
        return Ok(tensor.clone());
    }
    let values = tensor
        .to_f32_vec()?
        .into_iter()
        .map(bf16::from_f32)
        .collect::<Vec<_>>();
    Tensor::from_bf16(shape, &values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn most_recent_cache_drops_before_replacement() {
        let mut cache = None;
        let first = cached_or_build(&mut cache, 1usize, || Ok(vec![1])).unwrap();
        let reused = cached_or_build(&mut cache, 1usize, || Ok(vec![2])).unwrap();
        assert!(Rc::ptr_eq(&first, &reused));
        let second = cached_or_build(&mut cache, 2usize, || Ok(vec![2])).unwrap();
        assert_eq!(&*second, &[2]);
    }
}
