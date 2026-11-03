//! PI0.5 Ascend runner: eager + graph capture execution (BF16).

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Arc;

use apxinf_core::{DType, Device, Error, NormalGenerator, Result, Tensor};
use half::bf16;

use crate::vla::{
    Action, ExecutionMode, ExecutionPolicy, ImageLayout, InferenceSpec, InitialLatent,
    Observation, PreparationStatus, PreparedInference, VisionObservation, VlaRequest, VlaRuntime,
};

use crate::ascend_pi05::backend::{
    self, ops, transfers, Context, DeviceBuffer, ExecutionSession,
    ImageLayout as KernelImageLayout,
};
use crate::ascend_pi05::model::ModelVariant;
use crate::ascend_pi05::Pi05Config;
use apxinf_ascend::l3::graph::{capture, CapturedGraph};

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

/// Owning prepared PI0.5 inference plan (eager or graph-captured).
pub struct Pi05PreparedInference {
    spec: InferenceSpec,
    backend: Arc<Context>,
    config: Arc<Pi05Config>,
    model: ModelVariant,
    inputs: EagerInputs,
    normal_generator: RefCell<Box<dyn NormalGenerator>>,
    captured_graph: Option<CapturedGraph>,
    /// Output tensor from the capture run. After graph replay, this tensor's
    /// device buffer contains the new inference result.
    capture_output: Option<Tensor>,
}

impl Drop for Pi05PreparedInference {
    fn drop(&mut self) {
        // Synchronize to ensure no pending async ops on the stream.
        let _ = backend::synchronize(&self.backend);
        // Drop captured graph FIRST — aclmdlRIDestroy may reference
        // device memory encoded in the graph (workspace pool, output buffers).
        // Rust drops fields in declaration order, which would free
        // inputs.session (WorkspacePool) before captured_graph, causing
        // aclmdlRIDestroy to segfault on freed pool memory.
        self.captured_graph.take();
        // Drop capture_output next — it may reference pool memory.
        self.capture_output.take();
        // Remaining fields (inputs, session/pool, model, etc.) drop in
        // declaration order after this returns.
    }
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
                    if bytes.len() != raw.size() {
                        return Err(Error::Other(format!(
                            "PI0.5 expected {} raw image bytes, got {}",
                            raw.size(),
                            bytes.len()
                        )));
                    }
                    raw.copy_from_host(bytes)?;
                    None
                }
            },
            self.model.input_dtype(),
            patch_shape(&self.config),
            "patches",
        )?;
        if let Some(patches) = patches.as_ref() {
            transfers::copy_cpu_to_npu(patches, &inputs.patches)?;
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
                transfers::copy_cpu_to_npu(&noise, &inputs.noise)?;
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

        // RGB preprocessing if needed
        if let (Some(raw), Some(layout)) = (&inputs.raw_images, self.spec.image_layout) {
            self.model
                .preprocess_rgb(raw, &inputs.patches, kernel_image_layout(layout))?;
        }

        // Phase 5a diagnostics: reset counters before inference
        apxinf_ascend::l3::reset_diagnostics();
        let infer_start = std::time::Instant::now();

        let action = ops::with_session(&inputs.session, || {
            self.model.infer(
                &inputs.patches,
                &inputs.token_ids,
                self.spec.token_count,
                &inputs.noise,
                raw_rgb,
                request.metadata.valid_prefix_tokens,
            )
        })?;

        let infer_elapsed = infer_start.elapsed();
        let sync_start = std::time::Instant::now();

        // Single synchronization after the entire forward pass.
        // Replaces ~9600 per-op aclrtSynchronizeStream calls.
        backend::synchronize(&self.backend)?;

        let sync_elapsed = sync_start.elapsed();

        apxinf_ascend::l3::print_diagnostics();

        Ok(Action::new(action))
    }
}

impl PreparedInference for Pi05PreparedInference {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }

    fn status(&self) -> PreparationStatus {
        let mode = if self.captured_graph.is_some() {
            ExecutionMode::Graph
        } else {
            ExecutionMode::Eager
        };
        PreparationStatus::Ready {
            mode,
            fallback_reason: None,
        }
    }

    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        let observation = request.observation;
        observation.validate()?;
        if !self.spec.matches(observation) {
            return Err(Error::Other(format!(
                "prepared PI0.5 spec {:?} does not match observation {:?}",
                self.spec,
                observation.inference_spec()
            )));
        }
        self.update_eager_inputs(&self.inputs, request)?;
        let raw_rgb = matches!(&observation.vision, VisionObservation::RgbU8 { .. });

        // RGB preprocessing if needed (outside graph/closure — input update)
        if let (Some(raw), Some(layout)) = (&self.inputs.raw_images, self.spec.image_layout) {
            self.model
                .preprocess_rgb(raw, &self.inputs.patches, kernel_image_layout(layout))?;
        }

        apxinf_ascend::l3::reset_diagnostics();
        let infer_start = std::time::Instant::now();

        let action = if let Some(graph) = &self.captured_graph {
            // Graph replay: single aclmdlRIExecuteAsync replaces 4761 per-op dispatches.
            graph.replay()?;
            // The capture_output tensor's device buffer now contains the result.
            self.capture_output.clone().expect("capture output")
        } else {
            // Eager mode: run all 4761 ops through with_session.
            ops::with_session(&self.inputs.session, || {
                self.model.infer(
                    &self.inputs.patches,
                    &self.inputs.token_ids,
                    self.spec.token_count,
                    &self.inputs.noise,
                    raw_rgb,
                    request.metadata.valid_prefix_tokens,
                )
            })?
        };

        let infer_elapsed = infer_start.elapsed();
        let sync_start = std::time::Instant::now();

        backend::synchronize(&self.backend)?;

        let sync_elapsed = sync_start.elapsed();

        let mode_str = if self.captured_graph.is_some() { "graph" } else { "eager" };
        apxinf_ascend::l3::print_diagnostics();

        Ok(Action::new(action))
    }
}

/// PI0.5 Ascend model runner (eager only).
pub struct Pi05ModelRunner {
    backend: Arc<Context>,
    config: Arc<Pi05Config>,
    model: ModelVariant,
    prepared: RefCell<Option<(InferenceSpec, std::rc::Rc<Pi05PreparedInference>)>>,
}

impl Pi05ModelRunner {
    pub(in crate::ascend_pi05) fn new(
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

    fn allocate_prepared_buffers(&self, spec: &InferenceSpec) -> Result<PreparedBuffers> {
        self.backend.ensure_context();
        spec.validate()?;
        if spec.token_count > self.config.max_token_len {
            return Err(Error::Other(format!(
                "PI0.5 token count {} exceeds maximum {}",
                spec.token_count, self.config.max_token_len
            )));
        }
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
        let token_ids = DeviceBuffer::alloc_zeros(spec.token_count * 4, self.backend.device_id())?;
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

        // Phase 5a diagnostics: time the prepare phase
        apxinf_ascend::l3::reset_diagnostics();
        let prepare_start = std::time::Instant::now();

        let output = ops::prepare_with_session(&inputs.session, || {
            if let (Some(raw), Some(layout)) = (&inputs.raw_images, spec.image_layout) {
                self.model
                    .preprocess_rgb(raw, &inputs.patches, kernel_image_layout(layout))?;
            }
            self.model.infer(
                &inputs.patches,
                &inputs.token_ids,
                spec.token_count,
                &inputs.noise,
                raw_rgb,
                None,
            )
        })?;

        let prepare_closure_elapsed = prepare_start.elapsed();
        let sync_start = std::time::Instant::now();
        backend::synchronize(&self.backend)?;
        let sync_elapsed = sync_start.elapsed();

        apxinf_ascend::l3::print_diagnostics();

        drop(output);
        Ok(inputs)
    }

    fn build_eager(&self, spec: &InferenceSpec) -> Result<Pi05PreparedInference> {
        let build_start = std::time::Instant::now();
        let PreparedBuffers {
            patches,
            noise,
            token_ids,
            normal_generator,
        } = self.allocate_prepared_buffers(spec)?;
        let raw_images = spec
            .image_layout
            .is_some()
            .then(|| DeviceBuffer::alloc_zeros(image_bytes(&self.config), self.backend.device_id()))
            .transpose()?;
        let inputs = self.prepare_eager_inputs(spec, patches, noise, token_ids, raw_images)?;
        Ok(Pi05PreparedInference {
            spec: *spec,
            backend: Arc::clone(&self.backend),
            config: Arc::clone(&self.config),
            model: self.model.clone(),
            inputs,
            normal_generator: RefCell::new(normal_generator),
            captured_graph: None,
            capture_output: None,
        })
    }

    /// Build a graph-captured prepared inference.
    ///
    /// First runs a warmup pass (same as eager) to stabilize workspace pool
    /// allocations, then captures the forward pass into an RI graph.
    /// Subsequent `run` calls replay the graph (1 dispatch vs 4761).
    fn build_graphed(&self, spec: &InferenceSpec) -> Result<Pi05PreparedInference> {
        let build_start = std::time::Instant::now();
        let PreparedBuffers {
            patches,
            noise,
            token_ids,
            normal_generator,
        } = self.allocate_prepared_buffers(spec)?;
        let raw_images = spec
            .image_layout
            .is_some()
            .then(|| DeviceBuffer::alloc_zeros(image_bytes(&self.config), self.backend.device_id()))
            .transpose()?;
        let inputs = self.prepare_eager_inputs(spec, patches, noise, token_ids, raw_images)?;
        let raw_rgb = inputs.raw_images.is_some();

        // Preprocess RGB outside capture — it uses D2H copy which is
        // forbidden during RI capture.
        if let (Some(raw), Some(layout)) = (&inputs.raw_images, spec.image_layout) {
            self.model
                .preprocess_rgb(raw, &inputs.patches, kernel_image_layout(layout))?;
        }

        // Capture only the model.infer forward pass into an RI graph.
        let capture_start = std::time::Instant::now();
        let backend_ref = &self.backend;
        let model_ref = &self.model;
        let inputs_ref = &inputs;
        let token_count = spec.token_count;

        // Graph capture fixes position_offset to None (original mode).
        // LeRobot-compatible mode (valid_prefix_tokens) requires eager execution
        // because the offset value changes the RoPE frequencies baked into the graph.
        let (captured_graph, output) = capture(backend_ref, &inputs.session, || {
            model_ref.infer(
                &inputs_ref.patches,
                &inputs_ref.token_ids,
                token_count,
                &inputs_ref.noise,
                raw_rgb,
                None,
            )
        })?;
        let capture_elapsed = capture_start.elapsed();

        Ok(Pi05PreparedInference {
            spec: *spec,
            backend: Arc::clone(&self.backend),
            config: Arc::clone(&self.config),
            model: self.model.clone(),
            inputs,
            normal_generator: RefCell::new(normal_generator),
            captured_graph: Some(captured_graph),
            capture_output: Some(output),
        })
    }

    fn build_prepared(&self, spec: &InferenceSpec) -> Result<Pi05PreparedInference> {
        // APXINF_NO_GRAPH=1 forces eager mode (for op-distribution profiling).
        if std::env::var("APXINF_NO_GRAPH").is_ok() {
                return self.build_eager(spec);
        }
        // Try graph capture first; fall back to eager on failure.
        match self.build_graphed(spec) {
            Ok(prepared) => {
                        Ok(prepared)
            }
            Err(e) => {
                self.build_eager(spec)
            }
        }
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
        let total_start = std::time::Instant::now();
        let observation = request.observation;
        observation.validate()?;
        let spec = observation.inference_spec();
        let needs_prepare = self
            .prepared
            .borrow()
            .as_ref()
            .map_or(true, |(cached_spec, _)| *cached_spec != spec);
        if needs_prepare {
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
            cached_or_build(&mut cache, spec, || self.build_prepared(&spec))?
        };
        let prepare_elapsed = total_start.elapsed();
        let result = prepared.run(request);
        result
    }

    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        self.prepare_with_policy(spec, ExecutionPolicy::Eager)
    }

    fn prepare_with_policy(
        &self,
        spec: &InferenceSpec,
        _policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        Ok(Box::new(self.build_prepared(spec)?))
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
        "graph"
    }

    fn clear_prepared(&self) -> Result<()> {
        backend::synchronize(&self.backend)?;
        drop(self.prepared.borrow_mut().take());
        Ok(())
    }

    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        let t0 = std::time::Instant::now();
        let action = self.infer(request)?;
        let infer_elapsed = t0.elapsed();
        let t1 = std::time::Instant::now();
        let cpu_tensor = backend::to_cpu(action.tensor())?;
        let to_cpu_elapsed = t1.elapsed();
        let t2 = std::time::Instant::now();
        let result = cpu_tensor.to_f32_vec();
        let to_f32_elapsed = t2.elapsed();
        result
    }
    fn infer_host_f32_with_dump(
        &self,
        request: &VlaRequest<'_>,
    ) -> Result<(Vec<f32>, std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)>)> {
        use std::collections::BTreeMap;
        request.observation.validate()?;
        let observation = request.observation;
        let spec = observation.inference_spec();
        let prepared = self.build_eager(&spec)?;
        let raw_rgb = matches!(&observation.vision, VisionObservation::RgbU8 { .. });

        // CRITICAL: copy provided noise/patches/token_ids into the prepared
        // eager buffers (otherwise they stay at the zeros from
        // allocate_prepared_buffers). Mirrors run_eager/run update path.
        prepared.update_eager_inputs(&prepared.inputs, request)?;

        // RGB preprocessing if needed (outside session — input update).
        if let (Some(raw), Some(layout)) = (&prepared.inputs.raw_images, spec.image_layout) {
            self.model
                .preprocess_rgb(raw, &prepared.inputs.patches, kernel_image_layout(layout))?;
        }

        apxinf_ascend::l3::reset_diagnostics();
        let dump = ops::with_session(&prepared.inputs.session, || {
            self.model.infer_with_dump(
                &prepared.inputs.patches,
                &prepared.inputs.token_ids,
                spec.token_count,
                &prepared.inputs.noise,
                request.metadata.valid_prefix_tokens,
            )
        })?;
        backend::synchronize(&self.backend)?;

        // Copy all intermediate tensors to host as f32.
        let tensor_to_f32 = |t: &Tensor| -> Result<(Vec<f32>, Vec<usize>)> {
            let cpu = backend::to_cpu(t)?;
            let shape = cpu.shape().dims().to_vec();
            let data = cpu.to_f32_vec()?;
            Ok((data, shape))
        };

        let action_f32 = tensor_to_f32(&dump.action)?;
        let mut captures: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();

        // Vision
        captures.insert("vision.output".to_string(), tensor_to_f32(&dump.vision_output)?);
        for (i, layer_out) in dump.vision_layer_outputs.iter().enumerate() {
            captures.insert(
                format!("vision.layer.{i}.output"),
                tensor_to_f32(layer_out)?,
            );
        }

        // Prefix embedding
        captures.insert(
            "prefix.embedding".to_string(),
            tensor_to_f32(&dump.prefix_embedding)?,
        );

        // Language layers
        for (i, layer_out) in dump.language_layer_outputs.iter().enumerate() {
            captures.insert(
                format!("language.layer.{i}.output"),
                tensor_to_f32(layer_out)?,
            );
        }

        // Denoise steps
        for (step, (state, action_layers)) in dump.denoise_step_outputs.iter().enumerate() {
            captures.insert(
                format!("denoise.step.{step}.state"),
                tensor_to_f32(state)?,
            );
            for (layer, layer_out) in action_layers.iter().enumerate() {
                captures.insert(
                    format!("denoise.step.{step}.action_layer.{layer}.output"),
                    tensor_to_f32(layer_out)?,
                );
            }
        }

        // First transformer block internals per denoise step.
        // Keys: denoise.step.{step}.block0.{field}
        for (step, internals_opt) in dump.denoise_step_first_block_internals.iter().enumerate() {
            let Some(internals) = internals_opt else { continue };
            let prefix = format!("denoise.step.{step}.block0");
            captures.insert(format!("{prefix}.normalized"), tensor_to_f32(&internals.normalized)?);
            captures.insert(format!("{prefix}.qkv"), tensor_to_f32(&internals.qkv)?);
            captures.insert(format!("{prefix}.q_post_rope"), tensor_to_f32(&internals.q_post_rope)?);
            captures.insert(format!("{prefix}.k_post_rope"), tensor_to_f32(&internals.k_post_rope)?);
            captures.insert(format!("{prefix}.v"), tensor_to_f32(&internals.v)?);
            captures.insert(format!("{prefix}.attn_out"), tensor_to_f32(&internals.attn_out)?);
            captures.insert(format!("{prefix}.attn_proj"), tensor_to_f32(&internals.attn_proj)?);
            captures.insert(format!("{prefix}.post_attn_hidden"), tensor_to_f32(&internals.post_attn_hidden)?);
            captures.insert(format!("{prefix}.post_attn_norm"), tensor_to_f32(&internals.post_attn_norm)?);
            captures.insert(format!("{prefix}.geglu_out"), tensor_to_f32(&internals.geglu_out)?);
            captures.insert(format!("{prefix}.mlp_proj"), tensor_to_f32(&internals.mlp_proj)?);
            captures.insert(format!("{prefix}.mlp_hidden"), tensor_to_f32(&internals.mlp_hidden)?);
            captures.insert(format!("{prefix}.mlp_next_norm"), tensor_to_f32(&internals.mlp_next_norm)?);
        }

        Ok((action_f32.0, captures))
    }

    fn calibration_amax(&self, request: &VlaRequest<'_>) -> Result<BTreeMap<String, f32>> {
        request.observation.validate()?;
        let prepared = self.build_eager(&request.observation.inference_spec())?;
        prepared.run_eager(&prepared.inputs, request)?;
        // The calibrate method is on the model, called via the model variant
        self.model.calibrate(
            &prepared.inputs.patches,
            &prepared.inputs.token_ids,
            prepared.spec.token_count,
            &prepared.inputs.noise,
        )
    }

    fn calibration_plan(&self) -> Result<Vec<String>> {
        Ok(crate::ascend_pi05::Pi05CalibrationPlan::for_config(&self.config)
            .sites()
            .to_vec())
    }
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

fn copy_token_ids(buffer: &DeviceBuffer, token_ids: &[u32]) -> Result<()> {
    let bytes = token_ids
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    buffer.copy_from_host(&bytes)
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

fn cached_or_build<K, V>(
    cache: &mut Option<(K, std::rc::Rc<V>)>,
    key: K,
    build: impl FnOnce() -> Result<V>,
) -> Result<std::rc::Rc<V>>
where
    K: Copy + PartialEq,
{
    if let Some((cached_key, value)) = cache.as_ref() {
        if *cached_key == key {
            return Ok(std::rc::Rc::clone(value));
        }
    }
    drop(cache.take());
    let value = std::rc::Rc::new(build()?);
    *cache = Some((key, std::rc::Rc::clone(&value)));
    Ok(value)
}
