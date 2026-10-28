//! X-VLA runtime: observation binding, RGB preprocessing, and execution plans.
//!
//! The runtime is the VLA-family seam: it turns one [`VlaRequest`] into the
//! three-stage forward pass (DaViT view tower, Florence-2 merged encoder, linear
//! flow action head). RGB frames are normalized and patchified on device by the
//! X-VLA kernels, so no host-side pixel preprocessing happens here.
//!
//! Preparation can capture the three stages into CUDA graphs. Vision, language
//! and the ten-step action loop each own a graph whose inputs and results live
//! in fixed buffers; a plan captures once per `(views, domain)` pair and then
//! refreshes those buffers before every replay.

use std::cell::RefCell;
use std::path::Path;
use std::sync::Arc;

use apxinf_core::{Backend, DType, Device, Error, Graph, Result, SamplingBackend, Shape, Tensor};
use half::bf16;

use crate::accelerator::cuda::RuntimeBackend;
use crate::auto::{LoadOptions, LoadedModel};
use crate::vla::{
    Action, ExecutionMode, ExecutionPolicy, ImageLayout, InferenceSpec, InitialLatent,
    PreparationStatus, PreparedInference, VisionObservation, VlaContract, VlaRequest, VlaRuntime,
};

use super::backend::kernels::xvla_ops;
use super::backend::{kernels, transfers, Context, DeviceBuffer};
use super::config::XvlaConfig;
use super::model::action::generate_actions;
use super::model::text::{encode_vlm, encode_vlm_with_plan};
use super::model::vision::encode_image;
use super::weights::{ActionHeadWeights, VlmWeights};

/// First graph arena size; doubled when a warmup traversal exhausts it.
const INITIAL_WORKSPACE_BYTES: usize = 256 * 1024 * 1024;
const MAX_WORKSPACE_BYTES: usize = 2560 * 1024 * 1024;

pub(crate) fn register_builtin() {
    crate::registry::register("xvla-cuda", load_registered);
}

fn load_registered(
    path: &Path,
    _device: Device,
    backend: Arc<dyn Backend>,
    _options: &LoadOptions,
) -> Result<LoadedModel> {
    Ok(LoadedModel::Vla(Box::new(load_runtime(path, backend)?)))
}

fn load_runtime(path: &Path, backend: Arc<dyn Backend>) -> Result<XvlaRuntime> {
    let backend = crate::accelerator::cuda::downcast_arc(backend)
        .ok_or_else(|| Error::Other("X-VLA is only registered for CUDA".into()))?;
    let root = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf()
    };
    let config = Arc::new(XvlaConfig::from_checkpoint(&root)?);
    let ctx = backend.context();
    let vlm = Arc::new(VlmWeights::from_checkpoint(&config, path, ctx)?);
    let head = Arc::new(ActionHeadWeights::from_checkpoint(&config, path, ctx)?);
    Ok(XvlaRuntime {
        backend,
        config,
        vlm,
        head,
        prepared: RefCell::new(None),
    })
}

/// Loaded X-VLA checkpoint plus its execution resources.
pub(crate) struct XvlaRuntime {
    backend: Arc<RuntimeBackend>,
    config: Arc<XvlaConfig>,
    vlm: Arc<VlmWeights>,
    head: Arc<ActionHeadWeights>,
    prepared: RefCell<Option<(InferenceSpec, XvlaPreparedInference)>>,
}

impl XvlaRuntime {
    fn build_prepared(
        &self,
        spec: InferenceSpec,
        policy: ExecutionPolicy,
    ) -> Result<XvlaPreparedInference> {
        Ok(XvlaPreparedInference {
            spec,
            policy,
            backend: Arc::clone(&self.backend),
            config: Arc::clone(&self.config),
            vlm: Arc::clone(&self.vlm),
            head: Arc::clone(&self.head),
            graphs: RefCell::new(XvlaGraphState::default()),
        })
    }
}

impl VlaRuntime for XvlaRuntime {
    fn model_variant(&self) -> Option<&'static str> {
        Some("bf16")
    }

    fn contract(&self) -> VlaContract {
        let profile = &self.config.profile;
        VlaContract {
            action_shape: [profile.action_horizon, self.config.action_space.dim_action],
            // This runtime accepts raw RGB only; the patch row count is not a
            // fixed part of its wire contract.
            patch_shape: [0, 3],
            max_token_len: profile.tokens,
            num_views: profile.num_views,
            image_size: profile.image_size,
            // Input pixels covered by one stage-4 DaViT token (strides 4,2,2,2).
            patch_size: profile.image_size
                / self.config.vision.feature_sides(profile.image_size)[3],
            accepts_rgb_u8: true,
        }
    }

    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        let spec = request.observation.inference_spec();
        let needs_plan = {
            let cache = self.prepared.borrow();
            cache.as_ref().map_or(true, |(cached, _)| *cached != spec)
        };
        if needs_plan {
            let prepared = self.build_prepared(spec, ExecutionPolicy::PreferGraph)?;
            drop(self.prepared.borrow_mut().replace((spec, prepared)));
        }
        let cache = self.prepared.borrow();
        let (_, prepared) = cache
            .as_ref()
            .ok_or_else(|| Error::Other("X-VLA prepared plan was not installed".into()))?;
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
        spec.validate()?;
        Ok(Box::new(self.build_prepared(*spec, policy)?))
    }

    fn execution_mode(&self) -> &'static str {
        match self
            .prepared
            .borrow()
            .as_ref()
            .map(|(_, prepared)| prepared.status())
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
        transfers::to_cpu(action.tensor())?.to_f32_vec()
    }
}

#[derive(Default)]
struct XvlaGraphState {
    plan: Option<XvlaGraphs>,
    fallback: Option<String>,
}

/// Fixed-shape plan that captures the three stages once per `(views, domain)`.
struct XvlaPreparedInference {
    spec: InferenceSpec,
    policy: ExecutionPolicy,
    backend: Arc<RuntimeBackend>,
    config: Arc<XvlaConfig>,
    vlm: Arc<VlmWeights>,
    head: Arc<ActionHeadWeights>,
    graphs: RefCell<XvlaGraphState>,
}

impl PreparedInference for XvlaPreparedInference {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }

    fn status(&self) -> PreparationStatus {
        let state = self.graphs.borrow();
        match (&state.plan, &state.fallback) {
            (Some(_), _) => PreparationStatus::Ready {
                mode: ExecutionMode::Graph,
                fallback_reason: None,
            },
            (None, Some(reason)) => PreparationStatus::Ready {
                mode: ExecutionMode::Eager,
                fallback_reason: Some(reason.clone()),
            },
            (None, None) => PreparationStatus::Ready {
                mode: ExecutionMode::Eager,
                fallback_reason: None,
            },
        }
    }

    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        if !self.spec.matches(request.observation) {
            return Err(Error::Other(format!(
                "prepared X-VLA spec {:?} does not match observation {:?}",
                self.spec,
                request.observation.inference_spec()
            )));
        }
        if self.policy == ExecutionPolicy::Eager {
            return run_xvla(&self.backend, &self.config, &self.vlm, &self.head, request);
        }

        let parts = request_parts(&self.config, request)?;
        let mut state = self.graphs.borrow_mut();
        if state.fallback.is_some() {
            return run_xvla(&self.backend, &self.config, &self.vlm, &self.head, request);
        }
        let reusable = state
            .plan
            .as_ref()
            .is_some_and(|plan| plan.views == parts.views && plan.domain == parts.domain);
        if !reusable {
            // Release the previous arena before reserving the replacement.
            state.plan = None;
            match XvlaGraphs::capture(&self.backend, &self.config, &self.vlm, &self.head, &parts) {
                Ok(plan) => state.plan = Some(plan),
                Err(error) => {
                    if self.policy == ExecutionPolicy::RequireGraph {
                        return Err(error);
                    }
                    eprintln!("[apxinf] X-VLA graph capture unavailable, using eager: {error}");
                    state.fallback = Some(error.to_string());
                    return run_xvla(&self.backend, &self.config, &self.vlm, &self.head, request);
                }
            }
        }
        let plan = state
            .plan
            .as_ref()
            .ok_or_else(|| Error::Other("X-VLA graph plan was not installed".into()))?;
        plan.update(&self.backend, &parts)?;
        plan.replay()?;
        Ok(Action::new(plan.buffers.action.clone()))
    }
}

/// Validated request surfaces shared by eager and captured execution.
struct RequestParts<'a, 'lat> {
    bytes: &'a [u8],
    token_ids: &'a [u32],
    state: &'a Tensor,
    views: usize,
    domain: usize,
    latent: InitialLatent<'lat>,
}

fn request_parts<'a, 'lat>(
    config: &XvlaConfig,
    request: &'a VlaRequest<'lat>,
) -> Result<RequestParts<'a, 'lat>> {
    let observation = request.observation;
    observation.validate()?;
    let profile = &config.profile;
    let side = profile.image_size;
    if observation.token_ids.len() != profile.tokens {
        return Err(Error::Other(format!(
            "X-VLA expects {} instruction tokens, got {}",
            profile.tokens,
            observation.token_ids.len()
        )));
    }
    let bytes = match &observation.vision {
        VisionObservation::RgbU8 { bytes, layout } => {
            if *layout != ImageLayout::Nhwc {
                return Err(Error::Other(
                    "X-VLA RGB input must be NHWC; transpose channels-first frames in the policy"
                        .into(),
                ));
            }
            bytes
        }
        VisionObservation::Patches(_) => {
            return Err(Error::Other(
                "X-VLA does not accept preprocessed patches; supply raw RGB frames".into(),
            ))
        }
    };
    let frame_bytes = side * side * 3;
    if bytes.is_empty() || bytes.len() % frame_bytes != 0 {
        return Err(Error::Other(format!(
            "X-VLA RGB input must contain whole {side}x{side}x3 frames, got {} bytes",
            bytes.len()
        )));
    }
    let views = bytes.len() / frame_bytes;
    if views > profile.num_views {
        return Err(Error::Other(format!(
            "X-VLA accepts at most {} camera views, got {views}",
            profile.num_views
        )));
    }
    let state = observation
        .state
        .as_ref()
        .ok_or_else(|| Error::Other("X-VLA requires proprioception (Observation.state)".into()))?;
    let domain = request.metadata.embodiment_id.ok_or_else(|| {
        Error::Other("X-VLA requires a domain id (VlaMetadata.embodiment_id)".into())
    })?;
    if domain >= config.head.num_domains {
        return Err(Error::Other(format!(
            "X-VLA domain {domain} is out of range (domains = {})",
            config.head.num_domains
        )));
    }
    Ok(RequestParts {
        bytes,
        token_ids: &observation.token_ids,
        state,
        views,
        domain,
        latent: request.initial_latent,
    })
}

fn run_xvla(
    backend: &RuntimeBackend,
    config: &XvlaConfig,
    vlm: &VlmWeights,
    head: &ActionHeadWeights,
    request: &VlaRequest<'_>,
) -> Result<Action> {
    let parts = request_parts(config, request)?;
    let ctx = backend.context();
    let profile = &config.profile;
    let side = profile.image_size;
    let frame_bytes = side * side * 3;
    let proprio = upload_bf16(
        ctx,
        parts.state,
        &[config.action_space.dim_proprio],
        "proprioception",
    )?;
    let latent_shape = [profile.action_horizon, config.action_space.dim_action];
    let latent = match parts.latent {
        InitialLatent::Provided(tensor) => {
            upload_bf16(ctx, tensor, &latent_shape, "initial latent")?
        }
        InitialLatent::Generate { rng } => {
            let elements = latent_shape[0] * latent_shape[1];
            let buffer = DeviceBuffer::alloc_on(ctx, elements * DType::BF16.size_in_bytes())
                .map_err(Error::Cuda)?;
            let noise = buffer
                .as_tensor(Shape::new(latent_shape.to_vec()), DType::BF16)
                .map_err(Error::Cuda)?;
            let mut generator = backend.create_normal_generator(noise.clone())?;
            generator.generate(rng)?;
            noise
        }
    };

    let mut features: Vec<Option<Tensor>> = Vec::with_capacity(profile.num_views);
    for index in 0..parts.views {
        let frame = &parts.bytes[index * frame_bytes..(index + 1) * frame_bytes];
        let raw = xvla_ops::upload_bytes(ctx, frame)?;
        let tokens = xvla_ops::rgb_u8_to_normalized_tokens_bf16(
            ctx,
            &raw,
            side,
            side,
            xvla_ops::XVLA_IMAGE_MEAN,
            xvla_ops::XVLA_IMAGE_STD,
        )?;
        features.push(Some(encode_image(
            ctx,
            &vlm.vision,
            &tokens,
            config.layer_norm_eps,
        )?));
    }
    while features.len() < profile.num_views {
        features.push(None);
    }
    let refs: Vec<Option<&Tensor>> = features.iter().map(Option::as_ref).collect();
    let output = encode_vlm(ctx, vlm, &refs, parts.token_ids, config.layer_norm_eps)?;
    let action = generate_actions(
        ctx,
        head,
        config,
        &output.vlm_features,
        &output.aux_visual_inputs,
        &proprio,
        parts.domain,
        &latent,
    )?;
    Ok(Action::new(action))
}

/// Fixed device buffers and one CUDA graph per pipeline stage.
struct XvlaGraphs {
    views: usize,
    domain: usize,
    buffers: StageBuffers,
    workspace: kernels::GraphWorkspace,
    vision: Box<dyn Graph>,
    language: Box<dyn Graph>,
    flow: Box<dyn Graph>,
    // Keep every fixed weight and the config referenced by the captured work.
    _fixed: (Arc<XvlaConfig>, Arc<VlmWeights>, Arc<ActionHeadWeights>),
}

struct StageBuffers {
    raw_rgb: DeviceBuffer,
    view_features: Vec<Tensor>,
    vlm_features: Tensor,
    aux_visual: Tensor,
    state: Tensor,
    noise: Tensor,
    action: Tensor,
    indices: xvla_ops::RowGatherPlan,
}

impl XvlaGraphs {
    fn capture(
        backend: &Arc<RuntimeBackend>,
        config: &Arc<XvlaConfig>,
        vlm: &Arc<VlmWeights>,
        head: &Arc<ActionHeadWeights>,
        parts: &RequestParts<'_, '_>,
    ) -> Result<Self> {
        let ctx = backend.context();
        let profile = &config.profile;
        let side = profile.image_size;
        let frame_bytes = side * side * 3;
        let hidden = config.head.hidden_size;
        let image_tokens = config.vision.feature_sides(side)[3].pow(2) + 1;
        let aux_rows = profile.tokens * (profile.num_views - 1);
        let latent_shape = [profile.action_horizon, config.action_space.dim_action];

        let raw_rgb =
            DeviceBuffer::alloc_on(ctx, parts.views * frame_bytes).map_err(Error::Cuda)?;
        raw_rgb.copy_from_host(parts.bytes).map_err(Error::Cuda)?;
        let view_features = (0..profile.num_views)
            .map(|_| static_tensor(ctx, &[image_tokens, hidden]))
            .collect::<Result<Vec<_>>>()?;
        let vlm_features = static_tensor(ctx, &[image_tokens + profile.tokens, hidden])?;
        let aux_visual = static_tensor(ctx, &[aux_rows, hidden])?;
        let state = static_tensor(ctx, &[config.action_space.dim_proprio])?;
        let noise = static_tensor(ctx, &latent_shape)?;
        let action = static_tensor(ctx, &latent_shape)?;

        let state_host = host_bf16(
            parts.state,
            &[config.action_space.dim_proprio],
            "proprioception",
        )?;
        transfers::copy_cpu_to_cuda(&state_host, &state)?;
        match parts.latent {
            InitialLatent::Provided(tensor) => {
                let latent_host = host_bf16(tensor, &latent_shape, "initial latent")?;
                transfers::copy_cpu_to_cuda(&latent_host, &noise)?;
            }
            InitialLatent::Generate { rng } => {
                let mut generator = backend.create_normal_generator(noise.clone())?;
                generator.generate(rng)?;
            }
        }
        let indices = xvla_ops::prepare_row_gather(
            ctx,
            parts.token_ids,
            vlm.text.embedding.shape().dims()[0],
        )?;

        let buffers = StageBuffers {
            raw_rgb,
            view_features,
            vlm_features,
            aux_visual,
            state,
            noise,
            action,
            indices,
        };

        let (workspace, vision, language, flow) = {
            let body = StageBody {
                ctx,
                config: config.as_ref(),
                vlm: vlm.as_ref(),
                head: head.as_ref(),
                views: parts.views,
                domain: parts.domain,
                buffers: &buffers,
            };
            let mut capacity = INITIAL_WORKSPACE_BYTES;
            let (workspace, peak) = loop {
                let workspace = kernels::GraphWorkspace::new(capacity, ctx.device_id())?;
                match warmup(backend, &workspace, &body) {
                    Ok(peak) => break (workspace, peak),
                    Err(error)
                        if is_workspace_exhaustion(&error) && capacity < MAX_WORKSPACE_BYTES =>
                    {
                        capacity = (capacity * 2).min(MAX_WORKSPACE_BYTES);
                    }
                    Err(error) => return Err(error),
                }
            };
            eprintln!(
                "[apxinf] X-VLA warmup arena: capacity={} MiB peak={} MiB",
                workspace.capacity() / (1024 * 1024),
                peak / (1024 * 1024)
            );
            let (vision, _) = backend
                .capture_graph(|| kernels::with_workspace(&workspace, || body.vision()))
                .map_err(|error| Error::Other(format!("X-VLA vision graph capture: {error}")))?;
            let (language, _) = backend
                .capture_graph(|| kernels::with_workspace(&workspace, || body.language()))
                .map_err(|error| Error::Other(format!("X-VLA language graph capture: {error}")))?;
            let (flow, _) = backend
                .capture_graph(|| kernels::with_workspace(&workspace, || body.action()))
                .map_err(|error| Error::Other(format!("X-VLA action graph capture: {error}")))?;
            (workspace, vision, language, flow)
        };

        let plan = Self {
            views: parts.views,
            domain: parts.domain,
            buffers,
            workspace,
            vision,
            language,
            flow,
            _fixed: (Arc::clone(config), Arc::clone(vlm), Arc::clone(head)),
        };
        plan.vision.replay()?;
        plan.language.replay()?;
        plan.flow.replay()?;
        backend.synchronize()?;
        eprintln!(
            "[apxinf] X-VLA captured CUDA graphs: views={} domain={} arena={} MiB used={} MiB",
            plan.views,
            plan.domain,
            plan.workspace.capacity() / (1024 * 1024),
            plan.workspace.used() / (1024 * 1024)
        );
        Ok(plan)
    }

    fn update(&self, backend: &RuntimeBackend, parts: &RequestParts<'_, '_>) -> Result<()> {
        backend.synchronize()?;
        self.buffers
            .raw_rgb
            .copy_from_host(parts.bytes)
            .map_err(Error::Cuda)?;

        let state_dims = self.buffers.state.shape().dims().to_vec();
        let state_host = host_bf16(parts.state, &state_dims, "proprioception")?;
        transfers::copy_cpu_to_cuda(&state_host, &self.buffers.state)?;

        let latent_dims = self.buffers.noise.shape().dims().to_vec();
        match parts.latent {
            InitialLatent::Provided(tensor) => {
                let latent_host = host_bf16(tensor, &latent_dims, "initial latent")?;
                transfers::copy_cpu_to_cuda(&latent_host, &self.buffers.noise)?;
            }
            InitialLatent::Generate { rng } => {
                let mut generator = backend.create_normal_generator(self.buffers.noise.clone())?;
                generator.generate(rng)?;
            }
        }
        self.buffers.indices.update_rows(parts.token_ids)?;
        Ok(())
    }

    fn replay(&self) -> Result<()> {
        self.vision.replay()?;
        self.language.replay()?;
        self.flow.replay()?;
        Ok(())
    }
}

/// Borrowed view of the captured buffers and weights used inside a capture.
struct StageBody<'a> {
    ctx: &'a Context,
    config: &'a XvlaConfig,
    vlm: &'a VlmWeights,
    head: &'a ActionHeadWeights,
    views: usize,
    domain: usize,
    buffers: &'a StageBuffers,
}

impl StageBody<'_> {
    fn vision(&self) -> Result<()> {
        let side = self.config.profile.image_size;
        let frame_bytes = side * side * 3;
        for view in 0..self.views {
            let raw = self
                .buffers
                .raw_rgb
                .view(view * frame_bytes, frame_bytes)
                .map_err(Error::Cuda)?;
            let tokens = xvla_ops::rgb_u8_to_normalized_tokens_bf16(
                self.ctx,
                &raw,
                side,
                side,
                xvla_ops::XVLA_IMAGE_MEAN,
                xvla_ops::XVLA_IMAGE_STD,
            )?;
            let encoded = encode_image(
                self.ctx,
                &self.vlm.vision,
                &tokens,
                self.config.layer_norm_eps,
            )?;
            copy_into(&self.buffers.view_features[view], &encoded, self.ctx)?;
        }
        Ok(())
    }

    fn language(&self) -> Result<()> {
        let refs: Vec<Option<&Tensor>> = (0..self.config.profile.num_views)
            .map(|index| {
                if index < self.views {
                    Some(&self.buffers.view_features[index])
                } else {
                    None
                }
            })
            .collect();
        let merged = encode_vlm_with_plan(
            self.ctx,
            self.vlm,
            &refs,
            &self.buffers.indices,
            self.config.profile.tokens,
            self.config.layer_norm_eps,
        )?;
        copy_into(&self.buffers.vlm_features, &merged.vlm_features, self.ctx)?;
        copy_into(
            &self.buffers.aux_visual,
            &merged.aux_visual_inputs,
            self.ctx,
        )?;
        Ok(())
    }

    fn action(&self) -> Result<()> {
        let output = generate_actions(
            self.ctx,
            self.head,
            self.config,
            &self.buffers.vlm_features,
            &self.buffers.aux_visual,
            &self.buffers.state,
            self.domain,
            &self.buffers.noise,
        )?;
        copy_into(&self.buffers.action, &output, self.ctx)
    }
}

fn warmup(
    backend: &RuntimeBackend,
    workspace: &kernels::GraphWorkspace,
    body: &StageBody<'_>,
) -> Result<usize> {
    backend.synchronize()?;
    let mut peak = 0;
    for stage in 0..3 {
        let mut stable = false;
        for _ in 0..4 {
            let generation = backend.context().tuning().generation();
            kernels::prepare_with_workspace(workspace, || match stage {
                0 => body.vision(),
                1 => body.language(),
                _ => body.action(),
            })?;
            backend.synchronize()?;
            peak = peak.max(workspace.used());
            if backend.context().tuning().generation() == generation {
                stable = true;
                break;
            }
        }
        if !stable {
            return Err(Error::Other(
                "X-VLA GEMM tactics did not stabilize before graph capture".into(),
            ));
        }
    }
    Ok(peak)
}

fn is_workspace_exhaustion(error: &Error) -> bool {
    error.to_string().contains("workspace exhausted")
}

fn static_tensor(ctx: &Context, dims: &[usize]) -> Result<Tensor> {
    let elements: usize = dims.iter().product();
    let bytes = elements
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("X-VLA graph buffer size overflow".into()))?;
    let buffer = DeviceBuffer::alloc_on(ctx, bytes).map_err(Error::Cuda)?;
    buffer
        .as_tensor(Shape::new(dims.to_vec()), DType::BF16)
        .map_err(Error::Cuda)
}

fn copy_into(destination: &Tensor, source: &Tensor, ctx: &Context) -> Result<()> {
    if destination.dtype() != source.dtype() || destination.shape().dims() != source.shape().dims()
    {
        return Err(Error::Other(format!(
            "X-VLA graph stage copy mismatch: {} {:?} -> {} {:?}",
            source.dtype(),
            source.shape().dims(),
            destination.dtype(),
            destination.shape().dims()
        )));
    }
    let source_buffer = DeviceBuffer::from_tensor(source).map_err(Error::Cuda)?;
    let destination_buffer = DeviceBuffer::from_tensor(destination).map_err(Error::Cuda)?;
    destination_buffer
        .copy_from_device_async(&source_buffer, source.size_in_bytes(), ctx.stream())
        .map_err(Error::Cuda)
}

/// Convert one host float tensor of `dims` into a CPU BF16 tensor.
fn host_bf16(tensor: &Tensor, dims: &[usize], label: &str) -> Result<Tensor> {
    if tensor.device() != Device::Cpu {
        return Err(Error::Other(format!("X-VLA {label} must be a CPU tensor")));
    }
    if tensor.shape().dims() != dims {
        return Err(Error::Other(format!(
            "X-VLA {label} shape {:?} does not match {dims:?}",
            tensor.shape().dims()
        )));
    }
    match tensor.dtype() {
        DType::BF16 => Ok(tensor.clone()),
        DType::F32 | DType::F16 => {
            let rounded: Vec<bf16> = tensor
                .to_f32_vec()?
                .into_iter()
                .map(bf16::from_f32)
                .collect();
            Tensor::from_bf16(dims.to_vec(), &rounded)
        }
        dtype => Err(Error::Other(format!(
            "X-VLA {label} must be f32, f16 or bf16, got {dtype}"
        ))),
    }
}

/// Upload one host float tensor of `dims` as device BF16 storage.
fn upload_bf16(ctx: &Context, tensor: &Tensor, dims: &[usize], label: &str) -> Result<Tensor> {
    let host = host_bf16(tensor, dims, label)?;
    transfers::to_cuda(&host, ctx.device_id())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use apxinf_core::{Device, Shape, Tensor};

    use super::load_runtime;
    use crate::vla::{
        ImageLayout, Observation, VisionObservation, VlaMetadata, VlaRequest, VlaRuntime,
    };

    fn read_f32(path: &Path) -> Vec<f32> {
        let bytes = std::fs::read(path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn read_i64(path: &Path) -> Vec<i64> {
        let bytes = std::fs::read(path)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
        bytes
            .chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn pixels_to_rgb_u8(pixels: &[f32], side: usize) -> Vec<u8> {
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];
        let mut out = Vec::with_capacity(side * side * 3);
        for position in 0..side * side {
            for channel in 0..3 {
                let value = pixels[channel * side * side + position] * std[channel] + mean[channel];
                out.push((value * 255.0).round().clamp(0.0, 255.0) as u8);
            }
        }
        out
    }

    /// Full runtime binding: RGB frames -> normalize -> DaViT -> BART -> flow.
    /// The oracle stores normalized fp32 pixels; this harness quantizes them
    /// back to u8, so the gate is wider than the head-only parity test.
    #[test]
    #[ignore = "GPU parity harness: set XVLA_RUNTIME_CASE"]
    fn runtime_matches_oracle() {
        let case = std::env::var("XVLA_RUNTIME_CASE").expect("XVLA_RUNTIME_CASE is required");
        let case = Path::new(&case);
        let tolerance: f32 = std::env::var("XVLA_RUNTIME_TOL")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0.02);
        let checkpoint = std::env::var("XVLA_CKPT_DIR").unwrap_or_else(|_| {
            format!(
                "{}/refs/checkpoints/X-VLA-Libero",
                std::env::var("HOME").expect("HOME")
            )
        });
        let backend = crate::accelerator::create_backend(Device::Cuda(0)).unwrap();
        let runtime = load_runtime(Path::new(&checkpoint), backend).unwrap();
        let side = runtime.contract().image_size;

        let mask = std::fs::read(case.join("view_mask.u8")).unwrap();
        let mut rgb = Vec::new();
        let mut valid = 0usize;
        for present in mask {
            if present != 0 {
                let pixels = read_f32(&case.join(format!("pixel_{valid}.f32")));
                rgb.extend(pixels_to_rgb_u8(&pixels, side));
                valid += 1;
            }
        }
        let ids: Vec<u32> = read_i64(&case.join("input_ids.i64"))
            .into_iter()
            .map(|id| u32::try_from(id).unwrap())
            .collect();
        let proprio =
            Tensor::from_f32(Shape::new(vec![20]), &read_f32(&case.join("proprio.f32"))).unwrap();
        let x1 =
            Tensor::from_f32(Shape::new(vec![30, 20]), &read_f32(&case.join("x1.f32"))).unwrap();
        let domain = read_i64(&case.join("domain_id.i64"))[0] as usize;
        let observation = Observation {
            vision: VisionObservation::RgbU8 {
                bytes: rgb,
                layout: ImageLayout::Nhwc,
            },
            token_ids: ids,
            state: Some(proprio),
            action_mask: None,
        };
        let metadata = VlaMetadata {
            embodiment_id: Some(domain),
            ..VlaMetadata::default()
        };
        let request = VlaRequest::provided_with_metadata(&observation, &x1, metadata);
        let actual = runtime.infer_host_f32(&request).unwrap();
        let expected = read_f32(&case.join("final_action.f32"));
        assert_eq!(actual.len(), expected.len());
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        let mut dot = 0.0f64;
        let mut actual_norm = 0.0f64;
        let mut expected_norm = 0.0f64;
        for (actual, expected) in actual.iter().zip(&expected) {
            let difference = (actual - expected).abs();
            max_abs = max_abs.max(difference);
            sum_abs += difference as f64;
            dot += (*actual as f64) * (*expected as f64);
            actual_norm += (*actual as f64) * (*actual as f64);
            expected_norm += (*expected as f64) * (*expected as f64);
        }
        let cosine = dot / (actual_norm.sqrt() * expected_norm.sqrt());
        println!(
            "X-VLA runtime_action parity: max_abs={max_abs:.6} mean_abs={:.6} cosine={cosine:.8}",
            sum_abs / actual.len() as f64
        );
        assert!(
            max_abs <= tolerance,
            "X-VLA runtime parity failed: max_abs={max_abs} > {tolerance}"
        );
        assert!(cosine >= 0.999, "X-VLA runtime cosine {cosine} below 0.999");
    }
}
