//! Owning whole-model direct-planning plan: prepare once, bind and replay.
use super::runner::{rgb_bf16_lut, validate, Validated, ValidatedVision};
use crate::qwen_drive::{
    backend::{kernels, nvtx, transfers, tuning, DeviceBuffer},
    model::{
        blocks::direct::RawRgbInput, DirectExecution, DirectInputs, PlanningState, QwenDriveModel,
        VisionState,
    },
};
use crate::vla::{
    Action, ExecutionMode, ExecutionPolicy, InferenceSpec, InitialLatent, PreparationStatus,
    PreparedInference, VlaRequest,
};
use apxinf_core::{
    Backend, DType, Device, Error, Graph, NormalGenerator, Result, SamplingBackend, Tensor,
};
use std::{cell::RefCell, rc::Rc, sync::Arc};

struct WorkspaceExecution<'a> {
    workspace: &'a kernels::GraphWorkspace,
    preparing: bool,
    usage: Vec<usize>,
}
impl DirectExecution for WorkspaceExecution<'_> {
    fn run(&mut self, operation: &mut dyn FnMut() -> Result<Tensor>) -> Result<Tensor> {
        let result = if self.preparing {
            kernels::prepare_with_workspace(self.workspace, operation)
        } else {
            kernels::with_workspace(self.workspace, operation)
        };
        self.usage.push(self.workspace.used());
        result
    }
}
struct Mutable {
    state: PlanningState,
    generator: Box<dyn NormalGenerator>,
}
/// Graph output aliases plan storage until the next run. Plans are serial and
/// retain weights, stable inputs, KV/state, workspace and native execution choices.
pub(super) struct DirectPlan {
    // Drop executable before any captured allocation or model resource.
    graph: Option<Box<dyn Graph>>,
    output: Tensor,
    inputs: DirectInputs,
    workspace: kernels::GraphWorkspace,
    noise_workspace: kernels::GraphWorkspace,
    mutable: RefCell<Mutable>,
    model: Rc<QwenDriveModel>,
    spec: InferenceSpec,
    grids: Vec<[u32; 3]>,
    image_positions: Vec<bool>,
    pixel_dtype: DType,
    rgb_input: bool,
    raw_resize: Option<Vec<crate::vla::RawRgbResizeFrame>>,
    resize_algorithm: &'static str,
    steps: usize,
    prompt_len: usize,
    tuning: Arc<tuning::TuningSession>,
    generation: u64,
    fallback: Option<String>,
}
impl DirectPlan {
    pub(super) fn is_current(&self) -> bool {
        let current = self.model.backend().context().tuning();
        Arc::ptr_eq(&current, &self.tuning) && current.generation() == self.generation
    }
    pub(super) fn matches(&self, request: &VlaRequest<'_>, valid: &Validated<'_>) -> bool {
        self.spec.matches(request.observation)
            && request
                .metadata
                .planning
                .and_then(|p| p.reasoning.as_ref())
                .is_none()
            && self.grids == valid.grids
            && self.steps == valid.steps
            && self.prompt_len == valid.prompt_len
            && self.pixel_dtype == valid.input_dtype()
            && self.rgb_input == matches!(valid.vision, ValidatedVision::Rgb(_))
            && self.raw_resize.as_deref() == valid.raw_resize
            && self.resize_algorithm == kernels::pillow_bicubic::PILLOW_RESAMPLE_VERSION
            && self.inputs.pixels.shape().dims() == [valid.rows, valid.width]
            && self
                .image_positions
                .iter()
                .zip(&request.observation.token_ids)
                .all(|(&image, &id)| image == (id == self.model.config().image_token_id))
    }
    pub(super) fn execution_mode(&self) -> &'static str {
        if !self.is_current() {
            "invalidated"
        } else if self.graph.is_some() {
            "graph"
        } else if self.fallback.is_some() {
            "eager-fallback"
        } else {
            "eager"
        }
    }
    pub(super) fn prepare(
        model: Rc<QwenDriveModel>,
        sample: &VlaRequest<'_>,
        policy: ExecutionPolicy,
    ) -> Result<Self> {
        let _prepare = nvtx::range("qwen_drive/prepare");
        let valid = validate(&model, sample)?;
        if sample
            .metadata
            .planning
            .and_then(|p| p.reasoning.as_ref())
            .is_some()
        {
            return Err(Error::Other("qwen_drive whole-model preparation currently requires direct planning; variable-length reasoning uses eager inference".into()));
        }
        if crate::qwen_drive::diagnostics_enabled()
            || std::env::var_os("APXINF_QWEN_TRACE_DIR").is_some()
        {
            return Err(Error::Other(
                "qwen_drive preparation requires host tensor diagnostics to be disabled".into(),
            ));
        }
        let backend = model.backend().clone();
        let ctx = backend.context();
        let vision = Rc::new(VisionState::default());
        let mut state = model.new_state(vision.clone())?;
        // The BF16 placeholder only supplies shape during one-time preparation.
        // Replay binds the actual RGB bytes into stable device storage.
        let sample_pixels = match valid.vision {
            ValidatedVision::Patches(pixels) => (*pixels).clone(),
            ValidatedVision::Rgb(_) => Tensor::zeros(vec![valid.rows, valid.width], DType::BF16),
        };
        let rgb_frames = valid.rgb_frames();
        let pixel_dtype = valid.input_dtype();
        let rgb_input = matches!(valid.vision, ValidatedVision::Rgb(_));
        let raw_resize = valid.raw_resize.map(|frames| frames.to_vec());
        let grids = valid.grids.to_vec();
        let steps = valid.steps;
        let resize_frames = valid.resize_frames();
        let mut inputs = model.prepare_direct_inputs(
            &mut state,
            &vision,
            &sample.observation.token_ids,
            valid.prompt_len,
            &sample_pixels,
            valid.grids,
            valid.cond,
            valid.steps,
        )?;
        if let ValidatedVision::Rgb(bytes) = valid.vision {
            let resize = resize_frames
                .map(|frames| kernels::pillow_bicubic::PillowBicubicRgbPlan::new(ctx, &frames))
                .transpose()?;
            let rgb_bytes = if let Some(plan) = &resize {
                plan.raw().clone()
            } else {
                DeviceBuffer::alloc(bytes.len(), ctx.device_id()).map_err(Error::Cuda)?
            };
            inputs.rgb = Some(RawRgbInput {
                bytes: rgb_bytes,
                lut: rgb_bf16_lut(ctx)?,
                frames: rgb_frames,
                resize,
            });
        }
        let generator = backend.create_normal_generator(inputs.noise.clone())?;
        // One arena is reused sequentially for Vision, Language and Action inside
        // the same graph. Only persistent visual features and shared KV cross phases.
        let gib = std::env::var("APXINF_QWEN_GRAPH_WORKSPACE_GIB")
            .ok()
            .map(|s| s.parse::<usize>())
            .transpose()
            .map_err(|_| Error::Other("invalid APXINF_QWEN_GRAPH_WORKSPACE_GIB".into()))?
            .unwrap_or(32);
        let bytes = gib
            .checked_mul(1 << 30)
            .filter(|&n| n > 0)
            .ok_or_else(|| Error::Other("qwen_drive graph workspace capacity overflow".into()))?;
        let workspace = kernels::GraphWorkspace::new(bytes, ctx.device_id())?;
        let noise_workspace =
            kernels::GraphWorkspace::new(inputs.noise.numel() * 4 + 256, ctx.device_id())?;
        let spec = sample.observation.inference_spec();
        let image_positions = sample
            .observation
            .token_ids
            .iter()
            .map(|&id| id == model.config().image_token_id)
            .collect();
        let mut plan = Self {
            graph: None,
            output: inputs.noise.clone(),
            inputs,
            workspace,
            noise_workspace,
            mutable: RefCell::new(Mutable { state, generator }),
            spec,
            grids,
            image_positions,
            pixel_dtype,
            rgb_input,
            raw_resize,
            resize_algorithm: kernels::pillow_bicubic::PILLOW_RESAMPLE_VERSION,
            steps,
            prompt_len: valid.prompt_len,
            tuning: ctx.tuning().clone(),
            generation: ctx.tuning().generation(),
            fallback: None,
            model,
        };
        tuning::without_autotune(|| -> Result<()> {
            let valid = validate(&plan.model, sample)?;
            let mut mutable = plan.mutable.borrow_mut();
            plan.bind(sample, &valid, &mut mutable)?;
            // Resolve address-dependent native resources before capture; both
            // traversals reset the arena at identical phase boundaries.
            let mut preflight = WorkspaceExecution {
                workspace: &plan.workspace,
                preparing: true,
                usage: Vec::new(),
            };
            let output =
                match plan
                    .model
                    .forward_direct(&plan.inputs, &mut mutable.state, &mut preflight)
                {
                    Ok(output) => output,
                    Err(error) => {
                        // A partial RGB frame sequence can have queued work.
                        backend.synchronize()?;
                        return Err(error);
                    }
                };
            eprintln!(
                "qwen_drive prepared workspace: reserved={} bytes, phase_used={:?}",
                plan.workspace.capacity(),
                preflight.usage
            );
            backend.synchronize()?;
            plan.output = output;
            if policy != ExecutionPolicy::Eager {
                match backend.capture_graph(|| {
                    plan.model.forward_direct(
                        &plan.inputs,
                        &mut mutable.state,
                        &mut WorkspaceExecution {
                            workspace: &plan.workspace,
                            preparing: false,
                            usage: Vec::new(),
                        },
                    )
                }) {
                    Ok((graph, output)) => {
                        // Complete first-launch setup before reporting Ready.
                        graph.replay()?;
                        backend.synchronize()?;
                        plan.graph = Some(graph);
                        plan.output = output;
                    }
                    Err(error) if policy == ExecutionPolicy::PreferGraph => {
                        plan.fallback = Some(error.to_string());
                        eprintln!("qwen_drive whole-model capture fell back to eager: {error}");
                    }
                    Err(error) => return Err(error),
                }
            }
            Ok(())
        })?;
        plan.tuning = ctx.tuning();
        plan.generation = plan.tuning.generation();
        Ok(plan)
    }
    fn bind(
        &self,
        request: &VlaRequest<'_>,
        valid: &Validated<'_>,
        mutable: &mut Mutable,
    ) -> Result<()> {
        let ctx = self.model.backend().context();
        // Public boundary: host writes must not race a previous asynchronous run.
        ctx.synchronize().map_err(Error::Cuda)?;
        let copy = |source: &Tensor, destination: &Tensor| -> Result<()> {
            if source.device() == Device::Cpu {
                transfers::copy_cpu_to_cuda(source, destination)
            } else if source.device() == Device::Cuda(ctx.device_id()) {
                DeviceBuffer::from_tensor(destination)
                    .map_err(Error::Cuda)?
                    .copy_from_device_async(
                        &DeviceBuffer::from_tensor(source).map_err(Error::Cuda)?,
                        destination.numel() * destination.dtype().size_in_bytes(),
                        ctx.stream(),
                    )
                    .map_err(Error::Cuda)
            } else {
                Err(Error::Other("qwen_drive input is on another device".into()))
            }
        };
        match valid.vision {
            ValidatedVision::Patches(pixels) => copy(pixels, &self.inputs.pixels)?,
            ValidatedVision::Rgb(bytes) => self
                .inputs
                .rgb
                .as_ref()
                .ok_or_else(|| Error::Other("qwen_drive RGB plan has no raw input".into()))?
                .bytes
                .copy_from_host(bytes)
                .map_err(Error::Cuda)?,
        }
        let tokens: Vec<u8> = request
            .observation
            .token_ids
            .iter()
            .flat_map(|id| id.to_ne_bytes())
            .collect();
        self.inputs
            .tokens
            .copy_from_host(&tokens)
            .map_err(Error::Cuda)?;
        self.inputs
            .update_conditioning(self.model.config(), &valid.cond)?;
        match request.initial_latent {
            InitialLatent::Provided(noise) => copy(
                &noise.reshape(self.inputs.noise.shape().dims().to_vec())?,
                &self.inputs.noise,
            )?,
            InitialLatent::Generate { rng } => {
                mutable.generator.generate(rng)?;
                if self.model.config().noise_init_std != 1.0 {
                    let scaled = kernels::with_workspace(&self.noise_workspace, || {
                        kernels::elementwise::scale(
                            ctx,
                            &self.inputs.noise,
                            self.model.config().noise_init_std,
                        )
                    })?;
                    copy(&scaled, &self.inputs.noise)?;
                }
            }
        }
        Ok(())
    }
}
impl PreparedInference for DirectPlan {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }
    fn status(&self) -> PreparationStatus {
        if !self.is_current() {
            return PreparationStatus::Invalidated;
        }
        PreparationStatus::Ready {
            mode: if self.graph.is_some() {
                ExecutionMode::Graph
            } else {
                ExecutionMode::Eager
            },
            fallback_reason: self.fallback.clone(),
        }
    }
    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        if !self.is_current() {
            return Err(Error::Other(
                "qwen_drive prepared plan is stale after a tactic update".into(),
            ));
        }
        let valid = validate(&self.model, request)?;
        if !self.matches(request, &valid) {
            return Err(Error::Other("qwen_drive prepared profile mismatch; prepare again for changed shape, image-token positions, grids or steps".into()));
        }
        let mut mutable = self
            .mutable
            .try_borrow_mut()
            .map_err(|_| Error::Other("qwen_drive prepared plan is already executing".into()))?;
        tuning::without_autotune(|| {
            {
                let _bind = nvtx::range("qwen_drive/bind");
                self.bind(request, &valid, &mut mutable)?;
            }
            if let Some(graph) = &self.graph {
                let _replay = nvtx::range("qwen_drive/graph/replay_launch");
                graph.replay()?;
                Ok(Action::new(self.output.clone()))
            } else {
                self.model
                    .forward_direct(
                        &self.inputs,
                        &mut mutable.state,
                        &mut WorkspaceExecution {
                            workspace: &self.workspace,
                            preparing: false,
                            usage: Vec::new(),
                        },
                    )
                    .map(Action::new)
            }
        })
    }
}

#[cfg(test)]
#[path = "direct_tests.rs"]
mod tests;
