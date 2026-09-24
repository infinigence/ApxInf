//! Preparation and retained execution resources: the whole-model direct
//! planning plan, and the inherited opt-in per-GDN-layer decode graphs.
use super::runner::{validate, Validated};
use crate::qwen_drive::backend::{kernels, nvtx, transfers, tuning, Context, DeviceBuffer};
use crate::qwen_drive::model::{
    DirectExecution, DirectInputs, GdnExecution, GdnRequest, PlanningState, QwenDriveModel,
    VisionState,
};
use crate::vla::{
    Action, ExecutionMode, ExecutionPolicy, InferenceSpec, InitialLatent, PreparationStatus,
    PreparedInference, VlaRequest,
};
use apxinf_core::{
    Backend, DType, Device, Error, Graph, NormalGenerator, Result, SamplingBackend, Shape, Tensor,
};
use std::{cell::RefCell, rc::Rc, sync::Arc};
const DECODE_GRAPH_ARENA_BYTES: usize = 64 * 1024 * 1024;
const DECODE_GRAPH_WARMUP_STEPS: usize = 2;
fn persistent_tensor(ctx: &Context, shape: &[usize], dtype: DType) -> Result<Tensor> {
    let elements: usize = shape.iter().product();
    let bytes = elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| Error::Other("qwen_drive: persistent tensor size overflow".into()))?;
    DeviceBuffer::alloc(bytes.max(1), ctx.device_id())
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape.to_vec()), dtype)
        .map_err(Error::Cuda)
}
fn device_copy(ctx: &Context, destination: &Tensor, source: &Tensor, bytes: usize) -> Result<()> {
    let dst = DeviceBuffer::from_tensor(destination).map_err(Error::Cuda)?;
    let src = DeviceBuffer::from_tensor(source).map_err(Error::Cuda)?;
    dst.copy_from_device_async(&src, bytes, ctx.stream())
        .map_err(Error::Cuda)
}
fn decode_graph_layers() -> Option<Option<usize>> {
    static SETTING: std::sync::OnceLock<Option<Option<usize>>> = std::sync::OnceLock::new();
    *SETTING.get_or_init(|| match std::env::var("APXINF_QWEN_DECODE_GRAPH") {
        Err(_) => None,
        Ok(value) => match value.trim().parse::<usize>() {
            Ok(layer) => Some(Some(layer)),
            Err(_) => Some(None),
        },
    })
}
fn decode_graph_enabled_for(layer_idx: usize) -> bool {
    match decode_graph_layers() {
        None => false,
        Some(None) => true,
        Some(Some(only)) => only == layer_idx,
    }
}
#[derive(Default)]
pub(super) struct GdnGraphs {
    /// Captured GDN decode graphs, by layer and by the conv double-buffer
    /// parity the capture baked in.
    gdn_graphs: Vec<[Option<Box<dyn apxinf_core::Graph>>; 2]>,
    /// Whether one eager pass has already run under the arena for this layer
    /// and parity. cuBLASLt plans bind buffer addresses when they are prepared,
    /// and `may_prepare_native_resources` reports false once a workspace is
    /// bound unless the pass is a declared preflight, so a capture taken before
    /// that preflight records kernels set up against the driver-allocated
    /// buffers the eager steps used.
    gdn_graph_prepared: Vec<[bool; 2]>,
    /// Where a captured layer leaves its result: a workspace view, valid only
    /// until the arena is reused, so it is copied out after every replay.
    gdn_graph_out: Vec<[Option<Tensor>; 2]>,
    /// Fixed-address input a captured layer reads, and the per-layer landing
    /// buffer a replayed result is copied into.
    gdn_graph_in: Option<Tensor>,
    gdn_graph_result: Vec<Option<Tensor>>,
    /// Stable-address arena the captured bodies allocate from.
    gdn_graph_workspace: Option<kernels::GraphWorkspace>,
}
impl GdnGraphs {
    fn forward_gdn_captured(
        &mut self,
        request: &GdnRequest<'_>,
        eager: &mut dyn FnMut(Tensor) -> Result<Tensor>,
        x: Tensor,
        layer_idx: usize,
        parity: usize,
    ) -> Result<(Tensor, bool)> {
        let cuda = Arc::clone(request.backend);
        let ctx = cuda.context();
        let dims = x.shape().dims().to_vec();
        let bytes = x.numel() * DType::BF16.size_in_bytes();

        if self.gdn_graphs.len() != request.layer_count {
            self.gdn_graphs = (0..request.layer_count).map(|_| [None, None]).collect();
            self.gdn_graph_out = (0..request.layer_count).map(|_| [None, None]).collect();
            self.gdn_graph_result = vec![None; request.layer_count];
            self.gdn_graph_prepared = vec![[false, false]; request.layer_count];
        }
        if self.gdn_graph_workspace.is_none() {
            self.gdn_graph_workspace = Some(kernels::GraphWorkspace::new(
                DECODE_GRAPH_ARENA_BYTES,
                ctx.device_id(),
            )?);
        }
        // Diagnostic arm: take the arena and then run the body exactly as the
        // eager path does, with no workspace bound, no staging, no capture. The
        // only difference from a clean run is that a 64MB block now exists.
        // Every hypothesis that blames what the arena path *does* predicts this
        // is correct; a layout hypothesis -- some kernel writing past the end of
        // a buffer whose neighbour this allocation changed -- predicts it is
        // wrong, and that would explain a full-attention layer going bad while
        // the GDN layer that uses the arena stays exact.
        if std::env::var("APXINF_QWEN_DECODE_GRAPH_ARENA_ONLY")
            .map(|value| value.contains("alloc"))
            .unwrap_or(false)
        {
            return eager(x).map(|x| (x, false));
        }

        if self.gdn_graph_in.is_none() {
            self.gdn_graph_in = Some(persistent_tensor(ctx, &dims, DType::BF16)?);
        }
        if self.gdn_graph_result[layer_idx].is_none() {
            self.gdn_graph_result[layer_idx] = Some(persistent_tensor(ctx, &dims, DType::BF16)?);
        }

        // The capture reads this address, so every step stages its input here.
        let staged = self.gdn_graph_in.clone().expect("staging input");
        device_copy(ctx, &staged, &x, bytes)?;

        // Diagnostic arm: run the body inside the arena with no capture and no
        // replay. If this is already wrong, the fault is in the workspace
        // allocation path rather than in anything CUDA graphs do.
        if std::env::var_os("APXINF_QWEN_DECODE_GRAPH_ARENA_ONLY").is_some() {
            // `raw` drops the staging copies as well, which separates the arena
            // itself from the copies in and out of it.
            let raw = std::env::var("APXINF_QWEN_DECODE_GRAPH_ARENA_ONLY")
                .map(|value| value.contains("raw"))
                .unwrap_or(false);
            let input = if raw { x.clone() } else { staged.clone() };
            let prepare = std::env::var("APXINF_QWEN_DECODE_GRAPH_ARENA_ONLY")
                .map(|value| value.contains("prepare"))
                .unwrap_or(false);
            let workspace = self.gdn_graph_workspace.take().expect("graph arena");
            let produced = if prepare {
                kernels::prepare_with_workspace(&workspace, || eager(input))
            } else {
                kernels::with_workspace(&workspace, || eager(input))
            };
            self.gdn_graph_workspace = Some(workspace);
            let output = produced?;
            if raw {
                return Ok((output, false));
            }
            let landing = self.gdn_graph_result[layer_idx]
                .clone()
                .expect("landing buffer");
            device_copy(ctx, &landing, &output, bytes)?;
            return Ok((landing, false));
        }

        if !self.gdn_graph_prepared[layer_idx][parity] {
            // Declared preflight: executes, so its output is this step's real
            // result and the recurrent and conv state advance exactly once.
            let workspace = self.gdn_graph_workspace.take().expect("graph arena");
            let prepared = kernels::prepare_with_workspace(&workspace, || eager(staged.clone()));
            self.gdn_graph_workspace = Some(workspace);
            let output = prepared?;
            cuda.synchronize()?;
            self.gdn_graph_prepared[layer_idx][parity] = true;
            let landing = self.gdn_graph_result[layer_idx]
                .clone()
                .expect("landing buffer");
            device_copy(ctx, &landing, &output, bytes)?;
            return Ok((landing, false));
        }

        // A capture executes the Rust body, so the host-side conv flip advances
        // on that step and must not be advanced again after the replay below.
        let mut captured_now = false;
        if self.gdn_graphs[layer_idx][parity].is_none() {
            captured_now = true;
            cuda.synchronize()?;
            let workspace = self.gdn_graph_workspace.take().expect("graph arena");
            let captured = cuda
                .capture_graph(|| kernels::with_workspace(&workspace, || eager(staged.clone())));
            self.gdn_graph_workspace = Some(workspace);
            let (graph, output) = captured?;
            self.gdn_graph_out[layer_idx][parity] = Some(output);
            self.gdn_graphs[layer_idx][parity] = Some(graph);
        }

        self.gdn_graphs[layer_idx][parity]
            .as_ref()
            .expect("captured graph")
            .replay()?;

        // The captured output lives in the arena, which the next captured layer
        // reuses, so it is copied into this layer's own buffer before returning.
        let produced = self.gdn_graph_out[layer_idx][parity]
            .clone()
            .expect("captured output");
        let landing = self.gdn_graph_result[layer_idx]
            .clone()
            .expect("landing buffer");
        device_copy(ctx, &landing, &produced, bytes)?;
        Ok((landing, !captured_now))
    }
}
impl GdnExecution for GdnGraphs {
    fn run(
        &mut self,
        request: &GdnRequest<'_>,
        input: Tensor,
        eager: &mut dyn FnMut(Tensor) -> Result<Tensor>,
    ) -> Result<(Tensor, bool)> {
        if !request.decode
            || request.decode_step < DECODE_GRAPH_WARMUP_STEPS
            || !decode_graph_enabled_for(request.layer)
        {
            return eager(input).map(|output| (output, false));
        }
        self.forward_gdn_captured(request, eager, input, request.layer, request.parity)
    }
}

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
    steps: usize,
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
            && self.pixel_dtype == valid.pixels.dtype()
            && self.inputs.pixels.shape() == valid.pixels.shape()
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
        let inputs = model.prepare_direct_inputs(
            &mut state,
            &vision,
            &sample.observation.token_ids,
            valid.pixels,
            valid.grids,
            valid.cond,
            valid.steps,
        )?;
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
        let pixel_dtype = valid.pixels.dtype();
        let grids = valid.grids.to_vec();
        let steps = valid.steps;
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
            steps,
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
                plan.model
                    .forward_direct(&plan.inputs, &mut mutable.state, &mut preflight)?;
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
        copy(valid.pixels, &self.inputs.pixels)?;
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

/// Real-checkpoint graph contract. Fixtures are private canonical policy inputs.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        qwen_drive::backend::RuntimeBackend,
        vla::{Observation, PlanningOptions, VisionObservation, VlaMetadata},
        LoadOptions, LoadedModel,
    };
    use apxinf_core::RngKey;
    use std::path::Path;

    fn floats(path: &Path) -> Vec<f32> {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect()
    }
    fn host(action: Action) -> Vec<f32> {
        transfers::to_cpu(action.tensor())
            .unwrap()
            .to_f32_vec()
            .unwrap()
    }
    #[test]
    #[ignore = "requires CUDA, APXINF_QWEN_DRIVE_TEST_MODEL and APXINF_QWEN_DRIVE_TEST_INPUTS"]
    fn whole_direct_graph_rebinds_inputs_and_owns_its_lifetime() {
        let checkpoint = std::env::var("APXINF_QWEN_DRIVE_TEST_MODEL").unwrap();
        let fixture = std::env::var("APXINF_QWEN_DRIVE_TEST_INPUTS").unwrap();
        let fixture = Path::new(&fixture);
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(fixture.join("meta.json")).unwrap()).unwrap();
        let grids: Vec<[u32; 3]> = serde_json::from_value(meta["grids"].clone()).unwrap();
        let shape: Vec<usize> = serde_json::from_value(meta["pixels_shape"].clone()).unwrap();
        let tokens: Vec<u32> = std::fs::read(fixture.join("tokens.bin"))
            .unwrap()
            .chunks_exact(4)
            .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        let conditioning = floats(&fixture.join("conditioning.bin"));
        let pixels = floats(&fixture.join("pixels.bin"));
        let noise_values = floats(&fixture.join("noise.bin"));
        let noise = Tensor::from_f32(vec![50, 3], &noise_values).unwrap();
        let changed_noise = Tensor::from_f32(
            vec![50, 3],
            &noise_values.iter().map(|x| -x).collect::<Vec<_>>(),
        )
        .unwrap();
        let original = Observation {
            vision: VisionObservation::Patches(Tensor::from_f32(shape.clone(), &pixels).unwrap()),
            token_ids: tokens,
            state: Some(Tensor::from_f32(vec![conditioning.len()], &conditioning).unwrap()),
            action_mask: None,
        };
        let mut images = original.clone();
        images.vision = VisionObservation::Patches(
            Tensor::from_f32(shape, &pixels.iter().map(|x| x * 0.5).collect::<Vec<_>>()).unwrap(),
        );
        let mut text = original.clone();
        // Change a plain-text token while retaining all image positions and length.
        let slot = text.token_ids.iter_mut().find(|id| **id < 100000).unwrap();
        *slot = if *slot == 42 { 43 } else { 42 };
        let mut state = original.clone();
        let mut changed_conditioning = conditioning;
        changed_conditioning[0] += 0.25;
        state.state = Some(
            Tensor::from_f32(vec![changed_conditioning.len()], &changed_conditioning).unwrap(),
        );
        let options = PlanningOptions {
            num_steps: Some(10),
            reasoning: None,
        };
        let metadata = VlaMetadata {
            image_grid_thw: Some(&grids),
            planning: Some(&options),
            ..Default::default()
        };
        let request = VlaRequest::provided_with_metadata(&original, &noise, metadata);
        let backend = Arc::new(RuntimeBackend::new(0).unwrap());
        let loaded = crate::qwen_drive::load::load_registered(
            Path::new(&checkpoint),
            Device::Cuda(0),
            backend.clone(),
            &LoadOptions::default(),
        )
        .unwrap();
        let LoadedModel::Vla(runner) = loaded else {
            panic!("VLA required")
        };
        let observations = [&original, &images, &text, &state, &original, &original];
        let noises = [&noise, &noise, &noise, &noise, &changed_noise, &noise];
        let eager = runner
            .prepare_for(&request, ExecutionPolicy::Eager)
            .unwrap();
        let reference: Vec<_> = observations
            .iter()
            .zip(noises)
            .map(|(obs, latent)| {
                host(
                    eager
                        .run(&VlaRequest::provided_with_metadata(obs, latent, metadata))
                        .unwrap(),
                )
            })
            .collect();
        for (index, result) in reference[1..5].iter().enumerate() {
            assert_ne!(
                result,
                &reference[0],
                "fixture change {} must affect the output",
                index + 1
            );
        }
        assert_eq!(reference[0], reference[5]);
        let generated_reference = host(
            eager
                .run(&VlaRequest::generated_with_metadata(
                    &original,
                    RngKey::default(),
                    metadata,
                ))
                .unwrap(),
        );
        drop(eager);
        let graph = runner
            .prepare_for(&request, ExecutionPolicy::RequireGraph)
            .unwrap();
        assert_eq!(
            graph.status(),
            PreparationStatus::Ready {
                mode: ExecutionMode::Graph,
                fallback_reason: None
            }
        );
        for (index, ((obs, latent), expected)) in
            observations.iter().zip(noises).zip(&reference).enumerate()
        {
            let actual = host(
                graph
                    .run(&VlaRequest::provided_with_metadata(obs, latent, metadata))
                    .unwrap(),
            );
            assert_eq!(&actual, expected, "changed-input case {index}");
        }
        // Same length is insufficient: changed image placement/geometry or steps
        // must reject without overwriting or evicting a healthy explicit plan.
        let mut invalid = original.clone();
        invalid.token_ids.pop();
        assert!(graph
            .run(&VlaRequest::provided_with_metadata(
                &invalid, &noise, metadata
            ))
            .is_err());
        let different_steps = PlanningOptions {
            num_steps: Some(4),
            reasoning: None,
        };
        let different_meta = VlaMetadata {
            planning: Some(&different_steps),
            ..metadata
        };
        assert!(graph
            .run(&VlaRequest::provided_with_metadata(
                &original,
                &noise,
                different_meta
            ))
            .is_err());
        assert_eq!(host(graph.run(&request).unwrap()), reference[0]);
        let rng = RngKey::default();
        let generated = VlaRequest::generated_with_metadata(&original, rng, metadata);
        let first = host(graph.run(&generated).unwrap());
        assert_eq!(first, generated_reference);
        assert_eq!(host(graph.run(&generated).unwrap()), first);
        for changed_key in [
            RngKey::new(1, 0, 0),
            RngKey::new(0, 1, 0),
            RngKey::new(0, 0, 1),
        ] {
            let changed = VlaRequest::generated_with_metadata(&original, changed_key, metadata);
            assert_ne!(host(graph.run(&changed).unwrap()), first);
        }
        assert_eq!(host(graph.run(&generated).unwrap()), first);
        assert_eq!(host(graph.run(&request).unwrap()), reference[0]);
        let generation = backend.context().tuning().generation();
        runner.clear_prepared().unwrap();
        drop(runner);
        // Explicit plan owns weights/state even after the originating runner drops.
        assert_eq!(host(graph.run(&request).unwrap()), reference[0]);
        assert_eq!(backend.context().tuning().generation(), generation);
        backend
            .context()
            .install_tuning(tuning::TuningSession::inference(
                tuning::TacticStore::default(),
            ))
            .unwrap();
        assert_eq!(graph.status(), PreparationStatus::Invalidated);
        assert!(graph.run(&request).is_err());
    }
}
