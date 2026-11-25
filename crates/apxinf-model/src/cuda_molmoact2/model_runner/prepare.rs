//! Resource preparation, the raw eager execution of one request, and the prepared plan (graph or eager).
//!
//! * Raw eager (`ModelRunner::run`): everything belongs to the request and is created for it, so nothing
//!   from a previous observation can leak into the next. There is no `ExecutionSession`: every native
//!   GEMM/attention execution is created and destroyed per call and each GEMM synchronizes. The only
//!   persistent resource is the BF16 noise buffer the device-side generator fills. This is an internal
//!   path (the tuning pre-run and the reference check of the tests), not the execution mode of `infer`.
//! * Prepared (`Plan`): one fixed token structure (image layout, prefill segments, number of visible
//!   context rows). The plan owns every device buffer the computation touches: the uint8 image buffer,
//!   the BF16 noise buffer (+ its F32 staging buffer for host-supplied noise), the token / context-row
//!   tables, the F32 output, and an `ExecutionSession` arena that backs every intermediate tensor and
//!   keeps the prepared native executions. Normally the whole
//!   `vision -> connector -> VLM -> adapter -> 10 flow steps -> publish` traversal is captured as ONE
//!   CUDA graph; each request rewrites the buffer contents in place and replays it. A plan built with
//!   `ExecutionPolicy::Eager` (and the fallback plan of `PreferGraph`) is *prepared eager*: the same
//!   buffers and session, no capture; each request rewrites the buffers and the host enqueues the same
//!   traversal on the prepared executions, with one synchronization at the end. The traversal is
//!   `MolmoAct2Model::forward`, the same function the raw eager path calls.
use crate::cuda_molmoact2::backend::ops::{prepare_with_session, with_session, ExecutionSession};
use crate::cuda_molmoact2::backend::sampling::create_normal_generator;
use crate::cuda_molmoact2::backend::{capture, quantization, CapturedGraph, CudaBuffer, CudaContext, QuantizationArgs, QuantizationSemantic};
use crate::cuda_molmoact2::config::MolmoAct2Config;
use crate::cuda_molmoact2::model::{HostPlan, MolmoAct2Model, RequestPlan};
use crate::vla::*;
use apxinf_core::{DType, Error, NormalGenerator, Result, Shape, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

use super::runner::{invalid, ModelRunner};

/// Persistent per-runner state of the eager path: the BF16 `[horizon, width]` noise the generator writes.
pub(super) struct Resources {
    noise: Tensor,
    generator: Box<dyn NormalGenerator>,
}

fn alloc_tensor(ctx: &CudaContext, shape: Vec<usize>, dtype: DType) -> Result<Tensor> {
    let bytes = shape.iter().product::<usize>() * dtype.size_in_bytes();
    CudaBuffer::alloc(bytes, ctx.device_id())
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape), dtype)
        .map_err(Error::Cuda)
}

fn alloc_zeroed_tensor(ctx: &CudaContext, shape: Vec<usize>, dtype: DType) -> Result<Tensor> {
    let bytes = shape.iter().product::<usize>() * dtype.size_in_bytes();
    CudaBuffer::alloc_zeros(bytes, ctx.device_id())
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape), dtype)
        .map_err(Error::Cuda)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

impl Resources {
    pub fn new(ctx: &CudaContext, cfg: &MolmoAct2Config) -> Result<Self> {
        let ex = &cfg.expert;
        let noise = alloc_tensor(ctx, vec![ex.max_action_horizon, ex.max_action_dim], DType::BF16)?;
        let generator = create_normal_generator(ctx, noise.clone())?;
        Ok(Self { noise, generator })
    }
}

impl ModelRunner {
    /// Raw eager forward pass for one request (no plan, no session; see the module comment).
    pub(in crate::cuda_molmoact2) fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        self.validate_request(request)?;
        let observation = request.observation;
        let VisionObservation::RgbU8 { bytes, .. } = &observation.vision else {
            return Err(invalid("RGB input required"));
        };
        let (horizon, width) = (self.model.config().expert.max_action_horizon, self.model.config().expert.max_action_dim);

        // Request inputs: images, then the token-derived plan, then the noise.
        let rgb = CudaBuffer::alloc(bytes.len(), self.ctx.device_id()).map_err(Error::Cuda)?;
        rgb.copy_from_host_on(bytes, self.ctx.stream()).map_err(Error::Cuda)?;
        let plan = self.model.plan(&observation.token_ids)?;
        let noise = match request.initial_latent {
            InitialLatent::Provided(noise) => {
                let device = apxinf_cuda_new::transfers::to_cuda(
                    &Tensor::from_f32(vec![horizon, width], noise.as_f32()?)?,
                    self.ctx.device_id(),
                )?;
                self.model.noise_to_bf16(&device)?
            }
            InitialLatent::Generate { rng } => {
                // The generator fills the runner's buffer on the context stream; the flow reads it on the
                // same stream and every request ends with a synchronization, so one buffer is safe.
                let mut resources = self.resources.try_borrow_mut().map_err(|_| invalid("runner is already running"))?;
                resources.generator.generate(rng)?;
                resources.noise.clone()
            }
        };

        // Publish independent device storage: the result never aliases a scratch buffer.
        let mut out = alloc_tensor(&self.ctx, vec![horizon, width], DType::F32)?;
        self.model.forward(&rgb, &plan, &noise, &mut out)?;
        self.ctx.synchronize().map_err(Error::Cuda)?;
        Ok(Action::new(out))
    }
}

// ----------------------------------------------------------------------------- prepared plan

/// Upper bound of the probe arena that measures the traversal's peak (the real arena is sized from it).
const PROBE_ARENA_BYTES: usize = 3 << 30;
const MIB: usize = 1 << 20;

#[cfg(test)]
thread_local! {
    static INJECT_CAPTURE_FAILURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Make every following capture fail inside its capture scope (tests of the failure paths).
#[cfg(test)]
pub(in crate::cuda_molmoact2) fn inject_capture_failure(on: bool) {
    INJECT_CAPTURE_FAILURE.with(|flag| flag.set(on));
}

/// One prepared profile: fixed token structure, stable buffers, a workspace arena and the prepared
/// executions, plus (normally) the captured graph. `run` binds a request's inputs in place and executes
/// (graph replay, or the host-enqueued traversal when there is no graph); it never captures, tunes or
/// allocates workspace.
pub(in crate::cuda_molmoact2) struct Plan {
    // Dropped first: the graph references every buffer below.
    graph: Option<CapturedGraph>,
    session: ExecutionSession,
    plan: RequestPlan,
    structure: HostPlan,
    rgb: CudaBuffer,
    /// BF16 `[horizon, width]` noise the flow reads (graph input).
    noise: Tensor,
    /// F32 `[horizon, width]` device staging for host-supplied noise.
    noise_stage: Tensor,
    generator: RefCell<Box<dyn NormalGenerator>>,
    /// F32 `[horizon, width]` normalized padded actions (graph output).
    out: Tensor,
    spec: InferenceSpec,
    model: Rc<MolmoAct2Model>,
    ctx: Rc<CudaContext>,
    fallback_reason: Option<String>,
}

impl Plan {
    /// Build a plan for the structure of `sample` (which must already be validated). The sample only
    /// provides the token structure; no image or noise value of it is retained.
    pub fn build(
        model: Rc<MolmoAct2Model>,
        ctx: Rc<CudaContext>,
        sample: &VlaRequest<'_>,
        host: HostPlan,
        policy: ExecutionPolicy,
    ) -> Result<Self> {
        let cfg = model.config();
        let (horizon, width) = (cfg.expert.max_action_horizon, cfg.expert.max_action_dim);
        let size = cfg.vit.image_size;
        let device = ctx.device_id();
        let plan = model.plan_from(&host)?;
        let rgb = CudaBuffer::alloc_zeros(crate::cuda_molmoact2::geometry::NUM_VIEWS * size * size * 3, device).map_err(Error::Cuda)?;
        let noise = alloc_zeroed_tensor(&ctx, vec![horizon, width], DType::BF16)?;
        let noise_stage = alloc_zeroed_tensor(&ctx, vec![horizon, width], DType::F32)?;
        let generator = create_normal_generator(&ctx, noise.clone())?;
        let out = alloc_zeroed_tensor(&ctx, vec![horizon, width], DType::F32)?;

        let traverse = || -> Result<()> {
            let mut out = out.clone();
            model.forward(&rgb, &plan, &noise, &mut out)
        };

        // 1. Warm-up traversal in a large probe arena: it prepares nothing that is kept, it measures the
        //    arena the real traversal needs (the arena is deterministic: same calls, same allocations).
        let probe = ExecutionSession::with_capacity(PROBE_ARENA_BYTES, device)?;
        prepare_with_session(&probe, traverse)?;
        ctx.synchronize().map_err(Error::Cuda)?;
        let high_water = probe.used();
        drop(probe);
        let arena_bytes = (high_water + high_water / 8 + 16 * MIB).next_multiple_of(MIB);

        // 2. The real arena: prepare (native GEMM / attention executions) then capture the same traversal.
        let session = ExecutionSession::with_capacity(arena_bytes, device)?;
        prepare_with_session(&session, traverse)?;
        ctx.synchronize().map_err(Error::Cuda)?;

        // The captured traversal; tests can make it fail to exercise the capture-failure paths.
        let capture_body = || {
            with_session(&session, || {
                #[cfg(test)]
                if INJECT_CAPTURE_FAILURE.with(std::cell::Cell::get) {
                    return Err(Error::Other("injected capture failure".into()));
                }
                traverse()
            })
        };
        let (graph, fallback_reason) = match policy {
            ExecutionPolicy::Eager => (None, None),
            ExecutionPolicy::RequireGraph => (Some(capture(&ctx, capture_body)?), None),
            ExecutionPolicy::PreferGraph => match capture(&ctx, capture_body) {
                Ok(graph) => (Some(graph), None),
                Err(error) => {
                    eprintln!("[apxinf] MolmoAct2 graph capture unavailable, using eager: {error}");
                    (None, Some(error.to_string()))
                }
            },
        };
        ctx.synchronize().map_err(Error::Cuda)?;
        let spec = sample.observation.inference_spec();
        Ok(Self {
            graph,
            session,
            plan,
            structure: host,
            rgb,
            noise,
            noise_stage,
            generator: RefCell::new(generator),
            out,
            spec,
            model,
            ctx,
            fallback_reason,
        })
    }

    pub fn matches(&self, host: &HostPlan) -> bool {
        self.structure.same_structure(host)
    }

    pub fn is_graph(&self) -> bool {
        self.graph.is_some()
    }

    /// Bind the request's inputs into the plan's stable buffers and execute. `host` is the analysis of
    /// this request's tokens; a different structure is rejected before any buffer is touched.
    pub fn run_host(&self, request: &VlaRequest<'_>, host: &HostPlan) -> Result<Action> {
        let observation = request.observation;
        let VisionObservation::RgbU8 { bytes, .. } = &observation.vision else {
            return Err(invalid("RGB input required"));
        };
        if !self.structure.same_structure(host) {
            return Err(invalid("the prompt tokens do not have the image layout / context size this plan was prepared for; prepare again"));
        }
        let stream = self.ctx.stream();
        // Inputs are free: every run ends with a synchronization. Re-synchronize anyway so a plan can never
        // overwrite a buffer a previous (possibly failed) run still reads.
        self.ctx.synchronize().map_err(Error::Cuda)?;
        self.model.update_plan(&self.plan, host)?;
        self.rgb.copy_from_host_on(bytes, stream).map_err(Error::Cuda)?;
        match request.initial_latent {
            InitialLatent::Provided(noise) => {
                CudaBuffer::from_tensor(&self.noise_stage)
                    .map_err(Error::Cuda)?
                    .copy_from_host_on(&f32_bytes(noise.as_f32()?), stream)
                    .map_err(Error::Cuda)?;
                let mut bf16 = self.noise.clone();
                quantization(&self.ctx, QuantizationArgs::new(QuantizationSemantic::CastF32ToBf16, &self.noise_stage, &mut bf16))?;
            }
            InitialLatent::Generate { rng } => {
                self.generator.try_borrow_mut().map_err(|_| invalid("plan is already running"))?.generate(rng)?;
            }
        }
        match &self.graph {
            Some(graph) => graph.replay()?,
            None => {
                let mut out = self.out.clone();
                with_session(&self.session, || self.model.forward(&self.rgb, &self.plan, &self.noise, &mut out))?;
            }
        }
        // The graph output buffer is reused by the next run: publish an independent copy.
        let cfg = self.model.config();
        let (horizon, width) = (cfg.expert.max_action_horizon, cfg.expert.max_action_dim);
        let fresh = alloc_tensor(&self.ctx, vec![horizon, width], DType::F32)?;
        let dst = CudaBuffer::from_tensor(&fresh).map_err(Error::Cuda)?;
        let src = CudaBuffer::from_tensor(&self.out).map_err(Error::Cuda)?;
        dst.copy_from_device_async(&src, horizon * width * 4, stream).map_err(Error::Cuda)?;
        self.ctx.synchronize().map_err(Error::Cuda)?;
        Ok(Action::new(fresh))
    }
}

impl PreparedInference for Plan {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }

    fn status(&self) -> PreparationStatus {
        PreparationStatus::Ready {
            mode: if self.graph.is_some() { ExecutionMode::Graph } else { ExecutionMode::Eager },
            fallback_reason: self.fallback_reason.clone(),
        }
    }

    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        let observation = request.observation;
        observation.validate()?;
        if !self.spec.matches(observation) {
            return Err(invalid(&format!("prepared spec {:?} does not match observation {:?}", self.spec, observation.inference_spec())));
        }
        super::runner::validate_request(self.model.config(), request)?;
        let host = self.model.analyze(&observation.token_ids)?;
        self.run_host(request, &host)
    }
}
