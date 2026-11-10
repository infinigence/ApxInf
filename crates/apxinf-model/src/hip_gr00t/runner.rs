//! GR00T N1.7 on HIP as a [`VlaRuntime`].
//!
//! A prepared plan owns the fixed device inputs, the per-shape host
//! preparation and one activation arena; each run refills the inputs and
//! reuses the rest.
//!
//! **Why `prepare` refuses.** The portable [`InferenceSpec`] carries only a
//! token count and an image layout. GR00T's allocations also depend on
//! `image_grid_thw` (how many patch rows, and the per-frame segments the
//! vision attention batches over) and on `embodiment_id` (which adapters are
//! uploaded). A plan built from a spec alone would bind neither, so
//! [`VlaRuntime::prepare`] returns an error naming what is missing and
//! [`VlaRuntime::prepare_for`] -- which receives a whole request, metadata
//! included -- is the supported entry point.
//!
//! Execution is eager. Inputs sit at fixed addresses and every activation
//! comes from the arena in a fixed order, which are the preconditions for
//! capturing a run as one HIP graph; until that lands, a request for a graph
//! is refused rather than silently served eagerly.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::OnceLock;

use apxinf_core::{DType, Device, Error, Result, Tensor};
use apxinf_hip::graph::HipGraph;
use apxinf_hip::native::{self, HipArena};
use half::bf16;

use super::geometry::{build_backbone_token_groups, Gr00tBackboneTokenGroups};
use super::model::HipGr00tModel;
use super::vision::PreparedVision;
use super::backbone::PreparedBackbone;
use super::weights::Embodiment;
use crate::vla::{
    Action, ExecutionMode, ExecutionPolicy, InferenceSpec, InitialLatent, PreparationStatus,
    PreparedInference, VisionObservation, VlaContract, VlaRequest, VlaRuntime,
};

/// Everything about a request that changes an allocation or a dispatch shape.
///
/// This is the plan cache key, and the reason it is not [`InferenceSpec`]: two
/// requests with the same token count but different grids need different
/// plans, and serving one from the other's plan would be wrong rather than
/// slow.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PlanKey {
    patches: usize,
    grid: Vec<[u32; 3]>,
    token_count: usize,
    embodiment_id: usize,
    /// Row groups depend on which tokens are image placeholders, which the
    /// mask and the token IDs decide together.
    groups: Gr00tBackboneTokenGroups,
}

pub struct HipGr00tRunner {
    model: Rc<HipGr00tModel>,
    /// The implicit plan `infer` keeps, for the most recent key only: an arena
    /// is large, so a replacement is built after the old one is gone.
    prepared: RefCell<Option<(PlanKey, Rc<HipGr00tPrepared>)>>,
}

impl HipGr00tRunner {
    pub fn new(model: Rc<HipGr00tModel>) -> Self {
        Self {
            model,
            prepared: RefCell::new(None),
        }
    }

    pub fn model(&self) -> &Rc<HipGr00tModel> {
        &self.model
    }

    /// The cached plan, for a benchmark that wants to report what it did.
    ///
    /// Not part of `VlaRuntime`: a policy has no business reaching inside the
    /// plan, and a report does.
    pub fn prepared_for_report(&self) -> Option<Rc<HipGr00tPrepared>> {
        self.prepared.borrow().as_ref().map(|(_, plan)| Rc::clone(plan))
    }

    fn read_request(&self, request: &VlaRequest<'_>) -> Result<(PlanKey, Inputs)> {
        read_request(&self.model, request)
    }

    /// Build a concretely typed plan from a representative request.
    ///
    /// [`VlaRuntime::prepare_for`] wraps this in a `Box<dyn
    /// PreparedInference>`. Benches and tests that need family-specific
    /// detail -- [`HipGr00tPrepared::arena_usage`], for one -- use this
    /// instead of widening the portable trait with a GR00T concern.
    pub fn prepare_plan(
        &self,
        request: &VlaRequest<'_>,
        policy: ExecutionPolicy,
    ) -> Result<HipGr00tPrepared> {
        let (key, _) = self.read_request(request)?;
        self.build(key, request, policy)
    }

    fn build(
        &self,
        key: PlanKey,
        request: &VlaRequest<'_>,
        policy: ExecutionPolicy,
    ) -> Result<HipGr00tPrepared> {
        if policy == ExecutionPolicy::RequireGraph {
            return Err(Error::UnsupportedOp("GR00T HIP graph capture"));
        }
        HipGr00tPrepared::new(Rc::clone(&self.model), key, &request.observation.token_ids)
    }
}

/// Everything the request must carry, validated once, as the plan key plus
/// the host tensors to upload.
///
/// A free function rather than a method: a prepared plan has to re-derive the
/// key to check a request against itself, and it holds the model but no runner.
fn read_request(model: &HipGr00tModel, request: &VlaRequest<'_>) -> Result<(PlanKey, Inputs)> {
    let observation = request.observation;
    observation.validate()?;
    let config = model.config();
    let backbone = model.backbone_config();

    let pixel_values = match &observation.vision {
        VisionObservation::Patches(tensor) => tensor,
        VisionObservation::RgbU8 { .. } => {
            return Err(Error::Other(
                "GR00T on HIP expects pixel_values from the checkpoint's official processor, \
                 not raw RGB"
                    .into(),
            ))
        }
    };
    let patch_width = backbone.vision.in_channels
        * backbone.vision.temporal_patch_size
        * backbone.vision.patch_size
        * backbone.vision.patch_size;
    let dims = pixel_values.shape().dims();
    if dims.len() != 2 || dims[1] != patch_width {
        return Err(Error::ShapeMismatch {
            expected: format!("[patches, {patch_width}]"),
            got: pixel_values.shape().to_string(),
        });
    }
    let patches = dims[0];

    let grid = request
        .metadata
        .image_grid_thw
        .ok_or(Error::Contract("GR00T requires VlaMetadata.image_grid_thw"))?;
    let attention_mask = request
        .metadata
        .attention_mask
        .ok_or(Error::Contract("GR00T requires VlaMetadata.attention_mask"))?;
    let embodiment_id = request
        .metadata
        .embodiment_id
        .ok_or(Error::Contract("GR00T requires VlaMetadata.embodiment_id"))?;
    let groups = build_backbone_token_groups(
        &observation.token_ids,
        attention_mask,
        backbone.image_token_id,
    )?;

    let state = observation.state.as_ref().ok_or(Error::Contract(
        "GR00T requires a normalized, padded state tensor",
    ))?;
    let state = host_bf16(state, vec![1, config.state_input_dim()?], "state")?;
    let noise = match request.initial_latent {
        InitialLatent::Provided(latent) => latent,
        InitialLatent::Generate { .. } => {
            return Err(Error::Contract(
                "GR00T requires processor-owned initial noise; pass an explicit latent",
            ))
        }
    };
    let noise = host_bf16(
        noise,
        vec![config.action_horizon, config.max_action_dim],
        "initial latent",
    )?;

    Ok((
        PlanKey {
            patches,
            grid: grid.to_vec(),
            token_count: observation.token_ids.len(),
            embodiment_id,
            groups,
        },
        Inputs {
            pixel_values: host_bf16(pixel_values, vec![patches, patch_width], "pixel_values")?,
            state,
            noise,
            token_ids: observation.token_ids.clone(),
        },
    ))
}

/// Host values for one request. Everything here is refilled on every run;
/// anything that is *not* here is a function of the plan key.
struct Inputs {
    pixel_values: Tensor,
    state: Tensor,
    noise: Tensor,
    /// Carried per request because the plan key holds only the token *count*
    /// and the placeholder layout -- two prompts can share a plan.
    token_ids: Vec<u32>,
}

/// One fully bound plan: fixed device inputs, the per-shape host preparation,
/// the selected embodiment, and the activation arena.
pub struct HipGr00tPrepared {
    model: Rc<HipGr00tModel>,
    key: PlanKey,
    /// Derived for the trait's benefit; [`PlanKey`] is what `run` checks.
    spec: InferenceSpec,
    arena: HipArena,
    pixel_values: Tensor,
    state: Tensor,
    noise: Tensor,
    vision_plan: PreparedVision,
    backbone_plan: PreparedBackbone,
    /// The DiT's cross-attention row groups, uploaded once.
    ///
    /// These are a function of the plan key, so building them here rather than
    /// in the forward path keeps the forward free of device allocations and
    /// host-to-device copies -- the two things that would make it
    /// uncapturable, and two allocations plus two copies per inference either
    /// way.
    rows: super::action_head::DitRowIndices,
    embodiment: Rc<Embodiment>,
    /// The forward, captured as one launch, and the output tensor it writes.
    ///
    /// Built on a later `run_bound` rather than in `new`: capture records the
    /// addresses the arena hands out, so it has to observe a real run of this
    /// plan's shapes. Holding the output tensor alongside the graph is what
    /// makes the replay usable -- the replay writes the same arena slot, and
    /// nothing else returns it.
    ///
    /// `None` when capture is off or when it failed. A failure is not an
    /// inference failure: the eager path stays correct and stays in use.
    graph: RefCell<Option<(HipGraph, Tensor)>>,
    /// Inferences served by replaying the graph.
    ///
    /// Capture failing, or the switch being off, both leave the eager path in
    /// use and the right answer returned -- so without a count, a measurement
    /// that never reached the graph is indistinguishable from one that found
    /// no gain. The same reasoning as the GEMM and attention provider counts.
    replays: std::cell::Cell<usize>,
}

impl HipGr00tPrepared {
    fn new(model: Rc<HipGr00tModel>, key: PlanKey, token_ids: &[u32]) -> Result<Self> {
        let config = model.config();
        let backbone = model.backbone_config();
        let ctx = model.context();

        // Rebuild the mask the groups imply, so preparation does not need the
        // caller to hand the same mask twice.
        let mut attention_mask = vec![0u8; key.token_count];
        for index in key.groups.non_image.iter().chain(&key.groups.image) {
            attention_mask[*index] = 1;
        }
        let (vision_plan, backbone_plan) =
            model.prepare_shape(key.patches, &key.grid, token_ids, &attention_mask)?;
        let embodiment = model.embodiment(key.embodiment_id)?;

        let patch_width = backbone.vision.in_channels
            * backbone.vision.temporal_patch_size
            * backbone.vision.patch_size
            * backbone.vision.patch_size;
        let zeros = |dims: Vec<usize>| ctx.to_device(&Tensor::zeros(dims, DType::BF16));
        let arena = HipArena::new(ctx, model.arena_bytes(key.patches, key.token_count))?;
        let rows =
            super::action_head::DitRowIndices::new(ctx, &key.groups.non_image, &key.groups.image)?;
        Ok(Self {
            spec: InferenceSpec {
                token_count: key.token_count,
                image_layout: None,
            },
            pixel_values: zeros(vec![key.patches, patch_width])?,
            state: zeros(vec![1, config.state_input_dim()?])?,
            noise: zeros(vec![config.action_horizon, config.max_action_dim])?,
            model,
            key,
            arena,
            vision_plan,
            backbone_plan,
            rows,
            embodiment,
            graph: RefCell::new(None),
            replays: std::cell::Cell::new(0),
        })
    }

    fn run_bound(&self, inputs: &Inputs) -> Result<Action> {
        let ctx = self.model.context();
        // Inputs are uploaded outside any graph: they come from host tensors
        // the caller rebuilds per request, so a captured copy would replay
        // from a pointer that no longer exists. The device buffers they land
        // in are fixed, which is what lets the forward be captured at all.
        native::upload(ctx, &inputs.pixel_values, &self.pixel_values)?;
        native::upload(ctx, &inputs.state, &self.state)?;
        native::upload(ctx, &inputs.noise, &self.noise)?;
        self.backbone_plan.refill_tokens(&inputs.token_ids)?;
        self.arena.reset();

        if let Some((graph, output)) = self.graph.borrow().as_ref() {
            self.replays.set(self.replays.get() + 1);
            graph.launch(ctx)?;
            return Ok(Action::new(output.clone()));
        }

        let actions = self.model.infer(
            &self.arena,
            &self.pixel_values,
            &self.state,
            &self.noise,
            &self.vision_plan,
            &self.backbone_plan,
            &self.embodiment,
            &self.rows,
        )?;

        // Capture on the next run rather than this one. The first run of a
        // plan also does whatever first-touch work the libraries defer --
        // hipBLASLt's solution map, CK's module load -- and recording that
        // into the graph would replay it forever.
        if graph_capture_enabled() && self.graph.borrow().is_none() {
            self.capture(inputs)?;
        }
        Ok(Action::new(actions))
    }

    /// Record this plan's forward, so later runs replay it as one launch.
    ///
    /// Runs the forward once more under capture. That run's result is
    /// discarded -- the caller already has the answer from the eager run that
    /// preceded it -- but the arena slot it writes is the slot every replay
    /// writes, so it is kept alongside the graph.
    fn capture(&self, inputs: &Inputs) -> Result<()> {
        let ctx = self.model.context();
        // The eager run above left work queued against the arena; capture must
        // not interleave with it.
        ctx.synchronize()?;
        self.arena.reset();

        // SAFETY: the captured region allocates nothing on the device (the
        // arena hands out slices of one allocation, and the DiT row indices
        // moved to preparation), copies nothing from host memory, and neither
        // synchronizes nor downloads. Every replay is preceded by
        // `arena.reset()` in `run_bound`, so the addresses the graph records
        // are the addresses it replays against.
        let captured = unsafe {
            HipGraph::capture(ctx, || {
                self.model.infer(
                    &self.arena,
                    &self.pixel_values,
                    &self.state,
                    &self.noise,
                    &self.vision_plan,
                    &self.backbone_plan,
                    &self.embodiment,
                    &self.rows,
                )
            })
        };
        match captured {
            Ok((graph, output)) => {
                *self.graph.borrow_mut() = Some((graph, output));
                Ok(())
            }
            // Returned as an error, not swallowed.
            //
            // Two reasons. First, a capture that fails means the forward did
            // something a fixed profile must not -- allocated on the device,
            // copied from host memory rebuilt per call, or synchronized -- and
            // `CLAUDE.md` calls that unfinished work rather than debt.
            //
            // Second, and decisive: continuing is not actually available. The
            // failure that broke the capture latches an error on the stream,
            // and every later call on it reports "a previous error during
            // capture" instead of its own status. Clearing the thread-local
            // last error is not enough; the stream itself stays unusable. An
            // earlier version of this reported the problem and returned `Ok`,
            // and the next four inferences failed with an error that named
            // neither the cause nor the capture.
            Err(error) => Err(Error::Other(format!(
                "apxinf-hip: the GR00T forward did not capture ({error}). A fixed profile \
                 must be device-resident from input upload to output transfer: something \
                 in the forward allocates on the device, copies from host memory that is \
                 rebuilt per call, or synchronizes. Set APXINF_HIP_GRAPH=0 to run eagerly \
                 while that is fixed."
            ))),
        }
    }
}

/// Whether to capture, from `APXINF_HIP_GRAPH`. On by default.
///
/// Capture is not here for speed. Measured over six paired runs it is worth
/// -0.03 +/- 0.11 ms, which is nothing: the host issues 1759 launches in about
/// 3 ms against 74 ms of GPU work, so it never falls behind and there is no
/// gap for a graph to close.
///
/// It is here because capturing is an executable assertion that the forward is
/// device-resident, which `CLAUDE.md` requires of a fixed profile and
/// `doc/model-lifecycle.md` warns cannot be inferred from eager execution
/// working. Device allocation is illegal during capture, a captured copy from
/// host memory replays from a pointer that is gone, and a synchronize inside a
/// capture is an error -- so a forward that captures is a forward that does
/// none of those things, checked rather than asserted in a comment.
///
/// `APXINF_HIP_GRAPH=0` turns it off for A/B measurement and for the case the
/// capture itself is the suspect.
fn graph_capture_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("APXINF_HIP_GRAPH").unwrap_or_default().trim(),
            "0" | "false" | "off"
        )
    })
}

impl HipGr00tPrepared {
    /// High-water mark of the activation arena, and its capacity.
    ///
    /// `HipGr00tModel::arena_bytes` is an audited bound rather than an
    /// estimate; exposing both lets a test assert it stays above the real
    /// peak as the forward path changes.
    /// `(replays, graph nodes)`; `(0, 0)` when the forward never got captured.
    pub fn graph_usage(&self) -> (usize, usize) {
        (
            self.replays.get(),
            self.graph.borrow().as_ref().map_or(0, |(g, _)| g.nodes()),
        )
    }

    pub fn arena_usage(&self) -> (usize, usize) {
        (self.arena.peak(), self.arena.capacity())
    }
}

impl PreparedInference for HipGr00tPrepared {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }

    fn status(&self) -> PreparationStatus {
        PreparationStatus::Ready {
            mode: ExecutionMode::Eager,
            fallback_reason: None,
        }
    }

    /// The returned action lives in the arena: the next run overwrites it.
    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        // A plan is bound to far more than the portable spec, so the full key
        // is what has to match. Reconstructing it also re-validates the
        // request's shapes and metadata.
        let (key, inputs) = read_request(&self.model, request)?;
        if key != self.key {
            return Err(Error::Other(format!(
                "prepared GR00T plan is bound to {:?} but the request is {:?}",
                PlanSummary::from(&self.key),
                PlanSummary::from(&key)
            )));
        }
        self.run_bound(&inputs)
    }
}

/// The parts of a [`PlanKey`] worth printing in a mismatch message; the row
/// index lists are long and say nothing a reader can act on.
#[derive(Debug)]
struct PlanSummary {
    patches: usize,
    grid: Vec<[u32; 3]>,
    tokens: usize,
    embodiment: usize,
    image_rows: usize,
}

impl From<&PlanKey> for PlanSummary {
    fn from(key: &PlanKey) -> Self {
        Self {
            patches: key.patches,
            grid: key.grid.clone(),
            tokens: key.token_count,
            embodiment: key.embodiment_id,
            image_rows: key.groups.image.len(),
        }
    }
}

impl VlaRuntime for HipGr00tRunner {
    fn model_variant(&self) -> Option<&'static str> {
        Some("bf16")
    }

    fn contract(&self) -> VlaContract {
        let config = self.model.config();
        let vision = &self.model.backbone_config().vision;
        VlaContract {
            action_shape: [config.action_horizon, config.max_action_dim],
            // The official processor's row count is request-dependent; a zero
            // row count advertises that only the feature width is fixed.
            patch_shape: [
                0,
                vision.in_channels
                    * vision.temporal_patch_size
                    * vision.patch_size
                    * vision.patch_size,
            ],
            max_token_len: config.max_seq_len,
            num_views: 0,
            image_size: config.image_target_size.map_or(0, |shape| shape[0]),
            patch_size: vision.patch_size,
            accepts_rgb_u8: false,
        }
    }

    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        let (key, inputs) = self.read_request(request)?;
        let cached = self
            .prepared
            .borrow()
            .as_ref()
            .filter(|(cached, _)| *cached == key)
            .map(|(_, plan)| Rc::clone(plan));
        let prepared = match cached {
            Some(prepared) => prepared,
            None => {
                self.clear_prepared()?;
                let plan = Rc::new(self.build(key.clone(), request, ExecutionPolicy::PreferGraph)?);
                *self.prepared.borrow_mut() = Some((key, Rc::clone(&plan)));
                plan
            }
        };
        prepared.run_bound(&inputs)
    }

    fn prepare(&self, _spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        Err(Error::Other(
            "GR00T on HIP cannot prepare from an InferenceSpec: it carries only token_count and \
             image_layout, while GR00T's allocations also depend on image_grid_thw (patch rows \
             and per-frame attention segments) and embodiment_id (which adapters are uploaded). \
             Use prepare_for with a representative request so the plan binds every shape."
                .into(),
        ))
    }

    fn prepare_with_policy(
        &self,
        spec: &InferenceSpec,
        _policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        self.prepare(spec)
    }

    fn prepare_for(
        &self,
        sample: &VlaRequest<'_>,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        Ok(Box::new(self.prepare_plan(sample, policy)?))
    }

    fn execution_mode(&self) -> &'static str {
        if self.prepared.borrow().is_some() {
            "eager"
        } else {
            "unprepared"
        }
    }

    fn clear_prepared(&self) -> Result<()> {
        // The arena may still be read by queued work; let it finish first.
        self.model.context().synchronize()?;
        drop(self.prepared.borrow_mut().take());
        Ok(())
    }

    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        let action = self.infer(request)?;
        self.model.context().to_cpu(action.tensor())?.to_f32_vec()
    }
}

/// A host tensor of `shape` as BF16, converting from any float dtype and
/// accepting any leading singleton axes the processor may have left on.
fn host_bf16(tensor: &Tensor, shape: Vec<usize>, label: &str) -> Result<Tensor> {
    if tensor.device() != Device::Cpu {
        return Err(Error::Other(format!(
            "GR00T {label} must be a CPU tensor"
        )));
    }
    if tensor.numel() != shape.iter().product::<usize>() {
        return Err(Error::Other(format!(
            "GR00T {label} has {} elements, expected {shape:?}",
            tensor.numel()
        )));
    }
    if tensor.dtype() == DType::BF16 {
        return tensor.reshape(shape);
    }
    let values: Vec<bf16> = tensor
        .to_f32_vec()?
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    Tensor::from_bf16(shape, &values)
}
