//! The prepared plan: fixed device inputs, per-shape host preparation, the
//! selected embodiment, the activation arena, and -- when the policy asks and
//! the capture succeeds -- the forward recorded as one HIP graph launch.
//!
//! Preparation runs the model, deliberately: the first execution of a shape
//! also does whatever first-touch work the libraries defer (hipBLASLt's
//! solution map, CK's module load), and recording that into a graph would
//! replay it forever. So a plan under a graph policy warms up once eagerly,
//! then captures; by the time `run` is called nothing is left but the input
//! upload and the replay.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use apxinf_core::{DType, Error, Result, Tensor};
use apxinf_hip::graph::HipGraph;
use apxinf_hip::native::{self, HipArena};

use crate::hip_gr00t::model::blocks::action_head::DitRowIndices;
use crate::hip_gr00t::model::blocks::backbone::PreparedBackbone;
use crate::hip_gr00t::model::blocks::vision::PreparedVision;
use crate::hip_gr00t::model::HipGr00tModel;
use crate::hip_gr00t::weights::Embodiment;
use crate::vla::{Action, ExecutionMode, ExecutionPolicy, InferenceSpec, PreparationStatus, PreparedInference, VlaRequest};

use super::runner::{Inputs, PlanKey};

/// One fully bound plan: fixed device inputs, the per-shape host preparation,
/// the selected embodiment, and the activation arena.
pub struct HipGr00tPrepared {
    pub(super) model: Rc<HipGr00tModel>,
    pub(super) key: PlanKey,
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
    rows: DitRowIndices,
    embodiment: Rc<Embodiment>,
    /// The forward, captured as one launch, and the output tensor it writes.
    ///
    /// Holding the output tensor alongside the graph is what makes the replay
    /// usable -- the replay writes the same arena slot, and nothing else
    /// returns it.
    ///
    /// `None` when the policy did not ask for a graph, or when capture fell
    /// back; [`Self::fallback_reason`] says which.
    graph: RefCell<Option<(HipGraph, Tensor)>>,
    fallback_reason: Option<String>,
    /// Inferences served by replaying the graph.
    ///
    /// A capture that fell back, or a policy that never asked, both leave the
    /// eager path in use and the right answer returned -- so without a count,
    /// a measurement that never reached the graph is indistinguishable from
    /// one that found no gain. The same reasoning as the GEMM and attention
    /// provider counts.
    replays: Cell<usize>,
}

impl HipGr00tPrepared {
    /// Build and bind a plan, then warm it up and capture it if the policy
    /// asks. The warmup inputs are the caller's representative request.
    pub(super) fn new(
        model: Rc<HipGr00tModel>,
        key: PlanKey,
        inputs: &Inputs,
        policy: ExecutionPolicy,
    ) -> Result<Self> {
        let mut prepared = Self::allocate(model, key, &inputs.token_ids)?;

        // Warm up eagerly, whatever the policy: this run performs the
        // libraries' first-touch work for this shape and checks the forward
        // against real data before anything is recorded.
        prepared.run_bound(inputs)?;

        let (graph, fallback_reason) = match policy {
            ExecutionPolicy::Eager => (None, None),
            ExecutionPolicy::RequireGraph | ExecutionPolicy::PreferGraph => {
                match prepared.capture() {
                    Ok(graph) => (Some(graph), None),
                    Err(error) => {
                        if policy == ExecutionPolicy::RequireGraph {
                            return Err(error);
                        }
                        // PreferGraph falls back -- but only onto a stream
                        // that still works. A failure inside the captured
                        // region latches an error on the stream itself
                        // (measured: later calls report "a previous error
                        // during capture" instead of their own status), so
                        // probe before promising eager execution.
                        prepared.model.context().stream_status()?;
                        (None, Some(error.to_string()))
                    }
                }
            }
        };
        *prepared.graph.borrow_mut() = graph;
        prepared.fallback_reason = fallback_reason;
        if let Some(reason) = &prepared.fallback_reason {
            eprintln!("[apxinf] GR00T HIP graph capture unavailable, using eager: {reason}");
        }
        Ok(prepared)
    }

    fn allocate(model: Rc<HipGr00tModel>, key: PlanKey, token_ids: &[u32]) -> Result<Self> {
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
        let rows = DitRowIndices::new(ctx, &key.groups.non_image, &key.groups.image)?;
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
            fallback_reason: None,
            replays: Cell::new(0),
        })
    }

    /// Upload the request's host values and execute the plan.
    ///
    /// Never captures and never retries a capture: with capture moved into
    /// preparation, a run is exactly one upload plus one replay or eager
    /// forward.
    pub(super) fn run_bound(&self, inputs: &Inputs) -> Result<Action> {
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
        Ok(Action::new(actions))
    }

    /// Record this plan's forward, so later runs replay it as one launch.
    ///
    /// Runs the forward once more under capture. That run's result is
    /// discarded -- the warmup already returned the answer -- but the arena
    /// slot it writes is the slot every replay writes, so it is kept alongside
    /// the graph.
    fn capture(&self) -> Result<(HipGraph, Tensor)> {
        let ctx = self.model.context();
        // The eager warmup left work queued against the arena; capture must
        // not interleave with it.
        ctx.synchronize()?;
        self.arena.reset();

        // SAFETY: the captured region allocates nothing on the device (the
        // arena hands out slices of one allocation, and the DiT row indices
        // moved to preparation), copies nothing from host memory, and neither
        // synchronizes nor downloads. Every replay is preceded by
        // `arena.reset()` in `run_bound`, so the addresses the graph records
        // are the addresses it replays against.
        let (graph, output) = unsafe {
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
        }?;
        if graph.nodes() == 0 {
            return Err(Error::Other(
                "apxinf-hip: the GR00T capture recorded no nodes".into(),
            ));
        }
        Ok((graph, output))
    }

    /// `(replays, graph nodes)`; `(0, 0)` when the plan runs eagerly.
    pub fn graph_usage(&self) -> (usize, usize) {
        (
            self.replays.get(),
            self.graph.borrow().as_ref().map_or(0, |(g, _)| g.nodes()),
        )
    }

    /// High-water mark of the activation arena, and its capacity.
    ///
    /// `HipGr00tModel::arena_bytes` is an audited bound rather than an
    /// estimate; exposing both lets a test assert it stays above the real
    /// peak as the forward path changes.
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
            mode: if self.graph.borrow().is_some() {
                ExecutionMode::Graph
            } else {
                ExecutionMode::Eager
            },
            fallback_reason: self.fallback_reason.clone(),
        }
    }

    /// The returned action lives in the arena: the next run overwrites it.
    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        // A plan is bound to far more than the portable spec, so the full key
        // is what has to match. Reconstructing it also re-validates the
        // request's shapes and metadata.
        let (key, inputs) = super::runner::read_request(&self.model, request)?;
        if key != self.key {
            return Err(Error::Other(format!(
                "prepared GR00T plan is bound to {:?} but the request is {:?}",
                super::runner::PlanSummary::from(&self.key),
                super::runner::PlanSummary::from(&key)
            )));
        }
        self.run_bound(&inputs)
    }
}
