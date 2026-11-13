//! GR00T N1.7 on HIP as a [`VlaRuntime`].
//!
//! A prepared plan owns the fixed device inputs, the per-shape host
//! preparation and one activation arena; each run refills the inputs and
//! reuses the rest. Building that plan -- including its warmup and graph
//! capture -- lives in [`super::prepare`]; this file owns the plan cache and
//! the portable trait surface.
//!
//! **Why `prepare` refuses.** The portable [`InferenceSpec`] carries only a
//! token count and an image layout. GR00T's allocations also depend on
//! `image_grid_thw` (how many patch rows, and the per-frame segments the
//! vision attention batches over) and on `embodiment_id` (which adapters are
//! uploaded). A plan built from a spec alone would bind neither, so
//! [`VlaRuntime::prepare`] returns an error naming what is missing and
//! [`VlaRuntime::prepare_for`] -- which receives a whole request, metadata
//! included -- is the supported entry point.

use std::cell::RefCell;
use std::rc::Rc;

use apxinf_core::{Device, Result, Tensor};
use half::bf16;

use crate::hip_gr00t::model::token_groups::{build_backbone_token_groups, Gr00tBackboneTokenGroups};
use crate::hip_gr00t::model::HipGr00tModel;
use crate::vla::{
    Action, ExecutionPolicy, InferenceSpec, InitialLatent,
    PreparedInference, VisionObservation, VlaContract, VlaRequest, VlaRuntime,
};

use super::prepare::HipGr00tPrepared;

/// Everything about a request that changes an allocation or a dispatch shape.
///
/// This is the plan cache key, and the reason it is not [`InferenceSpec`]: two
/// requests with the same token count but different grids need different
/// plans, and serving one from the other's plan would be wrong rather than
/// slow.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::hip_gr00t::model_runner) struct PlanKey {
    pub(super) patches: usize,
    pub(super) grid: Vec<[u32; 3]>,
    pub(super) token_count: usize,
    pub(super) embodiment_id: usize,
    /// Row groups depend on which tokens are image placeholders, which the
    /// mask and the token IDs decide together.
    pub(super) groups: Gr00tBackboneTokenGroups,
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
        let (key, inputs) = self.read_request(request)?;
        HipGr00tPrepared::new(Rc::clone(&self.model), key, &inputs, policy)
    }

    fn read_request(&self, request: &VlaRequest<'_>) -> Result<(PlanKey, Inputs)> {
        read_request(&self.model, request)
    }
}

/// Everything the request must carry, validated once, as the plan key plus
/// the host tensors to upload.
///
/// A free function rather than a method: a prepared plan has to re-derive the
/// key to check a request against itself, and it holds the model but no runner.
pub(super) fn read_request(model: &HipGr00tModel, request: &VlaRequest<'_>) -> Result<(PlanKey, Inputs)> {
    let observation = request.observation;
    observation.validate()?;
    let config = model.config();
    let backbone = model.backbone_config();

    let pixel_values = match &observation.vision {
        VisionObservation::Patches(tensor) => tensor,
        VisionObservation::RgbU8 { .. } => {
            return Err(apxinf_core::Error::Other(
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
        return Err(apxinf_core::Error::ShapeMismatch {
            expected: format!("[patches, {patch_width}]"),
            got: pixel_values.shape().to_string(),
        });
    }
    let patches = dims[0];

    let grid = request
        .metadata
        .image_grid_thw
        .ok_or(apxinf_core::Error::Contract("GR00T requires VlaMetadata.image_grid_thw"))?;
    let attention_mask = request
        .metadata
        .attention_mask
        .ok_or(apxinf_core::Error::Contract("GR00T requires VlaMetadata.attention_mask"))?;
    let embodiment_id = request
        .metadata
        .embodiment_id
        .ok_or(apxinf_core::Error::Contract("GR00T requires VlaMetadata.embodiment_id"))?;
    let groups = build_backbone_token_groups(
        &observation.token_ids,
        attention_mask,
        backbone.image_token_id,
    )?;

    let state = observation.state.as_ref().ok_or(apxinf_core::Error::Contract(
        "GR00T requires a normalized, padded state tensor",
    ))?;
    let state = host_bf16(state, vec![1, config.state_input_dim()?], "state")?;
    let noise = match request.initial_latent {
        InitialLatent::Provided(latent) => latent,
        InitialLatent::Generate { .. } => {
            return Err(apxinf_core::Error::Contract(
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
pub(in crate::hip_gr00t::model_runner) struct Inputs {
    pub(super) pixel_values: Tensor,
    pub(super) state: Tensor,
    pub(super) noise: Tensor,
    /// Carried per request because the plan key holds only the token *count*
    /// and the placeholder layout -- two prompts can share a plan.
    pub(super) token_ids: Vec<u32>,
}

/// The parts of a [`PlanKey`] worth printing in a mismatch message; the row
/// index lists are long and say nothing a reader can act on.
#[derive(Debug)]
pub(super) struct PlanSummary {
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
                let plan = Rc::new(self.prepare_plan(request, ExecutionPolicy::PreferGraph)?);
                *self.prepared.borrow_mut() = Some((key, Rc::clone(&plan)));
                plan
            }
        };
        prepared.run_bound(&inputs)
    }

    fn prepare(&self, _spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        Err(apxinf_core::Error::Other(
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
        // The cached plan's actual mode, as `status()` reports it; the
        // implicit path always asks PreferGraph, so a plan here is eager only
        // when capture fell back.
        match self.prepared.borrow().as_ref() {
            Some((_, plan)) => match plan.status() {
                crate::vla::PreparationStatus::Ready { mode, .. } => match mode {
                    crate::vla::ExecutionMode::Graph => "graph",
                    crate::vla::ExecutionMode::Eager => "eager",
                },
                _ => "unprepared",
            },
            None => "unprepared",
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
        return Err(apxinf_core::Error::Other(format!(
            "GR00T {label} must be a CPU tensor"
        )));
    }
    if tensor.numel() != shape.iter().product::<usize>() {
        return Err(apxinf_core::Error::Other(format!(
            "GR00T {label} has {} elements, expected {shape:?}",
            tensor.numel()
        )));
    }
    if tensor.dtype() == apxinf_core::DType::BF16 {
        return tensor.reshape(shape);
    }
    let values: Vec<bf16> = tensor
        .to_f32_vec()?
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    Tensor::from_bf16(shape, &values)
}

