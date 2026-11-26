//! π0.5 on HIP as a [`VlaRuntime`].
//!
//! The runner owns the implicit plan cache and the portable trait surface;
//! building a plan — including its warmup and graph capture — lives in
//! [`super::prepare`].

use std::cell::RefCell;
use std::rc::Rc;

use apxinf_core::Result;

use super::super::config::Config;
use super::super::model::HipPi05Model;
use super::prepare::HipPi05Prepared;
use crate::vla::{
    Action, ExecutionMode, ExecutionPolicy, InferenceSpec, PreparationStatus, PreparedInference,
    VlaContract, VlaRequest, VlaRuntime,
};

pub struct HipPi05Runner {
    model: Rc<HipPi05Model>,
    /// The implicit plan `infer` keeps, for the most recent shape only: an
    /// arena is large, so a replacement is built after the old one is gone.
    prepared: RefCell<Option<(InferenceSpec, Rc<HipPi05Prepared>)>>,
}

impl HipPi05Runner {
    pub fn new(model: Rc<HipPi05Model>) -> Self {
        Self {
            model,
            prepared: RefCell::new(None),
        }
    }

    pub fn model(&self) -> &Rc<HipPi05Model> {
        &self.model
    }

    fn config(&self) -> &Config {
        self.model.config()
    }

    fn build(&self, spec: &InferenceSpec, policy: ExecutionPolicy) -> Result<HipPi05Prepared> {
        HipPi05Prepared::new(Rc::clone(&self.model), *spec, policy)
    }

    /// The concrete plan, for callers that want more than
    /// [`PreparedInference`] exposes (graph node counts, for measurement).
    pub fn prepare_plan(
        &self,
        spec: &InferenceSpec,
        policy: ExecutionPolicy,
    ) -> Result<HipPi05Prepared> {
        self.build(spec, policy)
    }
}

impl VlaRuntime for HipPi05Runner {
    fn model_variant(&self) -> Option<&'static str> {
        Some("bf16")
    }

    fn contract(&self) -> VlaContract {
        let c = self.config();
        VlaContract {
            action_shape: [c.action_horizon, c.action_dim],
            patch_shape: [
                c.num_views * c.patches_per_view(),
                3 * c.patch_size * c.patch_size,
            ],
            max_token_len: c.max_token_len,
            num_views: c.num_views,
            image_size: c.image_size,
            patch_size: c.patch_size,
            accepts_rgb_u8: true,
        }
    }

    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        request.observation.validate()?;
        let spec = request.observation.inference_spec();
        let cached = self
            .prepared
            .borrow()
            .as_ref()
            .filter(|(s, _)| *s == spec)
            .map(|(_, p)| Rc::clone(p));
        let prepared = match cached {
            Some(prepared) => prepared,
            None => {
                self.clear_prepared()?;
                let prepared = Rc::new(self.build(&spec, ExecutionPolicy::PreferGraph)?);
                *self.prepared.borrow_mut() = Some((spec, Rc::clone(&prepared)));
                prepared
            }
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
        Ok(Box::new(self.build(spec, policy)?))
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
        // The cached plan's actual mode, as `status()` reports it; the
        // implicit path always asks PreferGraph, so a plan here is eager only
        // when capture fell back.
        match self.prepared.borrow().as_ref() {
            Some((_, plan)) => match plan.status() {
                PreparationStatus::Ready { mode, .. } => match mode {
                    ExecutionMode::Graph => "graph",
                    ExecutionMode::Eager => "eager",
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
