//! Stable workspace allocation, real warmup and graph capture; retained plan resources.
use super::runner::*;
use crate::cuda_dreamzero::{
    backend,
    model::{DreamZeroModel, Profile, Workspace},
};
use crate::vla::{ExecutionPolicy, InferenceSpec, VlaRequest};
use apxinf_core::{Error, NormalGenerator, Result};
use std::{cell::RefCell, rc::Rc};
pub(super) struct Plan {
    pub(super) spec: InferenceSpec,
    pub(super) model: Rc<DreamZeroModel>,
    pub(super) session: Session,
    pub(super) profile: Profile,
    pub(super) work: RefCell<Workspace>,
    pub(super) normal: RefCell<Box<dyn NormalGenerator>>,
    pub(super) graphs: Option<[backend::CapturedGraph; 3]>,
    pub(super) sessions: [backend::ExecutionSession; 3],
    pub(super) fallback_reason: Option<String>,
}

impl DreamZeroModelRunner {
    pub(super) fn make_plan(
        &self,
        request: &VlaRequest<'_>,
        host: &HostInputs,
        policy: ExecutionPolicy,
    ) -> Result<Plan> {
        let episode = self.session.borrow();
        let profile = profile(
            host,
            &request.observation.token_ids,
            episode.as_ref().map(|e| &e.key),
        );
        let mut work = self.model.workspace(profile)?;
        let mut normal =
            backend::sampling::create_normal_generator(&self.model.ctx, work.noise().clone())?;
        bind(&self.model, &work, host, request, &mut *normal)?;
        self.model.bind_history(
            &work,
            if profile.initial() {
                None
            } else {
                episode.as_ref().map(|e| &e.history)
            },
        )?;
        let scratch = self.model.phase_scratch_bytes();
        let sessions = [
            backend::ExecutionSession::with_capacity(scratch[0], self.model.ctx.device_id())?,
            backend::ExecutionSession::with_capacity(scratch[1], self.model.ctx.device_id())?,
            backend::ExecutionSession::with_capacity(scratch[2], self.model.ctx.device_id())?,
        ];
        // Resolve each operator and its workspace on real input, without publishing history.
        sessions[0].prepare(|| self.model.vision(&mut work, true))?;
        sessions[1].prepare(|| self.model.language(&mut work, true))?;
        sessions[2].prepare(|| self.model.action(&mut work, true))?;
        self.model.ctx.synchronize().map_err(Error::Cuda)?;
        let mut graphs = None;
        let mut fallback_reason = None;
        if policy != ExecutionPolicy::Eager {
            let capture = (|| {
                let vision = sessions[0].run(|| {
                    backend::capture(&self.model.ctx, || self.model.vision(&mut work, false))
                })?;
                let language = sessions[1].run(|| {
                    backend::capture(&self.model.ctx, || self.model.language(&mut work, false))
                })?;
                let action = sessions[2].run(|| {
                    backend::capture(&self.model.ctx, || self.model.action(&mut work, false))
                })?;
                Ok::<_, Error>([vision, language, action])
            })();
            match capture {
                Ok(g) => graphs = Some(g),
                Err(e) if policy == ExecutionPolicy::PreferGraph => {
                    fallback_reason = Some(e.to_string())
                }
                Err(e) => return Err(e),
            }
        }
        Ok(Plan {
            spec: request.observation.inference_spec(),
            model: self.model.clone(),
            session: self.session.clone(),
            profile,
            work: RefCell::new(work),
            normal: RefCell::new(normal),
            graphs,
            sessions,
            fallback_reason,
        })
    }
}
