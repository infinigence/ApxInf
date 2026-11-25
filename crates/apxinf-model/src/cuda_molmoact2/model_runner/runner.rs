//! Request binding for the existing VLA interface: validation of what a request may carry, and the
//! public shape contract of this fixed LIBERO BF16 profile.
use crate::cuda_molmoact2::backend::ops::GemmPolicy;
use crate::cuda_molmoact2::backend::{AttentionPolicy, CudaBuffer, CudaContext};
use crate::cuda_molmoact2::config::MolmoAct2Config;
use crate::cuda_molmoact2::geometry::NUM_VIEWS;
use crate::cuda_molmoact2::model::{HostPlan, MolmoAct2Model};
use crate::cuda_molmoact2::weights::Bf16Weights;
use crate::vla::*;
use apxinf_core::{DType, Device, Error, Result};
use std::{cell::RefCell, rc::Rc};

use super::prepare::{Plan, Resources};

pub(super) fn invalid(message: &str) -> Error {
    Error::Other(format!("MolmoAct2: {message}"))
}

pub(in crate::cuda_molmoact2) struct ModelRunner {
    pub(super) ctx: Rc<CudaContext>,
    pub(super) model: Rc<MolmoAct2Model>,
    pub(super) resources: RefCell<Resources>,
    /// How `infer` executes. Every policy keeps the most recently used prepared plan (a plan reserves a
    /// workspace arena and the prepared native executions; a graph plan also owns the captured graph,
    /// so the implicit cache holds one; explicit `prepare_for` plans are caller-owned):
    /// * `Eager` — a *prepared eager* plan: stable buffers, an `ExecutionSession` and the prepared native
    ///   executions are kept and reused, the host enqueues every operator on each request, and nothing is
    ///   captured or replayed;
    /// * `PreferGraph` / `RequireGraph` — the same plan plus the captured whole-model graph (`PreferGraph`
    ///   falls back to the prepared eager plan, with the reason, when capture fails).
    ///
    /// The raw eager path (`ModelRunner::run`: nothing prepared, every execution created and destroyed
    /// per call, one synchronization per GEMM) is not an execution mode of `infer`; it only serves the
    /// tuning pre-run below.
    policy: ExecutionPolicy,
    /// The backend was loaded with online tuning: a real request is run through the raw eager path (tuning
    /// allowed) before a plan is prepared, so recipes are tuned on real data and preparation itself never
    /// tunes. `run` never builds a plan, so this does not recurse.
    tune: bool,
    prepared: RefCell<Option<Rc<Plan>>>,
}

impl ModelRunner {
    pub fn with_policy(
        ctx: Rc<CudaContext>,
        weights: Rc<Bf16Weights>,
        cfg: MolmoAct2Config,
        gemm: GemmPolicy,
        attention: AttentionPolicy,
        policy: ExecutionPolicy,
    ) -> Result<Self> {
        let tune = gemm.online_tune || attention.online_tune;
        let model = Rc::new(MolmoAct2Model::new(ctx.clone(), weights, cfg, gemm, attention)?);
        let resources = RefCell::new(Resources::new(&ctx, model.config())?);
        Ok(Self { ctx, model, resources, policy, tune, prepared: RefCell::new(None) })
    }

    /// The implicit plan, if one is prepared (tests).
    #[cfg(test)]
    pub(in crate::cuda_molmoact2) fn implicit_plan(&self) -> Option<Rc<Plan>> {
        self.prepared.borrow().clone()
    }

    fn horizon(&self) -> usize {
        self.model.config().expert.max_action_horizon
    }

    fn action_width(&self) -> usize {
        self.model.config().expert.max_action_dim
    }

    pub(super) fn validate_request(&self, request: &VlaRequest<'_>) -> Result<()> {
        validate_request(self.model.config(), request)
    }

    /// A plan for the structure of `sample`, optionally tuned on the real sample first.
    fn build_plan(&self, sample: &VlaRequest<'_>, host: HostPlan, policy: ExecutionPolicy) -> Result<Plan> {
        if self.tune {
            // Recipes tuned on the real request (a miss resolves once, then the plan only reuses them).
            self.run(sample)?;
        }
        Plan::build(self.model.clone(), self.ctx.clone(), sample, host, policy)
    }

    /// The implicit-cache path of `infer`, for every policy: reuse the plan when the request has the same
    /// structure (new content is written in place), otherwise replace it. With `Eager` the plan has no
    /// graph; the first call and a structure change prepare (probe, prepare), later calls of the same
    /// structure only rewrite the stable buffers and run the traversal on the prepared executions.
    fn infer_planned(&self, request: &VlaRequest<'_>) -> Result<Action> {
        self.validate_request(request)?;
        request.observation.validate()?;
        let host = self.model.analyze(&request.observation.token_ids)?;
        // The hit is looked up in its own scope: no handle to the cached plan may survive into the rebuild
        // below, otherwise the old arena, graph and native GEMM/Attention executions stay alive while the
        // replacement is prepared (two complete plans at once).
        let hit = self.prepared.borrow().as_ref().filter(|plan| plan.matches(&host)).cloned();
        if let Some(plan) = hit {
            return plan.run_host(request, &host);
        }
        // Structure change or first call: release the old plan first, then prepare its replacement.
        self.clear_prepared()?;
        let plan = Rc::new(self.build_plan(request, host.clone(), self.policy)?);
        *self.prepared.borrow_mut() = Some(plan.clone());
        plan.run_host(request, &host)
    }
}

/// Everything a request may carry. State is part of the prompt tokens in this model, so a
/// continuous state, masks, grids, embodiment ids and planning options are all rejected.
pub(super) fn validate_request(cfg: &MolmoAct2Config, request: &VlaRequest<'_>) -> Result<()> {
    let observation = request.observation;
    observation.validate()?;
    let size = cfg.vit.image_size;
    match &observation.vision {
        VisionObservation::RgbU8 { bytes, layout: ImageLayout::Nhwc } if bytes.len() == NUM_VIEWS * size * size * 3 => {}
        _ => return Err(invalid(&format!("expected {NUM_VIEWS} NHWC {size}x{size} RGB uint8 images in one buffer"))),
    }
    let max_tokens = cfg.text.max_positions;
    if observation.token_ids.len() > max_tokens {
        return Err(invalid(&format!("{} prompt tokens exceed the {max_tokens} positions of the model", observation.token_ids.len())));
    }
    if observation.state.is_some() {
        return Err(invalid("the state is encoded in the prompt tokens; a continuous state is not accepted"));
    }
    let metadata = &request.metadata;
    if observation.action_mask.is_some()
        || metadata.attention_mask.is_some()
        || metadata.image_grid_thw.is_some()
        || metadata.embodiment_id.is_some()
        || metadata.planning.is_some()
    {
        return Err(invalid("this profile takes no action mask, attention mask, grids, embodiment id or planning options"));
    }
    if let InitialLatent::Provided(noise) = request.initial_latent {
        let want = [1, cfg.expert.max_action_horizon, cfg.expert.max_action_dim];
        if noise.device() != Device::Cpu
            || noise.dtype() != DType::F32
            || noise.shape().dims() != want
            || noise.as_f32()?.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(&format!("supplied noise must be finite host FP32 {want:?}")));
        }
    }
    Ok(())
}

impl VlaRuntime for ModelRunner {
    fn model_variant(&self) -> Option<&'static str> {
        Some("bf16")
    }

    fn contract(&self) -> VlaContract {
        let cfg = self.model.config();
        let side = cfg.patch_side();
        VlaContract {
            action_shape: [self.horizon(), self.action_width()],
            patch_shape: [NUM_VIEWS * side * side, cfg.vit.patch_size * cfg.vit.patch_size * 3],
            max_token_len: cfg.text.max_positions,
            num_views: NUM_VIEWS,
            image_size: cfg.vit.image_size,
            patch_size: cfg.vit.patch_size,
            accepts_rgb_u8: true,
        }
    }

    /// The full noise tensor keeps its leading batch axis (`[1, horizon, width]`), as the official run.
    fn initial_latent_shape(&self) -> Vec<usize> {
        vec![1, self.horizon(), self.action_width()]
    }

    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        self.infer_planned(request)
    }

    /// The token structure of a request (image layout, prefill segments, visible context rows) is part of
    /// the compiled traversal and cannot be derived from a token count, so a plan is always built from a
    /// real sample: use `prepare_for`.
    fn prepare(&self, _spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        Err(invalid("a prepared plan needs the prompt tokens (image layout and context rows depend on them); use prepare_for with a sample request"))
    }

    fn prepare_with_policy(&self, _spec: &InferenceSpec, _policy: ExecutionPolicy) -> Result<Box<dyn PreparedInference>> {
        self.prepare(_spec)
    }

    /// Build a caller-owned plan for the structure of `sample`. `RequireGraph` fails when capture fails;
    /// `PreferGraph` reports an eager plan with its fallback reason. The plan keeps no image / noise value
    /// of the sample, and does not touch the runner's implicit cache.
    fn prepare_for(&self, sample: &VlaRequest<'_>, policy: ExecutionPolicy) -> Result<Box<dyn PreparedInference>> {
        self.validate_request(sample)?;
        sample.observation.validate()?;
        let host = self.model.analyze(&sample.observation.token_ids)?;
        Ok(Box::new(self.build_plan(sample, host, policy)?))
    }

    /// Evict the implicit plan (arena, captured graph). Caller-owned plans stay alive.
    fn clear_prepared(&self) -> Result<()> {
        self.ctx.synchronize().map_err(Error::Cuda)?;
        drop(self.prepared.borrow_mut().take());
        Ok(())
    }

    /// `eager` for an `Eager`-policy runner (prepared eager, or not yet prepared: it never becomes a
    /// graph); for the graph policies `unprepared`, `graph`, or `eager` after a `PreferGraph` fallback.
    fn execution_mode(&self) -> &'static str {
        if self.policy == ExecutionPolicy::Eager {
            return "eager";
        }
        match self.prepared.borrow().as_ref() {
            None => "unprepared",
            Some(plan) if plan.is_graph() => "graph",
            Some(_) => "eager",
        }
    }

    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        let action = self.infer(request)?;
        let mut bytes = vec![0u8; self.horizon() * self.action_width() * 4];
        CudaBuffer::from_tensor(action.tensor())
            .map_err(Error::Cuda)?
            .copy_to_host(&mut bytes)
            .map_err(Error::Cuda)?;
        Ok(bytes.chunks_exact(4).map(|b| f32::from_ne_bytes(b.try_into().unwrap())).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_molmoact2::config::tests::libero_json;
    use apxinf_core::Tensor;

    fn cfg() -> MolmoAct2Config {
        MolmoAct2Config::from_json(&libero_json()).unwrap()
    }

    fn observation() -> Observation {
        Observation {
            vision: VisionObservation::RgbU8 { bytes: vec![0; 2 * 378 * 378 * 3], layout: ImageLayout::Nhwc },
            token_ids: vec![1, 2, 3],
            state: None,
            action_mask: None,
        }
    }

    fn noise() -> Tensor {
        Tensor::zeros(vec![1, 10, 32], DType::F32)
    }

    #[test]
    fn a_complete_request_with_full_noise_is_accepted() {
        let (o, n) = (observation(), noise());
        assert!(validate_request(&cfg(), &VlaRequest::provided(&o, &n)).is_ok());
        assert!(validate_request(&cfg(), &VlaRequest::generated(&o, apxinf_core::RngKey::new(1, 0, 0))).is_ok());
    }

    #[test]
    fn wrong_images_tokens_and_unsupported_inputs_are_rejected() {
        let n = noise();
        let mut o = observation();
        o.vision = VisionObservation::RgbU8 { bytes: vec![0; 2 * 378 * 378 * 3 - 1], layout: ImageLayout::Nhwc };
        assert!(validate_request(&cfg(), &VlaRequest::provided(&o, &n)).is_err(), "short image buffer");
        o.vision = VisionObservation::RgbU8 { bytes: vec![0; 2 * 378 * 378 * 3], layout: ImageLayout::Nchw };
        assert!(validate_request(&cfg(), &VlaRequest::provided(&o, &n)).is_err(), "NCHW");
        o.vision = VisionObservation::Patches(Tensor::zeros(vec![2, 2], DType::F32));
        assert!(validate_request(&cfg(), &VlaRequest::provided(&o, &n)).is_err(), "patch input");

        let mut o = observation();
        o.token_ids.clear();
        assert!(validate_request(&cfg(), &VlaRequest::provided(&o, &n)).is_err(), "no tokens");
        let mut o = observation();
        o.token_ids = vec![0; 16385];
        assert!(validate_request(&cfg(), &VlaRequest::provided(&o, &n)).is_err(), "too many tokens");

        let mut o = observation();
        o.state = Some(Tensor::zeros(vec![8], DType::F32));
        assert!(validate_request(&cfg(), &VlaRequest::provided(&o, &n)).is_err(), "continuous state");
        let mut o = observation();
        o.action_mask = Some(Tensor::zeros(vec![10, 32], DType::F32));
        assert!(validate_request(&cfg(), &VlaRequest::provided(&o, &n)).is_err(), "action mask");
        let o = observation();
        let mut request = VlaRequest::provided(&o, &n);
        let planning = PlanningOptions::default();
        request.metadata.planning = Some(&planning);
        assert!(validate_request(&cfg(), &request).is_err(), "planning options");
        let mut request = VlaRequest::provided(&o, &n);
        request.metadata.embodiment_id = Some(0);
        assert!(validate_request(&cfg(), &request).is_err(), "embodiment id");
    }

    #[test]
    fn noise_must_be_the_full_finite_host_fp32_tensor() {
        let o = observation();
        for bad in [
            Tensor::zeros(vec![10, 32], DType::F32),
            Tensor::zeros(vec![1, 10, 7], DType::F32),
            Tensor::zeros(vec![1, 10, 32], DType::BF16),
        ] {
            assert!(validate_request(&cfg(), &VlaRequest::provided(&o, &bad)).is_err());
        }
        let mut values = vec![0.0f32; 320];
        for non_finite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            values[17] = non_finite;
            let n = Tensor::from_f32(vec![1, 10, 32], &values).unwrap();
            assert!(validate_request(&cfg(), &VlaRequest::provided(&o, &n)).is_err(), "{non_finite}");
        }
    }
}