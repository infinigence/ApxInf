//! Request binding and fixed-profile lifetime for the existing VLA interface.
use super::{model::Model, weights::Weights};
use crate::vla::*;
use apxinf_core::{DType, Device, Error, NormalGenerator, Result, Tensor};
use apxinf_cuda_new::ops::GemmPolicy;
use apxinf_cuda_new::{AttentionPolicy, CapturedGraph, CudaBuffer, CudaContext, ExecutionSession};
use std::{cell::RefCell, rc::Rc};

fn invalid(message: &str) -> Error {
    Error::Other(format!("Cosmos Policy: {message}"))
}
fn validate_spec(spec: &InferenceSpec) -> Result<()> {
    if spec.token_count != 1 || spec.image_layout != Some(ImageLayout::Nhwc) {
        return Err(invalid(
            "LIBERO direct requires one cached-task ID and NHWC RGB",
        ));
    }
    Ok(())
}
fn validate_request(request: &VlaRequest<'_>) -> Result<()> {
    validate_spec(&request.observation.inference_spec())?;
    let o = request.observation;
    match &o.vision {
        VisionObservation::RgbU8 {
            bytes,
            layout: ImageLayout::Nhwc,
        } if bytes.len() == 2 * 224 * 224 * 3 => {}
        _ => return Err(invalid("expected two 224x224 RGB images")),
    }
    if o.token_ids[0] >= 40 {
        return Err(invalid("cached-task ID is outside the supported table"));
    }
    let state = o
        .state
        .as_ref()
        .ok_or_else(|| invalid("normalized proprioception is required"))?;
    if state.device() != Device::Cpu
        || state.dtype() != DType::F32
        || state.shape().dims() != [9]
        || state.as_f32()?.iter().any(|v| !v.is_finite())
    {
        return Err(invalid("proprioception must be finite host FP32 [9]"));
    }
    if o.action_mask.is_some()
        || request.metadata.attention_mask.is_some()
        || request.metadata.image_grid_thw.is_some()
        || request.metadata.embodiment_id.is_some()
        || request.metadata.planning.is_some()
    {
        return Err(invalid(
            "direct profile does not take masks, embodiment overrides or planning options",
        ));
    }
    if let InitialLatent::Provided(noise) = request.initial_latent {
        if noise.device() != Device::Cpu
            || noise.dtype() != DType::F32
            || noise.shape().dims() != [1, 16, 9, 28, 28]
            || noise.as_f32()?.iter().any(|v| !v.is_finite())
        {
            return Err(invalid(
                "supplied noise must be finite host FP32 [1,16,9,28,28]",
            ));
        }
    }
    Ok(())
}
fn upload_f32(t: &Tensor, values: &[f32]) -> Result<()> {
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    CudaBuffer::from_tensor(t)
        .map_err(Error::Cuda)?
        .copy_from_host(&bytes)
        .map_err(Error::Cuda)
}
struct PlanState {
    // Graph is dropped before its model/session owners.
    graph: Option<CapturedGraph>,
    model: Model,
    session: ExecutionSession,
    generator: Box<dyn NormalGenerator>,
    gemm: GemmPolicy,
    attention: AttentionPolicy,
}
impl PlanState {
    fn bind(&mut self, request: &VlaRequest<'_>) -> Result<()> {
        validate_request(request)?;
        self.model.ctx.synchronize().map_err(Error::Cuda)?;
        if let VisionObservation::RgbU8 { bytes, .. } = &request.observation.vision {
            self.model.rgb.copy_from_host(bytes).map_err(Error::Cuda)?;
        }
        upload_f32(
            &self.model.proprio,
            request.observation.state.as_ref().unwrap().as_f32()?,
        )?;
        CudaBuffer::from_tensor(&self.model.task)
            .map_err(Error::Cuda)?
            .copy_from_host(&(request.observation.token_ids[0] as i32).to_ne_bytes())
            .map_err(Error::Cuda)?;
        match request.initial_latent {
            InitialLatent::Provided(noise) => upload_f32(&self.model.noise, noise.as_f32()?)?,
            InitialLatent::Generate { rng } => {
                self.generator.generate(rng)?;
            }
        }
        Ok(())
    }
    fn compute(&mut self) -> Result<()> {
        if let Some(graph) = &self.graph {
            graph.replay()?;
        } else {
            let Self {
                model,
                session,
                gemm,
                attention,
                ..
            } = self;
            session.run(|| model.forward(gemm, attention))?;
        }
        self.model.ctx.synchronize().map_err(Error::Cuda)
    }
}

#[derive(Clone)]
struct Plan {
    spec: InferenceSpec,
    status: PreparationStatus,
    state: Rc<RefCell<PlanState>>,
}
impl PreparedInference for Plan {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }
    fn status(&self) -> PreparationStatus {
        self.status.clone()
    }
    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        if !self.spec.matches(request.observation) {
            return Err(invalid("request does not match prepared profile"));
        }
        let mut state = self
            .state
            .try_borrow_mut()
            .map_err(|_| invalid("prepared inference is already running"))?;
        state.bind(request)?;
        state.compute()?;
        // Publish independent device storage. Retained actions survive replay,
        // plan eviction and runner destruction without exposing scratch buffers.
        let buffer =
            CudaBuffer::alloc(16 * 7 * 4, state.model.ctx.device_id()).map_err(Error::Cuda)?;
        let output = buffer
            .as_tensor(state.model.actions.shape().clone(), DType::F32)
            .map_err(Error::Cuda)?;
        apxinf_cuda_new::transfers::copy_cuda_to_cuda(
            &state.model.ctx,
            &state.model.actions,
            &output,
        )?;
        state.model.ctx.synchronize().map_err(Error::Cuda)?;
        Ok(Action::new(output))
    }
}

pub(super) struct ModelRunner {
    ctx: Rc<CudaContext>,
    weights: Rc<Weights>,
    gemm: GemmPolicy,
    attention: AttentionPolicy,
    implicit: RefCell<Option<Plan>>,
}
impl ModelRunner {
    pub fn new(
        ctx: Rc<CudaContext>,
        weights: Rc<Weights>,
        gemm: GemmPolicy,
        attention: AttentionPolicy,
    ) -> Self {
        Self {
            ctx,
            weights,
            gemm,
            attention,
            implicit: RefCell::new(None),
        }
    }
    fn plan(
        &self,
        spec: &InferenceSpec,
        policy: ExecutionPolicy,
        sample: Option<&VlaRequest<'_>>,
    ) -> Result<Plan> {
        validate_spec(spec)?;
        if let Some(sample) = sample {
            validate_request(sample)?;
            if !spec.matches(sample.observation) {
                return Err(invalid("sample/spec mismatch"));
            }
        }
        let model = Model::new(self.ctx.clone(), self.weights.clone())?;
        let generator =
            apxinf_cuda_new::sampling::create_normal_generator(&self.ctx, model.noise.clone())?;
        let mut gemm = self.gemm.clone();
        let mut attention = self.attention.clone();
        if sample.is_none() {
            gemm.online_tune = false;
            attention.online_tune = false;
        }
        let mut state = PlanState {
            graph: None,
            model,
            session: ExecutionSession::with_capacity(256 * 1024 * 1024, self.ctx.device_id())?,
            generator,
            gemm,
            attention,
        };
        if let Some(sample) = sample {
            state.bind(sample)?;
        } else {
            state
                .model
                .rgb
                .copy_from_host(&vec![0; 2 * 224 * 224 * 3])
                .map_err(Error::Cuda)?;
            upload_f32(&state.model.proprio, &[0.; 9])?;
            upload_f32(&state.model.noise, &vec![0.; 16 * 9 * 28 * 28])?;
            CudaBuffer::from_tensor(&state.model.task)
                .map_err(Error::Cuda)?
                .copy_from_host(&0i32.to_ne_bytes())
                .map_err(Error::Cuda)?;
        }
        {
            let PlanState {
                model,
                session,
                gemm,
                attention,
                ..
            } = &mut state;
            session.prepare(|| model.forward(gemm, attention))?;
        }
        self.ctx.synchronize().map_err(Error::Cuda)?;
        state.gemm.online_tune = false;
        state.attention.online_tune = false;
        let mut mode = ExecutionMode::Eager;
        let mut fallback_reason = None;
        if policy != ExecutionPolicy::Eager {
            let PlanState {
                model,
                session,
                gemm,
                attention,
                ..
            } = &mut state;
            match apxinf_cuda_new::capture(&self.ctx, || {
                session.run(|| model.forward(gemm, attention))
            }) {
                Ok(graph) => {
                    state.graph = Some(graph);
                    mode = ExecutionMode::Graph;
                }
                Err(e) if policy == ExecutionPolicy::PreferGraph => {
                    fallback_reason = Some(e.to_string())
                }
                Err(e) => return Err(e),
            }
        }
        Ok(Plan {
            spec: *spec,
            status: PreparationStatus::Ready {
                mode,
                fallback_reason,
            },
            state: Rc::new(RefCell::new(state)),
        })
    }
}
impl VlaRuntime for ModelRunner {
    fn model_variant(&self) -> Option<&'static str> {
        Some("bf16")
    }
    fn contract(&self) -> VlaContract {
        VlaContract {
            action_shape: [16, 7],
            patch_shape: [224 * 224 * 2, 3],
            max_token_len: 1,
            num_views: 2,
            image_size: 224,
            patch_size: 1,
            accepts_rgb_u8: true,
        }
    }
    fn initial_latent_shape(&self) -> Vec<usize> {
        vec![1, 16, 9, 28, 28]
    }
    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        validate_request(request)?;
        if self.implicit.borrow().is_none() {
            let plan = self.plan(
                &request.observation.inference_spec(),
                ExecutionPolicy::RequireGraph,
                Some(request),
            )?;
            *self.implicit.borrow_mut() = Some(plan);
        }
        let plan = self.implicit.borrow().as_ref().unwrap().clone();
        plan.run(request)
    }
    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        self.prepare_with_policy(spec, ExecutionPolicy::RequireGraph)
    }
    fn prepare_with_policy(
        &self,
        spec: &InferenceSpec,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        Ok(Box::new(self.plan(spec, policy, None)?))
    }
    fn prepare_for(
        &self,
        sample: &VlaRequest<'_>,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        Ok(Box::new(self.plan(
            &sample.observation.inference_spec(),
            policy,
            Some(sample),
        )?))
    }
    fn clear_prepared(&self) -> Result<()> {
        *self.implicit.borrow_mut() = None;
        Ok(())
    }
    fn execution_mode(&self) -> &'static str {
        match self.implicit.borrow().as_ref().map(|p| &p.status) {
            Some(PreparationStatus::Ready {
                mode: ExecutionMode::Graph,
                ..
            }) => "graph",
            Some(PreparationStatus::Ready {
                mode: ExecutionMode::Eager,
                ..
            }) => "eager",
            _ => "unprepared",
        }
    }
    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        let action = self.infer(request)?;
        let mut bytes = vec![0; 16 * 7 * 4];
        CudaBuffer::from_tensor(action.tensor())
            .map_err(Error::Cuda)?
            .copy_to_host(&mut bytes)
            .map_err(Error::Cuda)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{path::PathBuf, time::Instant};
    fn host(t: &Tensor) -> Result<Vec<f32>> {
        let mut bytes = vec![0; t.size_in_bytes()];
        CudaBuffer::from_tensor(t)
            .map_err(Error::Cuda)?
            .copy_to_host(&mut bytes)
            .map_err(Error::Cuda)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect())
    }
    fn observation() -> Observation {
        Observation {
            vision: VisionObservation::RgbU8 {
                bytes: vec![0; 2 * 224 * 224 * 3],
                layout: ImageLayout::Nhwc,
            },
            token_ids: vec![0],
            state: Some(Tensor::from_f32(vec![9], &[0.; 9]).unwrap()),
            action_mask: None,
        }
    }
    #[test]
    fn direct_request_validation_rejects_action_only_noise_and_wrong_metadata() {
        let mut o = observation();
        let n = Tensor::zeros(vec![1, 16, 9, 28, 28], DType::F32);
        assert!(validate_request(&VlaRequest::provided(&o, &n)).is_ok());
        let action = Tensor::zeros(vec![16, 7], DType::F32);
        assert!(validate_request(&VlaRequest::provided(&o, &action)).is_err());
        o.token_ids[0] = 40;
        assert!(validate_request(&VlaRequest::provided(&o, &n)).is_err());
        o.token_ids[0] = 0;
        o.state = Some(Tensor::from_f32(vec![9], &[f32::NAN; 9]).unwrap());
        assert!(validate_request(&VlaRequest::provided(&o, &n)).is_err());
        o = observation();
        let planning = PlanningOptions::default();
        let mut request = VlaRequest::provided(&o, &n);
        request.metadata.planning = Some(&planning);
        assert!(validate_request(&request).is_err());
        assert!(validate_spec(&InferenceSpec {
            token_count: 1,
            image_layout: Some(ImageLayout::Nchw)
        })
        .is_err());
    }
    #[test]
    #[ignore = "requires Thor, pinned converted checkpoint and official joint reference fixture"]
    fn official_joint_actions_preparation_and_lifetime() -> Result<()> {
        let root =
            PathBuf::from(std::env::var("APXINF_COSMOS_EVIDENCE").expect("APXINF_COSMOS_EVIDENCE"));
        let (fixture, _) =
            apxinf_loader::safetensors::load_native_path(&root.join("joint-fixture.safetensors"))
                .map_err(|e| Error::Other(e.to_string()))?;
        let mut o = observation();
        o.vision = VisionObservation::RgbU8 {
            bytes: std::fs::read(root.join("public-rgb.bin"))?,
            layout: ImageLayout::Nhwc,
        };
        o.token_ids = vec![std::fs::read_to_string(root.join("public-task-id.txt"))?
            .trim()
            .parse()
            .map_err(|_| invalid("fixture task ID"))?];
        o.state = Some(fixture["state"].clone());
        let request = VlaRequest::provided(&o, &fixture["noise"]);
        let tune = std::env::var("APXINF_COSMOS_TUNE").as_deref() == Ok("1");
        let ctx = Rc::new(CudaContext::new_with_autotune(0, tune).map_err(Error::Cuda)?);
        let cache_dir = Some(root.join("recipes").to_string_lossy().into_owned());
        let weights = Rc::new(Weights::load(&root.join("native-assets"), 0)?);
        let gemm = GemmPolicy {
            online_tune: tune,
            cache_dir: cache_dir.clone(),
            ..Default::default()
        };
        let attention = AttentionPolicy {
            online_tune: tune,
            cache_dir,
            ..Default::default()
        };
        let runner = ModelRunner::new(
            ctx.clone(),
            weights.clone(),
            gemm.clone(),
            attention.clone(),
        );
        eprintln!("Preparing eager full model (autotune={tune})");
        let eager = runner.plan(&o.inference_spec(), ExecutionPolicy::Eager, Some(&request))?;
        eprintln!("Eager full model prepared; executing fixed reference request");
        assert_eq!(
            eager.status(),
            PreparationStatus::Ready {
                mode: ExecutionMode::Eager,
                fallback_reason: None
            }
        );
        let retained = eager.run(&request)?;
        let eager_action = host(retained.tensor())?;
        let joint = host(&eager.state.borrow().model.joint)?;
        std::fs::write(
            root.join("joint-native-output.f32"),
            joint
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
        )?;
        let expected = fixture["expected_joint"].as_f32()?;
        for t in 0..9 {
            let (mut aa, mut bb, mut ab, mut ee, mut max_abs) = (0f64, 0f64, 0f64, 0f64, 0f64);
            for c in 0..16 {
                for p in 0..784 {
                    let i = (c * 9 + t) * 784 + p;
                    let (a, b) = (joint[i] as f64, expected[i] as f64);
                    aa += a * a;
                    bb += b * b;
                    ab += a * b;
                    ee += (a - b).powi(2);
                    max_abs = max_abs.max((a - b).abs());
                }
            }
            let relative = (ee / bb.max(1e-30)).sqrt();
            let cosine = ab / (aa * bb).sqrt();
            eprintln!("Joint frame {t}: relative_l2={relative} cosine={cosine} max_abs={max_abs}");
            assert!(
                relative <= 0.02 && cosine >= 0.999,
                "joint frame {t} differs"
            );
        }
        let expected = fixture["expected_actions"].as_f32()?;
        let max_abs = eager_action
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let rmse = (eager_action
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / 112.)
            .sqrt();
        eprintln!("Normalized actions: max_abs={max_abs} rmse={rmse}");
        assert!(max_abs <= 0.03 && rmse <= 0.01);
        for i in 0..16 {
            assert_eq!(eager_action[i * 7 + 6] > 0., expected[i * 7 + 6] > 0.);
        }
        drop(eager);
        eprintln!("Preparing full model graph");
        let graph = runner.plan(
            &o.inference_spec(),
            ExecutionPolicy::RequireGraph,
            Some(&request),
        )?;
        assert_eq!(
            graph.status(),
            PreparationStatus::Ready {
                mode: ExecutionMode::Graph,
                fallback_reason: None
            }
        );
        eprintln!("Full model graph prepared; checking replay and lifecycle");
        for _ in 0..3 {
            assert_eq!(host(graph.run(&request)?.tensor())?, eager_action);
        }
        assert_eq!(host(retained.tensor())?, eager_action);
        let mut changed = o.clone();
        changed.state = Some(Tensor::from_f32(vec![9], &[0.125; 9])?);
        let changed_result = host(
            graph
                .run(&VlaRequest::provided(&changed, &fixture["noise"]))?
                .tensor(),
        )?;
        assert_ne!(changed_result, eager_action);
        assert_eq!(host(graph.run(&request)?.tensor())?, eager_action);
        let mut changed_image = o.clone();
        if let VisionObservation::RgbU8 { bytes, .. } = &mut changed_image.vision {
            for pixel in &mut bytes[..224 * 224 * 3] {
                *pixel = 255 - *pixel;
            }
        }
        assert_ne!(
            host(
                graph
                    .run(&VlaRequest::provided(&changed_image, &fixture["noise"]))?
                    .tensor()
            )?,
            eager_action
        );
        let mut changed_task = o.clone();
        changed_task.token_ids[0] = (changed_task.token_ids[0] + 1) % 40;
        assert_ne!(
            host(
                graph
                    .run(&VlaRequest::provided(&changed_task, &fixture["noise"]))?
                    .tensor()
            )?,
            eager_action
        );
        let wrong_noise = Tensor::zeros(vec![16, 7], DType::F32);
        assert!(graph.run(&VlaRequest::provided(&o, &wrong_noise)).is_err());
        assert_eq!(host(graph.run(&request)?.tensor())?, eager_action);
        let rng = apxinf_core::RngKey::new(123, 0, 0);
        let random = host(graph.run(&VlaRequest::generated(&o, rng))?.tensor())?;
        assert_eq!(
            host(graph.run(&VlaRequest::generated(&o, rng))?.tensor())?,
            random
        );
        assert_ne!(
            host(
                graph
                    .run(&VlaRequest::generated(
                        &o,
                        apxinf_core::RngKey::new(124, 0, 0)
                    ))?
                    .tensor()
            )?,
            random
        );
        assert_eq!(host(runner.infer(&request)?.tensor())?, eager_action);
        assert_eq!(runner.execution_mode(), "graph");
        runner.clear_prepared()?;
        assert_eq!(runner.execution_mode(), "unprepared");
        drop(runner);
        assert_eq!(host(graph.run(&request)?.tensor())?, eager_action);
        let mut milliseconds = Vec::new();
        for _ in 0..2 {
            graph.run(&request)?;
        }
        for _ in 0..5 {
            let start = Instant::now();
            graph.run(&request)?;
            milliseconds.push(start.elapsed().as_secs_f64() * 1000.);
        }
        let memory = ctx.memory_info().map_err(Error::Cuda)?;
        std::fs::write(root.join("native-benchmark.json"),serde_json::to_vec_pretty(&serde_json::json!({
            "milliseconds":milliseconds,"includes":"request binding, full model graph, independent action publication and synchronization",
            "cuda_free_bytes_after":memory.0,"cuda_total_bytes":memory.1,
            "memory_caveat":"Thor unified system memory; not a per-process allocator peak",
            "normalized_action_max_abs":max_abs,"normalized_action_rmse":rmse
        })).unwrap())?;
        drop(graph);
        assert_eq!(host(retained.tensor())?, eager_action);
        let runner = ModelRunner::new(ctx, weights, gemm, attention);
        let placeholder =
            runner.prepare_with_policy(&o.inference_spec(), ExecutionPolicy::RequireGraph)?;
        assert_eq!(host(placeholder.run(&request)?.tensor())?, eager_action);
        drop(runner);
        assert_eq!(host(placeholder.run(&request)?.tensor())?, eager_action);
        drop(placeholder);
        assert_eq!(host(retained.tensor())?, eager_action);
        Ok(())
    }
}
