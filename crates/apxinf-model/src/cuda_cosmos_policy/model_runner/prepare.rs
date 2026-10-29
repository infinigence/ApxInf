//! Request binding and fixed-profile lifetime for the existing VLA interface.
use crate::cuda_cosmos_policy::backend::ops::GemmPolicy;
use crate::cuda_cosmos_policy::backend::{
    AttentionPolicy, CapturedGraph, CudaBuffer, CudaContext, ExecutionSession,
};
use crate::cuda_cosmos_policy::model::{CosmosPolicyModel, Inputs};
use crate::vla::*;
use apxinf_core::{DType, Error, NormalGenerator, Result, Tensor};
use std::{cell::RefCell, rc::Rc};

use super::runner::{invalid, validate_request, validate_spec, ModelRunner};
impl Inputs {
    fn allocate(ctx: &CudaContext) -> Result<Self> {
        let tensor = |shape: Vec<usize>, dtype: DType| {
            let bytes = shape.iter().product::<usize>() * dtype.size_in_bytes();
            CudaBuffer::alloc(bytes, ctx.device_id())
                .map_err(Error::Cuda)?
                .as_tensor(shape.into(), dtype)
                .map_err(Error::Cuda)
        };
        Ok(Self {
            rgb: CudaBuffer::alloc(2 * 224 * 224 * 3, ctx.device_id()).map_err(Error::Cuda)?,
            proprio: tensor(vec![1, 9], DType::F32)?,
            task: tensor(vec![1], DType::I32)?,
            noise: tensor(vec![1, 16 * 9 * 28 * 28], DType::F32)?,
        })
    }
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
    model: CosmosPolicyModel,
    ctx: Rc<CudaContext>,
    inputs: Inputs,
    session: ExecutionSession,
    generator: Box<dyn NormalGenerator>,
    gemm: GemmPolicy,
    attention: AttentionPolicy,
}
impl PlanState {
    fn bind(&mut self, request: &VlaRequest<'_>) -> Result<()> {
        validate_request(request)?;
        self.ctx.synchronize().map_err(Error::Cuda)?;
        if let VisionObservation::RgbU8 { bytes, .. } = &request.observation.vision {
            self.inputs.rgb.copy_from_host(bytes).map_err(Error::Cuda)?;
        }
        upload_f32(
            &self.inputs.proprio,
            request.observation.state.as_ref().unwrap().as_f32()?,
        )?;
        CudaBuffer::from_tensor(&self.inputs.task)
            .map_err(Error::Cuda)?
            .copy_from_host(&(request.observation.token_ids[0] as i32).to_ne_bytes())
            .map_err(Error::Cuda)?;
        match request.initial_latent {
            InitialLatent::Provided(noise) => upload_f32(&self.inputs.noise, noise.as_f32()?)?,
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
        self.ctx.synchronize().map_err(Error::Cuda)
    }
}

#[derive(Clone)]
pub(super) struct Plan {
    spec: InferenceSpec,
    pub(super) status: PreparationStatus,
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
        let buffer = CudaBuffer::alloc(16 * 7 * 4, state.ctx.device_id()).map_err(Error::Cuda)?;
        let output = buffer
            .as_tensor(state.model.actions().shape().clone(), DType::F32)
            .map_err(Error::Cuda)?;
        crate::cuda_cosmos_policy::backend::transfers::copy_cuda_to_cuda(
            &state.ctx,
            state.model.actions(),
            &output,
        )?;
        state.ctx.synchronize().map_err(Error::Cuda)?;
        Ok(Action::new(output))
    }
}

impl ModelRunner {
    pub(super) fn plan(
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
        let inputs = Inputs::allocate(&self.ctx)?;
        let model = CosmosPolicyModel::new(self.ctx.clone(), self.weights.clone(), &inputs)?;
        let workspace_bytes = model.workspace_bytes();
        let generator = crate::cuda_cosmos_policy::backend::sampling::create_normal_generator(
            &self.ctx,
            inputs.noise.clone(),
        )?;
        let mut gemm = self.gemm.clone();
        let mut attention = self.attention.clone();
        if sample.is_none() {
            gemm.online_tune = false;
            attention.online_tune = false;
        }
        let mut state = PlanState {
            graph: None,
            model,
            ctx: self.ctx.clone(),
            inputs,
            session: ExecutionSession::with_capacity(workspace_bytes, self.ctx.device_id())?,
            generator,
            gemm,
            attention,
        };
        if let Some(sample) = sample {
            state.bind(sample)?;
        } else {
            state
                .inputs
                .rgb
                .copy_from_host(&vec![0; 2 * 224 * 224 * 3])
                .map_err(Error::Cuda)?;
            upload_f32(&state.inputs.proprio, &[0.; 9])?;
            upload_f32(&state.inputs.noise, &vec![0.; 16 * 9 * 28 * 28])?;
            CudaBuffer::from_tensor(&state.inputs.task)
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
            match crate::cuda_cosmos_policy::backend::capture(&self.ctx, || {
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
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_cosmos_policy::weights::Bf16Weights;
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
        let weights = Rc::new(Bf16Weights::load(&root.join("native-assets"), 0)?);
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
        let joint = host(eager.state.borrow().model.joint())?;
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
