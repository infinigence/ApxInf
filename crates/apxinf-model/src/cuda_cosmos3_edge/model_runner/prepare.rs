//! Request binding and fixed-profile lifetime for the existing VLA interface.
use crate::cuda_cosmos3_edge::backend::ops::GemmPolicy;
use crate::cuda_cosmos3_edge::backend::{
    AttentionPolicy, CapturedGraph, Conv3dPolicy, CudaBuffer, CudaContext, ExecutionSession,
};
use crate::cuda_cosmos3_edge::model::{Cosmos3EdgeModel, Inputs};
use crate::vla::*;
use apxinf_core::{DType, Error, NormalGenerator, Result, Tensor};
use std::{cell::RefCell, rc::Rc};

use super::runner::{
    invalid, validate_request, validate_spec, ModelRunner, LATENT_SHAPE, NUM_TASKS, PIXEL_FRAMES,
    PIXEL_H, PIXEL_W,
};

fn tensor(ctx: &CudaContext, shape: Vec<usize>, dtype: DType) -> Result<Tensor> {
    let bytes = shape.iter().product::<usize>() * dtype.size_in_bytes();
    CudaBuffer::alloc(bytes, ctx.device_id())
        .map_err(Error::Cuda)?
        .as_tensor(shape.into(), dtype)
        .map_err(Error::Cuda)
}

impl Inputs {
    fn allocate(ctx: &CudaContext, text_tokens: usize) -> Result<Self> {
        Ok(Self {
            text: tensor(ctx, vec![1, text_tokens * 2048], DType::BF16)?,
            noise: tensor(ctx, LATENT_SHAPE.to_vec(), DType::F32)?,
            pixel: tensor(ctx, vec![1, 272, 368, 12], DType::BF16)?,
            rgb: CudaBuffer::alloc(PIXEL_FRAMES * PIXEL_H * PIXEL_W * 3, ctx.device_id())
                .map_err(Error::Cuda)?,
        })
    }
}
/// VAE convolutions follow the GEMM tuning/fallback/cache controls.
fn conv_policy(gemm: &GemmPolicy) -> Conv3dPolicy {
    Conv3dPolicy {
        online_tune: gemm.online_tune,
        allow_fallback: gemm.allow_fallback,
        cache_dir: gemm.cache_dir.clone(),
    }
}
fn upload_f32(ctx: &CudaContext, t: &Tensor, values: &[f32]) -> Result<()> {
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    CudaBuffer::from_tensor(t)
        .map_err(Error::Cuda)?
        .copy_from_host_on(&bytes, ctx.stream())
        .map_err(Error::Cuda)
}
struct PlanState {
    graph: Option<CapturedGraph>,
    model: Cosmos3EdgeModel,
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
        if let VisionObservation::RgbU8 { bytes, .. } = &request.observation.vision {
            self.inputs
                .rgb
                .copy_from_host_on(bytes, self.ctx.stream())
                .map_err(Error::Cuda)?;
        }
        match request.initial_latent {
            InitialLatent::Provided(noise) => {
                upload_f32(&self.ctx, &self.inputs.noise, noise.as_f32()?)?
            }
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
        let action = state.model.actions();
        let buffer = CudaBuffer::alloc(16 * 10 * 4, state.ctx.device_id()).map_err(Error::Cuda)?;
        let output = buffer
            .as_tensor(action.shape().clone(), DType::F32)
            .map_err(Error::Cuda)?;
        crate::cuda_cosmos3_edge::backend::quantization(
            &state.ctx,
            crate::cuda_cosmos3_edge::backend::QuantizationArgs::new(
                crate::cuda_cosmos3_edge::backend::QuantizationSemantic::CastBf16ToF32,
                action,
                &mut output.clone(),
            ),
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
        let inputs = Inputs::allocate(&self.ctx, self.weights.text_tokens)?;
        // Single cached prompt: the text table already matches the input buffer.
        crate::cuda_cosmos3_edge::backend::transfers::copy_cuda_to_cuda(
            &self.ctx,
            &self.weights.text,
            &inputs.text,
        )?;
        let mut gemm = self.gemm.clone();
        let mut attention = self.attention.clone();
        if sample.is_none() {
            gemm.online_tune = false;
            attention.online_tune = false;
        }
        let convolution = conv_policy(&gemm);
        let model = Cosmos3EdgeModel::new(
            self.ctx.clone(),
            self.weights.clone(),
            &inputs,
            &convolution,
            self.num_inference_steps,
        )?;
        let workspace_bytes = model.workspace_bytes();
        let generator = crate::cuda_cosmos3_edge::backend::sampling::create_normal_generator(
            &self.ctx,
            inputs.noise.clone(),
        )?;
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
                .copy_from_host_on(
                    &vec![0; PIXEL_FRAMES * PIXEL_H * PIXEL_W * 3],
                    state.ctx.stream(),
                )
                .map_err(Error::Cuda)?;
            upload_f32(
                &state.ctx,
                &state.inputs.noise,
                &vec![0.; LATENT_SHAPE.iter().product()],
            )?;
        }
        {
            let PlanState {
                model,
                session,
                gemm,
                attention,
                ..
            } = &mut state;
            model.prepare_constants(gemm, attention)?;
            session.prepare(|| model.forward(gemm, attention))?;
        }
        // Capture replays exactly the operators prepared above, so keep the
        // policies unchanged: the execution key includes `online_tune`, and
        // flipping it here would turn every lookup into a cache miss.
        self.ctx.synchronize().map_err(Error::Cuda)?;
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
            match crate::cuda_cosmos3_edge::backend::capture(&self.ctx, || {
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
    use crate::auto::{AutoModel, LoadOptions, LoadedModel};
    use crate::cuda_cosmos3_edge::model_runner::runner::{ACTION_SHAPE, PIXEL_H, PIXEL_W};
    use apxinf_core::Device;
    use std::path::{Path, PathBuf};

    fn floats(path: &Path) -> Result<Vec<f32>> {
        let bytes = std::fs::read(path)?;
        assert_eq!(bytes.len() % 4, 0);
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    }
    fn host(action: &Action) -> Result<Vec<f32>> {
        let mut bytes = vec![0; ACTION_SHAPE.iter().product::<usize>() * 4];
        CudaBuffer::from_tensor(action.tensor())
            .map_err(Error::Cuda)?
            .copy_to_host(&mut bytes)
            .map_err(Error::Cuda)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect())
    }

    /// Pinned independent FP32 reference, full-graph parity, rebinding and ownership.
    /// Fixture: rgb.u8, noise.f32, actions.f32. Assets are read in place.
    #[test]
    #[ignore = "requires CUDA, converted DROID assets and independent reference fixture"]
    fn cosmos3_edge_reference_and_preparation_lifetime() -> Result<()> {
        let root =
            PathBuf::from(std::env::var("APXINF_COSMOS3_ASSETS").expect("APXINF_COSMOS3_ASSETS"));
        let fixture =
            PathBuf::from(std::env::var("APXINF_COSMOS3_FIXTURE").expect("APXINF_COSMOS3_FIXTURE"));
        let output = PathBuf::from(
            std::env::var("APXINF_COSMOS3_OUTPUT_DIR").expect("APXINF_COSMOS3_OUTPUT_DIR"),
        );
        std::fs::create_dir_all(&output)?;
        let observation = Observation {
            vision: VisionObservation::RgbU8 {
                bytes: std::fs::read(fixture.join("rgb.u8"))?,
                layout: ImageLayout::Nhwc,
            },
            token_ids: vec![0],
            state: None,
            action_mask: None,
        };
        let noise = Tensor::from_f32(LATENT_SHAPE.to_vec(), &floats(&fixture.join("noise.f32"))?)?;
        let request = VlaRequest::provided(&observation, &noise);
        let options = LoadOptions {
            model_name: Some("cosmos3-edge".into()),
            model_variant: Some("bf16".into()),
            autotune: std::env::var_os("APXINF_COSMOS3_AUTOTUNE").is_some(),
            tuning_path: std::env::var("APXINF_COSMOS3_RECIPES")
                .ok()
                .map(PathBuf::from),
            ..Default::default()
        };
        let loaded = AutoModel::load_model(Device::Cuda(0), &root, &options)?;
        let LoadedModel::Vla(runner) = loaded else {
            panic!("expected VLA")
        };
        assert_eq!(runner.contract().action_shape, ACTION_SHAPE);
        assert_eq!(runner.initial_latent_shape(), LATENT_SHAPE);
        let eager = runner.prepare_for(&request, ExecutionPolicy::Eager)?;
        let original = host(&eager.run(&request)?)?;
        assert_eq!(
            host(&eager.run(&request)?)?,
            original,
            "request solver state was retained"
        );
        let reference = floats(&fixture.join("actions.f32"))?;
        assert_eq!(reference.len(), original.len());
        assert!(original.iter().all(|v| v.is_finite()));
        let dot: f64 = original
            .iter()
            .zip(&reference)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        let cosine = dot / (norm(&original) * norm(&reference));
        let max_abs = original
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let rmse = (original
            .iter()
            .zip(&reference)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum::<f64>()
            / original.len() as f64)
            .sqrt();
        assert!(
            cosine >= 0.9999 && max_abs <= 0.03 && rmse <= 0.01,
            "independent FP32 reference: cosine={cosine}, max_abs={max_abs}, rmse={rmse}"
        );
        let mut alternate = observation.clone();
        if let VisionObservation::RgbU8 { bytes, .. } = &mut alternate.vision {
            assert_eq!(bytes.len(), PIXEL_H * PIXEL_W * 3);
            bytes.iter_mut().for_each(|p| *p = 255 - *p);
        }
        let alternate_noise = Tensor::from_f32(
            LATENT_SHAPE.to_vec(),
            &noise.as_f32()?.iter().map(|v| -*v).collect::<Vec<_>>(),
        )?;
        let rgb_request = VlaRequest::provided(&alternate, &noise);
        let noise_request = VlaRequest::provided(&observation, &alternate_noise);
        let rgb_changed = host(&eager.run(&rgb_request)?)?;
        let noise_changed = host(&eager.run(&noise_request)?)?;
        assert_ne!(
            rgb_changed, original,
            "RGB rebinding did not reach model output"
        );
        assert_ne!(
            noise_changed, original,
            "noise rebinding did not reach model output"
        );
        let alternate_request = VlaRequest::provided(&alternate, &alternate_noise);
        let changed = host(&eager.run(&alternate_request)?)?;
        assert_ne!(changed, original);
        assert_eq!(host(&eager.run(&request)?)?, original);
        drop(eager);
        let placeholder = runner
            .prepare_with_policy(&observation.inference_spec(), ExecutionPolicy::RequireGraph)?;
        assert_eq!(host(&placeholder.run(&request)?)?, original);
        drop(placeholder);
        let graph = runner.prepare_for(&request, ExecutionPolicy::RequireGraph)?;
        assert!(matches!(
            graph.status(),
            PreparationStatus::Ready {
                mode: ExecutionMode::Graph,
                ..
            }
        ));
        assert_eq!(host(&graph.run(&rgb_request)?)?, rgb_changed);
        assert_eq!(host(&graph.run(&noise_request)?)?, noise_changed);
        let retained = graph.run(&request)?;
        assert_eq!(host(&retained)?, original);
        for _ in 0..4 {
            assert_eq!(host(&graph.run(&alternate_request)?)?, changed);
            assert_eq!(host(&graph.run(&request)?)?, original);
        }
        for _ in 0..5 {
            graph.run(&request)?;
        }
        let mut milliseconds = Vec::new();
        for _ in 0..20 {
            let begin = std::time::Instant::now();
            graph.run(&request)?;
            milliseconds.push(begin.elapsed().as_secs_f64() * 1000.0);
        }
        let mut sorted = milliseconds.clone();
        sorted.sort_by(f64::total_cmp);
        let median_ms = (sorted[9] + sorted[10]) / 2.0;
        let p95_ms = sorted[18];
        let generated = VlaRequest::generated(&observation, apxinf_core::RngKey::new(19, 0, 0));
        let generated_result = host(&graph.run(&generated)?)?;
        assert_eq!(host(&graph.run(&generated)?)?, generated_result);
        assert_ne!(generated_result, original);
        assert_ne!(
            host(&graph.run(&VlaRequest::generated(
                &observation,
                apxinf_core::RngKey::new(19, 0, 1)
            ))?)?,
            generated_result
        );
        let wrong_noise = Tensor::zeros(vec![16, 10], DType::F32);
        assert!(graph
            .run(&VlaRequest::provided(&observation, &wrong_noise))
            .is_err());
        assert_eq!(host(&graph.run(&request)?)?, original);
        runner.clear_prepared()?;
        drop(runner);
        assert_eq!(host(&graph.run(&request)?)?, original);
        drop(graph);
        assert_eq!(host(&retained)?, original);
        let report = serde_json::json!({"cosine":cosine,"max_abs":max_abs,"rmse":rmse,
            "eager_graph_bitwise":true,"changed_rgb_and_noise":true,"retained_action":true, "request_milliseconds":milliseconds,
            "request_median_ms":median_ms,"request_p95_ms":p95_ms,
            "timing_scope":"prepared run: RGB/noise upload + whole graph + independent device action publication; excludes D2H"});
        std::fs::write(
            output.join("acceptance.json"),
            serde_json::to_vec_pretty(&report).unwrap(),
        )?;
        std::fs::write(
            output.join("actions.f32"),
            original
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        eprintln!("COSMOS3_EDGE_ACCEPTANCE {report}");
        Ok(())
    }
}
