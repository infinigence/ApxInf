//! Fixed buffers, explicit request binding, RNG, and whole-model CUDA Graph
//! ownership. Forward computation is exclusively in Model.
use super::weights::host_tensor;
use super::{
    backend::{
        self, ops, sampling, transfers, Context, DeviceBuffer, ExecutionSession, PreparedPhase,
    },
    model::{Inputs, Model},
};
use crate::vla::{
    Action, ExecutionMode, ExecutionPolicy, ImageLayout, InferenceSpec, InitialLatent,
    PreparationStatus, PreparedInference, VisionObservation, VlaContract, VlaRequest, VlaRuntime,
};
use apxinf_core::{DType, Error, NormalGenerator, Result, Shape, Tensor};
use std::{cell::RefCell, rc::Rc};

pub(super) struct ModelRunner {
    ctx: Rc<Context>,
    model: Rc<Model>,
    cached: RefCell<Option<Box<dyn PreparedInference>>>,
}
impl ModelRunner {
    pub(super) fn new(ctx: Rc<Context>, model: Model) -> Self {
        Self {
            ctx,
            model: Rc::new(model),
            cached: RefCell::new(None),
        }
    }
}
fn check_spec(spec: &InferenceSpec) -> Result<()> {
    if spec.token_count != 128
        || !matches!(
            spec.image_layout,
            Some(ImageLayout::Nhwc) | Some(ImageLayout::Nchw)
        )
    {
        return Err(Error::Other(
            "Fast-WAM requires 128 padded tokens and two 224x224 RGB views".into(),
        ));
    }
    Ok(())
}
fn validate(request: &VlaRequest<'_>, spec: &InferenceSpec) -> Result<()> {
    check_spec(spec)?;
    request.observation.validate()?;
    if !spec.matches(request.observation) {
        return Err(Error::Other(
            "Fast-WAM prepared input specification changed".into(),
        ));
    }
    let obs = request.observation;
    if obs.token_ids.iter().any(|&x| x >= 256384) {
        return Err(Error::Other(
            "Fast-WAM token ID outside UMT5 vocabulary".into(),
        ));
    }
    match &obs.vision {
        VisionObservation::RgbU8 { bytes, .. } if bytes.len() == 2 * 224 * 224 * 3 => {}
        _ => return Err(Error::Other("Fast-WAM RGB byte count mismatch".into())),
    }
    let state = obs
        .state
        .as_ref()
        .ok_or_else(|| Error::Other("Fast-WAM requires normalized proprioceptive state".into()))?;
    if state.device() != apxinf_core::Device::Cpu
        || !matches!(state.shape().dims(), [8] | [1, 8])
        || state.to_f32_vec()?.iter().any(|x| !x.is_finite())
    {
        return Err(Error::Other(
            "Fast-WAM state must contain eight finite CPU values".into(),
        ));
    }
    if obs.action_mask.is_some()
        || request.metadata.embodiment_id.is_some()
        || request.metadata.image_grid_thw.is_some()
    {
        return Err(Error::Other(
            "Fast-WAM does not accept action masks, embodiment IDs or image grids".into(),
        ));
    }
    if let Some(p) = request.metadata.planning {
        if p.num_steps.is_some_and(|n| n != 10)
            || p.reasoning.is_some()
            || p.raw_rgb_resize.is_some()
        {
            return Err(Error::Other(
                "Fast-WAM supports ten action steps and already resized RGB".into(),
            ));
        }
    }
    if let Some(mask) = request.metadata.attention_mask {
        if mask.len() != 128
            || mask[0] != 1
            || mask.iter().any(|&x| x > 1)
            || mask.windows(2).any(|x| x == [0, 1])
        {
            return Err(Error::Other(
                "Fast-WAM requires a nonempty right-padded binary text mask".into(),
            ));
        }
    } else if obs.token_ids[0] == 0 || obs.token_ids.windows(2).any(|x| x[0] == 0 && x[1] != 0) {
        return Err(Error::Other(
            "Fast-WAM inferred token mask is not right padded".into(),
        ));
    }
    if let InitialLatent::Provided(noise) = request.initial_latent {
        if noise.device() != apxinf_core::Device::Cpu
            || noise.shape().dims() != [32, 7]
            || noise.to_f32_vec()?.iter().any(|x| !x.is_finite())
        {
            return Err(Error::Other(
                "Fast-WAM supplied noise must be finite CPU [32,7]".into(),
            ));
        }
    }
    Ok(())
}
impl Inputs {
    fn allocate(ctx: &Context, layout: ImageLayout) -> Result<Self> {
        let zeros =
            |dims: Vec<usize>, dtype| backend::upload(ctx, &Tensor::zeros(Shape::new(dims), dtype));
        Ok(Self {
            rgb: DeviceBuffer::alloc_zeros(224 * 448 * 3, ctx.device_id()).map_err(Error::Cuda)?,
            image_layout: match layout {
                ImageLayout::Nhwc => ops::ImageMemoryLayout::Nhwc,
                ImageLayout::Nchw => ops::ImageMemoryLayout::Nchw,
            },
            ids: zeros(vec![128], DType::I32)?,
            padding: zeros(vec![128], DType::F32)?,
            mask: zeros(vec![128], DType::BF16)?,
            state: zeros(vec![1, 8], DType::BF16)?,
            noise: zeros(vec![32, 7], DType::BF16)?,
        })
    }
    fn bind(
        &self,
        request: &VlaRequest<'_>,
        rng: &RefCell<Box<dyn NormalGenerator>>,
    ) -> Result<()> {
        let VisionObservation::RgbU8 { bytes, .. } = &request.observation.vision else {
            unreachable!()
        };
        self.rgb.copy_from_host(bytes).map_err(Error::Cuda)?;
        transfers::copy_cpu_to_cuda(
            &Tensor::from_i32(
                vec![128],
                &request
                    .observation
                    .token_ids
                    .iter()
                    .map(|&x| x as i32)
                    .collect::<Vec<_>>(),
            )?,
            &self.ids,
        )?;
        let mask = request
            .metadata
            .attention_mask
            .map(|x| x.to_vec())
            .unwrap_or_else(|| {
                request
                    .observation
                    .token_ids
                    .iter()
                    .map(|&id| u8::from(id != 0))
                    .collect()
            });
        let mask_values = mask.iter().map(|&x| x as f32).collect::<Vec<_>>();
        let padding = mask
            .iter()
            .map(|&x| if x == 0 { -3.3895314e38f32 } else { 0. })
            .collect::<Vec<_>>();
        transfers::copy_cpu_to_cuda(
            &host_tensor(vec![128], &mask_values, DType::BF16)?,
            &self.mask,
        )?;
        transfers::copy_cpu_to_cuda(&Tensor::from_f32(vec![128], &padding)?, &self.padding)?;
        transfers::copy_cpu_to_cuda(
            &host_tensor(
                vec![1, 8],
                &request.observation.state.as_ref().unwrap().to_f32_vec()?,
                DType::BF16,
            )?,
            &self.state,
        )?;
        match request.initial_latent {
            InitialLatent::Provided(x) => transfers::copy_cpu_to_cuda(
                &host_tensor(vec![32, 7], &x.to_f32_vec()?, DType::BF16)?,
                &self.noise,
            )?,
            InitialLatent::Generate { rng: key } => {
                rng.borrow_mut().generate(key)?;
            }
        }
        Ok(())
    }
}
enum Execution {
    Eager(ExecutionSession),
    Graph(PreparedPhase),
}
struct Plan {
    execution: Execution,
    inputs: Inputs,
    output: Tensor,
    rng: RefCell<Box<dyn NormalGenerator>>,
    spec: InferenceSpec,
    ctx: Rc<Context>,
    model: Rc<Model>,
    fallback: Option<String>,
}
impl Plan {
    fn build(
        runner: &ModelRunner,
        spec: &InferenceSpec,
        policy: ExecutionPolicy,
        sample: Option<&VlaRequest<'_>>,
    ) -> Result<Self> {
        check_spec(spec)?;
        if let Some(sample) = sample {
            validate(sample, spec)?;
        }
        let ctx = &runner.ctx;
        let inputs = Inputs::allocate(ctx, spec.image_layout.unwrap())?;
        let rng = RefCell::new(sampling::create_normal_generator(
            ctx,
            inputs.noise.clone(),
        )?);
        if let Some(sample) = sample {
            ctx.synchronize().map_err(Error::Cuda)?;
            inputs.bind(sample, &rng)?;
        }
        let output = ctx.allocate_output(Shape::new(vec![32, 7]), DType::F32)?;
        let operation = |autotune| {
            let action = runner.model.forward(ctx, &inputs, autotune)?;
            transfers::copy_cuda_to_cuda(ctx, &action, &output)
        };
        let make_session =
            || ExecutionSession::with_capacity(runner.model.workspace_bytes(), ctx.device_id());
        // Prepare/tune in the same stable arena that capture will reuse. An
        // unscoped warm forward would create and destroy thousands of native
        // GEMM resources, then repeat that work during actual preparation.
        let make_graph = || {
            let mut preparing = true;
            PreparedPhase::prepare_and_capture(ctx, make_session()?, || {
                let autotune = std::mem::replace(&mut preparing, false);
                operation(autotune)
            })
        };
        let mut fallback = None;
        let execution = match policy {
            ExecutionPolicy::Eager => {
                let session = make_session()?;
                ops::prepare_with_session(&session, || operation(true))?;
                Execution::Eager(session)
            }
            ExecutionPolicy::RequireGraph => Execution::Graph(make_graph()?),
            ExecutionPolicy::PreferGraph => match make_graph() {
                Ok(graph) => Execution::Graph(graph),
                Err(e) => {
                    fallback = Some(e.to_string());
                    let session = make_session()?;
                    ops::prepare_with_session(&session, || operation(true))?;
                    Execution::Eager(session)
                }
            },
        };
        Ok(Self {
            execution,
            inputs,
            output,
            rng,
            spec: *spec,
            ctx: Rc::clone(ctx),
            model: Rc::clone(&runner.model),
            fallback,
        })
    }
}
impl PreparedInference for Plan {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }
    fn status(&self) -> PreparationStatus {
        PreparationStatus::Ready {
            mode: match self.execution {
                Execution::Eager(_) => ExecutionMode::Eager,
                Execution::Graph(_) => ExecutionMode::Graph,
            },
            fallback_reason: self.fallback.clone(),
        }
    }
    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        validate(request, &self.spec)?;
        // Host uploads must not overwrite buffers still read by the previous
        // asynchronous execution on this nonblocking context stream.
        self.ctx.synchronize().map_err(Error::Cuda)?;
        self.inputs.bind(request, &self.rng)?;
        match &self.execution {
            Execution::Graph(graph) => graph.replay()?,
            Execution::Eager(session) => ops::with_session(session, || {
                let x = self.model.forward(&self.ctx, &self.inputs, false)?;
                transfers::copy_cuda_to_cuda(&self.ctx, &x, &self.output)
            })?,
        }
        Ok(Action::new(self.output.clone()))
    }
}
impl VlaRuntime for ModelRunner {
    fn model_variant(&self) -> Option<&'static str> {
        Some("bf16")
    }
    fn contract(&self) -> VlaContract {
        VlaContract {
            action_shape: [32, 7],
            patch_shape: [0, 0],
            max_token_len: 128,
            num_views: 2,
            image_size: 224,
            patch_size: 1,
            accepts_rgb_u8: true,
        }
    }
    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        let spec = request.observation.inference_spec();
        validate(request, &spec)?;
        if self
            .cached
            .borrow()
            .as_ref()
            .is_none_or(|p| p.spec() != &spec)
        {
            *self.cached.borrow_mut() =
                Some(self.prepare_for(request, ExecutionPolicy::RequireGraph)?);
        }
        self.cached.borrow().as_ref().unwrap().run(request)
    }
    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        self.prepare_with_policy(spec, ExecutionPolicy::RequireGraph)
    }
    fn prepare_with_policy(
        &self,
        spec: &InferenceSpec,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        Ok(Box::new(Plan::build(self, spec, policy, None)?))
    }
    fn prepare_for(
        &self,
        sample: &VlaRequest<'_>,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        Ok(Box::new(Plan::build(
            self,
            &sample.observation.inference_spec(),
            policy,
            Some(sample),
        )?))
    }
    fn clear_prepared(&self) -> Result<()> {
        *self.cached.borrow_mut() = None;
        Ok(())
    }
    fn execution_mode(&self) -> &'static str {
        match self.cached.borrow().as_ref().map(|p| p.status()) {
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
        transfers::to_cpu(self.infer(request)?.tensor())?.to_f32_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{path::PathBuf, time::Instant};
    fn compare(a: &[f32], b: &[f32]) -> (f64, f32) {
        assert_eq!(a.len(), b.len());
        let mut s = 0.;
        let mut n = 0.;
        let mut m = 0f32;
        for (&a, &b) in a.iter().zip(b) {
            assert!(a.is_finite());
            s += ((a - b) as f64).powi(2);
            n += (b as f64).powi(2);
            m = m.max((a - b).abs());
        }
        ((s / n.max(1e-30)).sqrt(), m)
    }
    fn observation_from(
        input: &std::collections::HashMap<String, Tensor>,
    ) -> crate::vla::Observation {
        crate::vla::Observation {
            vision: VisionObservation::RgbU8 {
                bytes: input["inputs.rgb"]
                    .to_f32_vec()
                    .unwrap()
                    .iter()
                    .map(|&x| x as u8)
                    .collect(),
                layout: ImageLayout::Nhwc,
            },
            token_ids: input["inputs.ids"]
                .to_f32_vec()
                .unwrap()
                .iter()
                .map(|&x| x as u32)
                .collect(),
            state: Some(input["inputs.state"].clone()),
            action_mask: None,
        }
    }
    #[test]
    fn fastwam_request_validation_rejects_unsupported_inputs_before_execution() {
        use crate::vla::{Observation, PlanningOptions, VlaMetadata};
        let mut ids = vec![0; 128];
        ids[..3].copy_from_slice(&[10, 20, 1]);
        let observation = Observation {
            vision: VisionObservation::RgbU8 {
                bytes: vec![0; 2 * 224 * 224 * 3],
                layout: ImageLayout::Nhwc,
            },
            token_ids: ids,
            state: Some(Tensor::from_f32(vec![8], &[0.; 8]).unwrap()),
            action_mask: None,
        };
        let noise = Tensor::from_f32(vec![32, 7], &[0.; 32 * 7]).unwrap();
        let check = |obs: &Observation, noise: &Tensor, metadata: VlaMetadata<'_>| {
            validate(
                &VlaRequest::provided_with_metadata(obs, noise, metadata),
                &obs.inference_spec(),
            )
        };
        assert!(check(&observation, &noise, VlaMetadata::default()).is_ok());
        let mut bad = observation.clone();
        bad.token_ids[0] = 256384;
        assert!(check(&bad, &noise, VlaMetadata::default()).is_err());
        bad.token_ids[0] = 0;
        assert!(check(&bad, &noise, VlaMetadata::default()).is_err());
        let mut bad = observation.clone();
        bad.token_ids[1] = 0;
        assert!(check(&bad, &noise, VlaMetadata::default()).is_err());
        let mut mask = [0; 128];
        mask[..3].fill(1);
        assert!(check(
            &observation,
            &noise,
            VlaMetadata {
                attention_mask: Some(&mask),
                ..Default::default()
            }
        )
        .is_ok());
        mask[1] = 0;
        assert!(check(
            &observation,
            &noise,
            VlaMetadata {
                attention_mask: Some(&mask),
                ..Default::default()
            }
        )
        .is_err());
        mask[1] = 2;
        assert!(check(
            &observation,
            &noise,
            VlaMetadata {
                attention_mask: Some(&mask),
                ..Default::default()
            }
        )
        .is_err());
        let mut bad = observation.clone();
        bad.state = None;
        assert!(check(&bad, &noise, VlaMetadata::default()).is_err());
        bad.state = Some(Tensor::from_f32(vec![8], &[f32::NAN; 8]).unwrap());
        assert!(check(&bad, &noise, VlaMetadata::default()).is_err());
        let bad_noise = Tensor::from_f32(vec![32, 7], &[f32::INFINITY; 32 * 7]).unwrap();
        assert!(check(&observation, &bad_noise, VlaMetadata::default()).is_err());
        assert!(check(
            &observation,
            &noise,
            VlaMetadata {
                embodiment_id: Some(0),
                ..Default::default()
            }
        )
        .is_err());
        let planning = PlanningOptions {
            num_steps: Some(9),
            ..Default::default()
        };
        assert!(check(
            &observation,
            &noise,
            VlaMetadata {
                planning: Some(&planning),
                ..Default::default()
            }
        )
        .is_err());
        let mut bad = observation.clone();
        if let VisionObservation::RgbU8 { bytes, .. } = &mut bad.vision {
            bytes.pop();
        }
        assert!(check(&bad, &noise, VlaMetadata::default()).is_err());
    }

    #[test]
    #[ignore = "requires Fast-WAM checkpoint, frozen encoders, reference captures and CUDA"]
    fn fastwam_native_reference_and_lifecycle() {
        let checkpoint = PathBuf::from(std::env::var("APXINF_FASTWAM_CHECKPOINT").unwrap());
        let text = PathBuf::from(std::env::var("APXINF_FASTWAM_TEXT").unwrap());
        let vae = PathBuf::from(std::env::var("APXINF_FASTWAM_VAE").unwrap());
        let fixture = PathBuf::from(std::env::var("APXINF_FASTWAM_CAPTURE").unwrap());
        let (input, _) =
            apxinf_loader::safetensors::load_native_path(&fixture.join("inputs.safetensors"))
                .unwrap();
        let (golden, _) =
            apxinf_loader::safetensors::load_native_path(&fixture.join("outputs.safetensors"))
                .unwrap();
        let observation = observation_from(&input);
        let noise = input["inputs.noise"].reshape(vec![32, 7]).unwrap();
        let request = VlaRequest::provided(&observation, &noise);
        let ctx = Rc::new(Context::new(0).unwrap());
        let mut gemm = ops::GemmPolicy::default();
        gemm.online_tune = false;
        gemm.cache_dir = Some(String::new());
        let mut attention = ops::AttentionPolicy::default();
        attention.online_tune = false;
        attention.cache_dir = Some(String::new());
        eprintln!("loading native Fast-WAM");
        let load_start = Instant::now();
        let model =
            super::super::load::build_model(&ctx, &checkpoint, &text, &vae, gemm, attention)
                .unwrap();
        let load_ms = load_start.elapsed().as_secs_f64() * 1000.;
        let arena_capacity = model.workspace_bytes();
        let runner = ModelRunner::new(ctx, model);
        eprintln!("prepare eager; load_ms={load_ms} arena_capacity={arena_capacity}");
        let prepare_start = Instant::now();
        let eager = runner
            .prepare_for(&request, ExecutionPolicy::Eager)
            .unwrap();
        let eager_prepare_ms = prepare_start.elapsed().as_secs_f64() * 1000.;
        let expected = golden["outputs.action"].to_f32_vec().unwrap();
        let actual = transfers::to_cpu(eager.run(&request).unwrap().tensor())
            .unwrap()
            .to_f32_vec()
            .unwrap();
        let (relative, max) = compare(&actual, &expected);
        eprintln!("native reference relative_l2={relative} max_abs={max}");
        if let Ok(path) = std::env::var("APXINF_FASTWAM_RESULT") {
            std::fs::write(
                path,
                serde_json::to_string_pretty(
                    &serde_json::json!({"actions":actual,"relative_l2":relative,"max_abs":max,"load_ms":load_ms,"eager_prepare_ms":eager_prepare_ms,"arena_capacity":arena_capacity}),
                )
                .unwrap(),
            )
            .unwrap();
        }
        assert!(
            relative < 0.02 && max < 0.04,
            "native end-to-end reference gate"
        );
        let mut eager_ms = Vec::new();
        for _ in 0..3 {
            let start = Instant::now();
            let repeat = transfers::to_cpu(eager.run(&request).unwrap().tensor())
                .unwrap()
                .to_f32_vec()
                .unwrap();
            eager_ms.push(start.elapsed().as_secs_f64() * 1000.);
            assert_eq!(repeat, actual);
        }
        drop(eager);
        eprintln!(
            "prepare whole-model graph; eager_prepare_ms={eager_prepare_ms} eager_ms={eager_ms:?}"
        );
        let prepare_start = Instant::now();
        let graph = runner
            .prepare_for(&request, ExecutionPolicy::RequireGraph)
            .unwrap();
        assert!(matches!(
            graph.status(),
            PreparationStatus::Ready {
                mode: ExecutionMode::Graph,
                ..
            }
        ));
        let graph_prepare_ms = prepare_start.elapsed().as_secs_f64() * 1000.;
        let run = |req: &VlaRequest<'_>| {
            transfers::to_cpu(graph.run(req).unwrap().tensor())
                .unwrap()
                .to_f32_vec()
                .unwrap()
        };
        let captured = run(&request);
        assert_eq!(captured, actual, "eager/graph must be identical");
        let mut precision_results = Vec::new();
        if let Ok(root) = std::env::var("APXINF_FASTWAM_SUITE") {
            let root = PathBuf::from(root);
            let mut results = Vec::new();
            let cases = std::env::var("APXINF_FASTWAM_CASES")
                .map(|count| count.parse::<usize>().unwrap())
                .unwrap_or(4);
            assert!(cases > 0);
            for case in 0..cases {
                let bf16 = root.join("suite-bfloat16").join(case.to_string());
                let f32 = root.join("suite-float32").join(case.to_string());
                let (inputs, _) =
                    apxinf_loader::safetensors::load_native_path(&bf16.join("inputs.safetensors"))
                        .unwrap();
                let (expected, _) =
                    apxinf_loader::safetensors::load_native_path(&bf16.join("outputs.safetensors"))
                        .unwrap();
                let (floor, _) =
                    apxinf_loader::safetensors::load_native_path(&f32.join("outputs.safetensors"))
                        .unwrap();
                let observation = observation_from(&inputs);
                let noise = inputs["inputs.noise"].reshape(vec![32, 7]).unwrap();
                let actual = run(&VlaRequest::provided(&observation, &noise));
                let expected = expected["outputs.action"].to_f32_vec().unwrap();
                let floor = floor["outputs.action"].to_f32_vec().unwrap();
                let native_error = compare(&actual, &floor);
                let reference_error = compare(&expected, &floor);
                let parity = compare(&actual, &expected);
                eprintln!("fixture {case}: native/F32={native_error:?} referenceBF16/F32={reference_error:?} native/BF16={parity:?}");
                results.push(serde_json::json!({"case":case,"native_f32":native_error,"reference_bf16_f32":reference_error,"native_bf16":parity,"actions":actual}));
            }
            std::fs::write(
                root.join("native-suite.json"),
                serde_json::to_string_pretty(&results).unwrap(),
            )
            .unwrap();
            precision_results = results;
        }
        assert_eq!(run(&request), captured);
        let mut changed = observation.clone();
        changed.state = Some(Tensor::from_f32(vec![8], &[0.3; 8]).unwrap());
        assert_ne!(run(&VlaRequest::provided(&changed, &noise)), captured);
        for _ in 0..3 {
            let _ = graph.run(&VlaRequest::provided(&changed, &noise)).unwrap();
            let _ = graph.run(&request).unwrap();
        }
        assert_eq!(
            run(&request),
            captured,
            "asynchronous rebinding must be ordered"
        );
        let mut changed = observation.clone();
        if let VisionObservation::RgbU8 { bytes, .. } = &mut changed.vision {
            bytes.fill(0);
        }
        assert_ne!(run(&VlaRequest::provided(&changed, &noise)), captured);
        let mut changed = observation.clone();
        changed.token_ids[0] = 42;
        assert_ne!(run(&VlaRequest::provided(&changed, &noise)), captured);
        let zero_noise = Tensor::zeros(Shape::new(vec![32, 7]), DType::F32);
        assert_ne!(
            run(&VlaRequest::provided(&observation, &zero_noise)),
            captured
        );
        assert_eq!(run(&request), captured);
        let generated = VlaRequest::generated(&observation, apxinf_core::RngKey::new(1, 2, 3));
        let first = run(&generated);
        assert_eq!(first, run(&generated));
        assert_ne!(
            first,
            run(&VlaRequest::generated(
                &observation,
                apxinf_core::RngKey::new(2, 2, 3)
            ))
        );
        let mut bad = observation.clone();
        bad.token_ids.pop();
        assert!(graph.run(&VlaRequest::provided(&bad, &noise)).is_err());
        let mut ms = Vec::new();
        for _ in 0..10 {
            let start = Instant::now();
            run(&request);
            ms.push(start.elapsed().as_secs_f64() * 1000.);
        }
        eprintln!("whole_model_host_latency_ms={ms:?}");
        let mut nchw = observation.clone();
        if let VisionObservation::RgbU8 { bytes, layout } = &mut nchw.vision {
            let source = bytes.clone();
            for view in 0..2 {
                for channel in 0..3 {
                    for y in 0..224 {
                        for x in 0..224 {
                            bytes[((view * 3 + channel) * 224 + y) * 224 + x] =
                                source[((view * 224 + y) * 224 + x) * 3 + channel];
                        }
                    }
                }
            }
            *layout = ImageLayout::Nchw;
        }
        let nchw_request = VlaRequest::provided(&nchw, &noise);
        assert!(
            graph.run(&nchw_request).is_err(),
            "layout changes invalidate a prepared profile"
        );
        let retained = graph.run(&request).unwrap();
        drop(graph);
        let nchw_plan = runner
            .prepare_for(&nchw_request, ExecutionPolicy::RequireGraph)
            .unwrap();
        assert_eq!(
            transfers::to_cpu(nchw_plan.run(&nchw_request).unwrap().tensor())
                .unwrap()
                .to_f32_vec()
                .unwrap(),
            captured
        );
        drop(nchw_plan);
        drop(runner);
        assert_eq!(
            transfers::to_cpu(retained.tensor())
                .unwrap()
                .to_f32_vec()
                .unwrap(),
            captured
        );
        if let Ok(path) = std::env::var("APXINF_FASTWAM_RESULT") {
            let mut report: serde_json::Value =
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            report["eager_ms"] = serde_json::json!(eager_ms);
            report["graph_ms"] = serde_json::json!(ms);
            report["graph_prepare_ms"] = serde_json::json!(graph_prepare_ms);
            report["lifecycle_passed"] = serde_json::json!(true);
            std::fs::write(path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
        }
        for result in &precision_results {
            assert!(
                    result["native_f32"][0].as_f64().unwrap()
                        <= result["reference_bf16_f32"][0].as_f64().unwrap() + 1e-6
                        && result["native_f32"][1].as_f64().unwrap()
                            <= result["reference_bf16_f32"][1].as_f64().unwrap() + 1e-6,
                    "case {} must preserve the canonical FP32 quality floor: native {:?}, reference {:?}",
                    result["case"], result["native_f32"], result["reference_bf16_f32"]
                );
        }
    }
}
