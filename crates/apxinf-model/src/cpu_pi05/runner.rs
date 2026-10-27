//! Synchronous CPU execution. Explicit plans retain the model and never claim graph readiness.
use super::config::{sinusoidal_time_embedding, Config};
use crate::cpu_pi05::model::{CpuPi05Model, StepModulation};
use crate::vla::{
    Action, ExecutionMode, ExecutionPolicy, ImageLayout, InferenceSpec, InitialLatent,
    PreparationStatus, PreparedInference, VisionObservation, VlaContract, VlaRequest, VlaRuntime,
};
use apxinf_core::{standard_normal_f32, DType, Device, Error, Result, Tensor};
use std::rc::Rc;

#[derive(Clone)]
pub(in crate::cpu_pi05) struct CpuPi05Runner {
    config: Config,
    model: Rc<CpuPi05Model>,
    modulation: Rc<Vec<StepModulation>>,
}
impl CpuPi05Runner {
    pub(in crate::cpu_pi05) fn new(config: Config, model: Rc<CpuPi05Model>) -> Result<Self> {
        let embeddings = (0..config.num_flow_steps)
            .map(|i| {
                let t = config.flow_start_time * (1. - i as f32 / config.num_flow_steps as f32);
                Tensor::from_f32(
                    vec![1, config.action_expert.width],
                    &sinusoidal_time_embedding(
                        t,
                        config.action_expert.width,
                        config.time_min_period,
                        config.time_max_period,
                    ),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let modulation = Rc::new(model.prepare_all_modulation(&embeddings)?);
        Ok(Self {
            config,
            model,
            modulation,
        })
    }
    fn validate_spec(&self, spec: &InferenceSpec) -> Result<()> {
        spec.validate()?;
        if spec.token_count > self.config.max_token_len {
            return Err(Error::Other(
                "PI0.5 CPU token count exceeds max_token_len".into(),
            ));
        }
        Ok(())
    }
    fn validate(&self, request: &VlaRequest<'_>) -> Result<()> {
        self.validate_spec(&request.observation.inference_spec())?;
        if request
            .observation
            .token_ids
            .iter()
            .any(|&id| id as usize >= self.config.vocab_size)
        {
            return Err(Error::Other("PI0.5 CPU token ID outside vocabulary".into()));
        }
        if request.observation.state.is_some()
            || request.observation.action_mask.is_some()
            || request.metadata.attention_mask.is_some()
            || request.metadata.image_grid_thw.is_some()
            || request.metadata.embodiment_id.is_some()
            || request.metadata.planning.is_some()
        {
            return Err(Error::Other("PI0.5 CPU accepts encoded state in tokens, all-valid views/tokens and no extra metadata or action mask".into()));
        }
        match &request.observation.vision {
            VisionObservation::Patches(p) => {
                host_f32(p, self.contract().patch_shape, "patches")?;
            }
            VisionObservation::RgbU8 { bytes, .. } => {
                if bytes.len()
                    != self.config.num_views * self.config.image_size * self.config.image_size * 3
                {
                    return Err(Error::Other("PI0.5 CPU RGB byte count mismatch".into()));
                }
            }
        }
        if let InitialLatent::Provided(t) = request.initial_latent {
            host_f32(t, self.action_shape(), "initial latent")?;
        }
        Ok(())
    }
    fn patches(&self, vision: &VisionObservation) -> Result<Tensor> {
        match vision {
            VisionObservation::Patches(p) => host_f32(p, self.contract().patch_shape, "patches"),
            VisionObservation::RgbU8 { bytes, layout } => {
                let c = &self.config;
                let size = c.image_size;
                let patch = c.patch_size;
                let mut out = Vec::with_capacity(bytes.len());
                // Checkpoint convolution flattening is channel, patch-y, patch-x.
                for view in 0..c.num_views {
                    for py in 0..size / patch {
                        for px in 0..size / patch {
                            for ch in 0..3 {
                                for dy in 0..patch {
                                    for dx in 0..patch {
                                        let y = py * patch + dy;
                                        let x = px * patch + dx;
                                        let i = match layout {
                                            ImageLayout::Nhwc => {
                                                ((view * size + y) * size + x) * 3 + ch
                                            }
                                            ImageLayout::Nchw => {
                                                ((view * 3 + ch) * size + y) * size + x
                                            }
                                        };
                                        out.push(bytes[i] as f32 * (2. / 255.) - 1.);
                                    }
                                }
                            }
                        }
                    }
                }
                Tensor::from_f32(self.contract().patch_shape.to_vec(), &out)
            }
        }
    }
}
fn host_f32(t: &Tensor, shape: [usize; 2], name: &str) -> Result<Tensor> {
    if t.device() != Device::Cpu
        || t.shape().dims() != shape
        || !matches!(t.dtype(), DType::F32 | DType::F16 | DType::BF16)
    {
        return Err(Error::Other(format!(
            "PI0.5 CPU {name} requires CPU floating point {shape:?}"
        )));
    }
    let values = t.to_f32_vec()?;
    if !values.iter().all(|v| v.is_finite()) {
        return Err(Error::Other(format!(
            "PI0.5 CPU {name} contains non-finite values"
        )));
    }
    if t.dtype() == DType::F32 {
        Ok(t.clone())
    } else {
        Tensor::from_f32(shape.to_vec(), &values)
    }
}
struct CpuPi05Prepared {
    runner: CpuPi05Runner,
    spec: InferenceSpec,
    fallback: bool,
}
impl PreparedInference for CpuPi05Prepared {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }
    fn status(&self) -> PreparationStatus {
        PreparationStatus::Ready {
            mode: ExecutionMode::Eager,
            fallback_reason: self
                .fallback
                .then(|| "CPU reference does not support graph capture".into()),
        }
    }
    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        if !self.spec.matches(request.observation) {
            return Err(Error::Other(
                "PI0.5 CPU prepared input specification mismatch".into(),
            ));
        }
        self.runner.infer(request)
    }
}
impl VlaRuntime for CpuPi05Runner {
    fn model_variant(&self) -> Option<&'static str> {
        Some("fp32_reference")
    }
    fn contract(&self) -> VlaContract {
        let c = &self.config;
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
        self.validate(request)?;
        let patches = self.patches(&request.observation.vision)?;
        let noise = match request.initial_latent {
            InitialLatent::Provided(t) => host_f32(t, self.action_shape(), "initial latent")?,
            InitialLatent::Generate { rng } => Tensor::from_f32(
                self.action_shape().to_vec(),
                &standard_normal_f32(self.action_shape().iter().product(), rng),
            )?,
        };
        let out = self.model.infer_with_modulation(
            &patches,
            &request.observation.token_ids,
            &noise,
            &self.modulation,
        )?;
        if !out.as_f32()?.iter().all(|v| v.is_finite()) {
            return Err(Error::Other("PI0.5 CPU produced non-finite action".into()));
        }
        Ok(Action::new(out))
    }
    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        self.prepare_with_policy(spec, ExecutionPolicy::Eager)
    }
    fn prepare_with_policy(
        &self,
        spec: &InferenceSpec,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        self.validate_spec(spec)?;
        if policy == ExecutionPolicy::RequireGraph {
            return Err(Error::Other(
                "PI0.5 CPU does not support RequireGraph".into(),
            ));
        }
        // Exercise the actual traversal before reporting Ready, without retaining request state.
        let vision = match spec.image_layout {
            None => VisionObservation::Patches(Tensor::zeros(
                self.contract().patch_shape.to_vec(),
                DType::F32,
            )),
            Some(layout) => VisionObservation::RgbU8 {
                bytes: vec![
                    0;
                    self.config.num_views
                        * self.config.image_size
                        * self.config.image_size
                        * 3
                ],
                layout,
            },
        };
        let observation = crate::vla::Observation {
            vision,
            token_ids: vec![0; spec.token_count],
            state: None,
            action_mask: None,
        };
        let noise = Tensor::zeros(self.action_shape().to_vec(), DType::F32);
        self.infer(&VlaRequest::provided(&observation, &noise))?;
        Ok(Box::new(CpuPi05Prepared {
            runner: self.clone(),
            spec: *spec,
            fallback: policy == ExecutionPolicy::PreferGraph,
        }))
    }
    fn prepare_for(
        &self,
        sample: &VlaRequest<'_>,
        policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        self.validate(sample)?;
        if policy == ExecutionPolicy::RequireGraph {
            return Err(Error::Other(
                "PI0.5 CPU does not support RequireGraph".into(),
            ));
        }
        self.infer(sample)?;
        Ok(Box::new(CpuPi05Prepared {
            runner: self.clone(),
            spec: sample.observation.inference_spec(),
            fallback: policy == ExecutionPolicy::PreferGraph,
        }))
    }
    fn clear_prepared(&self) -> Result<()> {
        Ok(())
    }
    fn execution_mode(&self) -> &'static str {
        "eager"
    }
    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        self.infer(request)?.tensor().to_f32_vec()
    }
}
