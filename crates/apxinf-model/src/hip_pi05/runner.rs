//! π0.5 on HIP as a [`VlaRuntime`]: a prepared plan owns fixed device inputs
//! and one activation arena, and each run refills the inputs and reuses both.
//!
//! Execution is eager for now. Inputs live at fixed addresses, every
//! activation comes from the arena in a fixed order, and a run never
//! synchronizes before its output is read — the preconditions for capturing a
//! run as one HIP graph, which is the next milestone. Until then a request for
//! a graph is refused rather than silently served eagerly.

use std::cell::RefCell;
use std::rc::Rc;

use apxinf_core::{DType, Device, Error, NormalGenerator, Result, Tensor};
use apxinf_hip::native::{self, HipArena, HipDeviceBytes};
use half::bf16;

use super::config::Config;
use super::model::HipPi05Model;
use crate::vla::{
    Action, ExecutionMode, ExecutionPolicy, ImageLayout, InferenceSpec, InitialLatent,
    PreparationStatus, PreparedInference, VisionObservation, VlaContract, VlaRequest, VlaRuntime,
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
        if policy == ExecutionPolicy::RequireGraph {
            return Err(Error::UnsupportedOp("π0.5 HIP graph capture"));
        }
        HipPi05Prepared::new(Rc::clone(&self.model), *spec)
    }
}

/// One fixed-shape plan: device inputs at fixed addresses, the arena, and the
/// generator that fills the noise buffer in place.
pub struct HipPi05Prepared {
    spec: InferenceSpec,
    model: Rc<HipPi05Model>,
    arena: HipArena,
    patches: Tensor,
    images: Option<HipDeviceBytes>,
    token_ids: HipDeviceBytes,
    noise: Tensor,
    normal_generator: RefCell<Box<dyn NormalGenerator>>,
}

impl HipPi05Prepared {
    fn new(model: Rc<HipPi05Model>, spec: InferenceSpec) -> Result<Self> {
        spec.validate()?;
        let config = model.config();
        if spec.token_count > config.max_token_len {
            return Err(Error::Other(format!(
                "π0.5 token count {} exceeds maximum {}",
                spec.token_count, config.max_token_len
            )));
        }
        let ctx = model.context();
        let patches = ctx.to_device(&Tensor::zeros(patch_shape(config), DType::BF16))?;
        let noise = ctx.to_device(&Tensor::zeros(noise_shape(config), DType::BF16))?;
        let images = spec
            .image_layout
            .map(|_| HipDeviceBytes::new(ctx, image_bytes(config)))
            .transpose()?;
        let token_ids = HipDeviceBytes::new(ctx, spec.token_count * 4)?;
        let normal_generator = native::normal_generator(ctx, noise.clone())?;
        let arena = HipArena::new(ctx, HipPi05Model::arena_bytes(config, spec.token_count))?;
        Ok(Self {
            spec,
            model,
            arena,
            patches,
            images,
            token_ids,
            noise,
            normal_generator: RefCell::new(normal_generator),
        })
    }

    fn bind_inputs(&self, request: &VlaRequest<'_>) -> Result<()> {
        let observation = request.observation;
        let config = self.model.config();
        let ctx = self.model.context();
        match &observation.vision {
            VisionObservation::Patches(patches) => {
                let patches = host_bf16(patches, patch_shape(config), "patches")?;
                native::upload(ctx, &patches, &self.patches)?;
            }
            VisionObservation::RgbU8 { bytes, .. } => {
                let images = self
                    .images
                    .as_ref()
                    .expect("an RGB plan holds an image buffer");
                if bytes.len() != images.len() {
                    return Err(Error::Other(format!(
                        "π0.5 expected {} raw image bytes, got {}",
                        images.len(),
                        bytes.len()
                    )));
                }
                images.upload(bytes)?;
            }
        }
        self.token_ids.upload_u32(&observation.token_ids)?;
        match request.initial_latent {
            InitialLatent::Provided(latent) => {
                let latent = host_bf16(latent, noise_shape(config), "initial latent")?;
                native::upload(ctx, &latent, &self.noise)?;
            }
            InitialLatent::Generate { rng } => {
                self.normal_generator.borrow_mut().generate(rng)?;
            }
        }
        Ok(())
    }
}

impl PreparedInference for HipPi05Prepared {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }

    fn status(&self) -> PreparationStatus {
        PreparationStatus::Ready {
            mode: ExecutionMode::Eager,
            fallback_reason: None,
        }
    }

    /// The returned action lives in the arena: the next run overwrites it.
    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        let observation = request.observation;
        observation.validate()?;
        if !self.spec.matches(observation) {
            return Err(Error::Other(format!(
                "prepared π0.5 spec {:?} does not match observation {:?}",
                self.spec,
                observation.inference_spec()
            )));
        }
        self.bind_inputs(request)?;
        let config = self.model.config();
        let ctx = self.model.context();
        self.arena.reset();
        if let (Some(images), Some(layout)) = (&self.images, self.spec.image_layout) {
            native::rgb_to_patches(
                ctx,
                images,
                kernel_layout(layout),
                config.num_views,
                config.image_size,
                config.patch_size,
                &self.patches,
            )?;
        }
        let actions = self.model.infer(
            &self.arena,
            &self.patches,
            &self.token_ids,
            self.spec.token_count,
            &self.noise,
        )?;
        Ok(Action::new(actions))
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
        if self.prepared.borrow().is_some() {
            "eager"
        } else {
            "unprepared"
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

fn kernel_layout(layout: ImageLayout) -> native::ImageLayout {
    match layout {
        ImageLayout::Nhwc => native::ImageLayout::Nhwc,
        ImageLayout::Nchw => native::ImageLayout::Nchw,
    }
}

fn patch_shape(config: &Config) -> Vec<usize> {
    vec![
        config.num_views * config.patches_per_view(),
        3 * config.patch_size * config.patch_size,
    ]
}

fn noise_shape(config: &Config) -> Vec<usize> {
    vec![config.action_horizon, config.action_dim]
}

fn image_bytes(config: &Config) -> usize {
    config.num_views * 3 * config.image_size * config.image_size
}

/// A host tensor of `shape` as BF16, converting from any float dtype.
fn host_bf16(tensor: &Tensor, shape: Vec<usize>, label: &str) -> Result<Tensor> {
    if tensor.device() != Device::Cpu {
        return Err(Error::Other(format!("π0.5 {label} must be a CPU tensor")));
    }
    if tensor.shape().dims() != shape {
        return Err(Error::Other(format!(
            "π0.5 {label} shape {:?} does not match {shape:?}",
            tensor.shape().dims()
        )));
    }
    if tensor.dtype() == DType::BF16 {
        return Ok(tensor.clone());
    }
    let values: Vec<bf16> = tensor
        .to_f32_vec()?
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    Tensor::from_bf16(shape, &values)
}
