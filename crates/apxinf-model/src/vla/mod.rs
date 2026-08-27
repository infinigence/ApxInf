//! Model-neutral interfaces for vision-language-action runtimes.

use apxinf_core::{Error, Result, RngKey, Tensor};

/// Model-independent dimensions exposed by every VLA runtime.
///
/// These values describe the bare observation-to-action contract used by
/// bindings and policy layers. Architecture-specific configuration remains in
/// the owning model directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VlaDimensions {
    pub action_horizon: usize,
    pub action_dim: usize,
    pub num_views: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub max_token_len: usize,
}

impl VlaDimensions {
    pub fn validate(&self) -> Result<()> {
        if self.action_horizon == 0
            || self.action_dim == 0
            || self.num_views == 0
            || self.image_size == 0
            || self.patch_size == 0
            || self.max_token_len == 0
        {
            return Err(Error::Other(format!(
                "VLA dimensions must be non-zero, got {self:?}"
            )));
        }
        if self.image_size % self.patch_size != 0 {
            return Err(Error::Other(format!(
                "VLA image size {} is not divisible by patch size {}",
                self.image_size, self.patch_size
            )));
        }
        Ok(())
    }

    pub fn patches_per_view(&self) -> usize {
        let side = self.image_size / self.patch_size;
        side * side
    }
}

/// Memory layout for an RGB `u8` observation batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageLayout {
    Nhwc,
    Nchw,
}

/// Vision input accepted by a VLA runtime.
#[derive(Clone, Debug)]
pub enum VisionObservation {
    /// Preprocessed patch rows. The model defines the expected dtype and shape.
    Patches(Tensor),
    /// Resized RGB images. The byte buffer contains the complete view batch.
    RgbU8 { bytes: Vec<u8>, layout: ImageLayout },
}

/// Complete input for one VLA inference.
#[derive(Clone, Debug)]
pub struct Observation {
    pub vision: VisionObservation,
    pub token_ids: Vec<u32>,
}

impl Observation {
    pub fn validate(&self) -> Result<()> {
        if self.token_ids.is_empty() {
            return Err(Error::Other("VLA observation has no token IDs".into()));
        }
        Ok(())
    }

    pub fn inference_spec(&self) -> InferenceSpec {
        InferenceSpec {
            token_count: self.token_ids.len(),
            image_layout: match self.vision {
                VisionObservation::Patches(_) => None,
                VisionObservation::RgbU8 { layout, .. } => Some(layout),
            },
        }
    }
}

/// Initial continuous latent used by a flow/diffusion VLA.
///
/// PI0.5 is the only current VLA implementation. Production callers may have
/// ApxInf generate standard-normal noise directly in its captured device
/// buffer; correctness fixtures can continue to inject an exact latent.
#[derive(Clone, Copy, Debug)]
pub enum InitialLatent<'a> {
    Generate { rng: RngKey },
    Provided(&'a Tensor),
}

/// Complete VLA request: an environment observation plus the model-generation
/// input that is deliberately not part of the observation itself.
#[derive(Clone, Copy, Debug)]
pub struct VlaRequest<'a> {
    pub observation: &'a Observation,
    pub initial_latent: InitialLatent<'a>,
}

impl<'a> VlaRequest<'a> {
    pub const fn generated(observation: &'a Observation, rng: RngKey) -> Self {
        Self {
            observation,
            initial_latent: InitialLatent::Generate { rng },
        }
    }

    pub const fn provided(observation: &'a Observation, latent: &'a Tensor) -> Self {
        Self {
            observation,
            initial_latent: InitialLatent::Provided(latent),
        }
    }
}

/// Fixed-shape contract established during preparation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InferenceSpec {
    pub token_count: usize,
    /// `None` means preprocessed patches; `Some` means raw RGB input.
    pub image_layout: Option<ImageLayout>,
}

impl InferenceSpec {
    pub fn validate(&self) -> Result<()> {
        if self.token_count == 0 {
            return Err(Error::Other(
                "VLA inference spec requires at least one token".into(),
            ));
        }
        Ok(())
    }

    pub fn matches(&self, observation: &Observation) -> bool {
        *self == observation.inference_spec()
    }
}

/// Model action output. The tensor stays on the runtime device unless the
/// caller explicitly asks its backend-facing integration to transfer it, or
/// uses [`VlaRuntime::infer_host_f32`] to get host values directly.
#[derive(Clone, Debug)]
pub struct Action {
    tensor: Tensor,
}

impl Action {
    pub fn new(tensor: Tensor) -> Self {
        Self { tensor }
    }

    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub fn into_tensor(self) -> Tensor {
        self.tensor
    }
}

/// A prepared, fixed-shape inference plan. Implementations own every resource
/// referenced by eager execution or a captured graph.
pub trait PreparedInference {
    fn spec(&self) -> &InferenceSpec;
    fn run(&self, request: &VlaRequest<'_>) -> Result<Action>;
}

/// Unified VLA runtime interface.
///
/// The boxed return keeps this trait object-safe so `LoadedModel::Vla` can
/// directly hold heterogeneous model runtimes.
pub trait VlaRuntime {
    /// Return the bare-model shape contract independently of model-family
    /// configuration types.
    fn dimensions(&self) -> VlaDimensions;

    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action>;
    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>>;

    /// Run inference and copy the resulting action to host as `f32`.
    ///
    /// [`infer`](Self::infer) returns an [`Action`] whose tensor lives on the
    /// runtime device. Consumers that need host values (servers writing actions
    /// back, benches checking outputs) would otherwise have to hold a backend
    /// handle and transfer it themselves — reaching around the abstraction.
    /// This convenience performs the device→host copy inside the runtime, which
    /// already owns the backend.
    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::DType;

    fn observation() -> Observation {
        Observation {
            vision: VisionObservation::RgbU8 {
                bytes: vec![0; 2 * 4 * 4 * 3],
                layout: ImageLayout::Nhwc,
            },
            token_ids: vec![1, 2, 3],
        }
    }

    #[test]
    fn observation_spec_contains_only_fixed_shape_routing_fields() {
        let observation = observation();
        assert_eq!(
            observation.inference_spec(),
            InferenceSpec {
                token_count: 3,
                image_layout: Some(ImageLayout::Nhwc),
            }
        );
        assert!(observation.validate().is_ok());

        let empty = Observation {
            vision: VisionObservation::Patches(Tensor::zeros((1, 2), DType::F32)),
            token_ids: Vec::new(),
        };
        assert!(empty.validate().is_err());
    }

    #[test]
    fn vla_request_keeps_provided_and_generated_latents_distinct() {
        let observation = observation();
        let latent = Tensor::zeros((1, 50, 7), DType::F32);
        let provided = VlaRequest::provided(&observation, &latent);
        match provided.initial_latent {
            InitialLatent::Provided(actual) => assert!(std::ptr::eq(actual, &latent)),
            InitialLatent::Generate { .. } => panic!("provided latent was changed to generated"),
        }

        let rng = RngKey::new(17, 23, 42);
        let generated = VlaRequest::generated(&observation, rng);
        match generated.initial_latent {
            InitialLatent::Generate { rng: actual } => assert_eq!(actual, rng),
            InitialLatent::Provided(_) => panic!("generated latent requires a caller tensor"),
        }
    }

    #[test]
    fn vla_dimensions_validate_model_independent_shape_contract() {
        let dimensions = VlaDimensions {
            action_horizon: 10,
            action_dim: 32,
            num_views: 2,
            image_size: 224,
            patch_size: 14,
            max_token_len: 200,
        };
        dimensions.validate().unwrap();
        assert_eq!(dimensions.patches_per_view(), 256);

        let invalid = VlaDimensions {
            patch_size: 15,
            ..dimensions
        };
        assert!(invalid.validate().is_err());
    }
}
