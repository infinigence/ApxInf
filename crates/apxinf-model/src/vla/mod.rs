//! Model-neutral interfaces for vision-language-action runtimes.

use std::collections::BTreeMap;

use apxinf_core::{Error, Result, RngKey, Tensor};

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
    /// Optional normalized proprioceptive state for models that project it
    /// directly instead of encoding it in the prompt.
    pub state: Option<Tensor>,
    /// Optional per-dimension action mask. Missing means every action
    /// dimension is active.
    pub action_mask: Option<Tensor>,
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
/// Production callers may have ApxInf generate standard-normal noise directly
/// in a captured device buffer; correctness fixtures can inject an exact
/// latent for reproducible parity checks.
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

/// Fixed public shape contract of a loaded VLA runtime.
///
/// Frontends use this to validate host inputs without parsing a concrete
/// model family's checkpoint config or repeating model-name switches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VlaContract {
    pub action_shape: [usize; 2],
    pub patch_shape: [usize; 2],
    pub max_token_len: usize,
    pub num_views: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub accepts_rgb_u8: bool,
}

/// Unified VLA runtime interface.
///
/// The boxed return keeps this trait object-safe so `LoadedModel::Vla` can
/// directly hold heterogeneous model runtimes.
pub trait VlaRuntime {
    /// Fixed input/output capabilities of this loaded checkpoint.
    fn contract(&self) -> VlaContract;

    /// Native action tensor shape produced by this loaded checkpoint.
    fn action_shape(&self) -> [usize; 2] {
        self.contract().action_shape
    }

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

    /// Discrete action-token output shape for autoregressive token VLAs.
    ///
    /// A runtime whose deployable output is a token sequence (π0-FAST) returns
    /// `Some([1, max_action_tokens])`; continuous-action runtimes keep `None`
    /// and use [`contract`](Self::contract) alone.
    fn action_token_shape(&self) -> Option<[usize; 2]> {
        None
    }

    /// Run inference and return the raw action token ids as `[1, steps]` `f32`.
    ///
    /// Integral values are exact in `f32` well beyond any model vocabulary.
    /// Token detokenization (FAST BPE + DCT) is action postprocessing and stays
    /// in the Python policy layer.
    ///
    /// `stop_token` ends an autoregressive decode early: the runtime emits the
    /// id, stops, and returns the shorter stream (π0-FAST's FAST stream ends at
    /// the `|` terminator after ~10% of `max_action_tokens`, and its detokenizer
    /// truncates there anyway). It rides this token-only entry point rather than
    /// [`Observation`] so a continuous-action family never carries — or has to
    /// spell out — a control knob that only token decoding reads.
    fn infer_action_tokens(
        &self,
        _request: &VlaRequest<'_>,
        _stop_token: Option<u32>,
    ) -> Result<Tensor> {
        Err(Error::Other(
            "this VLA runtime does not produce discrete action tokens".into(),
        ))
    }

    /// Collect named BF16 activation maxima for an FP8 calibration profile.
    fn calibration_amax(&self, _request: &VlaRequest<'_>) -> Result<BTreeMap<String, f32>> {
        Err(Error::Other(
            "activation calibration is not supported by this VLA runtime".into(),
        ))
    }

    /// Stable logical sites required by this runtime's calibration profile.
    fn calibration_plan(&self) -> Result<Vec<String>> {
        Err(Error::Other(
            "activation calibration is not supported by this VLA runtime".into(),
        ))
    }
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
            state: None,
            action_mask: None,
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
            state: None,
            action_mask: None,
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
}
