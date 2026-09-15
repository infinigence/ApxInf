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

/// Optional typed metadata emitted by preprocessors for VLA families whose
/// inputs include more than image patches and token IDs.
///
/// Keeping these fields on the request preserves the stable observation shape
/// used by existing PI0.5 and WallOSS callers. A runtime that requires one of
/// these fields validates it explicitly; other runtimes ignore the empty
/// default.
#[derive(Clone, Copy, Debug, Default)]
pub struct VlaMetadata<'a> {
    pub attention_mask: Option<&'a [u8]>,
    pub image_grid_thw: Option<&'a [[u32; 3]]>,
    pub embodiment_id: Option<usize>,
}

/// Complete VLA request: an environment observation plus the model-generation
/// input that is deliberately not part of the observation itself.
#[derive(Clone, Copy, Debug)]
pub struct VlaRequest<'a> {
    pub observation: &'a Observation,
    pub initial_latent: InitialLatent<'a>,
    pub metadata: VlaMetadata<'a>,
}

impl<'a> VlaRequest<'a> {
    pub const fn generated(observation: &'a Observation, rng: RngKey) -> Self {
        Self {
            observation,
            initial_latent: InitialLatent::Generate { rng },
            metadata: VlaMetadata {
                attention_mask: None,
                image_grid_thw: None,
                embodiment_id: None,
            },
        }
    }

    pub const fn provided(observation: &'a Observation, latent: &'a Tensor) -> Self {
        Self {
            observation,
            initial_latent: InitialLatent::Provided(latent),
            metadata: VlaMetadata {
                attention_mask: None,
                image_grid_thw: None,
                embodiment_id: None,
            },
        }
    }

    pub const fn generated_with_metadata(
        observation: &'a Observation,
        rng: RngKey,
        metadata: VlaMetadata<'a>,
    ) -> Self {
        Self {
            observation,
            initial_latent: InitialLatent::Generate { rng },
            metadata,
        }
    }

    pub const fn provided_with_metadata(
        observation: &'a Observation,
        latent: &'a Tensor,
        metadata: VlaMetadata<'a>,
    ) -> Self {
        Self {
            observation,
            initial_latent: InitialLatent::Provided(latent),
            metadata,
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

/// Requested execution policy. A runtime must explicitly support this contract;
/// legacy `prepare` implementations are not assumed to honor it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionPolicy {
    Eager,
    PreferGraph,
    RequireGraph,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionMode {
    Eager,
    Graph,
}

/// Readiness for the plan's fixed specification, not whole-model readiness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreparationStatus {
    /// The implementation has not adopted the explicit readiness contract.
    RuntimeManaged,
    Ready {
        mode: ExecutionMode,
        fallback_reason: Option<String>,
    },
    /// Execution choices changed; prepare a new plan before running again.
    Invalidated,
}

/// A prepared, fixed-shape inference plan. Implementations own every resource
/// referenced by eager execution or a captured graph.
pub trait PreparedInference {
    fn spec(&self) -> &InferenceSpec;

    fn status(&self) -> PreparationStatus {
        PreparationStatus::RuntimeManaged
    }

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

    /// Prepare without tuning on placeholder data. Compatible runs do not
    /// capture or autotune; unsupported implementations fail explicitly.
    fn prepare_with_policy(
        &self,
        _spec: &InferenceSpec,
        _policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        Err(Error::Other(
            "explicit VLA preparation policy is not supported".into(),
        ))
    }

    /// Optional real-input tuning followed by fixed-spec preparation. The
    /// sample is not retained as semantic request state.
    fn prepare_for(
        &self,
        _sample: &VlaRequest<'_>,
        _policy: ExecutionPolicy,
    ) -> Result<Box<dyn PreparedInference>> {
        Err(Error::Other(
            "real-input VLA preparation is not supported".into(),
        ))
    }

    /// Evict implicit cached plans. Explicitly owned plans remain alive.
    /// This is not a request/RNG reset.
    fn clear_prepared(&self) -> Result<()> {
        Err(Error::Other("VLA plan eviction is not supported".into()))
    }

    /// Current execution path for diagnostics and benchmarks. Implementations
    /// should report an eager fallback explicitly after a graph attempt.
    fn execution_mode(&self) -> &'static str {
        "runtime-managed"
    }

    /// Run inference and copy the resulting action to host as `f32`.
    ///
    /// [`infer`](Self::infer) returns an [`Action`] whose tensor lives on the
    /// runtime device. Consumers that need host values (servers writing actions
    /// back, benches checking outputs) would otherwise have to hold a backend
    /// handle and transfer it themselves — reaching around the abstraction.
    /// This convenience performs the device→host copy inside the runtime, which
    /// already owns the backend.
    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>>;

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
