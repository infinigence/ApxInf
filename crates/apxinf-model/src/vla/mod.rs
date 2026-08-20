//! Model-neutral interfaces for vision-language-action runtimes.

use apxinf_core::{Error, Result, Tensor};

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
    pub noise: Tensor,
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
    fn run(&self, observation: &Observation) -> Result<Action>;
}

/// Unified VLA runtime interface.
///
/// The boxed return keeps this trait object-safe so `LoadedModel::Vla` can
/// directly hold heterogeneous model runtimes.
pub trait VlaRuntime {
    fn infer(&self, observation: &Observation) -> Result<Action>;
    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>>;

    /// Run inference and copy the resulting action to host as `f32`.
    ///
    /// [`infer`](Self::infer) returns an [`Action`] whose tensor lives on the
    /// runtime device. Consumers that need host values (servers writing actions
    /// back, benches checking outputs) would otherwise have to hold a backend
    /// handle and transfer it themselves — reaching around the abstraction.
    /// This convenience performs the device→host copy inside the runtime, which
    /// already owns the backend.
    fn infer_host_f32(&self, observation: &Observation) -> Result<Vec<f32>>;
}
