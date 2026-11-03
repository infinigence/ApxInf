//! Execution lifecycle for Ascend PI0.5 (BF16 eager mode).
mod runner;

pub use runner::{Pi05ModelRunner, Pi05PreparedInference};

// Graph capture stubs — not yet implemented for Ascend
pub fn capture_patches(
    _model: &super::model::ModelVariant,
    _spec: &crate::vla::InferenceSpec,
    _patches: &apxinf_core::Tensor,
    _token_ids: &super::backend::DeviceBuffer,
    _noise: &apxinf_core::Tensor,
) -> apxinf_core::Result<CapturedGraph> {
    Err(apxinf_core::Error::Other(
        "Graph capture not yet implemented for Ascend".into(),
    ))
}

pub fn capture_rgb(
    _model: &super::model::ModelVariant,
    _spec: &crate::vla::InferenceSpec,
    _images: &super::backend::DeviceBuffer,
    _patches: &apxinf_core::Tensor,
    _token_ids: &super::backend::DeviceBuffer,
    _noise: &apxinf_core::Tensor,
) -> apxinf_core::Result<CapturedGraph> {
    Err(apxinf_core::Error::Other(
        "Graph capture not yet implemented for Ascend".into(),
    ))
}

pub struct CapturedGraph;
