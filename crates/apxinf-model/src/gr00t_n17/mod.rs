//! NVIDIA GR00T N1.7 vision-language-action model.
//!
//! The family owns its model orchestration, weight names, fixed LIBERO profile,
//! and CUDA Graph boundary. It shares only model-neutral ApxInf backend and
//! kernel interfaces with other families.

mod config;
#[cfg(feature = "cuda")]
mod action_executor;
#[cfg(feature = "cuda")]
mod backbone_executor;
#[cfg(feature = "cuda")]
mod vla_runtime;
mod backbone_weights;
mod weights;

#[cfg(feature = "cuda")]
pub use action_executor::GrootN17ActionExecutor;
#[cfg(feature = "cuda")]
pub use backbone_executor::GrootN17BackboneExecutor;
#[cfg(feature = "cuda")]
pub use vla_runtime::GrootN17VlaRuntime;
pub use backbone_weights::GrootN17BackboneWeights;

pub use config::{GrootN17Config, GrootN17DiffusionConfig, GrootN17VlSelfAttentionConfig};
pub use weights::{
    ActionEncoderWeights, AttentionWeights, CategoryMlpWeights, DitBlockWeights,
    FeedForwardWeights, GrootN17ActionWeights, LayerNormWeights, LinearWeights, VlBlockWeights,
};

#[cfg(feature = "cuda")]
pub(crate) fn register_builtin() {
    crate::registry::register("Gr00tN1d7-cuda", vla_runtime::load_registered);
}
