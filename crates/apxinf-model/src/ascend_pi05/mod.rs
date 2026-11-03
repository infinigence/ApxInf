//! PI0.5 vision-language-action model — Ascend NPU edition.
//!
//! Standalone Ascend port: mirrors `pi05/` but calls `apxinf-ascend`
//! L3 operator contracts directly. Hardware-agnostic modules (config, math,
//! host weights, packing, calibration) are copied locally to keep the model
//! family self-contained.

// Hardware-agnostic modules (copied from pi05/ to satisfy model family
// boundary checks — see scripts/check_model_family_boundaries.sh).
mod config;
mod math;

pub use config::{
    GemmaVariantConfig, ModelVariantChoice, Pi05Config, Pi05PerformanceProfile,
};
pub use math::{discretize_state, euler_flow_step, pi05_prompt, sinusoidal_time_embedding};

#[cfg(feature = "ascend")]
mod backend;

#[cfg(feature = "ascend")]
mod load;

#[cfg(feature = "ascend")]
mod model;

#[cfg(feature = "ascend")]
mod model_runner;

#[cfg(feature = "ascend")]
pub mod weights;

#[cfg(feature = "ascend")]
pub(crate) use load::load_with_ascend;

#[cfg(feature = "ascend")]
pub use backend::ImageLayout as AscendPi05ImageLayout;

#[cfg(feature = "ascend")]
pub use model::{
    build_bf16_model, upload_time_embeddings_bf16, Bf16Model, Bf16PrefixKvCache,
    Pi05CalibrationObserver,
};

#[cfg(feature = "ascend")]
pub use model_runner::{Pi05ModelRunner, Pi05PreparedInference};

#[cfg(feature = "ascend")]
pub use model_runner::{capture_patches, capture_rgb, CapturedGraph};

#[cfg(feature = "ascend")]
pub use weights::{
    Bf16Weights, Bf16LinearWeights, Bf16DeviceLayerNorm, Bf16DeviceVisionBlock,
    Bf16DeviceLanguageLayer, Bf16DeviceActionLayer,
    ActionLayerWeights, AdaRmsNormWeights, LanguageLayerWeights, LayerNormWeights,
    LinearWeights, Pi05Weights, VisionBlockWeights, VisionWeights,
    Pi05CalibrationPlan,
};
