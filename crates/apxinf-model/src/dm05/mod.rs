//! Dexmal DM05 vision-language-action model.
//!
//! OpenDM defines the reference semantics and checkpoint contract. This module
//! owns the ApxInf-native architecture, weight mapping and execution schedule;
//! device work is reachable only through the model-local `backend` seam and
//! model-neutral safe Rust operators.

#[cfg(feature = "cuda")]
mod backend;
mod config;
mod device_weights;
#[cfg(feature = "cuda")]
mod executor;
mod math;
#[cfg(feature = "cuda")]
mod runtime;
#[cfg(feature = "cuda")]
mod vla_runtime;
mod weights;

pub use config::{
    Dm05Config, Dm05LayerType, Dm05RopeConfig, Dm05TextConfig, Dm05VisionConfig,
    DM05_HISTORY_PAD_TOKEN_ID, DM05_IMAGE_TOKEN_ID,
};
pub use device_weights::{
    DeviceActionLayer, DeviceAttention, DeviceDm05Weights, DeviceGemmaRms, DeviceLanguageLayer,
    DeviceLayerNorm, DeviceLinear, DeviceMlp, DeviceVisionBlock,
};
#[cfg(feature = "cuda")]
pub use executor::{
    action_layer, action_projection, encode_vision, language_layer, merge_prefix, modulation_style,
    rope_kind_for_layer, ActionStyles, PrefixLayerOutput,
};
pub use math::{
    action_mask, prefix_attention_segments, projector_pool_matrix, time_values, AttentionSegment,
};
#[cfg(feature = "cuda")]
pub use runtime::{Dm05Bf16Runtime, Dm05PreparedShape, PrefixKvCache};
#[cfg(feature = "cuda")]
pub use vla_runtime::{Dm05PreparedInference, Dm05VlaRuntime};
pub use weights::{
    ActionLayerWeights, Dm05Weights, GemmaAttentionWeights, GemmaMlpWeights, GemmaRmsWeights,
    LanguageLayerWeights, LayerNormWeights, LinearWeights, VisionBlockWeights, VisionWeights,
};

#[cfg(feature = "cuda")]
pub(crate) fn register_builtin() {
    crate::registry::register("dm05-cuda", vla_runtime::load_registered);
}
