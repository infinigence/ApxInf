//! Physical Intelligence π0.5 vision-language-action model.
//!
//! OpenPI defines the model math, LeRobot defines the distributed checkpoint
//! contract. Loading selects compute Blocks; Network owns model dataflow;
//! Session and prepare own execution policy and stable graph resources.
//! CUDA crates expose kernels and device primitives.

#[cfg(feature = "cuda")]
mod backend;
mod config;
#[cfg(feature = "cuda")]
mod execution;
#[cfg(feature = "cuda")]
mod load;
mod math;
#[cfg(feature = "cuda")]
mod network;
mod weights;

pub use config::{ComputeVariant, GemmaVariantConfig, Pi05Config, Pi05PerformanceProfile};
#[cfg(feature = "cuda")]
pub use execution::{Pi05PreparedInference, Pi05Session};
pub use math::{discretize_state, euler_flow_step, pi05_prompt, sinusoidal_time_embedding};
#[cfg(feature = "cuda")]
pub use network::Pi05CalibrationObserver;
#[cfg(feature = "cuda")]
pub use network::{
    action_layer_bf16, language_layer_bf16, vision_layer_bf16, vision_patch_embed_bf16,
    Bf16ActionLayerOutput, Bf16LanguageLayerOutput,
};
#[cfg(feature = "cuda")]
pub use network::{
    action_layer_fp8_static, language_layer_fp8_static, vision_layer_fp8_static,
    vision_patch_embed_fp8_static, vision_patch_embed_fp8_static_native,
    vision_qkv_packed_from_env, Fp8StaticActionLayerOutput, Fp8StaticLanguageLayerOutput,
};
#[cfg(feature = "cuda")]
pub use network::{
    action_layer_int8_dynamic, language_layer_int8_dynamic, vision_layer_int8_dynamic,
    vision_patch_embed_int8_dynamic, Int8DynamicActionLayerOutput, Int8DynamicLanguageLayerOutput,
};
pub use weights::*;

#[cfg(feature = "cuda")]
pub(crate) fn register_builtin() {
    crate::registry::register("pi05-cuda", load::load_registered);
}

#[cfg(feature = "cuda")]
pub use backend::ImageLayout as Pi05ImageLayout;
#[cfg(feature = "cuda")]
pub use execution::{capture_patches, capture_rgb, CapturedGraph};
#[cfg(feature = "cuda")]
pub use network::{
    build_bf16_network, build_fp8_static_network, build_int8_dynamic_network,
    upload_time_embeddings_bf16, upload_time_embeddings_fp8_static,
    upload_time_embeddings_int8_dynamic,
};
#[cfg(feature = "cuda")]
pub use network::{Bf16Network, Fp8StaticNetwork, Int8DynamicNetwork};
#[cfg(feature = "cuda")]
pub use network::{Bf16PrefixKvCache, Fp8StaticPrefixKvCache, Int8DynamicPrefixKvCache};
