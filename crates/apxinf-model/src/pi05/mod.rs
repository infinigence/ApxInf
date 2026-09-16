//! Physical Intelligence π0.5 vision-language-action model.
//!
//! OpenPI defines the model math, LeRobot defines the distributed checkpoint
//! contract, and the CUDA fast path is specialized for the static two-view
//! Thor inference shape.  Keep architecture orchestration in this module;
//! CUDA crates expose only kernels and device primitives.

#[cfg(feature = "cuda")]
mod backend;
#[cfg(feature = "cuda")]
mod blocks;
#[cfg(feature = "cuda")]
mod calibration;
mod config;
#[cfg(feature = "cuda")]
mod load;
mod math;
#[cfg(feature = "cuda")]
mod network;
#[cfg(feature = "cuda")]
mod prepare;
#[cfg(feature = "cuda")]
mod session;
mod weights;

#[cfg(feature = "cuda")]
pub use blocks::bf16::{
    action_layer_bf16, language_layer_bf16, vision_layer_bf16, vision_patch_embed_bf16,
    Bf16ActionLayerOutput, Bf16LanguageLayerOutput,
};
#[cfg(feature = "cuda")]
pub use blocks::fp8_static::{
    action_layer_fp8_static, language_layer_fp8_static, vision_layer_fp8_static,
    vision_patch_embed_fp8_static, vision_patch_embed_fp8_static_native,
    vision_qkv_packed_from_env, Fp8StaticActionLayerOutput, Fp8StaticLanguageLayerOutput,
};
#[cfg(feature = "cuda")]
pub use blocks::int8_dynamic::{
    action_layer_int8_dynamic, language_layer_int8_dynamic, vision_layer_int8_dynamic,
    vision_patch_embed_int8_dynamic, Int8DynamicActionLayerOutput, Int8DynamicLanguageLayerOutput,
};
#[cfg(feature = "cuda")]
pub use calibration::Pi05CalibrationObserver;
pub use config::{ComputeVariant, GemmaVariantConfig, Pi05Config, Pi05PerformanceProfile};
pub use math::{discretize_state, euler_flow_step, pi05_prompt, sinusoidal_time_embedding};
#[cfg(feature = "cuda")]
pub use session::{Pi05PreparedInference, Pi05Session};
pub use weights::*;

#[cfg(feature = "cuda")]
pub(crate) fn register_builtin() {
    crate::registry::register("pi05-cuda", load::load_registered);
}

#[cfg(feature = "cuda")]
pub use backend::ImageLayout as Pi05ImageLayout;
#[cfg(feature = "cuda")]
pub use blocks::{Bf16PrefixKvCache, Fp8StaticPrefixKvCache, Int8DynamicPrefixKvCache};
#[cfg(feature = "cuda")]
pub use load::{
    build_bf16_network, build_fp8_static_network, build_int8_dynamic_network,
    upload_time_embeddings_bf16, upload_time_embeddings_fp8_static,
    upload_time_embeddings_int8_dynamic,
};
#[cfg(feature = "cuda")]
pub use prepare::{capture_patches, capture_rgb, CapturedGraph};
#[cfg(feature = "cuda")]
pub type Bf16Network = std::sync::Arc<network::Pi05Network<blocks::Bf16Blocks>>;
#[cfg(feature = "cuda")]
pub type Fp8StaticNetwork = std::sync::Arc<network::Pi05Network<blocks::Fp8StaticBlocks>>;
#[cfg(feature = "cuda")]
pub type Int8DynamicNetwork = std::sync::Arc<network::Pi05Network<blocks::Int8DynamicBlocks>>;
