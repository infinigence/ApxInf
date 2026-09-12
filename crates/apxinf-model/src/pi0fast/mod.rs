//! Physical Intelligence π0-FAST vision-language-action model.
//!
//! π0-FAST shares the PaliGemma backbone (SigLIP So400m/14 vision + Gemma-2B
//! text) with π0.5 but replaces flow-matching action generation with
//! autoregressive FAST action-token decoding: the same LM head emits discrete
//! action tokens that a FAST tokenizer maps back to continuous actions.
//!
//! OpenPI defines the model math and LeRobot the checkpoint contract; this
//! module keeps its own architecture code so the family stays independent of
//! `pi05`/`walloss`, composing only model-neutral kernels.

#[cfg(feature = "cuda")]
mod backend;
#[cfg(feature = "cuda")]
mod bf16_executor;
#[cfg(feature = "cuda")]
mod bf16_runtime;
#[cfg(feature = "cuda")]
mod vla_runtime;
mod bf16_weights;
mod config;
mod device_weights;
mod static_bf16_weights;
mod weights;

#[cfg(feature = "cuda")]
pub use bf16_executor::{
    language_layer_bf16, language_layer_cached_bf16, language_layer_cached_decode_bf16,
    vision_layer_bf16,
    vision_patch_embed_f32_bf16, Bf16LanguageLayerOutput,
};
pub use bf16_weights::{bf16_to_device, f32_to_device, Bf16LinearWeights};
#[cfg(feature = "cuda")]
pub use bf16_runtime::Pi0FastBf16Runtime;
#[cfg(feature = "cuda")]
pub use vla_runtime::Pi0FastVlaRuntime;
pub use config::{Pi0FastConfig, Pi0FastLanguageConfig, Pi0FastPerformanceProfile};
#[cfg(feature = "cuda")]
pub use static_bf16_weights::{
    Bf16DeviceLanguageLayer, Bf16DeviceLayerNorm, Bf16DeviceVisionBlock, StaticBf16Pi0FastWeights,
    VisionPatchEmbeddingF32,
};
pub use weights::{
    GemmaAttentionWeights, GemmaMlpWeights, LanguageLayerWeights, LayerNormWeights,
    LinearWeights, Pi0FastWeights, VisionBlockWeights, VisionWeights,
};

#[cfg(feature = "cuda")]
pub(crate) fn register_builtin() {
    crate::registry::register("pi0fast-cuda", vla_runtime::load_registered);
    crate::registry::register("pi0_fast-cuda", vla_runtime::load_registered);
}
