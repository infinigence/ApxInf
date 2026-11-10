//! Physical Intelligence π0.5 vision-language-action model.
//!
//! OpenPI defines the model math, LeRobot defines the distributed checkpoint
//! contract, and the CUDA fast path is specialized for the static two-view
//! Thor inference shape.  Keep architecture orchestration in this module;
//! CUDA crates expose only kernels and device primitives.

#[cfg(feature = "cuda")]
mod backend;
#[cfg(feature = "cuda")]
mod bf16_executor;
#[cfg(feature = "cuda")]
mod bf16_runtime;
mod bf16_weights;
mod config;
#[cfg(feature = "cuda")]
mod cuda_executor;
mod device_weights;
mod fp8;
#[cfg(feature = "cuda")]
mod int8_executor;
#[cfg(feature = "cuda")]
mod int8_runtime;
#[cfg(feature = "cuda")]
mod int8_weights;
mod math;
#[cfg(feature = "cuda")]
mod runtime;
mod schedule;
mod static_bf16_weights;
#[cfg(feature = "cuda")]
mod static_int8_weights;
mod static_weights;
#[cfg(feature = "cuda")]
mod vla_runtime;
mod weights;

#[cfg(feature = "cuda")]
pub use bf16_executor::{
    action_layer_bf16, language_layer_bf16, vision_layer_bf16, vision_patch_embed_bf16,
    Bf16ActionLayerOutput, Bf16LanguageLayerOutput,
};
#[cfg(feature = "cuda")]
pub use bf16_runtime::{
    upload_time_embeddings_bf16, Bf16PrefixKvCache, Pi05Bf16CapturedGraph, Pi05Bf16CudaRuntime,
};
pub use bf16_weights::{bf16_to_device, Bf16LinearWeights};
pub use config::{GemmaVariantConfig, Pi05Config, Pi05PerformanceProfile};
#[cfg(feature = "cuda")]
pub use cuda_executor::{
    action_layer, language_layer, vision_layer, vision_patch_embed, vision_patch_embed_fp8,
    vision_qkv_packed_from_env, ActionLayerOutput, LanguageLayerOutput, TransformerLayerScales,
    VisionLayerScales,
};
pub use device_weights::{fp16_to_device, Fp8LinearWeights};
pub use fp8::{
    decode_e4m3, dequantize_e4m3, encode_e4m3, quantize_e4m3, quantize_e4m3_absmax, Fp8Tensor,
    StaticFp8Calibration, E4M3_MAX,
};
#[cfg(feature = "cuda")]
pub use int8_executor::{
    action_layer_int8, language_layer_int8, vision_layer_int8, vision_patch_embed_int8,
    Int8ActionLayerOutput, Int8LanguageLayerOutput,
};
#[cfg(feature = "cuda")]
pub use int8_runtime::{
    upload_time_embeddings_int8, Int8PrefixKvCache, Pi05Int8CapturedGraph, Pi05Int8CudaRuntime,
};
#[cfg(feature = "cuda")]
pub use int8_weights::Int8LinearWeights;
pub use math::{discretize_state, euler_flow_step, pi05_prompt, sinusoidal_time_embedding};
#[cfg(feature = "cuda")]
pub use runtime::{
    upload_time_embeddings, Pi05ActivationScales, Pi05CapturedGraph, Pi05CudaRuntime,
    Pi05ImageLayout, PrefixKvCache,
};
pub use schedule::{GemmShape, Pi05ExecutionSchedule, Pi05Stage};
pub use static_bf16_weights::{
    Bf16DeviceActionLayer, Bf16DeviceLanguageLayer, Bf16DeviceLayerNorm, Bf16DeviceVisionBlock,
    StaticBf16Pi05Weights,
};
#[cfg(feature = "cuda")]
pub use static_int8_weights::{
    Int8DeviceActionLayer, Int8DeviceLanguageLayer, Int8DeviceLayerNorm, Int8DeviceVisionBlock,
    StaticInt8Pi05Weights,
};
pub use static_weights::{
    DeviceActionLayer, DeviceLanguageLayer, DeviceLayerNorm, DeviceVisionBlock,
    StaticFp8Pi05Weights,
};
#[cfg(feature = "cuda")]
pub use vla_runtime::{Pi05PreparedInference, Pi05VlaRuntime};
pub use weights::{
    ActionLayerWeights, AdaRmsNormWeights, GemmaAttentionWeights, GemmaMlpWeights,
    LanguageLayerWeights, LayerNormWeights, LinearWeights, Pi05Weights, VisionBlockWeights,
    VisionWeights,
};

#[cfg(feature = "cuda")]
pub(crate) fn register_builtin() {
    crate::registry::register("pi05-cuda", vla_runtime::load_registered);
}
