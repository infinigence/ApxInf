//! PI0.5 on Moore Threads GPUs, written directly against `apxinf_musa` the way
//! `pi05/` is written against `apxinf-cuda-new`.
//!
//! A self-contained port: it owns its config, checkpoint reading, BF16 device
//! weights, layer computations, inference plan and loader, and imports no other
//! model module. Agreement with the CUDA implementation comes from both passing
//! the same accuracy fixtures, not from shared code. Frontend config overrides
//! (`LoadOptions::config`, typed as `pi05::Pi05Config`) are converted by
//! `AutoModel`, outside this module, and arrive here as this port's own
//! [`Pi05Config`].
//!
//! BF16 and INT8 (W8A8) are the precision paths. Capture is the port's own:
//! [`model_runner`] prepares a traversal and records it, and a shape the
//! preparation never ran is refused by name.

mod backend;
pub mod config;
mod load;
mod math;
mod model;
mod model_runner;
mod weights;

pub use backend::ImageLayout as Pi05ImageLayout;
pub use config::{GemmaVariantConfig, ModelVariantChoice, Pi05Config};
pub(crate) use load::load;
pub use math::{discretize_state, euler_flow_step, pi05_prompt, sinusoidal_time_embedding};
pub use model::Bf16PrefixKvCache;
pub use model::Int8DynamicPrefixKvCache;
pub use model::{
    action_layer_bf16, build_bf16_model, build_int8_dynamic_model, language_layer_bf16,
    vision_layer_bf16, vision_patch_embed_bf16, Bf16ActionLayerOutput, Bf16LanguageLayerOutput,
};
pub use model::{
    action_layer_int8_dynamic, language_layer_int8_dynamic, vision_layer_int8_dynamic,
    vision_patch_embed_int8_dynamic, Int8DynamicActionLayerOutput, Int8DynamicLanguageLayerOutput,
};
pub use model_runner::{CapturedGraph, Pi05ModelRunner};
pub use weights::*;
