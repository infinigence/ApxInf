//! LingBot-VLA 2.0 (Robbyant) on cuda-new: Qwen3-VL-4B prefix plus a 36-layer MoE action expert, W8A8 ConvRot.
//!
//! Self-contained port: this module imports no other model family. See `doc/lingbot-vla2.md`.

mod backend;
mod config;
mod load;
pub(crate) mod math;
mod model;
mod model_runner;
mod weights;

pub use config::LingbotVla2Config;
pub use model_runner::LingbotVla2ModelRunner;

pub(crate) fn register_builtin() {
    crate::registry::register("lingbot_vla2", load::load_registered);
}
