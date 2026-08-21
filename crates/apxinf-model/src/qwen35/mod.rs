//! Qwen3.5 (Qwen3.8-27B-AWQ-INT4) text-model support.
//!
//! Only the language-model half is implemented (the service exposes
//! `multimodal: false`). Vision and MTP weights are ignored.

pub mod config;
pub mod cpu;
pub mod dequant;
pub mod safetensors_raw;
pub mod weights;

pub use config::{LayerType, Qwen35TextConfig};
