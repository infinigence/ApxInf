//! NVIDIA GR00T N1.7 VLA.
//!
//! The runtime consumes target-built TensorRT plans through ApxInf's native
//! CUDA layer. It does not import the reference Python package at runtime.

mod config;
mod engine_bundle;
#[cfg(feature = "cuda")]
mod runtime;

pub use config::{DiffusionConfig, Gr00tN17Config, VlSelfAttentionConfig};
pub use engine_bundle::{EngineBundle, EngineMetadata};

#[cfg(feature = "cuda")]
pub(crate) fn register_builtin() {
    crate::registry::register("Gr00tN1d7-cuda", runtime::load_registered);
    crate::registry::register("gr00t_n17-cuda", runtime::load_registered);
}
