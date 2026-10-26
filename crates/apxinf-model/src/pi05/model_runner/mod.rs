//! Execution lifecycle; model and weights never depend on this module.
#[cfg(feature = "cuda")]
mod prepare;
#[cfg(feature = "cuda")]
mod runner;
#[cfg(feature = "cuda")]
pub use prepare::{capture_patches, capture_rgb, CapturedGraph};
#[cfg(feature = "cuda")]
pub use runner::{Pi05ModelRunner, Pi05PreparedInference};

pub(in crate::pi05) mod cpu;
