//! Execution lifecycle; model and weights never depend on this module.
mod prepare;
mod runner;
pub use prepare::{capture_patches, capture_rgb, CapturedGraph};
pub use runner::{Pi05ModelRunner, Pi05PreparedInference};
