//! Execution lifecycle; network and weights never depend on this module.
mod prepare;
mod session;
pub use prepare::{capture_patches, capture_rgb, CapturedGraph};
pub use session::{Pi05PreparedInference, Pi05Session};
