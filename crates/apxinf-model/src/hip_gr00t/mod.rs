//! GR00T N1.7 on AMD GPUs, written directly against HIP (`apxinf_hip::native`)
//! the way `gr00t/` is written against `apxinf-cuda-new`.
//!
//! A self-contained port: it owns its config, checkpoint reader, device
//! weights, vision tower, text backbone, DiT action head, inference runner and
//! loader, and imports no other model module. Agreement with the CUDA
//! implementation comes from both passing the same accuracy fixtures, not from
//! shared code.
//!
//! BF16 only, eager. `select_layer` and the DiT depth come from the
//! checkpoint, not from the configuration defaults.

pub mod action_head;
pub mod assets;
pub mod backbone;
pub mod checkpoint;
pub mod config;
pub mod geometry;
pub mod load;
pub mod math;
pub mod model;
pub mod runner;
pub mod stages;
pub mod vision;
pub mod weights;

pub use config::{Gr00tConfig, Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig};
pub use load::load_runner;
pub use model::HipGr00tModel;
pub use runner::HipGr00tRunner;

pub(crate) use load::register_builtin;
