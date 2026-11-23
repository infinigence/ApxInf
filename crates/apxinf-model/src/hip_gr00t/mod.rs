//! GR00T N1.7 on AMD GPUs, written directly against HIP (`apxinf_hip::native`).
//!
//! A self-contained port: it owns its config, checkpoint reader, device
//! weights, vision tower, text backbone, DiT action head, inference runner and
//! loader, and imports no other model module. Agreement with the CUDA
//! implementation comes from both passing the same accuracy fixtures, not from
//! shared code.
//!
//! The module layout mirrors PI0.5: `model/` owns the forward math
//! (`model::blocks` holds the concrete BF16 blocks), `model_runner/` owns
//! request validation, the plan cache and execution, `weights/` owns checkpoint
//! mapping and the device weight tree. BF16 only; `select_layer` and the DiT
//! depth come from the checkpoint, not from the configuration defaults.

pub mod assets;
pub mod config;
pub mod load;
pub mod math;
pub mod model;
pub mod model_runner;
pub mod weights;

pub use config::{Gr00tConfig, Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig};
pub use load::load_runner;
pub use model::{HipGr00tModel, ModelVariant};
pub use model_runner::HipGr00tRunner;
/// The uploaded embodiment adapters, produced through the model interface and
/// opaque below it. Re-exported at the family root so the runner never names a
/// `weights::` path.
pub use weights::Embodiment;

pub(crate) use load::register_builtin;
