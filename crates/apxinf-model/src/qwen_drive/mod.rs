//! Qwen-Drive-1.0 model family: checkpoint configuration, weight mapping,
//! and the planning-expert executor.
//!
//! Model family isolation: every Qwen-Drive-specific type lives behind this
//! module; nothing here is mixed into the generic llama / qwen3vl paths.
//!
//! Device status (honest accounting):
//!
//! * `planner` is an explicitly host-side f32 correctness scaffold (bf16
//!   rounding at the reference's semantic rounding sites), not native GPU
//!   deployment; its device exit criterion is the CUDA executor gate from the
//!   execution ledger.
//! * The VLM scene-cache producer is pending reference qualification, so
//!   end-to-end planning activates only when both halves land.

pub mod config;
pub mod planner;
pub mod weights;

pub use config::{
    PlanningExpertConfig, QwenDriveConfig, QwenDriveTextConfig, QwenDriveVisionConfig,
};
pub use planner::{ExpertConditioning, PlanningExpertModel, SceneCache};
pub use weights::{QwenDriveExpertWeights, QwenDriveVlmWeights};
