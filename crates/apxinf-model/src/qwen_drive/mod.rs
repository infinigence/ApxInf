//! Qwen-Drive-1.0 model family: checkpoint configuration, weight mapping,
//! the native CUDA VLM executor, and the planning-expert device executor.
//!
//! Model family isolation: every Qwen-Drive-specific type lives behind this
//! module; nothing here is mixed into the generic llama / qwen3vl / pi05
//! paths.
//!
//! Device status (honest accounting):
//!
//! * `general` is the native CUDA VLM: hybrid gated-delta (linear attention)
//!   and gated full-attention layers with partial interleaved mRoPE, a ViT
//!   vision tower, the LlmTrait surface, and the VQA / direct-planning /
//!   reasoning-planning flows. All layer mathematics run on device through
//!   the model-neutral kernel facade plus the new linear-attention operators.
//! * `expert` is the native CUDA planning-expert executor (flow matching,
//!   adaLN, joint attention over the VLM's exported post-rotary scene caches).
//! * `planner` is the retained CPU f32 correctness scaffold from the saved K3
//!   base; it is the replay oracle for the device executor, not a deployment
//!   path.

/// Every development diagnostic in this model goes through here.
///
/// They were `eprintln!` calls, and on a normal request they ran: the vision
/// tower printed once per block, `generate` printed its entry, the tail of the
/// prompt, prefill completion and a decode heartbeat. That floods stderr for
/// anyone running inference, and several of the lines carry prompt and
/// generated token ids, which is request content and should not leave the
/// process because a developer left a probe in. Routing them through one macro
/// makes the switch the only way to reach any of them, and makes a new one
/// impossible to add without it.
macro_rules! qdiag {
    ($($arg:tt)*) => {{
        if $crate::qwen_drive::general::diagnostics_enabled() {
            eprintln!($($arg)*);
        }
    }};
}
// Declared above the modules on purpose: `macro_rules!` is in scope for
// everything that follows it in this file, which is how the submodules below
// reach it. Keep new modules below this line.

pub mod config;
pub mod planner;
pub mod weights;

#[cfg(feature = "cuda")]
pub mod backend;
#[cfg(feature = "cuda")]
pub mod device_weights;
#[cfg(feature = "cuda")]
pub mod expert;
#[cfg(feature = "cuda")]
pub mod general;
#[cfg(feature = "cuda")]
pub mod vision;

pub use config::{
    PlanningExpertConfig, QwenDriveConfig, QwenDriveTextConfig, QwenDriveVisionConfig,
};
#[cfg(feature = "cuda")]
pub use general::QwenDriveModel;
pub use planner::{ExpertConditioning, PlanningExpertModel, SceneCache};
pub use weights::{QwenDriveExpertWeights, QwenDriveVlmWeights};
