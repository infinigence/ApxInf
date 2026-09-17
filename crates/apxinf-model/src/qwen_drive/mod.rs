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
//! * `perception_fpn` and `perception_depth` are native feature-pyramid and
//!   depth-network components. The public perception path remains pending
//!   until camera projection, voxel pooling, BEV processing and output heads
//!   are connected and verified together.

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
pub mod perception_depth;
#[cfg(feature = "cuda")]
pub mod perception_fpn;
#[cfg(feature = "cuda")]
pub mod perception_view;
#[cfg(feature = "cuda")]
pub mod vision;

pub use config::{
    PlanningExpertConfig, QwenDriveConfig, QwenDriveTextConfig, QwenDriveVisionConfig,
};
#[cfg(feature = "cuda")]
pub use general::QwenDriveModel;
pub use planner::{ExpertConditioning, PlanningExpertModel, SceneCache};
pub use weights::{QwenDriveExpertWeights, QwenDriveVlmWeights};
