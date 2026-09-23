//! Model task composition. Planning reuses independently constructed BF16 blocks.
pub(crate) mod blocks;
mod vla;

pub(crate) use blocks::bf16::{BackboneState as PlanningState, VisionState};
pub(crate) use blocks::{GdnExecution, GdnRequest};
pub(crate) use vla::{PlanningInput, QwenDriveModel, ReasoningInput};
