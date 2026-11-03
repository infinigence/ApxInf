//! Precision-specific computation; graph ownership stays in model_runner.
mod bf16;
pub(super) use bf16::Bf16Blocks;
