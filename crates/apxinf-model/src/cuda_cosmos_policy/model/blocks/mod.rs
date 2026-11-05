//! Precision-specific computation. No graph or runner dependencies.
mod bf16;
pub(super) use bf16::Bf16Blocks;
