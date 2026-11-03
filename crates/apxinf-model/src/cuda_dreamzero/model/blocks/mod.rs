//! One precision implementation; the model never sees concrete modality storage.
mod bf16;
pub(in crate::cuda_dreamzero) use bf16::History;
pub(in crate::cuda_dreamzero) use bf16::{allocate, binary, staged_l2, staged_rms};
pub(super) use bf16::{Bf16Blocks, Workspace};
