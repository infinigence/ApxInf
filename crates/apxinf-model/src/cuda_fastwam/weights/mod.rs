//! Checkpoint storage and precision-specific device representations.
mod bf16;
mod host;
pub(in crate::cuda_fastwam) use bf16::*;
pub(in crate::cuda_fastwam) use host::host_tensor;
