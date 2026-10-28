//! Checkpoint storage and precision-specific device representations.
pub(in crate::cuda_fastwam) mod bf16;
mod host;
pub(in crate::cuda_fastwam) use bf16::LinearWeights;
pub(in crate::cuda_fastwam) use host::host_tensor;
