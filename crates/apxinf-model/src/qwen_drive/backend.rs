//! Compile-time backend seam for the Qwen-Drive model family.
//!
//! Executor code depends on this model-local alias and the model-neutral
//! kernel contract; adding another accelerator backend changes this seam,
//! not the layer topology. Mirrors the maintained pi05 pattern.

pub(crate) use crate::accelerator::cuda::{
    downcast_arc, kernels, transfers, Context, CublasTranspose, DeviceBuffer, RuntimeBackend,
};
