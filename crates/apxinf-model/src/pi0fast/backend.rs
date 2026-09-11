//! Compile-time backend seam for π0-FAST executors.
//!
//! Executor code depends on this model-local alias and the model-neutral
//! kernel contract. Adding another accelerator backend changes this seam,
//! not the layer topology.

pub(crate) use crate::accelerator::cuda::kernels::preprocess::ImageLayout;
pub(crate) use crate::accelerator::cuda::{
    kernels, Context, DeviceAddress, DeviceBuffer, RuntimeBackend,
};
