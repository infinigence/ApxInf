//! Compile-time accelerator seam for DM05.
//!
//! Model code imports only model-neutral safe Rust kernels and transfer/
//! workspace facilities through this module. Raw CUDA, FFI and vendor handles
//! remain owned by `apxinf-cuda`.

pub use crate::accelerator::cuda::kernels::preprocess::ImageLayout;
pub(crate) use crate::accelerator::cuda::{
    kernels, transfers, Context, DeviceBuffer, RuntimeBackend,
};
