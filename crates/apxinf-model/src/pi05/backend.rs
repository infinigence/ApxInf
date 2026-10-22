//! PI0.5's direct cuda-new surface.
//!
//! Other model families keep using the legacy CUDA backend through
//! `crate::accelerator`; PI0.5 owns a cuda-new context and calls its typed
//! operator contracts directly.

use apxinf_core::{Error, NormalGenerator, Result, Tensor};

pub(crate) use apxinf_cuda_new::{
    capture, ops, CudaBuffer as DeviceBuffer, CudaContext as Context, ExecutionSession,
};
pub(crate) use apxinf_cuda_new::{sampling, transfers};

pub(crate) fn to_device(ctx: &Context, tensor: &Tensor) -> Result<Tensor> {
    transfers::to_cuda(tensor, ctx.device_id())
}

pub(crate) fn to_cpu(tensor: &Tensor) -> Result<Tensor> {
    transfers::to_cpu(tensor)
}

pub(crate) fn synchronize(ctx: &Context) -> Result<()> {
    ctx.synchronize().map_err(Error::Cuda)
}

pub(crate) fn create_normal_generator(
    ctx: &Context,
    output: Tensor,
) -> Result<Box<dyn NormalGenerator>> {
    sampling::create_normal_generator(ctx, output)
}

/// Memory layout of a fixed-shape batch of RGB `uint8` images.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageLayout {
    Nhwc,
    Nchw,
}

impl std::fmt::Display for ImageLayout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nhwc => formatter.write_str("nhwc"),
            Self::Nchw => formatter.write_str("nchw"),
        }
    }
}
