//! PI0.5's Ascend backend surface.
//!
//! Mirrors `pi05::backend` but re-exports `apxinf-ascend` types instead
//! of `apxinf-cuda-new`. The L3 operator contracts (ops::*) have the same
//! API, so `blocks/bf16.rs` logic is identical — only the backend import changes.

use apxinf_core::{Error, NormalGenerator, Result, Tensor};

pub(crate) use apxinf_ascend::l3::{
    capture, ops, AscendBuffer as DeviceBuffer, AscendNewContext as Context, ExecutionSession,
};
pub(crate) use apxinf_ascend::l3::{sampling, transfers};

pub(crate) fn to_device(ctx: &Context, tensor: &Tensor) -> Result<Tensor> {
    ctx.ensure_context();
    // Use async H2D to avoid per-copy host blocking. The caller (Bf16Weights::from_host)
    // does a single synchronize after all weights are uploaded.
    transfers::to_npu_async(tensor, ctx.device_id(), ctx.raw_stream())
}

pub(crate) fn to_cpu(tensor: &Tensor) -> Result<Tensor> {
    transfers::to_cpu(tensor)
}

pub(crate) fn synchronize(ctx: &Context) -> Result<()> {
    ctx.ensure_context();
    ctx.synchronize().map_err(|e| Error::Other(format!("ascend sync: {e}")))
}

pub(crate) fn create_normal_generator(
    ctx: &Context,
    output: Tensor,
) -> Result<Box<dyn NormalGenerator>> {
    ctx.ensure_context();
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
