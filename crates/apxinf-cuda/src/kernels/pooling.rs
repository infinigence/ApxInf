//! Device pooling operators on contiguous feature maps.
use super::contracts::{checked_bytes, gpu_ptr};
use crate::{context::CudaContext, ffi, workspace::output_buffer};
use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

/// NCHW adaptive average pooling to [N,C,1,1].
pub fn global_mean_bf16(ctx: &CudaContext, x: &Tensor) -> Result<Tensor> {
    let d = x.shape().dims();
    if d.len() != 4 || x.dtype() != DType::BF16 || x.device() != Device::Cuda(ctx.device_id()) {
        return Err(Error::Other(
            "global mean requires NCHW BF16 on the context device".into(),
        ));
    }
    checked_bytes(DType::BF16, d, "global mean input")?;
    let rows = d[0]
        .checked_mul(d[1])
        .ok_or_else(|| Error::Other("global mean row overflow".into()))?;
    let spatial = d[2]
        .checked_mul(d[3])
        .ok_or_else(|| Error::Other("global mean spatial overflow".into()))?;
    let int = |v: usize| {
        i32::try_from(v).map_err(|_| Error::Other("global mean dimension overflow".into()))
    };
    let out = output_buffer(
        ctx,
        checked_bytes(DType::BF16, &[rows], "global mean output")?,
    )?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_global_mean_bf16(
            gpu_ptr(x)?,
            out.ptr(),
            int(rows)?,
            int(spatial)?,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(out.into_tensor(Shape::new(vec![d[0], d[1], 1, 1]), DType::BF16))
}

/// NCHW max pooling with a 2x2 window, stride 2, no padding and floor dimensions.
pub fn max_pool2x2_bf16(ctx: &CudaContext, x: &Tensor) -> Result<Tensor> {
    let d = x.shape().dims();
    if d.len() != 4
        || d[2] < 2
        || d[3] < 2
        || x.dtype() != DType::BF16
        || x.device() != Device::Cuda(ctx.device_id())
    {
        return Err(Error::Other(
            "max_pool2x2 requires nonempty NCHW BF16 input on the context device".into(),
        ));
    }
    checked_bytes(DType::BF16, d, "max_pool2x2 input")?;
    let shape = vec![d[0], d[1], d[2] / 2, d[3] / 2];
    let bytes = checked_bytes(DType::BF16, &shape, "max_pool2x2 output")?;
    let int = |v: usize| {
        i32::try_from(v).map_err(|_| Error::Other("max_pool2x2 dimension overflow".into()))
    };
    int(d[2]
        .checked_mul(d[3])
        .ok_or_else(|| Error::Other("max_pool2x2 spatial overflow".into()))?)?;
    let (height, width) = (int(d[2])?, int(d[3])?);
    let count =
        i64::try_from(bytes / 2).map_err(|_| Error::Other("max_pool2x2 size overflow".into()))?;
    let out = output_buffer(ctx, bytes)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_max_pool2x2_bf16(
            gpu_ptr(x)?,
            out.ptr(),
            height,
            width,
            count,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(out.into_tensor(Shape::new(shape), DType::BF16))
}
