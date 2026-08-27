//! Input preprocessing operator contracts.

use apxinf_core::{DType, Error, Result, Tensor};

use super::contracts::{gpu_ptr, make_gpu_tensor};
use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;
use crate::workspace::output_buffer;

pub fn im2col1d_bf16(
    ctx: &CudaContext,
    input: &Tensor,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    let dims = input.shape().dims();
    if input.dtype() != DType::BF16 || dims.len() != 2 || kernel == 0 || stride == 0 {
        return Err(Error::Other(
            "im2col1d: expected rank-2 BF16 input and positive kernel/stride".into(),
        ));
    }
    let (frames, channels) = (dims[0], dims[1]);
    let padded = frames
        .checked_add(2 * padding)
        .ok_or_else(|| Error::Other("im2col1d: padded length overflow".into()))?;
    if padded < kernel {
        return Err(Error::Other("im2col1d: kernel exceeds padded input".into()));
    }
    let output_frames = (padded - kernel) / stride + 1;
    let shape = apxinf_core::Shape::new(vec![output_frames, channels * kernel]);
    let output = output_buffer(
        ctx,
        output_frames * channels * kernel * DType::BF16.size_in_bytes(),
    )?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_im2col1d_bf16(
            gpu_ptr(input)?,
            output.ptr(),
            frames as i32,
            channels as i32,
            kernel as i32,
            stride as i32,
            padding as i32,
            output_frames as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        shape,
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

pub fn avg_pool1d_bf16(
    ctx: &CudaContext,
    input: &Tensor,
    kernel: usize,
    stride: usize,
) -> Result<Tensor> {
    let dims = input.shape().dims();
    if input.dtype() != DType::BF16
        || dims.len() != 2
        || kernel == 0
        || stride == 0
        || dims[0] < kernel
    {
        return Err(Error::Other("avg_pool1d: invalid BF16 input or parameters".into()));
    }
    let (frames, channels) = (dims[0], dims[1]);
    let output_frames = (frames - kernel) / stride + 1;
    let output = output_buffer(
        ctx,
        output_frames * channels * DType::BF16.size_in_bytes(),
    )?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_avg_pool1d_bf16(
            gpu_ptr(input)?,
            output.ptr(),
            frames as i32,
            channels as i32,
            kernel as i32,
            stride as i32,
            output_frames as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        apxinf_core::Shape::new(vec![output_frames, channels]),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}
/// Memory layout of a fixed-shape batch of RGB `uint8` images.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageLayout {
    /// `[views, image_size, image_size, 3]`.
    Nhwc,
    /// `[views, 3, image_size, image_size]`.
    Nchw,
}

impl ImageLayout {
    fn kernel_value(self) -> i32 {
        match self {
            Self::Nhwc => 0,
            Self::Nchw => 1,
        }
    }
}

impl std::fmt::Display for ImageLayout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nhwc => formatter.write_str("nhwc"),
            Self::Nchw => formatter.write_str("nchw"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn rgb_u8_to_patches_bf16(
    ctx: &CudaContext,
    images: &CudaBuffer,
    patches: &Tensor,
    views: usize,
    image_size: usize,
    patch_size: usize,
    layout: ImageLayout,
) -> Result<()> {
    if views == 0 || image_size == 0 || patch_size == 0 || image_size % patch_size != 0 {
        return Err(Error::Other(
            "invalid static inference BF16 image preprocessing shape".into(),
        ));
    }
    let expected_bytes = views * 3 * image_size * image_size;
    let side = image_size / patch_size;
    let expected_shape = [views * side * side, 3 * patch_size * patch_size];
    if images.device() != ctx.device_id()
        || images.len() != expected_bytes
        || patches.dtype() != DType::BF16
        || patches.shape().dims() != expected_shape
    {
        return Err(Error::Other(format!(
            "static inference BF16 raw image/preprocessed patch mismatch: image bytes {}, patches {} {:?}",
            images.len(),
            patches.dtype(),
            patches.shape().dims()
        )));
    }
    let layout = match layout {
        ImageLayout::Nhwc => 0,
        ImageLayout::Nchw => 1,
    };
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_rgb_u8_to_patches_bf16(
            images.ptr(),
            gpu_ptr(patches)?,
            views as i32,
            image_size as i32,
            patch_size as i32,
            layout,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}
/// Fused static inference image preprocessing for an already resized RGB image batch.
///
/// The input is `uint8` NHWC or NCHW. The output is patch-major E4M3 with
/// shape `[views*(image_size/patch_size)^2, 3*patch_size^2]`. Normalization,
/// channel-first patch reordering, FP16 boundary rounding, and static FP8
/// quantization are performed by one stream-ordered CUDA kernel.
#[allow(clippy::too_many_arguments, clippy::manual_is_multiple_of)]
pub fn rgb_u8_to_patches_e4m3(
    ctx: &CudaContext,
    images: &CudaBuffer,
    patches: &Tensor,
    views: usize,
    image_size: usize,
    patch_size: usize,
    layout: ImageLayout,
    scale: f32,
) -> Result<()> {
    if views == 0
        || image_size == 0
        || patch_size == 0
        || image_size % patch_size != 0
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(Error::Other(format!(
            "invalid static inference image preprocessing parameters: views={views}, image_size={image_size}, patch_size={patch_size}, scale={scale}"
        )));
    }
    let expected_image_bytes = views
        .checked_mul(3)
        .and_then(|value| value.checked_mul(image_size))
        .and_then(|value| value.checked_mul(image_size))
        .ok_or_else(|| Error::Other("static inference raw image size overflow".into()))?;
    let patches_per_side = image_size / patch_size;
    let expected_shape = [
        views * patches_per_side * patches_per_side,
        3 * patch_size * patch_size,
    ];
    if images.device() != ctx.device_id() || images.len() != expected_image_bytes {
        return Err(Error::Other(format!(
            "static inference raw images must contain exactly {expected_image_bytes} bytes on CUDA {}, got {} bytes on CUDA {}",
            ctx.device_id(),
            images.len(),
            images.device()
        )));
    }
    if patches.dtype() != DType::F8E4M3
        || patches.shape().dims() != expected_shape
        || patches.device() != apxinf_core::Device::Cuda(ctx.device_id())
    {
        return Err(Error::Other(format!(
            "static inference preprocessed output must be E4M3 {:?} on CUDA {}, got {} {:?} on {}",
            expected_shape,
            ctx.device_id(),
            patches.dtype(),
            patches.shape().dims(),
            patches.device()
        )));
    }
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_rgb_u8_to_patches_e4m3(
            images.ptr(),
            gpu_ptr(patches)?,
            views as i32,
            image_size as i32,
            patch_size as i32,
            layout.kernel_value(),
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}
