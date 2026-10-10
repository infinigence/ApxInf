//! Legacy `kernels::preprocess` names over the cuda-new gather operator.

use apxinf_core::{DType, Error, Result, Tensor};

use crate::{ops, CudaBuffer, CudaContext};

/// Memory layout of a resized RGB `u8` image batch. Mirrors the legacy enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageLayout {
    Nhwc,
    Nchw,
}

/// `rgb_u8_to_patches_bf16`: normalize a `u8` RGB batch to `[-1, 1]` and
/// patchify into `[views*(size/patch)^2, 3*patch^2]` BF16.
pub fn rgb_u8_to_patches_bf16(
    ctx: &CudaContext,
    images: &CudaBuffer,
    patches: &Tensor,
    views: usize,
    image_size: usize,
    patch_size: usize,
    layout: ImageLayout,
) -> Result<()> {
    if patches.dtype() != DType::BF16 {
        return Err(Error::Other(
            "cuda-new RGB preprocessing writes BF16 patches".into(),
        ));
    }
    // GatherArgs mutates only device memory; clone the handle for the &mut
    // contract without copying storage.
    let mut out = patches.clone();
    let args = ops::GatherArgs::rgb_to_patches(
        images,
        &mut out,
        ops::GatherPatchGeometry {
            views,
            image_size,
            patch_size,
            nhwc: layout == ImageLayout::Nhwc,
        },
    );
    ops::gather(ctx, args)
}

/// `rgb_u8_to_patches_f32`: the F32-output variant of the patchification,
/// used by pi0-fast whose vision tower keeps F32 until the first projection.
pub fn rgb_u8_to_patches_f32(
    ctx: &CudaContext,
    images: &CudaBuffer,
    patches: &Tensor,
    views: usize,
    image_size: usize,
    patch_size: usize,
    layout: ImageLayout,
) -> Result<()> {
    if patches.dtype() != DType::F32 {
        return Err(Error::Other(
            "rgb_u8_to_patches_f32 writes F32 patches".into(),
        ));
    }
    let mut out = patches.clone();
    let args = ops::GatherArgs::rgb_to_patches(
        images,
        &mut out,
        ops::GatherPatchGeometry {
            views,
            image_size,
            patch_size,
            nhwc: layout == ImageLayout::Nhwc,
        },
    );
    ops::gather(ctx, args)
}

/// `rgb_u8_to_normalized_temporal_merged_patches_bf16`: normalize a `u8` RGB
/// batch by mean/std (float64 rescale, float32 normalization — the
/// Transformers boundary) and patchify with temporal repetition and spatial
/// merge reordering into
/// `[views*(size/patch)^2, 3*temporal*patch^2]` BF16.
#[allow(clippy::too_many_arguments)]
pub fn rgb_u8_to_normalized_temporal_merged_patches_bf16(
    ctx: &CudaContext,
    images: &CudaBuffer,
    patches: &Tensor,
    views: usize,
    image_size: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
    layout: ImageLayout,
    rescale_factor: f64,
    image_mean: [f32; 3],
    image_std: [f32; 3],
) -> Result<()> {
    use crate::ffi::abi::{preprocess as abi, status};
    if !rescale_factor.is_finite()
        || rescale_factor <= 0.0
        || image_mean.iter().any(|value| !value.is_finite())
        || image_std
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(Error::Other(
            "invalid temporal-merged BF16 image preprocessing parameters".into(),
        ));
    }
    if views == 0
        || image_size == 0
        || patch_size == 0
        || temporal_patch_size == 0
        || merge_size == 0
        || image_size % (patch_size * merge_size) != 0
    {
        return Err(Error::Other(
            "invalid temporal-merged BF16 image preprocessing dimensions".into(),
        ));
    }
    let grid_size = image_size / patch_size;
    let patch_rows = views * grid_size * grid_size;
    let patch_width = 3 * temporal_patch_size * patch_size * patch_size;
    let expected_bytes = views * 3 * image_size * image_size;
    if images.device() != ctx.device_id() || images.len() != expected_bytes {
        return Err(Error::Other(format!(
            "temporal-merged raw images must contain exactly {expected_bytes} bytes on CUDA {}, got {} bytes on CUDA {}",
            ctx.device_id(),
            images.len(),
            images.device()
        )));
    }
    if patches.dtype() != DType::BF16 || patches.shape().dims() != [patch_rows, patch_width] {
        return Err(Error::Other(format!(
            "temporal-merged patches must be BF16 [{patch_rows}, {patch_width}], got {} {:?}",
            patches.dtype(),
            patches.shape().dims()
        )));
    }
    let to_i32 = |value: usize, what: &str| {
        i32::try_from(value).map_err(|_| Error::Other(format!("{what} exceeds i32")))
    };
    let patches_buffer = CudaBuffer::from_tensor(patches).map_err(Error::Cuda)?;
    unsafe {
        status::check(abi::apxinf_preprocess_temporal_merged_patches_bf16(
            images.ptr(),
            patches_buffer.ptr(),
            to_i32(views, "views")?,
            to_i32(image_size, "image size")?,
            to_i32(patch_size, "patch size")?,
            to_i32(temporal_patch_size, "temporal patch size")?,
            to_i32(merge_size, "merge size")?,
            match layout {
                ImageLayout::Nhwc => 1,
                ImageLayout::Nchw => 0,
            },
            rescale_factor,
            image_mean[0],
            image_mean[1],
            image_mean[2],
            image_std[0],
            image_std[1],
            image_std[2],
            ctx.stream().handle(),
        ))
    }
}

/// Patch-grid extent of one packed NHWC still frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RgbRectFrame {
    pub grid_h: usize,
    pub grid_w: usize,
}

/// Convert packed, heterogeneously sized NHWC still frames into the BF16
/// temporal-2 / spatial-merge-2 patch stream used by compatible vision towers.
/// `lut` contains 256 BF16 bit patterns; its construction belongs to the model.
/// The safe boundary checks every byte/row range before launching any frame.
pub fn rgb_u8_to_temporal2_merge2_rect_bf16(
    ctx: &CudaContext,
    rgb: &CudaBuffer,
    patches: &Tensor,
    lut: &CudaBuffer,
    frames: &[RgbRectFrame],
) -> Result<()> {
    use crate::ffi::abi::vla_la as abi;
    use crate::ffi::raw::cuda_runtime as raw;
    if frames.is_empty()
        || rgb.device() != ctx.device_id()
        || lut.device() != ctx.device_id()
        || lut.len() != 512
        || patches.device() != apxinf_core::Device::Cuda(ctx.device_id())
        || patches.dtype() != DType::BF16
    {
        return Err(Error::Other(
            "invalid RGB temporal-merged device inputs".into(),
        ));
    }
    let mut bytes = 0usize;
    let mut rows = 0usize;
    for frame in frames {
        if frame.grid_h == 0
            || frame.grid_w == 0
            || frame.grid_h % 2 != 0
            || frame.grid_w % 2 != 0
            || i32::try_from(frame.grid_h).is_err()
            || i32::try_from(frame.grid_w).is_err()
        {
            return Err(Error::Other("invalid rectangular RGB patch grid".into()));
        }
        let frame_rows = frame
            .grid_h
            .checked_mul(frame.grid_w)
            .filter(|&n| n <= i32::MAX as usize)
            .ok_or_else(|| Error::Other("RGB patch grid overflow".into()))?;
        rows = rows
            .checked_add(frame_rows)
            .ok_or_else(|| Error::Other("RGB patch row overflow".into()))?;
        bytes = bytes
            .checked_add(
                frame_rows
                    .checked_mul(16 * 16 * 3)
                    .ok_or_else(|| Error::Other("RGB byte count overflow".into()))?,
            )
            .ok_or_else(|| Error::Other("RGB byte count overflow".into()))?;
    }
    let output_bytes = rows
        .checked_mul(1536)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| Error::Other("RGB BF16 patch byte count overflow".into()))?;
    if rgb.len() != bytes || patches.shape().dims() != [rows, 1536] {
        return Err(Error::Other(
            "RGB byte count or BF16 patch shape does not match grids".into(),
        ));
    }
    let output = CudaBuffer::from_tensor(patches).map_err(Error::Cuda)?;
    if output.len() < output_bytes {
        return Err(Error::Other(
            "rectangular RGB BF16 patch buffer is too small".into(),
        ));
    }
    let mut byte_offset = 0usize;
    let mut word_offset = 0usize;
    let output_ptr = output.ptr() as *mut u16;
    for frame in frames {
        let frame_rows = frame.grid_h * frame.grid_w;
        unsafe {
            raw::check_cuda(abi::apxinf_cn_rgb_u8_to_temporal2_merge2_rect_bf16(
                (rgb.ptr() as *const u8).add(byte_offset).cast(),
                output_ptr.add(word_offset).cast(),
                lut.ptr().cast(),
                frame.grid_h as i32,
                frame.grid_w as i32,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)?;
        }
        byte_offset += frame_rows * 16 * 16 * 3;
        word_offset += frame_rows * 1536;
    }
    Ok(())
}
