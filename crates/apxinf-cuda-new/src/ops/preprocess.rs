//! Image preprocessing and patchification operators.

use apxinf_core::{DType, Error, Result, Tensor};

use crate::{ops, CudaBuffer, CudaContext};

mod planning;

pub use planning::{
    concat7_cols, flow_update, fourier_features, vision_position, VisionPositionArgs,
};

/// Memory layout of a resized RGB `u8` image batch. Mirrors the legacy enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageLayout {
    Nhwc,
    Nchw,
}

/// Patch geometry interpreted by [`preprocess`]. Uniform images share one
/// geometry; rectangular frames carry one patch-grid extent per frame.
#[derive(Clone, Copy, Debug)]
pub enum PatchGeometry<'a> {
    Uniform {
        views: usize,
        image_height: usize,
        image_width: usize,
        patch_size: usize,
        temporal_patch_size: usize,
        merge_size: usize,
    },
    RectangularFrames {
        frames: &'a [RgbRectFrame],
        patch_size: usize,
        temporal_patch_size: usize,
        merge_size: usize,
    },
}

/// Mathematical RGB normalization policy. Output dtype is inferred from the
/// output tensor rather than encoded in the public function name.
#[derive(Clone, Copy, Debug)]
pub enum PatchNormalization {
    /// Map each byte from `[0, 255]` to `[-1, 1]`.
    UnitRange,
    /// Apply the Transformers rescale/mean/std boundary.
    MeanStd {
        rescale_factor: f64,
        image_mean: [f32; 3],
        image_std: [f32; 3],
    },
    /// Apply a caller-supplied 256-entry BF16 lookup table.
    LookupTable,
}

#[derive(Clone, Copy, Debug)]
pub struct PreprocessSpec<'a> {
    pub layout: ImageLayout,
    pub geometry: PatchGeometry<'a>,
    pub normalization: PatchNormalization,
}

pub struct PreprocessArgs<'a> {
    pub images: &'a CudaBuffer,
    pub patches: &'a Tensor,
    pub lookup_table: Option<&'a CudaBuffer>,
    pub spec: PreprocessSpec<'a>,
}

/// Convert packed RGB bytes to patch rows according to one semantic Spec.
/// Dispatch preserves the existing kernels and their exact arithmetic.
pub fn preprocess(ctx: &CudaContext, args: PreprocessArgs<'_>) -> Result<()> {
    match (args.spec.geometry, args.spec.normalization) {
        (
            PatchGeometry::Uniform {
                views,
                image_height,
                image_width,
                patch_size,
                temporal_patch_size: 1,
                merge_size: 1,
            },
            PatchNormalization::UnitRange,
        ) if image_height == image_width && args.lookup_table.is_none() => unit_range_patches(
            ctx,
            args.images,
            args.patches,
            views,
            image_height,
            patch_size,
            args.spec.layout,
        ),
        (
            PatchGeometry::Uniform {
                views,
                image_height,
                image_width,
                patch_size,
                temporal_patch_size,
                merge_size,
            },
            PatchNormalization::MeanStd {
                rescale_factor,
                image_mean,
                image_std,
            },
        ) if args.lookup_table.is_none() => normalized_temporal_merged_patches(
            ctx,
            args.images,
            args.patches,
            views,
            image_height,
            image_width,
            patch_size,
            temporal_patch_size,
            merge_size,
            args.spec.layout,
            rescale_factor,
            image_mean,
            image_std,
        ),
        (
            PatchGeometry::RectangularFrames {
                frames,
                patch_size: 16,
                temporal_patch_size: 2,
                merge_size: 2,
            },
            PatchNormalization::LookupTable,
        ) if args.spec.layout == ImageLayout::Nhwc => temporal_merged_rect_patches(
            ctx,
            args.images,
            args.patches,
            args.lookup_table.ok_or_else(|| {
                Error::Other("lookup-table preprocessing requires a lookup table".into())
            })?,
            frames,
        ),
        _ => Err(Error::Other(
            "unsupported RGB preprocessing Spec/binding combination".into(),
        )),
    }
}

/// Normalize a `u8` RGB batch to `[-1, 1]` and patchify into
/// `[views*(size/patch)^2, 3*patch^2]`. Output dtype is read from `patches`.
fn unit_range_patches(
    ctx: &CudaContext,
    images: &CudaBuffer,
    patches: &Tensor,
    views: usize,
    image_size: usize,
    patch_size: usize,
    layout: ImageLayout,
) -> Result<()> {
    if !matches!(patches.dtype(), DType::F32 | DType::F16 | DType::BF16) {
        return Err(Error::Other(
            "RGB preprocessing writes F32, F16, or BF16 patches".into(),
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

/// `rgb_u8_to_normalized_temporal_merged_patches_bf16`: normalize a `u8` RGB
/// batch by mean/std (float64 rescale, float32 normalization — the
/// Transformers boundary) and patchify with temporal repetition and spatial
/// merge reordering into
/// `[views*(height/patch)*(width/patch), 3*temporal*patch^2]` BF16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TemporalMergedDimensions {
    expected_bytes: usize,
    patch_rows: usize,
    patch_width: usize,
    views: i32,
    image_height: i32,
    image_width: i32,
    patch_size: i32,
    temporal_patch_size: i32,
    merge_size: i32,
}

fn temporal_merged_dimensions(
    views: usize,
    image_height: usize,
    image_width: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
) -> Result<TemporalMergedDimensions> {
    if views == 0
        || image_height == 0
        || image_width == 0
        || patch_size == 0
        || temporal_patch_size == 0
        || merge_size == 0
    {
        return Err(Error::Other(
            "invalid temporal-merged BF16 image preprocessing dimensions".into(),
        ));
    }
    let patch_merge = patch_size
        .checked_mul(merge_size)
        .ok_or_else(|| Error::Other("temporal-merged patch/merge size overflow".into()))?;
    if image_height % patch_merge != 0 || image_width % patch_merge != 0 {
        return Err(Error::Other(
            "invalid temporal-merged BF16 image preprocessing dimensions".into(),
        ));
    }
    let expected_bytes = views
        .checked_mul(3)
        .and_then(|value| value.checked_mul(image_height))
        .and_then(|value| value.checked_mul(image_width))
        .ok_or_else(|| Error::Other("temporal-merged raw image size overflow".into()))?;
    let rows_per_view = (image_height / patch_size)
        .checked_mul(image_width / patch_size)
        .ok_or_else(|| Error::Other("temporal-merged patch row count overflow".into()))?;
    let patch_rows = views
        .checked_mul(rows_per_view)
        .ok_or_else(|| Error::Other("temporal-merged patch row count overflow".into()))?;
    let patch_width = 3usize
        .checked_mul(temporal_patch_size)
        .and_then(|value| value.checked_mul(patch_size))
        .and_then(|value| value.checked_mul(patch_size))
        .ok_or_else(|| Error::Other("temporal-merged patch width overflow".into()))?;
    let to_i32 = |value: usize, what: &str| {
        i32::try_from(value).map_err(|_| Error::Other(format!("{what} exceeds i32")))
    };
    Ok(TemporalMergedDimensions {
        expected_bytes,
        patch_rows,
        patch_width,
        views: to_i32(views, "views")?,
        image_height: to_i32(image_height, "image height")?,
        image_width: to_i32(image_width, "image width")?,
        patch_size: to_i32(patch_size, "patch size")?,
        temporal_patch_size: to_i32(temporal_patch_size, "temporal patch size")?,
        merge_size: to_i32(merge_size, "merge size")?,
    })
}

#[allow(clippy::too_many_arguments)]
fn normalized_temporal_merged_patches(
    ctx: &CudaContext,
    images: &CudaBuffer,
    patches: &Tensor,
    views: usize,
    image_height: usize,
    image_width: usize,
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
    let dimensions = temporal_merged_dimensions(
        views,
        image_height,
        image_width,
        patch_size,
        temporal_patch_size,
        merge_size,
    )?;
    if images.device() != ctx.device_id() || images.len() != dimensions.expected_bytes {
        return Err(Error::Other(format!(
            "temporal-merged raw images must contain exactly {} bytes on CUDA {}, got {} bytes on CUDA {}",
            dimensions.expected_bytes,
            ctx.device_id(),
            images.len(),
            images.device()
        )));
    }
    if patches.dtype() != DType::BF16
        || patches.shape().dims() != [dimensions.patch_rows, dimensions.patch_width]
    {
        return Err(Error::Other(format!(
            "temporal-merged patches must be BF16 [{}, {}], got {} {:?}",
            dimensions.patch_rows,
            dimensions.patch_width,
            patches.dtype(),
            patches.shape().dims()
        )));
    }
    let patches_buffer = CudaBuffer::from_tensor(patches).map_err(Error::Cuda)?;
    unsafe {
        status::check(abi::apxinf_preprocess_temporal_merged_patches_bf16(
            images.ptr(),
            patches_buffer.ptr(),
            dimensions.views,
            dimensions.image_height,
            dimensions.image_width,
            dimensions.patch_size,
            dimensions.temporal_patch_size,
            dimensions.merge_size,
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

#[cfg(test)]
mod temporal_merged_tests {
    use super::temporal_merged_dimensions;

    #[test]
    fn rectangular_temporal_merged_dimensions_match_the_packed_contract() {
        let dimensions = temporal_merged_dimensions(2, 8, 12, 2, 2, 2).unwrap();
        assert_eq!(dimensions.expected_bytes, 576);
        assert_eq!(dimensions.patch_rows, 48);
        assert_eq!(dimensions.patch_width, 24);
        assert_eq!(dimensions.image_height, 8);
        assert_eq!(dimensions.image_width, 12);
    }

    #[test]
    fn temporal_merged_dimensions_reject_partial_merge_groups() {
        assert!(temporal_merged_dimensions(1, 8, 10, 2, 2, 2).is_err());
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
fn temporal_merged_rect_patches(
    ctx: &CudaContext,
    rgb: &CudaBuffer,
    patches: &Tensor,
    lut: &CudaBuffer,
    frames: &[RgbRectFrame],
) -> Result<()> {
    use crate::ffi::abi::preprocess as abi;
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
            raw::check_cuda(abi::apxinf_preprocess_rgb_u8_to_temporal2_merge2_rect_bf16(
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

// ── smolvla pixel shuffle ───────────────────────────────────────────────────
//
// Ported verbatim from the legacy FP16 preprocessing wrappers over the
// `family-specific FP16` launchers (native/adapters/preprocess/fixed_f16.cu).

/// SmolVLA's 4x4 pixel shuffle. Input dtype selects the unchanged BF16 or F16
/// launcher.
pub fn pixel_shuffle_4(
    ctx: &CudaContext,
    input: &Tensor,
    tokens_per_view: usize,
) -> Result<Tensor> {
    match input.dtype() {
        DType::BF16 => pixel_shuffle_4_bf16(ctx, input, tokens_per_view),
        DType::F16 => pixel_shuffle_4_f16(ctx, input, tokens_per_view),
        _ => Err(Error::Other(
            "pixel-shuffle-4 requires BF16 or F16 input".into(),
        )),
    }
}

/// `pixel_shuffle_4_bf16`: SmolVLA's 4x4 pixel shuffle over per-view token
/// grids, `[views*T, W] -> [views*T/16, W*16]`.
fn pixel_shuffle_4_bf16(
    ctx: &CudaContext,
    input: &Tensor,
    tokens_per_view: usize,
) -> Result<Tensor> {
    use super::validation::{check_cuda, gpu_ptr, make_gpu_tensor};
    use crate::ffi::abi::preprocess as abi;
    let input_shape = input.shape().dims();
    if input.dtype() != DType::BF16
        || input_shape.len() != 2
        || tokens_per_view == 0
        || input_shape[0] % tokens_per_view != 0
        || tokens_per_view % 16 != 0
    {
        return Err(Error::Other(
            "BF16 pixel-shuffle-4 input/token shape mismatch".into(),
        ));
    }
    let views = input_shape[0] / tokens_per_view;
    let output_rows = views * tokens_per_view / 16;
    let output_columns = input_shape[1]
        .checked_mul(16)
        .ok_or_else(|| Error::Other("BF16 pixel-shuffle output width overflow".into()))?;
    let output_elements = output_rows
        .checked_mul(output_columns)
        .ok_or_else(|| Error::Other("BF16 pixel-shuffle output size overflow".into()))?;
    let output =
        crate::workspace::output_buffer(ctx, output_elements * DType::BF16.size_in_bytes())?;
    unsafe {
        check_cuda(abi::apxinf_preprocess_pixel_shuffle_4_bf16(
            gpu_ptr(input)?,
            output.ptr(),
            views as i32,
            tokens_per_view as i32,
            input_shape[1] as i32,
            ctx.stream().handle(),
        ))?;
    }
    Ok(make_gpu_tensor(
        apxinf_core::Shape::new(vec![output_rows, output_columns]),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

/// `pixel_shuffle_4_f16`: the FP16 variant of the 4x4 pixel shuffle.
fn pixel_shuffle_4_f16(
    ctx: &CudaContext,
    input: &Tensor,
    tokens_per_view: usize,
) -> Result<Tensor> {
    use super::validation::{check_cuda, gpu_ptr, make_gpu_tensor};
    use crate::ffi::abi::preprocess as abi;
    let input_shape = input.shape().dims();
    if input.dtype() != DType::F16
        || input_shape.len() != 2
        || tokens_per_view == 0
        || input_shape[0] % tokens_per_view != 0
        || tokens_per_view % 16 != 0
    {
        return Err(Error::Other(
            "FP16 pixel-shuffle-4 input/token shape mismatch".into(),
        ));
    }
    let views = input_shape[0] / tokens_per_view;
    let output_rows = views * tokens_per_view / 16;
    let output_columns = input_shape[1]
        .checked_mul(16)
        .ok_or_else(|| Error::Other("FP16 pixel-shuffle output width overflow".into()))?;
    let output_elements = output_rows
        .checked_mul(output_columns)
        .ok_or_else(|| Error::Other("FP16 pixel-shuffle output size overflow".into()))?;
    let output =
        crate::workspace::output_buffer(ctx, output_elements * DType::F16.size_in_bytes())?;
    unsafe {
        check_cuda(abi::apxinf_preprocess_pixel_shuffle_4_f16(
            gpu_ptr(input)?,
            output.ptr(),
            views as i32,
            tokens_per_view as i32,
            input_shape[1] as i32,
            ctx.stream().handle(),
        ))?;
    }
    Ok(make_gpu_tensor(
        apxinf_core::Shape::new(vec![output_rows, output_columns]),
        DType::F16,
        ctx.device_id(),
        output,
    ))
}
