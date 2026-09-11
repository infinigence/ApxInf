//! Input preprocessing operator contracts.

use apxinf_core::{DType, Error, Result, Tensor};

use super::contracts::gpu_ptr;
use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NormalizedTemporalMergedPreprocessDimensions {
    expected_bytes: usize,
    patch_rows: usize,
    patch_width: usize,
    views: i32,
    image_size: i32,
    patch_size: i32,
    temporal_patch_size: i32,
    merge_size: i32,
}

fn normalized_temporal_merged_preprocess_dimensions(
    views: usize,
    image_size: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
) -> Result<NormalizedTemporalMergedPreprocessDimensions> {
    if views == 0
        || image_size == 0
        || patch_size == 0
        || temporal_patch_size == 0
        || merge_size == 0
    {
        return Err(Error::Other(
            "invalid temporal-merged BF16 image preprocessing dimensions".into(),
        ));
    }

    let views_abi = i32::try_from(views)
        .map_err(|_| Error::Other("temporal-merged views exceed the CUDA ABI range".into()))?;
    let image_size_abi = i32::try_from(image_size).map_err(|_| {
        Error::Other("temporal-merged image size exceeds the CUDA ABI range".into())
    })?;
    let patch_size_abi = i32::try_from(patch_size).map_err(|_| {
        Error::Other("temporal-merged patch size exceeds the CUDA ABI range".into())
    })?;
    let temporal_patch_size_abi = i32::try_from(temporal_patch_size).map_err(|_| {
        Error::Other("temporal-merged temporal patch size exceeds the CUDA ABI range".into())
    })?;
    let merge_size_abi = i32::try_from(merge_size).map_err(|_| {
        Error::Other("temporal-merged merge size exceeds the CUDA ABI range".into())
    })?;

    let patch_merge = patch_size
        .checked_mul(merge_size)
        .ok_or_else(|| Error::Other("temporal-merged patch/merge size overflow".into()))?;
    i32::try_from(patch_merge).map_err(|_| {
        Error::Other("temporal-merged patch/merge size exceeds the CUDA kernel range".into())
    })?;
    if image_size % patch_merge != 0 {
        return Err(Error::Other(
            "invalid temporal-merged BF16 image preprocessing dimensions".into(),
        ));
    }

    let expected_bytes = views
        .checked_mul(3)
        .and_then(|value| value.checked_mul(image_size))
        .and_then(|value| value.checked_mul(image_size))
        .ok_or_else(|| Error::Other("temporal-merged raw image size overflow".into()))?;
    i64::try_from(expected_bytes).map_err(|_| {
        Error::Other("temporal-merged raw image size exceeds the CUDA index range".into())
    })?;

    let grid_size = image_size / patch_size;
    let rows_per_view = grid_size
        .checked_mul(grid_size)
        .ok_or_else(|| Error::Other("temporal-merged per-view patch row count overflow".into()))?;
    i32::try_from(rows_per_view).map_err(|_| {
        Error::Other(
            "temporal-merged per-view patch row count exceeds the CUDA kernel range".into(),
        )
    })?;
    let patch_rows = views
        .checked_mul(rows_per_view)
        .ok_or_else(|| Error::Other("temporal-merged patch row count overflow".into()))?;
    i32::try_from(patch_rows).map_err(|_| {
        Error::Other("temporal-merged patch row count exceeds the CUDA kernel range".into())
    })?;

    let patch_area = patch_size
        .checked_mul(patch_size)
        .ok_or_else(|| Error::Other("temporal-merged patch area overflow".into()))?;
    i32::try_from(patch_area).map_err(|_| {
        Error::Other("temporal-merged patch area exceeds the CUDA kernel range".into())
    })?;
    let patch_width = 3usize
        .checked_mul(temporal_patch_size)
        .and_then(|value| value.checked_mul(patch_area))
        .ok_or_else(|| Error::Other("temporal-merged patch width overflow".into()))?;
    i32::try_from(patch_width).map_err(|_| {
        Error::Other("temporal-merged patch width exceeds the CUDA kernel range".into())
    })?;

    let output_elements = patch_rows
        .checked_mul(patch_width)
        .ok_or_else(|| Error::Other("temporal-merged output element count overflow".into()))?;
    i64::try_from(output_elements).map_err(|_| {
        Error::Other("temporal-merged output element count exceeds the CUDA index range".into())
    })?;

    Ok(NormalizedTemporalMergedPreprocessDimensions {
        expected_bytes,
        patch_rows,
        patch_width,
        views: views_abi,
        image_size: image_size_abi,
        patch_size: patch_size_abi,
        temporal_patch_size: temporal_patch_size_abi,
        merge_size: merge_size_abi,
    })
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
/// Fused static inference image preprocessing into FP32 patch-major layout.
///
/// Identical addressing to [`rgb_u8_to_patches_bf16`], but the normalized
/// value stays in FP32. PaliGemma keeps the SigLIP patch embedding in FP32, so
/// quantizing the patch tensor to BF16 before the projection changes the
/// vision features enough to flip action tokens.
#[allow(clippy::too_many_arguments)]
pub fn rgb_u8_to_patches_f32(
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
            "invalid static inference FP32 image preprocessing shape".into(),
        ));
    }
    let expected_bytes = views * 3 * image_size * image_size;
    let side = image_size / patch_size;
    let expected_shape = [views * side * side, 3 * patch_size * patch_size];
    if images.device() != ctx.device_id()
        || images.len() != expected_bytes
        || patches.dtype() != DType::F32
        || patches.shape().dims() != expected_shape
    {
        return Err(Error::Other(format!(
            "static inference FP32 raw image/preprocessed patch mismatch: image bytes {}, patches {} {:?}",
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
        ffi::check_cuda(ffi::apxinf_static_rgb_u8_to_patches_f32(
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

/// temporal-merged still-image preprocessing for an already resized RGB image batch.
///
/// The output follows temporal-merged's temporal-patch and spatial-merge order and is
/// normalized channel-wise before the BF16 model-input rounding boundary.
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
    let dimensions = normalized_temporal_merged_preprocess_dimensions(
        views,
        image_size,
        patch_size,
        temporal_patch_size,
        merge_size,
    )?;
    let expected_shape = [dimensions.patch_rows, dimensions.patch_width];
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
        || patches.shape().dims() != expected_shape
        || patches.device() != apxinf_core::Device::Cuda(ctx.device_id())
    {
        return Err(Error::Other(format!(
            "temporal-merged patches must be BF16 {:?} on CUDA {}, got {} {:?} on {}",
            expected_shape,
            ctx.device_id(),
            patches.dtype(),
            patches.shape().dims(),
            patches.device()
        )));
    }
    unsafe {
        ffi::check_cuda(
            ffi::apxinf_rgb_u8_to_normalized_temporal_merged_patches_bf16(
                images.ptr(),
                gpu_ptr(patches)?,
                dimensions.views,
                dimensions.image_size,
                dimensions.patch_size,
                dimensions.temporal_patch_size,
                dimensions.merge_size,
                layout.kernel_value(),
                rescale_factor,
                image_mean[0],
                image_mean[1],
                image_mean[2],
                image_std[0],
                image_std[1],
                image_std[2],
                ctx.stream().handle(),
            ),
        )
        .map_err(Error::Cuda)
    }
}

#[cfg(test)]
mod tests {
    use super::normalized_temporal_merged_preprocess_dimensions;

    fn error_message<T: std::fmt::Debug>(result: apxinf_core::Result<T>) -> String {
        result.unwrap_err().to_string()
    }

    #[test]
    fn normalized_temporal_merged_dimensions_accept_representative_shape() {
        let dimensions =
            normalized_temporal_merged_preprocess_dimensions(2, 252, 14, 2, 2).unwrap();
        assert_eq!(dimensions.expected_bytes, 381_024);
        assert_eq!(dimensions.patch_rows, 648);
        assert_eq!(dimensions.patch_width, 1_176);
        assert_eq!(dimensions.views, 2);
        assert_eq!(dimensions.image_size, 252);
    }

    #[test]
    fn normalized_temporal_merged_dimensions_reject_int_kernel_intermediate_overflow() {
        let patch_merge = error_message(normalized_temporal_merged_preprocess_dimensions(
            1, 1, 50_000, 1, 50_000,
        ));
        assert!(patch_merge.contains("patch/merge size exceeds the CUDA kernel range"));

        let patch_area = error_message(normalized_temporal_merged_preprocess_dimensions(
            1, 46_341, 46_341, 1, 1,
        ));
        assert!(patch_area.contains("patch area exceeds the CUDA kernel range"));

        let rows = error_message(normalized_temporal_merged_preprocess_dimensions(
            1, 46_341, 1, 1, 1,
        ));
        assert!(rows.contains("per-view patch row count exceeds the CUDA kernel range"));

        let width = error_message(normalized_temporal_merged_preprocess_dimensions(
            1,
            1,
            1,
            715_827_883,
            1,
        ));
        assert!(width.contains("patch width exceeds the CUDA kernel range"));

        let total_rows = error_message(normalized_temporal_merged_preprocess_dimensions(
            2, 46_340, 1, 1, 1,
        ));
        assert!(total_rows.contains("patch row count exceeds the CUDA kernel range"));
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn normalized_temporal_merged_dimensions_reject_values_outside_ffi_abi() {
        let views = error_message(normalized_temporal_merged_preprocess_dimensions(
            i32::MAX as usize + 1,
            1,
            1,
            1,
            1,
        ));
        assert!(views.contains("views exceed the CUDA ABI range"));
    }
}
