//! X-VLA port operators: image normalization, convolution im2col, depthwise
//! convolution, exact (erf) GELU, sigmoid, and padded row gathers.
//!
//! These are the operators the X-VLA DaViT tower needs that the model-neutral
//! kernel set does not already provide. Layouts follow the model's token-major
//! convention: activations are `[tokens, channels]` with `tokens = height *
//! width` row-major.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use super::contracts::{bf16_output, check_cuda, gpu_ptr, optional_ptr, require_buffers};
use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;

/// CLIPImageProcessor normalization for the X-VLA vision tower.
pub const XVLA_IMAGE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const XVLA_IMAGE_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Upload packed host bytes (e.g. RGB u8) as a device buffer.
pub fn upload_bytes(ctx: &CudaContext, bytes: &[u8]) -> Result<CudaBuffer> {
    if bytes.is_empty() {
        return Err(Error::Other(
            "X-VLA upload requires a non-empty buffer".into(),
        ));
    }
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).map_err(Error::Cuda)?;
    buffer.copy_from_host(bytes).map_err(Error::Cuda)?;
    Ok(buffer)
}

/// `rgb` holds `height * width * 3` HWC u8 bytes; output is BF16 `[H*W, 3]`
/// token-major with `(pixel/255 - mean) / std` applied per channel.
pub fn rgb_u8_to_normalized_tokens_bf16(
    ctx: &CudaContext,
    rgb: &CudaBuffer,
    height: usize,
    width: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<Tensor> {
    if height == 0 || width == 0 {
        return Err(Error::Other(
            "X-VLA image normalization needs a non-zero size".into(),
        ));
    }
    let pixels = height
        .checked_mul(width)
        .ok_or_else(|| Error::Other("X-VLA image size overflow".into()))?;
    require_buffers(
        ctx,
        "X-VLA image normalization",
        &[("rgb", rgb, pixels.checked_mul(3).unwrap_or(usize::MAX))],
    )?;
    if std.iter().any(|value| *value == 0.0) {
        return Err(Error::Other(
            "X-VLA image normalization std must be non-zero".into(),
        ));
    }
    let output = bf16_output(ctx, pixels, 3)?;
    unsafe {
        check_cuda(ffi::apxinf_xvla_rgb_u8_to_normalized_tokens_bf16(
            rgb.ptr(),
            output.ptr(),
            height as i32,
            width as i32,
            mean.as_ptr(),
            std.as_ptr(),
            ctx.stream().handle(),
        ))
    }?;
    Ok(output.into_tensor(Shape::new(vec![pixels, 3]), DType::BF16))
}

/// im2col for a dense convolution over a token-major `[H*W, C]` input.
///
/// Output is `[out_h * out_w, C * kh * kw]` with column index
/// `((c*kh + ky)*kw + kx)`; out-of-bounds taps read as zero.
#[allow(clippy::too_many_arguments)]
pub fn im2col_bf16(
    ctx: &CudaContext,
    input: &Tensor,
    height: usize,
    width: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Result<Tensor> {
    let dims = input.shape().dims();
    if dims.len() != 2 || dims[0] != height * width {
        return Err(Error::Other(format!(
            "X-VLA im2col expects a token-major [{}, C] input, got {:?}",
            height * width,
            dims
        )));
    }
    if input.dtype() != DType::BF16 {
        return Err(Error::Other(format!(
            "X-VLA im2col expects BF16 input, got {}",
            input.dtype()
        )));
    }
    if kh == 0 || kw == 0 || stride_h == 0 || stride_w == 0 || height == 0 || width == 0 {
        return Err(Error::Other(
            "X-VLA im2col received a zero-sized parameter".into(),
        ));
    }
    if height + 2 * pad_h < kh || width + 2 * pad_w < kw {
        return Err(Error::Other(
            "X-VLA im2col kernel exceeds the padded input".into(),
        ));
    }
    let channels = dims[1];
    let out_h = (height + 2 * pad_h - kh) / stride_h + 1;
    let out_w = (width + 2 * pad_w - kw) / stride_w + 1;
    let rows = out_h * out_w;
    let cols = channels * kh * kw;
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        check_cuda(ffi::apxinf_xvla_im2col_bf16(
            gpu_ptr(input)?,
            output.ptr(),
            channels as i32,
            height as i32,
            width as i32,
            kh as i32,
            kw as i32,
            stride_h as i32,
            stride_w as i32,
            pad_h as i32,
            pad_w as i32,
            ctx.stream().handle(),
        ))
    }?;
    Ok(output.into_tensor(Shape::new(vec![rows, cols]), DType::BF16))
}

/// Depthwise 3×3, stride 1, pad 1 over a token-major `[H*W, C]` tensor.
///
/// `weight` is BF16 `[C, 9]` (`ky * 3 + kx` tap order), `bias` optional `[C]`.
#[allow(clippy::too_many_arguments)]
pub fn dwconv3x3_bf16(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    height: usize,
    width: usize,
) -> Result<Tensor> {
    let dims = input.shape().dims();
    if dims.len() != 2 || dims[0] != height * width || height == 0 || width == 0 {
        return Err(Error::Other(format!(
            "X-VLA depthwise conv expects a token-major [{}, C] input, got {:?}",
            height * width,
            dims
        )));
    }
    let channels = dims[1];
    if input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || weight.shape().dims() != [channels, 9]
    {
        return Err(Error::Other(format!(
            "X-VLA depthwise conv expects BF16 input and a [{channels}, 9] weight, got {} {:?}",
            weight.dtype(),
            weight.shape().dims()
        )));
    }
    if let Some(bias) = bias {
        if bias.dtype() != DType::BF16 || bias.shape().dims() != [channels] {
            return Err(Error::Other(format!(
                "X-VLA depthwise conv bias must be BF16 [{channels}]"
            )));
        }
    }
    let output = bf16_output(ctx, dims[0], channels)?;
    unsafe {
        check_cuda(ffi::apxinf_xvla_dwconv3x3_bf16(
            gpu_ptr(input)?,
            gpu_ptr(weight)?,
            optional_ptr(bias)?,
            output.ptr(),
            channels as i32,
            height as i32,
            width as i32,
            ctx.stream().handle(),
        ))
    }?;
    Ok(output.into_tensor(input.shape().clone(), DType::BF16))
}

/// Exact (erf) GELU, shape-preserving, BF16 in/out with FP32 math inside.
pub fn gelu_erf_bf16(ctx: &CudaContext, input: &Tensor) -> Result<Tensor> {
    elementwise_activation(ctx, input, ffi::apxinf_xvla_gelu_erf_bf16, "erf GELU")
}

/// Sigmoid, shape-preserving, BF16 in/out with FP32 math inside.
pub fn sigmoid_bf16(ctx: &CudaContext, input: &Tensor) -> Result<Tensor> {
    elementwise_activation(ctx, input, ffi::apxinf_xvla_sigmoid_bf16, "sigmoid")
}

/// Pack `[action | proprio | time]` rows for the flow head, zeroing the
/// gripper columns of the action and proprio blocks (ee6d preprocess).
pub fn pack_action_tokens_bf16(
    ctx: &CudaContext,
    action: &Tensor,
    proprio: &Tensor,
    time: &Tensor,
    gripper: [usize; 2],
) -> Result<Tensor> {
    let action_dims = action.shape().dims();
    let proprio_dims = proprio.shape().dims();
    let time_dims = time.shape().dims();
    if action_dims.len() != 2 || proprio_dims.len() != 1 || time_dims.len() != 1 {
        return Err(Error::Other(format!(
            "X-VLA action packing expects action [rows, dim], proprio [dim] and time [dim_time], got {:?} {:?} {:?}",
            action_dims, proprio_dims, time_dims
        )));
    }
    let (rows, dim_action) = (action_dims[0], action_dims[1]);
    if proprio_dims[0] != dim_action
        || time_dims[0] == 0
        || rows == 0
        || dim_action == 0
        || gripper[0] >= dim_action
        || gripper[1] >= dim_action
    {
        return Err(Error::Other(
            "X-VLA action packing has incompatible dimensions".into(),
        ));
    }
    if [action, proprio, time]
        .into_iter()
        .any(|tensor| tensor.dtype() != DType::BF16)
    {
        return Err(Error::Other(
            "X-VLA action packing requires BF16 inputs".into(),
        ));
    }
    let columns = dim_action * 2 + time_dims[0];
    let output = bf16_output(ctx, rows, columns)?;
    unsafe {
        check_cuda(ffi::apxinf_xvla_pack_action_tokens_bf16(
            gpu_ptr(action)?,
            gpu_ptr(proprio)?,
            gpu_ptr(time)?,
            output.ptr(),
            rows as i32,
            dim_action as i32,
            time_dims[0] as i32,
            gripper[0] as i32,
            gripper[1] as i32,
            ctx.stream().handle(),
        ))
    }?;
    Ok(output.into_tensor(Shape::new(vec![rows, columns]), DType::BF16))
}

/// Sigmoid applied to two columns, all other columns copied unchanged.
pub fn sigmoid_columns_bf16(
    ctx: &CudaContext,
    input: &Tensor,
    indices: [usize; 2],
) -> Result<Tensor> {
    let dims = input.shape().dims();
    if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
        return Err(Error::Other(format!(
            "X-VLA column sigmoid expects a non-empty matrix, got {dims:?}"
        )));
    }
    if input.dtype() != DType::BF16 || indices[0] >= dims[1] || indices[1] >= dims[1] {
        return Err(Error::Other(
            "X-VLA column sigmoid requires BF16 and in-range indices".into(),
        ));
    }
    let output = bf16_output(ctx, dims[0], dims[1])?;
    unsafe {
        check_cuda(ffi::apxinf_xvla_sigmoid_columns_bf16(
            gpu_ptr(input)?,
            output.ptr(),
            dims[0] as i32,
            dims[1] as i32,
            indices[0] as i32,
            indices[1] as i32,
            ctx.stream().handle(),
        ))
    }?;
    Ok(output.into_tensor(input.shape().clone(), DType::BF16))
}

/// Row gather with a caller-prepared index table.
///
/// Unlike the generic row gather this permits repeated rows and tables that
/// are longer than the source matrix, which is exactly what padded window
/// partition needs (padded slots index a trailing all-zero row).
pub struct RowGatherPlan {
    indices: CudaBuffer,
    rows: usize,
    input_rows: usize,
}

impl RowGatherPlan {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn input_rows(&self) -> usize {
        self.input_rows
    }

    /// Replace the device index table for a new instruction.
    ///
    /// CUDA-graph plans keep the table address stable across replays, so the
    /// token rows are refreshed here (outside capture) instead of allocating a
    /// new table inside the graph.
    pub fn update_rows(&self, rows: &[u32]) -> Result<()> {
        if rows.len() != self.rows {
            return Err(Error::Other(format!(
                "X-VLA row gather expects {} indices, got {}",
                self.rows,
                rows.len()
            )));
        }
        if let Some(row) = rows.iter().copied().find(|row| *row as usize >= self.input_rows) {
            return Err(Error::Other(format!(
                "X-VLA row index {row} is outside 0..{}",
                self.input_rows
            )));
        }
        let bytes: Vec<u8> = rows.iter().flat_map(|row| row.to_ne_bytes()).collect();
        self.indices.copy_from_host(&bytes).map_err(Error::Cuda)
    }
}

/// Prepare a device index table for [`gather_rows_bf16`].
pub fn prepare_row_gather(
    ctx: &CudaContext,
    rows: &[u32],
    input_rows: usize,
) -> Result<RowGatherPlan> {
    if rows.is_empty() || input_rows == 0 {
        return Err(Error::Other(
            "X-VLA row gather needs indices and a non-empty source".into(),
        ));
    }
    if let Some(row) = rows.iter().copied().find(|row| *row as usize >= input_rows) {
        return Err(Error::Other(format!(
            "X-VLA row index {row} is outside 0..{input_rows}"
        )));
    }
    let bytes: Vec<u8> = rows.iter().flat_map(|row| row.to_ne_bytes()).collect();
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).map_err(Error::Cuda)?;
    buffer.copy_from_host(&bytes).map_err(Error::Cuda)?;
    Ok(RowGatherPlan {
        indices: buffer,
        rows: rows.len(),
        input_rows,
    })
}

/// Gather BF16 rows (`input` must have `plan.input_rows()` rows).
pub fn gather_rows_bf16(ctx: &CudaContext, input: &Tensor, plan: &RowGatherPlan) -> Result<Tensor> {
    let dims = input.shape().dims();
    if dims.len() != 2 || dims[0] != plan.input_rows || input.dtype() != DType::BF16 {
        return Err(Error::Other(format!(
            "X-VLA row gather expects a BF16 [{}, C] source, got {} {:?}",
            plan.input_rows,
            input.dtype(),
            dims
        )));
    }
    require_buffers(
        ctx,
        "X-VLA row gather",
        &[(
            "indices",
            &plan.indices,
            plan.rows * std::mem::size_of::<u32>(),
        )],
    )?;
    let output = bf16_output(ctx, plan.rows, dims[1])?;
    unsafe {
        check_cuda(ffi::apxinf_xvla_gather_rows_bf16(
            gpu_ptr(input)?,
            plan.indices.ptr(),
            output.ptr(),
            plan.rows as i64,
            dims[1] as i32,
            ctx.stream().handle(),
        ))
    }?;
    Ok(output.into_tensor(Shape::new(vec![plan.rows, dims[1]]), DType::BF16))
}

fn elementwise_activation(
    ctx: &CudaContext,
    input: &Tensor,
    launch: unsafe extern "C" fn(
        *const std::ffi::c_void,
        *mut std::ffi::c_void,
        i64,
        ffi::cudaStream_t,
    ) -> ffi::cudaError_t,
    operation: &str,
) -> Result<Tensor> {
    if input.dtype() != DType::BF16 {
        return Err(Error::Other(format!(
            "X-VLA {operation} expects BF16 input, got {}",
            input.dtype()
        )));
    }
    let count: usize = input.shape().dims().iter().product();
    if count == 0 {
        return Err(Error::Other(format!(
            "X-VLA {operation} received an empty tensor"
        )));
    }
    let output = crate::workspace::output_buffer(
        ctx,
        count
            .checked_mul(DType::BF16.size_in_bytes())
            .ok_or_else(|| Error::Other(format!("X-VLA {operation} byte size overflow")))?,
    )?;
    unsafe {
        check_cuda(launch(
            gpu_ptr(input)?,
            output.ptr(),
            count as i64,
            ctx.stream().handle(),
        ))
    }?;
    Ok(output.into_tensor(input.shape().clone(), DType::BF16))
}
