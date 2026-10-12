//! Legacy `kernels::quantization` names over cuda-new quantization operators.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::{ops, CudaBuffer, CudaContext};

fn fixed_scale_e4m3(ctx: &CudaContext, input: &Tensor, scale: f32) -> Result<Tensor> {
    let mut out =
        ctx.allocate_output(Shape::new(input.shape().dims().to_vec()), DType::F8E4M3)?;
    let mut args =
        ops::QuantizationArgs::new(ops::QuantizationSemantic::FixedScaleE4m3, input, &mut out);
    args.scale = scale;
    ops::quantization(ctx, args)?;
    Ok(out)
}

/// `quantize_bf16_e4m3`: BF16 to E4M3 against one pre-calibrated scale.
pub fn quantize_bf16_e4m3(ctx: &CudaContext, input: &Tensor, scale: f32) -> Result<Tensor> {
    fixed_scale_e4m3(ctx, input, scale)
}

/// `quantize_f16_e4m3`: F16 to E4M3 against one pre-calibrated scale.
pub fn quantize_f16_e4m3(ctx: &CudaContext, input: &Tensor, scale: f32) -> Result<Tensor> {
    fixed_scale_e4m3(ctx, input, scale)
}

/// Row-quantized E4M3 tensor plus its per-row F32 scales. Mirrors the legacy
/// `DynamicFp8Tensor`.
pub struct DynamicFp8Tensor {
    pub values: Tensor,
    pub scales: Tensor,
}

/// `quantize_rows_bf16_e4m3`: quantize each BF16 row independently.
pub fn quantize_rows_bf16_e4m3(ctx: &CudaContext, input: &Tensor) -> Result<DynamicFp8Tensor> {
    let cols = input.shape().dims().get(1).copied().unwrap_or(0);
    quantize_rows_bf16_e4m3_padded(ctx, input, cols)
}

/// `quantize_rows_bf16_e4m3_padded`: rowwise quantization with zero-valued
/// FP8 padding columns appended up to `output_cols`.
pub fn quantize_rows_bf16_e4m3_padded(
    ctx: &CudaContext,
    input: &Tensor,
    output_cols: usize,
) -> Result<DynamicFp8Tensor> {
    let dims = input.shape().dims();
    if dims.len() != 2 {
        return Err(apxinf_core::Error::Other(
            "rowwise quantization requires a rank-2 input".into(),
        ));
    }
    let rows = dims[0];
    let mut values = ctx.allocate_output(Shape::new(vec![rows, output_cols]), DType::F8E4M3)?;
    let mut scales = ctx.allocate_output(Shape::new(vec![rows]), DType::F32)?;
    let mut args =
        ops::QuantizationArgs::new(ops::QuantizationSemantic::RowwiseE4m3, input, &mut values);
    args.scales = Some(&mut scales);
    ops::quantization(ctx, args)?;
    Ok(DynamicFp8Tensor { values, scales })
}

/// `slice_columns_bf16`: keep the leading `cols` columns of a BF16 matrix.
pub fn slice_columns_bf16(ctx: &CudaContext, input: &Tensor, cols: usize) -> Result<Tensor> {
    let rows = input.shape().dims().first().copied().unwrap_or(0);
    let mut out = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::BF16)?;
    let args =
        ops::QuantizationArgs::new(ops::QuantizationSemantic::SliceColumnsBf16, input, &mut out);
    ops::quantization(ctx, args)?;
    Ok(out)
}

/// `quantize_bf16_e4m3_packed8`: elementwise BF16->E4M3 with a static scale,
/// using the wide 8-element path when alignment allows.
pub fn quantize_bf16_e4m3_packed8(ctx: &CudaContext, input: &Tensor, scale: f32) -> Result<Tensor> {
    use crate::ffi::abi::{gr00t as abi, status};
    use crate::ffi::raw::cuda_runtime as raw;
    if input.dtype() != DType::BF16 {
        return Err(Error::DTypeMismatch {
            expected: DType::BF16,
            got: input.dtype(),
        });
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::Other(format!("invalid FP8 scale {scale}")));
    }
    let output = ctx.allocate_output(input.shape().clone(), DType::F8E4M3)?;
    let input_buffer = crate::CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let output_buffer = crate::CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(abi::apxinf_gr00t_quantize_bf16_e4m3_packed8(
            input_buffer.ptr(),
            output_buffer.ptr(),
            i64::try_from(input.numel()).map_err(|_| Error::Other("count exceeds i64".into()))?,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(output)
}

/// `concat_rows_quantize_bf16_e4m3`: stack two equal-width BF16 matrices by
/// rows and quantize the whole result to E4M3 with a shared scale.
pub fn concat_rows_quantize_bf16_e4m3(
    ctx: &CudaContext,
    first: &Tensor,
    second: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    use crate::ffi::abi::{gr00t as abi, status};
    use crate::ffi::raw::cuda_runtime as raw;
    let first_dims = first.shape().dims();
    let second_dims = second.shape().dims();
    if first_dims.len() != 2
        || second_dims.len() != 2
        || first.dtype() != DType::BF16
        || second.dtype() != DType::BF16
        || first_dims[1] != second_dims[1]
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(Error::Other(
            "BF16 row concatenation quantization requires equal-width matrices and a positive scale"
                .into(),
        ));
    }
    let rows = first_dims[0]
        .checked_add(second_dims[0])
        .ok_or_else(|| Error::Other("row concatenation size overflow".into()))?;
    let output = ctx.allocate_output(Shape::new(vec![rows, first_dims[1]]), DType::F8E4M3)?;
    let to_i32 = |value: usize, what: &str| {
        i32::try_from(value).map_err(|_| Error::Other(format!("{what} exceeds i32")))
    };
    let first_buffer = crate::CudaBuffer::from_tensor(first).map_err(Error::Cuda)?;
    let second_buffer = crate::CudaBuffer::from_tensor(second).map_err(Error::Cuda)?;
    let output_buffer = crate::CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(abi::apxinf_gr00t_concat_rows_quantize_bf16_e4m3(
            first_buffer.ptr(),
            second_buffer.ptr(),
            output_buffer.ptr(),
            to_i32(first_dims[0], "first rows")?,
            to_i32(second_dims[0], "second rows")?,
            to_i32(first_dims[1], "cols")?,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(output)
}

/// `slice_columns_f16`: keep the leading `output_cols` columns of an FP16
/// matrix. Ported from the legacy wrapper, which uses a strided 2D device
/// copy rather than a kernel.
pub fn slice_columns_f16(ctx: &CudaContext, input: &Tensor, output_cols: usize) -> Result<Tensor> {
    use super::contracts::make_gpu_tensor;
    if input.dtype() != DType::F16 {
        return Err(Error::DTypeMismatch {
            expected: DType::F16,
            got: input.dtype(),
        });
    }
    let shape = input.shape().dims();
    if shape.len() != 2 || shape[0] == 0 || output_cols == 0 || output_cols > shape[1] {
        return Err(Error::Other(format!(
            "FP16 column slice expects [rows,input_cols] with 0 < output_cols <= input_cols, got {shape:?} -> {output_cols}"
        )));
    }
    if output_cols == shape[1] {
        return Ok(input.clone());
    }
    let (rows, input_cols) = (shape[0], shape[1]);
    let output = crate::workspace::output_buffer(
        ctx,
        rows.checked_mul(output_cols)
            .and_then(|elements| elements.checked_mul(DType::F16.size_in_bytes()))
            .ok_or_else(|| Error::Other("FP16 column slice size overflow".into()))?,
    )?;
    crate::transfers::copy_tensor_2d_to_buffer(
        ctx,
        input,
        &output,
        0,
        output_cols * DType::F16.size_in_bytes(),
        input_cols * DType::F16.size_in_bytes(),
        output_cols * DType::F16.size_in_bytes(),
        rows,
    )?;
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, output_cols]),
        DType::F16,
        ctx.device_id(),
        output,
    ))
}
