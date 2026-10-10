//! Legacy `kernels::quantization` names over cuda-new quantization operators.

use apxinf_core::{DType, Result, Shape, Tensor};

use crate::{ops, CudaContext};

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
