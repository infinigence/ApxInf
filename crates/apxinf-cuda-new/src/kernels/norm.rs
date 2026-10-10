//! Legacy `kernels::norm` names over cuda-new norm operators.

use apxinf_core::{DType, Result, Shape, Tensor};

use crate::{ops, CudaContext};

/// `rms_bf16`: RMS-normalize a `[rows, cols]` BF16 activation.
pub fn rms_bf16(ctx: &CudaContext, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let output = ctx.allocate_output(Shape::new(input.shape().dims().to_vec()), DType::BF16)?;
    ops::mlp::rms_norm(ctx, input, weight, &output, eps)?;
    Ok(output)
}

/// `layer_bf16`: LayerNorm with weight and bias.
pub fn layer_bf16(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(Shape::new(input.shape().dims().to_vec()), DType::BF16)?;
    ops::layer_norm(ctx, ops::LayerNormArgs::new(input, weight, bias, &mut output, eps))?;
    Ok(output)
}

/// `rms_quantize_rows_bf16_e4m3`: RMS-normalize then rowwise-quantize to
/// E4M3, zero-padding to `output_cols`. Returns values plus per-row scales.
pub fn rms_quantize_rows_bf16_e4m3(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
    output_cols: usize,
) -> Result<super::quantization::DynamicFp8Tensor> {
    use crate::ffi::abi::{quant_fused as abi, status};
    use crate::CudaBuffer;
    let dims = input.shape().dims();
    if dims.len() != 2 || input.dtype() != DType::BF16 {
        return Err(apxinf_core::Error::Other(
            "fused RMS quantization requires a rank-2 BF16 input".into(),
        ));
    }
    let (rows, cols) = (dims[0], dims[1]);
    if output_cols < cols {
        return Err(apxinf_core::Error::Other(
            "fused RMS quantization output narrower than input".into(),
        ));
    }
    let values = ctx.allocate_output(Shape::new(vec![rows, output_cols]), DType::F8E4M3)?;
    let scales = ctx.allocate_output(Shape::new(vec![rows]), DType::F32)?;
    let to_i32 = |value: usize, what: &str| {
        i32::try_from(value).map_err(|_| apxinf_core::Error::Other(format!("{what} exceeds i32")))
    };
    let input_buffer = CudaBuffer::from_tensor(input).map_err(apxinf_core::Error::Cuda)?;
    let weight_buffer = CudaBuffer::from_tensor(weight).map_err(apxinf_core::Error::Cuda)?;
    let values_buffer = CudaBuffer::from_tensor(&values).map_err(apxinf_core::Error::Cuda)?;
    let scales_buffer = CudaBuffer::from_tensor(&scales).map_err(apxinf_core::Error::Cuda)?;
    unsafe {
        status::check(abi::apxinf_quant_rms_norm_rows_bf16_e4m3(
            input_buffer.ptr(),
            weight_buffer.ptr(),
            values_buffer.ptr(),
            scales_buffer.ptr(),
            to_i32(rows, "rows")?,
            to_i32(cols, "cols")?,
            to_i32(output_cols, "output cols")?,
            eps,
            ctx.stream().handle(),
        ))?;
    }
    Ok(super::quantization::DynamicFp8Tensor { values, scales })
}
