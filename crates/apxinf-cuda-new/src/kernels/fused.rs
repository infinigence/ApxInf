//! Legacy `kernels::fused` names over cuda-new norm operators.

use apxinf_core::{DType, Result, Shape, Tensor};

use crate::{ops, CudaContext};

/// Output pair of a fused residual + norm: the updated residual stream and
/// its normalized view. Mirrors the legacy struct of the same name.
pub struct ResidualNormTensors {
    pub hidden: Tensor,
    pub normalized: Tensor,
}

/// `bias_residual_bf16`: `hidden = projection + bias? + residual`.
pub fn bias_residual_bf16(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
) -> Result<Tensor> {
    let mut hidden =
        ctx.allocate_output(Shape::new(projection.shape().dims().to_vec()), DType::BF16)?;
    ops::bias_residual(
        ctx,
        ops::BiasResidualArgs::new(projection, bias, residual, &mut hidden),
    )?;
    Ok(hidden)
}

/// `bias_residual_rms_bf16`: residual update plus an RMS-normalized view.
pub fn bias_residual_rms_bf16(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    let shape = Shape::new(projection.shape().dims().to_vec());
    let mut hidden = ctx.allocate_output(shape.clone(), DType::BF16)?;
    let mut normalized = ctx.allocate_output(shape, DType::BF16)?;
    ops::bias_residual_rms_norm(
        ctx,
        ops::BiasResidualRmsNormArgs::new(
            projection,
            bias,
            residual,
            weight,
            &mut hidden,
            &mut normalized,
            eps,
        ),
    )?;
    Ok(ResidualNormTensors { hidden, normalized })
}

/// `bias_residual_layer_bf16`: residual update plus a LayerNorm-normalized view.
#[allow(clippy::too_many_arguments)]
pub fn bias_residual_layer_bf16(
    ctx: &CudaContext,
    projection: &Tensor,
    projection_bias: Option<&Tensor>,
    residual: &Tensor,
    norm_weight: &Tensor,
    norm_bias: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    let shape = Shape::new(projection.shape().dims().to_vec());
    let mut hidden = ctx.allocate_output(shape.clone(), DType::BF16)?;
    let mut normalized = ctx.allocate_output(shape, DType::BF16)?;
    ops::bias_residual_layer_norm(
        ctx,
        ops::BiasResidualLayerNormArgs::new(
            projection,
            projection_bias,
            residual,
            norm_weight,
            norm_bias,
            &mut hidden,
            &mut normalized,
            eps,
        ),
    )?;
    Ok(ResidualNormTensors { hidden, normalized })
}

/// Residual update plus a rowwise-quantized normalized view. Mirrors the
/// legacy struct of the same name.
pub struct DynamicResidualNormTensors {
    pub hidden: Tensor,
    pub normalized: super::quantization::DynamicFp8Tensor,
}

/// `bias_residual_rms_quantize_rows_bf16_e4m3`: residual update, RMS norm,
/// and rowwise E4M3 quantization in one launch. The hidden write rounds to
/// BF16 before the square-sum, matching the legacy contract bit for bit.
pub fn bias_residual_rms_quantize_rows_bf16_e4m3(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
    output_cols: usize,
) -> Result<DynamicResidualNormTensors> {
    use crate::ffi::abi::{quant_fused as abi, status};
    use crate::CudaBuffer;
    let dims = projection.shape().dims();
    if dims.len() != 2 || projection.dtype() != DType::BF16 {
        return Err(apxinf_core::Error::Other(
            "fused residual RMS quantization requires a rank-2 BF16 projection".into(),
        ));
    }
    let (rows, cols) = (dims[0], dims[1]);
    if output_cols < cols {
        return Err(apxinf_core::Error::Other(
            "fused residual RMS quantization output narrower than input".into(),
        ));
    }
    let hidden = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::BF16)?;
    let values = ctx.allocate_output(Shape::new(vec![rows, output_cols]), DType::F8E4M3)?;
    let scales = ctx.allocate_output(Shape::new(vec![rows]), DType::F32)?;
    let to_i32 = |value: usize, what: &str| {
        i32::try_from(value).map_err(|_| apxinf_core::Error::Other(format!("{what} exceeds i32")))
    };
    let projection_buffer = CudaBuffer::from_tensor(projection).map_err(apxinf_core::Error::Cuda)?;
    let residual_buffer = CudaBuffer::from_tensor(residual).map_err(apxinf_core::Error::Cuda)?;
    let weight_buffer = CudaBuffer::from_tensor(weight).map_err(apxinf_core::Error::Cuda)?;
    let bias_buffer = bias
        .map(CudaBuffer::from_tensor)
        .transpose()
        .map_err(apxinf_core::Error::Cuda)?;
    let hidden_buffer = CudaBuffer::from_tensor(&hidden).map_err(apxinf_core::Error::Cuda)?;
    let values_buffer = CudaBuffer::from_tensor(&values).map_err(apxinf_core::Error::Cuda)?;
    let scales_buffer = CudaBuffer::from_tensor(&scales).map_err(apxinf_core::Error::Cuda)?;
    unsafe {
        status::check(abi::apxinf_quant_bias_residual_rms_norm_rows_bf16_e4m3(
            projection_buffer.ptr(),
            bias_buffer
                .as_ref()
                .map_or(std::ptr::null(), |buffer| buffer.ptr() as *const _),
            residual_buffer.ptr(),
            weight_buffer.ptr(),
            hidden_buffer.ptr(),
            values_buffer.ptr(),
            scales_buffer.ptr(),
            to_i32(rows, "rows")?,
            to_i32(cols, "cols")?,
            to_i32(output_cols, "output cols")?,
            eps,
            ctx.stream().handle(),
        ))?;
    }
    Ok(DynamicResidualNormTensors {
        hidden,
        normalized: super::quantization::DynamicFp8Tensor { values, scales },
    })
}

/// `adaln_gate_residual_rms_bf16`: weighted adaLN with unit-offset gate —
/// `hidden = residual + proj * bf16(1 + gate)`, normalized =
/// `rms(hidden) * weight * (1 + scale) + shift`. Direct port of the legacy
/// fused kernel.
#[allow(clippy::too_many_arguments)]
pub fn adaln_gate_residual_rms_bf16(
    ctx: &CudaContext,
    projection: &Tensor,
    residual: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    shift: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    use crate::ffi::abi::vla_la as abi;
    use crate::ffi::raw::cuda_runtime as raw;
    use crate::CudaBuffer;
    use apxinf_core::{Device, Error};
    let dims = projection.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(
            "weighted AdaLN residual expects a 2D projection".into(),
        ));
    }
    let (rows, cols) = (dims[0], dims[1]);
    if rows == 0
        || cols == 0
        || cols > 8192
        || !eps.is_finite()
        || eps <= 0.0
        || projection.shape() != residual.shape()
        || [gate, weight, scale, shift]
            .iter()
            .any(|t| t.shape().dims() != [cols])
    {
        return Err(Error::Other(
            "weighted AdaLN residual shape/epsilon unsupported".into(),
        ));
    }
    for t in [projection, residual, gate, weight, scale, shift] {
        if t.dtype() != DType::BF16 || t.device() != Device::Cuda(ctx.device_id()) {
            return Err(Error::Other(
                "weighted AdaLN residual requires BF16 on the context device".into(),
            ));
        }
    }
    let m = i32::try_from(rows).map_err(|_| Error::Other("AdaLN row count overflow".into()))?;
    let hidden = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::BF16)?;
    let normalized = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::BF16)?;
    let bufs: Vec<CudaBuffer> = [projection, residual, gate, weight, scale, shift]
        .iter()
        .map(|t| CudaBuffer::from_tensor(t).map_err(apxinf_core::Error::Cuda))
        .collect::<Result<_>>()?;
    let hidden_buffer = CudaBuffer::from_tensor(&hidden).map_err(apxinf_core::Error::Cuda)?;
    let normalized_buffer =
        CudaBuffer::from_tensor(&normalized).map_err(apxinf_core::Error::Cuda)?;
    unsafe {
        raw::check_cuda(abi::apxinf_cn_adaln_gate_residual_rms_bf16(
            bufs[0].ptr(),
            bufs[1].ptr(),
            bufs[2].ptr(),
            bufs[3].ptr(),
            bufs[4].ptr(),
            bufs[5].ptr(),
            hidden_buffer.ptr(),
            normalized_buffer.ptr(),
            m,
            cols as i32,
            eps,
            ctx.stream().handle(),
        ))
        .map_err(apxinf_core::Error::Cuda)?;
    }
    Ok(ResidualNormTensors { hidden, normalized })
}
