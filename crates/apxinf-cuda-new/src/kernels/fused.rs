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

/// `bias_then_residual_bf16`: add the optional bias, round to BF16, then add
/// the residual and round again. Distinct from `bias_residual_bf16`, which
/// rounds only the final sum.
pub fn bias_then_residual_bf16(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
) -> Result<Tensor> {
    use crate::ffi::abi::{gr00t as abi, status};
    use crate::CudaBuffer;
    use apxinf_core::Error;
    let dims = projection.shape().dims();
    if dims.len() != 2
        || projection.dtype() != DType::BF16
        || residual.dtype() != DType::BF16
        || residual.shape() != projection.shape()
        || bias.is_some_and(|value| value.dtype() != DType::BF16 || value.shape().dims() != [dims[1]])
    {
        return Err(Error::Other(
            "static inference BF16 bias-then-residual has incompatible dtype or shape".into(),
        ));
    }
    let (rows, cols) = (dims[0], dims[1]);
    let output = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::BF16)?;
    let to_i32 = |value: usize, what: &str| {
        i32::try_from(value).map_err(|_| Error::Other(format!("{what} exceeds i32")))
    };
    let projection_buffer = CudaBuffer::from_tensor(projection).map_err(Error::Cuda)?;
    let residual_buffer = CudaBuffer::from_tensor(residual).map_err(Error::Cuda)?;
    let bias_buffer = bias.map(CudaBuffer::from_tensor).transpose().map_err(Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    unsafe {
        status::check(abi::apxinf_gr00t_bias_then_residual_bf16(
            projection_buffer.ptr(),
            bias_buffer
                .as_ref()
                .map_or(std::ptr::null(), |buffer| buffer.ptr() as *const _),
            residual_buffer.ptr(),
            output_buffer.ptr(),
            to_i32(rows, "rows")?,
            to_i32(cols, "cols")?,
            ctx.stream().handle(),
        ))?;
    }
    Ok(output)
}

/// `bias_residual_bf16_packed4`: packed-4 variant reading quads through
/// aligned 8-byte accesses; requires a bias.
pub fn bias_residual_bf16_packed4(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: &Tensor,
    residual: &Tensor,
) -> Result<Tensor> {
    use crate::ffi::abi::{gr00t as abi, status};
    use crate::CudaBuffer;
    use apxinf_core::Error;
    let dims = projection.shape().dims();
    if dims.len() != 2
        || projection.dtype() != DType::BF16
        || bias.dtype() != DType::BF16
        || residual.dtype() != DType::BF16
        || residual.shape() != projection.shape()
        || bias.shape().dims() != [dims[1]]
    {
        return Err(Error::Other(
            "packed-4 BF16 bias-residual has incompatible dtype or shape".into(),
        ));
    }
    let (rows, cols) = (dims[0], dims[1]);
    let output = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::BF16)?;
    let to_i32 = |value: usize, what: &str| {
        i32::try_from(value).map_err(|_| Error::Other(format!("{what} exceeds i32")))
    };
    let projection_buffer = CudaBuffer::from_tensor(projection).map_err(Error::Cuda)?;
    let bias_buffer = CudaBuffer::from_tensor(bias).map_err(Error::Cuda)?;
    let residual_buffer = CudaBuffer::from_tensor(residual).map_err(Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    unsafe {
        status::check(abi::apxinf_gr00t_bias_residual_bf16_packed4(
            projection_buffer.ptr(),
            bias_buffer.ptr(),
            residual_buffer.ptr(),
            output_buffer.ptr(),
            to_i32(rows, "rows")?,
            to_i32(cols, "cols")?,
            ctx.stream().handle(),
        ))?;
    }
    Ok(output)
}

/// `bias_then_residual_bf16_packed4`: packed-4 companion of
/// [`bias_then_residual_bf16`]; the bias is optional.
pub fn bias_then_residual_bf16_packed4(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
) -> Result<Tensor> {
    use crate::ffi::abi::{gr00t as abi, status};
    use crate::CudaBuffer;
    use apxinf_core::Error;
    let dims = projection.shape().dims();
    if dims.len() != 2
        || projection.dtype() != DType::BF16
        || residual.dtype() != DType::BF16
        || residual.shape() != projection.shape()
        || bias.is_some_and(|value| value.dtype() != DType::BF16 || value.shape().dims() != [dims[1]])
    {
        return Err(Error::Other(
            "packed-4 BF16 bias-then-residual has incompatible dtype or shape".into(),
        ));
    }
    let (rows, cols) = (dims[0], dims[1]);
    let output = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::BF16)?;
    let to_i32 = |value: usize, what: &str| {
        i32::try_from(value).map_err(|_| Error::Other(format!("{what} exceeds i32")))
    };
    let projection_buffer = CudaBuffer::from_tensor(projection).map_err(Error::Cuda)?;
    let residual_buffer = CudaBuffer::from_tensor(residual).map_err(Error::Cuda)?;
    let bias_buffer = bias.map(CudaBuffer::from_tensor).transpose().map_err(Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    unsafe {
        status::check(abi::apxinf_gr00t_bias_then_residual_bf16_packed4(
            projection_buffer.ptr(),
            bias_buffer
                .as_ref()
                .map_or(std::ptr::null(), |buffer| buffer.ptr() as *const _),
            residual_buffer.ptr(),
            output_buffer.ptr(),
            to_i32(rows, "rows")?,
            to_i32(cols, "cols")?,
            ctx.stream().handle(),
        ))?;
    }
    Ok(output)
}

/// `bias_residual_layer_bf16_cached_1024`: fused residual + LayerNorm at the
/// fixed width 1024, with the row carried in registers across both reductions.
#[allow(clippy::too_many_arguments)]
pub fn bias_residual_layer_bf16_cached_1024(
    ctx: &CudaContext,
    projection: &Tensor,
    projection_bias: Option<&Tensor>,
    residual: &Tensor,
    norm_weight: &Tensor,
    norm_bias: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    residual_layer_cached(
        ctx,
        projection,
        projection_bias,
        residual,
        norm_weight,
        norm_bias,
        eps,
        1024,
    )
}

/// `bias_then_residual_layer_bf16_cached_1536`: fused bias-then-residual +
/// LayerNorm at the fixed width 1536. The projection bias is required.
#[allow(clippy::too_many_arguments)]
pub fn bias_then_residual_layer_bf16_cached_1536(
    ctx: &CudaContext,
    projection: &Tensor,
    projection_bias: &Tensor,
    residual: &Tensor,
    norm_weight: &Tensor,
    norm_bias: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    residual_layer_cached(
        ctx,
        projection,
        Some(projection_bias),
        residual,
        norm_weight,
        norm_bias,
        eps,
        1536,
    )
}

/// `bias_then_residual_adaptive_layer_bf16_cached_1536`: fused bias-then-
/// residual + adaptive LayerNorm at the fixed width 1536.
#[allow(clippy::too_many_arguments)]
pub fn bias_then_residual_adaptive_layer_bf16_cached_1536(
    ctx: &CudaContext,
    projection: &Tensor,
    projection_bias: &Tensor,
    residual: &Tensor,
    modulation: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    use crate::ffi::abi::{gr00t as abi, status};
    use crate::CudaBuffer;
    use apxinf_core::Error;
    let dims = projection.shape().dims();
    if dims.len() != 2
        || dims[1] != 1536
        || projection.dtype() != DType::BF16
        || projection_bias.dtype() != DType::BF16
        || residual.dtype() != DType::BF16
        || modulation.dtype() != DType::BF16
        || residual.shape() != projection.shape()
        || modulation.shape().dims() != [dims[1] * 2]
    {
        return Err(Error::Other(
            "cached-1536 adaptive residual LayerNorm has incompatible dtype or shape".into(),
        ));
    }
    let (rows, cols) = (dims[0], dims[1]);
    let shape = Shape::new(vec![rows, cols]);
    let hidden = ctx.allocate_output(shape.clone(), DType::BF16)?;
    let normalized = ctx.allocate_output(shape, DType::BF16)?;
    let to_i32 = |value: usize, what: &str| {
        i32::try_from(value).map_err(|_| Error::Other(format!("{what} exceeds i32")))
    };
    let projection_buffer = CudaBuffer::from_tensor(projection).map_err(Error::Cuda)?;
    let projection_bias_buffer = CudaBuffer::from_tensor(projection_bias).map_err(Error::Cuda)?;
    let residual_buffer = CudaBuffer::from_tensor(residual).map_err(Error::Cuda)?;
    let modulation_buffer = CudaBuffer::from_tensor(modulation).map_err(Error::Cuda)?;
    let hidden_buffer = CudaBuffer::from_tensor(&hidden).map_err(Error::Cuda)?;
    let normalized_buffer = CudaBuffer::from_tensor(&normalized).map_err(Error::Cuda)?;
    unsafe {
        status::check(
            abi::apxinf_gr00t_bias_then_residual_adaptive_layer_norm_bf16_cached_1536(
                projection_buffer.ptr(),
                projection_bias_buffer.ptr(),
                residual_buffer.ptr(),
                modulation_buffer.ptr(),
                hidden_buffer.ptr(),
                normalized_buffer.ptr(),
                to_i32(rows, "rows")?,
                to_i32(cols, "cols")?,
                eps,
                ctx.stream().handle(),
            ),
        )?;
    }
    Ok(ResidualNormTensors { hidden, normalized })
}

/// `bias_residual_layer_quant_bf16_e4m3`: fused residual + LayerNorm whose
/// `normalized` output is a calibrated E4M3 matrix.
#[allow(clippy::too_many_arguments)]
pub fn bias_residual_layer_quant_bf16_e4m3(
    ctx: &CudaContext,
    projection: &Tensor,
    projection_bias: Option<&Tensor>,
    residual: &Tensor,
    norm_weight: &Tensor,
    norm_bias: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<ResidualNormTensors> {
    use crate::ffi::abi::{gr00t as abi, status};
    use crate::CudaBuffer;
    use apxinf_core::Error;
    let dims = projection.shape().dims();
    if dims.len() != 2
        || projection.dtype() != DType::BF16
        || residual.dtype() != DType::BF16
        || norm_weight.dtype() != DType::BF16
        || norm_bias.dtype() != DType::BF16
        || residual.shape() != projection.shape()
        || norm_weight.shape().dims() != [dims[1]]
        || norm_bias.shape().dims() != [dims[1]]
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(Error::Other(
            "quantized residual LayerNorm has incompatible dtype, shape, or scale".into(),
        ));
    }
    let (rows, cols) = (dims[0], dims[1]);
    let hidden = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::BF16)?;
    let normalized = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::F8E4M3)?;
    let to_i32 = |value: usize, what: &str| {
        i32::try_from(value).map_err(|_| Error::Other(format!("{what} exceeds i32")))
    };
    let projection_buffer = CudaBuffer::from_tensor(projection).map_err(Error::Cuda)?;
    let residual_buffer = CudaBuffer::from_tensor(residual).map_err(Error::Cuda)?;
    let norm_weight_buffer = CudaBuffer::from_tensor(norm_weight).map_err(Error::Cuda)?;
    let norm_bias_buffer = CudaBuffer::from_tensor(norm_bias).map_err(Error::Cuda)?;
    let bias_buffer = projection_bias
        .map(CudaBuffer::from_tensor)
        .transpose()
        .map_err(Error::Cuda)?;
    let hidden_buffer = CudaBuffer::from_tensor(&hidden).map_err(Error::Cuda)?;
    let normalized_buffer = CudaBuffer::from_tensor(&normalized).map_err(Error::Cuda)?;
    unsafe {
        status::check(abi::apxinf_gr00t_bias_residual_layer_norm_quant_bf16_e4m3(
            projection_buffer.ptr(),
            bias_buffer
                .as_ref()
                .map_or(std::ptr::null(), |buffer| buffer.ptr() as *const _),
            residual_buffer.ptr(),
            norm_weight_buffer.ptr(),
            norm_bias_buffer.ptr(),
            hidden_buffer.ptr(),
            normalized_buffer.ptr(),
            to_i32(rows, "rows")?,
            to_i32(cols, "cols")?,
            eps,
            scale,
            ctx.stream().handle(),
        ))?;
    }
    Ok(ResidualNormTensors { hidden, normalized })
}

/// Shared driver for the two residual-then-LayerNorm fixed-width kernels.
#[allow(clippy::too_many_arguments)]
fn residual_layer_cached(
    ctx: &CudaContext,
    projection: &Tensor,
    projection_bias: Option<&Tensor>,
    residual: &Tensor,
    norm_weight: &Tensor,
    norm_bias: &Tensor,
    eps: f32,
    width: usize,
) -> Result<ResidualNormTensors> {
    use crate::ffi::abi::{gr00t as abi, status};
    use crate::CudaBuffer;
    use apxinf_core::Error;
    let dims = projection.shape().dims();
    if dims.len() != 2
        || dims[1] != width
        || projection.dtype() != DType::BF16
        || residual.dtype() != DType::BF16
        || norm_weight.dtype() != DType::BF16
        || norm_bias.dtype() != DType::BF16
        || residual.shape() != projection.shape()
        || norm_weight.shape().dims() != [dims[1]]
        || norm_bias.shape().dims() != [dims[1]]
    {
        return Err(Error::Other(format!(
            "cached-{width} residual LayerNorm has incompatible dtype or shape"
        )));
    }
    let (rows, cols) = (dims[0], dims[1]);
    let shape = Shape::new(vec![rows, cols]);
    let hidden = ctx.allocate_output(shape.clone(), DType::BF16)?;
    let normalized = ctx.allocate_output(shape, DType::BF16)?;
    let to_i32 = |value: usize, what: &str| {
        i32::try_from(value).map_err(|_| Error::Other(format!("{what} exceeds i32")))
    };
    let projection_buffer = CudaBuffer::from_tensor(projection).map_err(Error::Cuda)?;
    let residual_buffer = CudaBuffer::from_tensor(residual).map_err(Error::Cuda)?;
    let norm_weight_buffer = CudaBuffer::from_tensor(norm_weight).map_err(Error::Cuda)?;
    let norm_bias_buffer = CudaBuffer::from_tensor(norm_bias).map_err(Error::Cuda)?;
    let bias_buffer = projection_bias
        .map(CudaBuffer::from_tensor)
        .transpose()
        .map_err(Error::Cuda)?;
    let hidden_buffer = CudaBuffer::from_tensor(&hidden).map_err(Error::Cuda)?;
    let normalized_buffer = CudaBuffer::from_tensor(&normalized).map_err(Error::Cuda)?;
    unsafe {
        let code = if width == 1024 {
            abi::apxinf_gr00t_bias_residual_layer_norm_bf16_cached_1024(
                projection_buffer.ptr(),
                bias_buffer
                    .as_ref()
                    .map_or(std::ptr::null(), |buffer| buffer.ptr() as *const _),
                residual_buffer.ptr(),
                norm_weight_buffer.ptr(),
                norm_bias_buffer.ptr(),
                hidden_buffer.ptr(),
                normalized_buffer.ptr(),
                to_i32(rows, "rows")?,
                to_i32(cols, "cols")?,
                eps,
                ctx.stream().handle(),
            )
        } else {
            abi::apxinf_gr00t_bias_then_residual_layer_norm_bf16_cached_1536(
                projection_buffer.ptr(),
                bias_buffer
                    .as_ref()
                    .map_or(std::ptr::null(), |buffer| buffer.ptr() as *const _),
                residual_buffer.ptr(),
                norm_weight_buffer.ptr(),
                norm_bias_buffer.ptr(),
                hidden_buffer.ptr(),
                normalized_buffer.ptr(),
                to_i32(rows, "rows")?,
                to_i32(cols, "cols")?,
                eps,
                ctx.stream().handle(),
            )
        };
        status::check(code)?;
    }
    Ok(ResidualNormTensors { hidden, normalized })
}

/// `try_fp8_bias_gelu_quant_e4m3_m41`: fixed-shape sm110 FP8 fusion. Not
/// vendored into cuda-new; callers fall back to the generic route.
#[allow(clippy::too_many_arguments)]
pub fn try_fp8_bias_gelu_quant_e4m3_m41(
    _ctx: &CudaContext,
    _activation: &Tensor,
    _weight: &Tensor,
    _bias: &Tensor,
    _activation_scale: f32,
    _weight_scale: f32,
    _output_scale: f32,
) -> Result<Option<Tensor>> {
    Ok(None)
}
