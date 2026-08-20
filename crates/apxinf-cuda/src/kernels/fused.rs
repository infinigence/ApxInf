//! Fused multi-operator contracts used by static inference plans.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use super::contracts::{
    bf16_output, f16_output, fp8_output, gpu_ptr, make_gpu_tensor, matrix_shape, matrix_tensor,
    optional_ptr,
};
use crate::context::CudaContext;
use crate::ffi;
pub struct ResidualNormTensors {
    pub hidden: Tensor,
    pub normalized: Tensor,
}
use crate::workspace::{fp8_emulation_required, may_prepare_native_resources, output_buffer};

/// FP8 GEMM with an FP16 bias epilogue and a graph-safe fallback.
pub fn gemm_bias_fp8(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    activation_scale: f32,
    weight_scale: f32,
) -> Result<Tensor> {
    if let Some(output) = try_fp8_gemm_bias_f16(
        ctx,
        activation,
        weight,
        bias,
        activation_scale,
        weight_scale,
    )? {
        return Ok(output);
    }
    let projection = super::gemm::fp8(
        ctx,
        activation,
        activation_scale,
        super::gemm::Fp8WeightView {
            values_e4m3: weight,
            scale: weight_scale,
        },
    )?;
    super::elementwise::bias_f16(ctx, &projection, Some(bias))
}

/// FP8 GEMM with bias/GELU epilogue and a graph-safe fallback.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bias_gelu_fp8(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    activation_scale: f32,
    weight_scale: f32,
    output_scale: f32,
) -> Result<Tensor> {
    if let Some(output) = try_fp8_gemm_bias_gelu_e4m3(
        ctx,
        activation,
        weight,
        bias,
        activation_scale,
        weight_scale,
        output_scale,
    )? {
        return Ok(output);
    }
    let projection = super::gemm::fp8(
        ctx,
        activation,
        activation_scale,
        super::gemm::Fp8WeightView {
            values_e4m3: weight,
            scale: weight_scale,
        },
    )?;
    super::activation::bias_gelu_quant_f16_e4m3(ctx, &projection, bias, output_scale)
}

/// FP8 GEMM with bias/residual epilogue and a graph-safe fallback.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bias_residual_fp8(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    activation_scale: f32,
    weight_scale: f32,
) -> Result<Tensor> {
    if let Some(output) = try_fp8_gemm_bias_residual_f16(
        ctx,
        activation,
        weight,
        bias,
        residual,
        activation_scale,
        weight_scale,
    )? {
        return Ok(output);
    }
    let projection = super::gemm::fp8(
        ctx,
        activation,
        activation_scale,
        super::gemm::Fp8WeightView {
            values_e4m3: weight,
            scale: weight_scale,
        },
    )?;
    bias_residual_f16(ctx, &projection, bias, residual)
}
pub fn bias_residual_bf16(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(projection, "bias residual")?;
    if projection.dtype() != DType::BF16
        || residual.dtype() != DType::BF16
        || residual.shape() != projection.shape()
        || bias.is_some_and(|value| value.dtype() != DType::BF16 || value.shape().dims() != [cols])
    {
        return Err(Error::Other(
            "static inference BF16 bias residual has incompatible dtype or shape".into(),
        ));
    }
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_bias_residual_bf16(
            gpu_ptr(projection)?,
            optional_ptr(bias)?,
            gpu_ptr(residual)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(matrix_tensor(ctx, rows, cols, output))
}

pub fn bias_residual_rms_bf16(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    let (rows, cols) = matrix_shape(projection, "residual RMSNorm")?;
    if projection.dtype() != DType::BF16
        || residual.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || residual.shape() != projection.shape()
        || weight.shape().dims() != [cols]
        || bias.is_some_and(|value| value.dtype() != DType::BF16 || value.shape().dims() != [cols])
    {
        return Err(Error::Other(
            "static inference BF16 residual RMSNorm shape mismatch".into(),
        ));
    }
    let hidden = bf16_output(ctx, rows, cols)?;
    let normalized = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_bias_residual_rms_norm_bf16(
            gpu_ptr(projection)?,
            optional_ptr(bias)?,
            gpu_ptr(residual)?,
            gpu_ptr(weight)?,
            hidden.ptr(),
            normalized.ptr(),
            rows as i32,
            cols as i32,
            eps,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(ResidualNormTensors {
        hidden: matrix_tensor(ctx, rows, cols, hidden),
        normalized: matrix_tensor(ctx, rows, cols, normalized),
    })
}

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
    let (rows, cols) = matrix_shape(projection, "residual LayerNorm")?;
    if [projection, residual, norm_weight, norm_bias]
        .into_iter()
        .any(|tensor| tensor.dtype() != DType::BF16)
        || residual.shape() != projection.shape()
        || norm_weight.shape().dims() != [cols]
        || norm_bias.shape().dims() != [cols]
        || projection_bias
            .is_some_and(|value| value.dtype() != DType::BF16 || value.shape().dims() != [cols])
    {
        return Err(Error::Other(
            "static inference BF16 residual LayerNorm shape mismatch".into(),
        ));
    }
    let hidden = bf16_output(ctx, rows, cols)?;
    let normalized = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_bias_residual_layer_norm_bf16(
            gpu_ptr(projection)?,
            optional_ptr(projection_bias)?,
            gpu_ptr(residual)?,
            gpu_ptr(norm_weight)?,
            gpu_ptr(norm_bias)?,
            hidden.ptr(),
            normalized.ptr(),
            rows as i32,
            cols as i32,
            eps,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(ResidualNormTensors {
        hidden: matrix_tensor(ctx, rows, cols, hidden),
        normalized: matrix_tensor(ctx, rows, cols, normalized),
    })
}

pub fn adaptive_gate_residual_bf16(
    ctx: &CudaContext,
    projection: &Tensor,
    residual: &Tensor,
    style: &Tensor,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(projection, "Ada gate residual")?;
    if [projection, residual, style]
        .into_iter()
        .any(|tensor| tensor.dtype() != DType::BF16)
        || projection.shape() != residual.shape()
        || style.shape().dims() != [3 * cols]
    {
        return Err(Error::Other(
            "static inference BF16 Ada gate residual shape mismatch".into(),
        ));
    }
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_ada_gate_residual_bf16(
            gpu_ptr(projection)?,
            gpu_ptr(residual)?,
            gpu_ptr(style)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(matrix_tensor(ctx, rows, cols, output))
}

pub fn adaptive_gate_residual_rms_bf16(
    ctx: &CudaContext,
    projection: &Tensor,
    residual: &Tensor,
    gate_style: &Tensor,
    norm_style: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    let (rows, cols) = matrix_shape(projection, "Ada residual RMSNorm")?;
    if [projection, residual, gate_style, norm_style]
        .into_iter()
        .any(|tensor| tensor.dtype() != DType::BF16)
        || projection.shape() != residual.shape()
        || gate_style.shape().dims() != [3 * cols]
        || norm_style.shape().dims() != [3 * cols]
    {
        return Err(Error::Other(
            "static inference BF16 Ada residual RMSNorm shape mismatch".into(),
        ));
    }
    let hidden = bf16_output(ctx, rows, cols)?;
    let normalized = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_ada_gate_residual_rms_norm_bf16(
            gpu_ptr(projection)?,
            gpu_ptr(residual)?,
            gpu_ptr(gate_style)?,
            gpu_ptr(norm_style)?,
            hidden.ptr(),
            normalized.ptr(),
            rows as i32,
            cols as i32,
            eps,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(ResidualNormTensors {
        hidden: matrix_tensor(ctx, rows, cols, hidden),
        normalized: matrix_tensor(ctx, rows, cols, normalized),
    })
}
pub fn try_fp8_gemm_bias_f16(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    activation_scale: f32,
    weight_scale: f32,
) -> Result<Option<Tensor>> {
    if fp8_emulation_required(ctx)? {
        return Ok(None);
    }
    let a = activation.shape().dims();
    let b = weight.shape().dims();
    if activation.dtype() != DType::F8E4M3
        || weight.dtype() != DType::F8E4M3
        || a.len() != 2
        || b.len() != 2
        || a[1] != b[0]
        || bias.dtype() != DType::F16
        || bias.shape().dims() != [b.get(1).copied().unwrap_or(0)]
    {
        return Err(Error::Other(format!(
            "static inference fused bias GEMM shape mismatch: {a:?} @ {b:?}, bias {:?}",
            bias.shape().dims()
        )));
    }
    let (m, k, n) = (a[0], a[1], b[1]);
    let output = f16_output(ctx, m, n)?;
    if may_prepare_native_resources() {
        let status = unsafe {
            ffi::apxinf_static_prepare_fp8_gemm_bias_f16(
                gpu_ptr(bias)?,
                m as i32,
                n as i32,
                k as i32,
            )
        };
        if status != ffi::CUBLAS_STATUS_SUCCESS {
            return Ok(None);
        }
    }
    let status = unsafe {
        ffi::apxinf_static_fp8_gemm_bias_f16(
            gpu_ptr(activation)?,
            gpu_ptr(weight)?,
            gpu_ptr(bias)?,
            output.ptr(),
            m as i32,
            n as i32,
            k as i32,
            activation_scale * weight_scale,
            ctx.stream().handle(),
        )
    };
    if status != ffi::CUBLAS_STATUS_SUCCESS {
        return Ok(None);
    }
    Ok(Some(make_gpu_tensor(
        Shape::new(vec![m, n]),
        DType::F16,
        ctx.device_id(),
        output,
    )))
}

pub fn try_fp8_gemm_bias_gelu_e4m3(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    activation_scale: f32,
    weight_scale: f32,
    output_scale: f32,
) -> Result<Option<Tensor>> {
    if fp8_emulation_required(ctx)? {
        return Ok(None);
    }
    if activation.dtype() != DType::F8E4M3 || weight.dtype() != DType::F8E4M3 {
        return Err(Error::Other(format!(
            "static inference fused GELU GEMM expects E4M3 operands, got {} and {}",
            activation.dtype(),
            weight.dtype()
        )));
    }
    let a = activation.shape().dims();
    let b = weight.shape().dims();
    if a.len() != 2
        || b.len() != 2
        || a[1] != b[0]
        || bias.dtype() != DType::F16
        || bias.shape().dims() != [b[1]]
        || !output_scale.is_finite()
        || output_scale <= 0.0
    {
        return Err(Error::Other(format!(
            "static inference fused GELU GEMM shape mismatch: {a:?} @ {b:?}, bias {:?}, output scale {output_scale}",
            bias.shape().dims()
        )));
    }
    let (m, k, n) = (a[0], a[1], b[1]);
    let output = output_buffer(ctx, m * n)?;
    if may_prepare_native_resources() {
        let status = unsafe {
            ffi::apxinf_static_prepare_fp8_gemm_bias_gelu_e4m3(
                gpu_ptr(bias)?,
                m as i32,
                n as i32,
                k as i32,
                output_scale,
            )
        };
        if status != ffi::CUBLAS_STATUS_SUCCESS {
            return Ok(None);
        }
    }
    let status = unsafe {
        ffi::apxinf_static_fp8_gemm_bias_gelu_e4m3(
            gpu_ptr(activation)?,
            gpu_ptr(weight)?,
            gpu_ptr(bias)?,
            output.ptr(),
            m as i32,
            n as i32,
            k as i32,
            activation_scale * weight_scale,
            output_scale,
            ctx.stream().handle(),
        )
    };
    if status != ffi::CUBLAS_STATUS_SUCCESS {
        #[cfg(test)]
        eprintln!("fused FP8 GELU GEMM returned cuBLAS status {status}");
        return Ok(None);
    }
    Ok(Some(make_gpu_tensor(
        Shape::new(vec![m, n]),
        DType::F8E4M3,
        ctx.device_id(),
        output,
    )))
}

pub fn try_fp8_gemm_bias_residual_f16(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    activation_scale: f32,
    weight_scale: f32,
) -> Result<Option<Tensor>> {
    if fp8_emulation_required(ctx)? {
        return Ok(None);
    }
    if activation.dtype() != DType::F8E4M3 || weight.dtype() != DType::F8E4M3 {
        return Err(Error::Other(format!(
            "static inference fused residual GEMM expects E4M3 operands, got {} and {}",
            activation.dtype(),
            weight.dtype()
        )));
    }
    let a = activation.shape().dims();
    let b = weight.shape().dims();
    let bias_matches = bias.is_none_or(|value| {
        value.dtype() == DType::F16 && value.shape().dims() == [b.get(1).copied().unwrap_or(0)]
    });
    if a.len() != 2
        || b.len() != 2
        || a[1] != b[0]
        || residual.dtype() != DType::F16
        || residual.shape().dims() != [a[0], b[1]]
        || !bias_matches
    {
        return Err(Error::Other(format!(
            "static inference fused residual GEMM shape mismatch: {a:?} @ {b:?}, bias {:?}, residual {:?}",
            bias.map(|value| value.shape().dims()),
            residual.shape().dims()
        )));
    }
    let (m, k, n) = (a[0], a[1], b[1]);
    let output = output_buffer(ctx, m * n * DType::F16.size_in_bytes())?;
    let bias_pointer = match bias {
        Some(value) => gpu_ptr(value)?,
        None => std::ptr::null(),
    };
    if may_prepare_native_resources() {
        let status = unsafe {
            ffi::apxinf_static_prepare_fp8_gemm_bias_residual_f16(
                bias_pointer,
                m as i32,
                n as i32,
                k as i32,
            )
        };
        if status != ffi::CUBLAS_STATUS_SUCCESS {
            return Ok(None);
        }
    }
    let status = unsafe {
        ffi::apxinf_static_fp8_gemm_bias_residual_f16(
            gpu_ptr(activation)?,
            gpu_ptr(weight)?,
            bias_pointer,
            gpu_ptr(residual)?,
            output.ptr(),
            m as i32,
            n as i32,
            k as i32,
            activation_scale * weight_scale,
            ctx.stream().handle(),
        )
    };
    if status != ffi::CUBLAS_STATUS_SUCCESS {
        #[cfg(test)]
        eprintln!("fused FP8 residual GEMM returned cuBLAS status {status}");
        return Ok(None);
    }
    Ok(Some(make_gpu_tensor(
        Shape::new(vec![m, n]),
        DType::F16,
        ctx.device_id(),
        output,
    )))
}

pub fn bias_residual_rms_quant_f16_e4m3(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<ResidualNormTensors> {
    let (rows, cols) = matrix_shape(projection, "fused residual RMSNorm")?;
    if projection.dtype() != DType::F16
        || residual.dtype() != DType::F16
        || residual.shape() != projection.shape()
        || weight.dtype() != DType::F16
        || weight.shape().dims() != [cols]
        || bias.is_some_and(|x| x.dtype() != DType::F16 || x.shape().dims() != [cols])
    {
        return Err(Error::Other(
            "static inference fused residual RMSNorm shape mismatch".into(),
        ));
    }
    let hidden = f16_output(ctx, rows, cols)?;
    let normalized = fp8_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_bias_residual_rms_norm_quant_f16_e4m3(
            gpu_ptr(projection)?,
            bias.map(gpu_ptr)
                .transpose()?
                .unwrap_or(std::ptr::null_mut()),
            gpu_ptr(residual)?,
            gpu_ptr(weight)?,
            hidden.ptr(),
            normalized.ptr(),
            rows as i32,
            cols as i32,
            eps,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(ResidualNormTensors {
        hidden: make_gpu_tensor(
            Shape::new(vec![rows, cols]),
            DType::F16,
            ctx.device_id(),
            hidden,
        ),
        normalized: make_gpu_tensor(
            Shape::new(vec![rows, cols]),
            DType::F8E4M3,
            ctx.device_id(),
            normalized,
        ),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn bias_residual_layer_quant_f16_e4m3(
    ctx: &CudaContext,
    projection: &Tensor,
    projection_bias: Option<&Tensor>,
    residual: &Tensor,
    norm_weight: &Tensor,
    norm_bias: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<ResidualNormTensors> {
    let (rows, cols) = matrix_shape(projection, "fused residual LayerNorm")?;
    if projection.dtype() != DType::F16
        || residual.dtype() != DType::F16
        || residual.shape() != projection.shape()
        || norm_weight.dtype() != DType::F16
        || norm_bias.dtype() != DType::F16
        || norm_weight.shape().dims() != [cols]
        || norm_bias.shape().dims() != [cols]
        || projection_bias.is_some_and(|x| x.dtype() != DType::F16 || x.shape().dims() != [cols])
    {
        return Err(Error::Other(
            "static inference fused residual LayerNorm shape mismatch".into(),
        ));
    }
    let hidden = f16_output(ctx, rows, cols)?;
    let normalized = fp8_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_bias_residual_layer_norm_quant_f16_e4m3(
            gpu_ptr(projection)?,
            projection_bias
                .map(gpu_ptr)
                .transpose()?
                .unwrap_or(std::ptr::null_mut()),
            gpu_ptr(residual)?,
            gpu_ptr(norm_weight)?,
            gpu_ptr(norm_bias)?,
            hidden.ptr(),
            normalized.ptr(),
            rows as i32,
            cols as i32,
            eps,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(ResidualNormTensors {
        hidden: make_gpu_tensor(
            Shape::new(vec![rows, cols]),
            DType::F16,
            ctx.device_id(),
            hidden,
        ),
        normalized: make_gpu_tensor(
            Shape::new(vec![rows, cols]),
            DType::F8E4M3,
            ctx.device_id(),
            normalized,
        ),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn adaptive_gate_residual_rms_quant_f16_e4m3(
    ctx: &CudaContext,
    projection: &Tensor,
    residual: &Tensor,
    gate_style: &Tensor,
    norm_style: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<ResidualNormTensors> {
    let (rows, cols) = matrix_shape(projection, "fused Ada residual RMSNorm")?;
    if projection.dtype() != DType::F16
        || residual.dtype() != DType::F16
        || residual.shape() != projection.shape()
        || gate_style.dtype() != DType::F16
        || norm_style.dtype() != DType::F16
        || gate_style.shape().dims() != [3 * cols]
        || norm_style.shape().dims() != [3 * cols]
    {
        return Err(Error::Other(
            "static inference fused Ada residual RMSNorm shape mismatch".into(),
        ));
    }
    let hidden = f16_output(ctx, rows, cols)?;
    let normalized = fp8_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(
            ffi::apxinf_static_ada_gate_residual_rms_norm_quant_f16_e4m3(
                gpu_ptr(projection)?,
                gpu_ptr(residual)?,
                gpu_ptr(gate_style)?,
                gpu_ptr(norm_style)?,
                hidden.ptr(),
                normalized.ptr(),
                rows as i32,
                cols as i32,
                eps,
                scale,
                ctx.stream().handle(),
            ),
        )
        .map_err(Error::Cuda)?;
    }
    Ok(ResidualNormTensors {
        hidden: make_gpu_tensor(
            Shape::new(vec![rows, cols]),
            DType::F16,
            ctx.device_id(),
            hidden,
        ),
        normalized: make_gpu_tensor(
            Shape::new(vec![rows, cols]),
            DType::F8E4M3,
            ctx.device_id(),
            normalized,
        ),
    })
}

pub fn bias_residual_f16(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(projection, "bias residual")?;
    if projection.dtype() != DType::F16
        || residual.dtype() != DType::F16
        || residual.shape() != projection.shape()
        || bias.is_some_and(|x| x.dtype() != DType::F16 || x.shape().dims() != [cols])
    {
        return Err(Error::Other(
            "static inference bias residual has incompatible dtype or shape".into(),
        ));
    }
    let output = f16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_bias_residual_f16(
            gpu_ptr(projection)?,
            bias.map(gpu_ptr)
                .transpose()?
                .unwrap_or(std::ptr::null_mut()),
            gpu_ptr(residual)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::F16,
        ctx.device_id(),
        output,
    ))
}

pub fn adaptive_gate_residual_f16(
    ctx: &CudaContext,
    projection: &Tensor,
    residual: &Tensor,
    style: &Tensor,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(projection, "Ada gate residual")?;
    if projection.dtype() != DType::F16
        || residual.dtype() != DType::F16
        || style.dtype() != DType::F16
        || residual.shape() != projection.shape()
        || style.shape().dims() != [3 * cols]
    {
        return Err(Error::Other(
            "static inference Ada gate residual has incompatible dtype or shape".into(),
        ));
    }
    let output = f16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_ada_gate_residual_f16(
            gpu_ptr(projection)?,
            gpu_ptr(residual)?,
            gpu_ptr(style)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::F16,
        ctx.device_id(),
        output,
    ))
}
