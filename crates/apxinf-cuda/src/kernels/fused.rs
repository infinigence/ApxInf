//! Fused multi-operator contracts used by static inference plans.

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use super::contracts::{
    bf16_output, f16_output, fp8_output, gpu_ptr, make_gpu_tensor, matrix_shape, matrix_tensor,
    optional_ptr,
};
use crate::context::CudaContext;
use crate::ffi;
use crate::tuning::{
    DeviceFingerprint, Epilogue, GemmLayout, GemmOp, GemmTuningKey, ScaleMode, TacticBackend,
    TuningDType,
};
pub struct ResidualNormTensors {
    pub hidden: Tensor,
    pub normalized: Tensor,
}
pub struct DynamicResidualNormTensors {
    pub hidden: Tensor,
    pub normalized: super::quantization::DynamicFp8Tensor,
}
use crate::workspace::{fp8_emulation_required, may_prepare_native_resources, output_buffer};

pub(crate) fn fp8_fused_tuning_key(
    ctx: &CudaContext,
    m: usize,
    n: usize,
    k: usize,
    output_dtype: TuningDType,
    epilogue: Epilogue,
) -> GemmTuningKey {
    GemmTuningKey {
        op: GemmOp::Fp8F16,
        device: DeviceFingerprint::from(ctx.caps()),
        m,
        n,
        k,
        activation_dtype: TuningDType::F8E4M3,
        weight_dtype: TuningDType::F8E4M3,
        output_dtype,
        layout: GemmLayout::RowMajor,
        scale_mode: ScaleMode::PerTensor,
        epilogue,
        workspace_limit: usize::MAX,
    }
}

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
            dual_geglu_interleaved: false,
            dual_geglu_auto_interleaved: None,
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
            dual_geglu_interleaved: false,
            dual_geglu_auto_interleaved: None,
        },
    )?;
    super::activation::bias_gelu_quant_f16_e4m3(ctx, &projection, bias, output_scale)
}

/// Explicit FP8 GEMM + bias + GELU-tanh + quantization for M=41, K=1536, N=6144.
///
/// The epilogue preserves the production BF16 projection and GELU rounding
/// boundaries before writing E4M3. Generic FP8 dispatch never selects it.
#[allow(clippy::too_many_arguments)]
pub fn try_fp8_bias_gelu_quant_e4m3_m41(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    activation_scale: f32,
    weight_scale: f32,
    output_scale: f32,
) -> Result<Option<Tensor>> {
    if ctx.caps().sm != 110
        || activation.shape().dims() != [41, 1536]
        || weight.shape().dims() != [1536, 6144]
        || bias.shape().dims() != [6144]
    {
        return Ok(None);
    }
    if activation.dtype() != DType::F8E4M3
        || weight.dtype() != DType::F8E4M3
        || bias.dtype() != DType::BF16
    {
        return Err(Error::Other(format!(
            "FP8 M41 bias/GELU/quantization fusion expects E4M3 activation/weight and BF16 bias, got {}, {}, and {}",
            activation.dtype(),
            weight.dtype(),
            bias.dtype()
        )));
    }
    if !activation_scale.is_finite()
        || activation_scale <= 0.0
        || !weight_scale.is_finite()
        || weight_scale <= 0.0
        || !output_scale.is_finite()
        || output_scale <= 0.0
    {
        return Err(Error::Other(format!(
            "invalid FP8 M41 bias/GELU/quantization fusion scales: activation={activation_scale}, weight={weight_scale}, output={output_scale}"
        )));
    }

    #[cfg(apxinf_cutlass_gemm)]
    {
        let output = fp8_output(ctx, 41, 6144)?;
        let status = unsafe {
            ffi::apxinf_static_cutlass_fp8_gemm_bias_gelu_quant_e4m3_m41(
                gpu_ptr(activation)?,
                gpu_ptr(weight)?,
                gpu_ptr(bias)?,
                output.ptr(),
                41,
                6144,
                1536,
                activation_scale * weight_scale,
                output_scale,
                ctx.stream().handle(),
            )
        };
        if status != 0 {
            return Err(Error::Cuda(format!(
                "FP8 M41 bias/GELU/quantization CUTLASS M64xN128xK128 launch failed ({status})"
            )));
        }
        return Ok(Some(make_gpu_tensor(
            Shape::new(vec![41, 6144]),
            DType::F8E4M3,
            ctx.device_id(),
            output,
        )));
    }

    #[cfg(not(apxinf_cutlass_gemm))]
    Ok(None)
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
            dual_geglu_interleaved: false,
            dual_geglu_auto_interleaved: None,
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

pub fn bias_residual_bf16_packed4(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: &Tensor,
    residual: &Tensor,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(projection, "packed4 bias residual")?;
    if projection.dtype() != DType::BF16
        || residual.dtype() != DType::BF16
        || residual.shape() != projection.shape()
        || bias.dtype() != DType::BF16
        || bias.shape().dims() != [cols]
        || cols % 4 != 0
    {
        return Err(Error::Other(
            "static inference packed4 BF16 bias residual has incompatible dtype or shape".into(),
        ));
    }
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_bias_residual_bf16_packed4(
            gpu_ptr(projection)?,
            gpu_ptr(bias)?,
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

pub fn bias_then_residual_bf16(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(projection, "bias then residual")?;
    if projection.dtype() != DType::BF16
        || residual.dtype() != DType::BF16
        || residual.shape() != projection.shape()
        || bias.is_some_and(|value| value.dtype() != DType::BF16 || value.shape().dims() != [cols])
    {
        return Err(Error::Other(
            "static inference BF16 bias-then-residual has incompatible dtype or shape".into(),
        ));
    }
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_bias_then_residual_bf16(
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

pub fn bias_then_residual_bf16_packed4(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(projection, "packed4 bias then residual")?;
    if projection.dtype() != DType::BF16
        || residual.dtype() != DType::BF16
        || residual.shape() != projection.shape()
        || cols % 4 != 0
        || bias.is_some_and(|value| value.dtype() != DType::BF16 || value.shape().dims() != [cols])
    {
        return Err(Error::Other(
            "static inference packed4 BF16 bias-then-residual has incompatible dtype or shape"
                .into(),
        ));
    }
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_bias_then_residual_bf16_packed4(
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

pub fn bias_residual_f16_bf16(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(projection, "mixed bias residual")?;
    if projection.dtype() != DType::F16
        || residual.dtype() != DType::BF16
        || residual.shape() != projection.shape()
        || bias.is_some_and(|value| value.dtype() != DType::BF16 || value.shape().dims() != [cols])
    {
        return Err(Error::Other(
            "static inference mixed bias residual has incompatible dtype or shape".into(),
        ));
    }
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_bias_residual_f16_bf16(
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

/// Fuse BF16 bias/residual addition, RMSNorm, and dynamic per-row E4M3
/// quantization for the next rowwise GEMM.
pub fn bias_residual_rms_quantize_rows_bf16_e4m3(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
    output_cols: usize,
) -> Result<DynamicResidualNormTensors> {
    let (rows, cols) = matrix_shape(projection, "dynamic residual RMSNorm quantization")?;
    if projection.dtype() != DType::BF16
        || residual.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || residual.shape() != projection.shape()
        || weight.shape().dims() != [cols]
        || bias.is_some_and(|value| value.dtype() != DType::BF16 || value.shape().dims() != [cols])
        || output_cols < cols
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Error::Other(
            "dynamic residual RMSNorm quantization has incompatible input".into(),
        ));
    }
    let hidden = bf16_output(ctx, rows, cols)?;
    let normalized = output_buffer(
        ctx,
        rows.checked_mul(output_cols)
            .ok_or_else(|| Error::Other("dynamic residual RMSNorm output size overflow".into()))?,
    )?;
    let scales = output_buffer(ctx, rows * DType::F32.size_in_bytes())?;
    unsafe {
        ffi::check_cuda(
            ffi::apxinf_dynamic_bias_residual_rms_norm_quantize_rows_bf16_e4m3(
                gpu_ptr(projection)?,
                optional_ptr(bias)?,
                gpu_ptr(residual)?,
                gpu_ptr(weight)?,
                hidden.ptr(),
                normalized.ptr(),
                scales.ptr(),
                rows as i32,
                cols as i32,
                output_cols as i32,
                eps,
                ctx.stream().handle(),
            ),
        )
        .map_err(Error::Cuda)?;
    }
    Ok(DynamicResidualNormTensors {
        hidden: matrix_tensor(ctx, rows, cols, hidden),
        normalized: super::quantization::DynamicFp8Tensor {
            values: make_gpu_tensor(
                Shape::new(vec![rows, output_cols]),
                DType::F8E4M3,
                ctx.device_id(),
                normalized,
            ),
            scales: make_gpu_tensor(Shape::new(vec![rows]), DType::F32, ctx.device_id(), scales),
        },
    })
}

pub fn bias_residual_rms_quant_f16_bf16_e4m3(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<ResidualNormTensors> {
    let (rows, cols) = matrix_shape(projection, "mixed residual RMSNorm quantization")?;
    if projection.dtype() != DType::F16
        || residual.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || residual.shape() != projection.shape()
        || weight.shape().dims() != [cols]
        || bias.is_some_and(|value| value.dtype() != DType::BF16 || value.shape().dims() != [cols])
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(Error::Other(
            "static inference mixed residual RMSNorm quantization has incompatible input".into(),
        ));
    }
    let hidden = bf16_output(ctx, rows, cols)?;
    let normalized = fp8_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(
            ffi::apxinf_static_bias_residual_rms_norm_quant_f16_bf16_e4m3(
                gpu_ptr(projection)?,
                optional_ptr(bias)?,
                gpu_ptr(residual)?,
                gpu_ptr(weight)?,
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
        hidden: matrix_tensor(ctx, rows, cols, hidden),
        normalized: make_gpu_tensor(
            Shape::new(vec![rows, cols]),
            DType::F8E4M3,
            ctx.device_id(),
            normalized,
        ),
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

/// Opt-in BF16 residual LayerNorm with an FP8 E4M3 normalized output.
///
/// The normalized value is rounded to BF16 before quantization, matching the
/// legacy `bias_residual_layer_bf16` then `quantize_bf16_e4m3` contract.  This
/// is deliberately a separate entrypoint so existing public dispatch remains
/// unchanged.
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
    let (rows, cols) = matrix_shape(projection, "quantized residual LayerNorm")?;
    if [projection, residual, norm_weight, norm_bias]
        .into_iter()
        .any(|tensor| tensor.dtype() != DType::BF16)
        || residual.shape() != projection.shape()
        || norm_weight.shape().dims() != [cols]
        || norm_bias.shape().dims() != [cols]
        || projection_bias
            .is_some_and(|value| value.dtype() != DType::BF16 || value.shape().dims() != [cols])
        || !eps.is_finite()
        || eps <= 0.0
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(Error::Other(
            "static inference quantized BF16 residual LayerNorm shape or scale mismatch".into(),
        ));
    }
    let hidden = bf16_output(ctx, rows, cols)?;
    let normalized = fp8_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_bias_residual_layer_norm_quant_bf16_e4m3(
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
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(ResidualNormTensors {
        hidden: matrix_tensor(ctx, rows, cols, hidden),
        normalized: make_gpu_tensor(
            Shape::new(vec![rows, cols]),
            DType::F8E4M3,
            ctx.device_id(),
            normalized,
        ),
    })
}

/// Explicit width-1536 bias + residual + adaptive LayerNorm composition.
/// Public/default normalization dispatch does not select this helper.
pub fn bias_then_residual_adaptive_layer_bf16_cached_1536(
    ctx: &CudaContext,
    projection: &Tensor,
    projection_bias: &Tensor,
    residual: &Tensor,
    modulation: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    let (rows, cols) = matrix_shape(projection, "cached bias residual adaptive LayerNorm")?;
    if (ctx.caps().sm == 110 && rows != 41)
        || cols != 1536
        || [projection, projection_bias, residual, modulation]
            .into_iter()
            .any(|tensor| tensor.dtype() != DType::BF16)
        || residual.shape() != projection.shape()
        || projection_bias.shape().dims() != [cols]
        || modulation.shape().dims() != [2 * cols]
    {
        return Err(Error::Other(
            "cached BF16 bias residual adaptive LayerNorm requires width 1536 (41 rows on SM110) and matching inputs"
                .into(),
        ));
    }
    let hidden = bf16_output(ctx, rows, cols)?;
    let normalized = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(
            ffi::apxinf_static_bias_then_residual_adaptive_layer_norm_bf16_cached_1536(
                gpu_ptr(projection)?,
                gpu_ptr(projection_bias)?,
                gpu_ptr(residual)?,
                gpu_ptr(modulation)?,
                hidden.ptr(),
                normalized.ptr(),
                rows as i32,
                cols as i32,
                eps,
                ctx.stream().handle(),
            ),
        )
        .map_err(Error::Cuda)?;
    }
    Ok(ResidualNormTensors {
        hidden: matrix_tensor(ctx, rows, cols, hidden),
        normalized: matrix_tensor(ctx, rows, cols, normalized),
    })
}

/// GR00T may opt into this exact-width composition when the caller needs the
/// legacy bias-rounding boundary before adding the residual. Public residual
/// and LayerNorm helpers do not dispatch here automatically.
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
    let (rows, cols) = matrix_shape(projection, "cached bias residual LayerNorm")?;
    if cols != 1536
        || [
            projection,
            projection_bias,
            residual,
            norm_weight,
            norm_bias,
        ]
        .into_iter()
        .any(|tensor| tensor.dtype() != DType::BF16)
        || residual.shape() != projection.shape()
        || projection_bias.shape().dims() != [cols]
        || norm_weight.shape().dims() != [cols]
        || norm_bias.shape().dims() != [cols]
    {
        return Err(Error::Other(
            "cached BF16 bias residual LayerNorm requires width 1536 and matching shapes".into(),
        ));
    }
    let hidden = bf16_output(ctx, rows, cols)?;
    let normalized = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(
            ffi::apxinf_static_bias_then_residual_layer_norm_bf16_cached_1536(
                gpu_ptr(projection)?,
                gpu_ptr(projection_bias)?,
                gpu_ptr(residual)?,
                gpu_ptr(norm_weight)?,
                gpu_ptr(norm_bias)?,
                hidden.ptr(),
                normalized.ptr(),
                rows as i32,
                cols as i32,
                eps,
                ctx.stream().handle(),
            ),
        )
        .map_err(Error::Cuda)?;
    }
    Ok(ResidualNormTensors {
        hidden: matrix_tensor(ctx, rows, cols, hidden),
        normalized: matrix_tensor(ctx, rows, cols, normalized),
    })
}

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
    let (rows, cols) = matrix_shape(projection, "cached residual LayerNorm")?;
    if cols != 1024
        || [projection, residual, norm_weight, norm_bias]
            .into_iter()
            .any(|tensor| tensor.dtype() != DType::BF16)
        || residual.shape() != projection.shape()
        || norm_weight.shape().dims() != [cols]
        || norm_bias.shape().dims() != [cols]
        || projection_bias
            .is_some_and(|value| value.dtype() != DType::BF16 || value.shape().dims() != [cols])
    {
        return Err(Error::Other(
            "cached BF16 residual LayerNorm requires width 1024 and matching shapes".into(),
        ));
    }
    let hidden = bf16_output(ctx, rows, cols)?;
    let normalized = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(
            ffi::apxinf_static_bias_residual_layer_norm_bf16_cached_1024(
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
            ),
        )
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
    let key = fp8_fused_tuning_key(ctx, m, n, k, TuningDType::F16, Epilogue::Bias);
    let plan = super::gemm::resolve_fused_fp8_plan(
        ctx,
        &key,
        DType::F16,
        1.0,
        || unsafe {
            ffi::check_cublas(ffi::apxinf_static_prepare_fp8_gemm_bias_f16(
                gpu_ptr(bias)?,
                m as i32,
                n as i32,
                k as i32,
            ))
            .map_err(Error::Cuda)
        },
        |candidate_output| unsafe {
            ffi::check_cublas(ffi::apxinf_static_fp8_gemm_bias_f16(
                gpu_ptr(activation)?,
                gpu_ptr(weight)?,
                gpu_ptr(bias)?,
                candidate_output.ptr(),
                m as i32,
                n as i32,
                k as i32,
                activation_scale * weight_scale,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)
        },
    )?;
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
            if plan.tactic.backend != TacticBackend::Vendor {
                ctx.gemm_plans().fallback(ctx, &key)?;
            }
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
        if plan.tactic.backend != TacticBackend::Vendor {
            ctx.gemm_plans().fallback(ctx, &key)?;
        }
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
    let key = fp8_fused_tuning_key(ctx, m, n, k, TuningDType::F8E4M3, Epilogue::BiasGelu);
    let plan = super::gemm::resolve_fused_fp8_plan(
        ctx,
        &key,
        DType::F8E4M3,
        output_scale,
        || unsafe {
            ffi::check_cublas(ffi::apxinf_static_prepare_fp8_gemm_bias_gelu_e4m3(
                gpu_ptr(bias)?,
                m as i32,
                n as i32,
                k as i32,
                output_scale,
            ))
            .map_err(Error::Cuda)
        },
        |candidate_output| unsafe {
            ffi::check_cublas(ffi::apxinf_static_fp8_gemm_bias_gelu_e4m3(
                gpu_ptr(activation)?,
                gpu_ptr(weight)?,
                gpu_ptr(bias)?,
                candidate_output.ptr(),
                m as i32,
                n as i32,
                k as i32,
                activation_scale * weight_scale,
                output_scale,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)
        },
    )?;
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
            if plan.tactic.backend != TacticBackend::Vendor {
                ctx.gemm_plans().fallback(ctx, &key)?;
            }
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
        if plan.tactic.backend != TacticBackend::Vendor {
            ctx.gemm_plans().fallback(ctx, &key)?;
        }
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
    let bias_pointer = match bias {
        Some(value) => gpu_ptr(value)?,
        None => std::ptr::null(),
    };
    let key = fp8_fused_tuning_key(ctx, m, n, k, TuningDType::F16, Epilogue::BiasResidual);
    let plan = super::gemm::resolve_fused_fp8_plan(
        ctx,
        &key,
        DType::F16,
        1.0,
        || unsafe {
            ffi::check_cublas(ffi::apxinf_static_prepare_fp8_gemm_bias_residual_f16(
                bias_pointer,
                m as i32,
                n as i32,
                k as i32,
            ))
            .map_err(Error::Cuda)
        },
        |candidate_output| unsafe {
            ffi::check_cublas(ffi::apxinf_static_fp8_gemm_bias_residual_f16(
                gpu_ptr(activation)?,
                gpu_ptr(weight)?,
                bias_pointer,
                gpu_ptr(residual)?,
                candidate_output.ptr(),
                m as i32,
                n as i32,
                k as i32,
                activation_scale * weight_scale,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)
        },
    )?;
    let output = output_buffer(ctx, m * n * DType::F16.size_in_bytes())?;
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
            if plan.tactic.backend != TacticBackend::Vendor {
                ctx.gemm_plans().fallback(ctx, &key)?;
            }
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
        if plan.tactic.backend != TacticBackend::Vendor {
            ctx.gemm_plans().fallback(ctx, &key)?;
        }
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

/// Unit-offset gated residual followed by weighted adaptive RMSNorm.
/// Preserves BF16 rounding after (1 + gate), gate multiplication, residual add,
/// weighted normalization, (1 + scale), scale multiplication and shift addition.
/// Returns the residual as well as normalized rows for the next sublayer.
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
    let (rows, cols) = matrix_shape(projection, "weighted AdaLN residual")?;
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
        let bytes = super::contracts::checked_bytes(
            DType::BF16,
            t.shape().dims(),
            "weighted AdaLN residual",
        )?;
        let buffer = crate::buffer::CudaBuffer::from_tensor(t).map_err(Error::Cuda)?;
        super::contracts::require_buffers(
            ctx,
            "weighted AdaLN residual",
            &[("input", &buffer, bytes)],
        )?;
    }
    let m = i32::try_from(rows).map_err(|_| Error::Other("AdaLN row count overflow".into()))?;
    let hidden = bf16_output(ctx, rows, cols)?;
    let normalized = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_adaln_gate_residual_rms_bf16(
            gpu_ptr(projection)?,
            gpu_ptr(residual)?,
            gpu_ptr(gate)?,
            gpu_ptr(weight)?,
            gpu_ptr(scale)?,
            gpu_ptr(shift)?,
            hidden.ptr(),
            normalized.ptr(),
            m,
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
