//! Normalization operator contracts.

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use super::contracts::{
    bf16_output, check_cuda, checked_bytes, fp8_output, gpu_ptr, make_gpu_tensor, matrix_shape,
    matrix_tensor, require_buffers, require_finite, unsupported_dtype,
};
use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;
use crate::workspace::output_buffer;

/// RMS normalization into caller-owned storage.
#[allow(clippy::too_many_arguments)]
pub fn rms_into(
    ctx: &CudaContext,
    dtype: DType,
    input: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    cols: usize,
    rows: usize,
    eps: f32,
) -> Result<()> {
    require_finite("RMSNorm", &[eps])?;
    if eps <= 0.0 {
        return Err(Error::Other("RMSNorm epsilon must be positive".into()));
    }
    let matrix = checked_bytes(dtype, &[rows, cols], "RMSNorm")?;
    let weight_size = checked_bytes(dtype, &[cols], "RMSNorm")?;
    require_buffers(
        ctx,
        "RMSNorm",
        &[
            ("input", input, matrix),
            ("weight", weight, weight_size),
            ("output", output, matrix),
        ],
    )?;
    let status = unsafe {
        match dtype {
            DType::F32 => ffi::apxinf_rms_norm_f32(
                input.ptr(),
                weight.ptr(),
                output.ptr(),
                cols as u32,
                rows as u32,
                eps,
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_rms_norm_bf16(
                input.ptr(),
                weight.ptr(),
                output.ptr(),
                cols as u32,
                rows as u32,
                eps,
                ctx.stream().handle(),
            ),
            dtype => {
                return Err(Error::Other(format!(
                    "decode RMSNorm does not support {dtype}"
                )))
            }
        }
    };
    check_cuda(status)
}

/// Fused residual add and BF16 RMS normalization into caller-owned storage.
#[allow(clippy::too_many_arguments)]
pub fn residual_add_rms_bf16_into(
    ctx: &CudaContext,
    residual: &CudaBuffer,
    delta: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    cols: usize,
    rows: usize,
    eps: f32,
) -> Result<()> {
    require_finite("residual RMSNorm", &[eps])?;
    let matrix = checked_bytes(DType::BF16, &[rows, cols], "residual RMSNorm")?;
    let weight_size = checked_bytes(DType::BF16, &[cols], "residual RMSNorm")?;
    require_buffers(
        ctx,
        "residual RMSNorm",
        &[
            ("residual", residual, matrix),
            ("delta", delta, matrix),
            ("weight", weight, weight_size),
            ("output", output, matrix),
        ],
    )?;
    check_cuda(unsafe {
        ffi::apxinf_rms_norm_add_bf16(
            residual.ptr(),
            delta.ptr(),
            weight.ptr(),
            output.ptr(),
            cols as u32,
            rows as u32,
            eps,
            ctx.stream().handle(),
        )
    })
}

/// RMS normalization on CUDA. Dispatches on `input.dtype()`.
pub fn rms(ctx: &CudaContext, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let device_id = ctx.device_id();
    let dims = input.shape().dims();
    let rows = if dims.len() == 1 { 1 } else { dims[0] };
    let cols = if dims.len() == 1 {
        dims[0]
    } else {
        dims[dims.len() - 1]
    };

    let out_bytes = input.size_in_bytes();
    let out_buf = output_buffer(ctx, out_bytes)?;

    unsafe {
        let res = match input.dtype() {
            DType::F32 => ffi::apxinf_rms_norm_f32(
                gpu_ptr(input)?,
                gpu_ptr(weight)?,
                out_buf.ptr(),
                cols as u32,
                rows as u32,
                eps,
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_rms_norm_bf16(
                gpu_ptr(input)?,
                gpu_ptr(weight)?,
                out_buf.ptr(),
                cols as u32,
                rows as u32,
                eps,
                ctx.stream().handle(),
            ),
            dtype => return unsupported_dtype("rms_norm", dtype),
        };
        ffi::check_cuda(res).map_err(Error::Cuda)?;
    }

    Ok(make_gpu_tensor(
        input.shape().clone(),
        input.dtype(),
        device_id,
        out_buf,
    ))
}

/// LayerNorm (bf16 only). `input` shape `[rows, cols]` (or `[cols]` for
/// rows=1). Weight + bias are `[cols]`. Vision blocks have both.
pub fn layer(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    if input.dtype() != DType::BF16 {
        return Err(Error::Other("layer_norm: only BF16 supported".into()));
    }
    let device_id = ctx.device_id();
    let dims = input.shape().dims();
    let rows = if dims.len() == 1 { 1 } else { dims[0] };
    let cols = if dims.len() == 1 {
        dims[0]
    } else {
        dims[dims.len() - 1]
    };
    let out_buf = output_buffer(ctx, input.size_in_bytes())?;
    unsafe {
        let res = ffi::apxinf_layer_norm_bf16(
            gpu_ptr(input)?,
            gpu_ptr(weight)?,
            gpu_ptr(bias)?,
            out_buf.ptr(),
            cols as u32,
            rows as u32,
            eps,
            ctx.stream().handle(),
        );
        ffi::check_cuda(res).map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        input.shape().clone(),
        DType::BF16,
        device_id,
        out_buf,
    ))
}

/// LayerNorm followed by a row-broadcast adaptive affine transform.
///
/// `input` is `[rows, cols]`; `modulation` is `[2 * cols]` in
/// `[scale, shift]` order. This is the model-neutral primitive used by
/// diffusion transformer blocks such as GR00T's DiT.
pub fn adaptive_layer(
    ctx: &CudaContext,
    input: &Tensor,
    modulation: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "adaptive LayerNorm")?;
    require_finite("adaptive LayerNorm", &[eps])?;
    if rows == 0 || cols == 0 {
        return Err(Error::Other(
            "adaptive LayerNorm requires non-zero rows and columns".into(),
        ));
    }
    let modulation_columns = cols
        .checked_mul(2)
        .ok_or_else(|| Error::Other("adaptive LayerNorm modulation width overflow".into()))?;
    if eps <= 0.0
        || input.dtype() != DType::BF16
        || modulation.dtype() != DType::BF16
        || modulation.shape().dims() != [modulation_columns]
    {
        return Err(Error::Other(format!(
            "adaptive LayerNorm expects BF16 [rows,{cols}] and modulation [{}], got {} {:?} and {} {:?}",
            modulation_columns,
            input.dtype(),
            input.shape().dims(),
            modulation.dtype(),
            modulation.shape().dims()
        )));
    }
    let expected_device = Device::Cuda(ctx.device_id());
    for tensor in [input, modulation] {
        if tensor.device() != expected_device {
            return Err(Error::DeviceMismatch {
                expected: expected_device,
                got: tensor.device(),
            });
        }
    }
    let rows_u32 = u32::try_from(rows)
        .map_err(|_| Error::Other("adaptive LayerNorm row count exceeds CUDA ABI".into()))?;
    let cols_u32 = u32::try_from(cols)
        .map_err(|_| Error::Other("adaptive LayerNorm width exceeds CUDA ABI".into()))?;
    let output = bf16_output(ctx, rows, cols)?;
    check_cuda(unsafe {
        ffi::apxinf_adaptive_layer_norm_bf16(
            gpu_ptr(input)?,
            gpu_ptr(modulation)?,
            output.ptr(),
            rows_u32,
            cols_u32,
            eps,
            ctx.stream().handle(),
        )
    })?;
    Ok(matrix_tensor(ctx, rows, cols, output))
}

/// Adaptive LayerNorm producing both the exact BF16 result and its calibrated
/// E4M3 representation. The quantized output matches applying the standalone
/// BF16-to-E4M3 quantizer to the returned BF16 tensor.
pub fn adaptive_layer_quant_bf16_e4m3(
    ctx: &CudaContext,
    input: &Tensor,
    modulation: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<(Tensor, Tensor)> {
    let (rows, cols) = matrix_shape(input, "adaptive LayerNorm quantization")?;
    require_finite("adaptive LayerNorm quantization", &[eps, scale])?;
    let modulation_columns = cols
        .checked_mul(2)
        .ok_or_else(|| Error::Other("adaptive LayerNorm modulation width overflow".into()))?;
    if rows == 0
        || cols == 0
        || eps <= 0.0
        || scale <= 0.0
        || input.dtype() != DType::BF16
        || modulation.dtype() != DType::BF16
        || modulation.shape().dims() != [modulation_columns]
    {
        return Err(Error::Other(
            "adaptive LayerNorm quantization expects aligned BF16 input/modulation and positive epsilon/scale".into(),
        ));
    }
    let expected_device = Device::Cuda(ctx.device_id());
    if input.device() != expected_device || modulation.device() != expected_device {
        return Err(Error::DeviceMismatch {
            expected: expected_device,
            got: input.device(),
        });
    }
    let rows = u32::try_from(rows)
        .map_err(|_| Error::Other("adaptive LayerNorm row count exceeds CUDA ABI".into()))?;
    let cols = u32::try_from(cols)
        .map_err(|_| Error::Other("adaptive LayerNorm width exceeds CUDA ABI".into()))?;
    let bf16 = bf16_output(ctx, rows as usize, cols as usize)?;
    let fp8 = fp8_output(ctx, rows as usize, cols as usize)?;
    check_cuda(unsafe {
        ffi::apxinf_adaptive_layer_norm_quant_bf16_e4m3(
            gpu_ptr(input)?,
            gpu_ptr(modulation)?,
            bf16.ptr(),
            fp8.ptr(),
            rows,
            cols,
            eps,
            scale,
            ctx.stream().handle(),
        )
    })?;
    let shape = Shape::new(vec![rows as usize, cols as usize]);
    Ok((
        make_gpu_tensor(shape.clone(), DType::BF16, ctx.device_id(), bf16),
        make_gpu_tensor(shape, DType::F8E4M3, ctx.device_id(), fp8),
    ))
}
pub fn rms_bf16(ctx: &CudaContext, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "RMSNorm")?;
    if input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || weight.shape().dims() != [cols]
    {
        return Err(Error::Other(
            "static inference BF16 RMSNorm shape mismatch".into(),
        ));
    }
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_rms_norm_bf16(
            gpu_ptr(input)?,
            gpu_ptr(weight)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            eps,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(matrix_tensor(ctx, rows, cols, output))
}

/// Fuse BF16 RMSNorm with dynamic per-row E4M3 quantization.
pub fn rms_quantize_rows_bf16_e4m3(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
    output_cols: usize,
) -> Result<super::quantization::DynamicFp8Tensor> {
    let (rows, cols) = matrix_shape(input, "dynamic RMSNorm quantization")?;
    if input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || weight.shape().dims() != [cols]
        || output_cols < cols
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Error::Other(
            "dynamic RMSNorm quantization has incompatible input".into(),
        ));
    }
    let values = output_buffer(
        ctx,
        rows.checked_mul(output_cols)
            .ok_or_else(|| Error::Other("dynamic RMSNorm output size overflow".into()))?,
    )?;
    let scales = output_buffer(ctx, rows * DType::F32.size_in_bytes())?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_dynamic_rms_norm_quantize_rows_bf16_e4m3(
            gpu_ptr(input)?,
            gpu_ptr(weight)?,
            values.ptr(),
            scales.ptr(),
            rows as i32,
            cols as i32,
            output_cols as i32,
            eps,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(super::quantization::DynamicFp8Tensor {
        values: make_gpu_tensor(
            Shape::new(vec![rows, output_cols]),
            DType::F8E4M3,
            ctx.device_id(),
            values,
        ),
        scales: make_gpu_tensor(Shape::new(vec![rows]), DType::F32, ctx.device_id(), scales),
    })
}

pub fn layer_bf16(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "LayerNorm")?;
    if [input, weight, bias]
        .into_iter()
        .any(|tensor| tensor.dtype() != DType::BF16)
        || weight.shape().dims() != [cols]
        || bias.shape().dims() != [cols]
    {
        return Err(Error::Other(
            "static inference BF16 LayerNorm shape mismatch".into(),
        ));
    }
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_layer_norm_bf16(
            gpu_ptr(input)?,
            gpu_ptr(weight)?,
            gpu_ptr(bias)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            eps,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(matrix_tensor(ctx, rows, cols, output))
}

pub fn adaptive_rms_bf16(
    ctx: &CudaContext,
    input: &Tensor,
    style: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "AdaRMSNorm")?;
    if input.dtype() != DType::BF16
        || style.dtype() != DType::BF16
        || style.shape().dims() != [3 * cols]
    {
        return Err(Error::Other(
            "static inference BF16 AdaRMSNorm shape mismatch".into(),
        ));
    }
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_ada_rms_norm_bf16(
            gpu_ptr(input)?,
            gpu_ptr(style)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            eps,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(matrix_tensor(ctx, rows, cols, output))
}
pub fn rms_quant_f16_e4m3(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "RMSNorm")?;
    if input.dtype() != DType::F16
        || weight.dtype() != DType::F16
        || weight.shape().dims() != [cols]
    {
        return Err(Error::Other(
            "static inference RMSNorm expects FP16 [rows,cols] and FP16 [cols] scale".into(),
        ));
    }
    let output = fp8_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_rms_norm_quant_f16_e4m3(
            gpu_ptr(input)?,
            gpu_ptr(weight)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            eps,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::F8E4M3,
        ctx.device_id(),
        output,
    ))
}

pub fn rms_quant_bf16_e4m3(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "BF16 RMSNorm quantization")?;
    if input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || weight.shape().dims() != [cols]
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(Error::Other(
            "static inference BF16 RMSNorm quantization has incompatible dtype, shape, or scale"
                .into(),
        ));
    }
    let output = fp8_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_rms_norm_quant_bf16_e4m3(
            gpu_ptr(input)?,
            gpu_ptr(weight)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            eps,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::F8E4M3,
        ctx.device_id(),
        output,
    ))
}

pub fn layer_quant_f16_e4m3(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "LayerNorm")?;
    if input.dtype() != DType::F16
        || weight.dtype() != DType::F16
        || bias.dtype() != DType::F16
        || weight.shape().dims() != [cols]
        || bias.shape().dims() != [cols]
    {
        return Err(Error::Other(
            "static inference LayerNorm expects FP16 [rows,cols] and FP16 [cols] affine tensors"
                .into(),
        ));
    }
    let output = fp8_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_layer_norm_quant_f16_e4m3(
            gpu_ptr(input)?,
            gpu_ptr(weight)?,
            gpu_ptr(bias)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            eps,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::F8E4M3,
        ctx.device_id(),
        output,
    ))
}

pub fn layer_quant_bf16_e4m3(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "LayerNorm")?;
    if input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || bias.dtype() != DType::BF16
        || weight.shape().dims() != [cols]
        || bias.shape().dims() != [cols]
    {
        return Err(Error::Other(
            "static inference LayerNorm expects BF16 [rows,cols] and BF16 [cols] affine tensors"
                .into(),
        ));
    }
    if !eps.is_finite() || eps <= 0.0 || !scale.is_finite() || scale <= 0.0 {
        return Err(Error::Other(
            "static inference LayerNorm epsilon and FP8 scale must be finite and positive".into(),
        ));
    }
    let output = fp8_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_layer_norm_quant_bf16_e4m3(
            gpu_ptr(input)?,
            gpu_ptr(weight)?,
            gpu_ptr(bias)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            eps,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::F8E4M3,
        ctx.device_id(),
        output,
    ))
}

pub fn adaptive_rms_quant_f16_e4m3(
    ctx: &CudaContext,
    input: &Tensor,
    style: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "AdaRMSNorm")?;
    if input.dtype() != DType::F16
        || style.dtype() != DType::F16
        || style.shape().dims() != [3 * cols]
    {
        return Err(Error::Other(
            "static inference AdaRMSNorm expects FP16 input and [3*cols] style".into(),
        ));
    }
    let output = fp8_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_ada_rms_norm_quant_f16_e4m3(
            gpu_ptr(input)?,
            gpu_ptr(style)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            eps,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::F8E4M3,
        ctx.device_id(),
        output,
    ))
}
