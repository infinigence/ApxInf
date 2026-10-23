//! Explicit SM110 operators. Unsupported inputs return `None` for caller fallback.
//! These functions never change shared dispatch state.
use super::contracts::{
    bf16_output, checked_bytes, fp8_output, gpu_ptr, make_gpu_tensor, matrix_shape, matrix_tensor,
    optional_ptr, require_buffers,
};
use super::{fused::ResidualNormTensors, rope::QkvTensors};
use crate::{buffer::CudaBuffer, context::CudaContext, ffi, workspace::output_buffer};
use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

fn supported(ctx: &CudaContext, inputs: &[&Tensor]) -> bool {
    ctx.caps().sm == 110
        && inputs.iter().all(|input| {
            input.dtype() == DType::BF16
                && input.device() == Device::Cuda(ctx.device_id())
                && gpu_ptr(input).is_ok_and(|pointer| pointer as usize % 16 == 0)
                && CudaBuffer::from_tensor(input).is_ok_and(|buffer| {
                    buffer.device() == ctx.device_id() && buffer.len() >= input.size_in_bytes()
                })
                && input
                    .shape()
                    .dims()
                    .iter()
                    .all(|&d| d > 0 && d <= i32::MAX as usize)
        })
}

fn matrix(input: &Tensor) -> Option<(usize, usize)> {
    match input.shape().dims() {
        &[rows, cols] if rows > 0 && cols > 0 => Some((rows, cols)),
        _ => None,
    }
}

fn positive(value: f32) -> bool {
    value.is_finite() && value > 0.0
}

fn check_cuda(status: ffi::cudaError_t) -> Result<()> {
    ffi::check_cuda(status).map_err(Error::Cuda)
}

/// Explicit SM110 path; returns `None` when unsupported.
pub fn try_layer_quant_bf16_e4m3_sm110(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<Option<Tensor>> {
    let Some((_, cols)) = matrix(input) else {
        return Ok(None);
    };
    if !supported(ctx, &[input, weight, bias])
        || cols != 1024
        || weight.shape().dims() != [cols]
        || bias.shape().dims() != [cols]
        || !positive(eps)
        || !positive(scale)
    {
        return Ok(None);
    }

    let (rows, cols) = matrix_shape(input, "LayerNorm")?;
    let output = fp8_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_layer_norm_quant_bf16_e4m3_sm110(
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
    Ok(Some(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::F8E4M3,
        ctx.device_id(),
        output,
    )))
}

/// Explicit SM110 path; returns `None` when unsupported.
pub fn try_bias_gelu_quant_bf16_e4m3_sm110(
    ctx: &CudaContext,
    input: &Tensor,
    bias: &Tensor,
    scale: f32,
) -> Result<Option<Tensor>> {
    let Some((rows, cols)) = matrix(input) else {
        return Ok(None);
    };
    if !supported(ctx, &[input, bias])
        || cols % 8 != 0
        || rows > 65535
        || bias.shape().dims() != [cols]
        || !positive(scale)
    {
        return Ok(None);
    }

    let (rows, cols) = matrix_shape(input, "bias GELU quantization")?;
    let output = fp8_output(ctx, rows, cols)?;
    check_cuda(unsafe {
        ffi::apxinf_static_bias_gelu_quant_bf16_e4m3_sm110(
            gpu_ptr(input)?,
            gpu_ptr(bias)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            scale,
            ctx.stream().handle(),
        )
    })?;
    Ok(Some(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::F8E4M3,
        ctx.device_id(),
        output,
    )))
}

/// Explicit SM110 path; returns `None` when unsupported.
pub fn try_concat_rows_quantize_bf16_e4m3_sm110(
    ctx: &CudaContext,
    first: &Tensor,
    second: &Tensor,
    scale: f32,
) -> Result<Option<Tensor>> {
    let Some((_, cols)) = matrix(first) else {
        return Ok(None);
    };
    let Some((_, second_cols)) = matrix(second) else {
        return Ok(None);
    };
    if !supported(ctx, &[first, second]) || cols != second_cols || cols % 8 != 0 || !positive(scale)
    {
        return Ok(None);
    }

    let (first_rows, cols) = matrix_shape(first, "row concatenation quantization")?;
    let (second_rows, _) = matrix_shape(second, "row concatenation quantization")?;
    let rows = first_rows
        .checked_add(second_rows)
        .ok_or_else(|| Error::Other("row concatenation quantization size overflow".into()))?;
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| Error::Other("row concatenation quantization size overflow".into()))?;
    let output = output_buffer(ctx, elements)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_concat_rows_quantize_bf16_e4m3_sm110(
            gpu_ptr(first)?,
            gpu_ptr(second)?,
            output.ptr(),
            first_rows as i32,
            second_rows as i32,
            cols as i32,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(Some(make_gpu_tensor(
        apxinf_core::Shape::new(vec![rows, cols]),
        DType::F8E4M3,
        ctx.device_id(),
        output,
    )))
}

/// Explicit SM110 path; returns `None` when unsupported.
pub fn try_bias_residual_bf16_sm110(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
) -> Result<Option<Tensor>> {
    let Some((_, cols)) = matrix(projection) else {
        return Ok(None);
    };
    if !supported(ctx, &[projection, residual])
        || cols % 8 != 0
        || residual.shape() != projection.shape()
        || bias.is_some_and(|b| !supported(ctx, &[b]) || b.shape().dims() != [cols])
    {
        return Ok(None);
    }

    let (rows, cols) = matrix_shape(projection, "bias residual")?;
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_bias_residual_bf16_sm110(
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
    Ok(Some(matrix_tensor(ctx, rows, cols, output)))
}

/// Explicit SM110 path; returns `None` when unsupported.
pub fn try_bias_residual_layer_quant_bf16_e4m3_sm110(
    ctx: &CudaContext,
    projection: &Tensor,
    projection_bias: Option<&Tensor>,
    residual: &Tensor,
    norm_weight: &Tensor,
    norm_bias: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<Option<ResidualNormTensors>> {
    let Some((_, cols)) = matrix(projection) else {
        return Ok(None);
    };
    if !supported(ctx, &[projection, residual, norm_weight, norm_bias])
        || cols != 1024
        || residual.shape() != projection.shape()
        || norm_weight.shape().dims() != [cols]
        || norm_bias.shape().dims() != [cols]
        || !positive(eps)
        || !positive(scale)
        || projection_bias.is_some_and(|b| !supported(ctx, &[b]) || b.shape().dims() != [cols])
    {
        return Ok(None);
    }

    let (rows, cols) = matrix_shape(projection, "quantized residual LayerNorm")?;
    let hidden = bf16_output(ctx, rows, cols)?;
    let normalized = fp8_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(
            ffi::apxinf_static_bias_residual_layer_norm_quant_bf16_e4m3_sm110(
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
            ),
        )
        .map_err(Error::Cuda)?;
    }
    Ok(Some(ResidualNormTensors {
        hidden: matrix_tensor(ctx, rows, cols, hidden),
        normalized: make_gpu_tensor(
            Shape::new(vec![rows, cols]),
            DType::F8E4M3,
            ctx.device_id(),
            normalized,
        ),
    }))
}

/// Four-pair precomputed vision RoPE for the measured SM110 shapes.
pub fn try_split_qkv_bias_apply_vision_2d_precomputed_sm110(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: &Tensor,
    n_heads: usize,
    head_dim: usize,
    rotation_table: &CudaBuffer,
) -> Result<Option<QkvTensors>> {
    let Some((seq, width)) = matrix(qkv) else {
        return Ok(None);
    };
    if !supported(ctx, &[qkv, bias])
        || n_heads != 16
        || head_dim != 64
        || !matches!(seq, 256 | 512)
        || width != 3 * n_heads * head_dim
        || bias.shape().dims() != [width]
        || rotation_table.device() != ctx.device_id()
        || rotation_table.ptr() as usize % 32 != 0
    {
        return Ok(None);
    }

    let (seq_len, width) = matrix_shape(qkv, "precomputed vision fused QKV 2D-RoPE")?;
    let qkv_bytes = checked_bytes(
        DType::BF16,
        &[seq_len, width],
        "precomputed vision fused QKV",
    )?;
    let bias_bytes = checked_bytes(DType::BF16, &[width], "precomputed vision fused QKV bias")?;
    let output_bytes = checked_bytes(
        DType::BF16,
        &[seq_len, n_heads, head_dim],
        "precomputed vision fused QKV output",
    )?;
    let rotation_bytes = seq_len
        .checked_mul(head_dim / 2)
        .and_then(|count| count.checked_mul(2 * std::mem::size_of::<f32>()))
        .ok_or_else(|| Error::Other("precomputed vision rotation size overflow".into()))?;
    let qkv_buffer = CudaBuffer::from_tensor(qkv).map_err(Error::Cuda)?;
    let bias_buffer = CudaBuffer::from_tensor(bias).map_err(Error::Cuda)?;
    require_buffers(
        ctx,
        "precomputed vision fused QKV 2D-RoPE",
        &[
            ("qkv", &qkv_buffer, qkv_bytes),
            ("bias", &bias_buffer, bias_bytes),
            ("rotation table", rotation_table, rotation_bytes),
        ],
    )?;
    let q = output_buffer(ctx, output_bytes)?;
    let k = output_buffer(ctx, output_bytes)?;
    let v = output_buffer(ctx, output_bytes)?;
    check_cuda(unsafe {
        ffi::apxinf_qkv_split_bias_vision_rope_precomputed_bf16_sm110(
            qkv_buffer.ptr(),
            bias_buffer.ptr(),
            q.ptr(),
            k.ptr(),
            v.ptr(),
            u32::try_from(head_dim)
                .map_err(|_| Error::Other("vision fused QKV head dimension exceeds u32".into()))?,
            u32::try_from(n_heads)
                .map_err(|_| Error::Other("vision fused QKV head count exceeds u32".into()))?,
            u32::try_from(seq_len)
                .map_err(|_| Error::Other("vision fused QKV sequence exceeds u32".into()))?,
            rotation_table.ptr(),
            ctx.stream().handle(),
        )
    })?;
    let shape = Shape::new(vec![seq_len, n_heads, head_dim]);
    Ok(Some(QkvTensors {
        q: make_gpu_tensor(shape.clone(), DType::BF16, ctx.device_id(), q),
        k: make_gpu_tensor(shape.clone(), DType::BF16, ctx.device_id(), k),
        v: make_gpu_tensor(shape, DType::BF16, ctx.device_id(), v),
    }))
}
