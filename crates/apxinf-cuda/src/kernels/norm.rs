//! Normalization operator contracts.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use super::contracts::{
    bf16_output, check_cuda, checked_bytes, fp8_output, gpu_ptr, make_gpu_tensor, matrix_shape,
    matrix_tensor, require_buffers, require_finite, unsupported_dtype,
};
use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;

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
    let out_buf = CudaBuffer::alloc_zeros(out_bytes, device_id).map_err(Error::Cuda)?;

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
    let out_buf = CudaBuffer::alloc_zeros(input.size_in_bytes(), device_id).map_err(Error::Cuda)?;
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

/// Affine-free LayerNorm followed by timestep shift/scale.
/// `style` is `[2 * cols]` or `[1, 2 * cols]`, broadcast across rows and laid
/// out as `(scale, shift)` unless `shift_first` is set.
pub fn adaptive_layer_bf16(
    ctx: &CudaContext,
    input: &Tensor,
    style: &Tensor,
    eps: f32,
    shift_first: bool,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "AdaLayerNorm")?;
    let style_dims = style.shape().dims();
    if input.dtype() != DType::BF16
        || style.dtype() != DType::BF16
        || (style_dims != [2 * cols] && style_dims != [1, 2 * cols])
    {
        return Err(Error::Other(
            "static inference BF16 AdaLayerNorm shape mismatch".into(),
        ));
    }
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_ada_layer_norm_bf16(
            gpu_ptr(input)?,
            gpu_ptr(style)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            eps,
            shift_first,
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
