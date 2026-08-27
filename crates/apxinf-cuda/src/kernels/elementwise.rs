//! Elementwise operator contracts.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use super::contracts::{
    bf16_output, check_cuda, checked_bytes, f16_output, gpu_ptr, make_gpu_tensor, matrix_shape,
    matrix_tensor, require_buffers, require_finite, unsupported_dtype,
};
use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;
use crate::workspace::output_buffer;

pub fn add_into(
    ctx: &CudaContext,
    dtype: DType,
    a: &CudaBuffer,
    b: &CudaBuffer,
    output: &CudaBuffer,
    count: usize,
) -> Result<()> {
    let bytes = checked_bytes(dtype, &[count], "decode add")?;
    require_buffers(
        ctx,
        "decode add",
        &[("A", a, bytes), ("B", b, bytes), ("output", output, bytes)],
    )?;
    let count = u32::try_from(count)
        .map_err(|_| Error::Other("decode add element count exceeds u32".into()))?;
    let status = unsafe {
        match dtype {
            DType::F32 => {
                ffi::apxinf_add_f32(a.ptr(), b.ptr(), output.ptr(), count, ctx.stream().handle())
            }
            DType::BF16 => {
                ffi::apxinf_add_bf16(a.ptr(), b.ptr(), output.ptr(), count, ctx.stream().handle())
            }
            dtype => {
                return Err(apxinf_core::Error::Other(format!(
                    "decode add does not support {dtype}"
                )))
            }
        }
    };
    check_cuda(status)
}

pub fn mul_into(
    ctx: &CudaContext,
    dtype: DType,
    a: &CudaBuffer,
    b: &CudaBuffer,
    output: &CudaBuffer,
    count: usize,
) -> Result<()> {
    let bytes = checked_bytes(dtype, &[count], "decode multiply")?;
    require_buffers(
        ctx,
        "decode multiply",
        &[("A", a, bytes), ("B", b, bytes), ("output", output, bytes)],
    )?;
    let count = u32::try_from(count)
        .map_err(|_| Error::Other("decode multiply element count exceeds u32".into()))?;
    let status = unsafe {
        match dtype {
            DType::F32 => {
                ffi::apxinf_mul_f32(a.ptr(), b.ptr(), output.ptr(), count, ctx.stream().handle())
            }
            DType::BF16 => {
                ffi::apxinf_mul_bf16(a.ptr(), b.ptr(), output.ptr(), count, ctx.stream().handle())
            }
            dtype => {
                return Err(apxinf_core::Error::Other(format!(
                    "decode multiply does not support {dtype}"
                )))
            }
        }
    };
    check_cuda(status)
}

/// Scale caller-owned storage without allocating or synchronizing.
///
/// Keeping this primitive separate from a following [`add_into`] preserves an
/// explicit dtype boundary such as `round_bf16(velocity * dt)` before the
/// addition in a two-stage Euler update. The caller owns both fixed-address
/// buffers, so the operation is safe to record in a CUDA Graph.
pub fn scale_into(
    ctx: &CudaContext,
    dtype: DType,
    input: &CudaBuffer,
    output: &CudaBuffer,
    count: usize,
    scale_factor: f32,
) -> Result<()> {
    require_finite("scale", &[scale_factor])?;
    let bytes = checked_bytes(dtype, &[count], "scale")?;
    require_buffers(
        ctx,
        "scale",
        &[("input", input, bytes), ("output", output, bytes)],
    )?;
    let count = u32::try_from(count)
        .map_err(|_| Error::Other("caller-owned scale element count exceeds u32".into()))?;
    let status = unsafe {
        match dtype {
            DType::F32 => ffi::apxinf_scale_f32(
                input.ptr(),
                output.ptr(),
                count,
                scale_factor,
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_scale_bf16(
                input.ptr(),
                output.ptr(),
                count,
                scale_factor,
                ctx.stream().handle(),
            ),
            dtype => {
                return Err(Error::Other(format!(
                    "caller-owned scale does not support {dtype}"
                )))
            }
        }
    };
    check_cuda(status)
}

/// Multiply BF16 matrix rows by the gate third of a packed
/// `[scale, shift, gate]` style tensor into caller-owned storage.
///
/// A subsequent [`add_into`] preserves the intermediate BF16 product instead
/// of fusing multiply and residual addition into one rounding boundary.
pub fn mul_style_gate_bf16_into(
    ctx: &CudaContext,
    input: &CudaBuffer,
    style: &CudaBuffer,
    output: &CudaBuffer,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let matrix = checked_bytes(DType::BF16, &[rows, cols], "style gate multiply")?;
    let style_bytes = checked_bytes(DType::BF16, &[3, cols], "style gate multiply")?;
    require_buffers(
        ctx,
        "style gate multiply",
        &[
            ("input", input, matrix),
            ("style", style, style_bytes),
            ("output", output, matrix),
        ],
    )?;
    let rows =
        i32::try_from(rows).map_err(|_| Error::Other("style gate row count exceeds i32".into()))?;
    let cols = i32::try_from(cols)
        .map_err(|_| Error::Other("style gate column count exceeds i32".into()))?;
    check_cuda(unsafe {
        ffi::apxinf_style_gate_mul_bf16(
            input.ptr(),
            style.ptr(),
            output.ptr(),
            rows,
            cols,
            ctx.stream().handle(),
        )
    })
}

/// Graph-workspace wrapper for [`mul_style_gate_bf16_into`].
pub fn mul_style_gate_bf16(ctx: &CudaContext, input: &Tensor, style: &Tensor) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "style gate multiply")?;
    let style_width = cols
        .checked_mul(3)
        .ok_or_else(|| Error::Other("style gate width overflow".into()))?;
    let expected_device = apxinf_core::Device::Cuda(ctx.device_id());
    if input.dtype() != DType::BF16
        || style.dtype() != DType::BF16
        || style.shape().dims() != [style_width]
        || input.device() != expected_device
        || style.device() != expected_device
    {
        return Err(Error::Other(
            "style gate multiply expects CUDA BF16 input [rows,cols] and style [3*cols]".into(),
        ));
    }
    let input = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let style = CudaBuffer::from_tensor(style).map_err(Error::Cuda)?;
    let output = bf16_output(ctx, rows, cols)?;
    mul_style_gate_bf16_into(ctx, &input, &style, &output, rows, cols)?;
    Ok(matrix_tensor(ctx, rows, cols, output))
}

/// Exact two-stage BF16 Euler expression:
/// `round_bf16(state + round_bf16(velocity * dt))`.
///
/// Both intermediates come from the active graph workspace, so callers do not
/// allocate raw buffers or collapse the two dtype boundaries.
pub fn euler_two_stage_bf16(
    ctx: &CudaContext,
    state: &Tensor,
    velocity: &Tensor,
    dt: f32,
) -> Result<Tensor> {
    require_finite("two-stage Euler", &[dt])?;
    let expected_device = apxinf_core::Device::Cuda(ctx.device_id());
    if state.dtype() != DType::BF16
        || velocity.dtype() != DType::BF16
        || state.shape() != velocity.shape()
        || state.device() != expected_device
        || velocity.device() != expected_device
    {
        return Err(Error::Other(
            "two-stage Euler expects matching CUDA BF16 state and velocity tensors".into(),
        ));
    }
    let state_buffer = CudaBuffer::from_tensor(state).map_err(Error::Cuda)?;
    let velocity_buffer = CudaBuffer::from_tensor(velocity).map_err(Error::Cuda)?;
    let scaled = output_buffer(ctx, state.size_in_bytes())?;
    let output = output_buffer(ctx, state.size_in_bytes())?;
    scale_into(
        ctx,
        DType::BF16,
        &velocity_buffer,
        &scaled,
        state.numel(),
        dt,
    )?;
    add_into(
        ctx,
        DType::BF16,
        &state_buffer,
        &scaled,
        &output,
        state.numel(),
    )?;
    Ok(make_gpu_tensor(
        state.shape().clone(),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

/// Broadcast-add a bias vector `[cols]` over rows of `input` `[rows, cols]`.
/// bf16 only.
pub fn add_bias(ctx: &CudaContext, input: &Tensor, bias: &Tensor) -> Result<Tensor> {
    if input.dtype() != DType::BF16 {
        return Err(Error::Other("add_bias: only BF16 supported".into()));
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
        let res = ffi::apxinf_add_bias_bf16(
            gpu_ptr(input)?,
            gpu_ptr(bias)?,
            out_buf.ptr(),
            cols as u32,
            rows as u32,
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

pub fn add(ctx: &CudaContext, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape() != b.shape()
        || a.dtype() != b.dtype()
        || a.device() != b.device()
        || a.device() != apxinf_core::Device::Cuda(ctx.device_id())
    {
        return Err(Error::Other(
            "add expects matching tensors on the active CUDA device".into(),
        ));
    }
    if !matches!(a.dtype(), DType::F32 | DType::BF16) {
        return unsupported_dtype("add", a.dtype());
    }
    let a_buffer = CudaBuffer::from_tensor(a).map_err(Error::Cuda)?;
    let b_buffer = CudaBuffer::from_tensor(b).map_err(Error::Cuda)?;
    let out_buf = output_buffer(ctx, a.size_in_bytes())?;
    add_into(ctx, a.dtype(), &a_buffer, &b_buffer, &out_buf, a.numel())?;
    Ok(make_gpu_tensor(
        a.shape().clone(),
        a.dtype(),
        ctx.device_id(),
        out_buf,
    ))
}

/// Element-wise multiply on CUDA. Dispatches on dtype.
pub fn mul(ctx: &CudaContext, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape() != b.shape()
        || a.dtype() != b.dtype()
        || a.device() != b.device()
        || a.device() != apxinf_core::Device::Cuda(ctx.device_id())
    {
        return Err(Error::Other(
            "multiply expects matching tensors on the active CUDA device".into(),
        ));
    }
    if !matches!(a.dtype(), DType::F32 | DType::BF16) {
        return unsupported_dtype("mul", a.dtype());
    }
    let a_buffer = CudaBuffer::from_tensor(a).map_err(Error::Cuda)?;
    let b_buffer = CudaBuffer::from_tensor(b).map_err(Error::Cuda)?;
    let out_buf = output_buffer(ctx, a.size_in_bytes())?;
    mul_into(ctx, a.dtype(), &a_buffer, &b_buffer, &out_buf, a.numel())?;
    Ok(make_gpu_tensor(
        a.shape().clone(),
        a.dtype(),
        ctx.device_id(),
        out_buf,
    ))
}

/// Multiply every element by a scalar. Dispatches on dtype.
pub fn scale(ctx: &CudaContext, input: &Tensor, scale_factor: f32) -> Result<Tensor> {
    if input.device() != apxinf_core::Device::Cuda(ctx.device_id()) {
        return Err(Error::Other(
            "scale expects input on the active CUDA device".into(),
        ));
    }
    if !matches!(input.dtype(), DType::F32 | DType::BF16) {
        return unsupported_dtype("scale", input.dtype());
    }
    let input_buffer = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let out_buf = output_buffer(ctx, input.size_in_bytes())?;
    scale_into(
        ctx,
        input.dtype(),
        &input_buffer,
        &out_buf,
        input.numel(),
        scale_factor,
    )?;
    Ok(make_gpu_tensor(
        input.shape().clone(),
        input.dtype(),
        ctx.device_id(),
        out_buf,
    ))
}
pub fn bias_bf16(ctx: &CudaContext, input: &Tensor, value: Option<&Tensor>) -> Result<Tensor> {
    super::activation::bias_activation(ctx, input, value, 0)
}

pub fn concat_rows_bf16(ctx: &CudaContext, first: &Tensor, second: &Tensor) -> Result<Tensor> {
    let (first_rows, cols) = matrix_shape(first, "row concatenation")?;
    let (second_rows, second_cols) = matrix_shape(second, "row concatenation")?;
    if first.dtype() != DType::BF16 || second.dtype() != DType::BF16 || cols != second_cols {
        return Err(Error::Other(
            "static inference BF16 row concatenation requires matrices with equal widths".into(),
        ));
    }
    let output = bf16_output(ctx, first_rows + second_rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_concat_rows_bf16(
            gpu_ptr(first)?,
            gpu_ptr(second)?,
            output.ptr(),
            first_rows as i32,
            second_rows as i32,
            cols as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(matrix_tensor(ctx, first_rows + second_rows, cols, output))
}

pub fn euler_update_bf16(
    ctx: &CudaContext,
    state: &Tensor,
    velocity: &Tensor,
    dt: f32,
) -> Result<Tensor> {
    if state.dtype() != DType::BF16
        || velocity.dtype() != DType::BF16
        || state.shape() != velocity.shape()
    {
        return Err(Error::Other(
            "static inference BF16 Euler update expects matching tensors".into(),
        ));
    }
    let output = output_buffer(ctx, state.size_in_bytes())?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_euler_update_bf16(
            gpu_ptr(state)?,
            gpu_ptr(velocity)?,
            output.ptr(),
            state.numel() as i64,
            dt,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        state.shape().clone(),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}
pub fn bias_f16(ctx: &CudaContext, input: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "bias")?;
    if input.dtype() != DType::F16
        || bias.is_some_and(|x| x.dtype() != DType::F16 || x.shape().dims() != [cols])
    {
        return Err(Error::Other(
            "static inference bias expects an FP16 matrix and matching bias".into(),
        ));
    }
    let output = f16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_bias_f16(
            gpu_ptr(input)?,
            bias.map(gpu_ptr)
                .transpose()?
                .unwrap_or(std::ptr::null_mut()),
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

pub fn concat_rows_f16(ctx: &CudaContext, first: &Tensor, second: &Tensor) -> Result<Tensor> {
    let (first_rows, cols) = matrix_shape(first, "row concatenation")?;
    let (second_rows, second_cols) = matrix_shape(second, "row concatenation")?;
    if first.dtype() != DType::F16 || second.dtype() != DType::F16 || cols != second_cols {
        return Err(Error::Other(
            "static inference row concatenation expects FP16 matrices with equal widths".into(),
        ));
    }
    let output = f16_output(ctx, first_rows + second_rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_concat_rows_f16(
            gpu_ptr(first)?,
            gpu_ptr(second)?,
            output.ptr(),
            first_rows as i32,
            second_rows as i32,
            cols as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![first_rows + second_rows, cols]),
        DType::F16,
        ctx.device_id(),
        output,
    ))
}

pub fn euler_update_f16(
    ctx: &CudaContext,
    state: &Tensor,
    velocity: &Tensor,
    dt: f32,
) -> Result<Tensor> {
    if state.dtype() != DType::F16
        || velocity.dtype() != DType::F16
        || state.shape() != velocity.shape()
    {
        return Err(Error::Other(
            "static inference Euler update expects matching FP16 tensors".into(),
        ));
    }
    let output = output_buffer(ctx, state.size_in_bytes())?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_euler_update_f16(
            gpu_ptr(state)?,
            gpu_ptr(velocity)?,
            output.ptr(),
            state.numel() as i64,
            dt,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        state.shape().clone(),
        DType::F16,
        ctx.device_id(),
        output,
    ))
}
