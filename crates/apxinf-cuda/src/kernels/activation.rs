//! Activation operator contracts.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use super::contracts::{
    bf16_output, check_cuda, checked_bytes, f16_output, fp8_output, gpu_ptr, make_gpu_tensor,
    matrix_shape, matrix_tensor, optional_ptr, require_buffers, unsupported_dtype,
};
use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;
use crate::workspace::output_buffer;

/// Allocation-free SiLU into caller-owned decode storage.
pub fn silu_into(
    ctx: &CudaContext,
    dtype: DType,
    input: &CudaBuffer,
    output: &CudaBuffer,
    count: usize,
) -> Result<()> {
    let bytes = checked_bytes(dtype, &[count], "decode SiLU")?;
    require_buffers(
        ctx,
        "decode SiLU",
        &[("input", input, bytes), ("output", output, bytes)],
    )?;
    let status = unsafe {
        match dtype {
            DType::F32 => ffi::apxinf_silu_f32(
                input.ptr(),
                output.ptr(),
                count as u32,
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_silu_bf16(
                input.ptr(),
                output.ptr(),
                count as u32,
                ctx.stream().handle(),
            ),
            dtype => {
                return Err(apxinf_core::Error::Other(format!(
                    "decode SiLU does not support {dtype}"
                )))
            }
        }
    };
    check_cuda(status)
}

/// Fused BF16 SiLU(gate) * up into caller-owned decode storage.
pub fn silu_mul_bf16_into(
    ctx: &CudaContext,
    gate_up: &CudaBuffer,
    output: &CudaBuffer,
    intermediate: usize,
) -> Result<()> {
    require_buffers(
        ctx,
        "decode SiLU multiply",
        &[
            (
                "gate_up",
                gate_up,
                checked_bytes(DType::BF16, &[2, intermediate], "decode SiLU multiply")?,
            ),
            (
                "output",
                output,
                checked_bytes(DType::BF16, &[intermediate], "decode SiLU multiply")?,
            ),
        ],
    )?;
    check_cuda(unsafe {
        ffi::apxinf_silu_mul_bf16(
            gate_up.ptr(),
            output.ptr(),
            intermediate as u32,
            ctx.stream().handle(),
        )
    })
}

/// SiLU (Swish) activation on CUDA.
pub fn silu(ctx: &CudaContext, input: &Tensor) -> Result<Tensor> {
    let device_id = ctx.device_id();
    let count = input.numel() as u32;

    let out_bytes = input.size_in_bytes();
    let out_buf = output_buffer(ctx, out_bytes)?;

    unsafe {
        let res = match input.dtype() {
            apxinf_core::DType::F32 => {
                ffi::apxinf_silu_f32(gpu_ptr(input)?, out_buf.ptr(), count, ctx.stream().handle())
            }
            apxinf_core::DType::BF16 => {
                ffi::apxinf_silu_bf16(gpu_ptr(input)?, out_buf.ptr(), count, ctx.stream().handle())
            }
            dtype => return unsupported_dtype("silu", dtype),
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

/// GELU with tanh approximation (bf16). Element-wise.
pub fn gelu_tanh(ctx: &CudaContext, input: &Tensor) -> Result<Tensor> {
    if input.dtype() != DType::BF16 {
        return Err(Error::Other("gelu_tanh: only BF16 supported".into()));
    }
    let device_id = ctx.device_id();
    let count = input.numel() as u32;
    let out_buf = output_buffer(ctx, input.size_in_bytes())?;
    unsafe {
        let res =
            ffi::apxinf_gelu_tanh_bf16(gpu_ptr(input)?, out_buf.ptr(), count, ctx.stream().handle());
        ffi::check_cuda(res).map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        input.shape().clone(),
        DType::BF16,
        device_id,
        out_buf,
    ))
}
pub(super) fn bias_activation(
    ctx: &CudaContext,
    input: &Tensor,
    bias: Option<&Tensor>,
    activation: i32,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "bias activation")?;
    if input.dtype() != DType::BF16
        || bias.is_some_and(|value| value.dtype() != DType::BF16 || value.shape().dims() != [cols])
    {
        return Err(Error::Other(
            "static inference BF16 bias activation has incompatible dtype or shape".into(),
        ));
    }
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_bias_activation_bf16(
            gpu_ptr(input)?,
            optional_ptr(bias)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            activation,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(matrix_tensor(ctx, rows, cols, output))
}

pub fn bias_gelu_bf16(ctx: &CudaContext, input: &Tensor, value: Option<&Tensor>) -> Result<Tensor> {
    bias_activation(ctx, input, value, 1)
}

pub fn bias_silu_bf16(ctx: &CudaContext, input: &Tensor, value: Option<&Tensor>) -> Result<Tensor> {
    bias_activation(ctx, input, value, 2)
}

/// Broadcast-add an optional BF16 bias and apply ReLU on device.
pub fn bias_relu_bf16(ctx: &CudaContext, input: &Tensor, value: Option<&Tensor>) -> Result<Tensor> {
    bias_activation(ctx, input, value, 3)
}

pub fn geglu_bf16(ctx: &CudaContext, gate_up: &Tensor) -> Result<Tensor> {
    let (rows, twice_inner) = matrix_shape(gate_up, "GeGLU")?;
    if gate_up.dtype() != DType::BF16 || twice_inner % 2 != 0 {
        return Err(Error::Other(
            "static inference BF16 GeGLU expects [rows,2*inner]".into(),
        ));
    }
    let inner = twice_inner / 2;
    let output = bf16_output(ctx, rows, inner)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_geglu_bf16(
            gpu_ptr(gate_up)?,
            output.ptr(),
            rows as i32,
            inner as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(matrix_tensor(ctx, rows, inner, output))
}
pub fn bias_gelu_quant_f16_e4m3(
    ctx: &CudaContext,
    input: &Tensor,
    bias: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "bias GELU")?;
    if input.dtype() != DType::F16 || bias.dtype() != DType::F16 || bias.shape().dims() != [cols] {
        return Err(Error::Other(
            "static inference bias GELU expects FP16 matrix and matching bias".into(),
        ));
    }
    let output = fp8_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_bias_gelu_quant_f16_e4m3(
            gpu_ptr(input)?,
            gpu_ptr(bias)?,
            output.ptr(),
            rows as i32,
            cols as i32,
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

pub fn bias_silu_quant_f16_e4m3(
    ctx: &CudaContext,
    input: &Tensor,
    bias: Option<&Tensor>,
    scale: f32,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "bias SiLU")?;
    if input.dtype() != DType::F16
        || bias.is_some_and(|x| x.dtype() != DType::F16 || x.shape().dims() != [cols])
    {
        return Err(Error::Other(
            "static inference bias SiLU expects an FP16 matrix and matching bias".into(),
        ));
    }
    let output = fp8_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_bias_silu_quant_f16_e4m3(
            gpu_ptr(input)?,
            bias.map(gpu_ptr)
                .transpose()?
                .unwrap_or(std::ptr::null_mut()),
            output.ptr(),
            rows as i32,
            cols as i32,
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

pub fn bias_silu_f16(ctx: &CudaContext, input: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "bias SiLU")?;
    if input.dtype() != DType::F16
        || bias.is_some_and(|x| x.dtype() != DType::F16 || x.shape().dims() != [cols])
    {
        return Err(Error::Other(
            "static inference bias SiLU expects an FP16 matrix and matching bias".into(),
        ));
    }
    let output = f16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_bias_silu_f16(
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

pub fn geglu_quant_f16_e4m3(ctx: &CudaContext, gate_up: &Tensor, scale: f32) -> Result<Tensor> {
    let (rows, twice_inner) = matrix_shape(gate_up, "GeGLU")?;
    if gate_up.dtype() != DType::F16 || twice_inner % 2 != 0 {
        return Err(Error::Other(
            "static inference GeGLU expects FP16 [rows,2*inner]".into(),
        ));
    }
    let inner = twice_inner / 2;
    let output = fp8_output(ctx, rows, inner)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_geglu_quant_f16_e4m3(
            gpu_ptr(gate_up)?,
            output.ptr(),
            rows as i32,
            inner as i32,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, inner]),
        DType::F8E4M3,
        ctx.device_id(),
        output,
    ))
}
