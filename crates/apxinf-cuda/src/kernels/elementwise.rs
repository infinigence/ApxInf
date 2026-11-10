//! Elementwise operator contracts.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use super::contracts::{
    bf16_output, check_cuda, checked_bytes, f16_output, gpu_ptr, make_gpu_tensor, matrix_shape,
    matrix_tensor, require_buffers, unsupported_dtype,
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
    let status = unsafe {
        match dtype {
            DType::F32 => ffi::apxinf_add_f32(
                a.ptr(),
                b.ptr(),
                output.ptr(),
                count as u32,
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_add_bf16(
                a.ptr(),
                b.ptr(),
                output.ptr(),
                count as u32,
                ctx.stream().handle(),
            ),
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
    let status = unsafe {
        match dtype {
            DType::F32 => ffi::apxinf_mul_f32(
                a.ptr(),
                b.ptr(),
                output.ptr(),
                count as u32,
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_mul_bf16(
                a.ptr(),
                b.ptr(),
                output.ptr(),
                count as u32,
                ctx.stream().handle(),
            ),
            dtype => {
                return Err(apxinf_core::Error::Other(format!(
                    "decode multiply does not support {dtype}"
                )))
            }
        }
    };
    check_cuda(status)
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
    let device_id = ctx.device_id();
    let count = a.numel() as u32;

    let out_bytes = a.size_in_bytes();
    let out_buf = CudaBuffer::alloc_zeros(out_bytes, device_id).map_err(Error::Cuda)?;

    unsafe {
        let res = match a.dtype() {
            DType::F32 => ffi::apxinf_add_f32(
                gpu_ptr(a)?,
                gpu_ptr(b)?,
                out_buf.ptr(),
                count,
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_add_bf16(
                gpu_ptr(a)?,
                gpu_ptr(b)?,
                out_buf.ptr(),
                count,
                ctx.stream().handle(),
            ),
            dtype => return unsupported_dtype("add", dtype),
        };
        ffi::check_cuda(res).map_err(Error::Cuda)?;
    }

    Ok(make_gpu_tensor(
        a.shape().clone(),
        a.dtype(),
        device_id,
        out_buf,
    ))
}

/// Element-wise multiply on CUDA. Dispatches on dtype.
pub fn mul(ctx: &CudaContext, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let device_id = ctx.device_id();
    let count = a.numel() as u32;

    let out_bytes = a.size_in_bytes();
    let out_buf = CudaBuffer::alloc_zeros(out_bytes, device_id).map_err(Error::Cuda)?;

    unsafe {
        let res = match a.dtype() {
            DType::F32 => ffi::apxinf_mul_f32(
                gpu_ptr(a)?,
                gpu_ptr(b)?,
                out_buf.ptr(),
                count,
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_mul_bf16(
                gpu_ptr(a)?,
                gpu_ptr(b)?,
                out_buf.ptr(),
                count,
                ctx.stream().handle(),
            ),
            dtype => return unsupported_dtype("mul", dtype),
        };
        ffi::check_cuda(res).map_err(Error::Cuda)?;
    }

    Ok(make_gpu_tensor(
        a.shape().clone(),
        a.dtype(),
        device_id,
        out_buf,
    ))
}

/// Multiply every element by a scalar. Dispatches on dtype.
pub fn scale(ctx: &CudaContext, input: &Tensor, scale_factor: f32) -> Result<Tensor> {
    let device_id = ctx.device_id();
    let count = input.numel() as u32;

    let out_bytes = input.size_in_bytes();
    let out_buf = CudaBuffer::alloc_zeros(out_bytes, device_id).map_err(Error::Cuda)?;

    unsafe {
        let res = match input.dtype() {
            DType::F32 => ffi::apxinf_scale_f32(
                gpu_ptr(input)?,
                out_buf.ptr(),
                count,
                scale_factor,
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_scale_bf16(
                gpu_ptr(input)?,
                out_buf.ptr(),
                count,
                scale_factor,
                ctx.stream().handle(),
            ),
            dtype => return unsupported_dtype("scale", dtype),
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

/// Copy a contiguous column range from a CUDA BF16 matrix without touching
/// host memory. This is used to separate Qwen3.5's fused q/gate projection.
pub fn slice_columns_bf16(
    ctx: &CudaContext,
    input: &Tensor,
    start: usize,
    width: usize,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "column slice")?;
    if input.dtype() != DType::BF16
        || input.device() != apxinf_core::Device::Cuda(ctx.device_id())
        || width == 0
        || start.checked_add(width).is_none()
        || start + width > cols
    {
        return Err(Error::Other(format!(
            "BF16 column slice expects CUDA matrix and range within [0,{cols}), got start={start} width={width}"
        )));
    }
    let source = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let source_offset = start * DType::BF16.size_in_bytes();
    let source_view = source
        .view(source_offset, source.len() - source_offset)
        .map_err(Error::Cuda)?;
    let output = bf16_output(ctx, rows, width)?;
    let width_bytes = width * DType::BF16.size_in_bytes();
    let source_pitch = cols * DType::BF16.size_in_bytes();
    let destination_pitch = width_bytes;
    unsafe {
        ffi::check_cuda(ffi::cudaMemcpy2DAsync(
            output.ptr(),
            destination_pitch,
            source_view.ptr(),
            source_pitch,
            width_bytes,
            rows,
            ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(matrix_tensor(ctx, rows, width, output))
}

/// Stack `[1, width]` BF16 rows into one `[rows.len(), width]` matrix by
/// device-to-device copies. Used by the batched decode engine to regroup
/// per-request results (conv/attention outputs) into a batch for the shared
/// projections; the copies preserve every row bit-for-bit.
pub fn stack_rows_bf16(ctx: &CudaContext, rows: &[&Tensor]) -> Result<Tensor> {
    if rows.is_empty() {
        return Err(Error::Other(
            "BF16 row stack expects at least one row".into(),
        ));
    }
    let (first_rows, width) = matrix_shape(rows[0], "row stack")?;
    if first_rows != 1 {
        return Err(Error::Other(format!(
            "BF16 row stack expects [1, width] rows, got {first_rows} rows"
        )));
    }
    let expected_device = apxinf_core::Device::Cuda(ctx.device_id());
    for row in rows {
        let (row_count, row_width) = matrix_shape(row, "row stack")?;
        if row.dtype() != DType::BF16
            || row.device() != expected_device
            || row_count != 1
            || row_width != width
        {
            return Err(Error::Other(format!(
                "BF16 row stack expects CUDA BF16 [1, {width}] rows, got {:?} {}",
                row.shape().dims(),
                row.dtype()
            )));
        }
    }
    let output = bf16_output(ctx, rows.len(), width)?;
    let row_bytes = width * DType::BF16.size_in_bytes();
    for (index, row) in rows.iter().enumerate() {
        let source = CudaBuffer::from_tensor(row).map_err(Error::Cuda)?;
        // Stream-ordered copy: a synchronous cudaMemcpy here would fence the
        // whole stream once per row per layer in the batched decode loop.
        unsafe {
            ffi::check_cuda(ffi::cudaMemcpy2DAsync(
                (output.ptr() as *mut u8).wrapping_add(index * row_bytes) as *mut std::ffi::c_void,
                row_bytes,
                source.ptr(),
                row_bytes,
                row_bytes,
                1,
                ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)?;
        }
    }
    Ok(matrix_tensor(ctx, rows.len(), width, output))
}

/// Zero-copy view of row `index` of a `[rows, width]` BF16 device matrix as
/// a `[1, width]` tensor sharing the same storage. The batched decode engine
/// hands these to read-only per-request kernels (convolution, recurrence,
/// RoPE, flash decode) without any copy or allocation.
pub fn row_view_bf16(input: &Tensor, index: usize) -> Result<Tensor> {
    let dims = input.shape().dims();
    if input.dtype() != DType::BF16 || dims.len() != 2 || index >= dims[0] {
        return Err(Error::Other(format!(
            "BF16 row view expects a CUDA BF16 matrix with row {index}, got {dims:?}"
        )));
    }
    let width = dims[1];
    let row_bytes = width * DType::BF16.size_in_bytes();
    let source = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let view = source
        .view(index * row_bytes, row_bytes)
        .map_err(Error::Cuda)?;
    Ok(view.into_tensor(apxinf_core::Shape::new(vec![1, width]), DType::BF16))
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
