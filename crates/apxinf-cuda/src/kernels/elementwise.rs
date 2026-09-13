//! Elementwise operator contracts.

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use super::contracts::{
    bf16_output, check_cuda, checked_bytes, f16_output, gpu_ptr, make_gpu_tensor, matrix_shape,
    matrix_tensor, require_buffers, unsupported_dtype,
};
use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;
use crate::workspace::output_buffer;

/// Reorder a contiguous NHWC BF16 tensor to NCHW through cuDNN.
pub fn nhwc_to_nchw_bf16(ctx: &CudaContext, x: &Tensor) -> Result<Tensor> {
    image_layout_bf16(ctx, x, true)
}

/// Reorder a contiguous NCHW BF16 tensor to NHWC through cuDNN.
pub fn nchw_to_nhwc_bf16(ctx: &CudaContext, x: &Tensor) -> Result<Tensor> {
    image_layout_bf16(ctx, x, false)
}

fn image_layout_bf16(ctx: &CudaContext, x: &Tensor, to_first: bool) -> Result<Tensor> {
    let d = x.shape().dims();
    if d.len() != 4
        || x.dtype() != DType::BF16
        || x.device() != apxinf_core::Device::Cuda(ctx.device_id())
    {
        return Err(Error::Other(
            "image layout conversion requires a 4D BF16 tensor on the context device".into(),
        ));
    }
    let bytes = checked_bytes(DType::BF16, d, "image layout conversion")?;
    let (n, c, h, w) = if to_first {
        (d[0], d[3], d[1], d[2])
    } else {
        (d[0], d[1], d[2], d[3])
    };
    let dims = [n, c, h, w].map(|v| {
        i32::try_from(v).map_err(|_| Error::Other("image layout dimension overflow".into()))
    });
    let [ni, ci, hi, wi] = dims;
    let (ni, ci, hi, wi) = (ni?, ci?, hi?, wi?);
    if !crate::workspace::may_prepare_native_resources() {
        return Err(Error::Other(
            "cuDNN image layout conversion needs prepared descriptors for capture".into(),
        ));
    }
    use crate::cudnn::{api, check, Descriptor};
    let api = api().map_err(Error::Cuda)?;
    let handle = Descriptor::new(api, api.create, api.destroy).map_err(Error::Cuda)?;
    let xd = Descriptor::new(api, api.create_tensor, api.destroy_tensor).map_err(Error::Cuda)?;
    let yd = Descriptor::new(api, api.create_tensor, api.destroy_tensor).map_err(Error::Cuda)?;
    let out = output_buffer(ctx, bytes)?;
    unsafe {
        check(api, (api.set_stream)(handle.raw, ctx.stream().handle())).map_err(Error::Cuda)?;
        check(
            api,
            (api.set_tensor)(xd.raw, if to_first { 1 } else { 0 }, 9, ni, ci, hi, wi),
        )
        .map_err(Error::Cuda)?;
        check(
            api,
            (api.set_tensor)(yd.raw, if to_first { 0 } else { 1 }, 9, ni, ci, hi, wi),
        )
        .map_err(Error::Cuda)?;
        let one = 1.0f32;
        let zero = 0.0f32;
        check(
            api,
            (api.transform)(
                handle.raw,
                (&one as *const f32).cast(),
                xd.raw,
                gpu_ptr(x)?,
                (&zero as *const f32).cast(),
                yd.raw,
                out.ptr(),
            ),
        )
        .map_err(Error::Cuda)?;
    }
    let shape = if to_first {
        vec![n, c, h, w]
    } else {
        vec![n, h, w, c]
    };
    Ok(out.into_tensor(Shape::new(shape), DType::BF16))
}

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
    let out_buf = output_buffer(ctx, out_bytes)?;

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

/// Broadcast [N,C,1,1] into a contiguous NCHW feature map.
pub fn expand_spatial_bf16(
    ctx: &CudaContext,
    x: &Tensor,
    height: usize,
    width: usize,
) -> Result<Tensor> {
    let d = x.shape().dims();
    if d.len() != 4
        || d[2..] != [1, 1]
        || x.dtype() != DType::BF16
        || x.device() != Device::Cuda(ctx.device_id())
    {
        return Err(Error::Other(
            "spatial expansion requires CUDA BF16 [N,C,1,1]".into(),
        ));
    }
    let shape = vec![d[0], d[1], height, width];
    let bytes = checked_bytes(DType::BF16, &shape, "spatial expansion")?;
    let spatial = i32::try_from(
        height
            .checked_mul(width)
            .ok_or_else(|| Error::Other("spatial overflow".into()))?,
    )
    .map_err(|_| Error::Other("spatial dimension overflow".into()))?;
    let count =
        i64::try_from(bytes / 2).map_err(|_| Error::Other("spatial output overflow".into()))?;
    let out = output_buffer(ctx, bytes)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_expand_spatial_bf16(
            gpu_ptr(x)?,
            out.ptr(),
            spatial,
            count,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        Shape::new(shape),
        DType::BF16,
        ctx.device_id(),
        out,
    ))
}

/// Concatenate NCHW feature maps along channels using device-to-device copies.
pub fn concat_channels_bf16(ctx: &CudaContext, inputs: &[&Tensor]) -> Result<Tensor> {
    let first = inputs
        .first()
        .ok_or_else(|| Error::Other("empty channel concatenation".into()))?;
    let d = first.shape().dims();
    if d.len() != 4 {
        return Err(Error::Other("channel concatenation requires NCHW".into()));
    }
    let mut channels = 0usize;
    for t in inputs {
        let s = t.shape().dims();
        if s.len() != 4
            || s[0] != d[0]
            || s[2..] != d[2..]
            || t.dtype() != DType::BF16
            || t.device() != Device::Cuda(ctx.device_id())
        {
            return Err(Error::Other(
                "channel concatenation shape/device mismatch".into(),
            ));
        }
        checked_bytes(DType::BF16, s, "channel concatenation input")?;
        channels = channels
            .checked_add(s[1])
            .ok_or_else(|| Error::Other("channel overflow".into()))?;
    }
    let shape = vec![d[0], channels, d[2], d[3]];
    let bytes = checked_bytes(DType::BF16, &shape, "channel concatenation output")?;
    let out = output_buffer(ctx, bytes)?;
    let pitch = bytes / d[0];
    let mut offset = 0;
    for t in inputs {
        let source_pitch = t.size_in_bytes() / d[0];
        crate::transfers::copy_tensor_2d_to_buffer(
            ctx,
            t,
            &out,
            offset,
            pitch,
            source_pitch,
            source_pitch,
            d[0],
        )?;
        offset += source_pitch;
    }
    Ok(make_gpu_tensor(
        Shape::new(shape),
        DType::BF16,
        ctx.device_id(),
        out,
    ))
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

pub fn gather_rows_bf16(
    ctx: &CudaContext,
    input: &Tensor,
    indices: &CudaBuffer,
    rows: usize,
) -> Result<Tensor> {
    let (input_rows, cols) = matrix_shape(input, "row gather")?;
    if input.dtype() != DType::BF16 || rows == 0 || rows > input_rows {
        return Err(Error::Other(
            "static inference BF16 row gather has incompatible shape".into(),
        ));
    }
    require_buffers(
        ctx,
        "row gather",
        &[("indices", indices, rows * std::mem::size_of::<u32>())],
    )?;
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_gather_rows_bf16(
            gpu_ptr(input)?,
            indices.ptr(),
            output.ptr(),
            rows as i32,
            cols as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(matrix_tensor(ctx, rows, cols, output))
}

/// Replace selected rows according to a device `u32` map. `u32::MAX` keeps
/// the base row; every other value selects a row from `replacement`.
pub fn replace_rows_bf16(
    ctx: &CudaContext,
    base: &Tensor,
    replacement: &Tensor,
    row_map: &CudaBuffer,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(base, "row replacement")?;
    let (_, replacement_cols) = matrix_shape(replacement, "row replacement")?;
    if base.dtype() != DType::BF16 || replacement.dtype() != DType::BF16 || cols != replacement_cols
    {
        return Err(Error::Other(
            "static inference BF16 row replacement has incompatible shape".into(),
        ));
    }
    require_buffers(
        ctx,
        "row replacement",
        &[("row_map", row_map, rows * std::mem::size_of::<u32>())],
    )?;
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_replace_rows_bf16(
            gpu_ptr(base)?,
            gpu_ptr(replacement)?,
            row_map.ptr(),
            output.ptr(),
            rows as i32,
            cols as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(matrix_tensor(ctx, rows, cols, output))
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
