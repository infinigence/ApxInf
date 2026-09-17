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

/// Concatenate equally tall BF16 matrices along their column dimension.
///
/// Unlike the portable backend packing helper, this execution-time variant
/// allocates from the active graph workspace and is therefore safe to use
/// while a fixed-shape CUDA graph is being captured.
pub fn concat_columns_bf16(ctx: &CudaContext, tensors: &[&Tensor]) -> Result<Tensor> {
    let first = tensors
        .first()
        .ok_or_else(|| Error::Other("column concatenation requires at least one tensor".into()))?;
    let (rows, first_cols) = matrix_shape(first, "column concatenation")?;
    if first.dtype() != DType::BF16 || rows == 0 || first_cols == 0 {
        return Err(Error::Other(
            "column concatenation requires non-empty BF16 matrices".into(),
        ));
    }
    let expected_device = Device::Cuda(ctx.device_id());
    let mut total_cols = 0usize;
    for tensor in tensors {
        let (tensor_rows, tensor_cols) = matrix_shape(tensor, "column concatenation")?;
        if tensor.dtype() != DType::BF16 || tensor_rows != rows || tensor_cols == 0 {
            return Err(Error::Other(
                "column concatenation requires equally tall, non-empty BF16 matrices".into(),
            ));
        }
        if tensor.device() != expected_device {
            return Err(Error::DeviceMismatch {
                expected: expected_device,
                got: tensor.device(),
            });
        }
        total_cols = total_cols
            .checked_add(tensor_cols)
            .ok_or_else(|| Error::Other("column concatenation width overflow".into()))?;
    }
    let row_bytes = total_cols
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| Error::Other("column concatenation row size overflow".into()))?;
    let output = output_buffer(
        ctx,
        rows.checked_mul(row_bytes)
            .ok_or_else(|| Error::Other("column concatenation output size overflow".into()))?,
    )?;
    let mut column_offset = 0usize;
    for tensor in tensors {
        let tensor_cols = tensor.shape().dims()[1];
        let tensor_row_bytes = tensor_cols * DType::BF16.size_in_bytes();
        crate::transfers::copy_tensor_2d_to_buffer(
            ctx,
            tensor,
            &output,
            column_offset * DType::BF16.size_in_bytes(),
            row_bytes,
            tensor_row_bytes,
            tensor_row_bytes,
            rows,
        )?;
        column_offset += tensor_cols;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, total_cols]),
        DType::BF16,
        ctx.device_id(),
        output,
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

/// Applies three independent BF16 biases to equally-shaped fresh Q/K/V
/// projections in one launch. Inputs are consumed before their device storage
/// is mutated, preventing safe callers from observing aliases.
pub fn bias_qkv_in_place_bf16(
    ctx: &CudaContext,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_bias: &Tensor,
    key_bias: &Tensor,
    value_bias: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (rows, cols) = matrix_shape(&query, "fused QKV bias")?;
    let expected_device = Device::Cuda(ctx.device_id());
    for (name, tensor) in [("query", &query), ("key", &key), ("value", &value)] {
        if tensor.dtype() != DType::BF16
            || tensor.device() != expected_device
            || tensor.shape().dims() != [rows, cols]
        {
            return Err(Error::Other(format!(
                "fused QKV bias {name} must be CUDA BF16 [{rows},{cols}], got {:?} on {}",
                tensor.shape().dims(),
                tensor.device()
            )));
        }
    }
    for (name, tensor) in [
        ("query bias", query_bias),
        ("key bias", key_bias),
        ("value bias", value_bias),
    ] {
        if tensor.dtype() != DType::BF16
            || tensor.device() != expected_device
            || tensor.shape().dims() != [cols]
        {
            return Err(Error::Other(format!(
                "fused QKV bias {name} must be CUDA BF16 [{cols}], got {:?} on {}",
                tensor.shape().dims(),
                tensor.device()
            )));
        }
    }
    if cols % 4 != 0 {
        return Err(Error::Other(format!(
            "fused QKV bias width must be divisible by 4, got {cols}"
        )));
    }
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_bias_qkv_in_place_bf16(
            gpu_ptr(&query)?,
            gpu_ptr(&key)?,
            gpu_ptr(&value)?,
            gpu_ptr(query_bias)?,
            gpu_ptr(key_bias)?,
            gpu_ptr(value_bias)?,
            rows as i32,
            cols as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok((query, key, value))
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

/// Bounds-checked row-selection metadata uploaded once to a CUDA device.
#[derive(Clone)]
pub struct PreparedRowIndices {
    buffer: CudaBuffer,
    matrix_rows: usize,
    row_count: usize,
    unique: bool,
}

impl PreparedRowIndices {
    fn validate(&self, ctx: &CudaContext, matrix_rows: usize, operation: &str) -> Result<()> {
        if self.matrix_rows != matrix_rows {
            return Err(Error::Other(format!(
                "{operation} indices were prepared for {} rows, got {matrix_rows}",
                self.matrix_rows
            )));
        }
        let required_bytes = self
            .row_count
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| Error::Other(format!("{operation} index byte size overflow")))?;
        require_buffers(
            ctx,
            operation,
            &[("row indices", &self.buffer, required_bytes)],
        )
    }
}

pub fn prepare_row_indices(
    ctx: &CudaContext,
    rows: &[usize],
    input_rows: usize,
) -> Result<PreparedRowIndices> {
    if rows.is_empty() {
        return Err(Error::Other(
            "CUDA row selection requires at least one row".into(),
        ));
    }
    let indices = rows
        .iter()
        .map(|&row| {
            if row >= input_rows {
                return Err(Error::Other(format!(
                    "CUDA row index {row} is outside 0..{input_rows}"
                )));
            }
            u32::try_from(row).map_err(|_| Error::Other("CUDA row index exceeds u32".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    let required_bytes = indices
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| Error::Other("CUDA row-index byte size overflow".into()))?;
    let bytes = indices
        .iter()
        .flat_map(|index| index.to_ne_bytes())
        .collect::<Vec<_>>();
    debug_assert_eq!(bytes.len(), required_bytes);
    let buffer = CudaBuffer::alloc(required_bytes, ctx.device_id()).map_err(Error::Cuda)?;
    buffer.copy_from_host(&bytes).map_err(Error::Cuda)?;
    let unique = rows
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .len()
        == rows.len();
    Ok(PreparedRowIndices {
        buffer,
        matrix_rows: input_rows,
        row_count: rows.len(),
        unique,
    })
}

/// Gather BF16 rows using a caller-owned device `u32` index buffer.
///
/// This is the stable low-level contract used by existing model runtimes that
/// keep their row order on the GPU.
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

pub fn gather_rows_bf16_prepared(
    ctx: &CudaContext,
    input: &Tensor,
    indices: &PreparedRowIndices,
) -> Result<Tensor> {
    let (input_rows, _) = matrix_shape(input, "prepared row gather")?;
    if input.dtype() != DType::BF16 {
        return Err(Error::Other(
            "CUDA prepared row gather requires BF16 input".into(),
        ));
    }
    indices.validate(ctx, input_rows, "CUDA prepared row gather")?;
    gather_rows_bf16(ctx, input, &indices.buffer, indices.row_count)
}

pub fn scatter_rows_bf16_prepared(
    ctx: &CudaContext,
    destination: &Tensor,
    indices: &PreparedRowIndices,
    source: &Tensor,
    add: bool,
) -> Result<Tensor> {
    let (destination_rows, columns) =
        matrix_shape(destination, "prepared row scatter destination")?;
    let (source_rows, source_columns) = matrix_shape(source, "prepared row scatter source")?;
    if destination.dtype() != DType::BF16
        || source.dtype() != DType::BF16
        || source_rows != indices.row_count
        || source_columns != columns
    {
        return Err(Error::Other(format!(
            "CUDA prepared row scatter expects BF16 [{}, {columns}] source, got {} {:?}",
            indices.row_count,
            source.dtype(),
            source.shape().dims()
        )));
    }
    let expected_device = Device::Cuda(ctx.device_id());
    for tensor in [destination, source] {
        if tensor.device() != expected_device {
            return Err(Error::DeviceMismatch {
                expected: expected_device,
                got: tensor.device(),
            });
        }
    }
    indices.validate(ctx, destination_rows, "CUDA prepared row scatter")?;
    if !indices.unique {
        return Err(Error::Other(
            "CUDA prepared row scatter requires unique destination rows".into(),
        ));
    }
    let output = bf16_output(ctx, destination_rows, columns)?;
    unsafe {
        ffi::check_cuda(ffi::cudaMemcpyAsync(
            output.ptr(),
            gpu_ptr(destination)?,
            destination.size_in_bytes(),
            ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
        ffi::check_cuda(ffi::apxinf_static_scatter_rows_bf16(
            gpu_ptr(source)?,
            indices.buffer.ptr(),
            output.ptr(),
            i32::try_from(indices.row_count)
                .map_err(|_| Error::Other("CUDA row scatter count exceeds i32".into()))?,
            i32::try_from(columns)
                .map_err(|_| Error::Other("CUDA row scatter width exceeds i32".into()))?,
            if add { 1 } else { 0 },
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(matrix_tensor(ctx, destination_rows, columns, output))
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

/// Return a zero-copy view over contiguous rows of a CUDA matrix.
pub fn contiguous_rows(
    ctx: &CudaContext,
    input: &Tensor,
    first_row: usize,
    row_count: usize,
) -> Result<Tensor> {
    let (rows, columns) = matrix_shape(input, "contiguous row slice")?;
    let end = first_row
        .checked_add(row_count)
        .ok_or_else(|| Error::Other("CUDA row slice range overflow".into()))?;
    if row_count == 0 || end > rows {
        return Err(Error::Other(format!(
            "CUDA row slice [{first_row}..{end}] is outside 0..{rows}"
        )));
    }
    if input.device() != Device::Cuda(ctx.device_id()) {
        return Err(Error::DeviceMismatch {
            expected: Device::Cuda(ctx.device_id()),
            got: input.device(),
        });
    }
    let row_bytes = columns
        .checked_mul(input.dtype().size_in_bytes())
        .ok_or_else(|| Error::Other("CUDA row slice byte width overflow".into()))?;
    let byte_offset = first_row
        .checked_mul(row_bytes)
        .ok_or_else(|| Error::Other("CUDA row slice byte offset overflow".into()))?;
    let byte_len = row_count
        .checked_mul(row_bytes)
        .ok_or_else(|| Error::Other("CUDA row slice byte length overflow".into()))?;
    let buffer = CudaBuffer::from_tensor(input)
        .map_err(Error::Cuda)?
        .view(byte_offset, byte_len)
        .map_err(Error::Cuda)?;
    Ok(buffer.into_tensor(Shape::new(vec![row_count, columns]), input.dtype()))
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

/// Reorder a contiguous NCHW BF16 tensor to NHWC through cuDNN.
pub fn nchw_to_nhwc_bf16(ctx: &CudaContext, x: &Tensor) -> Result<Tensor> {
    image_layout_bf16(ctx, x, false)
}

/// Reorder a contiguous NHWC BF16 tensor to NCHW through cuDNN.
pub fn nhwc_to_nchw_bf16(ctx: &CudaContext, x: &Tensor) -> Result<Tensor> {
    image_layout_bf16(ctx, x, true)
}
