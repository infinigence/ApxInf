//! Legacy `kernels::elementwise` names over cuda-new operators.

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use crate::ffi::abi::{elementwise as abi, status};
use crate::{ops, CudaBuffer, CudaContext};

use super::contracts::{bf16_output, matrix_shape, matrix_tensor};

fn invalid(message: &str) -> Error {
    Error::Other(message.into())
}

/// `add`: elementwise `a + b` into a fresh tensor. Dispatches on dtype like
/// the legacy helper: BF16 routes to the cuda-new operator, F16 takes the
/// ported smolvla launcher (the Xavier FP16 path).
pub fn add(ctx: &CudaContext, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.dtype() == DType::F16 {
        use super::contracts::{check_cuda, gpu_ptr, make_gpu_tensor};
        use crate::ffi::abi::smolvla as abi;
        if b.dtype() != DType::F16 || b.shape() != a.shape() {
            return Err(invalid("FP16 add requires matching FP16 tensors"));
        }
        let output = crate::workspace::output_buffer(ctx, a.size_in_bytes())?;
        unsafe {
            check_cuda(abi::apxinf_smolvla_add_f16(
                gpu_ptr(a)?,
                gpu_ptr(b)?,
                output.ptr(),
                u32::try_from(a.numel())
                    .map_err(|_| invalid("FP16 add element count exceeds u32"))?,
                ctx.stream().handle(),
            ))?;
        }
        return Ok(make_gpu_tensor(
            a.shape().clone(),
            DType::F16,
            ctx.device_id(),
            output,
        ));
    }
    let output = ctx.allocate_output(Shape::new(a.shape().dims().to_vec()), DType::BF16)?;
    ops::elementwise_add(ctx, a, b, &output)?;
    Ok(output)
}

/// `scale`: elementwise `input * factor` into a fresh tensor. Dispatches on
/// dtype like the legacy helper: BF16 routes to the cuda-new operator, F16
/// takes the ported smolvla launcher (the Xavier FP16 path).
pub fn scale(ctx: &CudaContext, input: &Tensor, factor: f32) -> Result<Tensor> {
    if input.dtype() == DType::F16 {
        use super::contracts::{check_cuda, gpu_ptr, make_gpu_tensor};
        use crate::ffi::abi::smolvla as abi;
        if !factor.is_finite() {
            return Err(invalid("scale factor must be finite"));
        }
        let output = crate::workspace::output_buffer(ctx, input.size_in_bytes())?;
        unsafe {
            check_cuda(abi::apxinf_smolvla_scale_f16(
                gpu_ptr(input)?,
                output.ptr(),
                u32::try_from(input.numel())
                    .map_err(|_| invalid("FP16 scale element count exceeds u32"))?,
                factor,
                ctx.stream().handle(),
            ))?;
        }
        return Ok(make_gpu_tensor(
            input.shape().clone(),
            DType::F16,
            ctx.device_id(),
            output,
        ));
    }
    let output = ctx.allocate_output(Shape::new(input.shape().dims().to_vec()), DType::BF16)?;
    ops::elementwise_scale(ctx, input, &output, factor)?;
    Ok(output)
}

/// `bias_bf16`: broadcast-add an optional `[cols]` bias over rows. A missing
/// bias returns the input unchanged, matching the legacy contract.
pub fn bias_bf16(ctx: &CudaContext, input: &Tensor, value: Option<&Tensor>) -> Result<Tensor> {
    let Some(bias) = value else {
        return Ok(input.clone());
    };
    let output = ctx.allocate_output(Shape::new(input.shape().dims().to_vec()), DType::BF16)?;
    ops::elementwise_add_bias(ctx, input, bias, &output)?;
    Ok(output)
}

/// `concat_rows_bf16`: stack two row-major matrices with equal columns.
pub fn concat_rows_bf16(ctx: &CudaContext, first: &Tensor, second: &Tensor) -> Result<Tensor> {
    ops::concat_rows(ctx, first, second)
}

/// `gather_rows_bf16`: gather `rows` whole rows of `input` by u32 indices
/// held in a device buffer.
pub fn gather_rows_bf16(
    ctx: &CudaContext,
    input: &Tensor,
    indices: &CudaBuffer,
    rows: usize,
) -> Result<Tensor> {
    let dims = input.shape().dims();
    if dims.len() != 2 || input.dtype() != DType::BF16 || rows == 0 || rows > dims[0] {
        return Err(Error::Other("row gather has incompatible shape".into()));
    }
    let cols = dims[1];
    let output = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::BF16)?;
    let input_buffer = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    let rows_i64 = i64::try_from(rows).map_err(|_| Error::Other("gather rows exceed i64".into()))?;
    let cols_i64 = i64::try_from(cols).map_err(|_| Error::Other("gather cols exceed i64".into()))?;
    unsafe {
        status::check(abi::apxinf_elementwise_gather_rows_bf16(
            input_buffer.ptr(),
            indices.ptr(),
            output_buffer.ptr(),
            rows_i64,
            cols_i64,
            ctx.stream().handle(),
        ))?;
    }
    Ok(output)
}

/// `replace_rows_bf16`: `output[r] = row_map[r] == u32::MAX ? base[r]
/// : replacement[row_map[r]]`.
pub fn replace_rows_bf16(
    ctx: &CudaContext,
    base: &Tensor,
    replacement: &Tensor,
    row_map: &CudaBuffer,
) -> Result<Tensor> {
    let dims = base.shape().dims();
    let replacement_dims = replacement.shape().dims();
    if dims.len() != 2
        || replacement_dims.len() != 2
        || dims[1] != replacement_dims[1]
        || base.dtype() != DType::BF16
        || replacement.dtype() != DType::BF16
    {
        return Err(invalid("row replacement has incompatible shapes"));
    }
    let (rows, cols) = (dims[0], dims[1]);
    let output = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::BF16)?;
    let base_buffer = CudaBuffer::from_tensor(base).map_err(apxinf_core::Error::Cuda)?;
    let replacement_buffer =
        CudaBuffer::from_tensor(replacement).map_err(apxinf_core::Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(apxinf_core::Error::Cuda)?;
    let rows_i64 =
        i64::try_from(rows).map_err(|_| invalid("row replacement rows exceed i64"))?;
    let cols_i64 =
        i64::try_from(cols).map_err(|_| invalid("row replacement cols exceed i64"))?;
    unsafe {
        status::check(abi::apxinf_elementwise_replace_rows_bf16(
            base_buffer.ptr(),
            replacement_buffer.ptr(),
            row_map.ptr(),
            output_buffer.ptr(),
            rows_i64,
            cols_i64,
            ctx.stream().handle(),
        ))?;
    }
    Ok(output)
}

/// `euler_update_bf16`: `output = state + velocity * dt`, the flow-matching
/// integration step.
pub fn euler_update_bf16(
    ctx: &CudaContext,
    state: &Tensor,
    velocity: &Tensor,
    dt: f32,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(Shape::new(state.shape().dims().to_vec()), DType::BF16)?;
    let mut args =
        ops::PointwiseArgs::new(ops::PointwiseSemantic::EulerUpdate, state, &mut output);
    args.secondary = Some(velocity);
    args.dt = dt;
    ops::pointwise(ctx, args)?;
    Ok(output)
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
        if self.buffer.device() != ctx.device_id() {
            return Err(Error::Other(format!(
                "{operation} row indices are on CUDA{}, expected CUDA{}",
                self.buffer.device(),
                ctx.device_id()
            )));
        }
        if self.buffer.len() < required_bytes {
            return Err(Error::Other(format!(
                "{operation} row indices require {required_bytes} bytes, has {}",
                self.buffer.len()
            )));
        }
        Ok(())
    }
}

/// Upload a bounds-checked `u32` row-index table to the device once, so a
/// repeated gather/scatter can reuse it across launches.
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

/// `gather_rows_bf16_prepared`: gather rows using prepared index metadata.
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

/// `scatter_rows_bf16_prepared`: copy `destination` into a fresh output, then
/// scatter `source` rows to the prepared destinations. `add` selects
/// accumulate-vs-overwrite.
pub fn scatter_rows_bf16_prepared(
    ctx: &CudaContext,
    destination: &Tensor,
    indices: &PreparedRowIndices,
    source: &Tensor,
    add: bool,
) -> Result<Tensor> {
    use crate::ffi::abi::gr00t as gr00t;
    use crate::ffi::raw::cuda_runtime as raw;
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
    let destination_buffer = CudaBuffer::from_tensor(destination).map_err(Error::Cuda)?;
    let source_buffer = CudaBuffer::from_tensor(source).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(raw::cudaMemcpyAsync(
            output.ptr(),
            destination_buffer.ptr(),
            destination.size_in_bytes(),
            raw::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
        raw::check_cuda(gr00t::apxinf_gr00t_scatter_rows_bf16(
            source_buffer.ptr(),
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

/// `contiguous_rows`: a zero-copy view over contiguous rows of a CUDA matrix.
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

/// `concat_columns_bf16`: concatenate equally tall BF16 matrices along their
/// column dimension with a 2D strided device copy per input.
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
    let output = bf16_output(
        ctx,
        rows,
        total_cols,
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
    Ok(matrix_tensor(ctx, rows, total_cols, output))
}

/// `bias_qkv_in_place_bf16`: apply three independent BF16 biases to equally
/// shaped fresh Q/K/V projections in a single launch, mutating them in place.
/// Widths must be divisible by four.
pub fn bias_qkv_in_place_bf16(
    ctx: &CudaContext,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_bias: &Tensor,
    key_bias: &Tensor,
    value_bias: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    use crate::ffi::abi::gr00t as gr00t;
    use crate::ffi::raw::cuda_runtime as raw;
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
    let query_buffer = CudaBuffer::from_tensor(&query).map_err(Error::Cuda)?;
    let key_buffer = CudaBuffer::from_tensor(&key).map_err(Error::Cuda)?;
    let value_buffer = CudaBuffer::from_tensor(&value).map_err(Error::Cuda)?;
    let query_bias_buffer = CudaBuffer::from_tensor(query_bias).map_err(Error::Cuda)?;
    let key_bias_buffer = CudaBuffer::from_tensor(key_bias).map_err(Error::Cuda)?;
    let value_bias_buffer = CudaBuffer::from_tensor(value_bias).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(gr00t::apxinf_gr00t_bias_qkv_in_place_bf16(
            query_buffer.ptr(),
            key_buffer.ptr(),
            value_buffer.ptr(),
            query_bias_buffer.ptr(),
            key_bias_buffer.ptr(),
            value_bias_buffer.ptr(),
            rows as i32,
            cols as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok((query, key, value))
}

// ── smolvla FP16 surface ────────────────────────────────────────────────────
//
// Ported verbatim from the legacy `kernels::elementwise` FP16 wrappers over
// the `apxinf_smolvla_*` launchers (native/adapters/smolvla/ops.cu).

/// `bias_f16`: broadcast-add an optional `[cols]` FP16 bias over rows.
pub fn bias_f16(ctx: &CudaContext, input: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    use super::contracts::{check_cuda, f16_output, gpu_ptr, make_gpu_tensor, optional_ptr};
    use crate::ffi::abi::smolvla as abi;
    let (rows, cols) = matrix_shape(input, "bias")?;
    if input.dtype() != DType::F16
        || bias.is_some_and(|x| x.dtype() != DType::F16 || x.shape().dims() != [cols])
    {
        return Err(invalid(
            "static inference bias expects an FP16 matrix and matching bias",
        ));
    }
    let output = f16_output(ctx, rows, cols)?;
    unsafe {
        check_cuda(abi::apxinf_smolvla_bias_f16(
            gpu_ptr(input)?,
            optional_ptr(bias)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            ctx.stream().handle(),
        ))?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::F16,
        ctx.device_id(),
        output,
    ))
}

/// `concat_rows_f16`: stack two row-major FP16 matrices with equal columns.
/// Routes to the dtype-generic cuda-new `concat_rows` layout semantic (two
/// device copies), which is byte-exact.
pub fn concat_rows_f16(ctx: &CudaContext, first: &Tensor, second: &Tensor) -> Result<Tensor> {
    if first.dtype() != DType::F16 || second.dtype() != DType::F16 {
        return Err(invalid(
            "static inference row concatenation expects FP16 matrices",
        ));
    }
    ops::concat_rows(ctx, first, second)
}

/// `concat_columns_f16`: concatenate equally tall FP16 matrices along their
/// column dimension with a 2D strided device copy per input.
pub fn concat_columns_f16(ctx: &CudaContext, tensors: &[&Tensor]) -> Result<Tensor> {
    use super::contracts::output_tensor;
    let first = tensors
        .first()
        .ok_or_else(|| invalid("column concatenation requires at least one tensor"))?;
    let (rows, first_cols) = matrix_shape(first, "column concatenation")?;
    if first.dtype() != DType::F16 || rows == 0 || first_cols == 0 {
        return Err(invalid(
            "column concatenation requires non-empty FP16 matrices",
        ));
    }
    let expected_device = Device::Cuda(ctx.device_id());
    let mut total_cols = 0usize;
    for tensor in tensors {
        let (tensor_rows, tensor_cols) = matrix_shape(tensor, "column concatenation")?;
        if tensor.dtype() != DType::F16 || tensor_rows != rows || tensor_cols == 0 {
            return Err(invalid(
                "column concatenation requires equally tall, non-empty FP16 matrices",
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
            .ok_or_else(|| invalid("column concatenation width overflow"))?;
    }
    let row_bytes = total_cols
        .checked_mul(DType::F16.size_in_bytes())
        .ok_or_else(|| invalid("column concatenation row size overflow"))?;
    let output = crate::workspace::output_buffer(
        ctx,
        rows.checked_mul(row_bytes)
            .ok_or_else(|| invalid("column concatenation output size overflow"))?,
    )?;
    let mut column_offset = 0usize;
    for tensor in tensors {
        let tensor_cols = tensor.shape().dims()[1];
        let tensor_row_bytes = tensor_cols * DType::F16.size_in_bytes();
        crate::transfers::copy_tensor_2d_to_buffer(
            ctx,
            tensor,
            &output,
            column_offset * DType::F16.size_in_bytes(),
            row_bytes,
            tensor_row_bytes,
            tensor_row_bytes,
            rows,
        )?;
        column_offset += tensor_cols;
    }
    Ok(output_tensor(
        ctx,
        Shape::new(vec![rows, total_cols]),
        DType::F16,
        output,
    ))
}

/// `euler_update_f16`: `output = state + velocity * dt` in FP16.
pub fn euler_update_f16(
    ctx: &CudaContext,
    state: &Tensor,
    velocity: &Tensor,
    dt: f32,
) -> Result<Tensor> {
    use super::contracts::{check_cuda, gpu_ptr, make_gpu_tensor};
    use crate::ffi::abi::smolvla as abi;
    if state.dtype() != DType::F16
        || velocity.dtype() != DType::F16
        || state.shape() != velocity.shape()
    {
        return Err(invalid(
            "static inference Euler update expects matching FP16 tensors",
        ));
    }
    let output = crate::workspace::output_buffer(ctx, state.size_in_bytes())?;
    unsafe {
        check_cuda(abi::apxinf_smolvla_euler_update_f16(
            gpu_ptr(state)?,
            gpu_ptr(velocity)?,
            output.ptr(),
            state.numel() as i64,
            dt,
            ctx.stream().handle(),
        ))?;
    }
    Ok(make_gpu_tensor(
        state.shape().clone(),
        DType::F16,
        ctx.device_id(),
        output,
    ))
}

// ── Decode fast-path `_into` surface (caller-owned buffers, graph-safe) ───

/// Allocation-free elementwise add into caller-owned decode storage.
pub fn add_into(
    ctx: &CudaContext,
    dtype: DType,
    a: &CudaBuffer,
    b: &CudaBuffer,
    output: &CudaBuffer,
    count: usize,
) -> Result<()> {
    use super::contracts::{check_cuda, checked_bytes, require_buffers};
    use crate::ffi::abi::llm_decode as llmdec;
    let bytes = checked_bytes(dtype, &[count], "decode add")?;
    require_buffers(
        ctx,
        "decode add",
        &[("A", a, bytes), ("B", b, bytes), ("output", output, bytes)],
    )?;
    let status = unsafe {
        match dtype {
            DType::F32 => llmdec::apxinf_llmdec_add_f32(
                a.ptr(),
                b.ptr(),
                output.ptr(),
                count as u32,
                ctx.stream().handle(),
            ),
            DType::BF16 => llmdec::apxinf_llmdec_add_bf16(
                a.ptr(),
                b.ptr(),
                output.ptr(),
                count as u32,
                ctx.stream().handle(),
            ),
            dtype => {
                return Err(Error::Other(format!(
                    "decode add does not support {dtype}"
                )))
            }
        }
    };
    check_cuda(status)
}

/// Allocation-free elementwise multiply into caller-owned decode storage.
pub fn mul_into(
    ctx: &CudaContext,
    dtype: DType,
    a: &CudaBuffer,
    b: &CudaBuffer,
    output: &CudaBuffer,
    count: usize,
) -> Result<()> {
    use super::contracts::{check_cuda, checked_bytes, require_buffers};
    use crate::ffi::abi::llm_decode as llmdec;
    let bytes = checked_bytes(dtype, &[count], "decode multiply")?;
    require_buffers(
        ctx,
        "decode multiply",
        &[("A", a, bytes), ("B", b, bytes), ("output", output, bytes)],
    )?;
    let status = unsafe {
        match dtype {
            DType::F32 => llmdec::apxinf_llmdec_mul_f32(
                a.ptr(),
                b.ptr(),
                output.ptr(),
                count as u32,
                ctx.stream().handle(),
            ),
            DType::BF16 => llmdec::apxinf_llmdec_mul_bf16(
                a.ptr(),
                b.ptr(),
                output.ptr(),
                count as u32,
                ctx.stream().handle(),
            ),
            dtype => {
                return Err(Error::Other(format!(
                    "decode multiply does not support {dtype}"
                )))
            }
        }
    };
    check_cuda(status)
}
