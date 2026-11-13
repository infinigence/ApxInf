//! Row-reduction elementwise operators: row mean, per-row argmax, and a
//! full-tensor fill, each with a fixed implementation. The arithmetic matches
//! the legacy speech kernels bit for bit: the mean sums rows sequentially in
//! f32 and the argmax breaks ties toward the smaller index.

use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::ffi::abi::{elementwise as abi, status};
use crate::ops::validation::check_cuda;
use crate::{CudaBuffer, CudaContext};

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

fn dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        _ => Err(invalid(
            "row elementwise operators support F32, F16, and BF16",
        )),
    }
}

/// The zero fill is a byte memset, so any fixed-width dtype whose zero is all
/// zero bits is supported.
fn zero_dtype_code(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        DType::I32 => Ok(3),
        DType::I8 => Ok(4),
        _ => Err(invalid(
            "fill zero supports F32, F16, BF16, I32, and I8",
        )),
    }
}

fn tensor_storage(
    ctx: &CudaContext,
    tensor: &Tensor,
    dtype: DType,
    elements: usize,
    label: &str,
) -> Result<CudaBuffer> {
    if tensor.device() != Device::Cuda(ctx.device_id()) || tensor.dtype() != dtype {
        return Err(invalid(format!("{label} device/dtype mismatch")));
    }
    if tensor.shape().numel() != elements {
        return Err(invalid(format!(
            "{label} element count mismatch: expected {elements}, got {}",
            tensor.shape().numel()
        )));
    }
    let expected = elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| invalid(format!("{label} size overflow")))?;
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    if buffer.len() < expected || (buffer.ptr() as usize) % dtype.size_in_bytes() != 0 {
        return Err(invalid(format!("{label} storage is invalid")));
    }
    Ok(buffer)
}

/// Mean over the rows of a rank-2 view, producing one row of `out`.
pub fn mean_rows(ctx: &CudaContext, input: &Tensor, out: &Tensor) -> Result<()> {
    let dtype = input.dtype();
    let code = dtype_code(dtype)?;
    if out.dtype() != dtype {
        return Err(invalid("row mean dtype mismatch"));
    }
    let dims = input.shape().dims().to_vec();
    if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
        return Err(invalid("row mean expects a non-empty rank-2 input"));
    }
    let (rows, cols) = (dims[0], dims[1]);
    if out.numel() != cols {
        return Err(invalid("row mean output must have one row"));
    }
    let input_buffer = tensor_storage(ctx, input, dtype, rows * cols, "row mean input")?;
    let out_buffer = tensor_storage(ctx, out, dtype, cols, "row mean output")?;
    unsafe {
        status::check(abi::apxinf_elementwise_mean_rows(
            input_buffer.ptr(),
            out_buffer.ptr(),
            rows as i64,
            cols as i64,
            code,
            ctx.stream().handle(),
        ))?;
    }
    Ok(())
}

/// Per-row argmax with first-index tie breaking, one I32 index per row.
pub fn argmax_rows(ctx: &CudaContext, logits: &Tensor, index: &Tensor) -> Result<()> {
    let dtype = logits.dtype();
    let code = dtype_code(dtype)?;
    let dims = logits.shape().dims().to_vec();
    if dims.len() != 2 || dims[0] == 0 || dims[1] == 0 {
        return Err(invalid("row argmax expects a non-empty rank-2 input"));
    }
    let (rows, cols) = (dims[0], dims[1]);
    if index.dtype() != DType::I32
        || index.device() != Device::Cuda(ctx.device_id())
        || index.numel() != rows
    {
        return Err(invalid("row argmax index must be one I32 per row"));
    }
    let logits_buffer = tensor_storage(ctx, logits, dtype, rows * cols, "row argmax logits")?;
    let index_buffer = tensor_storage(ctx, index, DType::I32, rows, "row argmax index")?;
    unsafe {
        status::check(abi::apxinf_elementwise_argmax_rows(
            logits_buffer.ptr(),
            index_buffer.ptr(),
            rows as i64,
            cols as i64,
            code,
            ctx.stream().handle(),
        ))?;
    }
    Ok(())
}

/// Set every element of `tensor` to zero. Zero bytes are zero in every
/// supported dtype, so this is a memset.
pub fn fill_zero(ctx: &CudaContext, tensor: &Tensor) -> Result<()> {
    let dtype = tensor.dtype();
    let code = zero_dtype_code(dtype)?;
    if tensor.numel() == 0 {
        return Ok(());
    }
    if tensor.device() != Device::Cuda(ctx.device_id()) {
        return Err(invalid("fill zero device mismatch"));
    }
    let buffer = tensor_storage(ctx, tensor, dtype, tensor.numel(), "fill zero")?;
    unsafe {
        check_cuda(abi::apxinf_elementwise_fill_zero(
            buffer.ptr(),
            tensor.numel() as i64,
            code,
            ctx.stream().handle(),
        ))?;
    }
    Ok(())
}
