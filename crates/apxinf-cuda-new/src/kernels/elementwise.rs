//! Legacy `kernels::elementwise` names over cuda-new operators.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::ffi::abi::{elementwise as abi, status};
use crate::{ops, CudaBuffer, CudaContext};

fn invalid(message: &str) -> Error {
    Error::Other(message.into())
}

/// `add`: elementwise `a + b` into a fresh tensor.
pub fn add(ctx: &CudaContext, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let output = ctx.allocate_output(Shape::new(a.shape().dims().to_vec()), DType::BF16)?;
    ops::elementwise_add(ctx, a, b, &output)?;
    Ok(output)
}

/// `scale`: elementwise `input * factor` into a fresh tensor.
pub fn scale(ctx: &CudaContext, input: &Tensor, factor: f32) -> Result<Tensor> {
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
