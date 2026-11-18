//! Cache operations — reserve_prefix and concat_rows.
//!
//! These manage prefix KV cache tensors via device memory allocation + D2D copies.

use apxinf_core::{DType, Device, Error, Result, Shape, Storage, Tensor};

use crate::l3::context::AscendNewContext;

/// Allocate a persistent `[total_rows, cols]` cache, zero-initialize it,
/// then copy `prefix` (`[prefix_rows, cols]`) into the first rows.
pub fn reserve_prefix(ctx: &AscendNewContext, prefix: &Tensor, total_rows: usize) -> Result<Tensor> {
    let dtype = prefix.dtype();
    let prefix_dims = prefix.shape().dims();
    let cols = prefix_dims[prefix_dims.len() - 1];
    let prefix_rows = prefix_dims[0];

    if prefix_rows > total_rows {
        return Err(Error::Other(format!(
            "reserve_prefix: prefix_rows({prefix_rows}) > total_rows({total_rows})"
        )));
    }

    // Allocate [total_rows, cols] zero-initialized
    let cache_shape = Shape::new(vec![total_rows, cols]);
    let cache = ctx.allocate_output(cache_shape.clone(), dtype)?;

    // Copy prefix into the first rows
    let cache_ptr = match cache.storage() {
        Storage::Gpu { handle, .. } => handle.ptr(),
        _ => return Err(Error::Other("reserve_prefix: cache not on device".into())),
    };
    let prefix_ptr = match prefix.storage() {
        Storage::Gpu { handle, .. } => handle.ptr(),
        _ => return Err(Error::Other("reserve_prefix: prefix not on device".into())),
    };

    let copy_numel = prefix_rows * cols;
    if copy_numel > 0 {
        super::aclnn_d2d_copy(cache_ptr, prefix_ptr, copy_numel, dtype, ctx.raw_stream())?;
    }

    Ok(cache)
}

/// Concatenate two rank-2 tensors along the row dimension.
/// Both must have the same number of columns and dtype.
pub fn concat_rows(ctx: &AscendNewContext, first: &Tensor, second: &Tensor) -> Result<Tensor> {
    let dtype = first.dtype();
    let first_dims = first.shape().dims();
    let second_dims = second.shape().dims();
    let cols = first_dims[first_dims.len() - 1];
    let first_rows = first_dims[0];
    let second_rows = second_dims[0];

    if second_dims[second_dims.len() - 1] != cols {
        return Err(Error::Other(format!(
            "concat_rows: column mismatch {} vs {}",
            cols, second_dims[second_dims.len() - 1]
        )));
    }
    if second.dtype() != dtype {
        return Err(Error::Other("concat_rows: dtype mismatch".into()));
    }

    let total_rows = first_rows + second_rows;
    let out_shape = Shape::new(vec![total_rows, cols]);
    let out = ctx.allocate_output(out_shape.clone(), dtype)?;

    let out_ptr = match out.storage() {
        Storage::Gpu { handle, .. } => handle.ptr(),
        _ => return Err(Error::Other("concat_rows: out not on device".into())),
    };
    let first_ptr = match first.storage() {
        Storage::Gpu { handle, .. } => handle.ptr(),
        _ => return Err(Error::Other("concat_rows: first not on device".into())),
    };
    let second_ptr = match second.storage() {
        Storage::Gpu { handle, .. } => handle.ptr(),
        _ => return Err(Error::Other("concat_rows: second not on device".into())),
    };

    let first_numel = first_rows * cols;
    let second_numel = second_rows * cols;
    let elem_size = dtype.size_in_bytes();

    // Copy first tensor via aclnnInplaceCopy (RI-capture compatible)
    if first_numel > 0 {
        super::aclnn_d2d_copy(out_ptr, first_ptr, first_numel, dtype, ctx.raw_stream())?;
    }

    // Copy second tensor
    if second_numel > 0 {
        super::aclnn_d2d_copy(
            out_ptr + first_numel * elem_size,
            second_ptr,
            second_numel,
            dtype,
            ctx.raw_stream(),
        )?;
    }

    Ok(out)
}
