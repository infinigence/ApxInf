//! Shared validation helpers for caller-owned kernel buffers.
//!
//! **Every rejection here is an `Error::Other` naming the operation, the
//! argument and both sides of the comparison.** `apxinf-core` offers
//! `DTypeMismatch` and `DeviceMismatch`, and an earlier version of this file
//! used them for some checks and a formatted message for others — which meant
//! the most useful half of the message ("which operator, which operand") was
//! present only for the cases someone had got round to. Nothing in the
//! workspace matches on those variants, so the variant buys nothing here and the
//! message is the product; `apxinf-cuda`'s `contracts.rs` — the file this one
//! ported — reads the same way.

use std::ffi::c_void;

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use crate::buffer::MusaBuffer;
use crate::context::MusaContext;
use crate::error::musa_error;

/// Assert that `tensor` lives on `ctx`'s device.
///
/// Earns a function of its own because a same-shaped tensor on *another* device
/// passes every shape check and is then handed to a kernel as a raw pointer —
/// a device fault rather than a wrong answer — and because the rule was spelled
/// out per entry point, which is how two of them came to omit it entirely.
pub(super) fn require_device(
    ctx: &MusaContext,
    tensor: &Tensor,
    operation: &str,
    name: &str,
) -> Result<()> {
    let expected = Device::Musa(ctx.device_id());
    if tensor.device() != expected {
        return Err(Error::Other(format!(
            "MUSA {operation} {name} is on {}, but this context is {expected}",
            tensor.device()
        )));
    }
    Ok(())
}

/// [`require_device`] plus the dtype an entry point expects. For operands whose
/// shape the caller checks itself.
pub(super) fn require_dtype_on(
    ctx: &MusaContext,
    tensor: &Tensor,
    dtype: DType,
    operation: &str,
    name: &str,
) -> Result<()> {
    if tensor.dtype() != dtype {
        return Err(Error::Other(format!(
            "MUSA {operation} {name} expects {dtype}, got {}",
            tensor.dtype()
        )));
    }
    require_device(ctx, tensor, operation, name)
}

/// Require `tensor` to be BF16 on `ctx`'s device and shaped exactly like
/// `reference`.
///
/// The residual-style operators take one operand the caller has already
/// validated with [`require_matrix_bf16`] and a second one that only has to
/// agree with it; spelling that second check out per operator is how five
/// copies of the same three-part test appeared with slightly different
/// omissions.
pub(super) fn require_matching_bf16(
    ctx: &MusaContext,
    operation: &str,
    name: &str,
    reference: &Tensor,
    tensor: &Tensor,
) -> Result<()> {
    require_dtype_on(ctx, tensor, DType::BF16, operation, name)?;
    if tensor.shape().dims() != reference.shape().dims() {
        return Err(Error::Other(format!(
            "MUSA {operation} needs matching shapes, got {:?} and {:?}",
            reference.shape().dims(),
            tensor.shape().dims()
        )));
    }
    Ok(())
}

pub(super) fn checked_bytes(dtype: DType, dims: &[usize], operation: &str) -> Result<usize> {
    if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Err(Error::Other(format!(
            "{operation} does not support {dtype}"
        )));
    }
    if dims.iter().any(|dimension| *dimension == 0) {
        return Err(Error::Other(format!(
            "{operation} dimensions must be non-zero: {dims:?}"
        )));
    }
    dims.iter()
        .try_fold(dtype.size_in_bytes(), |bytes, dimension| {
            bytes
                .checked_mul(*dimension)
                .ok_or_else(|| Error::Other(format!("{operation} byte size overflow")))
        })
}

/// Narrow a host dimension to the width a kernel ABI takes, naming the
/// operation and the argument it came from.
///
/// Every native operator needs this on the way to its `extern "C"` boundary.
/// Each one used to spell the check out itself — as a free function, a closure
/// or inline, with the argument order varying between them — so the message and
/// the rule live here instead of in eight near-copies.
#[cfg(apxinf_musa_available)]
pub(super) fn checked_i32(operation: &str, name: &str, value: usize) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| Error::Other(format!("MUSA {operation} {name} {value} does not fit in i32")))
}

/// The same for the one ABI that takes an element count as `int64_t`.
#[cfg(apxinf_musa_available)]
pub(super) fn checked_i64(operation: &str, name: &str, value: usize) -> Result<i64> {
    i64::try_from(value)
        .map_err(|_| Error::Other(format!("MUSA {operation} {name} {value} does not fit in i64")))
}

pub(super) fn require_buffers(
    ctx: &MusaContext,
    operation: &str,
    buffers: &[(&str, &MusaBuffer, usize)],
) -> Result<()> {
    for (name, buffer, required) in buffers {
        if buffer.device() != ctx.device_id() {
            return Err(Error::Other(format!(
                "{operation} {name} is on MUSA{}, expected MUSA{}",
                buffer.device(),
                ctx.device_id()
            )));
        }
        if buffer.len() < *required {
            return Err(Error::Other(format!(
                "{operation} {name} requires {required} bytes, has {}",
                buffer.len()
            )));
        }
    }
    Ok(())
}

pub(super) fn require_finite(operation: &str, values: &[f32]) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(Error::Other(format!(
            "{operation} received a non-finite scalar"
        )))
    }
}

pub(super) fn tensor_ptr(tensor: &Tensor) -> Result<*mut c_void> {
    let buffer = MusaBuffer::from_tensor(tensor).map_err(musa_error)?;
    Ok(buffer.ptr())
}

/// [`tensor_ptr`] for an operand the native ABI treats as optional: `None`
/// becomes the null pointer the kernels document as "absent".
pub(super) fn optional_tensor_ptr(tensor: Option<&Tensor>) -> Result<*mut c_void> {
    match tensor {
        Some(tensor) => tensor_ptr(tensor),
        None => Ok(std::ptr::null_mut()),
    }
}

pub(super) fn matrix_shape(tensor: &Tensor, operation: &str) -> Result<(usize, usize)> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!(
            "MUSA {operation} expects a 2D tensor, got {dims:?}"
        )));
    }
    Ok((dims[0], dims[1]))
}

/// Require a 2-D BF16 tensor on `ctx`'s device and return its extent.
///
/// Nearly every BF16 operator opens with this, so it lives here rather than
/// being restated per operator; the error names both the operation and which
/// argument failed.
pub(super) fn require_matrix_bf16(
    ctx: &MusaContext,
    tensor: &Tensor,
    operation: &str,
    name: &str,
) -> Result<(usize, usize)> {
    let expected = Device::Musa(ctx.device_id());
    if tensor.dtype() != DType::BF16 {
        return Err(Error::Other(format!(
            "MUSA {operation} {name} expects BF16, got {}",
            tensor.dtype()
        )));
    }
    if tensor.device() != expected {
        return Err(Error::Other(format!(
            "MUSA {operation} {name} is on {}, but this context is {expected}",
            tensor.device()
        )));
    }
    matrix_shape(tensor, operation).map_err(|_| {
        Error::Other(format!(
            "MUSA {operation} {name} expects a 2D BF16 tensor, got {:?}",
            tensor.shape().dims()
        ))
    })
}

/// Require a caller-provided output to be a `[rows, cols]` BF16 tensor on
/// `ctx`'s device.
///
/// Typed operators write in place, so the extent is part of their contract
/// rather than something they can infer from an allocation they made.
pub(super) fn require_output_matrix(
    ctx: &MusaContext,
    tensor: &Tensor,
    rows: usize,
    cols: usize,
    operation: &str,
    name: &str,
) -> Result<()> {
    if require_matrix_bf16(ctx, tensor, operation, name)? != (rows, cols) {
        return Err(Error::Other(format!(
            "MUSA {operation} {name} expects [{rows}, {cols}], got {:?}",
            tensor.shape().dims()
        )));
    }
    Ok(())
}

/// Require a 1-D BF16 vector of exactly `len` elements on `ctx`'s device,
/// when the operand is one the operator accepts as absent.
pub(super) fn require_optional_vector_bf16(
    ctx: &MusaContext,
    tensor: Option<&Tensor>,
    len: usize,
    operation: &str,
    name: &str,
) -> Result<()> {
    if let Some(tensor) = tensor {
        require_vector_bf16(ctx, tensor, len, operation, name)?;
    }
    Ok(())
}

/// Require a 1-D BF16 vector of exactly `len` elements on `ctx`'s device.
pub(super) fn require_vector_bf16(
    ctx: &MusaContext,
    tensor: &Tensor,
    len: usize,
    operation: &str,
    name: &str,
) -> Result<()> {
    let dims = tensor.shape().dims();
    let expected = Device::Musa(ctx.device_id());
    if tensor.dtype() != DType::BF16 || dims != [len] || tensor.device() != expected {
        return Err(Error::Other(format!(
            "MUSA {operation} {name} expects a BF16 [{len}] vector on {expected}, got {} {:?}",
            tensor.dtype(),
            dims
        )));
    }
    Ok(())
}

/// Require a caller-provided output to be a BF16 tensor of an exact rank-N
/// extent on `ctx`'s device.
pub(super) fn require_output_tensor(
    ctx: &MusaContext,
    tensor: &Tensor,
    dims: &[usize],
    operation: &str,
    name: &str,
) -> Result<()> {
    let expected = Device::Musa(ctx.device_id());
    if tensor.dtype() != DType::BF16 || tensor.shape().dims() != dims || tensor.device() != expected
    {
        return Err(Error::Other(format!(
            "MUSA {operation} {name} expects BF16 {dims:?} on {expected}, got {} {:?} on {}",
            tensor.dtype(),
            tensor.shape().dims(),
            tensor.device()
        )));
    }
    Ok(())
}

/// Require a 1-D BF16 vector of **at least** `len` elements on `ctx`'s
/// device.
///
/// Used where a kernel reads a prefix of a longer vector the model passes
/// wholesale — PI0.5's modulation carries a third segment some operators do not
/// consume.
pub(super) fn require_vector_at_least_bf16(
    ctx: &MusaContext,
    tensor: &Tensor,
    len: usize,
    operation: &str,
    name: &str,
) -> Result<()> {
    let dims = tensor.shape().dims();
    let expected = Device::Musa(ctx.device_id());
    if tensor.dtype() != DType::BF16
        || dims.len() != 1
        || dims[0] < len
        || tensor.device() != expected
    {
        return Err(Error::Other(format!(
            "MUSA {operation} {name} expects a BF16 vector of at least {len} elements on {expected}, got {} {:?}",
            tensor.dtype(),
            dims
        )));
    }
    Ok(())
}

pub(super) fn output_tensor(
    ctx: &MusaContext,
    shape: Shape,
    dtype: DType,
    buffer: MusaBuffer,
) -> Tensor {
    debug_assert_eq!(buffer.device(), ctx.device_id());
    buffer.into_tensor(shape, dtype)
}

pub(super) fn matrix_tensor(
    ctx: &MusaContext,
    rows: usize,
    cols: usize,
    buffer: MusaBuffer,
) -> Tensor {
    output_tensor(ctx, Shape::new(vec![rows, cols]), DType::BF16, buffer)
}

pub(super) fn bf16_output(ctx: &MusaContext, rows: usize, cols: usize) -> Result<MusaBuffer> {
    crate::workspace::output_buffer(
        ctx,
        rows.checked_mul(cols)
            .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
            .ok_or_else(|| Error::Other("BF16 output size overflow".into()))?,
    )
}

/// Allocate an output of any rank from the active workspace.
///
/// The attention operators return `[tokens, heads, head_dim]`, so the 2-D
/// `bf16_output` is not enough on its own; this keeps the element-count
/// arithmetic in one place for every dtype an operator allocates.
pub(super) fn output_tensor_dims(
    ctx: &MusaContext,
    dims: &[usize],
    dtype: DType,
) -> Result<Tensor> {
    let bytes = checked_bytes(dtype, dims, "MUSA operator output")?;
    Ok(output_tensor(
        ctx,
        Shape::new(dims.to_vec()),
        dtype,
        crate::workspace::output_buffer(ctx, bytes)?,
    ))
}

/// A BF16 output of any rank, from the active workspace.
pub(super) fn bf16_tensor(ctx: &MusaContext, dims: &[usize]) -> Result<Tensor> {
    output_tensor_dims(ctx, dims, DType::BF16)
}

/// A f32 output tensor, for the operators whose intermediate results cannot be
/// narrowed to bf16 without moving the numbers. The vision attention is one:
/// its scores are a whole 72-term dot product, and rounding those to bf16 puts
/// the result past the gate the port is held to.
pub(super) fn f32_tensor(ctx: &MusaContext, dims: &[usize]) -> Result<Tensor> {
    output_tensor_dims(ctx, dims, DType::F32)
}

pub(super) fn f16_output(ctx: &MusaContext, rows: usize, cols: usize) -> Result<MusaBuffer> {
    crate::workspace::output_buffer(
        ctx,
        rows.checked_mul(cols)
            .and_then(|elements| elements.checked_mul(DType::F16.size_in_bytes()))
            .ok_or_else(|| Error::Other("FP16 output size overflow".into()))?,
    )
}

