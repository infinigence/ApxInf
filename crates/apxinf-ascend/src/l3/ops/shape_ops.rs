//! Shape and manipulation operations — Transpose, Cast, Cat, Expand, Sub, Div, Clamp.
//!
//! Uses `aclnnPermute`, `aclnnCast`, `aclnnCat`, `aclnnExpand`,
//! `aclnnSub`, `aclnnDiv`, `aclnnClamp`.
//! These cover the P0/P1 operator gaps identified by CANN profiling.

use apxinf_core::{DType, Error, Result, Storage, Tensor};

use crate::l3::context::AscendNewContext;

// ── device_ptr helper ─────────────────────────────────────────────────

fn device_ptr(tensor: &Tensor) -> Result<usize> {
    match tensor.storage() {
        Storage::Gpu { handle, .. } => Ok(handle.ptr()),
        _ => Err(Error::Other(format!(
            "expected NPU tensor, got {:?}",
            tensor.device()
        ))),
    }
}

// ── Transpose / Permute ───────────────────────────────────────────────

/// Permute (transpose) tensor dimensions.
///
/// `dims` is the permutation of dimensions, e.g. `[0, 2, 1, 3]` for
/// swapping dims 1 and 2 in a 4D tensor.
pub fn transpose(
    ctx: &AscendNewContext,
    input: &Tensor,
    dims: &[i64],
    out: &mut Tensor,
) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = input.dtype();

    let input_ptr = device_ptr(input)?;
    let out_ptr = device_ptr(out)?;

    let acl_input =
        crate::acl_tensor::AclTensor::from_device(input_ptr, input.shape(), dtype);
    let acl_out =
        crate::acl_tensor::AclTensor::from_device(out_ptr, out.shape(), dtype);
    let acl_dims = crate::acl_tensor::AclIntArray::new(dims);

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnPermuteGetWorkspaceSize(
                acl_input.raw(),
                acl_dims.raw(),
                acl_out.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnPermute(ws, ws_size, exec, s)
        },
        stream,
        "aclnnPermute",
    )
    .map_err(|e| Error::Other(format!("transpose: {e}")))
}

// ── Cast (type conversion) ────────────────────────────────────────────

/// Cast tensor to a different dtype.
pub fn cast(
    ctx: &AscendNewContext,
    input: &Tensor,
    target_dtype: DType,
    out: &mut Tensor,
) -> Result<()> {
    let stream = ctx.raw_stream();
    let input_dtype = input.dtype();

    let input_ptr = device_ptr(input)?;
    let out_ptr = device_ptr(out)?;

    let acl_input =
        crate::acl_tensor::AclTensor::from_device(input_ptr, input.shape(), input_dtype);
    let acl_out =
        crate::acl_tensor::AclTensor::from_device(out_ptr, out.shape(), target_dtype);
    let acl_target = crate::acl_tensor::acl_dtype(target_dtype);

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnCastGetWorkspaceSize(
                acl_input.raw(),
                acl_target,
                acl_out.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnCast(ws, ws_size, exec, s)
        },
        stream,
        "aclnnCast",
    )
    .map_err(|e| Error::Other(format!("cast: {e}")))
}

// ── Cat (concatenate) ─────────────────────────────────────────────────

/// Concatenate tensors along a dimension.
pub fn cat(
    ctx: &AscendNewContext,
    tensors: &[&Tensor],
    dim: i64,
    out: &mut Tensor,
) -> Result<()> {
    if tensors.is_empty() {
        return Err(Error::Other("cat: empty tensor list".into()));
    }

    let stream = ctx.raw_stream();
    let dtype = tensors[0].dtype();

    let out_ptr = device_ptr(out)?;
    let acl_out =
        crate::acl_tensor::AclTensor::from_device(out_ptr, out.shape(), dtype);

    // Build aclTensor list
    let acl_tensors: Vec<crate::acl_tensor::AclTensor> = tensors
        .iter()
        .map(|t| {
            let ptr = device_ptr(t).unwrap();
            crate::acl_tensor::AclTensor::from_device(ptr, t.shape(), dtype)
        })
        .collect();
    let raw_ptrs: Vec<*const crate::ffi::aclnn::aclTensor> =
        acl_tensors.iter().map(|t| t.raw() as *const _).collect();
    let acl_list = crate::acl_tensor::AclTensorList::new(&raw_ptrs);

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnCatGetWorkspaceSize(
                acl_list.raw(),
                dim,
                acl_out.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnCat(ws, ws_size, exec, s)
        },
        stream,
        "aclnnCat",
    )
    .map_err(|e| Error::Other(format!("cat: {e}")))
}

// ── Expand (broadcast) ────────────────────────────────────────────────

/// Expand tensor to a target shape (broadcasting).
pub fn expand(
    ctx: &AscendNewContext,
    input: &Tensor,
    target_shape: &[i64],
    out: &mut Tensor,
) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = input.dtype();

    let input_ptr = device_ptr(input)?;
    let out_ptr = device_ptr(out)?;

    let acl_input =
        crate::acl_tensor::AclTensor::from_device(input_ptr, input.shape(), dtype);
    let acl_out =
        crate::acl_tensor::AclTensor::from_device(out_ptr, out.shape(), dtype);
    let acl_shape = crate::acl_tensor::AclIntArray::new(target_shape);

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnExpandGetWorkspaceSize(
                acl_input.raw(),
                acl_shape.raw(),
                acl_out.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnExpand(ws, ws_size, exec, s)
        },
        stream,
        "aclnnExpand",
    )
    .map_err(|e| Error::Other(format!("expand: {e}")))
}

// ── Sub (subtraction) ─────────────────────────────────────────────────

/// Element-wise subtraction: out = self - alpha * other.
pub fn sub(
    ctx: &AscendNewContext,
    lhs: &Tensor,
    rhs: &Tensor,
    alpha: f32,
    out: &mut Tensor,
) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = lhs.dtype();
    let shape = lhs.shape().clone();

    let lhs_ptr = device_ptr(lhs)?;
    let rhs_ptr = device_ptr(rhs)?;
    let out_ptr = device_ptr(out)?;

    let acl_lhs = crate::acl_tensor::AclTensor::from_device(lhs_ptr, &shape, dtype);
    let acl_rhs = crate::acl_tensor::AclTensor::from_device(rhs_ptr, rhs.shape(), dtype);
    let acl_out = crate::acl_tensor::AclTensor::from_device(out_ptr, &shape, dtype);
    let acl_alpha = crate::acl_tensor::AclScalar::f32(alpha);

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnSubGetWorkspaceSize(
                acl_lhs.raw(),
                acl_rhs.raw(),
                acl_alpha.raw(),
                acl_out.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnSub(ws, ws_size, exec, s)
        },
        stream,
        "aclnnSub",
    )
    .map_err(|e| Error::Other(format!("sub: {e}")))
}

// ── Div (division) ────────────────────────────────────────────────────

/// Element-wise division: out = self / other.
pub fn div(
    ctx: &AscendNewContext,
    lhs: &Tensor,
    rhs: &Tensor,
    out: &mut Tensor,
) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = lhs.dtype();
    let shape = lhs.shape().clone();

    let lhs_ptr = device_ptr(lhs)?;
    let rhs_ptr = device_ptr(rhs)?;
    let out_ptr = device_ptr(out)?;

    let acl_lhs = crate::acl_tensor::AclTensor::from_device(lhs_ptr, &shape, dtype);
    let acl_rhs = crate::acl_tensor::AclTensor::from_device(rhs_ptr, rhs.shape(), dtype);
    let acl_out = crate::acl_tensor::AclTensor::from_device(out_ptr, &shape, dtype);

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnDivGetWorkspaceSize(
                acl_lhs.raw(),
                acl_rhs.raw(),
                acl_out.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnDiv(ws, ws_size, exec, s)
        },
        stream,
        "aclnnDiv",
    )
    .map_err(|e| Error::Other(format!("div: {e}")))
}

// ── Clamp (clip to [min, max]) ────────────────────────────────────────

/// Clamp tensor values to [min, max].
pub fn clamp(
    ctx: &AscendNewContext,
    input: &Tensor,
    min_val: f32,
    max_val: f32,
    out: &mut Tensor,
) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = input.dtype();
    let shape = input.shape().clone();

    let input_ptr = device_ptr(input)?;
    let out_ptr = device_ptr(out)?;

    let acl_input = crate::acl_tensor::AclTensor::from_device(input_ptr, &shape, dtype);
    let acl_out = crate::acl_tensor::AclTensor::from_device(out_ptr, &shape, dtype);
    let acl_min = crate::acl_tensor::AclScalar::f32(min_val);
    let acl_max = crate::acl_tensor::AclScalar::f32(max_val);

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnClampGetWorkspaceSize(
                acl_input.raw(),
                acl_min.raw(),
                acl_max.raw(),
                acl_out.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnClamp(ws, ws_size, exec, s)
        },
        stream,
        "aclnnClamp",
    )
    .map_err(|e| Error::Other(format!("clamp: {e}")))
}
