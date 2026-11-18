//! L3 operator contracts for Ascend NPU.
//!
//! Each function mirrors `apxinf_cuda_new::ops::*` and is called by
//! PI0.5's hardware-agnostic `blocks/bf16.rs`.

pub mod attention;
pub mod cache;
pub mod gather;
pub mod gemm;
pub mod norm;
pub mod pointwise;
pub mod rope;
pub mod shape_ops;

// Re-export session helpers
pub use crate::l3::workspace::{with_session, prepare_with_session, ExecutionSession};

/// D2D copy using `aclnnInplaceCopy` (RI-capture compatible).
///
/// Replaces `aclrtMemcpy` D2D calls which are forbidden during graph capture
/// (error 107030). This goes through `run_op`, so it benefits from workspace
/// pooling and skip-sync when inside a session.
pub fn aclnn_d2d_copy(
    dst: usize,
    src: usize,
    numel: usize,
    dtype: apxinf_core::DType,
    stream: crate::ffi::acl::aclrtStream,
) -> apxinf_core::Result<()> {
    if numel == 0 {
        return Ok(());
    }
    let shape = apxinf_core::Shape::new(vec![numel, 1]);
    let acl_dst = crate::acl_tensor::AclTensor::from_device(dst, &shape, dtype);
    let acl_src = crate::acl_tensor::AclTensor::from_device(src, &shape, dtype);

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnInplaceCopyGetWorkspaceSize(
                acl_dst.raw(),
                acl_src.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnInplaceCopy(ws, ws_size, exec, s)
        },
        stream,
        "aclnnInplaceCopy (d2d)",
    )
    .map_err(|e| apxinf_core::Error::Other(format!("aclnn_d2d_copy: {e}")))
}

/// In-place add: `self_ref += alpha * other` (modifies self_ref in place).
///
/// Uses `aclnnInplaceAdd` which avoids the temp-buffer + copy-back pattern
/// of `aclnnAdd` (which requires out != self). RI-capture compatible.
pub fn aclnn_inplace_add(
    self_ref: usize,
    other: usize,
    self_shape: &apxinf_core::Shape,
    other_shape: &apxinf_core::Shape,
    dtype: apxinf_core::DType,
    stream: crate::ffi::acl::aclrtStream,
) -> apxinf_core::Result<()> {
    let acl_self = crate::acl_tensor::AclTensor::from_device(self_ref, self_shape, dtype);
    let acl_other = crate::acl_tensor::AclTensor::from_device(other, other_shape, dtype);
    let acl_alpha = crate::acl_tensor::AclScalar::f32(1.0);

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnInplaceAddGetWorkspaceSize(
                acl_self.raw(),
                acl_other.raw(),
                acl_alpha.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnInplaceAdd(ws, ws_size, exec, s)
        },
        stream,
        "aclnnInplaceAdd",
    )
    .map_err(|e| apxinf_core::Error::Other(format!("aclnn_inplace_add: {e}")))
}

/// Slice a 2D tensor `[rows, total_cols]` along dim=1 from `start` to `end`
/// into a contiguous output `[rows, end-start]` using `aclnnSlice`.
///
/// Replaces per-token D2D copy loop for QKV split (3 calls vs 3*seq_len copies).
pub fn aclnn_slice_2d(
    src_ptr: usize,
    src_shape: &apxinf_core::Shape,
    out_ptr: usize,
    out_rows: usize,
    out_cols: usize,
    start: i64,
    dtype: apxinf_core::DType,
    stream: crate::ffi::acl::aclrtStream,
) -> apxinf_core::Result<()> {
    let out_shape = apxinf_core::Shape::new(vec![out_rows, out_cols]);
    let acl_src = crate::acl_tensor::AclTensor::from_device(src_ptr, src_shape, dtype);
    let acl_out = crate::acl_tensor::AclTensor::from_device(out_ptr, &out_shape, dtype);

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnSliceGetWorkspaceSize(
                acl_src.raw(),
                1, // dim=1 (column slice)
                start,
                start + out_cols as i64,
                1, // step=1
                acl_out.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnSlice(ws, ws_size, exec, s)
        },
        stream,
        "aclnnSlice (qkv split)",
    )
    .map_err(|e| apxinf_core::Error::Other(format!("aclnn_slice_2d: {e}")))
}

/// Split a 2D tensor `[rows, total_cols]` along dim=1 into multiple contiguous
/// output tensors using `aclnnSplitWithSize` (single op, replaces per-token loop).
///
/// `out_ptrs` must have the same length as `split_sizes`. Each output tensor is
/// `[rows, split_sizes[i]]` and contiguous.
pub fn aclnn_split_with_size(
    src_ptr: usize,
    src_shape: &apxinf_core::Shape,
    out_ptrs: &[usize],
    split_sizes: &[i64],
    dtype: apxinf_core::DType,
    stream: crate::ffi::acl::aclrtStream,
) -> apxinf_core::Result<()> {
    let dims = src_shape.dims();
    debug_assert_eq!(dims.len(), 2);
    let rows = dims[0] as i64;
    debug_assert_eq!(out_ptrs.len(), split_sizes.len());

    let acl_src = crate::acl_tensor::AclTensor::from_device(src_ptr, src_shape, dtype);

    // Build output tensors: each [rows, split_sizes[i]], contiguous
    let acl_outs: Vec<crate::acl_tensor::AclTensor> = out_ptrs
        .iter()
        .zip(split_sizes.iter())
        .map(|(&ptr, &cols)| {
            let out_shape = apxinf_core::Shape::new(vec![rows as usize, cols as usize]);
            crate::acl_tensor::AclTensor::from_device(ptr, &out_shape, dtype)
        })
        .collect();
    let out_raws: Vec<*const crate::ffi::aclnn::aclTensor> =
        acl_outs.iter().map(|t| t.raw() as *const _).collect();
    let out_list = crate::acl_tensor::AclTensorList::new(&out_raws);
    let split_sizes_arr = crate::acl_tensor::AclIntArray::new(split_sizes);

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnSplitWithSizeGetWorkspaceSize(
                acl_src.raw(),
                split_sizes_arr.raw(),
                1, // dim=1 (column split)
                out_list.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnSplitWithSize(ws, ws_size, exec, s)
        },
        stream,
        "aclnnSplitWithSize (qkv split)",
    )
    .map_err(|e| apxinf_core::Error::Other(format!("aclnn_split_with_size: {e}")))?;
    Ok(())
}

// Re-export all op functions and their arg types
pub use attention::{attention, AttentionArgs, AttentionMask, AttentionPolicy};
pub use cache::{concat_rows, reserve_prefix};
pub use gather::{gather, GatherArgs, GatherPatchGeometry, GatherSemantic};
pub use gemm::{
    gemm, gemm_bias, gemm_geglu, GemmArgs, GemmBiasArgs, GemmGegluArgs, GemmImpl, GemmPolicy,
    GemmQuantization, WeightVersion,
};
pub use norm::{
    ada_gate_residual, ada_gate_residual_rms_norm, adaptive_rms_norm,
    add_layer_norm, bias_residual, bias_residual_layer_norm, bias_residual_rms_norm,
    bias_then_residual, inplace_add_rms_norm_zero_copy, layer_norm, rms_norm,
    AdaGateResidualArgs, AdaGateResidualRmsNormArgs, AdaptiveRmsNormArgs,
    AddLayerNormArgs, BiasResidualArgs, BiasResidualLayerNormArgs, BiasResidualRmsNormArgs,
    BiasThenResidualArgs, LayerNormArgs, RmsNormArgs,
};
pub use pointwise::{pointwise, PointwiseActivation, PointwiseArgs, PointwiseSemantic};
pub use rope::{rope, RopeArgs, RopeSemantic};
pub use shape_ops::{transpose, cast, cat, expand, sub, div, clamp};

// ── Legacy raw helpers (torch_npu graph alignment) ─────────────────────
// Bare device-pointer / stream helpers used by the fused-op BF16 reference model to match
// the same op sequence as the torch_npu FULL_FUSION path (1 op per aclnn call).

/// Scalar add: `out = input + scalar` via `aclnnAdds`.
pub fn aclnn_adds(
    input_ptr: usize,
    out_ptr: usize,
    shape: &apxinf_core::Shape,
    scalar: f32,
    dtype: apxinf_core::DType,
    stream: crate::ffi::acl::aclrtStream,
) -> apxinf_core::Result<()> {
    if shape.numel() == 0 {
        return Ok(());
    }
    let acl_in = crate::acl_tensor::AclTensor::from_device(input_ptr, shape, dtype);
    let acl_out = crate::acl_tensor::AclTensor::from_device(out_ptr, shape, dtype);
    let acl_scalar = crate::acl_tensor::AclScalar::f32(scalar);
    let acl_alpha = crate::acl_tensor::AclScalar::f32(1.0);
    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnAddsGetWorkspaceSize(
                acl_in.raw(),
                acl_scalar.raw(),
                acl_alpha.raw(),
                acl_out.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe { crate::ffi::aclnn::aclnnAdds(ws, ws_size, exec, s) },
        stream,
        "aclnnAdds",
    )
    .map_err(|e| apxinf_core::Error::Other(format!("aclnn_adds: {e}")))
}

/// Scalar mul: `out = input * scalar` via `aclnnMuls`.
pub fn aclnn_muls(
    input_ptr: usize,
    out_ptr: usize,
    shape: &apxinf_core::Shape,
    scalar: f32,
    dtype: apxinf_core::DType,
    stream: crate::ffi::acl::aclrtStream,
) -> apxinf_core::Result<()> {
    if shape.numel() == 0 {
        return Ok(());
    }
    let acl_in = crate::acl_tensor::AclTensor::from_device(input_ptr, shape, dtype);
    let acl_out = crate::acl_tensor::AclTensor::from_device(out_ptr, shape, dtype);
    let acl_scalar = crate::acl_tensor::AclScalar::f32(scalar);
    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnMulsGetWorkspaceSize(
                acl_in.raw(),
                acl_scalar.raw(),
                acl_out.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe { crate::ffi::aclnn::aclnnMuls(ws, ws_size, exec, s) },
        stream,
        "aclnnMuls",
    )
    .map_err(|e| apxinf_core::Error::Other(format!("aclnn_muls: {e}")))
}

/// Element-wise mul: `out = a * b` via `aclnnMul`.
pub fn aclnn_mul(
    a_ptr: usize,
    b_ptr: usize,
    out_ptr: usize,
    a_shape: &apxinf_core::Shape,
    b_shape: &apxinf_core::Shape,
    dtype: apxinf_core::DType,
    stream: crate::ffi::acl::aclrtStream,
) -> apxinf_core::Result<()> {
    let acl_a = crate::acl_tensor::AclTensor::from_device(a_ptr, a_shape, dtype);
    let acl_b = crate::acl_tensor::AclTensor::from_device(b_ptr, b_shape, dtype);
    let acl_out = crate::acl_tensor::AclTensor::from_device(out_ptr, a_shape, dtype);
    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnMulGetWorkspaceSize(
                acl_a.raw(),
                acl_b.raw(),
                acl_out.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe { crate::ffi::aclnn::aclnnMul(ws, ws_size, exec, s) },
        stream,
        "aclnnMul",
    )
    .map_err(|e| apxinf_core::Error::Other(format!("aclnn_mul: {e}")))
}

/// Gelu (tanh approximation): `out = gelu(input)` via `aclnnGeluV2` with
/// approximate=1 to match torch_npu gelu_pytorch_tanh.
pub fn aclnn_gelu(
    input_ptr: usize,
    out_ptr: usize,
    shape: &apxinf_core::Shape,
    dtype: apxinf_core::DType,
    stream: crate::ffi::acl::aclrtStream,
) -> apxinf_core::Result<()> {
    let acl_in = crate::acl_tensor::AclTensor::from_device(input_ptr, shape, dtype);
    let acl_out = crate::acl_tensor::AclTensor::from_device(out_ptr, shape, dtype);
    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnGeluV2GetWorkspaceSize(
                acl_in.raw(),
                1, // approximate=1 → tanh
                acl_out.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe { crate::ffi::aclnn::aclnnGeluV2(ws, ws_size, exec, s) },
        stream,
        "aclnnGeluV2",
    )
    .map_err(|e| apxinf_core::Error::Other(format!("aclnn_gelu: {e}")))
}

/// Element-wise add: `out = a + alpha * b` via `aclnnAdd`.
pub fn aclnn_add(
    a_ptr: usize,
    b_ptr: usize,
    out_ptr: usize,
    a_shape: &apxinf_core::Shape,
    b_shape: &apxinf_core::Shape,
    alpha: f32,
    dtype: apxinf_core::DType,
    stream: crate::ffi::acl::aclrtStream,
) -> apxinf_core::Result<()> {
    let acl_a = crate::acl_tensor::AclTensor::from_device(a_ptr, a_shape, dtype);
    let acl_b = crate::acl_tensor::AclTensor::from_device(b_ptr, b_shape, dtype);
    let acl_out = crate::acl_tensor::AclTensor::from_device(out_ptr, a_shape, dtype);
    let acl_alpha = crate::acl_tensor::AclScalar::f32(alpha);
    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnAddGetWorkspaceSize(
                acl_a.raw(),
                acl_b.raw(),
                acl_alpha.raw(),
                acl_out.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe { crate::ffi::aclnn::aclnnAdd(ws, ws_size, exec, s) },
        stream,
        "aclnnAdd",
    )
    .map_err(|e| apxinf_core::Error::Other(format!("aclnn_add: {e}")))
}
