//! The norm family's per-head RMSNorm, for Qwen3.5's `q_norm` and `k_norm`.
//!
//! Not a Spec pipeline: it writes its input in place where every Spec semantic
//! writes a separate output, and it reduces over the last axis of a
//! `[tokens, heads, head_dim]` activation that the Spec's `[rows, cols]` shape
//! cannot name without folding heads into rows. So it keeps flat parameters per
//! `doc/cuda-new-design.md` section 5.
//!
//! Its affine is `(1 + weight)`, the same form as
//! [`Affine::Weight { offset: true }`](super::Affine::Weight), but it is the
//! family's *third* reduction tree -- a warp shuffle into static shared memory,
//! summed serially by thread 0 -- so it is not bit-comparable with either
//! `norm.cuh` or `norm_offset.cu`.
//!
//! Moved here from `ops/attn_ops.rs`, which was named after a call site rather
//! than a mathematical family (section 1.3).

use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::{norm as abi, status};
use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::CudaContext;

/// Per-head RMSNorm in place over `[rows, head_dim]`, for q_norm and k_norm.
pub fn head_rms_norm(
    ctx: &CudaContext,
    data: &Tensor,
    weight: &Tensor,
    epsilon: f32,
) -> Result<()> {
    let dims = data.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(invalid("head RMSNorm expects [rows, head_dim]"));
    }
    let data_buffer = tensor_storage(ctx, data, DType::BF16, &dims)?;
    let weight_buffer = tensor_storage(ctx, weight, DType::BF16, &[dims[1]])?;
    unsafe {
        status::check(abi::apxinf_norm_head_rms(
            data_buffer.ptr(),
            weight_buffer.ptr(),
            dims[0] as i64,
            dims[1] as i64,
            epsilon,
            ctx.stream().handle(),
        ))
    }
}
