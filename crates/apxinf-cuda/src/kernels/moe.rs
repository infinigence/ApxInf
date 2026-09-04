//! Mixture-of-experts routing and token permutation contracts.
//!
//! These operators are model-neutral: any top-k softmax router with
//! per-token expert weights can use them. Expert GEMMs themselves are
//! executed through [`super::gemm`] (dense BF16 after dequantization) or the
//! packed W4A16 GEMV in [`super::gemm::w4a16`].

use apxinf_core::{DType, Error, Result};

use super::contracts::{check_cuda, require_buffers};
use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;

/// Softmax over `experts` router logits per token, keep the `k` largest.
///
/// * `logits`: BF16 `[tokens, experts]` (experts <= 256).
/// * `topk_idx`: i32 `[tokens, k]`, `topk_weight`: f32 `[tokens, k]`, both in
///   descending probability order.
/// * `renormalize`: divide the kept weights by their sum (`norm_topk_prob`).
#[allow(clippy::too_many_arguments)]
pub fn router_topk_into(
    ctx: &CudaContext,
    logits: &CudaBuffer,
    tokens: usize,
    experts: usize,
    k: usize,
    renormalize: bool,
    topk_idx: &CudaBuffer,
    topk_weight: &CudaBuffer,
) -> Result<()> {
    if tokens == 0 || experts == 0 || experts > 256 || k == 0 || k > experts || k > 32 {
        return Err(Error::Other(format!(
            "MoE router: unsupported tokens={tokens} experts={experts} k={k}"
        )));
    }
    require_buffers(
        ctx,
        "MoE router",
        &[
            ("logits", logits, tokens * experts * DType::BF16.size_in_bytes()),
            ("topk_idx", topk_idx, tokens * k * 4),
            ("topk_weight", topk_weight, tokens * k * 4),
        ],
    )?;
    unsafe {
        check_cuda(ffi::apxinf_moe_router_topk_bf16(
            logits.ptr(),
            topk_idx.ptr(),
            topk_weight.ptr(),
            tokens as i32,
            experts as i32,
            k as i32,
            renormalize as i32,
            ctx.stream().handle(),
        ))
    }
}

/// `output[r, :] = x[source_rows[r], :]` for BF16 rows of width `cols`.
pub fn gather_rows_into(
    ctx: &CudaContext,
    x: &CudaBuffer,
    x_rows: usize,
    source_rows: &CudaBuffer,
    rows: usize,
    cols: usize,
    output: &CudaBuffer,
) -> Result<()> {
    if rows == 0 || cols == 0 || x_rows == 0 {
        return Err(Error::Other("MoE gather needs non-zero dimensions".into()));
    }
    let elem = DType::BF16.size_in_bytes();
    require_buffers(
        ctx,
        "MoE gather",
        &[
            ("x", x, x_rows * cols * elem),
            ("source_rows", source_rows, rows * 4),
            ("output", output, rows * cols * elem),
        ],
    )?;
    unsafe {
        check_cuda(ffi::apxinf_gather_rows_bf16(
            x.ptr(),
            source_rows.ptr(),
            output.ptr(),
            rows as i32,
            cols as i32,
            ctx.stream().handle(),
        ))
    }
}

/// `output[m, :] = sum_{s<k} weight[m*k+s] * y[slot_rows[m*k+s], :]`.
///
/// `y` holds `tokens * k` expert-output rows in any order; `slot_rows` maps
/// each (token, slot) to its row in `y`.
#[allow(clippy::too_many_arguments)]
pub fn weighted_gather_sum_into(
    ctx: &CudaContext,
    y: &CudaBuffer,
    slot_rows: &CudaBuffer,
    weight: &CudaBuffer,
    tokens: usize,
    k: usize,
    cols: usize,
    output: &CudaBuffer,
) -> Result<()> {
    if tokens == 0 || k == 0 || cols == 0 {
        return Err(Error::Other("MoE combine needs non-zero dimensions".into()));
    }
    let elem = DType::BF16.size_in_bytes();
    require_buffers(
        ctx,
        "MoE combine",
        &[
            ("y", y, tokens * k * cols * elem),
            ("slot_rows", slot_rows, tokens * k * 4),
            ("weight", weight, tokens * k * 4),
            ("output", output, tokens * cols * elem),
        ],
    )?;
    unsafe {
        check_cuda(ffi::apxinf_weighted_gather_sum_bf16(
            y.ptr(),
            slot_rows.ptr(),
            weight.ptr(),
            output.ptr(),
            tokens as i32,
            k as i32,
            cols as i32,
            ctx.stream().handle(),
        ))
    }
}

/// Row-wise SwiGLU: `gate_up` BF16 `[rows, 2*inter]` -> `output` `[rows, inter]`.
pub fn silu_mul_rows_into(
    ctx: &CudaContext,
    gate_up: &CudaBuffer,
    rows: usize,
    inter: usize,
    output: &CudaBuffer,
) -> Result<()> {
    if rows == 0 || inter == 0 {
        return Err(Error::Other("row SwiGLU needs non-zero dimensions".into()));
    }
    let elem = DType::BF16.size_in_bytes();
    require_buffers(
        ctx,
        "row SwiGLU",
        &[
            ("gate_up", gate_up, rows * 2 * inter * elem),
            ("output", output, rows * inter * elem),
        ],
    )?;
    unsafe {
        check_cuda(ffi::apxinf_silu_mul_rows_bf16(
            gate_up.ptr(),
            output.ptr(),
            rows as i32,
            inter as i32,
            ctx.stream().handle(),
        ))
    }
}
