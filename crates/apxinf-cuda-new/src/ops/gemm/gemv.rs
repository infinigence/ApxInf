//! The M=1 projections: FP8 and NVFP4 GEMV.
//!
//! These are GEMMs, so they belong to this family rather than to the MLP that
//! calls them (`doc/cuda-new-design.md` section 1.3). They are not members of
//! the *tunable* shape, though: one implementation each and nothing whose
//! relative speed needs measuring, so per section 1.2 question 1 they skip the
//! candidate registry, the recipe and `prepare`/`enqueue` entirely, and keep
//! flat parameters.
//!
//! Moved here from `ops/mlp.rs`.

use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::{gemm as abi, status};
use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::CudaContext;

/// Single-token FP8 projection: `y[n] = alpha * sum_k weight[n,k] * a[k]`.
///
/// `weight` is `[N, K]`, the orientation a checkpoint stores.
///
/// **Measured slower than the general GEMM at M=1** -- 0.94 ms against
/// 0.358 ms on a 10240x5120 projection. One row per block with a scalar FP8
/// conversion per element leaves each thread ~20 bytes of work, so launch and
/// conversion overhead dominate. The GEMM path itself only reaches ~146 GB/s
/// here, so a vectorized GEMV (16-byte loads, one warp per row, several rows
/// per block) should beat both; this one does not, and callers should prefer
/// the GEMM until that exists.
pub fn fp8_gemv(
    ctx: &CudaContext,
    weight: &Tensor,
    activation: &Tensor,
    output: &Tensor,
    alpha: f32,
) -> Result<()> {
    let weight_dims = weight.shape().dims().to_vec();
    if weight_dims.len() != 2 {
        return Err(invalid("FP8 GEMV weight must be [N, K]"));
    }
    let (n, k) = (weight_dims[0], weight_dims[1]);
    let weight_buffer = tensor_storage(ctx, weight, DType::F8E4M3, &weight_dims)?;
    let activation_dims = activation.shape().dims().to_vec();
    if activation_dims.iter().product::<usize>() != k {
        return Err(invalid("FP8 GEMV activation must hold K elements"));
    }
    let activation_buffer = tensor_storage(ctx, activation, DType::F8E4M3, &activation_dims)?;
    let output_dims = output.shape().dims().to_vec();
    if output_dims.iter().product::<usize>() != n {
        return Err(invalid("FP8 GEMV output must hold N elements"));
    }
    let output_buffer = tensor_storage(ctx, output, DType::BF16, &output_dims)?;
    unsafe {
        status::check(abi::apxinf_gemm_fp8_gemv(
            weight_buffer.ptr(),
            activation_buffer.ptr(),
            output_buffer.ptr(),
            n as i64,
            k as i64,
            alpha,
            ctx.stream().handle(),
        ))
    }
}

/// Single-token NVFP4 projection.
///
/// Scales are plain row-major `[rows, K/16]` — the layout a checkpoint stores.
/// The CUTLASS atom layout exists for the tcgen05 MMA; a GEMV indexes scales
/// directly, so a decode-only path needs no relayout.
#[allow(clippy::too_many_arguments)]
pub fn nvfp4_gemv(
    ctx: &CudaContext,
    weight: &Tensor,
    weight_scales: &Tensor,
    activation: &Tensor,
    activation_scales: &Tensor,
    output: &Tensor,
    alpha: f32,
) -> Result<()> {
    let weight_dims = weight.shape().dims().to_vec();
    if weight_dims.len() != 2 {
        return Err(invalid("NVFP4 GEMV weight must be [N, K/2]"));
    }
    let (n, k) = (weight_dims[0], weight_dims[1] * 2);
    let weight_buffer = tensor_storage(ctx, weight, DType::E2M1Pair, &weight_dims)?;
    let weight_scale_dims = weight_scales.shape().dims().to_vec();
    let weight_scale_buffer =
        tensor_storage(ctx, weight_scales, DType::F8E4M3, &weight_scale_dims)?;
    if weight_scale_buffer.len() < n * k / 16 {
        return Err(invalid("NVFP4 GEMV weight scales must hold N*K/16 entries"));
    }
    let activation_dims = activation.shape().dims().to_vec();
    let activation_buffer = tensor_storage(ctx, activation, DType::E2M1Pair, &activation_dims)?;
    if activation_buffer.len() < k / 2 {
        return Err(invalid("NVFP4 GEMV activation must hold K/2 bytes"));
    }
    let activation_scale_dims = activation_scales.shape().dims().to_vec();
    let activation_scale_buffer =
        tensor_storage(ctx, activation_scales, DType::F8E4M3, &activation_scale_dims)?;
    if activation_scale_buffer.len() < k / 16 {
        return Err(invalid("NVFP4 GEMV activation scales must hold K/16 entries"));
    }
    let output_dims = output.shape().dims().to_vec();
    if output_dims.iter().product::<usize>() != n {
        return Err(invalid("NVFP4 GEMV output must hold N elements"));
    }
    let output_buffer = tensor_storage(ctx, output, DType::BF16, &output_dims)?;
    unsafe {
        status::check(abi::apxinf_gemm_nvfp4_gemv(
            weight_buffer.ptr(),
            weight_scale_buffer.ptr(),
            activation_buffer.ptr(),
            activation_scale_buffer.ptr(),
            output_buffer.ptr(),
            n as i64,
            k as i64,
            alpha,
            ctx.stream().handle(),
        ))
    }
}
