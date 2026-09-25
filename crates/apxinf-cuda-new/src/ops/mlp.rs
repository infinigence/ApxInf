//! MLP building blocks: RMSNorm, SwiGLU and residual addition.
//!
//! These have a single implementation each and nothing to select between, so
//! they are direct entry points rather than registry-backed operators
//! (`doc/adding-new-kernels.md` section 6).

use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::{mlp as abi, status};
use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::CudaContext;

/// `y[r,c] = x[r,c] / sqrt(mean(x[r,:]^2) + epsilon) * weight[c]`
///
/// The reduction runs in f32 regardless of the BF16 storage.
pub fn rms_norm(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    epsilon: f32,
) -> Result<()> {
    let dims = input.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(invalid("RMSNorm requires a rank-2 tensor"));
    }
    let (rows, width) = (dims[0], dims[1]);
    let input_buffer = tensor_storage(ctx, input, DType::BF16, &dims)?;
    let weight_buffer = tensor_storage(ctx, weight, DType::BF16, &[width])?;
    let output_buffer = tensor_storage(ctx, output, DType::BF16, &dims)?;
    unsafe {
        status::check(abi::apxinf_rms_norm_bf16(
            input_buffer.ptr(),
            weight_buffer.ptr(),
            output_buffer.ptr(),
            rows as i64,
            width as i64,
            epsilon,
            ctx.stream().handle(),
        ))
    }
}

/// `y[r,c] = silu(fused[r,c]) * fused[r,width+c]`
///
/// `fused` is `[rows, 2*width]` with gate first, the layout one fused gate/up
/// GEMM produces. This is SwiGLU; `gemm_geglu` computes the GELU variant and
/// is not interchangeable.
pub fn swiglu(ctx: &CudaContext, fused: &Tensor, output: &Tensor) -> Result<()> {
    let fused_dims = fused.shape().dims().to_vec();
    let output_dims = output.shape().dims().to_vec();
    if fused_dims.len() != 2 || output_dims.len() != 2 {
        return Err(invalid("SwiGLU requires rank-2 tensors"));
    }
    if fused_dims[0] != output_dims[0] || fused_dims[1] != output_dims[1] * 2 {
        return Err(invalid(
            "SwiGLU input must be [rows, 2*width] for an output of [rows, width]",
        ));
    }
    let fused_buffer = tensor_storage(ctx, fused, DType::BF16, &fused_dims)?;
    let output_buffer = tensor_storage(ctx, output, DType::BF16, &output_dims)?;
    unsafe {
        status::check(abi::apxinf_swiglu_bf16(
            fused_buffer.ptr(),
            output_buffer.ptr(),
            output_dims[0] as i64,
            output_dims[1] as i64,
            ctx.stream().handle(),
        ))
    }
}

/// `accumulator += addend`, elementwise. Residual connections.
pub fn add_into(ctx: &CudaContext, addend: &Tensor, accumulator: &Tensor) -> Result<()> {
    let dims = addend.shape().dims().to_vec();
    if dims != accumulator.shape().dims() {
        return Err(invalid("residual add requires matching shapes"));
    }
    let addend_buffer = tensor_storage(ctx, addend, DType::BF16, &dims)?;
    let accumulator_buffer = tensor_storage(ctx, accumulator, DType::BF16, &dims)?;
    unsafe {
        status::check(abi::apxinf_add_bf16(
            addend_buffer.ptr(),
            accumulator_buffer.ptr(),
            dims.iter().product::<usize>() as i64,
            ctx.stream().handle(),
        ))
    }
}

/// Quantize BF16 to E4M3 against a single per-tensor scale.
///
/// The FP8 projections in a ModelOpt checkpoint carry scalar `weight_scale`
/// and `input_scale`, not the `[M]`/`[N]` vectors [`GemmQuantization::Fp8`]
/// expects. Quantizing here against `input_scale` lets the projection run as
/// `Fp8UnitScale` with `alpha = weight_scale * input_scale`, so attention and
/// GDN need no new quantization contract.
pub fn quantize_fp8_per_tensor(
    ctx: &CudaContext,
    input: &Tensor,
    output: &Tensor,
    input_scale: f32,
) -> Result<()> {
    let dims = input.shape().dims().to_vec();
    if dims != output.shape().dims() {
        return Err(invalid("FP8 quantization requires matching shapes"));
    }
    if !(input_scale > 0.0) || !input_scale.is_finite() {
        return Err(invalid("FP8 input_scale must be finite and positive"));
    }
    let source = tensor_storage(ctx, input, DType::BF16, &dims)?;
    let destination = tensor_storage(ctx, output, DType::F8E4M3, &dims)?;
    unsafe {
        status::check(abi::apxinf_quantize_fp8_per_tensor(
            source.ptr(),
            destination.ptr(),
            dims.iter().product::<usize>() as i64,
            input_scale,
            ctx.stream().handle(),
        ))
    }
}

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
        status::check(abi::apxinf_fp8_gemv(
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
        status::check(abi::apxinf_nvfp4_gemv(
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
