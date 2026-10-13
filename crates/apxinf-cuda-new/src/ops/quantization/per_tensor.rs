//! The quantization family's per-tensor FP8 member.
//!
//! Moved here from `ops/mlp.rs`, which was named after a call site rather than
//! a mathematical family (`doc/cuda-new-design.md` section 1.3).
//!
//! [`QuantizationSemantic::FixedScaleE4m3`] computes the same arithmetic
//! through the family's Spec, and merging the two is the obvious follow-up —
//! but not a rename. That kernel's BF16 path is a scalar grid-stride loop where
//! this one is 8-wide (uint4 loads, uint64 stores), and it clamps through
//! `fminf`/`fmaxf`, which maps a NaN input to `-448` rather than to the FP8 NaN
//! encoding this convert produces. Both differences are observable, so the
//! merge needs a measurement and a NaN contract, not a redirect.
//!
//! [`QuantizationSemantic::FixedScaleE4m3`]: super::QuantizationSemantic::FixedScaleE4m3

use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::{quantization as abi, status};
use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::CudaContext;

/// Quantize BF16 to E4M3 against a single per-tensor scale.
///
/// The FP8 projections in a ModelOpt checkpoint carry scalar `weight_scale`
/// and `input_scale`, not the `[M]`/`[N]` vectors [`GemmQuantization::Fp8`]
/// expects. Quantizing here against `input_scale` lets the projection run as
/// `Fp8UnitScale` with `alpha = weight_scale * input_scale`, so attention and
/// GDN need no new quantization contract.
///
/// [`GemmQuantization::Fp8`]: crate::ops::GemmQuantization::Fp8
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
        status::check(abi::apxinf_quantization_fp8_per_tensor(
            source.ptr(),
            destination.ptr(),
            dims.iter().product::<usize>() as i64,
            input_scale,
            ctx.stream().handle(),
        ))
    }
}
