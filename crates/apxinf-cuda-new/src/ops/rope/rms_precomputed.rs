use super::contracts::{invalid, tensor_storage};
use crate::ffi::abi::{rope as abi, status};
use crate::{CudaBuffer, CudaContext};
use apxinf_core::{DType, Result, Tensor};
use std::rc::Rc;

/// Per-head RMSNorm followed by half-split rotary embedding.
/// Input/output: BF16 [tokens, heads, dim]; weight: BF16 [dim]; table:
/// FP32 [tokens, 2, dim/2], cosine then sine, broadcast over heads.
/// The RMS result rounds to BF16 before separately rounded FP32 rotary
/// products/sums and final BF16 conversion. Exact in-place output is allowed.
pub fn rms_norm_apply_precomputed(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    table: &Tensor,
    epsilon: f32,
    output: &Tensor,
) -> Result<()> {
    let d = input.shape().dims();
    if d.len() != 3
        || d.contains(&0)
        || d[2] > 2048
        || d[2] % 2 != 0
        || input.shape() != output.shape()
        || weight.shape().dims() != [d[2]]
        || table.shape().dims() != [d[0], 2, d[2] / 2]
        || !epsilon.is_finite()
        || epsilon <= 0.0
    {
        return Err(invalid(
            "RMSNorm precomputed RoPE shape or epsilon mismatch",
        ));
    }
    let rows = d[0]
        .checked_mul(d[1])
        .filter(|&n| n <= i32::MAX as usize)
        .ok_or_else(|| invalid("RMSNorm precomputed RoPE row count exceeds range"))?;
    let count = rows
        .checked_mul(d[2])
        .ok_or_else(|| invalid("RMSNorm RoPE size overflow"))?;
    let input = tensor_storage(ctx, input, DType::BF16, count)?;
    let output = tensor_storage(ctx, output, DType::BF16, count)?;
    let weight = tensor_storage(ctx, weight, DType::BF16, d[2])?;
    let table = tensor_storage(ctx, table, DType::F32, d[0] * d[2])?;
    let overlaps = |a: &CudaBuffer, b: &CudaBuffer| {
        let a0 = a.ptr() as usize;
        let b0 = b.ptr() as usize;
        a0 < b0.saturating_add(b.len()) && b0 < a0.saturating_add(a.len())
    };
    if (input.ptr() != output.ptr() && overlaps(&input, &output))
        || overlaps(&weight, &output)
        || overlaps(&table, &output)
    {
        return Err(invalid(
            "RMSNorm RoPE rejects partial input/output and weight/table output overlap",
        ));
    }
    let spec = abi::RmsPrecomputedSpec {
        version: 1,
        heads: d[1] as u32,
        dim: d[2] as u32,
        reserved: 0,
        tokens: d[0] as i64,
    };
    let bindings = abi::RmsPrecomputedBindings {
        input: input.ptr().cast_const(),
        weight: weight.ptr().cast_const(),
        table: table.ptr().cast_const(),
        output: output.ptr(),
        stream: ctx.stream().handle(),
        epsilon,
    };
    crate::workspace::validate_capture_target(ctx.device_id(), bindings.stream as usize)?;
    // Fixed operators do not need a prepared execution, but graphs must own all
    // storage referenced by their captured kernels even after caller tensors drop.
    let storage = Rc::new(vec![input, output, weight, table]);
    crate::workspace::retain_resource(&storage);
    unsafe {
        status::check(abi::apxinf_rope_rms_precomputed_launch(
            ctx.runtime(),
            &spec,
            &bindings,
        ))
    }
}
