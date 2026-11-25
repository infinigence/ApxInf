//! Small tensor helpers shared by the block compositions.

use crate::cuda_lingbot_vla2::backend::{ops, Context, DeviceBuffer};
use crate::cuda_lingbot_vla2::weights::device::Int8Linear;
use apxinf_core::{DType, Error, Result, Shape, Tensor};

pub(super) fn alloc(ctx: &Context, shape: &[usize], dtype: DType) -> Result<Tensor> {
    ctx.allocate_output(Shape::new(shape.to_vec()), dtype)
}

/// `count` consecutive rows of a contiguous `[*, cols]` tensor starting at `row0`, as a zero-copy view.
pub(super) fn rows(t: &Tensor, row0: usize, count: usize, cols: usize) -> Result<Tensor> {
    byte_view(t, row0 * cols, count * cols, vec![count, cols])
}

/// `len` elements starting at element `start`, viewed with `shape`.
pub(super) fn byte_view(t: &Tensor, start: usize, len: usize, shape: Vec<usize>) -> Result<Tensor> {
    let size = t.dtype().size_in_bytes();
    DeviceBuffer::from_tensor(t)
        .map_err(Error::Cuda)?
        .view(start * size, len * size)
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape), t.dtype())
        .map_err(Error::Cuda)
}

/// W8A8 GEMM of rotated INT8 rows against one rotated INT8 weight: BF16 `[m, n]`.
pub(super) fn int8_linear(
    ctx: &Context,
    input: &ops::quantization::DynamicI8Tensor,
    weight: &Int8Linear,
    policy: &ops::GemmPolicy,
) -> Result<Tensor> {
    let [m, k] = input.values.shape().dims() else {
        return Err(Error::Other(
            "W8A8 activation must be a [rows, K] matrix".into(),
        ));
    };
    if *k != weight.k {
        return Err(Error::Other(format!(
            "W8A8 activation width {k} does not match the weight's K {}",
            weight.k
        )));
    }
    let mut out = alloc(ctx, &[*m, weight.n], DType::BF16)?;
    let mut args = ops::GemmArgs::row_channel_quantized(
        &input.values,
        &input.scales,
        &weight.weight,
        &weight.scales,
        &mut out,
    )
    .with_repacked_weight(&weight.layout);
    let accumulation = args.policy.accumulation_dtype;
    args.policy = policy.clone();
    args.policy.accumulation_dtype = accumulation;
    ops::gemm(ctx, args)?;
    Ok(out)
}

/// Rotate each 256-wide column group and quantise per row (the activation half of ConvRot).
pub(super) fn rotate_quantize(
    ctx: &Context,
    x: &Tensor,
) -> Result<ops::quantization::DynamicI8Tensor> {
    ops::quantization::rotate_quantize_rows_i8(ctx, x)
}

/// `y = x + bias` broadcast over rows, BF16, out of place.
pub(super) fn add_bias(ctx: &Context, x: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let out = alloc(ctx, x.shape().dims(), DType::BF16)?;
    ops::elementwise_add_bias(ctx, x, bias, &out)?;
    Ok(out)
}

/// Dense BF16 `x [m, k] @ w [k, n] (+ bias)`.
pub(super) fn dense(
    ctx: &Context,
    x: &Tensor,
    w: &crate::cuda_lingbot_vla2::weights::device::DenseLinear,
    policy: &ops::GemmPolicy,
) -> Result<Tensor> {
    let m = x.shape().dims()[0];
    let n = w.weight.shape().dims()[1];
    let mut out = alloc(ctx, &[m, n], DType::BF16)?;
    let mut args = ops::GemmArgs::new(x, &w.weight, &mut out);
    args.policy = policy.clone();
    args.weight_version = Some(ops::WeightVersion::new(0));
    ops::gemm(ctx, args)?;
    // The bias is a separate pass. The fused `gemm_bias` candidate list contains cuBLASLt epilogues that fail on
    // sm_89 for the K = 64 state/action projections; the plain GEMM plus `add_bias` is valid for every shape here.
    match &w.bias {
        Some(bias) => add_bias(ctx, &out, bias),
        None => Ok(out),
    }
}

pub(super) fn rms(ctx: &Context, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let out = alloc(ctx, x.shape().dims(), DType::BF16)?;
    ops::norm::norm(ctx, ops::NormArgs::rms(x, weight, &out, eps))?;
    Ok(out)
}

pub(super) fn layer_norm(
    ctx: &Context,
    x: &Tensor,
    w: &Tensor,
    b: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let out = alloc(ctx, x.shape().dims(), DType::BF16)?;
    ops::norm::norm(ctx, ops::NormArgs::layer(x, w, b, &out, eps))?;
    Ok(out)
}
