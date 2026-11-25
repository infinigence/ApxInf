//! Small tensor helpers shared by the block compositions.
use crate::cuda_minimax_h3_image::backend::ops::{self, GemmArgs, GemmPolicy};
use crate::cuda_minimax_h3_image::backend::{CudaBuffer, CudaContext};
use crate::cuda_minimax_h3_image::weights::{DenseLinear, Int8Linear};
use apxinf_core::{DType, Error, Result, Shape, Tensor};

pub(super) fn alloc(ctx: &CudaContext, shape: &[usize], dtype: DType) -> Result<Tensor> {
    ctx.allocate_output(Shape::new(shape.to_vec()), dtype)
}

/// `rows` consecutive rows of a contiguous matrix, starting at `row0`, as a zero-copy view.
pub(in crate::cuda_minimax_h3_image) fn rows(
    t: &Tensor,
    row0: usize,
    count: usize,
    cols: usize,
) -> Result<Tensor> {
    byte_view(t, row0 * cols, count * cols, vec![count, cols])
}

/// `len` elements starting at element `start`, viewed with `shape`.
pub(super) fn byte_view(t: &Tensor, start: usize, len: usize, shape: Vec<usize>) -> Result<Tensor> {
    let size = t.dtype().size_in_bytes();
    CudaBuffer::from_tensor(t)
        .map_err(Error::Cuda)?
        .view(start * size, len * size)
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape), t.dtype())
        .map_err(Error::Cuda)
}

/// W8A8 GEMM of already quantized rows against one ConvRot-rotated INT8 weight, BF16 output `[rows, n]`.
pub(super) fn int8_linear(
    ctx: &CudaContext,
    input: &ops::quantization::DynamicI8Tensor,
    weight: &Int8Linear,
    policy: &GemmPolicy,
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
    let m = *m;
    let mut out = alloc(ctx, &[m, weight.n], DType::BF16)?;
    let mut args = GemmArgs::row_channel_quantized(
        &input.values,
        &input.scales,
        &weight.weight,
        &weight.channel_scales,
        &mut out,
    )
    .with_repacked_weight(&weight.layout);
    // the constructor fixed INT32 accumulation for INT8 inputs; the caller's policy supplies the rest
    let accumulation = args.policy.accumulation_dtype;
    args.policy = policy.clone();
    args.policy.accumulation_dtype = accumulation;
    ops::gemm(ctx, args)?;
    Ok(out)
}

/// Plain GEMM `input [m, k] @ weight [k, n] (+ bias)`; the output dtype is the input dtype.
pub(super) fn dense_linear(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &DenseLinear,
    policy: &GemmPolicy,
) -> Result<Tensor> {
    let m = input.shape().dims()[0];
    let n = weight.weight.shape().dims()[1];
    let mut out = alloc(ctx, &[m, n], input.dtype())?;
    let mut args = GemmArgs::new(input, &weight.weight, &mut out);
    args.policy = policy.clone();
    args.weight_version = Some(ops::WeightVersion::new(0));
    match &weight.bias {
        Some(bias) => ops::gemm_bias(ctx, ops::GemmBiasArgs { gemm: args, bias })?,
        None => ops::gemm(ctx, args)?,
    }
    Ok(out)
}

/// BF16 GEMM `input [m, k] @ weight [k, n] (+ bias)` producing BF16.
pub(super) fn dense_linear_bf16(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    policy: &GemmPolicy,
) -> Result<Tensor> {
    let m = input.shape().dims()[0];
    let n = weight.shape().dims()[1];
    let mut out = alloc(ctx, &[m, n], DType::BF16)?;
    let mut args = GemmArgs::new(input, weight, &mut out);
    args.policy = policy.clone();
    args.weight_version = Some(ops::WeightVersion::new(0));
    match bias {
        Some(bias) => ops::gemm_bias(ctx, ops::GemmBiasArgs { gemm: args, bias })?,
        None => ops::gemm(ctx, args)?,
    }
    Ok(out)
}

/// An output tensor with `like`'s shape and the given dtype.
pub(in crate::cuda_minimax_h3_image) fn alloc_like(
    ctx: &CudaContext,
    like: &Tensor,
    dtype: DType,
) -> Result<Tensor> {
    alloc(ctx, like.shape().dims(), dtype)
}
