//! Family composition helpers; device ownership and kernels stay in cuda-new.
use super::super::weights::{upload, Linear};
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use apxinf_cuda_new::ops::{embedding_gather, gemm, GemmArgs, GemmPolicy, WeightVersion};
use apxinf_cuda_new::{
    pointwise, quantization, CudaBuffer, CudaContext, PointwiseActivation, PointwiseArgs,
    PointwiseSemantic, QuantizationArgs, QuantizationSemantic,
};

pub fn alloc(ctx: &CudaContext, shape: impl Into<Shape>, dtype: DType) -> Result<Tensor> {
    let shape = shape.into();
    let size = shape
        .numel()
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| Error::Other("tensor size overflow".into()))?;
    CudaBuffer::alloc(size, ctx.device_id())
        .map_err(Error::Cuda)?
        .as_tensor(shape, dtype)
        .map_err(Error::Cuda)
}
pub fn bf(ctx: &CudaContext, rows: usize, cols: usize) -> Result<Tensor> {
    alloc(ctx, vec![rows, cols], DType::BF16)
}
pub fn matrix(t: &Tensor, cols: usize) -> Result<Tensor> {
    t.reshape(vec![t.numel() / cols, cols])
}
pub fn indices(ctx: &CudaContext, values: &[i32]) -> Result<Tensor> {
    upload(
        &Tensor::from_i32(vec![values.len()], values)?,
        ctx.device_id(),
    )
}
pub fn constant(ctx: &CudaContext, cols: usize, value: f32) -> Result<Tensor> {
    upload(
        &Tensor::from_bf16(vec![cols], &vec![half::bf16::from_f32(value); cols])?,
        ctx.device_id(),
    )
}
pub fn linear(
    ctx: &CudaContext,
    input: &Tensor,
    w: &Linear,
    output: &Tensor,
    policy: &GemmPolicy,
) -> Result<()> {
    let mut output = output.clone();
    let mut args = GemmArgs::new(input, &w.weight, &mut output);
    args.policy = policy.clone();
    args.weight_version = Some(WeightVersion::new(1));
    gemm(ctx, args)?;
    if let Some(bias) = &w.bias {
        activate(ctx, &output, Some(bias), PointwiseActivation::None, &output)?;
    }
    Ok(())
}
pub fn activate(
    ctx: &CudaContext,
    input: &Tensor,
    bias: Option<&Tensor>,
    activation: PointwiseActivation,
    out: &Tensor,
) -> Result<()> {
    let mut out = out.clone();
    let mut args = PointwiseArgs::new(PointwiseSemantic::BiasActivation, input, &mut out);
    args.bias = bias;
    args.activation = activation;
    pointwise(ctx, args)
}
pub fn multiply(ctx: &CudaContext, a: &Tensor, b: &Tensor, out: &Tensor) -> Result<()> {
    let mut out = out.clone();
    let mut args = PointwiseArgs::new(PointwiseSemantic::Multiply, a, &mut out);
    args.secondary = Some(b);
    pointwise(ctx, args)
}
pub fn add(ctx: &CudaContext, a: &Tensor, b: &Tensor, out: &Tensor) -> Result<()> {
    let mut out = out.clone();
    let mut args = PointwiseArgs::new(PointwiseSemantic::EulerUpdate, a, &mut out);
    args.secondary = Some(b);
    args.dt = 1.;
    pointwise(ctx, args)
}
pub fn cast(ctx: &CudaContext, input: &Tensor, out: &Tensor) -> Result<()> {
    let semantic = match (input.dtype(), out.dtype()) {
        (DType::BF16, DType::F32) => QuantizationSemantic::CastBf16ToF32,
        (DType::F32, DType::BF16) => QuantizationSemantic::CastF32ToBf16,
        _ => return Err(Error::Other("unsupported Cosmos intermediate cast".into())),
    };
    quantization(
        ctx,
        QuantizationArgs::new(semantic, input, &mut out.clone()),
    )
}
pub fn gather(
    ctx: &CudaContext,
    input: &Tensor,
    ids: &Tensor,
    cols: usize,
    out: &Tensor,
) -> Result<()> {
    embedding_gather(ctx, &matrix(input, cols)?, ids, &matrix(out, cols)?)
}
