//! Fast-WAM's safe cuda-new seam. No provider selection belongs in this family.
use apxinf_core::{DType, Error, Result, Shape, Tensor};
pub(super) use apxinf_cuda_new::{
    ops, sampling, transfers, CudaBuffer as DeviceBuffer, CudaContext as Context, ExecutionSession,
    PreparedPhase,
};

pub(super) fn reshape(x: &Tensor, dims: Vec<usize>) -> Result<Tensor> {
    x.reshape(Shape::new(dims))
}

pub(super) fn cast(ctx: &Context, x: &Tensor, dtype: DType) -> Result<Tensor> {
    if x.dtype() == dtype {
        return Ok(x.clone());
    }
    let semantic = match (x.dtype(), dtype) {
        (DType::BF16, DType::F32) => ops::QuantizationSemantic::CastBf16ToF32,
        (DType::F32, DType::BF16) => ops::QuantizationSemantic::CastF32ToBf16,
        _ => {
            return Err(Error::Other(
                "Fast-WAM supports BF16/F32 conversion only".into(),
            ))
        }
    };
    let mut out = ctx.allocate_output(x.shape().clone(), dtype)?;
    ops::quantization(ctx, ops::QuantizationArgs::new(semantic, x, &mut out))?;
    Ok(out)
}

pub(super) fn linear(
    ctx: &Context,
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    policy: &ops::GemmPolicy,
) -> Result<Tensor> {
    let x = cast(ctx, x, weight.dtype())?;
    let dims = x.shape().dims();
    if dims.len() != 2 || weight.shape().dims().len() != 2 || dims[1] != weight.shape().dims()[0] {
        return Err(Error::Other(format!(
            "Fast-WAM linear mismatch {:?} @ {:?}",
            x.shape(),
            weight.shape()
        )));
    }
    let mut out = ctx.allocate_output(
        Shape::new(vec![dims[0], weight.shape().dims()[1]]),
        weight.dtype(),
    )?;
    let mut args = ops::GemmArgs::new(&x, weight, &mut out);
    args.policy = policy.clone();
    if let Some(bias) = bias {
        ops::gemm_bias(ctx, ops::GemmBiasArgs { gemm: args, bias })?;
    } else {
        ops::gemm(ctx, args)?;
    }
    Ok(out)
}

/// PyTorch's cuDNN convolution materializes the projection in its output
/// dtype before adding bias. This differs from the Linear GEMM epilogue.
pub(super) fn convolution_projection(
    ctx: &Context,
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    policy: &ops::GemmPolicy,
) -> Result<Tensor> {
    let projection = linear(ctx, x, weight, None, policy)?;
    let Some(bias) = bias else {
        return Ok(projection);
    };
    let mut out = ctx.allocate_output(projection.shape().clone(), projection.dtype())?;
    let mut args = ops::PointwiseArgs::new(
        ops::PointwiseSemantic::BiasActivation,
        &projection,
        &mut out,
    );
    args.bias = Some(bias);
    ops::pointwise(ctx, args)?;
    Ok(out)
}

pub(super) fn activation(
    ctx: &Context,
    x: &Tensor,
    activation: ops::PointwiseActivation,
) -> Result<Tensor> {
    let mut out = ctx.allocate_output(x.shape().clone(), x.dtype())?;
    let mut args = ops::PointwiseArgs::new(ops::PointwiseSemantic::BiasActivation, x, &mut out);
    args.activation = activation;
    ops::pointwise(ctx, args)?;
    Ok(out)
}

pub(super) fn add(ctx: &Context, x: &Tensor, y: &Tensor) -> Result<Tensor> {
    let mut out = ctx.allocate_output(x.shape().clone(), x.dtype())?;
    let mut args = ops::PointwiseArgs::new(ops::PointwiseSemantic::Add, x, &mut out);
    args.secondary = Some(y);
    ops::pointwise(ctx, args)?;
    Ok(out)
}

pub(super) fn upload(ctx: &Context, x: &Tensor) -> Result<Tensor> {
    transfers::to_cuda(x, ctx.device_id())
}

pub(super) fn binary(
    ctx: &Context,
    x: &Tensor,
    y: &Tensor,
    semantic: ops::PointwiseSemantic,
) -> Result<Tensor> {
    let mut out = ctx.allocate_output(x.shape().clone(), x.dtype())?;
    let mut args = ops::PointwiseArgs::new(semantic, x, &mut out);
    args.secondary = Some(y);
    ops::pointwise(ctx, args)?;
    Ok(out)
}

pub(super) fn host_tensor(shape: Vec<usize>, values: &[f32], dtype: DType) -> Result<Tensor> {
    match dtype {
        DType::F32 => Tensor::from_f32(shape, values),
        DType::BF16 => Tensor::from_bf16(
            shape,
            &values
                .iter()
                .copied()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>(),
        ),
        _ => Err(Error::Other(
            "Fast-WAM checkpoint expects BF16 or F32 weights".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA"]
    fn fastwam_convolution_rounds_before_bias() {
        let ctx = Context::new(0).unwrap();
        let upload_values = |shape, values: &[f32]| {
            upload(&ctx, &host_tensor(shape, values, DType::BF16).unwrap()).unwrap()
        };
        let input = upload_values(vec![1, 2], &[1., 1.]);
        let weight = upload_values(vec![2, 1], &[1., 0.00390625]);
        let bias = upload_values(vec![1], &[-1.]);
        let policy = ops::GemmPolicy::default();
        // 1 + 2^-8 is the BF16 halfway point: convolution rounds it to 1
        // before cancellation, whereas Linear retains 2^-8 after adding -1.
        let conv = convolution_projection(&ctx, &input, &weight, Some(&bias), &policy).unwrap();
        let linear = linear(&ctx, &input, &weight, Some(&bias), &policy).unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(
            transfers::to_cpu(&conv).unwrap().to_f32_vec().unwrap(),
            vec![0.]
        );
        assert_eq!(
            transfers::to_cpu(&linear).unwrap().to_f32_vec().unwrap(),
            vec![0.00390625]
        );
    }
}
