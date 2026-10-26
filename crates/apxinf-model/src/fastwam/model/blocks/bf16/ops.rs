//! Precision-specific projection, activation and rounding composition.
use crate::fastwam::backend::{ops, Context};
use crate::fastwam::weights::{cast, Linear};
use apxinf_core::{Error, Result, Shape, Tensor};

pub(in crate::fastwam) fn reshape(x: &Tensor, dims: Vec<usize>) -> Result<Tensor> {
    x.reshape(Shape::new(dims))
}

pub(in crate::fastwam) fn linear(
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
pub(in crate::fastwam) fn convolution_projection(
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

pub(in crate::fastwam) fn activation(
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

pub(in crate::fastwam) fn add(ctx: &Context, x: &Tensor, y: &Tensor) -> Result<Tensor> {
    let mut out = ctx.allocate_output(x.shape().clone(), x.dtype())?;
    let mut args = ops::PointwiseArgs::new(ops::PointwiseSemantic::Add, x, &mut out);
    args.secondary = Some(y);
    ops::pointwise(ctx, args)?;
    Ok(out)
}

pub(in crate::fastwam) fn binary(
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

impl Linear {
    pub(in crate::fastwam) fn forward_conv(
        &self,
        ctx: &Context,
        x: &Tensor,
        policy: &ops::GemmPolicy,
    ) -> Result<Tensor> {
        convolution_projection(ctx, x, &self.weight, self.bias.as_ref(), policy)
    }

    pub(in crate::fastwam) fn forward(
        &self,
        ctx: &Context,
        x: &Tensor,
        policy: &ops::GemmPolicy,
    ) -> Result<Tensor> {
        linear(ctx, x, &self.weight, self.bias.as_ref(), policy)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fastwam::backend::{transfers, upload};
    use crate::fastwam::weights::host_tensor;
    use apxinf_core::DType;

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
