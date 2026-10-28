//! Device weight representation for the BF16 model, including explicit F32 tensors.
use super::host::{host_tensor, validate_tensor, Bias, Store};
use crate::cuda_fastwam::backend::{self, ops, Context};
use apxinf_core::{DType, Error, Result, Tensor};
pub(in crate::cuda_fastwam) mod dit;
pub(in crate::cuda_fastwam) mod text;
pub(in crate::cuda_fastwam) mod vae;
pub(in crate::cuda_fastwam) struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}
impl Linear {
    pub(in crate::cuda_fastwam) fn validate(
        &self,
        name: &str,
        input: usize,
        output: usize,
    ) -> Result<()> {
        validate_tensor(name, &self.weight, &[input, output], DType::BF16)?;
        if let Some(bias) = &self.bias {
            validate_tensor(name, bias, &[output], DType::BF16)?;
        }
        Ok(())
    }
}

impl Store {
    pub(in crate::cuda_fastwam) fn upload(&mut self, ctx: &Context, name: &str) -> Result<Tensor> {
        backend::upload(ctx, &self.take(name)?)
    }

    pub(in crate::cuda_fastwam) fn linear(
        &mut self,
        ctx: &Context,
        prefix: &str,
        bias: Bias,
    ) -> Result<Linear> {
        // Validate checkpoint semantics before allocating device weights.
        let (raw, bias) = self.take_linear(prefix, bias)?;
        let (n, k) = (raw.shape().dims()[0], raw.shape().dims()[1]);
        let src = raw.to_f32_vec()?;
        let mut dst = vec![0.; src.len()];
        for j in 0..n {
            for i in 0..k {
                dst[i * n + j] = src[j * k + i];
            }
        }
        let weight = backend::upload(ctx, &host_tensor(vec![k, n], &dst, raw.dtype())?)?;
        let bias = bias
            .as_ref()
            .map(|bias| backend::upload(ctx, bias))
            .transpose()?;
        Ok(Linear { weight, bias })
    }
}
pub(in crate::cuda_fastwam) fn cast(ctx: &Context, x: &Tensor, dtype: DType) -> Result<Tensor> {
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
