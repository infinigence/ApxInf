//! Device weight representation for the BF16 model, including explicit F32 tensors.
use super::host::{host_tensor, validate_tensor, Bias, Store};
use crate::cuda_fastwam::backend::{self, ops, Context};
use apxinf_core::{DType, Error, Result, Tensor};
pub(in crate::cuda_fastwam) mod dit;
pub(in crate::cuda_fastwam) mod text;
pub(in crate::cuda_fastwam) mod vae;
pub(in crate::cuda_fastwam) struct LinearWeights {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}
impl LinearWeights {
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

pub(in crate::cuda_fastwam) fn upload_tensor(
    store: &mut Store,
    ctx: &Context,
    name: &str,
) -> Result<Tensor> {
    backend::upload(ctx, &store.take(name)?)
}

pub(in crate::cuda_fastwam) fn load_linear(
    store: &mut Store,
    ctx: &Context,
    prefix: &str,
    bias: Bias,
) -> Result<LinearWeights> {
    // Validate checkpoint semantics before allocating device weights.
    let (raw, bias) = store.take_linear(prefix, bias)?;
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
    Ok(LinearWeights { weight, bias })
}

fn cast_weight(ctx: &Context, x: &Tensor, dtype: DType) -> Result<Tensor> {
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

/// Loaded device representations; fixed timestep computation belongs to Blocks.
pub(in crate::cuda_fastwam) struct Bf16Weights {
    pub video: (dit::ExpertWeights, dit::TimeWeights),
    pub action: (dit::ExpertWeights, dit::TimeWeights),
    pub proprio: LinearWeights,
    pub text: text::TextEncoderWeights,
    pub vae: vae::VaeWeights,
    pub rgb_table: Tensor,
}

impl Bf16Weights {
    pub(in crate::cuda_fastwam) fn load(
        ctx: &Context,
        path: &std::path::Path,
        text_path: &std::path::Path,
        vae_path: &std::path::Path,
    ) -> Result<Self> {
        let mut store = Store::load(path)?;
        let video = dit::ExpertWeights::load(&mut store, ctx, true)?;
        let action = dit::ExpertWeights::load(&mut store, ctx, false)?;
        let proprio = load_linear(&mut store, ctx, "model.proprio_encoder", Bias::Required)?;
        proprio.validate("proprio encoder", 8, 4096)?;
        drop(store);
        let rgb = (0..256)
            .map(|v| {
                let unit = half::bf16::from_f32(v as f32 / 255.).to_f32();
                half::bf16::from_f32(half::bf16::from_f32(unit * 2.).to_f32() - 1.).to_f32()
            })
            .collect::<Vec<_>>();
        Ok(Self {
            video,
            action,
            proprio,
            text: text::TextEncoderWeights::load(ctx, text_path)?,
            vae: vae::VaeWeights::load(ctx, vae_path, DType::BF16)?,
            rgb_table: backend::upload(ctx, &host_tensor(vec![256, 1], &rgb, DType::BF16)?)?,
        })
    }
}
