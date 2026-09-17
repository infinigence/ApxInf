//! Native 3D feature encoder following multi-camera voxel aggregation.
use super::backend::{kernels, transfers, Context};
use super::perception_fpn::take;
use apxinf_core::{DType, Device, Error, Result, Tensor};
use kernels::{convolution, norm};
use std::collections::HashMap;

struct Layer {
    weight: Tensor,
    bias: Tensor,
    mean: Tensor,
    invstd: Tensor,
    gamma: Tensor,
    beta: Tensor,
}

pub struct PerceptionViewEncoder {
    channels: usize,
    layers: Vec<Layer>,
}

impl PerceptionViewEncoder {
    pub fn load(
        ctx: &Context,
        weights: &mut HashMap<String, Tensor>,
        channels: usize,
    ) -> Result<Self> {
        if channels == 0 {
            return Err(Error::Other("empty perception view encoder".into()));
        }
        let mut layers = Vec::with_capacity(3);
        for i in 0..3 {
            let p = format!("bev_modeling.view_trans.conv_layer.{i}");
            let name = format!("{p}.1.running_var");
            let var = weights
                .remove(&name)
                .ok_or_else(|| Error::Other(format!("missing {name}")))?;
            if var.device() != Device::Cpu
                || var.shape().dims() != [channels]
                || !matches!(var.dtype(), DType::F32 | DType::BF16)
            {
                return Err(Error::Other(format!("invalid {name}")));
            }
            let inv: Vec<f32> = var
                .to_f32_vec()?
                .into_iter()
                .map(|v| {
                    let rounded = half::bf16::from_f32(v).to_f32();
                    (1.0 / (f64::from(rounded) + 1e-5).sqrt()) as f32
                })
                .collect();
            if inv.iter().any(|v| !v.is_finite()) {
                return Err(Error::Other(format!("invalid running variance in {name}")));
            }
            let invstd =
                transfers::to_cuda(&Tensor::from_f32(vec![channels], &inv)?, ctx.device_id())?;
            layers.push(Layer {
                weight: take(
                    ctx,
                    weights,
                    &format!("{p}.0.weight"),
                    &[channels, channels, 3, 3, 3],
                )?,
                bias: take(ctx, weights, &format!("{p}.0.bias"), &[channels])?,
                mean: take(ctx, weights, &format!("{p}.1.running_mean"), &[channels])?,
                gamma: take(ctx, weights, &format!("{p}.1.weight"), &[channels])?,
                beta: take(ctx, weights, &format!("{p}.1.bias"), &[channels])?,
                invstd,
            });
        }
        Ok(Self { channels, layers })
    }

    /// Input is the BF16 sum over cameras/sweeps in `[B,C,D,H,W]` order.
    pub fn forward(&self, ctx: &Context, input: &Tensor) -> Result<Tensor> {
        let d = input.shape().dims();
        if d.len() != 5 || d[1] != self.channels {
            return Err(Error::Other(
                "view encoder expects matching NCDHW features".into(),
            ));
        }
        let mut x = input.clone();
        for layer in &self.layers {
            x = convolution::conv3d(
                ctx,
                &x,
                &layer.weight,
                Some(&layer.bias),
                convolution::Conv3dSpec {
                    padding: [1, 1, 1],
                    ..Default::default()
                },
            )?;
            x = norm::batch_relu_bf16(
                ctx,
                &x,
                &layer.mean,
                &layer.invstd,
                &layer.gamma,
                &layer.beta,
            )?;
        }
        Ok(x)
    }
}
