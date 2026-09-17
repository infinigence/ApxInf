//! Native Qwen-Drive DepthNet: residual convolutions and atrous spatial pooling.
//! Topology and checkpoint names follow the Apache-2.0 Qwen-Drive reference.
use super::backend::{kernels, Context};
use super::perception_fpn::take;
use apxinf_core::{Error, Result, Tensor};
use kernels::{activation, convolution, elementwise, norm, pooling};
use std::collections::HashMap;

struct Conv {
    weight: Tensor,
    bias: Option<Tensor>,
    spec: convolution::Conv2dSpec,
}
struct ConvNorm {
    conv: Conv,
    gamma: Tensor,
    beta: Tensor,
}

impl Conv {
    fn load(
        ctx: &Context,
        map: &mut HashMap<String, Tensor>,
        name: &str,
        input: usize,
        output: usize,
        kernel: usize,
        padding: usize,
        dilation: usize,
        bias: bool,
    ) -> Result<Self> {
        Ok(Self {
            weight: take(
                ctx,
                map,
                &format!("{name}.weight"),
                &[output, input, kernel, kernel],
            )?,
            bias: if bias {
                Some(take(ctx, map, &format!("{name}.bias"), &[output])?)
            } else {
                None
            },
            spec: convolution::Conv2dSpec {
                padding: [padding, padding],
                dilation: [dilation, dilation],
                ..Default::default()
            },
        })
    }
    fn forward(&self, ctx: &Context, x: &Tensor) -> Result<Tensor> {
        convolution::conv2d(ctx, x, &self.weight, self.bias.as_ref(), self.spec)
    }
}
impl ConvNorm {
    fn load(
        ctx: &Context,
        map: &mut HashMap<String, Tensor>,
        conv: &str,
        norm: &str,
        input: usize,
        output: usize,
        kernel: usize,
        padding: usize,
        dilation: usize,
        bias: bool,
    ) -> Result<Self> {
        Ok(Self {
            conv: Conv::load(
                ctx, map, conv, input, output, kernel, padding, dilation, bias,
            )?,
            gamma: take(ctx, map, &format!("{norm}.weight"), &[output])?,
            beta: take(ctx, map, &format!("{norm}.bias"), &[output])?,
        })
    }
    fn forward(&self, ctx: &Context, x: &Tensor, relu: bool) -> Result<Tensor> {
        let x = self.conv.forward(ctx, x)?;
        let x = norm::group_bf16_rounded(ctx, &x, &self.gamma, &self.beta, 32, 1e-5)?;
        if relu {
            activation::relu_bf16(ctx, &x)
        } else {
            Ok(x)
        }
    }
}

pub struct PerceptionDepthNet {
    channels: usize,
    reduce: ConvNorm,
    residual: Vec<(ConvNorm, ConvNorm)>,
    atrous: Vec<ConvNorm>,
    global: ConvNorm,
    combine: ConvNorm,
    output: Conv,
}
impl PerceptionDepthNet {
    pub fn load(
        ctx: &Context,
        map: &mut HashMap<String, Tensor>,
        channels: usize,
        depth_channels: usize,
        aspp_channels: usize,
    ) -> Result<Self> {
        if channels == 0
            || channels % 32 != 0
            || aspp_channels == 0
            || aspp_channels % 32 != 0
            || depth_channels == 0
        {
            return Err(Error::Other("invalid DepthNet channel geometry".into()));
        }
        let p = "bev_modeling.depth_net";
        let reduce = ConvNorm::load(
            ctx,
            map,
            &format!("{p}.reduce_conv.0"),
            &format!("{p}.reduce_conv.1"),
            channels,
            channels,
            3,
            1,
            1,
            true,
        )?;
        let mut residual = Vec::new();
        for i in 0..3 {
            let p = format!("{p}.depth_conv.{i}");
            residual.push((
                ConvNorm::load(
                    ctx,
                    map,
                    &format!("{p}.conv1"),
                    &format!("{p}.gn1"),
                    channels,
                    channels,
                    3,
                    1,
                    1,
                    false,
                )?,
                ConvNorm::load(
                    ctx,
                    map,
                    &format!("{p}.conv2"),
                    &format!("{p}.gn2"),
                    channels,
                    channels,
                    3,
                    1,
                    1,
                    false,
                )?,
            ));
        }
        let ap = format!("{p}.depth_conv.3");
        let mut atrous = Vec::new();
        for (i, dilation) in [1, 6, 12, 18].into_iter().enumerate() {
            let kernel = if i == 0 { 1 } else { 3 };
            let padding = if i == 0 { 0 } else { dilation };
            atrous.push(ConvNorm::load(
                ctx,
                map,
                &format!("{ap}.aspp{}.atrous_conv", i + 1),
                &format!("{ap}.aspp{}.bn", i + 1),
                channels,
                aspp_channels,
                kernel,
                padding,
                dilation,
                false,
            )?);
        }
        let global = ConvNorm::load(
            ctx,
            map,
            &format!("{ap}.global_avg_pool.1"),
            &format!("{ap}.global_avg_pool.2"),
            channels,
            aspp_channels,
            1,
            0,
            1,
            false,
        )?;
        let combined = aspp_channels
            .checked_mul(5)
            .ok_or_else(|| Error::Other("DepthNet channels overflow".into()))?;
        let combine = ConvNorm::load(
            ctx,
            map,
            &format!("{ap}.conv1"),
            &format!("{ap}.bn1"),
            combined,
            channels,
            1,
            0,
            1,
            false,
        )?;
        let output = Conv::load(
            ctx,
            map,
            &format!("{p}.depth_conv.4"),
            channels,
            depth_channels,
            1,
            0,
            1,
            true,
        )?;
        Ok(Self {
            channels,
            reduce,
            residual,
            atrous,
            global,
            combine,
            output,
        })
    }

    /// Return NCHW depth logits. The caller owns depth-axis softmax and projection.
    pub fn forward(&self, ctx: &Context, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape().dims();
        if shape.len() != 4 || shape[1] != self.channels {
            return Err(Error::Other(
                "DepthNet expects NCHW features with matching channels".into(),
            ));
        }
        let mut x = self.reduce.forward(ctx, input, true)?;
        for (first, second) in &self.residual {
            let y = second.forward(ctx, &first.forward(ctx, &x, true)?, false)?;
            x = activation::relu_bf16(ctx, &elementwise::add(ctx, &x, &y)?)?;
        }
        let mut branches = self
            .atrous
            .iter()
            .map(|c| c.forward(ctx, &x, true))
            .collect::<Result<Vec<_>>>()?;
        let global = self
            .global
            .forward(ctx, &pooling::global_mean_bf16(ctx, &x)?, true)?;
        branches.push(elementwise::expand_spatial_bf16(
            ctx, &global, shape[2], shape[3],
        )?);
        let refs = branches.iter().collect::<Vec<_>>();
        let combined = elementwise::concat_channels_bf16(ctx, &refs)?;
        self.output
            .forward(ctx, &self.combine.forward(ctx, &combined, true)?)
    }
}
