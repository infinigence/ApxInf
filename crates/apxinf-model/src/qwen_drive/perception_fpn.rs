//! Native SimpleFPN composition for the Qwen-Drive perception feature taps.
//! Architecture follows the Apache-2.0 Qwen-Drive SimpleFPN implementation.
use super::backend::{kernels, transfers, Context};
use apxinf_core::{DType, Device, Error, Result, Tensor};
use kernels::{
    convolution::{conv2d, conv_transpose2d, Conv2dSpec},
    linear_attention, norm, pooling,
};
use std::collections::HashMap;

struct Conv {
    weight: Tensor,
    bias: Option<Tensor>,
}
struct Norm {
    weight: Tensor,
    bias: Tensor,
}
enum Resize {
    Identity,
    Pool,
    Up(Conv),
    UpTwice {
        first: Conv,
        norm: Norm,
        second: Conv,
    },
}
struct Stage {
    resize: Resize,
    first: Conv,
    first_norm: Norm,
    second: Conv,
    second_norm: Norm,
}

pub struct PerceptionFpn {
    input_channels: usize,
    stages: Vec<Stage>,
}

pub(super) fn take(
    ctx: &Context,
    map: &mut HashMap<String, Tensor>,
    name: &str,
    shape: &[usize],
) -> Result<Tensor> {
    let t = map
        .remove(name)
        .ok_or_else(|| Error::Other(format!("missing perception weight {name}")))?;
    if t.shape().dims() != shape || t.device() != Device::Cpu {
        return Err(Error::Other(format!(
            "perception weight {name} must be CPU {shape:?}, got {:?} {:?}",
            t.shape().dims(),
            t.device()
        )));
    }
    let t = if t.dtype() == DType::BF16 {
        t
    } else if t.dtype() == DType::F32 {
        let values: Vec<half::bf16> = t
            .to_f32_vec()?
            .iter()
            .map(|&v| half::bf16::from_f32(v))
            .collect();
        Tensor::from_bf16(t.shape().clone(), &values)?
    } else {
        return Err(Error::Other(format!(
            "unsupported perception weight dtype for {name}"
        )));
    };
    transfers::to_cuda(&t, ctx.device_id())
}
fn convolution(
    ctx: &Context,
    map: &mut HashMap<String, Tensor>,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    transpose: bool,
    bias: bool,
) -> Result<Conv> {
    let shape = if transpose {
        vec![input, output, kernel, kernel]
    } else {
        vec![output, input, kernel, kernel]
    };
    Ok(Conv {
        weight: take(ctx, map, &format!("{prefix}.weight"), &shape)?,
        bias: if bias {
            Some(take(ctx, map, &format!("{prefix}.bias"), &[output])?)
        } else {
            None
        },
    })
}
fn normalization(
    ctx: &Context,
    map: &mut HashMap<String, Tensor>,
    prefix: &str,
    channels: usize,
) -> Result<Norm> {
    Ok(Norm {
        weight: take(ctx, map, &format!("{prefix}.weight"), &[channels])?,
        bias: take(ctx, map, &format!("{prefix}.bias"), &[channels])?,
    })
}
impl Norm {
    fn forward(&self, ctx: &Context, x: &Tensor) -> Result<Tensor> {
        norm::channel_layer_bf16_rounded(ctx, x, &self.weight, &self.bias, 1e-6)
    }
}
impl Conv {
    fn forward(&self, ctx: &Context, x: &Tensor, padding: usize) -> Result<Tensor> {
        conv2d(
            ctx,
            x,
            &self.weight,
            self.bias.as_ref(),
            Conv2dSpec {
                padding: [padding, padding],
                ..Default::default()
            },
        )
    }
    fn up(&self, ctx: &Context, x: &Tensor) -> Result<Tensor> {
        conv_transpose2d(
            ctx,
            x,
            &self.weight,
            self.bias.as_ref(),
            Conv2dSpec {
                stride: [2, 2],
                ..Default::default()
            },
        )
    }
}
impl PerceptionFpn {
    /// Load one checkpoint-owned pyramid (adaptor: [4,2,1,0.5], ViT neck: [1]).
    pub fn load(
        ctx: &Context,
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        scales: &[f32],
    ) -> Result<Self> {
        if input_channels == 0 || output_channels == 0 || scales.is_empty() {
            return Err(Error::Other("empty perception pyramid".into()));
        }
        let mut stages = Vec::with_capacity(scales.len());
        for (index, &scale) in scales.iter().enumerate() {
            let prefix = format!("{prefix}.stages.{index}");
            let (resize, start, width) = if scale == 4.0 {
                if input_channels % 4 != 0 {
                    return Err(Error::Other(
                        "4x pyramid channels must be divisible by four".into(),
                    ));
                }
                (
                    Resize::UpTwice {
                        first: convolution(
                            ctx,
                            map,
                            &format!("{prefix}.0"),
                            input_channels,
                            input_channels / 2,
                            2,
                            true,
                            true,
                        )?,
                        norm: normalization(ctx, map, &format!("{prefix}.1"), input_channels / 2)?,
                        second: convolution(
                            ctx,
                            map,
                            &format!("{prefix}.3"),
                            input_channels / 2,
                            input_channels / 4,
                            2,
                            true,
                            true,
                        )?,
                    },
                    4,
                    input_channels / 4,
                )
            } else if scale == 2.0 {
                if input_channels % 2 != 0 {
                    return Err(Error::Other("2x pyramid channels must be even".into()));
                }
                (
                    Resize::Up(convolution(
                        ctx,
                        map,
                        &format!("{prefix}.0"),
                        input_channels,
                        input_channels / 2,
                        2,
                        true,
                        true,
                    )?),
                    1,
                    input_channels / 2,
                )
            } else if scale == 1.0 {
                (Resize::Identity, 0, input_channels)
            } else if scale == 0.5 {
                (Resize::Pool, 1, input_channels)
            } else {
                return Err(Error::Other(format!(
                    "unsupported perception pyramid scale {scale}"
                )));
            };
            stages.push(Stage {
                resize,
                first: convolution(
                    ctx,
                    map,
                    &format!("{prefix}.{start}"),
                    width,
                    output_channels,
                    1,
                    false,
                    false,
                )?,
                first_norm: normalization(
                    ctx,
                    map,
                    &format!("{prefix}.{}", start + 1),
                    output_channels,
                )?,
                second: convolution(
                    ctx,
                    map,
                    &format!("{prefix}.{}", start + 2),
                    output_channels,
                    output_channels,
                    3,
                    false,
                    false,
                )?,
                second_norm: normalization(
                    ctx,
                    map,
                    &format!("{prefix}.{}", start + 3),
                    output_channels,
                )?,
            });
        }
        Ok(Self {
            input_channels,
            stages,
        })
    }

    /// Input and each output are contiguous NCHW BF16 feature maps on the device.
    pub fn forward(&self, ctx: &Context, x: &Tensor) -> Result<Vec<Tensor>> {
        if x.shape().dims().len() != 4 || x.shape().dims()[1] != self.input_channels {
            return Err(Error::Other(
                "perception pyramid input channels mismatch".into(),
            ));
        }
        let mut outputs = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            let resized = match &stage.resize {
                Resize::Identity => x.clone(),
                Resize::Pool => pooling::max_pool2x2_bf16(ctx, x)?,
                Resize::Up(conv) => conv.up(ctx, x)?,
                Resize::UpTwice {
                    first,
                    norm,
                    second,
                } => {
                    let y = norm.forward(ctx, &first.up(ctx, x)?)?;
                    second.up(ctx, &linear_attention::gelu_exact(ctx, &y)?)?
                }
            };
            let y = stage
                .first_norm
                .forward(ctx, &stage.first.forward(ctx, &resized, 0)?)?;
            let y = stage
                .second_norm
                .forward(ctx, &stage.second.forward(ctx, &y, 1)?)?;
            outputs.push(y);
        }
        Ok(outputs)
    }
}
