use super::super::backend::{transfers, Context};
use apxinf_core::{DType, Error, Result, Tensor};
use std::collections::HashMap;

pub(in crate::dreamzero) struct Conv {
    pub input: usize,
    pub padded_input: usize,
    pub channel_block: usize,
    pub output: usize,
    pub kernel: [usize; 3],
    pub weight: Tensor,
    pub native_weight: Option<Tensor>,
    pub bias: Tensor,
}
pub(in crate::dreamzero) struct Residual {
    pub norm1: Tensor,
    pub conv1: Conv,
    pub norm2: Tensor,
    pub conv2: Conv,
    pub shortcut: Option<Conv>,
}
pub(in crate::dreamzero) enum Down {
    Residual(Residual),
    Spatial { conv: Conv, temporal: Option<Conv> },
}
pub(in crate::dreamzero) struct Attention {
    pub norm: Tensor,
    pub q: Conv,
    pub k: Conv,
    pub v: Conv,
    pub out: Conv,
}
pub(in crate::dreamzero) struct VaeWeights {
    pub input: Conv,
    pub down: Vec<Down>,
    pub middle1: Residual,
    pub attention: Attention,
    pub middle2: Residual,
    pub final_norm: Tensor,
    pub final_conv: Conv,
    pub moments: Conv,
    pub mean: Tensor,
    pub inverse_std: Tensor,
}
fn take(map: &mut HashMap<String, Tensor>, name: &str, shape: &[usize]) -> Result<Tensor> {
    let name = format!("action_head.vae.model.{name}");
    let x = map
        .remove(&name)
        .ok_or_else(|| Error::Other(format!("missing DreamZero {name}")))?;
    if x.dtype() != DType::BF16 || x.shape().dims() != shape {
        return Err(Error::Other(format!(
            "DreamZero {name}: expected BF16 {shape:?}, got {} {:?}",
            x.dtype(),
            x.shape().dims()
        )));
    }
    Ok(x)
}
fn upload(ctx: &Context, x: Tensor) -> Result<Tensor> {
    transfers::to_cuda(&x, ctx.device_id())
}
fn norm(
    ctx: &Context,
    map: &mut HashMap<String, Tensor>,
    name: &str,
    dim: usize,
    images: bool,
) -> Result<Tensor> {
    let shape = if images {
        vec![dim, 1, 1]
    } else {
        vec![dim, 1, 1, 1]
    };
    upload(ctx, take(map, name, &shape)?.reshape(vec![1, dim])?)
}
fn pack(
    ctx: &Context,
    data: &[half::bf16],
    input: usize,
    output: usize,
    kernel: [usize; 3],
    bias: Tensor,
    channel_block: usize,
) -> Result<Conv> {
    let taps: usize = kernel.iter().product();
    let padded_input = input.div_ceil(channel_block) * channel_block;
    let columns = padded_input * taps;
    let mut matrix = vec![half::bf16::ZERO; output * columns];
    for o in 0..output {
        for c in 0..input {
            for tap in 0..taps {
                let k = ((c / channel_block) * taps + tap) * channel_block + c % channel_block;
                matrix[k * output + o] = data[(o * input + c) * taps + tap];
            }
        }
    }
    Ok(Conv {
        native_weight: None,
        input,
        padded_input,
        channel_block,
        output,
        kernel,
        weight: upload(ctx, Tensor::from_bf16(vec![columns, output], &matrix)?)?,
        bias: upload(ctx, bias.reshape(vec![1, output])?)?,
    })
}
fn conv(
    ctx: &Context,
    map: &mut HashMap<String, Tensor>,
    name: &str,
    input: usize,
    output: usize,
    kernel: [usize; 3],
    is_2d: bool,
) -> Result<Conv> {
    let shape = if is_2d {
        vec![output, input, kernel[1], kernel[2]]
    } else {
        vec![output, input, kernel[0], kernel[1], kernel[2]]
    };
    let weight = take(map, &format!("{name}.weight"), &shape)?;
    let bias = take(map, &format!("{name}.bias"), &[output])?;
    let mut result = pack(
        ctx,
        weight.as_bf16()?,
        input,
        output,
        kernel,
        bias,
        input.min(64),
    )?;
    // The RGB stem and four 88x160 convolutions require the reference's
    // convolution reduction order at BF16 rounding boundaries.
    // Preserve OIDHW weights for the native convolution provider; remaining
    // layers use the aligned unfold/GEMM representation.
    if kernel == [3; 3]
        && (name == "encoder.conv1"
            || name.starts_with("encoder.downsamples.6.")
            || name.starts_with("encoder.downsamples.7."))
    {
        result.native_weight = Some(upload(ctx, weight)?);
    }
    Ok(result)
}
fn residual(
    ctx: &Context,
    map: &mut HashMap<String, Tensor>,
    name: &str,
    input: usize,
    output: usize,
) -> Result<Residual> {
    Ok(Residual {
        norm1: norm(ctx, map, &format!("{name}.residual.0.gamma"), input, false)?,
        conv1: conv(
            ctx,
            map,
            &format!("{name}.residual.2"),
            input,
            output,
            [3; 3],
            false,
        )?,
        norm2: norm(ctx, map, &format!("{name}.residual.3.gamma"), output, false)?,
        conv2: conv(
            ctx,
            map,
            &format!("{name}.residual.6"),
            output,
            output,
            [3; 3],
            false,
        )?,
        shortcut: if input != output {
            Some(conv(
                ctx,
                map,
                &format!("{name}.shortcut"),
                input,
                output,
                [1; 3],
                false,
            )?)
        } else {
            None
        },
    })
}
impl VaeWeights {
    pub fn from_map(ctx: &Context, map: &mut HashMap<String, Tensor>) -> Result<Self> {
        let input = conv(ctx, map, "encoder.conv1", 3, 96, [3; 3], false)?;
        let mut down = Vec::new();
        let mut index = 0;
        let mut input_dim = 96;
        for (stage, dim) in [96, 192, 384, 384].into_iter().enumerate() {
            for _ in 0..2 {
                down.push(Down::Residual(residual(
                    ctx,
                    map,
                    &format!("encoder.downsamples.{index}"),
                    input_dim,
                    dim,
                )?));
                input_dim = dim;
                index += 1;
            }
            if stage < 3 {
                let prefix = format!("encoder.downsamples.{index}");
                down.push(Down::Spatial {
                    conv: conv(
                        ctx,
                        map,
                        &format!("{prefix}.resample.1"),
                        dim,
                        dim,
                        [1, 3, 3],
                        true,
                    )?,
                    temporal: if stage > 0 {
                        Some(conv(
                            ctx,
                            map,
                            &format!("{prefix}.time_conv"),
                            dim,
                            dim,
                            [3, 1, 1],
                            false,
                        )?)
                    } else {
                        None
                    },
                });
                index += 1;
            }
        }
        let middle1 = residual(ctx, map, "encoder.middle.0", 384, 384)?;
        let qkv = take(map, "encoder.middle.1.to_qkv.weight", &[1152, 384, 1, 1])?;
        let bias = take(map, "encoder.middle.1.to_qkv.bias", &[1152])?;
        let split = |i: usize| {
            pack(
                ctx,
                &qkv.as_bf16()?[i * 384 * 384..(i + 1) * 384 * 384],
                384,
                384,
                [1; 3],
                Tensor::from_bf16(vec![384], &bias.as_bf16()?[i * 384..(i + 1) * 384])?,
                64,
            )
        };
        let attention = Attention {
            norm: norm(ctx, map, "encoder.middle.1.norm.gamma", 384, true)?,
            q: split(0)?,
            k: split(1)?,
            v: split(2)?,
            out: conv(ctx, map, "encoder.middle.1.proj", 384, 384, [1; 3], true)?,
        };
        let middle2 = residual(ctx, map, "encoder.middle.2", 384, 384)?;
        let final_norm = norm(ctx, map, "encoder.head.0.gamma", 384, false)?;
        let final_conv = conv(ctx, map, "encoder.head.2", 384, 32, [3; 3], false)?;
        let moments = conv(ctx, map, "conv1", 32, 32, [1; 3], false)?;
        let mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715,
            0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
        ];
        let std = [
            2.8184f32, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526,
            2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
        ];
        let constant =
            |values: Vec<half::bf16>| upload(ctx, Tensor::from_bf16(vec![1, 16], &values)?);
        let mean = constant(mean.into_iter().map(half::bf16::from_f32).collect())?;
        let inverse_std = constant(
            std.into_iter()
                .map(|s| half::bf16::from_f32(1. / s))
                .collect(),
        )?;
        Ok(Self {
            input,
            down,
            middle1,
            attention,
            middle2,
            final_norm,
            final_conv,
            moments,
            mean,
            inverse_std,
        })
    }
}
