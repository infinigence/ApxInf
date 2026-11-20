//! Audio VAE decoder composition (BigVGAN): a 1x1 lift, a k=7 convolution, seven stages of transposed convolution followed by three
//! anti-aliased residual blocks whose outputs are averaged, a final anti-aliased activation and a k=7 projection to one channel. FP32 throughout, on
//! the 1-D convolution operators of `apxinf-cuda-new`. The two stereo channels are decoded as a batch of two mono signals.
use super::util::alloc;
use crate::cuda_minimax_h3::backend::ops::{
    self, Conv1dArgs, Conv1dPadding, ConvTranspose1dArgs, SnakeBetaArgs,
};
use crate::cuda_minimax_h3::backend::CudaContext;
use crate::cuda_minimax_h3::config::*;
use crate::cuda_minimax_h3::weights::{AudioActivation, AudioConv, AudioResBlock, AudioVaeWeights};
use apxinf_core::{DType, Result, Tensor};
use std::rc::Rc;

/// Taps of the anti-aliasing filters (`UpSample1d` / `LowPassFilter1d` of the reference): replicate padding of the input before the
/// transposed convolution, the columns cropped from its output, and the left padding of the low-pass convolution.
const UP_PAD: usize = AUDIO_FILTER / 2 - 1;
const UP_CROP: usize = UP_PAD * 2 + (AUDIO_FILTER - 2) / 2;
const DOWN_PAD_LEFT: usize = AUDIO_FILTER / 2 - 1;

pub(in crate::cuda_minimax_h3) struct AudioVaeBlocks {
    ctx: Rc<CudaContext>,
    weights: Rc<AudioVaeWeights>,
}

fn dims(x: &Tensor) -> [usize; 3] {
    let d = x.shape().dims();
    [d[0], d[1], d[2]]
}

impl AudioVaeBlocks {
    pub fn new(ctx: Rc<CudaContext>, weights: Rc<AudioVaeWeights>) -> Self {
        Self { ctx, weights }
    }

    pub fn weights(&self) -> &AudioVaeWeights {
        &self.weights
    }

    /// `y = conv1d(x)` with `dilation` and zero padding that keeps the length (`(kernel - 1) * dilation / 2` on the left).
    fn same_conv(&self, x: &Tensor, conv: &AudioConv, dilation: usize) -> Result<Tensor> {
        let [batch, _, len] = dims(x);
        let out_channels = conv.weight.shape().dims()[0];
        let out = alloc(&self.ctx, &[batch, out_channels, len], DType::F32)?;
        ops::conv1d(
            &self.ctx,
            Conv1dArgs {
                input: x,
                weight: &conv.weight,
                bias: conv.bias.as_ref(),
                out: &out,
                stride: 1,
                dilation,
                padding: Conv1dPadding::Zeros {
                    before: (conv.kernel - 1) * dilation / 2,
                },
                depthwise: false,
            },
        )?;
        Ok(out)
    }

    /// `Activation1d`: x2 upsample with a Kaiser-sinc filter, SnakeBeta, x2 low-pass downsample.
    fn activation(&self, x: &Tensor, act: &AudioActivation) -> Result<Tensor> {
        let ctx = &*self.ctx;
        let [batch, channels, len] = dims(x);
        let up = alloc(ctx, &[batch, channels, 2 * len], DType::F32)?;
        ops::conv_transpose1d(
            ctx,
            ConvTranspose1dArgs {
                input: x,
                weight: &act.up_filter,
                bias: None,
                out: &up,
                stride: 2,
                replicate_pad: UP_PAD,
                crop_left: UP_CROP,
                groups: channels,
                scale: 2.0,
            },
        )?;
        ops::snake_beta(
            ctx,
            SnakeBetaArgs {
                input: &up,
                alpha: &act.alpha,
                beta: &act.beta,
                out: &up,
            },
        )?;
        let down = alloc(ctx, &[batch, channels, len], DType::F32)?;
        ops::conv1d(
            ctx,
            Conv1dArgs {
                input: &up,
                weight: &act.down_filter,
                bias: None,
                out: &down,
                stride: 2,
                dilation: 1,
                padding: Conv1dPadding::Replicate {
                    before: DOWN_PAD_LEFT,
                },
                depthwise: true,
            },
        )?;
        Ok(down)
    }

    /// `AMPBlock1`: three units of (activation, dilated conv, activation, conv) with a skip connection after each.
    fn resblock(&self, x: &Tensor, block: &AudioResBlock) -> Result<Tensor> {
        let ctx = &*self.ctx;
        let mut x = x.clone();
        for unit in 0..block.dilations.len() {
            let t = self.activation(&x, &block.activations[2 * unit])?;
            let t = self.same_conv(&t, &block.convs1[unit], block.dilations[unit])?;
            let t = self.activation(&t, &block.activations[2 * unit + 1])?;
            let t = self.same_conv(&t, &block.convs2[unit], 1)?;
            let sum = alloc_like(ctx, &t)?;
            ops::elementwise_add(ctx, &t, &x, &sum)?;
            x = sum;
        }
        Ok(x)
    }

    /// Decode denormalised latents `[batch, AUDIO_CHANNELS, frames]` (FP32) into waveforms `[batch, 1, frames * 800]`, before the clamp to
    /// `[-1, 1]` the caller applies. `tap(name, tensor)` sees the activations after `dec_in`, `conv_pre`, every `up{i}` and `stage{i}`,
    /// `act_post` and `conv_post` (the tests compare them stage by stage).
    pub fn decode(
        &self,
        latents: &Tensor,
        tap: &mut dyn FnMut(&str, &Tensor) -> Result<()>,
    ) -> Result<Tensor> {
        let ctx = &*self.ctx;
        let w = &*self.weights;
        let [batch, _, frames] = dims(latents);
        let lifted = alloc(ctx, &[batch, AUDIO_MEL_DIM, frames], DType::F32)?;
        ops::conv1d(
            ctx,
            Conv1dArgs {
                input: latents,
                weight: &w.dec_in.weight,
                bias: w.dec_in.bias.as_ref(),
                out: &lifted,
                stride: 1,
                dilation: 1,
                padding: Conv1dPadding::Zeros { before: 0 },
                depthwise: false,
            },
        )?;
        tap("dec_in", &lifted)?;
        let mut x = self.same_conv(&lifted, &w.conv_pre, 1)?;
        tap("conv_pre", &x)?;
        for (i, stage) in w.stages.iter().enumerate() {
            let [_, _, len] = dims(&x);
            let out_channels = stage.up.weight.shape().dims()[1];
            let upsampled = alloc(ctx, &[batch, out_channels, len * stage.stride], DType::F32)?;
            ops::conv_transpose1d(
                ctx,
                ConvTranspose1dArgs {
                    input: &x,
                    weight: &stage.up.weight,
                    bias: stage.up.bias.as_ref(),
                    out: &upsampled,
                    stride: stage.stride,
                    replicate_pad: 0,
                    crop_left: stage.padding,
                    groups: 1,
                    scale: 1.0,
                },
            )?;
            tap(&format!("up{i}"), &upsampled)?;
            let mut sum: Option<Tensor> = None;
            for block in &stage.resblocks {
                let y = self.resblock(&upsampled, block)?;
                sum = Some(match sum {
                    None => y,
                    Some(total) => {
                        let next = alloc_like(ctx, &y)?;
                        ops::elementwise_add(ctx, &total, &y, &next)?;
                        next
                    }
                });
            }
            let total = sum.expect("a stage has resblocks");
            let mean = alloc_like(ctx, &total)?;
            ops::elementwise_scale(ctx, &total, &mean, 1.0 / stage.resblocks.len() as f32)?;
            tap(&format!("stage{i}"), &mean)?;
            x = mean;
        }
        let activated = self.activation(&x, &w.activation_post)?;
        tap("act_post", &activated)?;
        let wave = self.same_conv(&activated, &w.conv_post, 1)?;
        tap("conv_post", &wave)?;
        Ok(wave)
    }
}

fn alloc_like(ctx: &CudaContext, like: &Tensor) -> Result<Tensor> {
    alloc(ctx, like.shape().dims(), like.dtype())
}
