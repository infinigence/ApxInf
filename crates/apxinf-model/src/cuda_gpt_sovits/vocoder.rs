//! V5 HiFiGAN-style vocoder (48 kHz).

use apxinf_core::{Error, Result, Tensor};
use apxinf_cuda_new::CudaContext as Context;

use super::config::{VOC_RESBLOCK_KERNELS, VOC_UPSAMPLE_KERNELS, VOC_UPSAMPLE_RATES};
use super::ops;
use super::weights::{Conv, Weights};

struct ResBlock {
    convs1: Vec<Conv>,
    convs2: Vec<Conv>,
}

pub struct Vocoder {
    conv_pre: Conv,
    ups: Vec<Conv>,
    resblocks: Vec<ResBlock>,
    conv_post: Conv,
    resblock_scales: Vec<Tensor>,
    resblock_zeros: Vec<Tensor>,
}

impl Vocoder {
    pub fn load(weights: &mut Weights, ctx: &Context) -> Result<Self> {
        let conv_pre = Conv::load(weights, ctx, "vocoder.conv_pre")?;
        let mut ups = Vec::with_capacity(5);
        for index in 0..5 {
            ups.push(Conv::load(weights, ctx, &format!("vocoder.ups.{index}"))?);
        }
        let mut resblocks = Vec::with_capacity(15);
        for index in 0..15 {
            let mut convs1 = Vec::with_capacity(3);
            let mut convs2 = Vec::with_capacity(3);
            for kernel in 0..3 {
                convs1.push(Conv::load(
                    weights,
                    ctx,
                    &format!("vocoder.res.{index}.convs1.{kernel}"),
                )?);
                convs2.push(Conv::load(
                    weights,
                    ctx,
                    &format!("vocoder.res.{index}.convs2.{kernel}"),
                )?);
            }
            resblocks.push(ResBlock { convs1, convs2 });
        }
        let resblock_channels = [256usize, 128, 64, 32, 16];
        let resblock_scales = resblock_channels
            .iter()
            .map(|&channels| {
                super::weights::fp16_tensor(ctx, &[channels], &vec![1.0 / 3.0; channels])
            })
            .collect::<Result<Vec<_>>>()?;
        let resblock_zeros = resblock_channels
            .iter()
            .map(|&channels| super::weights::fp16_tensor(ctx, &[channels], &vec![0.0; channels]))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_pre,
            ups,
            resblocks,
            conv_post: Conv::load(weights, ctx, "vocoder.conv_post")?,
            resblock_scales,
            resblock_zeros,
        })
    }

    pub fn forward(&self, ctx: &Context, mel: &Tensor) -> Result<Tensor> {
        let len = mel.shape().dims()[0];
        // Every convolution below binds the channel-last buffers directly.
        // A stage input is read twice — once by the activation feeding the
        // convolutions, once as the residual a resblock adds back — so its bias
        // cannot stay deferred. Materialize it where the ordinary path rounded
        // it: `round_f16(x + bias)`.
        let conv_pre_bias = self.conv_pre.deferred_bias()?;
        let x = self.conv_pre.apply_direct(
            ctx,
            &mel.reshape(apxinf_core::Shape::new(vec![1, len, 100]))?,
            1,
            3,
            1,
            1,
        )?;
        let mut x = ops::add_bias_rounded(ctx, &x, conv_pre_bias)?;
        for (index, up) in self.ups.iter().enumerate() {
            x = ops::leaky_relu(ctx, &x)?;
            let kernel = VOC_UPSAMPLE_KERNELS[index];
            x = up.transpose_direct(
                ctx,
                &x,
                VOC_UPSAMPLE_RATES[index],
                (kernel - VOC_UPSAMPLE_RATES[index]) / 2,
            )?;
            let up_bias = up.deferred_bias()?;
            x = ops::add_bias_rounded(ctx, &x, up_bias)?;
            let channels = x.shape().dims()[2];
            let length = x.shape().dims()[1];
            let mut sum = ops::alloc(ctx, &[1, length, channels])?;
            ops::fill_zero(ctx, &sum)?;
            for kernel_index in 0..3 {
                let block = &self.resblocks[index * 3 + kernel_index];
                let mut hidden = x.clone();
                let kernel = VOC_RESBLOCK_KERNELS[kernel_index];
                for pair in 0..3 {
                    let dilation = [1usize, 3, 5][pair];
                    let padded = dilation * (kernel - 1) / 2;
                    let convs1 = &block.convs1[pair];
                    let convs2 = &block.convs2[pair];
                    let first_bias = convs1.deferred_bias()?;
                    let second_bias = convs2.deferred_bias()?;
                    let first = ops::leaky_relu(ctx, &hidden)?;
                    let first = convs1.apply_direct(ctx, &first, 1, padded, dilation, 1)?;
                    let second = ops::leaky_relu_bias(ctx, &first, first_bias)?;
                    let second = convs2.apply_direct(ctx, &second, 1, (kernel - 1) / 2, 1, 1)?;
                    hidden = ops::residual_add_bias(ctx, &second, second_bias, &hidden)?;
                }
                sum = ops::residual_add(ctx, &sum, &hidden)?;
            }
            x = ops::affine(
                ctx,
                &sum,
                &self.resblock_scales[index],
                &self.resblock_zeros[index],
            )?;
            // The affine carries no bias, so the next stage activates plainly.
        }
        // PyTorch calls F.leaky_relu(x) without an explicit slope here,
        // so this final activation uses the framework default 0.01. The
        // upsample and ResBlock activations above explicitly use 0.1.
        x = ops::leaky_relu_one_percent(ctx, &x)?;
        x = self.conv_post.apply(ctx, &x, 1, 3, 1, 1)?;
        let output = ops::tanh(ctx, &x)?;
        Ok(output)
    }

    /// Returns mono PCM at 48 kHz.
    pub fn synth(&self, ctx: &Context, mel: &Tensor) -> Result<Vec<f32>> {
        let output = self.forward(ctx, mel)?;
        Self::pcm_from_output(ctx, &output)
    }

    /// Convert the `[1, 1, samples]` waveform tensor into host PCM.
    pub fn pcm_from_output(ctx: &Context, output: &Tensor) -> Result<Vec<f32>> {
        ops::sync(ctx)?;
        let host = apxinf_cuda_new::transfers::to_cpu(output)?;
        let bytes = host
            .storage()
            .as_cpu()
            .ok_or_else(|| Error::Other("vocoder output is not host storage".into()))?;
        let samples: Vec<f32> = match output.dtype() {
            apxinf_core::DType::F16 => bytes
                .chunks_exact(2)
                .map(|chunk| {
                    half::f16::from_bits(u16::from_ne_bytes([chunk[0], chunk[1]])).to_f32()
                })
                .collect(),
            apxinf_core::DType::F32 => bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
            other => {
                return Err(Error::Other(format!(
                    "unsupported vocoder output dtype {other}"
                )))
            }
        };
        Ok(samples)
    }
}
