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

pub(crate) struct VocoderDebug {
    pub conv_pre: Tensor,
    pub first_upsample: Tensor,
    pub first_upsample_resblock: Tensor,
    pub first_resblocks: Vec<Tensor>,
    pub first_resblock_pairs: Vec<Tensor>,
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
        self.forward_impl(ctx, mel, false).map(|(output, _)| output)
    }

    pub(crate) fn forward_with_debug(
        &self,
        ctx: &Context,
        mel: &Tensor,
    ) -> Result<(Tensor, VocoderDebug)> {
        let (output, debug) = self.forward_impl(ctx, mel, true)?;
        let debug = debug.ok_or_else(|| Error::Other("vocoder debug trace missing".into()))?;
        Ok((output, debug))
    }

    fn forward_impl(
        &self,
        ctx: &Context,
        mel: &Tensor,
        trace_debug: bool,
    ) -> Result<(Tensor, Option<VocoderDebug>)> {
        let len = mel.shape().dims()[0];
        let mut x = self.conv_pre.apply(
            ctx,
            &mel.reshape(apxinf_core::Shape::new(vec![1, len, 100]))?,
            1,
            3,
            1,
            1,
        )?;
        let mut conv_pre_debug = trace_debug.then(|| x.clone());
        let mut first_upsample_debug = None;
        let mut first_resblocks_debug = Vec::new();
        let mut first_resblock_pairs_debug = Vec::new();
        let mut first_upsample_resblock_debug = None;
        for (index, up) in self.ups.iter().enumerate() {
            x = ops::leaky_relu(ctx, &x)?;
            let kernel = VOC_UPSAMPLE_KERNELS[index];
            x = up.transpose(
                ctx,
                &x,
                VOC_UPSAMPLE_RATES[index],
                (kernel - VOC_UPSAMPLE_RATES[index]) / 2,
            )?;
            if index == 0 && trace_debug {
                first_upsample_debug = Some(x.clone());
            }
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
                    let first = ops::leaky_relu(ctx, &hidden)?;
                    let first = block.convs1[pair].apply(ctx, &first, 1, padded, dilation, 1)?;
                    if index == 0 && kernel_index == 2 && trace_debug {
                        first_resblock_pairs_debug.push(first.clone());
                    }
                    let second = ops::leaky_relu(ctx, &first)?;
                    let second =
                        block.convs2[pair].apply(ctx, &second, 1, (kernel - 1) / 2, 1, 1)?;
                    if index == 0 && kernel_index == 2 && trace_debug {
                        first_resblock_pairs_debug.push(second.clone());
                    }
                    hidden = ops::residual_add(ctx, &second, &hidden)?;
                }
                sum = ops::residual_add(ctx, &sum, &hidden)?;
                if index == 0 && trace_debug {
                    first_resblocks_debug.push(hidden.clone());
                }
            }
            x = ops::affine(
                ctx,
                &sum,
                &self.resblock_scales[index],
                &self.resblock_zeros[index],
            )?;
            if index == 0 && trace_debug {
                first_upsample_resblock_debug = Some(x.clone());
            }
        }
        // PyTorch calls F.leaky_relu(x) without an explicit slope here,
        // so this final activation uses the framework default 0.01. The
        // upsample and ResBlock activations above explicitly use 0.1.
        x = ops::leaky_relu_one_percent(ctx, &x)?;
        x = self.conv_post.apply(ctx, &x, 1, 3, 1, 1)?;
        let output = ops::tanh(ctx, &x)?;
        let debug = if trace_debug {
            Some(VocoderDebug {
                conv_pre: conv_pre_debug
                    .take()
                    .ok_or_else(|| Error::Other("vocoder conv_pre debug trace missing".into()))?,
                first_upsample: first_upsample_debug.take().ok_or_else(|| {
                    Error::Other("vocoder first upsample debug trace missing".into())
                })?,
                first_resblocks: first_resblocks_debug,
                first_resblock_pairs: first_resblock_pairs_debug,
                first_upsample_resblock: first_upsample_resblock_debug.take().ok_or_else(|| {
                    Error::Other("vocoder first upsample debug trace missing".into())
                })?,
            })
        } else {
            None
        };
        Ok((output, debug))
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
            apxinf_core::DType::BF16 => bytes
                .chunks_exact(2)
                .map(|chunk| {
                    half::bf16::from_bits(u16::from_ne_bytes([chunk[0], chunk[1]])).to_f32()
                })
                .collect(),
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

#[cfg(test)]
mod capture_tests {
    use super::*;
    #[test]
    #[ignore = "requires APXINF_GPT_CAPTURE_ROOT and APXINF_GPT_CHECKPOINT"]
    fn real_mel_eager_graph_benchmark() {
        let root = std::path::PathBuf::from(std::env::var("APXINF_GPT_CAPTURE_ROOT").unwrap());
        let checkpoint = std::path::PathBuf::from(std::env::var("APXINF_GPT_CHECKPOINT").unwrap());
        let ctx = Context::new(0).unwrap();
        let mut weights = Weights::load(&checkpoint).unwrap();
        let vocoder = Vocoder::load(&mut weights, &ctx).unwrap();
        for pair in ["pair1", "pair2"] {
            let manifest: serde_json::Value = serde_json::from_slice(
                &std::fs::read(root.join(pair).join("raw/manifest.json")).unwrap(),
            )
            .unwrap();
            let entries = manifest["entries"].as_array().unwrap();
            let load = |name: &str| {
                let entry = entries.iter().find(|e| e["name"] == name).unwrap();
                let bytes = std::fs::read(
                    root.join(pair)
                        .join("raw")
                        .join(entry["path"].as_str().unwrap()),
                )
                .unwrap();
                bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect::<Vec<_>>()
            };
            let nchw_mel = load("vocoder.0.mel_in");
            let frames = nchw_mel.len() / 100;
            let mut nlc_mel = vec![0f32; nchw_mel.len()];
            for t in 0..frames {
                for c in 0..100 {
                    nlc_mel[t * 100 + c] = nchw_mel[c * frames + t];
                }
            }
            let mel = super::super::weights::fp16_tensor(&ctx, &[frames, 100], &nlc_mel).unwrap();
            let reference = load("vocoder.0.wav_out");
            let start = std::time::Instant::now();
            let eager = vocoder.forward(&ctx, &mel).unwrap();
            ctx.synchronize().unwrap();
            println!("{pair} eager_seconds={:.6}", start.elapsed().as_secs_f64());
            let eager_pcm = Vocoder::pcm_from_output(&ctx, &eager).unwrap();
            let session = apxinf_cuda_new::ops::ExecutionSession::with_capacity(
                (2usize * 1024 * 1024 * 1024 * frames + 499) / 500,
                0,
            )
            .unwrap();
            let output = ops::alloc(&ctx, &[1, frames * 480, 1]).unwrap();
            let forward = || {
                let generated = vocoder.forward(&ctx, &mel)?;
                apxinf_cuda_new::transfers::copy_cuda_to_cuda(&ctx, &generated, &output)
            };
            apxinf_cuda_new::ops::prepare_with_session(&session, forward).unwrap();
            ctx.synchronize().unwrap();
            let graph = apxinf_cuda_new::capture(&ctx, || {
                apxinf_cuda_new::ops::with_session(&session, forward)
            })
            .unwrap();
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            let start = std::time::Instant::now();
            for _ in 0..3 {
                graph.replay().unwrap();
            }
            ctx.synchronize().unwrap();
            let seconds = start.elapsed().as_secs_f64() / 3.0;
            let pcm = Vocoder::pcm_from_output(&ctx, &output).unwrap();
            let mut error = 0f64;
            let mut signal = 0f64;
            let mut max_abs = 0f32;
            let mut graph_error = 0f32;
            for ((&a, &r), &e) in pcm.iter().zip(&reference).zip(&eager_pcm) {
                error += f64::from(a - r).powi(2);
                signal += f64::from(r).powi(2);
                max_abs = max_abs.max((a - r).abs());
                graph_error = graph_error.max((a - e).abs());
            }
            let snr = 10.0 * (signal / error).log10();
            println!("{pair} graph_seconds={seconds:.6} samples={} audio_seconds={} snr={snr:.4} max_abs={max_abs:.6} eager_graph_max={graph_error:.8}",pcm.len(),pcm.len() as f64/48000.0);
            assert_eq!(pcm.len(), reference.len());
            assert_eq!(graph_error, 0.0);
            assert!(snr >= 25.0);
        }
    }
}
