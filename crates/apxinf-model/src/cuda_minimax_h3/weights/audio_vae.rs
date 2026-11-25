//! Device weights of the audio VAE decoder (`minimax_h3_audio_vae_fp32.safetensors`): a BigVGAN vocoder over latent frames. The file is FP32 with
//! weight normalisation already folded into plain convolutions, and the decoder stays FP32 on the device (260 MB, resident).
use super::bf16::{f32_vec, upload};
use super::host::Checkpoint;
use crate::cuda_minimax_h3::backend::CudaContext;
use crate::cuda_minimax_h3::config::*;
use apxinf_core::{DType, Error, Result, Tensor};
use std::path::Path;

/// A convolution: weight `[out, in, kernel]` (`[in, out, kernel]` for a transposed one) and an optional bias `[out]`.
pub(in crate::cuda_minimax_h3) struct AudioConv {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub kernel: usize,
}

/// One anti-aliased SnakeBeta activation: x2 upsample filter, the exponentiated per-channel parameters and the x2 low-pass filter.
pub(in crate::cuda_minimax_h3) struct AudioActivation {
    /// Depthwise filters expanded to every channel: `[channels, 1, AUDIO_FILTER]`.
    pub up_filter: Tensor,
    pub down_filter: Tensor,
    pub alpha: Tensor,
    pub beta: Tensor,
}

/// An AMP residual block: three (activation, dilated conv, activation, conv) units.
pub(in crate::cuda_minimax_h3) struct AudioResBlock {
    pub dilations: [usize; 3],
    pub convs1: Vec<AudioConv>,
    pub convs2: Vec<AudioConv>,
    /// Six activations, alternating the two of each unit.
    pub activations: Vec<AudioActivation>,
}

pub(in crate::cuda_minimax_h3) struct AudioStage {
    pub up: AudioConv,
    pub stride: usize,
    pub padding: usize,
    pub resblocks: Vec<AudioResBlock>,
}

pub(in crate::cuda_minimax_h3) struct AudioVaeWeights {
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
    pub dec_in: AudioConv,
    pub conv_pre: AudioConv,
    pub stages: Vec<AudioStage>,
    pub activation_post: AudioActivation,
    pub conv_post: AudioConv,
}

struct Reader<'a> {
    file: Checkpoint,
    ctx: &'a CudaContext,
}

impl Reader<'_> {
    fn floats(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        Ok(f32_vec(&self.file.read(name, Some(DType::F32), shape)?))
    }

    fn tensor(&mut self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let values = self.floats(name, shape)?;
        upload(
            &Tensor::from_f32(shape.to_vec(), &values)?,
            self.ctx.device_id(),
        )
    }

    /// A convolution at `prefix`: `shape` is the weight's shape and `bias` its bias length (`None` for no bias). The bias length is
    /// `shape[0]` for a convolution and `shape[1]` for a transposed one, whose weight is `[in, out, kernel]`.
    fn conv(&mut self, prefix: &str, shape: [usize; 3], bias: Option<usize>) -> Result<AudioConv> {
        let weight = self.tensor(&format!("{prefix}.weight"), &shape)?;
        let bias = match bias {
            Some(len) => Some(self.tensor(&format!("{prefix}.bias"), &[len])?),
            None => None,
        };
        Ok(AudioConv {
            weight,
            bias,
            kernel: shape[2],
        })
    }

    /// The activation at `prefix` (`<prefix>.act.alpha`, `.act.beta`, `.upsample.filter`, `.downsample.lowpass.filter`).
    fn activation(&mut self, prefix: &str, channels: usize) -> Result<AudioActivation> {
        let expand = |filter: &[f32]| -> Vec<f32> {
            (0..channels).flat_map(|_| filter.iter().copied()).collect()
        };
        let up = self.floats(&format!("{prefix}.upsample.filter"), &[1, 1, AUDIO_FILTER])?;
        let down = self.floats(
            &format!("{prefix}.downsample.lowpass.filter"),
            &[1, 1, AUDIO_FILTER],
        )?;
        // alpha and beta are stored in log scale; the activation uses their exponentials
        let alpha: Vec<f32> = self
            .floats(&format!("{prefix}.act.alpha"), &[channels])?
            .iter()
            .map(|v| v.exp())
            .collect();
        let beta: Vec<f32> = self
            .floats(&format!("{prefix}.act.beta"), &[channels])?
            .iter()
            .map(|v| v.exp())
            .collect();
        let device = self.ctx.device_id();
        Ok(AudioActivation {
            up_filter: upload(
                &Tensor::from_f32(vec![channels, 1, AUDIO_FILTER], &expand(&up))?,
                device,
            )?,
            down_filter: upload(
                &Tensor::from_f32(vec![channels, 1, AUDIO_FILTER], &expand(&down))?,
                device,
            )?,
            alpha: upload(&Tensor::from_f32(vec![channels], &alpha)?, device)?,
            beta: upload(&Tensor::from_f32(vec![channels], &beta)?, device)?,
        })
    }
}

/// Channels of the activations after stage `i` (`AUDIO_BASE_CHANNELS / 2^(i + 1)`).
pub(in crate::cuda_minimax_h3) fn stage_channels(stage: usize) -> usize {
    AUDIO_BASE_CHANNELS >> (stage + 1)
}

impl AudioVaeWeights {
    pub fn load(path: &Path, ctx: &CudaContext) -> Result<Self> {
        let mut file = Checkpoint::open(path)?;
        // the DAC encoder, its attention head and the posterior projections only serve encoding, which this family does not do
        let skipped = ["encoder.", "pre_block.", "mean_proj.", "logs_proj."]
            .iter()
            .map(|p| file.ignore_prefix(p))
            .sum::<usize>();
        if skipped == 0 {
            return Err(Error::Other(format!(
                "{}: not a MiniMax-H3 audio VAE (no encoder tensors)",
                path.display()
            )));
        }
        let mut r = Reader { file, ctx };
        let latents_mean = r.floats("latents_mean", &[AUDIO_CHANNELS])?;
        let latents_std = r.floats("latents_std", &[AUDIO_CHANNELS])?;
        let dec_in = r.conv(
            "dec_in_proj",
            [AUDIO_MEL_DIM, AUDIO_CHANNELS, 1],
            Some(AUDIO_MEL_DIM),
        )?;
        let conv_pre = r.conv(
            "decoder.conv_pre",
            [AUDIO_BASE_CHANNELS, AUDIO_MEL_DIM, 7],
            Some(AUDIO_BASE_CHANNELS),
        )?;
        let mut stages = Vec::with_capacity(AUDIO_UPSAMPLE.len());
        for (i, &(stride, kernel)) in AUDIO_UPSAMPLE.iter().enumerate() {
            let (cin, cout) = (AUDIO_BASE_CHANNELS >> i, stage_channels(i));
            let up = r.conv(
                &format!("decoder.ups.{i}.0"),
                [cin, cout, kernel],
                Some(cout),
            )?;
            let mut resblocks = Vec::with_capacity(AUDIO_RESBLOCK_KERNELS.len());
            for (j, &block_kernel) in AUDIO_RESBLOCK_KERNELS.iter().enumerate() {
                let p = format!("decoder.resblocks.{}", i * AUDIO_RESBLOCK_KERNELS.len() + j);
                let mut convs1 = Vec::new();
                let mut convs2 = Vec::new();
                for u in 0..AUDIO_RESBLOCK_DILATIONS.len() {
                    convs1.push(r.conv(
                        &format!("{p}.convs1.{u}"),
                        [cout, cout, block_kernel],
                        Some(cout),
                    )?);
                    convs2.push(r.conv(
                        &format!("{p}.convs2.{u}"),
                        [cout, cout, block_kernel],
                        Some(cout),
                    )?);
                }
                let mut activations = Vec::new();
                for a in 0..2 * AUDIO_RESBLOCK_DILATIONS.len() {
                    activations.push(r.activation(&format!("{p}.activations.{a}"), cout)?);
                }
                resblocks.push(AudioResBlock {
                    dilations: AUDIO_RESBLOCK_DILATIONS,
                    convs1,
                    convs2,
                    activations,
                });
            }
            stages.push(AudioStage {
                up,
                stride,
                padding: (kernel - stride) / 2,
                resblocks,
            });
        }
        let last = stage_channels(AUDIO_UPSAMPLE.len() - 1);
        let activation_post = r.activation("decoder.activation_post", last)?;
        let conv_post = r.conv("decoder.conv_post", [1, last, 7], None)?;
        r.file.finish()?;
        Ok(Self {
            latents_mean,
            latents_std,
            dec_in,
            conv_pre,
            stages,
            activation_post,
            conv_post,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_channels_halve_from_the_base_width() {
        assert_eq!(
            (0..7).map(stage_channels).collect::<Vec<_>>(),
            vec![512, 256, 128, 64, 32, 16, 8]
        );
        // the upsample rates multiply to the 800 samples of a latent frame
        assert_eq!(
            AUDIO_UPSAMPLE.iter().map(|(s, _)| s).product::<usize>(),
            AUDIO_SAMPLES_PER_LATENT
        );
        // each stage's transposed convolution crops (k - s) / 2 per side, so it multiplies the length by its stride
        for (s, k) in AUDIO_UPSAMPLE {
            assert_eq!((k - s) % 2, 0);
        }
    }
}
