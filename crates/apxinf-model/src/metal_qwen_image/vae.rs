//! VAE decoding as mflux runs it under `--low-ram`: `Qwen21VAE.decode` on
//! NCHW tensors (each convolution in NHWC), tiled with host-side blending
//! (`VAETiler.decode_image_tiled`, `TilingConfig()` defaults), then RGB8.
use super::ops::Ops;
use super::tables::cos_ramp;
use super::weights::{Conv, Decoder, Residual, Up, LATENT};
use apxinf_core::Result;
use apxinf_mlx::{Array, MlxDType};

/// `Qwen21VAEConfig.LATENTS_MEAN` and `LATENTS_STD`.
#[rustfmt::skip]
#[allow(clippy::approx_constant)]
const LATENTS_MEAN: [f32; LATENT] = [
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722,
    1.8873, -1.7177, -0.651, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363,
    2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
    -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955,
    -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.107, 1.7376,
    -1.0914, -1.9435, -0.2784, -1.368, 0.4809, -0.4433, 0.3764, 0.5729,
    -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505,
    0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1566, 0.0304,
];
#[rustfmt::skip]
const LATENTS_STD: [f32; LATENT] = [
    3.2001, 3.2936, 3.4321, 3.0091, 3.106, 4.0379, 4.0705, 3.791,
    3.0785, 3.65, 3.9308, 3.0904, 2.8778, 3.7675, 3.732, 5.0756,
    3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489,
    3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467,
    3.9037, 3.2947, 5.077, 3.5075, 3.27, 3.4767, 2.8063, 5.1125,
    3.532, 4.7833, 3.1284, 4.181, 3.8527, 3.8317, 3.5603, 4.3867,
    3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.44, 3.4957,
    3.8747, 3.7608, 3.5735, 3.149, 3.7662, 3.6746, 3.4563, 3.8161,
];

/// Pixels per latent, and the tiler's 512 px tiles with 8 latent (128 px) overlap.
const SCALE: usize = 16;
const TILE: usize = 512 / SCALE;
const OVERLAP: usize = 8;

struct Vae<'a> {
    w: &'a Decoder,
    ops: &'a Ops,
}

impl Vae<'_> {
    /// `VAEOperations.conv` in text mode: the NCHW input is convolved in NHWC.
    fn conv(&self, x: &Array, c: &Conv) -> Result<Array> {
        let padding = c.weight.shape()[1] / 2;
        x.transpose(&[0, 2, 3, 1])?
            .conv2d(&c.weight, [1, 1], [padding; 2], [1, 1], 1)?
            .add(&c.bias)?
            .transpose(&[0, 3, 1, 2])
    }

    /// `ChannelNorm` in text mode: L2 over channels, times sqrt(C) and gamma.
    fn norm(&self, x: &Array, gamma: &Array) -> Result<Array> {
        let channels = gamma.shape()[0];
        let value = x.cast(MlxDType::F32)?;
        let norm = value
            .mul(&value)?
            .sum(&[1], true)?
            .sqrt()?
            .maximum(&self.ops.f32(1e-12)?)?;
        let scale = self.ops.f32((channels as f64).sqrt() as f32)?;
        value
            .div(&norm)?
            .mul(&scale)?
            .mul(&gamma.reshape(&[1, channels, 1, 1])?)?
            .cast(x.dtype())
    }

    fn residual(&self, x: &Array, r: &Residual) -> Result<Array> {
        let shortcut = match &r.shortcut {
            Some(c) => self.conv(x, c)?,
            None => x.clone(),
        };
        let y = self.conv(&self.ops.silu(&self.norm(x, &r.norm1)?)?, &r.conv1)?;
        let y = self.conv(&self.ops.silu(&self.norm(&y, &r.norm2)?)?, &r.conv2)?;
        y.add(&shortcut)
    }

    /// `AttentionBlock` in text mode, with its explicit softmax attention.
    fn attention(&self, x: &Array) -> Result<Array> {
        let normalized = self.norm(x, &self.w.attn_norm)?.transpose(&[0, 2, 3, 1])?;
        let [b, h, w, c] = *normalized.shape() else {
            unreachable!("decoder activations are NCHW")
        };
        let conv1x1 = |x: &Array, k: &Conv| -> Result<Array> {
            x.conv2d(&k.weight, [1, 1], [0, 0], [1, 1], 1)?.add(&k.bias)
        };
        let qkv = conv1x1(&normalized, &self.w.attn_qkv)?.reshape(&[b, h * w, 3, c])?;
        let part = |i: usize| -> Result<Array> {
            qkv.slice(&[0, 0, i, 0], &[b, h * w, i + 1, c])?
                .reshape(&[b, h * w, c])
        };
        let (q, k, v) = (part(0)?, part(1)?, part(2)?);
        let scale = self.ops.f32(1.0)?.div(&self.ops.f32(c as f32)?.sqrt()?)?;
        let scores = q.matmul(&k.transpose(&[0, 2, 1])?)?.mul(&scale)?;
        let output = scores.softmax_with(-1, false)?.matmul(&v)?;
        let output = conv1x1(&output.reshape(&[b, h, w, c])?, &self.w.attn_proj)?;
        output.transpose(&[0, 3, 1, 2])?.add(x)
    }

    /// `Qwen21DupUp`: parameterless channel-repeat shortcut of an up block.
    fn dup_up(x: &Array, up: &Up, temporal: usize) -> Result<Array> {
        let [b, _, h, w] = *x.shape() else {
            unreachable!("decoder activations are NCHW")
        };
        let out = up.out_dim;
        let repeats = out * temporal * 4 / up.in_dim;
        x.repeat(repeats, 1)?
            .reshape(&[b, out, temporal, 2, 2, 1, h, w])?
            .transpose(&[0, 1, 5, 2, 6, 3, 7, 4])?
            .reshape(&[b, out, temporal, 2 * h, 2 * w])?
            .slice_axis(2, temporal - 1, temporal)?
            .reshape(&[b, out, 2 * h, 2 * w])
    }

    fn up_block(&self, x: &Array, up: &Up, temporal: usize) -> Result<Array> {
        let mut y = x.clone();
        for r in &up.resnets {
            y = self.residual(&y, r)?;
        }
        let Some(upsampler) = &up.upsampler else {
            return Ok(y);
        };
        let shortcut = Self::dup_up(x, up, temporal)?;
        // Resample: nearest 2x in NHWC, then a 3x3 convolution.
        let y = y
            .transpose(&[0, 2, 3, 1])?
            .repeat(2, 1)?
            .repeat(2, 2)?
            .conv2d(&upsampler.weight, [1, 1], [1, 1], [1, 1], 1)?
            .add(&upsampler.bias)?
            .transpose(&[0, 3, 1, 2])?;
        y.add(&shortcut)
    }

    /// `Qwen21VAE.decode` for BF16 latents `[1, 64, h, w]`: FP32 `[1, 3, 16h, 16w]`.
    fn decode(&self, latents: &Array) -> Result<Array> {
        let stream = self.ops.stream();
        let std = Array::from_f32(stream, &[1, LATENT, 1, 1], &LATENTS_STD)?;
        let mean = Array::from_f32(stream, &[1, LATENT, 1, 1], &LATENTS_MEAN)?;
        let x = self.conv(&latents.mul(&std)?.add(&mean)?, &self.w.post_quant)?;
        let mut x = self.conv(&x, &self.w.conv_in)?;
        x = self.residual(&x, &self.w.mid[0])?;
        x = self.attention(&x)?;
        x = self.residual(&x, &self.w.mid[1])?;
        // Temporal factor of each up block's shortcut: reversed (F, T, T, T).
        for (up, temporal) in self.w.up.iter().zip([2, 2, 2, 1, 1]) {
            x = self.up_block(&x, up, temporal)?;
        }
        let x = self.conv(
            &self.ops.silu(&self.norm(&x, &self.w.norm_out)?)?,
            &self.w.conv_out,
        )?;
        let [b, _, h, w] = *x.shape() else {
            unreachable!("decoder output is NCHW")
        };
        x.slice(&[0, 0, 0, 0], &[b, 3, h, w])
    }
}

/// One decoded tile as host `[h, w, 3]` FP32.
fn decode_tile(vae: &Vae, latents: &Array) -> Result<Vec<f32>> {
    let decoded = vae.decode(latents)?.cast(MlxDType::F32)?;
    decoded.transpose(&[0, 2, 3, 1])?.contiguous()?.to_f32_vec()
}

/// Final latents `[1, N, 64]` BF16 for an image of `width` x `height` pixels: RGB8.
pub fn decode(
    w: &Decoder,
    ops: &Ops,
    latents: &Array,
    width: usize,
    height: usize,
) -> Result<Vec<u8>> {
    let vae = Vae { w, ops };
    let (lh, lw) = (height / SCALE, width / SCALE);
    // Qwen21LatentCreator.unpack_latents, single frame.
    let latents = latents
        .reshape(&[1, lh, lw, LATENT])?
        .transpose(&[0, 3, 1, 2])?;
    let pixels = if lh <= TILE && lw <= TILE {
        decode_tile(&vae, &latents)?
    } else {
        let ramp = cos_ramp(OVERLAP * SCALE);
        let mut out = vec![0f32; height * width * 3];
        let mut count = vec![0f32; height * width];
        let stride = TILE - OVERLAP;
        for y in (0..lh).step_by(stride) {
            let y_end = (y + TILE).min(lh);
            for x in (0..lw).step_by(stride) {
                let x_end = (x + TILE).min(lw);
                // Skip slivers that the previous tile's overlap already covers.
                if (y > 0 && y_end - y <= OVERLAP) || (x > 0 && x_end - x <= OVERLAP) {
                    continue;
                }
                let tile = latents.slice(&[0, 0, y, x], &[1, LATENT, y_end, x_end])?;
                let tile = decode_tile(&vae, &tile)?;
                let (th, tw) = ((y_end - y) * SCALE, (x_end - x) * SCALE);
                let weights = |n: usize, head: bool, tail: bool| {
                    let overlap = (OVERLAP * SCALE).min(n - 1);
                    let mut w = vec![1f32; n];
                    if head {
                        w[..overlap].copy_from_slice(&ramp[..overlap]);
                    }
                    if tail {
                        for (i, r) in ramp[..overlap].iter().enumerate() {
                            w[n - overlap + i] = 1.0 - r;
                        }
                    }
                    w
                };
                let wh = weights(th, y > 0, y_end < lh);
                let ww = weights(tw, x > 0, x_end < lw);
                for i in 0..th {
                    for j in 0..tw {
                        let weight = wh[i] * ww[j];
                        let at = (y * SCALE + i) * width + x * SCALE + j;
                        for ch in 0..3 {
                            out[at * 3 + ch] += tile[(i * tw + j) * 3 + ch] * weight;
                        }
                        count[at] += weight;
                    }
                }
            }
        }
        for (i, value) in out.iter_mut().enumerate() {
            *value /= count[i / 3].max(1e-6);
        }
        out
    };
    // ImageUtil.to_pil: clip(x / 2 + 0.5, 0, 1), then (x * 255).round() as u8.
    Ok(pixels
        .iter()
        .map(|&x| ((x / 2.0 + 0.5).clamp(0.0, 1.0) * 255.0).round_ties_even() as u8)
        .collect())
}
