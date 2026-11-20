//! Device weights of the ViT VAE decoder (`minimax_h3_image_vae_fp16.safetensors`). The file stores F16; linears are
//! uploaded as BF16 `[in, out]` because the decoder computes in BF16 (the DiT's activation dtype in this family).
use super::bf16::{f16_vec, transpose, upload};
use super::host::Checkpoint;
use crate::cuda_minimax_h3_image::backend::CudaContext;
use crate::cuda_minimax_h3_image::config::*;
use apxinf_core::{DType, Error, Result, Tensor};
use std::path::Path;

/// A BF16 linear: weight `[in, out]` and optional bias `[out]`.
pub(in crate::cuda_minimax_h3_image) struct VaeLinear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

/// An FP32 linear (weight `[in, out]`, bias `[out]`) for the small latent-embedding stage, which the reference runs
/// on the raw latents and which is too small to be worth rounding to BF16.
pub(in crate::cuda_minimax_h3_image) struct VaeLinearF32 {
    pub weight: Tensor,
    pub bias: Tensor,
}

pub(in crate::cuda_minimax_h3_image) struct VaeBlock {
    pub norm1: Tensor,
    pub norm2: Tensor,
    /// The fused `to_qkv` projection is stored per head as `[q(64), k(64), v(64)]` (`view(.., heads, 3 * head_dim)` then
    /// `chunk(3)`), unlike the DiT's block-wise `chunk(3)`. It is de-interleaved at load into three projections so each
    /// output is a contiguous `[tokens, heads, head_dim]` tensor.
    pub q: VaeLinear,
    pub k: VaeLinear,
    pub v: VaeLinear,
    pub out: VaeLinear,
    /// Gated SiLU feed-forward: `w1` rows are `[gate; up]`, split like the DiT's `fc1`.
    pub gate: VaeLinear,
    pub up: VaeLinear,
    pub down: VaeLinear,
    /// Residual scales as packed styles `[0; 0; scale]` (BF16 `[3 * VAE_DIM]`): `x + scale * branch` is a
    /// gated residual whose gate is the third segment.
    pub style1: Tensor,
    pub style2: Tensor,
}

pub(in crate::cuda_minimax_h3_image) struct VaeWeights {
    /// `x_embedder(post_quant_conv(z * std + mean))` as one FP32 linear on normalised latents: all three steps are
    /// affine, so they are composed in FP64 at load (see [`fold_embedding`]).
    pub embed: VaeLinearF32,
    pub ones: Tensor,
    pub registers: Tensor,
    pub blocks: Vec<VaeBlock>,
    pub norm_out_weight: Tensor,
    pub norm_out_bias: Tensor,
    pub proj_out: VaeLinear,
}

struct Reader<'a> {
    file: Checkpoint,
    ctx: &'a CudaContext,
}

fn bf16_bytes_of(values: &[half::f16]) -> Vec<half::bf16> {
    values
        .iter()
        .map(|v| half::bf16::from_f32(v.to_f32()))
        .collect()
}

/// Compose `x_embedder(post_quant_conv(z * std + mean))` into one affine map `z -> z @ W + b`.
/// `pq_weight` is `[out, in]` row-major (the 1x1 convolution), `embed_weight` is `[VAE_DIM, LATENT_CHANNELS]`.
/// Returns `W` as row-major `[LATENT_CHANNELS, VAE_DIM]` and `b` as `[VAE_DIM]`.
pub(super) fn fold_embedding(
    pq_weight: &[f32],
    pq_bias: &[f32],
    embed_weight: &[f32],
    embed_bias: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let c = LATENT_CHANNELS;
    let mut weight = vec![0f32; c * VAE_DIM];
    let mut bias = vec![0f32; VAE_DIM];
    for d in 0..VAE_DIM {
        let mut b = embed_bias[d] as f64;
        for o in 0..c {
            let e = embed_weight[d * c + o] as f64;
            b += e * pq_bias[o] as f64;
            for i in 0..c {
                let p = pq_weight[o * c + i] as f64;
                weight[i * VAE_DIM + d] += (e * p * LATENTS_STD[i] as f64) as f32;
                b += e * p * LATENTS_MEAN[i] as f64;
            }
        }
        bias[d] = b as f32;
    }
    (weight, bias)
}

/// Channel permutation of one 64-channel head so that the decoder's rotation of channel pairs `(i, i + 16)` for
/// `i < 16` becomes the half-split rotation over all 64 channels that `rms_norm_apply_precomputed` implements:
/// permuted channel `c` holds original channel `P[c]`, pairs `(j, j + 32)` rotate, and `j >= 16` are the
/// pass-through channels `32..64` paired with themselves under identity coefficients.
pub(in crate::cuda_minimax_h3_image) fn vae_rope_permutation() -> [usize; VAE_HEAD_DIM] {
    let mut p = [0usize; VAE_HEAD_DIM];
    for j in 0..VAE_HEAD_DIM / 2 {
        if j < VAE_ROPE_DIM / 2 {
            p[j] = j;
            p[j + VAE_HEAD_DIM / 2] = j + VAE_ROPE_DIM / 2;
        } else {
            p[j] = j + VAE_ROPE_DIM / 2;
            p[j + VAE_HEAD_DIM / 2] = j + VAE_HEAD_DIM / 2;
        }
    }
    p
}

/// Row indices (into a `[heads * 3 * head_dim, k]` per-head-interleaved qkv matrix) of projection `part`
/// (0 q, 1 k, 2 v). Q and K rows are channel-permuted within each head for the rotation; V rows keep their order.
pub(super) fn qkv_rows(part: usize) -> Vec<usize> {
    let permutation = vae_rope_permutation();
    (0..VAE_HEADS)
        .flat_map(move |h| {
            (0..VAE_HEAD_DIM).map(move |d| {
                let channel = if part < 2 { permutation[d] } else { d };
                h * 3 * VAE_HEAD_DIM + part * VAE_HEAD_DIM + channel
            })
        })
        .collect()
}

impl Reader<'_> {
    fn vector(&mut self, name: &str, len: usize) -> Result<Tensor> {
        let raw = self.file.read(name, Some(DType::F16), &[len])?;
        let values = bf16_bytes_of(&f16_vec(&raw));
        upload(
            &Tensor::from_bf16(vec![len], &values)?,
            self.ctx.device_id(),
        )
    }

    /// Read a `[rows, inp]` F16 matrix with bias and cut it into linears, one per row selection. Each selection lists the
    /// source rows (in order) that form one `[inp, selection.len()]` BF16 linear.
    fn split(
        &mut self,
        prefix: &str,
        rows: usize,
        inp: usize,
        selections: &[Vec<usize>],
    ) -> Result<Vec<VaeLinear>> {
        let all = f16_vec(&self.file.read(
            &format!("{prefix}.weight"),
            Some(DType::F16),
            &[rows, inp],
        )?);
        let biases = f16_vec(&self.file.read(
            &format!("{prefix}.bias"),
            Some(DType::F16),
            &[rows],
        )?);
        selections
            .iter()
            .map(|selection| {
                let mut part = Vec::with_capacity(selection.len() * inp);
                for &r in selection {
                    part.extend_from_slice(&all[r * inp..(r + 1) * inp]);
                }
                let packed = bf16_bytes_of(&transpose(&part, selection.len(), inp));
                let weight = upload(
                    &Tensor::from_bf16(vec![inp, selection.len()], &packed)?,
                    self.ctx.device_id(),
                )?;
                let bias: Vec<_> = selection.iter().map(|&r| biases[r]).collect();
                let bias = upload(
                    &Tensor::from_bf16(vec![selection.len()], &bf16_bytes_of(&bias))?,
                    self.ctx.device_id(),
                )?;
                Ok(VaeLinear {
                    weight,
                    bias: Some(bias),
                })
            })
            .collect()
    }

    /// `[0; 0; scale]` as BF16 `[3 * VAE_DIM]`, the packed-style form of a per-channel residual scale.
    fn residual_style(&mut self, name: &str) -> Result<Tensor> {
        let scale = bf16_bytes_of(&f16_vec(&self.file.read(
            name,
            Some(DType::F16),
            &[VAE_DIM],
        )?));
        let mut style = vec![half::bf16::ZERO; 2 * VAE_DIM];
        style.extend(scale);
        upload(
            &Tensor::from_bf16(vec![3 * VAE_DIM], &style)?,
            self.ctx.device_id(),
        )
    }

    fn linear(&mut self, prefix: &str, out: usize, inp: usize) -> Result<VaeLinear> {
        Ok(self
            .split(prefix, out, inp, &[(0..out).collect()])?
            .remove(0))
    }
}

impl VaeWeights {
    pub fn load(path: &Path, ctx: &CudaContext) -> Result<Self> {
        let mut file = Checkpoint::open(path)?;
        // The encoder and its quantizer are only needed to encode reference images, which this family does not do.
        let skipped = file.ignore_prefix("encoder.") + file.ignore_prefix("quant_conv.");
        if skipped == 0 {
            return Err(Error::Other(format!(
                "{}: not a MiniMax-H3-Image VAE (no encoder tensors)",
                path.display()
            )));
        }
        file.ignore_prefix("decoder.mask_token"); // unused at inference; present so the file loads without leftovers
        let mut r = Reader { file, ctx };

        // a 1x1 convolution over channels is a linear on channels; read the three affine stages and fold them
        let f32s =
            |bytes: Vec<u8>| -> Vec<f32> { f16_vec(&bytes).iter().map(|v| v.to_f32()).collect() };
        let pq_w = f32s(r.file.read(
            "post_quant_conv.weight",
            Some(DType::F16),
            &[LATENT_CHANNELS, LATENT_CHANNELS, 1, 1],
        )?);
        let pq_b = f32s(r.file.read(
            "post_quant_conv.bias",
            Some(DType::F16),
            &[LATENT_CHANNELS],
        )?);
        let e_w = f32s(r.file.read(
            "decoder.x_embedder.weight",
            Some(DType::F16),
            &[VAE_DIM, LATENT_CHANNELS],
        )?);
        let e_b = f32s(
            r.file
                .read("decoder.x_embedder.bias", Some(DType::F16), &[VAE_DIM])?,
        );
        let (folded_w, folded_b) = fold_embedding(&pq_w, &pq_b, &e_w, &e_b);
        let embed = VaeLinearF32 {
            weight: upload(
                &Tensor::from_f32(vec![LATENT_CHANNELS, VAE_DIM], &folded_w)?,
                ctx.device_id(),
            )?,
            bias: upload(
                &Tensor::from_f32(vec![VAE_DIM], &folded_b)?,
                ctx.device_id(),
            )?,
        };
        let ones = upload(
            &Tensor::from_bf16(vec![VAE_HEAD_DIM], &[half::bf16::ONE; VAE_HEAD_DIM])?,
            ctx.device_id(),
        )?;
        let registers = {
            let raw = r.file.read(
                "decoder.register_tokens",
                Some(DType::F16),
                &[1, VAE_REGISTERS, VAE_DIM],
            )?;
            upload(
                &Tensor::from_bf16(vec![VAE_REGISTERS, VAE_DIM], &bf16_bytes_of(&f16_vec(&raw)))?,
                ctx.device_id(),
            )?
        };
        let mut blocks = Vec::with_capacity(VAE_BLOCKS);
        for i in 0..VAE_BLOCKS {
            let p = format!("decoder.transformer_blocks.{i}");
            let mut qkv = r
                .split(
                    &format!("{p}.attn.to_qkv"),
                    3 * VAE_DIM,
                    VAE_DIM,
                    &[qkv_rows(0), qkv_rows(1), qkv_rows(2)],
                )?
                .into_iter();
            // `w1` is `[gate; up]` in contiguous halves (comfy `_swiglu_eager`: `gate, up = x.chunk(2)`)
            let halves = [(0..VAE_FFN).collect(), (VAE_FFN..2 * VAE_FFN).collect()];
            let mut ff = r
                .split(&format!("{p}.ff.w1"), 2 * VAE_FFN, VAE_DIM, &halves)?
                .into_iter();
            blocks.push(VaeBlock {
                norm1: r.vector(&format!("{p}.norm1.weight"), VAE_DIM)?,
                norm2: r.vector(&format!("{p}.norm2.weight"), VAE_DIM)?,
                q: qkv.next().unwrap(),
                k: qkv.next().unwrap(),
                v: qkv.next().unwrap(),
                out: r.linear(&format!("{p}.attn.to_out"), VAE_DIM, VAE_DIM)?,
                gate: ff.next().unwrap(),
                up: ff.next().unwrap(),
                down: r.linear(&format!("{p}.ff.w2"), VAE_DIM, VAE_FFN)?,
                style1: r.residual_style(&format!("{p}.scale1"))?,
                style2: r.residual_style(&format!("{p}.scale2"))?,
            });
        }
        let norm_out_weight = r.vector("decoder.norm_out.weight", VAE_DIM)?;
        let norm_out_bias = r.vector("decoder.norm_out.bias", VAE_DIM)?;
        let proj_out = r.linear("decoder.proj_out", VAE_OUT_DIM, VAE_DIM)?;
        r.file.finish()?;
        Ok(Self {
            embed,
            ones,
            registers,
            blocks,
            norm_out_weight,
            norm_out_bias,
            proj_out,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qkv_rows_are_the_per_head_interleaved_thirds() {
        let (q, k, v) = (qkv_rows(0), qkv_rows(1), qkv_rows(2));
        for part in [&q, &k, &v] {
            assert_eq!(part.len(), VAE_DIM);
        }
        // head 0 q is rows 0..64, k is 64..128, v is 128..192; head 1 q starts at 192
        assert_eq!((q[0], q[63], q[64]), (0, 63, 192));
        assert_eq!((k[0], k[63], k[64]), (64, 127, 256));
        assert_eq!((v[0], v[63], v[64]), (128, 191, 320));
        // together the three selections cover every row of the fused projection exactly once
        let mut all: Vec<usize> = q.into_iter().chain(k).chain(v).collect();
        all.sort_unstable();
        assert_eq!(all, (0..3 * VAE_DIM).collect::<Vec<_>>());
    }

    #[test]
    fn folded_embedding_equals_the_three_stages() {
        let c = LATENT_CHANNELS;
        let pq_w: Vec<f32> = (0..c * c)
            .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.03)
            .collect();
        let pq_b: Vec<f32> = (0..c).map(|i| (i as f32 - 12.0) * 0.01).collect();
        let e_w: Vec<f32> = (0..VAE_DIM * c)
            .map(|i| ((i * 13 % 17) as f32 - 8.0) * 0.02)
            .collect();
        let e_b: Vec<f32> = (0..VAE_DIM).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
        let (w, b) = fold_embedding(&pq_w, &pq_b, &e_w, &e_b);
        let z: Vec<f32> = (0..c).map(|i| ((i * 5 % 9) as f32 - 4.0) * 0.3).collect();
        for d in [0, 1, 511, VAE_DIM - 1] {
            // reference: denormalise, 1x1 conv, embed
            let conv: Vec<f64> = (0..c)
                .map(|o| {
                    pq_b[o] as f64
                        + (0..c)
                            .map(|i| {
                                pq_w[o * c + i] as f64
                                    * (z[i] as f64 * LATENTS_STD[i] as f64 + LATENTS_MEAN[i] as f64)
                            })
                            .sum::<f64>()
                })
                .collect();
            let direct =
                e_b[d] as f64 + (0..c).map(|o| e_w[d * c + o] as f64 * conv[o]).sum::<f64>();
            let folded = b[d] as f64
                + (0..c)
                    .map(|i| z[i] as f64 * w[i * VAE_DIM + d] as f64)
                    .sum::<f64>();
            assert!(
                (direct - folded).abs() < 1e-4 * direct.abs().max(1.0),
                "d={d}: {direct} vs {folded}"
            );
        }
    }

    #[test]
    fn rope_permutation_pairs_rotated_channels_half_a_head_apart() {
        let p = vae_rope_permutation();
        let mut seen = [false; VAE_HEAD_DIM];
        for &c in &p {
            assert!(!seen[c]);
            seen[c] = true;
        }
        assert!(seen.iter().all(|s| *s));
        // the decoder rotates (i, i + 16) for i < 16; after permutation those sit at (j, j + 32)
        for i in 0..16 {
            assert_eq!(p[i + 32], p[i] + 16);
        }
        // pass-through channels 32..64 pair with themselves, 16 apart in the model and 32 apart after permutation
        for j in 16..32 {
            assert_eq!(p[j] + 16, p[j + 32]);
            assert!(p[j] >= 32 && p[j + 32] >= 48);
        }
        // q rows of head 1 start after head 0's 192 rows; the permutation applies inside the head
        let q = qkv_rows(0);
        assert_eq!(q[64..68], [192, 193, 194, 195]);
        assert_eq!(q[32], 16);
        let v = qkv_rows(2);
        assert_eq!(v[32], 128 + 32);
    }
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;

    /// `APXINF_H3_IMAGE_VAE=/path/minimax_h3_image_vae_fp16.safetensors cargo test --release -p apxinf-model \
    ///  --features cuda --lib -- --ignored loads_the_released_vae`
    #[test]
    #[ignore = "requires the Comfy-Org VAE checkpoint and a CUDA GPU"]
    fn loads_the_released_vae() {
        let path = std::env::var("APXINF_H3_IMAGE_VAE").expect("APXINF_H3_IMAGE_VAE");
        let ctx = CudaContext::new(0).unwrap();
        let weights = VaeWeights::load(Path::new(&path), &ctx).unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(weights.blocks.len(), VAE_BLOCKS);
        assert_eq!(
            weights.embed.weight.shape().dims(),
            [LATENT_CHANNELS, VAE_DIM]
        );
        assert_eq!(weights.registers.shape().dims(), [VAE_REGISTERS, VAE_DIM]);
        assert_eq!(
            weights.blocks[0].q.weight.shape().dims(),
            [VAE_DIM, VAE_DIM]
        );
        assert_eq!(
            weights.blocks[23].gate.weight.shape().dims(),
            [VAE_DIM, VAE_FFN]
        );
        assert_eq!(
            weights.proj_out.weight.shape().dims(),
            [VAE_DIM, VAE_OUT_DIM]
        );
    }
}
