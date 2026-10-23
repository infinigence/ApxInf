//! Complete joint-latent transformer. All 1,764 tokens pass through every block.
//! Scratch is reused between blocks and denoising steps; it contains no runner
//! or capture policy. Cached text is supplied by the family model.
use super::super::weights::{upload, Attention, DitWeights, Modulation};
use super::tensor_ops::*;
use apxinf_core::{DType, Result, Tensor};
use apxinf_cuda_new::ops::GemmPolicy;
use apxinf_cuda_new::{
    attention, layer_norm, rms_norm, AttentionArgs, AttentionPolicy, CudaContext, LayerNormArgs,
    PointwiseActivation, RmsNormArgs,
};

const TOKENS: usize = 9 * 14 * 14;
const WIDTH: usize = 2048;

struct Rotary {
    swap: Tensor,
    cos: Tensor,
    signed_sin: Tensor,
    swapped: Tensor,
    a: Tensor,
    b: Tensor,
}
impl Rotary {
    fn new(ctx: &CudaContext, w: &DitWeights, ntk: [f32; 3]) -> Result<Self> {
        // Source buffers have already been rounded to BF16 by checkpoint load.
        let temporal = w.temporal_range.as_bf16()?;
        let spatial = w.spatial_range.as_bf16()?;
        let seq = w.position_seq.as_bf16()?;
        let mut cos = Vec::with_capacity(TOKENS * WIDTH);
        let mut sin = Vec::with_capacity(TOKENS * WIDTH);
        let mut swap = Vec::with_capacity(TOKENS * WIDTH);
        for token in 0..TOKENS {
            let coordinates = [token / 196, token / 14 % 14, token % 14];
            for head in 0..16 {
                for c in 0..128 {
                    let half = c % 64;
                    let (axis, exponent) = if half < 22 {
                        (0, temporal[half])
                    } else if half < 43 {
                        (1, spatial[half - 22])
                    } else {
                        (2, spatial[half - 43])
                    };
                    let angle = seq[coordinates[axis]].to_f32()
                        * (1. / (10000. * ntk[axis]).powf(exponent.to_f32()));
                    cos.push(angle.cos());
                    sin.push(angle.sin() * if c < 64 { -1. } else { 1. });
                    swap.push((token * WIDTH + head * 128 + (c + 64) % 128) as i32);
                }
            }
        }
        Ok(Self {
            swap: indices(ctx, &swap)?,
            cos: upload(
                &Tensor::from_f32(vec![TOKENS, WIDTH], &cos)?,
                ctx.device_id(),
            )?,
            signed_sin: upload(
                &Tensor::from_f32(vec![TOKENS, WIDTH], &sin)?,
                ctx.device_id(),
            )?,
            swapped: bf(ctx, TOKENS, WIDTH)?,
            a: alloc(ctx, vec![TOKENS, WIDTH], DType::F32)?,
            b: alloc(ctx, vec![TOKENS, WIDTH], DType::F32)?,
        })
    }
    fn apply(&self, ctx: &CudaContext, input: &Tensor) -> Result<()> {
        gather(ctx, input, &self.swap, 1, &self.swapped)?;
        cast(ctx, input, &self.a)?;
        cast(ctx, &self.swapped, &self.b)?;
        multiply(ctx, &self.a, &self.cos, &self.a)?;
        multiply(ctx, &self.b, &self.signed_sin, &self.b)?;
        add(ctx, &self.a, &self.b, &self.a)?;
        cast(ctx, &self.a, input)
    }
}

pub(super) struct Dit {
    pub patches: Tensor,
    pub output: Tensor,
    pub hidden: Tensor,
    normalized: Tensor,
    ones: Tensor,
    zeros: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    text_k: Tensor,
    text_v: Tensor,
    attended: Tensor,
    projected: Tensor,
    mlp: Tensor,
    time_hidden: Tensor,
    time_global: Tensor,
    time_normalized: Tensor,
    time_activated: Tensor,
    modulation_hidden: Tensor,
    modulation: Tensor,
    final_modulation: Tensor,
    final_global: Tensor,
    final_global_ids: Tensor,
    broadcast_ids: [Tensor; 3],
    final_ids: [Tensor; 2],
    scale: Tensor,
    shift: Tensor,
    gate: Tensor,
    rotary: Rotary,
}
impl Dit {
    pub fn prepare(ctx: &CudaContext, w: &DitWeights, ntk: [f32; 3]) -> Result<Self> {
        let maps = [0, 1, 2].map(|part| {
            indices(
                ctx,
                &(0..TOKENS)
                    .map(|r| (r / 196 * 3 + part) as i32)
                    .collect::<Vec<_>>(),
            )
        });
        let [shift, scale, gate] = maps;
        let maps = [0, 1].map(|part| {
            indices(
                ctx,
                &(0..TOKENS)
                    .map(|r| (r / 196 * 2 + part) as i32)
                    .collect::<Vec<_>>(),
            )
        });
        let [final_shift, final_scale] = maps;
        let final_global_ids = indices(
            ctx,
            &(0..9)
                .flat_map(|t| [t * 3, t * 3 + 1])
                .collect::<Vec<i32>>(),
        )?;
        Ok(Self {
            patches: bf(ctx, TOKENS, 72)?,
            output: bf(ctx, TOKENS, 64)?,
            hidden: bf(ctx, TOKENS, WIDTH)?,
            normalized: bf(ctx, TOKENS, WIDTH)?,
            ones: constant(ctx, WIDTH, 1.)?,
            zeros: constant(ctx, WIDTH, 0.)?,
            q: bf(ctx, TOKENS, WIDTH)?,
            k: bf(ctx, TOKENS, WIDTH)?,
            v: bf(ctx, TOKENS, WIDTH)?,
            text_k: bf(ctx, 512, WIDTH)?,
            text_v: bf(ctx, 512, WIDTH)?,
            attended: bf(ctx, TOKENS, WIDTH)?,
            projected: bf(ctx, TOKENS, WIDTH)?,
            mlp: bf(ctx, TOKENS, 8192)?,
            time_hidden: bf(ctx, 9, WIDTH)?,
            time_global: bf(ctx, 9, 6144)?,
            time_normalized: bf(ctx, 9, WIDTH)?,
            time_activated: bf(ctx, 9, WIDTH)?,
            modulation_hidden: bf(ctx, 9, 256)?,
            modulation: bf(ctx, 9, 6144)?,
            final_modulation: bf(ctx, 9, 4096)?,
            final_global: bf(ctx, 9, 4096)?,
            final_global_ids,
            broadcast_ids: [shift?, scale?, gate?],
            final_ids: [final_shift?, final_scale?],
            scale: bf(ctx, TOKENS, WIDTH)?,
            shift: bf(ctx, TOKENS, WIDTH)?,
            gate: bf(ctx, TOKENS, WIDTH)?,
            rotary: Rotary::new(ctx, w, ntk)?,
        })
    }
    fn modulate(
        &self,
        ctx: &CudaContext,
        w: &Modulation,
        final_layer: bool,
        policy: &GemmPolicy,
    ) -> Result<()> {
        linear(
            ctx,
            &self.time_activated,
            &w.down,
            &self.modulation_hidden,
            policy,
        )?;
        let (modulation, global, ids) = if final_layer {
            (
                &self.final_modulation,
                &self.final_global,
                &self.final_ids[..],
            )
        } else {
            (&self.modulation, &self.time_global, &self.broadcast_ids[..])
        };
        linear(ctx, &self.modulation_hidden, &w.up, modulation, policy)?;
        add(ctx, modulation, global, modulation)?;
        gather(ctx, modulation, &ids[0], WIDTH, &self.shift)?;
        gather(ctx, modulation, &ids[1], WIDTH, &self.scale)?;
        if !final_layer {
            gather(ctx, modulation, &ids[2], WIDTH, &self.gate)?;
        }
        // These three BF16 rounding boundaries are present in the reference.
        activate(
            ctx,
            &self.scale,
            Some(&self.ones),
            PointwiseActivation::None,
            &self.scale,
        )?;
        layer_norm(
            ctx,
            LayerNormArgs::new(
                &self.hidden,
                &self.ones,
                &self.zeros,
                &mut self.normalized.clone(),
                1e-6,
            ),
        )?;
        multiply(ctx, &self.normalized, &self.scale, &self.normalized)?;
        add(ctx, &self.normalized, &self.shift, &self.normalized)
    }
    fn attend(
        &self,
        ctx: &CudaContext,
        w: &Attention,
        text: Option<&Tensor>,
        gemm: &GemmPolicy,
        policy: &AttentionPolicy,
    ) -> Result<()> {
        let (context, k, v) = match text {
            Some(t) => (t, &self.text_k, &self.text_v),
            None => (&self.normalized, &self.k, &self.v),
        };
        linear(ctx, &self.normalized, &w.q, &self.q, gemm)?;
        linear(ctx, context, &w.k, k, gemm)?;
        linear(ctx, context, &w.v, v, gemm)?;
        rms_norm(
            ctx,
            RmsNormArgs::new(
                &matrix(&self.q, 128)?,
                &w.q_norm,
                &mut matrix(&self.q, 128)?,
                1e-6,
            ),
        )?;
        rms_norm(
            ctx,
            RmsNormArgs::new(&matrix(k, 128)?, &w.k_norm, &mut matrix(k, 128)?, 1e-6),
        )?;
        if text.is_none() {
            self.rotary.apply(ctx, &self.q)?;
            self.rotary.apply(ctx, k)?;
        }
        let q = self.q.reshape(vec![1, TOKENS, 16, 128])?;
        let k = k.reshape(vec![1, k.numel() / WIDTH, 16, 128])?;
        let v = v.reshape(k.shape().clone())?;
        let mut out = self.attended.reshape(q.shape().clone())?;
        let mut args = AttentionArgs::new(&q, &k, &v, &mut out);
        args.policy = policy.clone();
        attention(ctx, args)?;
        linear(ctx, &self.attended, &w.output, &self.projected, gemm)
    }
    fn residual(&self, ctx: &CudaContext) -> Result<()> {
        multiply(ctx, &self.projected, &self.gate, &self.projected)?;
        add(ctx, &self.hidden, &self.projected, &self.hidden)
    }
    pub fn forward(
        &self,
        ctx: &CudaContext,
        w: &DitWeights,
        time: &Tensor,
        text: &Tensor,
        gemm: &GemmPolicy,
        attention: &AttentionPolicy,
    ) -> Result<()> {
        self.forward_observed(ctx, w, time, text, gemm, attention, |_, _| Ok(()))
    }
    pub fn forward_observed(
        &self,
        ctx: &CudaContext,
        w: &DitWeights,
        time: &Tensor,
        text: &Tensor,
        gemm: &GemmPolicy,
        attention: &AttentionPolicy,
        mut observe: impl FnMut(&str, &Tensor) -> Result<()>,
    ) -> Result<()> {
        linear(ctx, &self.patches, &w.patch, &self.hidden, gemm)?;
        observe("patch", &self.hidden)?;
        linear(ctx, time, &w.time_in, &self.time_hidden, gemm)?;
        activate(
            ctx,
            &self.time_hidden,
            None,
            PointwiseActivation::Silu,
            &self.time_hidden,
        )?;
        linear(ctx, &self.time_hidden, &w.time_out, &self.time_global, gemm)?;
        rms_norm(
            ctx,
            RmsNormArgs::new(time, &w.time_norm, &mut self.time_normalized.clone(), 1e-6),
        )?;
        activate(
            ctx,
            &self.time_normalized,
            None,
            PointwiseActivation::Silu,
            &self.time_activated,
        )?;
        gather(
            ctx,
            &self.time_global,
            &self.final_global_ids,
            WIDTH,
            &self.final_global,
        )?;
        observe("time_lora", &self.time_global)?;
        for (index, block) in w.blocks.iter().enumerate() {
            self.modulate(ctx, &block.self_modulation, false, gemm)?;
            self.attend(ctx, &block.self_attention, None, gemm, attention)?;
            self.residual(ctx)?;
            self.modulate(ctx, &block.cross_modulation, false, gemm)?;
            self.attend(ctx, &block.cross_attention, Some(text), gemm, attention)?;
            self.residual(ctx)?;
            self.modulate(ctx, &block.mlp_modulation, false, gemm)?;
            linear(ctx, &self.normalized, &block.mlp_up, &self.mlp, gemm)?;
            activate(
                ctx,
                &self.mlp,
                None,
                PointwiseActivation::GeluExact,
                &self.mlp,
            )?;
            linear(ctx, &self.mlp, &block.mlp_down, &self.projected, gemm)?;
            self.residual(ctx)?;
            match index {
                0 => observe("block0", &self.hidden)?,
                13 => observe("block13", &self.hidden)?,
                27 => observe("block27", &self.hidden)?,
                _ => {}
            }
        }
        self.modulate(ctx, &w.final_modulation, true, gemm)?;
        linear(
            ctx,
            &self.normalized,
            &w.final_projection,
            &self.output,
            gemm,
        )?;
        observe("final", &self.output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::Error;
    use apxinf_cuda_new::{CudaBuffer, ExecutionSession};
    use std::path::PathBuf;
    #[test]
    #[ignore = "requires Thor, converted official weights and fixed reference DiT fixture"]
    fn official_full_joint_denoiser_layers() -> Result<()> {
        let root =
            PathBuf::from(std::env::var("APXINF_COSMOS_EVIDENCE").expect("APXINF_COSMOS_EVIDENCE"));
        let ctx = CudaContext::new(0).map_err(Error::Cuda)?;
        let weights = DitWeights::load(&root.join("native-assets/model.safetensors"), 0)?;
        let (fixture, _) =
            apxinf_loader::safetensors::load_native_path(&root.join("dit-fixture.safetensors"))
                .map_err(|e| Error::Other(e.to_string()))?;
        let ntk = 3f64.powf(42. / 40.) as f32;
        let dit = Dit::prepare(&ctx, &weights, [1., ntk, ntk])?;
        let patches = upload(&fixture["input"], 0)?;
        apxinf_cuda_new::transfers::copy_cuda_to_cuda(&ctx, &patches, &dit.patches)?;
        let time = upload(&fixture["time"], 0)?;
        let text = upload(&fixture["text"], 0)?;
        let gemm = GemmPolicy {
            online_tune: false,
            ..Default::default()
        };
        let attention = AttentionPolicy {
            online_tune: false,
            ..Default::default()
        };
        let session = ExecutionSession::with_capacity(256 * 1024 * 1024, 0)?;
        let mut failures = Vec::new();
        session.prepare(|| {
            dit.forward_observed(
                &ctx,
                &weights,
                &time,
                &text,
                &gemm,
                &attention,
                |name, tensor| {
                    ctx.synchronize().map_err(Error::Cuda)?;
                    let mut bytes = vec![0; tensor.size_in_bytes()];
                    CudaBuffer::from_tensor(tensor)
                        .map_err(Error::Cuda)?
                        .copy_to_host(&mut bytes)
                        .map_err(Error::Cuda)?;
                    std::fs::write(root.join(format!("dit-native-{name}.bf16")), &bytes)?;
                    let actual: Vec<f64> = bytes
                        .chunks_exact(2)
                        .map(|b| {
                            half::bf16::from_bits(u16::from_ne_bytes(b.try_into().unwrap()))
                                .to_f32() as f64
                        })
                        .collect();
                    let expected: Vec<f64> = fixture[name]
                        .as_bf16()?
                        .iter()
                        .map(|v| v.to_f32() as f64)
                        .collect();
                    assert_eq!(actual.len(), expected.len());
                    for (frame, (a, b)) in actual
                        .chunks_exact(actual.len() / 9)
                        .zip(expected.chunks_exact(expected.len() / 9))
                        .enumerate()
                    {
                        let error = a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
                        let aa = a.iter().map(|a| a * a).sum::<f64>();
                        let bb = b.iter().map(|b| b * b).sum::<f64>();
                        let ab = a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
                        let relative = (error / bb.max(1e-30)).sqrt();
                        let cosine = ab / (aa * bb).sqrt();
                        let max_abs = a.iter().zip(b).map(|(a,b)| (a-b).abs()).fold(0f64, f64::max);
                        eprintln!(
                            "DiT {name} frame {frame}: relative_l2={relative} cosine={cosine} max_abs={max_abs}"
                        );
                        if relative > 0.02
                            || cosine < 0.999
                            || !relative.is_finite()
                            || !cosine.is_finite()
                        {
                            failures.push(format!("{name}/{frame}"));
                        }
                    }
                    Ok(())
                },
            )
        })?;
        assert!(failures.is_empty(), "reference differences: {failures:?}");
        let eager = std::fs::read(root.join("dit-native-final.bf16"))?;
        let graph = apxinf_cuda_new::capture(&ctx, || {
            session.run(|| dit.forward(&ctx, &weights, &time, &text, &gemm, &attention))
        })?;
        graph.replay()?;
        ctx.synchronize().map_err(Error::Cuda)?;
        let mut captured = vec![0; eager.len()];
        CudaBuffer::from_tensor(&dit.output)
            .map_err(Error::Cuda)?
            .copy_to_host(&mut captured)
            .map_err(Error::Cuda)?;
        assert_eq!(captured, eager);
        Ok(())
    }
}
