//! Causal Wan2.1 encoder. Cache edges point at preceding chunk activations;
//! every inference recomputes the singleton chunk before consuming those edges.
//! No activation state crosses observation boundaries.
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use apxinf_cuda_new::ops::{embedding_gather, GemmPolicy};
use apxinf_cuda_new::{
    attention, bias_residual, channel_l2_norm, pointwise, AttentionArgs, AttentionPolicy,
    BiasResidualArgs, ChannelL2NormArgs, CudaBuffer, CudaContext, PointwiseActivation,
    PointwiseArgs, PointwiseSemantic,
};

use super::super::weights::{upload, Conv, VaeResidual, VaeWeights};
use super::convolution::ConvWorkspace;

fn alloc(ctx: &CudaContext, shape: Vec<usize>) -> Result<Tensor> {
    let bytes = shape.iter().product::<usize>() * 2;
    CudaBuffer::alloc(bytes, ctx.device_id())
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape), DType::BF16)
        .map_err(Error::Cuda)
}

fn matrix(t: &Tensor) -> Result<Tensor> {
    let channels = *t.shape().dims().last().unwrap();
    t.reshape(vec![t.numel() / channels, channels])
}

fn tail(input: &Tensor, frames: usize) -> Result<Tensor> {
    let mut shape = input.shape().dims().to_vec();
    let count = shape[0].min(frames);
    let frame_bytes = input.size_in_bytes() / shape[0];
    let offset = (shape[0] - count) * frame_bytes;
    shape[0] = count;
    CudaBuffer::from_tensor(input)
        .map_err(Error::Cuda)?
        .view(offset, count * frame_bytes)
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape), DType::BF16)
        .map_err(Error::Cuda)
}

fn causal(
    ctx: &CudaContext,
    weights: &Conv,
    input: &Tensor,
    previous: Option<&Tensor>,
) -> Result<ConvWorkspace> {
    let history = previous.map(|p| tail(p, 2)).transpose()?;
    let past = history.as_ref().map_or(0, |h| h.shape().dims()[0]);
    ConvWorkspace::prepare(
        ctx,
        weights,
        input,
        history.as_ref(),
        [1; 3],
        [2 - past, 1, 1],
        [0, 1, 1],
    )
}

fn normalize_activate(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    output: &Tensor,
) -> Result<()> {
    let input = matrix(input)?;
    let mut output = matrix(output)?;
    channel_l2_norm(
        ctx,
        ChannelL2NormArgs {
            input: &input,
            weight,
            out: &mut output,
            eps: 1e-12,
        },
    )?;
    let input = output.clone();
    let mut args = PointwiseArgs::new(PointwiseSemantic::BiasActivation, &input, &mut output);
    args.activation = PointwiseActivation::Silu;
    pointwise(ctx, args)
}

fn add(ctx: &CudaContext, a: &Tensor, b: &Tensor, out: &Tensor) -> Result<()> {
    bias_residual(
        ctx,
        BiasResidualArgs::new(&matrix(a)?, None, &matrix(b)?, &mut matrix(out)?),
    )
}

struct Residual {
    input: Tensor,
    norm1: Tensor,
    conv1: ConvWorkspace,
    norm2: Tensor,
    conv2: ConvWorkspace,
    shortcut: Option<ConvWorkspace>,
    output: Tensor,
}

impl Residual {
    fn prepare(
        ctx: &CudaContext,
        w: &VaeResidual,
        input: &Tensor,
        previous: Option<&Self>,
    ) -> Result<Self> {
        let norm1 = alloc(ctx, input.shape().dims().to_vec())?;
        let conv1 = causal(ctx, &w.conv1, &norm1, previous.map(|p| &p.norm1))?;
        let norm2 = alloc(ctx, conv1.output.shape().dims().to_vec())?;
        let conv2 = causal(ctx, &w.conv2, &norm2, previous.map(|p| &p.norm2))?;
        let shortcut = w
            .shortcut
            .as_ref()
            .map(|weight| ConvWorkspace::prepare(ctx, weight, input, None, [1; 3], [0; 3], [0; 3]))
            .transpose()?;
        let output = alloc(ctx, conv2.output.shape().dims().to_vec())?;
        Ok(Self {
            input: input.clone(),
            norm1,
            conv1,
            norm2,
            conv2,
            shortcut,
            output,
        })
    }

    fn forward(&mut self, ctx: &CudaContext, w: &VaeResidual, policy: &GemmPolicy) -> Result<()> {
        normalize_activate(ctx, &self.input, &w.norm1, &self.norm1)?;
        self.conv1.forward(ctx, &w.conv1, policy)?;
        normalize_activate(ctx, &self.conv1.output, &w.norm2, &self.norm2)?;
        self.conv2.forward(ctx, &w.conv2, policy)?;
        let residual = match (&mut self.shortcut, &w.shortcut) {
            (Some(shortcut), Some(weights)) => {
                shortcut.forward(ctx, weights, policy)?;
                &shortcut.output
            }
            _ => &self.input,
        };
        add(ctx, &self.conv2.output, residual, &self.output)
    }
}

struct Downsample {
    spatial: ConvWorkspace,
    temporal: Option<ConvWorkspace>,
}

struct SpatialAttention {
    input: Tensor,
    normalized: Tensor,
    qkv: ConvWorkspace,
    ids: [Tensor; 3],
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attended: Tensor,
    projection: ConvWorkspace,
    output: Tensor,
}

impl SpatialAttention {
    fn prepare(ctx: &CudaContext, w: &VaeWeights, input: &Tensor) -> Result<Self> {
        let dims = input.shape().dims();
        let (t, pixels, c) = (dims[0], dims[1] * dims[2], dims[3]);
        let normalized = alloc(ctx, dims.to_vec())?;
        let qkv = ConvWorkspace::prepare(
            ctx,
            &w.attention_qkv,
            &normalized,
            None,
            [1; 3],
            [0; 3],
            [0; 3],
        )?;
        let ids = [0, 1, 2].map(|part| {
            let indices: Vec<i32> = (0..t * pixels).map(|row| (row * 3 + part) as i32).collect();
            upload(
                &Tensor::from_i32(vec![t * pixels], &indices)?,
                ctx.device_id(),
            )
        });
        let [q_ids, k_ids, v_ids] = ids;
        let q = alloc(ctx, vec![t, pixels, 1, c])?;
        let k = alloc(ctx, vec![t, pixels, 1, c])?;
        let v = alloc(ctx, vec![t, pixels, 1, c])?;
        let attended = alloc(ctx, dims.to_vec())?;
        let projection = ConvWorkspace::prepare(
            ctx,
            &w.attention_output,
            &attended,
            None,
            [1; 3],
            [0; 3],
            [0; 3],
        )?;
        Ok(Self {
            input: input.clone(),
            normalized,
            qkv,
            ids: [q_ids?, k_ids?, v_ids?],
            q,
            k,
            v,
            attended,
            projection,
            output: alloc(ctx, dims.to_vec())?,
        })
    }

    fn forward(
        &mut self,
        ctx: &CudaContext,
        w: &VaeWeights,
        gemm: &GemmPolicy,
        attention_policy: &AttentionPolicy,
    ) -> Result<()> {
        channel_l2_norm(
            ctx,
            ChannelL2NormArgs {
                input: &matrix(&self.input)?,
                weight: &w.attention_norm,
                out: &mut matrix(&self.normalized)?,
                eps: 1e-12,
            },
        )?;
        self.qkv.forward(ctx, &w.attention_qkv, gemm)?;
        let rows = self.q.numel() / 384;
        let table = self.qkv.output.reshape(vec![rows * 3, 384])?;
        for (ids, output) in self.ids.iter().zip([&self.q, &self.k, &self.v]) {
            embedding_gather(ctx, &table, ids, &output.reshape(vec![rows, 384])?)?;
        }
        let mut output = self.attended.reshape(self.q.shape().clone())?;
        let mut args = AttentionArgs::new(&self.q, &self.k, &self.v, &mut output);
        args.policy = attention_policy.clone();
        attention(ctx, args)?;
        self.projection.forward(ctx, &w.attention_output, gemm)?;
        add(ctx, &self.projection.output, &self.input, &self.output)
    }
}

pub(super) struct VaeChunk {
    input: Tensor,
    input_conv: ConvWorkspace,
    blocks: Vec<[Residual; 2]>,
    downsample: Vec<Downsample>,
    middle_before: Residual,
    attention: SpatialAttention,
    middle_after: Residual,
    head_norm: Tensor,
    head_conv: ConvWorkspace,
    mean_projection: ConvWorkspace,
    mean_indices: Tensor,
    negative_mean: Tensor,
    inverse_std: Tensor,
    pub output: Tensor,
}

impl VaeChunk {
    pub fn prepare(
        ctx: &CudaContext,
        w: &VaeWeights,
        input: &Tensor,
        previous: Option<&Self>,
    ) -> Result<Self> {
        let input_conv = causal(ctx, &w.input, input, previous.map(|p| &p.input))?;
        let mut current = input_conv.output.clone();
        let mut blocks = Vec::new();
        let mut downsample = Vec::new();
        for (level, weights) in w.down_blocks.iter().enumerate() {
            let first = Residual::prepare(
                ctx,
                &weights[0],
                &current,
                previous.map(|p| &p.blocks[level][0]),
            )?;
            let second = Residual::prepare(
                ctx,
                &weights[1],
                &first.output,
                previous.map(|p| &p.blocks[level][1]),
            )?;
            current = second.output.clone();
            blocks.push([first, second]);
            if level < 3 {
                let spatial = ConvWorkspace::prepare(
                    ctx,
                    &w.downsample[level].spatial,
                    &current,
                    None,
                    [1, 2, 2],
                    [0; 3],
                    [0, 1, 1],
                )?;
                let temporal = match (&w.downsample[level].temporal, previous) {
                    (Some(weight), Some(p)) => {
                        let history = tail(&p.downsample[level].spatial.output, 1)?;
                        Some(ConvWorkspace::prepare(
                            ctx,
                            weight,
                            &spatial.output,
                            Some(&history),
                            [2, 1, 1],
                            [0; 3],
                            [0; 3],
                        )?)
                    }
                    _ => None, // The initial singleton bypasses temporal decimation.
                };
                current = temporal
                    .as_ref()
                    .map_or(&spatial.output, |t| &t.output)
                    .clone();
                downsample.push(Downsample { spatial, temporal });
            }
        }
        let middle_before = Residual::prepare(
            ctx,
            &w.middle_before,
            &current,
            previous.map(|p| &p.middle_before),
        )?;
        let attention = SpatialAttention::prepare(ctx, w, &middle_before.output)?;
        let middle_after = Residual::prepare(
            ctx,
            &w.middle_after,
            &attention.output,
            previous.map(|p| &p.middle_after),
        )?;
        let head_norm = alloc(ctx, middle_after.output.shape().dims().to_vec())?;
        let head_conv = causal(ctx, &w.output, &head_norm, previous.map(|p| &p.head_norm))?;
        let mean_projection = ConvWorkspace::prepare(
            ctx,
            &w.mean_projection,
            &head_conv.output,
            None,
            [1; 3],
            [0; 3],
            [0; 3],
        )?;
        let rows = mean_projection.output.numel() / 32;
        let indices: Vec<i32> = (0..rows)
            .flat_map(|r| (0..16).map(move |c| (r * 32 + c) as i32))
            .collect();
        let mean_indices = upload(
            &Tensor::from_i32(vec![rows * 16], &indices)?,
            ctx.device_id(),
        )?;
        let mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715,
            0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
        ];
        let std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652,
            1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
        ];
        let negative: Vec<_> = mean.into_iter().map(|v| half::bf16::from_f32(-v)).collect();
        let inverse: Vec<_> = std
            .into_iter()
            .map(|v| half::bf16::from_f32(1.0 / half::bf16::from_f32(v).to_f32()))
            .collect();
        let negative_mean = upload(&Tensor::from_bf16(vec![16], &negative)?, ctx.device_id())?;
        let inverse_std = upload(
            &Tensor::from_bf16(vec![rows, 16], &inverse.repeat(rows))?,
            ctx.device_id(),
        )?;
        let mut shape = mean_projection.output.shape().dims().to_vec();
        shape[3] = 16;
        let output = alloc(ctx, shape)?;
        Ok(Self {
            input: input.clone(),
            input_conv,
            blocks,
            downsample,
            middle_before,
            attention,
            middle_after,
            head_norm,
            head_conv,
            mean_projection,
            mean_indices,
            negative_mean,
            inverse_std,
            output,
        })
    }

    pub fn forward(
        &mut self,
        ctx: &CudaContext,
        w: &VaeWeights,
        gemm: &GemmPolicy,
        attention: &AttentionPolicy,
    ) -> Result<()> {
        self.input_conv.forward(ctx, &w.input, gemm)?;
        for level in 0..4 {
            for block in 0..2 {
                self.blocks[level][block].forward(ctx, &w.down_blocks[level][block], gemm)?;
            }
            if level < 3 {
                let down = &mut self.downsample[level];
                down.spatial
                    .forward(ctx, &w.downsample[level].spatial, gemm)?;
                if let (Some(temporal), Some(weights)) =
                    (&mut down.temporal, &w.downsample[level].temporal)
                {
                    temporal.forward(ctx, weights, gemm)?;
                }
            }
        }
        self.middle_before.forward(ctx, &w.middle_before, gemm)?;
        self.attention.forward(ctx, w, gemm, attention)?;
        self.middle_after.forward(ctx, &w.middle_after, gemm)?;
        normalize_activate(
            ctx,
            &self.middle_after.output,
            &w.output_norm,
            &self.head_norm,
        )?;
        self.head_conv.forward(ctx, &w.output, gemm)?;
        self.mean_projection
            .forward(ctx, &w.mean_projection, gemm)?;
        embedding_gather(
            ctx,
            &self
                .mean_projection
                .output
                .reshape(vec![self.mean_projection.output.numel(), 1])?,
            &self.mean_indices,
            &self.output.reshape(vec![self.output.numel(), 1])?,
        )?;
        let input = matrix(&self.output)?;
        let mut out = input.clone();
        let mut args = PointwiseArgs::new(PointwiseSemantic::BiasActivation, &input, &mut out);
        args.bias = Some(&self.negative_mean);
        pointwise(ctx, args)?;
        let mut args = PointwiseArgs::new(PointwiseSemantic::Multiply, &input, &mut out);
        args.secondary = Some(&self.inverse_std);
        pointwise(ctx, args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_cuda_new::ExecutionSession;
    use std::path::PathBuf;

    // External fixtures are captured from the pinned official checkpoint. Keep
    // their bytes outside Git; this test never downloads or fabricates a golden.
    #[test]
    #[ignore = "requires Thor, official converted weights and reference VAE fixture"]
    fn official_vae_conditioning_prefix() -> Result<()> {
        let root =
            PathBuf::from(std::env::var("APXINF_COSMOS_EVIDENCE").expect("APXINF_COSMOS_EVIDENCE"));
        let ctx = CudaContext::new(0).map_err(Error::Cuda)?;
        let weights = VaeWeights::load(&root.join("native-assets/vae_encoder.safetensors"), 0)?;
        let (fixture, _) =
            apxinf_loader::safetensors::load_native_path(&root.join("vae-fixture.safetensors"))
                .map_err(|e| Error::Other(e.to_string()))?;
        let input = upload(&fixture["input"], 0)?;
        let buffer = CudaBuffer::from_tensor(&input).map_err(Error::Cuda)?;
        let frame_bytes = 224 * 224 * 3 * 2;
        let first = buffer
            .view(0, frame_bytes)
            .map_err(Error::Cuda)?
            .as_tensor(Shape::new(vec![1, 224, 224, 3]), DType::BF16)
            .map_err(Error::Cuda)?;
        let rest = buffer
            .view(frame_bytes, 16 * frame_bytes)
            .map_err(Error::Cuda)?
            .as_tensor(Shape::new(vec![16, 224, 224, 3]), DType::BF16)
            .map_err(Error::Cuda)?;
        let mut first = VaeChunk::prepare(&ctx, &weights, &first, None)?;
        let mut rest = VaeChunk::prepare(&ctx, &weights, &rest, Some(&first))?;
        let session = ExecutionSession::with_capacity(256 * 1024 * 1024, 0)?;
        let gemm = GemmPolicy {
            online_tune: false,
            ..Default::default()
        };
        let attention = AttentionPolicy {
            online_tune: false,
            ..Default::default()
        };
        let collect = |first: &VaeChunk, rest: &VaeChunk| -> Result<Vec<u8>> {
            let mut bytes = Vec::new();
            for output in [&first.output, &rest.output] {
                let start = bytes.len();
                bytes.resize(start + output.size_in_bytes(), 0);
                CudaBuffer::from_tensor(output)
                    .map_err(Error::Cuda)?
                    .copy_to_host(&mut bytes[start..])
                    .map_err(Error::Cuda)?;
            }
            Ok(bytes)
        };
        session.prepare(|| {
            first.forward(&ctx, &weights, &gemm, &attention)?;
            rest.forward(&ctx, &weights, &gemm, &attention)
        })?;
        ctx.synchronize().map_err(Error::Cuda)?;
        let eager = collect(&first, &rest)?;
        let graph = apxinf_cuda_new::capture(&ctx, || {
            session.run(|| {
                first.forward(&ctx, &weights, &gemm, &attention)?;
                rest.forward(&ctx, &weights, &gemm, &attention)
            })
        })?;
        graph.replay()?;
        ctx.synchronize().map_err(Error::Cuda)?;
        let bytes = collect(&first, &rest)?;
        assert_eq!(bytes, eager, "VAE eager and captured outputs differ");
        std::fs::write(root.join("vae-native-output.bf16"), &bytes)?;
        let actual: Vec<_> = bytes
            .chunks_exact(2)
            .map(|b| half::bf16::from_bits(u16::from_ne_bytes([b[0], b[1]])).to_f32() as f64)
            .collect();
        let expected: Vec<_> = fixture["expected"]
            .as_bf16()?
            .iter()
            .map(|v| v.to_f32() as f64)
            .collect();
        assert_eq!(actual.len(), expected.len());
        for (frame, (a, b)) in actual
            .chunks_exact(28 * 28 * 16)
            .zip(expected.chunks_exact(28 * 28 * 16))
            .enumerate()
        {
            let error = a.iter().zip(b).map(|(a, b)| (a - b).powi(2)).sum::<f64>();
            let aa = a.iter().map(|a| a * a).sum::<f64>();
            let bb = b.iter().map(|b| b * b).sum::<f64>();
            let ab = a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
            let relative = (error / bb.max(1e-30)).sqrt();
            let cosine = ab / (aa * bb).sqrt();
            let max_abs = a
                .iter()
                .zip(b)
                .map(|(a, b)| (a - b).abs())
                .fold(0f64, f64::max);
            eprintln!(
                "VAE frame {frame}: relative_l2={relative} cosine={cosine} max_abs={max_abs}"
            );
            assert!(
                relative <= 0.02 && cosine >= 0.999,
                "VAE frame {frame} differs from official capture"
            );
        }
        Ok(())
    }
}
