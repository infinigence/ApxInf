//! BF16 computation and stable intermediate storage, behind Bf16Blocks.
mod tensor_ops {
    //! Family composition helpers; device ownership and kernels stay in cuda-new.
    use crate::cuda_cosmos_policy::backend::ops::{
        embedding_gather, gemm, GemmArgs, GemmPolicy, WeightVersion,
    };
    use crate::cuda_cosmos_policy::backend::{
        pointwise, quantization, CudaBuffer, CudaContext, PointwiseActivation, PointwiseArgs,
        PointwiseSemantic, QuantizationArgs, QuantizationSemantic,
    };
    use crate::cuda_cosmos_policy::weights::{upload, Linear};
    use apxinf_core::{DType, Error, Result, Shape, Tensor};

    pub fn alloc(ctx: &CudaContext, shape: impl Into<Shape>, dtype: DType) -> Result<Tensor> {
        let shape = shape.into();
        let size = shape
            .numel()
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| Error::Other("tensor size overflow".into()))?;
        CudaBuffer::alloc(size, ctx.device_id())
            .map_err(Error::Cuda)?
            .as_tensor(shape, dtype)
            .map_err(Error::Cuda)
    }
    pub fn bf(ctx: &CudaContext, rows: usize, cols: usize) -> Result<Tensor> {
        alloc(ctx, vec![rows, cols], DType::BF16)
    }
    pub fn matrix(t: &Tensor, cols: usize) -> Result<Tensor> {
        t.reshape(vec![t.numel() / cols, cols])
    }
    pub fn indices(ctx: &CudaContext, values: &[i32]) -> Result<Tensor> {
        upload(
            &Tensor::from_i32(vec![values.len()], values)?,
            ctx.device_id(),
        )
    }
    pub fn constant(ctx: &CudaContext, cols: usize, value: f32) -> Result<Tensor> {
        upload(
            &Tensor::from_bf16(vec![cols], &vec![half::bf16::from_f32(value); cols])?,
            ctx.device_id(),
        )
    }
    pub fn linear(
        ctx: &CudaContext,
        input: &Tensor,
        w: &Linear,
        output: &Tensor,
        policy: &GemmPolicy,
    ) -> Result<()> {
        let mut output = output.clone();
        let mut args = GemmArgs::new(input, &w.weight, &mut output);
        args.policy = policy.clone();
        args.weight_version = Some(WeightVersion::new(1));
        gemm(ctx, args)?;
        if let Some(bias) = &w.bias {
            activate(ctx, &output, Some(bias), PointwiseActivation::None, &output)?;
        }
        Ok(())
    }
    pub fn activate(
        ctx: &CudaContext,
        input: &Tensor,
        bias: Option<&Tensor>,
        activation: PointwiseActivation,
        out: &Tensor,
    ) -> Result<()> {
        let mut out = out.clone();
        let mut args = PointwiseArgs::new(PointwiseSemantic::BiasActivation, input, &mut out);
        args.bias = bias;
        args.activation = activation;
        pointwise(ctx, args)
    }
    pub fn multiply(ctx: &CudaContext, a: &Tensor, b: &Tensor, out: &Tensor) -> Result<()> {
        let mut out = out.clone();
        let mut args = PointwiseArgs::new(PointwiseSemantic::Multiply, a, &mut out);
        args.secondary = Some(b);
        pointwise(ctx, args)
    }
    pub fn add(ctx: &CudaContext, a: &Tensor, b: &Tensor, out: &Tensor) -> Result<()> {
        let mut out = out.clone();
        let mut args = PointwiseArgs::new(PointwiseSemantic::EulerUpdate, a, &mut out);
        args.secondary = Some(b);
        args.dt = 1.;
        pointwise(ctx, args)
    }
    pub fn cast(ctx: &CudaContext, input: &Tensor, out: &Tensor) -> Result<()> {
        let semantic = match (input.dtype(), out.dtype()) {
            (DType::BF16, DType::F32) => QuantizationSemantic::CastBf16ToF32,
            (DType::F32, DType::BF16) => QuantizationSemantic::CastF32ToBf16,
            _ => return Err(Error::Other("unsupported Cosmos intermediate cast".into())),
        };
        quantization(
            ctx,
            QuantizationArgs::new(semantic, input, &mut out.clone()),
        )
    }
    pub fn gather(
        ctx: &CudaContext,
        input: &Tensor,
        ids: &Tensor,
        cols: usize,
        out: &Tensor,
    ) -> Result<()> {
        embedding_gather(ctx, &matrix(input, cols)?, ids, &matrix(out, cols)?)
    }
}
mod convolution {
    use crate::cuda_cosmos_policy::backend::ops::{gemm, GemmArgs, GemmPolicy, WeightVersion};
    use crate::cuda_cosmos_policy::backend::{pointwise, PointwiseArgs, PointwiseSemantic};
    use crate::cuda_cosmos_policy::backend::{
        CudaBuffer, CudaContext, PreparedUnfold3d, Unfold3dArgs, Unfold3dGeometry,
    };
    use apxinf_core::{DType, Error, Result, Shape, Tensor};

    use crate::cuda_cosmos_policy::weights::Conv;

    /// Stable per-invocation lowering resources. The shared column buffer is
    /// reused between tiles, with all launches ordered on the model stream.
    pub(in crate::cuda_cosmos_policy::model::blocks::bf16) struct ConvWorkspace {
        tiles: Vec<Tile>,
        pub output: Tensor,
    }

    struct Tile {
        unfold: PreparedUnfold3d,
        columns: Tensor,
        destination: Tensor,
        output: Tensor,
    }

    impl ConvWorkspace {
        pub fn prepare(
            ctx: &CudaContext,
            weights: &Conv,
            input: &Tensor,
            history: Option<&Tensor>,
            stride: [usize; 3],
            padding_before: [usize; 3],
            padding_after: [usize; 3],
        ) -> Result<Self> {
            let dims = input.shape().dims();
            if dims.len() != 4 || dims[3] != weights.input_channels || dims.contains(&0) {
                return Err(Error::Other(
                    "Cosmos VAE convolution input geometry mismatch".into(),
                ));
            }
            let mut output_shape = vec![0; 4];
            for axis in 0..3 {
                let padded = dims[axis]
                    + padding_before[axis]
                    + padding_after[axis]
                    + if axis == 0 {
                        history.map_or(0, |h| h.shape().dims()[0])
                    } else {
                        0
                    };
                if stride[axis] == 0 || padded < weights.kernel[axis] {
                    return Err(Error::Other(
                        "Cosmos VAE convolution padding/stride mismatch".into(),
                    ));
                }
                output_shape[axis] = (padded - weights.kernel[axis]) / stride[axis] + 1;
            }
            output_shape[3] = weights.output_channels;
            let rows: usize = output_shape[..3].iter().product();
            let tile_rows = rows.min(4096);
            let k = weights.input_channels * weights.kernel.iter().product::<usize>();
            let columns_buffer =
                CudaBuffer::alloc(tile_rows * k * 2, ctx.device_id()).map_err(Error::Cuda)?;
            let output_buffer =
                CudaBuffer::alloc(rows * weights.output_channels * 2, ctx.device_id())
                    .map_err(Error::Cuda)?;
            // Bind every full tile to one GEMM output address. Distinct final
            // destinations would otherwise create one vendor handle/workspace per
            // tile, even though execution is sequential and the weights are shared.
            let tile_buffer =
                CudaBuffer::alloc(tile_rows * weights.output_channels * 2, ctx.device_id())
                    .map_err(Error::Cuda)?;
            let output = output_buffer
                .as_tensor(Shape::new(output_shape), DType::BF16)
                .map_err(Error::Cuda)?;
            let mut tiles = Vec::new();
            for start in (0..rows).step_by(tile_rows) {
                let count = tile_rows.min(rows - start);
                let mut columns = columns_buffer
                    .view(0, count * k * 2)
                    .map_err(Error::Cuda)?
                    .as_tensor(Shape::new(vec![count, k]), DType::BF16)
                    .map_err(Error::Cuda)?;
                let destination = output_buffer
                    .view(
                        start * weights.output_channels * 2,
                        count * weights.output_channels * 2,
                    )
                    .map_err(Error::Cuda)?
                    .as_tensor(
                        Shape::new(vec![count, weights.output_channels]),
                        DType::BF16,
                    )
                    .map_err(Error::Cuda)?;
                let tile_output = tile_buffer
                    .view(0, count * weights.output_channels * 2)
                    .map_err(Error::Cuda)?
                    .as_tensor(
                        Shape::new(vec![count, weights.output_channels]),
                        DType::BF16,
                    )
                    .map_err(Error::Cuda)?;
                let unfold = PreparedUnfold3d::new(
                    ctx,
                    Unfold3dArgs {
                        input,
                        history,
                        row_start: start,
                        out: &mut columns,
                        geometry: Unfold3dGeometry {
                            kernel: weights.kernel,
                            stride,
                            padding_before,
                            padding_after,
                        },
                    },
                )?;
                tiles.push(Tile {
                    unfold,
                    columns,
                    destination,
                    output: tile_output,
                });
            }
            Ok(Self { tiles, output })
        }

        pub fn forward(
            &mut self,
            ctx: &CudaContext,
            weights: &Conv,
            policy: &GemmPolicy,
        ) -> Result<()> {
            for tile in &mut self.tiles {
                tile.unfold.enqueue()?;
                let mut args =
                    GemmArgs::new(&tile.columns, &weights.linear.weight, &mut tile.output);
                args.policy = policy.clone();
                args.weight_version = Some(WeightVersion::new(1));
                gemm(ctx, args)?;
                if let Some(bias) = &weights.linear.bias {
                    // Preserve the convolution-output BF16 rounding before bias.
                    let input = tile.output.clone();
                    let mut args = PointwiseArgs::new(
                        PointwiseSemantic::BiasActivation,
                        &input,
                        &mut tile.output,
                    );
                    args.bias = Some(bias);
                    pointwise(ctx, args)?;
                }
                crate::cuda_cosmos_policy::backend::transfers::copy_cuda_to_cuda(
                    ctx,
                    &tile.output,
                    &tile.destination,
                )?;
            }
            Ok(())
        }
    }
}
mod vae {
    //! Causal Wan2.1 encoder. Cache edges point at preceding chunk activations;
    //! every inference recomputes the singleton chunk before consuming those edges.
    //! No activation state crosses observation boundaries.
    use crate::cuda_cosmos_policy::backend::ops::{embedding_gather, GemmPolicy};
    use crate::cuda_cosmos_policy::backend::{
        attention, channel_l2_norm, pointwise, residual_add, AttentionArgs, AttentionPolicy,
        BiasOrder, ChannelL2NormArgs, CudaBuffer, CudaContext, PointwiseActivation, PointwiseArgs,
        PointwiseSemantic, ResidualAddArgs,
    };
    use apxinf_core::{DType, Error, Result, Shape, Tensor};

    use super::convolution::ConvWorkspace;
    use crate::cuda_cosmos_policy::weights::{upload, Conv, VaeResidual, VaeWeights};

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
        residual_add(
            ctx,
            ResidualAddArgs {
                input: &matrix(a)?,
                bias: None,
                residual: &matrix(b)?,
                hidden: &matrix(out)?,
                gate: None,
                bias_order: BiasOrder::SingleRound,
            },
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
                .map(|weight| {
                    ConvWorkspace::prepare(ctx, weight, input, None, [1; 3], [0; 3], [0; 3])
                })
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

        fn forward(
            &mut self,
            ctx: &CudaContext,
            w: &VaeResidual,
            policy: &GemmPolicy,
        ) -> Result<()> {
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
                let indices: Vec<i32> =
                    (0..t * pixels).map(|row| (row * 3 + part) as i32).collect();
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

    pub(in crate::cuda_cosmos_policy::model::blocks::bf16) struct VaeChunk {
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
                -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134,
                -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
            ];
            let std = [
                2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526,
                2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
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
        use crate::cuda_cosmos_policy::backend::ExecutionSession;
        use std::path::PathBuf;

        // External fixtures are captured from the pinned official checkpoint. Keep
        // their bytes outside Git; this test never downloads or fabricates a golden.
        #[test]
        #[ignore = "requires Thor, official converted weights and reference VAE fixture"]
        fn official_vae_conditioning_prefix() -> Result<()> {
            let root = PathBuf::from(
                std::env::var("APXINF_COSMOS_EVIDENCE").expect("APXINF_COSMOS_EVIDENCE"),
            );
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
            let graph = crate::cuda_cosmos_policy::backend::capture(&ctx, || {
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
}
mod dit {
    //! Complete joint-latent transformer. All 1,764 tokens pass through every block.
    //! Scratch is reused between blocks and denoising steps; it contains no runner
    //! or capture policy. Cached text is supplied by the family model.
    use super::tensor_ops::*;
    use crate::cuda_cosmos_policy::backend::ops::GemmPolicy;
    use crate::cuda_cosmos_policy::backend::{
        attention, norm, AttentionArgs, AttentionPolicy, CudaContext, NormArgs, PointwiseActivation,
    };
    use crate::cuda_cosmos_policy::weights::{upload, Attention, DitWeights, Modulation};
    use apxinf_core::{DType, Result, Tensor};

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

    pub(in crate::cuda_cosmos_policy::model::blocks::bf16) struct Dit {
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
            norm(
                ctx,
                NormArgs::layer(
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
            norm(
                ctx,
                NormArgs::rms(
                    &matrix(&self.q, 128)?,
                    &w.q_norm,
                    &mut matrix(&self.q, 128)?,
                    1e-6,
                ),
            )?;
            norm(
                ctx,
                NormArgs::rms(&matrix(k, 128)?, &w.k_norm, &mut matrix(k, 128)?, 1e-6),
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
            norm(
                ctx,
                NormArgs::rms(time, &w.time_norm, &mut self.time_normalized.clone(), 1e-6),
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
        use crate::cuda_cosmos_policy::backend::{CudaBuffer, ExecutionSession};
        use apxinf_core::Error;
        use std::path::PathBuf;
        #[test]
        #[ignore = "requires Thor, converted official weights and fixed reference DiT fixture"]
        fn official_full_joint_denoiser_layers() -> Result<()> {
            let root = PathBuf::from(
                std::env::var("APXINF_COSMOS_EVIDENCE").expect("APXINF_COSMOS_EVIDENCE"),
            );
            let ctx = CudaContext::new(0).map_err(Error::Cuda)?;
            let weights = DitWeights::load(&root.join("native-assets/model.safetensors"), 0)?;
            let (fixture, _) =
                apxinf_loader::safetensors::load_native_path(&root.join("dit-fixture.safetensors"))
                    .map_err(|e| Error::Other(e.to_string()))?;
            let ntk = 3f64.powf(42. / 40.) as f32;
            let dit = Dit::prepare(&ctx, &weights, [1., ntk, ntk])?;
            let patches = upload(&fixture["input"], 0)?;
            crate::cuda_cosmos_policy::backend::transfers::copy_cuda_to_cuda(
                &ctx,
                &patches,
                &dit.patches,
            )?;
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
            let graph = crate::cuda_cosmos_policy::backend::capture(&ctx, || {
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
}
use self::{dit::Dit, tensor_ops::*, vae::VaeChunk};
use crate::cuda_cosmos_policy::backend::ops::GemmPolicy;
use crate::cuda_cosmos_policy::backend::{
    AttentionPolicy, CudaBuffer, CudaContext, F64RecurrenceState, GatherArgs, GatherPatchGeometry,
    GatherSemantic, PreparedIndexedMean, PreparedRecurrence, RecurrenceCoefficients,
};
use crate::cuda_cosmos_policy::model::{Inputs, SolverStep};
use crate::cuda_cosmos_policy::weights::{upload, Bf16Weights};
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use std::rc::Rc;
const PIXELS: usize = 224 * 224;
const LATENT: usize = 16 * 9 * 28 * 28;
const FRAME: usize = 16 * 28 * 28;

fn slice(t: &Tensor, start: usize, len: usize, shape: Vec<usize>) -> Result<Tensor> {
    CudaBuffer::from_tensor(t)
        .map_err(Error::Cuda)?
        .view(
            start * t.dtype().size_in_bytes(),
            len * t.dtype().size_in_bytes(),
        )
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape), t.dtype())
        .map_err(Error::Cuda)
}
fn constant_f32(ctx: &CudaContext, values: &[f32]) -> Result<Tensor> {
    upload(
        &Tensor::from_f32(vec![1, values.len()], values)?,
        ctx.device_id(),
    )
}
fn copy(ctx: &CudaContext, a: &Tensor, b: &Tensor) -> Result<()> {
    crate::cuda_cosmos_policy::backend::transfers::copy_cuda_to_cuda(
        ctx,
        &a.reshape(vec![a.numel()])?,
        &b.reshape(vec![b.numel()])?,
    )
}

struct Step {
    input_scale: Tensor,
    output_scale: Tensor,
    time: Tensor,
    update: Option<PreparedRecurrence>,
}

/// Fixed LIBERO direct-policy computation and stable device storage. The runner
/// owns preparation/capture and request binding; this module only computes.
pub(in crate::cuda_cosmos_policy::model) struct Bf16Blocks {
    pub ctx: Rc<CudaContext>,
    weights: Rc<Bf16Weights>,
    pub rgb: CudaBuffer,
    pub proprio: Tensor,
    pub task: Tensor,
    pub noise: Tensor,
    pub actions: Tensor,
    pub joint: Tensor,
    images: Tensor,
    video: Tensor,
    first: VaeChunk,
    rest: VaeChunk,
    condition_source: Tensor,
    condition_map: Tensor,
    condition_bf16: Tensor,
    condition: Tensor,
    latent_input: Tensor,
    latent_view: Tensor,
    patch_map: Tensor,
    unpatch_map: Tensor,
    network_output: Tensor,
    text: Tensor,
    dit: Dit,
    state: Tensor,
    scratch: Tensor,
    scaled_noise: Tensor,
    eighty: Tensor,
    reset: PreparedRecurrence,
    steps: Vec<Step>,
    readout: PreparedIndexedMean,
}
impl Bf16Blocks {
    pub fn new(
        ctx: Rc<CudaContext>,
        weights: Rc<Bf16Weights>,
        inputs: &Inputs,
        schedule: &[SolverStep],
    ) -> Result<Self> {
        let rgb = inputs.rgb.clone();
        let images = bf(&ctx, 2 * PIXELS, 3)?;
        let video = upload(
            &Tensor::from_bf16(
                vec![17, 224, 224, 3],
                &vec![half::bf16::NEG_ONE; 17 * PIXELS * 3],
            )?,
            ctx.device_id(),
        )?;
        let first_input = slice(&video, 0, PIXELS * 3, vec![1, 224, 224, 3])?;
        let rest_input = slice(&video, PIXELS * 3, 16 * PIXELS * 3, vec![16, 224, 224, 3])?;
        let first = VaeChunk::prepare(&ctx, &weights.vae, &first_input, None)?;
        let rest = VaeChunk::prepare(&ctx, &weights.vae, &rest_input, Some(&first))?;
        let proprio = inputs.proprio.clone();
        let task = inputs.task.clone();
        let noise = inputs.noise.clone();
        // Last element is the zero used for every unconditioned slot.
        let condition_source = upload(
            &Tensor::from_bf16(
                vec![5 * FRAME + 10, 1],
                &vec![half::bf16::ZERO; 5 * FRAME + 10],
            )?,
            ctx.device_id(),
        )?;
        let mut cond_map = Vec::with_capacity(LATENT);
        for c in 0..16 {
            for t in 0..9 {
                for p in 0..784 {
                    cond_map.push(if t == 1 {
                        (5 * FRAME + (c * 784 + p) % 9) as i32
                    } else if t < 4 {
                        ((t * 784 + p) * 16 + c) as i32
                    } else {
                        (5 * FRAME + 9) as i32
                    });
                }
            }
        }
        let condition_map = indices(&ctx, &cond_map)?;
        let condition_bf16 = bf(&ctx, 1, LATENT)?;
        let condition = alloc(&ctx, vec![1, LATENT], DType::F32)?;
        let mut latent = vec![half::bf16::ZERO; 18 * 9 * 784];
        latent[16 * 9 * 784..16 * 9 * 784 + 4 * 784].fill(half::bf16::ONE);
        let latent_input = upload(
            &Tensor::from_bf16(vec![1, latent.len()], &latent)?,
            ctx.device_id(),
        )?;
        let latent_view = slice(&latent_input, 0, LATENT, vec![1, LATENT])?;
        let mut patch = Vec::with_capacity(1764 * 72);
        for t in 0..9 {
            for y in 0..14 {
                for x in 0..14 {
                    for c in 0..18 {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                patch.push(
                                    (c * 9 * 784 + t * 784 + (y * 2 + dy) * 28 + x * 2 + dx) as i32,
                                );
                            }
                        }
                    }
                }
            }
        }
        let patch_map = indices(&ctx, &patch)?;
        let mut unpatch = Vec::with_capacity(LATENT);
        for c in 0..16 {
            for t in 0..9 {
                for y in 0..28 {
                    for x in 0..28 {
                        unpatch.push(
                            ((t * 196 + y / 2 * 14 + x / 2) * 64 + (y % 2 * 2 + x % 2) * 16 + c)
                                as i32,
                        );
                    }
                }
            }
        }
        let unpatch_map = indices(&ctx, &unpatch)?;
        let network_output = bf(&ctx, 1, LATENT)?;
        let text = bf(&ctx, 512, 1024)?;
        // Pinned LIBERO runtime: spatial extrapolation 3, temporal 1.
        let spatial_ntk = 3f64.powf(42. / 40.) as f32;
        let dit = Dit::prepare(&ctx, &weights.dit, [1., spatial_ntk, spatial_ntk])?;
        let state = alloc(&ctx, vec![1, LATENT], DType::F32)?;
        let scratch = alloc(&ctx, vec![1, LATENT], DType::F32)?;
        let joint = alloc(&ctx, vec![1, LATENT], DType::F32)?;
        let scaled_noise = alloc(&ctx, vec![1, LATENT], DType::F32)?;
        let eighty = constant_f32(&ctx, &vec![80.; LATENT])?;
        let recurrence = F64RecurrenceState::new(&ctx, LATENT)?;
        let reset = recurrence.prepare_reset(&ctx, &scaled_noise, &state)?;
        let mut steps = Vec::new();
        for step in schedule {
            let sigma = step.sigma;
            let sigma = sigma as f32;
            let t = sigma / (sigma + 1.);
            let clean = 1. - t;
            let mut input_scale = Vec::with_capacity(LATENT);
            let mut output_scale = Vec::with_capacity(LATENT);
            for _ in 0..16 {
                for frame in 0..9 {
                    for _ in 0..784 {
                        input_scale.push(if frame < 4 { 0. } else { clean });
                        output_scale.push(if frame < 4 { 0. } else { -t });
                    }
                }
            }
            let mut time = Vec::with_capacity(9 * 2048);
            for frame in 0..9 {
                let time_value = if frame < 4 {
                    0.
                } else {
                    half::bf16::from_f32(t).to_f32()
                };
                let angles: Vec<_> = (0..1024)
                    .map(|d| time_value * (-(10000f32).ln() * (d as f32) / 1024.).exp())
                    .collect();
                time.extend(angles.iter().map(|a| half::bf16::from_f32(a.cos())));
                time.extend(angles.iter().map(|a| half::bf16::from_f32(a.sin())));
            }
            let update = step
                .recurrence
                .map(|[a, b, c, d]| {
                    recurrence.prepare_update(
                        &ctx,
                        &joint,
                        &state,
                        RecurrenceCoefficients { a, b, c, d },
                    )
                })
                .transpose()?;
            steps.push(Step {
                input_scale: constant_f32(&ctx, &input_scale)?,
                output_scale: constant_f32(&ctx, &output_scale)?,
                time: upload(&Tensor::from_bf16(vec![9, 2048], &time)?, ctx.device_id())?,
                update,
            });
        }
        let actions = alloc(&ctx, vec![16, 7], DType::F32)?;
        let action_map: Vec<u32> = (0..FRAME)
            .map(|i| ((i / 784) * 9 * 784 + 4 * 784 + i % 784) as u32)
            .collect();
        let readout = PreparedIndexedMean::new(&ctx, &joint, &actions, &action_map)?;
        Ok(Self {
            ctx,
            weights,
            rgb,
            proprio,
            task,
            noise,
            actions,
            joint,
            images,
            video,
            first,
            rest,
            condition_source,
            condition_map,
            condition_bf16,
            condition,
            latent_input,
            latent_view,
            patch_map,
            unpatch_map,
            network_output,
            text,
            dit,
            state,
            scratch,
            scaled_noise,
            eighty,
            reset,
            steps,
            readout,
        })
    }
    pub fn encode_observation(
        &mut self,
        gemm: &GemmPolicy,
        attention: &AttentionPolicy,
    ) -> Result<()> {
        let ctx = &self.ctx;
        let mut images = self.images.clone();
        let mut args = GatherArgs::rgb_to_patches(
            &self.rgb,
            &mut images,
            GatherPatchGeometry {
                views: 2,
                image_size: 224,
                patch_size: 1,
                nhwc: true,
            },
        );
        args.semantic = GatherSemantic::RgbToPatchesStaged;
        crate::cuda_cosmos_policy::backend::gather(ctx, args)?;
        for (view, start) in [(0, 5), (1, 9)] {
            let source = slice(&self.images, view * PIXELS * 3, PIXELS * 3, vec![PIXELS, 3])?;
            for frame in start..start + 4 {
                copy(
                    ctx,
                    &source,
                    &slice(&self.video, frame * PIXELS * 3, PIXELS * 3, vec![PIXELS, 3])?,
                )?;
            }
        }
        self.first
            .forward(ctx, &self.weights.vae, gemm, attention)?;
        self.rest.forward(ctx, &self.weights.vae, gemm, attention)?;
        copy(
            ctx,
            &self.first.output,
            &slice(&self.condition_source, 0, FRAME, vec![FRAME, 1])?,
        )?;
        copy(
            ctx,
            &self.rest.output,
            &slice(&self.condition_source, FRAME, 4 * FRAME, vec![4 * FRAME, 1])?,
        )?;
        cast(
            ctx,
            &self.proprio,
            &slice(&self.condition_source, 5 * FRAME, 9, vec![1, 9])?,
        )?;
        gather(
            ctx,
            &self.condition_source,
            &self.condition_map,
            1,
            &self.condition_bf16,
        )?;
        cast(ctx, &self.condition_bf16, &self.condition)?;
        gather(ctx, &self.weights.text, &self.task, 512 * 1024, &self.text)?;
        Ok(())
    }
    pub fn initial_state(&self) -> Result<()> {
        let ctx = &self.ctx;
        multiply(ctx, &self.noise, &self.eighty, &self.scaled_noise)?;
        self.reset.enqueue()?;
        Ok(())
    }
    pub fn denoise_step(
        &self,
        index: usize,
        gemm: &GemmPolicy,
        attention: &AttentionPolicy,
    ) -> Result<()> {
        let ctx = &self.ctx;
        let step = &self.steps[index];
        multiply(ctx, &self.state, &step.input_scale, &self.scratch)?;
        add(ctx, &self.scratch, &self.condition, &self.scratch)?;
        cast(ctx, &self.scratch, &self.latent_view)?;
        gather(
            ctx,
            &self.latent_input,
            &self.patch_map,
            1,
            &self.dit.patches,
        )?;
        self.dit.forward(
            ctx,
            &self.weights.dit,
            &step.time,
            &self.text,
            gemm,
            attention,
        )?;
        gather(
            ctx,
            &self.dit.output,
            &self.unpatch_map,
            1,
            &self.network_output,
        )?;
        cast(ctx, &self.network_output, &self.scratch)?;
        multiply(ctx, &self.scratch, &step.output_scale, &self.scratch)?;
        multiply(ctx, &self.state, &step.input_scale, &self.joint)?;
        add(ctx, &self.joint, &self.scratch, &self.joint)?;
        add(ctx, &self.joint, &self.condition, &self.joint)?;
        if let Some(update) = &step.update {
            update.enqueue()?;
        }
        Ok(())
    }
    pub fn finish(&self) -> Result<()> {
        self.readout.enqueue()
    }
    pub fn actions(&self) -> &Tensor {
        &self.actions
    }
    #[cfg(test)]
    pub fn joint(&self) -> &Tensor {
        &self.joint
    }
    pub fn workspace_bytes(&self) -> usize {
        256 * 1024 * 1024
    }
}
