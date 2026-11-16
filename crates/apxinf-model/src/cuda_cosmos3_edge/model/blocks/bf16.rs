//! BF16 computation and stable intermediate storage, behind Bf16Blocks.
mod tensor_ops {
    //! Family composition helpers; device ownership and kernels stay in cuda-new.
    use crate::cuda_cosmos3_edge::backend::ops::{
        embedding_gather, gemm, gemm_relu_squared, GemmArgs, GemmPolicy, WeightVersion,
    };
    use crate::cuda_cosmos3_edge::backend::{
        pointwise, quantization, CudaContext, PointwiseActivation, PointwiseArgs,
        PointwiseSemantic, QuantizationArgs, QuantizationSemantic,
    };
    use crate::cuda_cosmos3_edge::weights::{upload, Linear};
    use apxinf_core::{DType, Error, Result, Shape, Tensor};

    pub fn alloc(ctx: &CudaContext, shape: impl Into<Shape>, dtype: DType) -> Result<Tensor> {
        // Session-aware allocation: routes through the graph workspace arena so
        // the fixed-shape forward is CUDA Graph-capturable. Direct `cudaMalloc`
        // here would invalidate capture and add a synchronous malloc/free pair
        // for every scratch tensor on the eager path.
        ctx.allocate_output(shape.into(), dtype)
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
    /// Up-projection with the MoT activation fused into the GEMM epilogue.
    /// The fused kernel rounds the accumulator to BF16 before `relu^2`, so it
    /// reproduces `linear` followed by `relu2` exactly; the weights carry no
    /// bias (the contract rejects one).
    pub fn linear_relu_squared(
        ctx: &CudaContext,
        input: &Tensor,
        w: &Linear,
        output: &Tensor,
        policy: &GemmPolicy,
    ) -> Result<()> {
        debug_assert!(w.bias.is_none(), "fused relu2 GEMM does not admit a bias");
        let mut output = output.clone();
        let mut args = GemmArgs::new(input, &w.weight, &mut output);
        args.policy = policy.clone();
        args.weight_version = Some(WeightVersion::new(1));
        gemm_relu_squared(ctx, args)
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
    pub fn scale_add(
        ctx: &CudaContext,
        a: &Tensor,
        dt: f32,
        b: &Tensor,
        out: &Tensor,
    ) -> Result<()> {
        let mut out = out.clone();
        let mut args = PointwiseArgs::new(PointwiseSemantic::EulerUpdate, a, &mut out);
        args.secondary = Some(b);
        args.dt = dt;
        pointwise(ctx, args)
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
    use crate::cuda_cosmos3_edge::backend::{
        fused_add_channel_l2_norm_silu, pointwise, residual_add, BiasOrder, Conv3dArgs,
        Conv3dPolicy, CudaBuffer, CudaContext, FusedAddChannelL2NormArgs, PointwiseArgs,
        PointwiseSemantic, PreparedConv3d, ResidualAddArgs, Unfold3dGeometry,
    };
    use apxinf_core::{DType, Error, Result, Shape, Tensor};

    use crate::cuda_cosmos3_edge::weights::Conv;

    /// Per-convolution scratch budget. Direct cuDNN convolution must stage a
    /// padded copy of its input when padding is asymmetric (the stride-2 spatial
    /// downsample) or history is separate (temporal downsample); at 16x272x368x160
    /// that copy is 516 MiB. Below the budget the convolution falls back to
    /// 4096-row im2col + GEMM tiles, which measured ~10x slower on thor3. Causal
    /// 3x3x3 convolutions read a zero-led timeline instead and need no copy, so
    /// only the downsample convolutions retain large scratch (~0.9 GiB total).
    const CONV3D_WORKSPACE_LIMIT: usize = 768 * 1024 * 1024;

    /// One prepared BF16 convolution and its stable output. Execution, packed
    /// weights and workspace are bound at preparation, so enqueue is capturable.
    pub(in crate::cuda_cosmos3_edge::model::blocks::bf16) struct ConvWorkspace {
        convolution: PreparedConv3d,
        pub output: Tensor,
    }

    impl ConvWorkspace {
        pub fn prepare(
            policy: &Conv3dPolicy,
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
            let bytes = output_shape.iter().product::<usize>() * 2;
            let mut output = CudaBuffer::alloc(bytes, ctx.device_id())
                .map_err(Error::Cuda)?
                .as_tensor(Shape::new(output_shape), DType::BF16)
                .map_err(Error::Cuda)?;
            let convolution = PreparedConv3d::with_policy(
                ctx,
                Conv3dArgs {
                    input,
                    history,
                    weight: &weights.linear.weight,
                    geometry: Unfold3dGeometry {
                        kernel: weights.kernel,
                        stride,
                        padding_before,
                        padding_after,
                    },
                    out: &mut output,
                    workspace_limit: CONV3D_WORKSPACE_LIMIT,
                },
                policy,
            )?;
            Ok(Self {
                convolution,
                output,
            })
        }

        pub fn forward(&mut self, ctx: &CudaContext, weights: &Conv) -> Result<()> {
            self.convolution.enqueue()?;
            if let Some(bias) = &weights.linear.bias {
                // Preserve the convolution-output BF16 rounding before bias.
                let input = self.output.reshape(vec![
                    self.output.numel() / weights.output_channels,
                    weights.output_channels,
                ])?;
                let mut output = input.clone();
                let mut args =
                    PointwiseArgs::new(PointwiseSemantic::BiasActivation, &input, &mut output);
                args.bias = Some(bias);
                pointwise(ctx, args)?;
            }
            Ok(())
        }

        fn matrix_shape(&self, weights: &Conv) -> Vec<usize> {
            vec![
                self.output.numel() / weights.output_channels,
                weights.output_channels,
            ]
        }

        /// Bias the convolution output in place and write its channel norm + SiLU
        /// into `normalized`, in one pass.
        pub fn forward_normalized(
            &mut self,
            ctx: &CudaContext,
            weights: &Conv,
            norm_weight: &Tensor,
            normalized: &Tensor,
        ) -> Result<()> {
            self.convolution.enqueue()?;
            let shape = self.matrix_shape(weights);
            let input = self.output.reshape(shape.clone())?;
            let mut normalized = normalized.reshape(shape)?;
            fused_add_channel_l2_norm_silu(
                ctx,
                FusedAddChannelL2NormArgs {
                    input: &input,
                    bias: weights.linear.bias.as_ref(),
                    residual: None,
                    hidden: &input,
                    weight: norm_weight,
                    normalized: &mut normalized,
                    eps: 1e-12,
                },
            )
        }

        /// `output = bf16(bf16(conv + bias) + residual)`, plus its channel norm +
        /// SiLU into `normalized` for the consumer, in one pass.
        pub fn forward_residual_normalized(
            &mut self,
            ctx: &CudaContext,
            weights: &Conv,
            residual: &Tensor,
            output: &Tensor,
            norm_weight: &Tensor,
            normalized: &Tensor,
        ) -> Result<()> {
            self.convolution.enqueue()?;
            let shape = self.matrix_shape(weights);
            let mut normalized = normalized.reshape(shape.clone())?;
            fused_add_channel_l2_norm_silu(
                ctx,
                FusedAddChannelL2NormArgs {
                    input: &self.output.reshape(shape.clone())?,
                    bias: weights.linear.bias.as_ref(),
                    residual: Some(&residual.reshape(shape.clone())?),
                    hidden: &output.reshape(shape)?,
                    weight: norm_weight,
                    normalized: &mut normalized,
                    eps: 1e-12,
                },
            )
        }

        /// `output = bf16(bf16(conv + bias) + residual)` without materializing
        /// the biased convolution result.
        pub fn forward_residual(
            &mut self,
            ctx: &CudaContext,
            weights: &Conv,
            residual: &Tensor,
            output: &Tensor,
        ) -> Result<()> {
            self.convolution.enqueue()?;
            let shape = self.matrix_shape(weights);
            residual_add(
                ctx,
                ResidualAddArgs {
                    input: &self.output.reshape(shape.clone())?,
                    bias: weights.linear.bias.as_ref(),
                    residual: &residual.reshape(shape.clone())?,
                    hidden: &output.reshape(shape)?,
                    gate: None,
                    bias_order: BiasOrder::BiasThenResidual,
                },
            )
        }
    }
}
mod vae {
    //! Causal Wan2.1 encoder. Cache edges point at preceding chunk activations;
    //! every inference recomputes the singleton chunk before consuming those edges.
    //! No activation state crosses observation boundaries.
    use crate::cuda_cosmos3_edge::backend::ops::embedding_gather;
    use crate::cuda_cosmos3_edge::backend::{
        attention, channel_l2_norm_silu, fused_add_channel_l2_norm_silu, norm, residual_add,
        AttentionArgs, AttentionPolicy, BiasOrder, ChannelL2NormArgs, Conv3dPolicy, CudaBuffer,
        CudaContext, FusedAddChannelL2NormArgs, NormArgs, ResidualAddArgs,
    };
    use apxinf_core::{DType, Error, Result, Shape, Tensor};

    use super::convolution::ConvWorkspace;
    use crate::cuda_cosmos3_edge::weights::{upload, Conv, VaeResidual, VaeWeights};

    fn alloc(ctx: &CudaContext, shape: Vec<usize>) -> Result<Tensor> {
        let bytes = shape.iter().product::<usize>() * 2;
        CudaBuffer::alloc(bytes, ctx.device_id())
            .map_err(Error::Cuda)?
            .as_tensor(Shape::new(shape), DType::BF16)
            .map_err(Error::Cuda)
    }

    /// One chronological activation shared by the singleton and the 16-frame
    /// chunk, led by two zero frames. A causal 3x3x3 convolution reads
    /// `causal` (two frames of history or zero padding, then `current`) with
    /// symmetric spatial padding, so cuDNN runs it directly without staging a
    /// padded copy. Producers write only `current`; the zero lead never changes.
    pub(in crate::cuda_cosmos3_edge::model::blocks::bf16) struct CausalActivation {
        storage: CudaBuffer,
        pub current: Tensor,
        causal: Tensor,
        end: usize,
        capacity: usize,
        frame_bytes: usize,
    }

    impl CausalActivation {
        pub fn prepare(
            ctx: &CudaContext,
            shape: &[usize],
            total_frames: usize,
            previous: Option<&Self>,
        ) -> Result<Self> {
            if shape.len() != 4 || shape.contains(&0) || total_frames < shape[0] {
                return Err(Error::Other("invalid causal activation extent".into()));
            }
            let capacity = total_frames + 2;
            let frame_bytes = shape[1..].iter().product::<usize>() * 2;
            let (storage, start) = match previous {
                Some(previous) => {
                    if previous.capacity != capacity
                        || previous.current.shape().dims()[1..] != shape[1..]
                    {
                        return Err(Error::Other(
                            "causal activation history geometry mismatch".into(),
                        ));
                    }
                    (previous.storage.clone(), previous.end)
                }
                None => (
                    CudaBuffer::alloc_zeros(capacity * frame_bytes, ctx.device_id())
                        .map_err(Error::Cuda)?,
                    2,
                ),
            };
            let end = start + shape[0];
            if end > capacity {
                return Err(Error::Other(
                    "causal activation exceeds prepared timeline".into(),
                ));
            }
            let view = |start: usize, frames: usize| -> Result<Tensor> {
                let mut dims = shape.to_vec();
                dims[0] = frames;
                storage
                    .view(start * frame_bytes, frames * frame_bytes)
                    .map_err(Error::Cuda)?
                    .as_tensor(Shape::new(dims), DType::BF16)
                    .map_err(Error::Cuda)
            };
            let current = view(start, shape[0])?;
            let causal = view(start - 2, shape[0] + 2)?;
            Ok(Self {
                storage,
                current,
                causal,
                end,
                capacity,
                frame_bytes,
            })
        }

        /// Every produced frame of the timeline, after the zero lead.
        pub fn frames(&self) -> Result<Tensor> {
            let mut dims = self.current.shape().dims().to_vec();
            dims[0] = self.capacity - 2;
            self.storage
                .view(2 * self.frame_bytes, dims[0] * self.frame_bytes)
                .map_err(Error::Cuda)?
                .as_tensor(Shape::new(dims), DType::BF16)
                .map_err(Error::Cuda)
        }
    }

    fn causal_activation(
        policy: &Conv3dPolicy,
        ctx: &CudaContext,
        weights: &Conv,
        input: &CausalActivation,
    ) -> Result<ConvWorkspace> {
        ConvWorkspace::prepare(
            policy,
            ctx,
            weights,
            &input.causal,
            None,
            [1; 3],
            [0, 1, 1],
            [0, 1, 1],
        )
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

    fn normalize_activate(
        ctx: &CudaContext,
        input: &Tensor,
        weight: &Tensor,
        output: &Tensor,
    ) -> Result<()> {
        // Wan RMS_norm (channel L2 norm * sqrt(C) * gamma), then SiLU of the
        // BF16-rounded result, in one pass.
        channel_l2_norm_silu(
            ctx,
            ChannelL2NormArgs {
                input: &matrix(input)?,
                weight,
                out: &mut matrix(output)?,
                eps: 1e-12,
            },
        )
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
        norm1: CausalActivation,
        conv1: ConvWorkspace,
        norm2: CausalActivation,
        conv2: ConvWorkspace,
        shortcut: Option<ConvWorkspace>,
        output: Tensor,
    }

    impl Residual {
        fn prepare(
            policy: &Conv3dPolicy,
            ctx: &CudaContext,
            w: &VaeResidual,
            input: &Tensor,
            total_frames: usize,
            previous: Option<&Self>,
        ) -> Result<Self> {
            let norm1 = CausalActivation::prepare(
                ctx,
                input.shape().dims(),
                total_frames,
                previous.map(|p| &p.norm1),
            )?;
            let conv1 = causal_activation(policy, ctx, &w.conv1, &norm1)?;
            let norm2 = CausalActivation::prepare(
                ctx,
                conv1.output.shape().dims(),
                total_frames,
                previous.map(|p| &p.norm2),
            )?;
            let conv2 = causal_activation(policy, ctx, &w.conv2, &norm2)?;
            let shortcut = w
                .shortcut
                .as_ref()
                .map(|weight| {
                    ConvWorkspace::prepare(policy, ctx, weight, input, None, [1; 3], [0; 3], [0; 3])
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

        fn forward(&mut self, ctx: &CudaContext, w: &VaeResidual) -> Result<()> {
            normalize_activate(ctx, &self.input, &w.norm1, &self.norm1.current)?;
            self.forward_normalized(ctx, w, None)
        }

        /// The producer has already written norm1. With `next`, the residual
        /// update also writes the following block's normalization `(weight, out)`.
        fn forward_normalized(
            &mut self,
            ctx: &CudaContext,
            w: &VaeResidual,
            next: Option<(&Tensor, &Tensor)>,
        ) -> Result<()> {
            self.conv1
                .forward_normalized(ctx, &w.conv1, &w.norm2, &self.norm2.current)?;
            let residual = match (&mut self.shortcut, &w.shortcut) {
                (Some(shortcut), Some(weights)) => {
                    shortcut.forward(ctx, weights)?;
                    &shortcut.output
                }
                _ => &self.input,
            };
            match next {
                Some((weight, normalized)) => self.conv2.forward_residual_normalized(
                    ctx,
                    &w.conv2,
                    residual,
                    &self.output,
                    weight,
                    normalized,
                ),
                None => self
                    .conv2
                    .forward_residual(ctx, &w.conv2, residual, &self.output),
            }
        }
    }

    struct Downsample {
        spatial: ConvWorkspace,
        temporal: Option<ConvWorkspace>,
        avg_shortcut: ConvWorkspace,
        output: Tensor,
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
        fn prepare(
            policy: &Conv3dPolicy,
            ctx: &CudaContext,
            w: &VaeWeights,
            input: &Tensor,
        ) -> Result<Self> {
            let dims = input.shape().dims();
            let (t, pixels, c) = (dims[0], dims[1] * dims[2], dims[3]);
            let normalized = alloc(ctx, dims.to_vec())?;
            let qkv = ConvWorkspace::prepare(
                policy,
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
                policy,
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
            attention_policy: &AttentionPolicy,
        ) -> Result<()> {
            norm(
                ctx,
                NormArgs::rms(
                    &matrix(&self.input)?,
                    &w.attention_norm,
                    &matrix(&self.normalized)?,
                    1e-12,
                ),
            )?;
            self.qkv.forward(ctx, &w.attention_qkv)?;
            let c = *self.q.shape().dims().last().unwrap_or(&384);
            let rows = self.q.numel() / c;
            let table = self.qkv.output.reshape(vec![rows * 3, c])?;
            for (ids, output) in self.ids.iter().zip([&self.q, &self.k, &self.v]) {
                embedding_gather(ctx, &table, ids, &output.reshape(vec![rows, c])?)?;
            }
            let mut output = self.attended.reshape(self.q.shape().clone())?;
            let mut args = AttentionArgs::new(&self.q, &self.k, &self.v, &mut output);
            args.policy = attention_policy.clone();
            attention(ctx, args)?;
            self.projection.forward(ctx, &w.attention_output)?;
            add(ctx, &self.projection.output, &self.input, &self.output)
        }
    }

    pub(in crate::cuda_cosmos3_edge::model::blocks::bf16) struct VaeChunk {
        input_conv: ConvWorkspace,
        blocks: Vec<[Residual; 2]>,
        downsample: Vec<Downsample>,
        /// The last down block has no resampling, but its residual shortcut
        /// still applies: `output = blocks[3][1].output + blocks[3][0].input`.
        tail_output: Tensor,
        middle_before: Residual,
        attention: SpatialAttention,
        middle_after: Residual,
        head_norm: CausalActivation,
        head_conv: ConvWorkspace,
        mean_projection: ConvWorkspace,
        pub output: Tensor,
    }

    impl VaeChunk {
        pub fn prepare(
            policy: &Conv3dPolicy,
            ctx: &CudaContext,
            w: &VaeWeights,
            input: &CausalActivation,
            total_frames: usize,
            previous: Option<&Self>,
        ) -> Result<Self> {
            let input_conv = causal_activation(policy, ctx, &w.input, input)?;
            let mut current = input_conv.output.clone();
            let mut total_frames = total_frames;
            let mut blocks = Vec::new();
            let mut downsample = Vec::new();
            let mut tail_output = None;
            for (level, weights) in w.down_blocks.iter().enumerate() {
                let block_input = current.clone();
                let first = Residual::prepare(
                    policy,
                    ctx,
                    &weights[0],
                    &current,
                    total_frames,
                    previous.map(|p| &p.blocks[level][0]),
                )?;
                let second = Residual::prepare(
                    policy,
                    ctx,
                    &weights[1],
                    &first.output,
                    total_frames,
                    previous.map(|p| &p.blocks[level][1]),
                )?;
                current = second.output.clone();
                blocks.push([first, second]);
                if level < 3 {
                    let spatial = ConvWorkspace::prepare(
                        policy,
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
                                policy,
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
                    let (ft, fs) = if level == 0 {
                        (1usize, 2usize)
                    } else {
                        (2usize, 2usize)
                    };
                    let t = block_input.shape().dims()[0];
                    let pad_t = (ft - t % ft) % ft;
                    let avg_shortcut = ConvWorkspace::prepare(
                        policy,
                        ctx,
                        &w.avg_shortcut[level],
                        &block_input,
                        None,
                        [ft, fs, fs],
                        [pad_t, 0, 0],
                        [0, 0, 0],
                    )?;
                    let output = alloc(ctx, avg_shortcut.output.shape().dims().to_vec())?;
                    current = output.clone();
                    downsample.push(Downsample {
                        spatial,
                        temporal,
                        avg_shortcut,
                        output,
                    });
                    if w.downsample[level].temporal.is_some() {
                        total_frames = (total_frames - 1) / 2 + 1;
                    }
                } else {
                    let output = alloc(ctx, current.shape().dims().to_vec())?;
                    current = output.clone();
                    tail_output = Some(output);
                }
            }
            let tail_output = tail_output
                .ok_or_else(|| Error::Other("Cosmos VAE needs a final down block".into()))?;
            let middle_before = Residual::prepare(
                policy,
                ctx,
                &w.middle_before,
                &current,
                total_frames,
                previous.map(|p| &p.middle_before),
            )?;
            let attention = SpatialAttention::prepare(policy, ctx, w, &middle_before.output)?;
            let middle_after = Residual::prepare(
                policy,
                ctx,
                &w.middle_after,
                &attention.output,
                total_frames,
                previous.map(|p| &p.middle_after),
            )?;
            let head_norm = CausalActivation::prepare(
                ctx,
                middle_after.output.shape().dims(),
                total_frames,
                previous.map(|p| &p.head_norm),
            )?;
            let head_conv = causal_activation(policy, ctx, &w.output, &head_norm)?;
            let mean_projection = ConvWorkspace::prepare(
                policy,
                ctx,
                &w.mean_projection,
                &head_conv.output,
                None,
                [1; 3],
                [0; 3],
                [0; 3],
            )?;
            let output = mean_projection.output.clone();
            Ok(Self {
                input_conv,
                blocks,
                downsample,
                tail_output,
                middle_before,
                attention,
                middle_after,
                head_norm,
                head_conv,
                mean_projection,
                output,
            })
        }

        pub fn forward(
            &mut self,
            ctx: &CudaContext,
            w: &VaeWeights,
            attention: &AttentionPolicy,
        ) -> Result<()> {
            // Every producer also writes its consumer's channel norm + SiLU, so
            // each large activation is read once by the normalization.
            self.input_conv.forward_normalized(
                ctx,
                &w.input,
                &w.down_blocks[0][0].norm1,
                &self.blocks[0][0].norm1.current,
            )?;
            for level in 0..4 {
                let (first, second) = self.blocks[level].split_at_mut(1);
                first[0].forward_normalized(
                    ctx,
                    &w.down_blocks[level][0],
                    Some((&w.down_blocks[level][1].norm1, &second[0].norm1.current)),
                )?;
                if level < 3 {
                    second[0].forward_normalized(ctx, &w.down_blocks[level][1], None)?;
                    let down = &mut self.downsample[level];
                    down.avg_shortcut.forward(ctx, &w.avg_shortcut[level])?;
                    let next = (
                        &w.down_blocks[level + 1][0].norm1,
                        &self.blocks[level + 1][0].norm1.current,
                    );
                    let shortcut = &down.avg_shortcut.output;
                    if let (Some(temporal), Some(weights)) =
                        (&mut down.temporal, &w.downsample[level].temporal)
                    {
                        down.spatial.forward(ctx, &w.downsample[level].spatial)?;
                        temporal.forward_residual_normalized(
                            ctx,
                            weights,
                            shortcut,
                            &down.output,
                            next.0,
                            next.1,
                        )?;
                    } else {
                        down.spatial.forward_residual_normalized(
                            ctx,
                            &w.downsample[level].spatial,
                            shortcut,
                            &down.output,
                            next.0,
                            next.1,
                        )?;
                    }
                } else {
                    second[0].forward_normalized(ctx, &w.down_blocks[level][1], None)?;
                    let shape = vec![self.tail_output.numel() / 640, 640];
                    fused_add_channel_l2_norm_silu(
                        ctx,
                        FusedAddChannelL2NormArgs {
                            input: &second[0].output.reshape(shape.clone())?,
                            bias: None,
                            residual: Some(&first[0].input.reshape(shape.clone())?),
                            hidden: &self.tail_output.reshape(shape.clone())?,
                            weight: &w.middle_before.norm1,
                            normalized: &mut self.middle_before.norm1.current.reshape(shape)?,
                            eps: 1e-12,
                        },
                    )?;
                }
            }
            self.middle_before
                .forward_normalized(ctx, &w.middle_before, None)?;
            self.attention.forward(ctx, w, attention)?;
            self.middle_after.forward(ctx, &w.middle_after)?;
            normalize_activate(
                ctx,
                &self.middle_after.output,
                &w.output_norm,
                &self.head_norm.current,
            )?;
            self.head_conv.forward(ctx, &w.output)?;
            self.mean_projection.forward(ctx, &w.mean_projection)?;
            Ok(())
        }
    }
}
mod transformer {
    use super::tensor_ops::activate;
    use crate::cuda_cosmos3_edge::backend::ops::{
        rms_norm_apply_precomputed, rope_apply_fractional_mrope_into as apply_mrope_f32_into,
    };
    use crate::cuda_cosmos3_edge::backend::{
        fused_add_norm, norm, Affine, BiasOrder, CudaBuffer, CudaContext, FusedAddNormArgs,
        NormArgs, NormKind, NormNumerics, PointwiseActivation,
    };
    use apxinf_core::{Error, Result, Tensor};

    pub const W: usize = 2048;
    pub const HEADS: usize = 16;
    pub const KV: usize = 8;
    pub const D: usize = 128;
    pub const EPS: f32 = 1e-5;
    pub const THETA: f32 = 1e8;
    pub const VOCAB: usize = 131072;
    pub const MLP: usize = 9216;

    pub fn make_pos_ids(len: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(len * 3);
        for i in 0..len {
            out.extend([i as f32, i as f32, i as f32]);
        }
        out
    }

    /// GEN-stream mRoPE ids: 5 latent frames of 17x23 patches, then the action
    /// tokens, temporally fps-modulated after the UND prompt and modality margin.
    pub fn gen_pos_ids(s_und: usize, s_gen: usize) -> Vec<f32> {
        let media = s_und as f64 + 15000.0;
        let (hp, wp) = (17usize, 23usize);
        let hw = hp * wp;
        let mut v = Vec::with_capacity(s_gen * 3);
        for tok in 0..s_gen {
            let (t, h, w) = if tok < hw * 5 {
                let ti = tok / hw;
                let rem = tok % hw;
                let hi = rem / wp;
                let wi = rem % wp;
                (media + ti as f64 / 1.25 * 6.0, hi as f64, wi as f64)
            } else {
                let ai = (tok - hw * 5) as f64;
                (media + (ai + 1.0) / 5.0 * 6.0, 0.0, 0.0)
            };
            v.push(t as f32);
            v.push(h as f32);
            v.push(w as f32);
        }
        v
    }

    /// Per-token mRoPE cos/sin table `[seq, 2, D/2]` (cosine block, then sine)
    /// for the precomputed norm+rotation kernel. Mirrors the interleaved axis
    /// assignment and FP32 angle arithmetic of the on-device mRoPE kernel.
    pub fn mrope_table(ids: &[f32]) -> Vec<f32> {
        let (sec_h, sec_w) = (20usize, 20usize);
        let half = D / 2;
        let mut out = Vec::with_capacity(ids.len() / 3 * D);
        for pos in ids.chunks_exact(3) {
            let angle = |p: usize| {
                let axis = match p % 3 {
                    1 if p < sec_h * 3 => 1,
                    2 if p < sec_w * 3 => 2,
                    _ => 0,
                };
                let freq = 1.0f32 / THETA.powf(2.0 * p as f32 / D as f32);
                pos[axis] * freq
            };
            out.extend((0..half).map(|p| angle(p).cos()));
            out.extend((0..half).map(|p| angle(p).sin()));
        }
        out
    }

    /// Per-head RMSNorm then mRoPE in one pass, `x [seq, heads, D]` into `out`
    /// (exact in-place allowed, or a view such as the joint K tail).
    pub(super) fn norm_mrope(
        ctx: &CudaContext,
        x: &Tensor,
        weight: &Tensor,
        table: &Tensor,
        out: &Tensor,
    ) -> Result<()> {
        rms_norm_apply_precomputed(ctx, x, weight, table, EPS, out)
    }

    /// `hidden = x + delta` in place, then `normalized = rms(hidden) * weight`.
    pub(super) fn add_rms(
        ctx: &CudaContext,
        x: &Tensor,
        delta: &Tensor,
        weight: &Tensor,
        normalized: &Tensor,
    ) -> Result<()> {
        fused_add_norm(
            ctx,
            FusedAddNormArgs {
                input: delta,
                bias: None,
                residual: x,
                gate: None,
                hidden: x,
                normalized: Some(normalized),
                kind: NormKind::Rms,
                affine: Affine::Weight {
                    weight,
                    bias: None,
                    offset: false,
                },
                quantized: None,
                numerics: NormNumerics::Flexible,
                bias_order: BiasOrder::SingleRound,
                eps: EPS,
            },
        )
    }

    /// mRoPE over `[seq, heads, D]` into `out` (may be a view), reading device
    /// positions uploaded once at construction so the forward stays capturable.
    pub(super) fn mrope(
        ctx: &CudaContext,
        x: &Tensor,
        positions: &CudaBuffer,
        out: &Tensor,
    ) -> Result<()> {
        let heads = x.shape().dims()[1];
        apply_mrope_f32_into(ctx, x, out, heads, D, THETA, [24, 20, 20], positions)
    }

    pub(super) fn rms(ctx: &CudaContext, x: &Tensor, weight: &Tensor, out: &Tensor) -> Result<()> {
        norm(ctx, NormArgs::rms(x, weight, out, EPS)).map(|_| ())
    }

    /// In-place `relu(x)^2` in one pass over the MLP activation.
    pub(super) fn relu2(ctx: &CudaContext, up: &Tensor) -> Result<()> {
        activate(ctx, up, None, PointwiseActivation::ReluSquared, up)
    }

    /// One UND tower layer; emits the per-layer K/V for the GEN cross-attention.

    pub fn slice(t: &Tensor, start: usize, len: usize, shape: Vec<usize>) -> Result<Tensor> {
        let elem = t.dtype().size_in_bytes();
        crate::cuda_cosmos3_edge::backend::CudaBuffer::from_tensor(t)
            .map_err(Error::Cuda)?
            .view(start * elem, len * elem)
            .map_err(Error::Cuda)?
            .as_tensor(apxinf_core::Shape::new(shape), t.dtype())
            .map_err(Error::Cuda)
    }
}

use crate::cuda_cosmos3_edge::backend::ops::GemmPolicy;
use crate::cuda_cosmos3_edge::backend::{AttentionPolicy, CudaBuffer, CudaContext};
use crate::cuda_cosmos3_edge::weights::Bf16Weights;
use apxinf_core::Tensor;
use apxinf_core::{DType, Result};
use std::rc::Rc;

const S_VID: usize = 5 * 17 * 23; // 1955
const S_ACT: usize = 16;
const S_GEN: usize = S_VID + S_ACT; // 1971
const LAT: usize = 5 * 34 * 46 * 48; // 375360
const LAT_A: usize = 16 * 64; // 1024 packed action dims
/// Per-channel Wan2.2 VAE latent statistics (`vae/config.json`): the encoder's
/// posterior mean is normalized as `(x - mean[c]) / std[c]` before conditioning.
const VAE_MEAN: [f32; 48] = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557, -0.1382, 0.0542, 0.2813,
    0.0891, 0.157, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108,
    -0.2158, 0.2502, -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.123, -0.0313,
    -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.052, 0.3748, 0.0152, 0.1957, 0.1433, -0.2944,
    0.3573, -0.0548, -0.1681, -0.0667,
];
const VAE_STD: [f32; 48] = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.499, 0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901,
    0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944,
    0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.06, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
];

pub(in crate::cuda_cosmos3_edge::model) struct Bf16Blocks {
    pub ctx: Rc<CudaContext>,
    weights: Rc<Bf16Weights>,
    text: Tensor,
    noise: Tensor,
    pub actions: Tensor,
    und_k: Vec<Tensor>,
    und_v: Vec<Tensor>,
    /// UND mRoPE positions (`[seq, 3]` f32), uploaded once so the forward
    /// performs no host-to-device transfer and stays graph-capturable.
    pos_und: CudaBuffer,
    /// The same positions as FP32 cos/sin tables for fused head-norm + mRoPE.
    rope_und: Tensor,
    rope_gen: Tensor,
    /// droid_lerobot gripper inversion at the JSON boundary: `1 - value` on the
    /// last raw action dim, as `scale * x + bias` per element.
    gripper_scale: Tensor,
    gripper_bias: Tensor,
    s_und: usize,
    steps: usize,
    /// Run the MLP up-projection as one GEMM whose epilogue applies the
    /// squared ReLU. The separate activation pass is skipped when this is set;
    /// `APXINF_COSMOS3_SCALAR_RELU2=1` restores the split path.
    fused_relu2: bool,
    // scratch
    und_x: Tensor,
    und_norm: Tensor,
    und_q: Tensor,
    und_k1: Tensor,
    und_attn: Tensor,
    und_mlp: Tensor,
    gen_x: Tensor,
    gen_norm: Tensor,
    gen_q: Tensor,
    gen_k: Tensor,
    gen_attn: Tensor,
    gen_mlp: Tensor,
    k_all: Tensor,
    v_all: Tensor,
    latent: Tensor,
    latent_pred: Tensor,
    action_latent: Tensor,
    action_pred: Tensor,
    video_tokens: Tensor,
    video_pred: Tensor,
    time: Tensor,
    patch_map: Tensor,
    unpatch_map: Tensor,
    action_map: Tensor,
    rgb: CudaBuffer,
    pixel: Tensor,
    /// The singleton frame of the zero-led causal timeline the VAE input
    /// convolution reads; request pixels are staged here each forward.
    pixel_frames: Tensor,
    vae_first: vae::VaeChunk,
    te_bcast: Tensor,
    gen_rows: Tensor,
    state: Tensor,
    solver_x0: Vec<Tensor>,
    solver_last: Tensor,
    solver_scratch: Tensor,
    zero: Tensor,
    solver_hist: usize,
    solver_have: bool,
    solver_lower: i32,
    solver_this: i32,
    sigmas: Vec<f32>,
    perm: Tensor,
    cond_map: Tensor,
    cond: Tensor,
    cond_bf16: Tensor,
    latent_scale: Tensor,
    latent_shift: Tensor,
}

fn alloc2(ctx: &CudaContext, shape: Vec<usize>, dtype: DType) -> Result<Tensor> {
    let bytes = shape.iter().product::<usize>() * dtype.size_in_bytes();
    CudaBuffer::alloc(bytes, ctx.device_id())
        .map_err(apxinf_core::Error::Cuda)?
        .as_tensor(shape.into(), dtype)
        .map_err(apxinf_core::Error::Cuda)
}

impl Bf16Blocks {
    pub fn new(
        ctx: Rc<CudaContext>,
        weights: Rc<Bf16Weights>,
        inputs: &super::super::Inputs,
        schedule: &[super::super::SolverStep],
        convolution: &crate::cuda_cosmos3_edge::backend::Conv3dPolicy,
    ) -> Result<Self> {
        let s_und = weights.text_tokens;
        // The fused epilogue measured ~20 ms/request faster than the separate
        // relu2 pass. `APXINF_COSMOS3_SCALAR_RELU2=1` restores the old path;
        // the fused provider admits only the M=1971 up-projection shape and
        // falls back to GEMM + staged relu2 elsewhere.
        let fused_relu2 = std::env::var_os("APXINF_COSMOS3_SCALAR_RELU2").is_none();
        let bf = |r: usize, c: usize| alloc2(&ctx, vec![r, c], DType::BF16);
        let mut und_k = Vec::with_capacity(28);
        let mut und_v = Vec::with_capacity(28);
        for _ in 0..28 {
            und_k.push(bf(s_und, 1024)?);
            und_v.push(bf(s_und, 1024)?);
        }
        // patch map: token (t,ty,tx) x width (dr,dc,c<48) -> latent flat index
        let mut patch = Vec::with_capacity(S_VID * 192);
        for t in 0..5usize {
            for ty in 0..17usize {
                for tx in 0..23usize {
                    for dr in 0..2usize {
                        for dc in 0..2usize {
                            for c in 0..48usize {
                                let flat = ((t * 34 + (2 * ty + dr)) * 46 + (2 * tx + dc)) * 48 + c;
                                patch.push(flat as i32);
                            }
                        }
                    }
                }
            }
        }
        let patch_map = tensor_ops::indices(&ctx, &patch)?;
        let mut unpatch = vec![0i32; LAT];
        for t in 0..5usize {
            for h in 0..34usize {
                for w in 0..46usize {
                    for c in 0..48usize {
                        let tk = (t * 17 + h / 2) * 23 + w / 2;
                        let widx = ((h % 2) * 2 + (w % 2)) * 48 + c;
                        unpatch[((t * 34 + h) * 46 + w) * 48 + c] = (tk * 192 + widx) as i32;
                    }
                }
            }
        }
        let unpatch_map = tensor_ops::indices(&ctx, &unpatch)?;
        let mut amap = Vec::with_capacity(16 * 10);
        for r in 0..16usize {
            for c in 0..10usize {
                amap.push((r * 64 + c) as i32);
            }
        }
        let action_map = tensor_ops::indices(&ctx, &amap)?;
        let pixel = inputs.pixel.clone();
        let pixel_first = vae::CausalActivation::prepare(&ctx, &[1, 272, 368, 12], 1, None)?;
        let pixel_frames = pixel_first.frames()?;
        let vae_first =
            vae::VaeChunk::prepare(convolution, &ctx, &weights.vae, &pixel_first, 1, None)?;
        let gen_rows = tensor_ops::indices(&ctx, &vec![0i32; S_GEN])?;
        let mut cmap = Vec::with_capacity(34 * 46 * 48);
        for h in 0..34usize {
            for w in 0..46usize {
                for c in 0..48usize {
                    cmap.push((((h * 46) + w) * 96 + c) as i32);
                }
            }
        }
        let cond_map = tensor_ops::indices(&ctx, &cmap)?;
        // Per-step sinusoidal timestep embedding [steps, 256] (host -> device).
        let mut time_vals = Vec::with_capacity(schedule.len() * 256);
        for step in schedule {
            let t = step.timestep * 0.001;
            // reference TimestepEmbedder: cat([cos(args), sin(args)]) — cos block then sin block
            for d in 0..128 {
                let freq = (-(10000f32).ln() * d as f32 / 128.).exp();
                time_vals.push(half::bf16::from_f32((t * freq).cos()));
            }
            for d in 0..128 {
                let freq = (-(10000f32).ln() * d as f32 / 128.).exp();
                time_vals.push(half::bf16::from_f32((t * freq).sin()));
            }
        }
        let time = crate::cuda_cosmos3_edge::weights::upload(
            &Tensor::from_bf16(vec![schedule.len(), 256], &time_vals)?,
            ctx.device_id(),
        )?;
        let latent = alloc2(&ctx, vec![LAT + LAT_A, 1], DType::BF16)?;
        let latent_pred = alloc2(&ctx, vec![LAT + LAT_A, 1], DType::BF16)?;
        let action_latent = transformer::slice(&latent, LAT, LAT_A, vec![16, 64])?;
        let action_pred = transformer::slice(&latent_pred, LAT, LAT_A, vec![16, 64])?;
        let mut scale_vals = Vec::with_capacity(34 * 46 * 48);
        let mut shift_vals = Vec::with_capacity(34 * 46 * 48);
        for _ in 0..(34 * 46) {
            for c in 0..48 {
                scale_vals.push(1.0 / VAE_STD[c]);
                shift_vals.push(-VAE_MEAN[c] / VAE_STD[c]);
            }
        }
        let latent_scale = crate::cuda_cosmos3_edge::weights::upload(
            &Tensor::from_f32(vec![34 * 46, 48], &scale_vals)?,
            ctx.device_id(),
        )?;
        let latent_shift = crate::cuda_cosmos3_edge::weights::upload(
            &Tensor::from_f32(vec![34 * 46, 48], &shift_vals)?,
            ctx.device_id(),
        )?;
        let mut scale = vec![half::bf16::ONE; 160];
        let mut bias = vec![half::bf16::ZERO; 160];
        for r in 0..16usize {
            scale[r * 10 + 9] = half::bf16::from_f32(-1.0);
            bias[r * 10 + 9] = half::bf16::from_f32(1.0);
        }
        let gripper_scale = crate::cuda_cosmos3_edge::weights::upload(
            &Tensor::from_bf16(vec![160, 1], &scale)?,
            ctx.device_id(),
        )?;
        let gripper_bias = crate::cuda_cosmos3_edge::weights::upload(
            &Tensor::from_bf16(vec![160, 1], &bias)?,
            ctx.device_id(),
        )?;
        Ok(Self {
            ctx: ctx.clone(),
            weights,
            text: inputs.text.clone(),
            noise: inputs.noise.clone(),
            actions: alloc2(&ctx, vec![16, 10], DType::BF16)?,
            und_k,
            und_v,
            pos_und: positions(&ctx, &transformer::make_pos_ids(s_und))?,
            rope_und: rope_table(&ctx, &transformer::make_pos_ids(s_und))?,
            rope_gen: rope_table(&ctx, &transformer::gen_pos_ids(s_und, S_GEN))?,
            gripper_scale,
            gripper_bias,
            s_und,
            steps: schedule.len(),
            fused_relu2,
            und_x: bf(s_und, 2048)?,
            und_norm: bf(s_und, 2048)?,
            und_q: bf(s_und, 2048)?,
            und_k1: bf(s_und, 1024)?,
            und_attn: bf(s_und, 2048)?,
            und_mlp: bf(s_und, 9216)?,
            gen_x: bf(S_GEN, 2048)?,
            gen_norm: bf(S_GEN, 2048)?,
            gen_q: bf(S_GEN, 2048)?,
            gen_k: bf(S_GEN, 1024)?,
            gen_attn: bf(S_GEN, 2048)?,
            gen_mlp: bf(S_GEN, 9216)?,
            k_all: bf(s_und + S_GEN, 1024)?,
            v_all: bf(s_und + S_GEN, 1024)?,
            latent,
            latent_pred,
            state: alloc2(&ctx, vec![LAT + LAT_A, 1], DType::F32)?,
            solver_x0: vec![
                alloc2(&ctx, vec![LAT + LAT_A, 1], DType::F32)?,
                alloc2(&ctx, vec![LAT + LAT_A, 1], DType::F32)?,
                alloc2(&ctx, vec![LAT + LAT_A, 1], DType::F32)?,
            ],
            solver_last: alloc2(&ctx, vec![LAT + LAT_A, 1], DType::F32)?,
            solver_scratch: alloc2(&ctx, vec![LAT + LAT_A, 1], DType::F32)?,
            zero: upload_f32_zero(&ctx, LAT + LAT_A)?,
            solver_hist: 0,
            solver_have: false,
            solver_lower: 0,
            solver_this: 1,
            sigmas: {
                let mut v: Vec<f32> = schedule.iter().map(|s| s.sigma as f32).collect();
                v.push(0.0);
                v
            },
            perm: {
                let mut pv = Vec::with_capacity(LAT);
                for t in 0..5usize {
                    for h in 0..34usize {
                        for w in 0..46usize {
                            for c in 0..48usize {
                                pv.push((((c * 5 + t) * 34 + h) * 46 + w) as i32);
                            }
                        }
                    }
                }
                tensor_ops::indices(&ctx, &pv)?
            },
            cond_map,
            cond: alloc2(&ctx, vec![34 * 46 * 48, 1], DType::F32)?,
            cond_bf16: alloc2(&ctx, vec![34 * 46 * 48, 1], DType::BF16)?,
            latent_scale,
            latent_shift,
            action_latent,
            action_pred,
            video_tokens: bf(S_VID, 192)?,
            video_pred: bf(S_VID, 192)?,
            time,
            patch_map,
            unpatch_map,
            action_map,
            rgb: inputs.rgb.clone(),
            pixel,
            pixel_frames,
            vae_first,
            te_bcast: bf(S_GEN, 2048)?,
            gen_rows,
        })
    }

    pub fn prepare_constants(&mut self, _g: &GemmPolicy, _a: &AttentionPolicy) -> Result<()> {
        Ok(())
    }

    /// VAE-encode the singleton observation into the frame-0 condition latent.
    /// The 16-frame rollout chunk is dropped: its latent output was never read
    /// by the policy path, and the causal VAE makes frame 0 independent of it.
    pub fn encode_observation(&mut self, _g: &GemmPolicy, a: &AttentionPolicy) -> Result<()> {
        use tensor_ops::cast;
        let ctx = self.ctx.clone();
        crate::cuda_cosmos3_edge::backend::ops::rgb_patch2x2(&ctx, &self.rgb, &self.pixel)?;
        crate::cuda_cosmos3_edge::backend::transfers::copy_cuda_to_cuda(
            &ctx,
            &self.pixel,
            &self.pixel_frames,
        )?;
        self.vae_first.forward(&ctx, &self.weights.vae, a)?;
        // Extract frame 0's first 48 channels (the mean) -> cond, cast to F32.
        let flat = self
            .vae_first
            .output
            .clone()
            .reshape(vec![34 * 46 * 96, 1])?;
        tensor_ops::gather(&ctx, &flat, &self.cond_map, 1, &self.cond_bf16)?;
        cast(&ctx, &self.cond_bf16, &self.cond)?;
        // normalize the condition latent: (x - mean[c]) / std[c] per channel
        let cond_m = self.cond.reshape(vec![34 * 46, 48])?;
        tensor_ops::multiply(&ctx, &cond_m, &self.latent_scale, &cond_m)?;
        tensor_ops::add(&ctx, &cond_m, &self.latent_shift, &cond_m)?;
        Ok(())
    }

    pub fn und_prefill(&mut self, g: &GemmPolicy, a: &AttentionPolicy) -> Result<()> {
        use crate::cuda_cosmos3_edge::backend::{attention, AttentionArgs, AttentionMask};
        use tensor_ops::linear;
        use transformer::{add_rms, mrope, norm_mrope, relu2, rms};
        let ctx = self.ctx.clone();
        let s = self.s_und;
        let und = &self.weights.transformer.und;
        crate::cuda_cosmos3_edge::backend::transfers::copy_cuda_to_cuda(
            &ctx,
            &self.text.reshape(vec![s, 2048])?,
            &self.und_x,
        )?;
        // As in GEN, each residual add produces the next norm in the same pass.
        rms(&ctx, &self.und_x, &und[0].input_norm, &self.und_norm)?;
        for i in 0..28 {
            let w = &und[i];
            linear(&ctx, &self.und_norm, &w.k, &self.und_k1, g)?;
            // V feeds both this layer's attention and the GEN K/V cache.
            linear(&ctx, &self.und_norm, &w.v, &self.und_v[i], g)?;
            // emit GEN K/V
            norm_mrope(
                &ctx,
                &self.und_k1.reshape(vec![s, 8, 128])?,
                &w.k_norm_for_gen,
                &self.rope_und,
                &self.und_k[i].reshape(vec![s, 8, 128])?,
            )?;
            if i + 1 == und.len() {
                // GEN only reads per-layer K/V; the last layer's attention and
                // MLP outputs are never consumed.
                break;
            }
            linear(&ctx, &self.und_norm, &w.q, &self.und_q, g)?;
            let uq = self.und_q.reshape(vec![s, 16, 128])?;
            let uk = self.und_k1.reshape(vec![s, 8, 128])?;
            let q = tensor_ops::alloc(&ctx, vec![s, 16, 128], apxinf_core::DType::BF16)?;
            mrope(&ctx, &uq, &self.pos_und, &q)?;
            let k = tensor_ops::alloc(&ctx, vec![s, 8, 128], apxinf_core::DType::BF16)?;
            mrope(&ctx, &uk, &self.pos_und, &k)?;
            let mut o = tensor_ops::alloc(&ctx, vec![1, s, 16, 128], apxinf_core::DType::BF16)?;
            let qch = q.reshape(vec![1, s, 16, 128])?;
            let kch = k.reshape(vec![1, s, 8, 128])?;
            let vch = self.und_v[i].reshape(vec![1, s, 8, 128])?;
            let mut args = AttentionArgs::new(&qch, &kch, &vch, &mut o);
            args.mask = AttentionMask::Causal;
            args.policy = a.clone();
            attention(&ctx, args)?;
            let o2 = o.reshape(vec![s, 2048])?;
            linear(&ctx, &o2, &w.o, &self.und_attn, g)?;
            add_rms(
                &ctx,
                &self.und_x,
                &self.und_attn,
                &w.post_norm,
                &self.und_norm,
            )?;
            linear(&ctx, &self.und_norm, &w.mlp_up, &self.und_mlp, g)?;
            relu2(&ctx, &self.und_mlp)?;
            let down = tensor_ops::bf(&ctx, s, 2048)?;
            linear(&ctx, &self.und_mlp, &w.mlp_down, &down, g)?;
            add_rms(
                &ctx,
                &self.und_x,
                &down,
                &und[i + 1].input_norm,
                &self.und_norm,
            )?;
        }
        Ok(())
    }

    fn reinject_cond(&self) -> Result<()> {
        // reinject the VAE observation into frame 0 (THWC first 34*46*48 elements)
        let dst = transformer::slice(&self.state, 0, 34 * 46 * 48, vec![34 * 46 * 48, 1])?;
        crate::cuda_cosmos3_edge::backend::transfers::copy_cuda_to_cuda(
            &self.ctx, &self.cond, &dst,
        )?;
        // zero the padded action dims [10:64] (reference reinjectConditioning)
        for r in 0..16usize {
            let act_dst = transformer::slice(&self.state, LAT + r * 64 + 10, 54, vec![54, 1])?;
            let act_src = transformer::slice(&self.zero, LAT + r * 64 + 10, 54, vec![54, 1])?;
            crate::cuda_cosmos3_edge::backend::transfers::copy_cuda_to_cuda(
                &self.ctx, &act_src, &act_dst,
            )?;
        }
        Ok(())
    }

    pub fn initial_state(&mut self) -> Result<()> {
        // Solver order/history belongs to one request. Reset host traversal
        // state before eager execution and capture so replay has fixed coefficients.
        self.solver_hist = 0;
        self.solver_have = false;
        self.solver_lower = 0;
        self.solver_this = 1;
        // transpose the reference CTHW noise into our THWC state layout (video)
        let nf = transformer::slice(&self.noise, 0, LAT, vec![LAT, 1])?;
        let video_state = transformer::slice(&self.state, 0, LAT, vec![LAT, 1])?;
        tensor_ops::gather(&self.ctx, &nf, &self.perm, 1, &video_state)?;
        // action noise -> packed F32 state tail
        let src_action = transformer::slice(&self.noise, LAT, LAT_A, vec![16, 64])?;
        let dst_action = transformer::slice(&self.state, LAT, LAT_A, vec![16, 64])?;
        crate::cuda_cosmos3_edge::backend::transfers::copy_cuda_to_cuda(
            &self.ctx,
            &src_action,
            &dst_action,
        )?;
        self.reinject_cond()
    }

    pub fn denoise_step(
        &mut self,
        index: usize,
        g: &GemmPolicy,
        a: &AttentionPolicy,
    ) -> Result<()> {
        use crate::cuda_cosmos3_edge::backend::{attention, AttentionArgs, PointwiseActivation};
        use tensor_ops::{activate, add, bf, cast, gather, linear};
        use transformer::{relu2, rms, slice};
        let ctx = self.ctx.clone();
        // seed the BF16 latent from the F32 solver state, then patchify
        cast(&ctx, &self.state, &self.latent)?;
        gather(&ctx, &self.latent, &self.patch_map, 1, &self.video_tokens)?;
        let vid_proj = transformer::slice(&self.gen_x, 0, S_VID * 2048, vec![S_VID, 2048])?;
        linear(
            &ctx,
            &self.video_tokens,
            &self.weights.transformer.proj_in,
            &vid_proj,
            g,
        )?;
        // action tokens
        let mut at = bf(&ctx, S_ACT, 2048)?;
        linear(
            &ctx,
            &self.action_latent,
            &self.weights.transformer.action_in,
            &at,
            g,
        )?;
        activate(
            &ctx,
            &at,
            Some(&self.weights.transformer.action_modality_embed),
            PointwiseActivation::None,
            &at,
        )?;
        // timestep embedding -> add to all tokens
        let mut th = bf(&ctx, 1, 2048)?;
        let t_in = transformer::slice(&self.time, index * 256, 256, vec![1, 256])?;
        linear(&ctx, &t_in, &self.weights.transformer.time.l1, &th, g)?;
        activate(&ctx, &th, None, PointwiseActivation::Silu, &th)?;
        let mut te = bf(&ctx, 1, 2048)?;
        linear(&ctx, &th, &self.weights.transformer.time.l2, &te, g)?;
        // broadcast the per-step timestep embedding to every token and add it
        let te_row = te.reshape(vec![1, 2048])?;
        gather(&ctx, &te_row, &self.gen_rows, 2048, &self.te_bcast)?;
        // reference gates the timestep by token_noisy_mask: frame-0 patch tokens are
        // clean (mask=0) and get no timestep; frames 1..4 + action tokens are noisy.
        let frame0 = 17 * 23; // one latent frame = 17x23 patch tokens
        let vid_rest = slice(
            &self.gen_x,
            frame0 * 2048,
            (S_VID - frame0) * 2048,
            vec![S_VID - frame0, 2048],
        )?;
        let te_rest = slice(
            &self.te_bcast,
            frame0 * 2048,
            (S_VID - frame0) * 2048,
            vec![S_VID - frame0, 2048],
        )?;
        add(&ctx, &vid_rest, &te_rest, &vid_rest)?;
        let te_act = slice(&self.te_bcast, 0, S_ACT * 2048, vec![S_ACT, 2048])?;
        add(&ctx, &at, &te_act, &at)?;
        // action part in gen_x tail
        let gen_tail = slice(&self.gen_x, S_VID * 2048, S_ACT * 2048, vec![S_ACT, 2048])?;
        crate::cuda_cosmos3_edge::backend::transfers::copy_cuda_to_cuda(&ctx, &at, &gen_tail)?;
        // 28 GEN layers
        // Each residual add is fused with the norm that consumes it: the MLP
        // residual of layer i produces layer i+1's input norm (or the final norm).
        rms(
            &ctx,
            &self.gen_x,
            &self.weights.transformer.gen[0].input_norm,
            &self.gen_norm,
        )?;
        for i in 0..28 {
            let w = &self.weights.transformer.gen[i];
            linear(&ctx, &self.gen_norm, &w.q, &self.gen_q, g)?;
            linear(&ctx, &self.gen_norm, &w.k, &self.gen_k, g)?;
            // Joint attention reads [UND ; GEN] K/V from one buffer shared by
            // all layers: refresh this layer's UND head, then write the GEN tail
            // in place (V straight from the GEMM, K straight from mRoPE).
            let dst_k_head = slice(&self.k_all, 0, self.s_und * 1024, vec![self.s_und, 1024])?;
            crate::cuda_cosmos3_edge::backend::transfers::copy_cuda_to_cuda(
                &ctx,
                &self.und_k[i],
                &dst_k_head,
            )?;
            let dst_v_head = slice(&self.v_all, 0, self.s_und * 1024, vec![self.s_und, 1024])?;
            crate::cuda_cosmos3_edge::backend::transfers::copy_cuda_to_cuda(
                &ctx,
                &self.und_v[i],
                &dst_v_head,
            )?;
            let dst_v_tail = slice(
                &self.v_all,
                self.s_und * 1024,
                S_GEN * 1024,
                vec![S_GEN, 1024],
            )?;
            linear(&ctx, &self.gen_norm, &w.v, &dst_v_tail, g)?;
            let qr = self.gen_q.reshape(vec![S_GEN, 16, 128])?;
            transformer::norm_mrope(&ctx, &qr, &w.norm_q, &self.rope_gen, &qr)?;
            let gk = self.gen_k.reshape(vec![S_GEN, 8, 128])?;
            let dst_k_tail = slice(
                &self.k_all,
                self.s_und * 1024,
                S_GEN * 1024,
                vec![S_GEN, 8, 128],
            )?;
            transformer::norm_mrope(&ctx, &gk, &w.norm_k, &self.rope_gen, &dst_k_tail)?;
            let mut o = tensor_ops::alloc(&ctx, vec![1, S_GEN, 16, 128], apxinf_core::DType::BF16)?;
            let qch = qr.reshape(vec![1, S_GEN, 16, 128])?;
            let kch = self.k_all.reshape(vec![1, self.s_und + S_GEN, 8, 128])?;
            let vch = self.v_all.reshape(vec![1, self.s_und + S_GEN, 8, 128])?;
            let mut args = AttentionArgs::new(&qch, &kch, &vch, &mut o);
            args.policy = a.clone();
            attention(&ctx, args)?;
            let o2 = o.reshape(vec![S_GEN, 2048])?;
            linear(&ctx, &o2, &w.o, &self.gen_attn, g)?;
            transformer::add_rms(
                &ctx,
                &self.gen_x,
                &self.gen_attn,
                &w.post_norm,
                &self.gen_norm,
            )?;
            if self.fused_relu2 {
                tensor_ops::linear_relu_squared(&ctx, &self.gen_norm, &w.mlp_up, &self.gen_mlp, g)?;
            } else {
                linear(&ctx, &self.gen_norm, &w.mlp_up, &self.gen_mlp, g)?;
                relu2(&ctx, &self.gen_mlp)?;
            }
            let down = bf(&ctx, S_GEN, 2048)?;
            linear(&ctx, &self.gen_mlp, &w.mlp_down, &down, g)?;
            let next_norm = match self.weights.transformer.gen.get(i + 1) {
                Some(next) => &next.input_norm,
                None => &self.weights.transformer.norm_gen,
            };
            transformer::add_rms(&ctx, &self.gen_x, &down, next_norm, &self.gen_norm)?;
        }
        // final norm (fused above) + proj_out video -> video_pred, unpatchify -> latent_pred
        let vid = slice(&self.gen_norm, 0, S_VID * 2048, vec![S_VID, 2048])?;
        linear(
            &ctx,
            &vid,
            &self.weights.transformer.proj_out,
            &self.video_pred,
            g,
        )?;
        let vid_pred = transformer::slice(&self.latent_pred, 0, LAT, vec![LAT, 1])?;
        gather(&ctx, &self.video_pred, &self.unpatch_map, 1, &vid_pred)?;
        // action head
        let act = slice(
            &self.gen_norm,
            S_VID * 2048,
            S_ACT * 2048,
            vec![S_ACT, 2048],
        )?;
        linear(
            &ctx,
            &act,
            &self.weights.transformer.action_out,
            &self.action_pred,
            g,
        )?;
        // ---- exact UniPC flow-matching step on the packed F32 state (video + action)
        let k = index as i32;
        let sigma_s0 = self.sigmas[k as usize];
        let mut vel = tensor_ops::alloc(&ctx, vec![LAT + LAT_A, 1], apxinf_core::DType::F32)?;
        cast(&ctx, &self.latent_pred, &vel)?; // velocity (prediction)
        let x0 = self.solver_x0[(k % 3) as usize].clone();
        tensor_ops::scale_add(&ctx, &self.state, -sigma_s0, &vel, &x0)?; // x0 = state - sigma*vel
        if k > 0 && self.solver_have {
            let c = unipc_coeffs(
                &self.sigmas,
                self.solver_this,
                k as usize,
                (k - 1) as usize,
                k,
                true,
                self.solver_hist,
            );
            let m0 = self.solver_x0[((k - 1) % 3) as usize].clone();
            let mp = if c.use_mp {
                Some(self.solver_x0[((k - 2) % 3) as usize].clone())
            } else {
                None
            };
            tensor_ops::scale_add(
                &ctx,
                &self.zero,
                c.c_x,
                &self.solver_last,
                &self.solver_scratch,
            )?;
            tensor_ops::scale_add(
                &ctx,
                &self.solver_scratch,
                c.c_m0,
                &m0,
                &self.solver_scratch,
            )?;
            if let Some(mp) = &mp {
                tensor_ops::scale_add(
                    &ctx,
                    &self.solver_scratch,
                    c.c_mp,
                    mp,
                    &self.solver_scratch,
                )?;
            }
            tensor_ops::scale_add(
                &ctx,
                &self.solver_scratch,
                c.c_mt,
                &x0,
                &self.solver_scratch,
            )?;
            std::mem::swap(&mut self.solver_scratch, &mut self.solver_last);
        } else {
            crate::cuda_cosmos3_edge::backend::transfers::copy_cuda_to_cuda(
                &ctx,
                &self.state,
                &self.solver_last,
            )?;
        }
        self.solver_hist = (self.solver_hist + 1).min(2);
        let mut this_order = std::cmp::min(2, self.steps as i32 - k);
        this_order = std::cmp::min(this_order, self.solver_lower + 1);
        self.solver_this = this_order;
        self.solver_have = true;
        let c = unipc_coeffs(
            &self.sigmas,
            this_order,
            (k + 1) as usize,
            k as usize,
            k,
            false,
            self.solver_hist,
        );
        let mp = if c.use_mp {
            Some(self.solver_x0[((k - 1) % 3) as usize].clone())
        } else {
            None
        };
        tensor_ops::scale_add(
            &ctx,
            &self.zero,
            c.c_x,
            &self.solver_last,
            &self.solver_scratch,
        )?;
        tensor_ops::scale_add(
            &ctx,
            &self.solver_scratch,
            c.c_m0,
            &x0,
            &self.solver_scratch,
        )?;
        if let Some(mp) = &mp {
            tensor_ops::scale_add(&ctx, &self.solver_scratch, c.c_mp, mp, &self.solver_scratch)?;
        }
        crate::cuda_cosmos3_edge::backend::transfers::copy_cuda_to_cuda(
            &ctx,
            &self.solver_scratch,
            &self.state,
        )?;
        self.solver_lower = (self.solver_lower + 1).min(2);
        // reference re-imposes frame-0 conditioning after every denoise step
        self.reinject_cond()?;
        Ok(())
    }

    pub fn finish(&self) -> Result<()> {
        // denoised action state lives in the packed F32 state tail; cast to BF16
        // then gather the raw 10 action dims.
        let dst_action = transformer::slice(&self.state, LAT, LAT_A, vec![16, 64])?;
        tensor_ops::cast(&self.ctx, &dst_action, &self.action_latent)?;
        let ap = self.action_latent.reshape(vec![16 * 64, 1])?;
        let out = self.actions.reshape(vec![16 * 10, 1])?;
        tensor_ops::gather(&self.ctx, &ap, &self.action_map, 1, &out)?;
        // droid_lerobot 的 gripper（最后一个 action 维）在 JSON 边界要 1-value 翻转，
        // 对应参考 writeActionJson 里的 invertGripper。
        tensor_ops::multiply(&self.ctx, &out, &self.gripper_scale, &out)?;
        tensor_ops::add(&self.ctx, &out, &self.gripper_bias, &out)?;
        Ok(())
    }
    pub fn actions(&self) -> &Tensor {
        &self.actions
    }
    pub fn workspace_bytes(&self) -> usize {
        1 << 30
    }
}

fn lambda_of(s: f32) -> f32 {
    (1.0 - s).ln() - s.ln()
}

struct UniCoeffs {
    c_x: f32,
    c_m0: f32,
    c_mp: f32,
    c_mt: f32,
    use_mp: bool,
}

fn unipc_coeffs(
    sigmas: &[f32],
    order: i32,
    idx_t: usize,
    idx_s0: usize,
    k: i32,
    corrector: bool,
    hist: usize,
) -> UniCoeffs {
    let sigma_t = sigmas[idx_t];
    let sigma_s0 = sigmas[idx_s0];
    let alpha_t = 1.0 - sigma_t;
    let lambda_s0 = lambda_of(sigma_s0);
    let inf = f32::INFINITY;
    let lambda_t = if sigma_t <= 0.0 {
        inf
    } else {
        lambda_of(sigma_t)
    };
    let h = lambda_t - lambda_s0;
    let hh = -h;
    let hh_neg_inf = hh == -inf;
    let h_phi1 = if hh_neg_inf { -1.0 } else { hh.exp_m1() };
    let b_h = if hh_neg_inf { -1.0 } else { hh.exp_m1() };
    let mut rk = 1.0f32;
    let mut use_mp = false;
    if order >= 2 && hist >= 2 {
        let si = if corrector { k - 2 } else { k - 1 };
        if si >= 0 {
            rk = (lambda_of(sigmas[si as usize]) - lambda_s0) / h;
            use_mp = true;
        }
    }
    let mut rho_first = 0.5f32;
    let mut rho_last = 0.5f32;
    if corrector && order >= 2 {
        let h_phi_k0 = if hh_neg_inf { -1.0 } else { h_phi1 / hh - 1.0 };
        let h_phi_k1 = (if hh_neg_inf { 0.0 } else { h_phi_k0 / hh }) - 0.5;
        let b0 = h_phi_k0 / b_h;
        let b1 = 2.0 * h_phi_k1 / b_h;
        let det = 1.0 - rk;
        if det.abs() > f32::EPSILON {
            rho_first = (b0 - b1) / det;
            rho_last = (b1 - rk * b0) / det;
        }
    }
    let mut c_x = sigma_t / sigma_s0;
    let mut c_m0 = -alpha_t * h_phi1;
    let mut c_mp = 0.0;
    let mut c_mt = 0.0;
    if use_mp {
        let dd = -alpha_t * b_h * rho_first / rk;
        c_mp = dd;
        c_m0 -= dd;
    }
    if corrector {
        let dt = -alpha_t * b_h * rho_last;
        c_mt = dt;
        c_m0 -= dt;
    }
    UniCoeffs {
        c_x: c_x,
        c_m0: c_m0,
        c_mp: c_mp,
        c_mt: c_mt,
        use_mp,
    }
}

fn positions(ctx: &CudaContext, ids: &[f32]) -> Result<CudaBuffer> {
    let t = crate::cuda_cosmos3_edge::weights::upload(
        &Tensor::from_f32(vec![ids.len() / 3, 3], ids)?,
        ctx.device_id(),
    )?;
    CudaBuffer::from_tensor(&t).map_err(apxinf_core::Error::Cuda)
}

fn rope_table(ctx: &CudaContext, ids: &[f32]) -> Result<Tensor> {
    crate::cuda_cosmos3_edge::weights::upload(
        &Tensor::from_f32(
            vec![ids.len() / 3, 2, transformer::D / 2],
            &transformer::mrope_table(ids),
        )?,
        ctx.device_id(),
    )
}

fn upload_f32_zero(ctx: &CudaContext, n: usize) -> Result<Tensor> {
    let t = Tensor::from_f32(vec![n, 1], &vec![0f32; n])?;
    Ok(crate::cuda_cosmos3_edge::weights::upload(
        &t,
        ctx.device_id(),
    )?)
}
