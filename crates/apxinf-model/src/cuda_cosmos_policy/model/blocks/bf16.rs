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
    /// DiT MLP up projection. The checkpoint's `mlp.layer1` has no bias, which
    /// is also the only form the staged exact-GELU GEMM accepts. The biased
    /// branch is kept as an explicit unfused composition rather than silently
    /// reading a different rounding order if a future checkpoint adds one.
    pub fn linear_gelu_exact(
        ctx: &CudaContext,
        input: &Tensor,
        w: &Linear,
        output: &Tensor,
        policy: &GemmPolicy,
    ) -> Result<()> {
        if w.bias.is_some() {
            linear(ctx, input, w, output, policy)?;
            return activate(ctx, output, None, PointwiseActivation::GeluExact, output);
        }
        let mut out = output.clone();
        let mut args = GemmArgs::new(input, &w.weight, &mut out);
        args.policy = policy.clone();
        args.weight_version = Some(WeightVersion::new(1));
        crate::cuda_cosmos_policy::backend::ops::gemm_gelu_exact(ctx, args)
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
    use crate::cuda_cosmos_policy::backend::{
        fused_add_channel_l2_norm_silu, pointwise, residual_add, BiasOrder, Conv3dArgs,
        Conv3dPolicy, CudaBuffer, CudaContext, FusedAddChannelL2NormArgs, PointwiseArgs,
        PointwiseSemantic, PreparedConv3d, ResidualAddArgs, Unfold3dGeometry,
    };
    use crate::cuda_cosmos_policy::weights::Conv;
    use apxinf_core::{DType, Error, Result, Shape, Tensor};

    /// Per-convolution scratch budget. Every `ConvWorkspace` keeps its own
    /// lowered tile buffers alive for the plan's lifetime, so this value is
    /// effectively multiplied by the ~24 VAE convolutions rather than shared.
    /// Lowering derives `tile_rows = limit / (2 * (k + n) * 2)`, so the bound
    /// is what actually decides the tile: 512 MiB kept every level at its
    /// 4096-row cap and retained 543 MiB of scratch; 32 MiB retains ~280 MiB
    /// and costs ~210 extra tile launches for the whole request. Smaller
    /// values trade launches faster than they release bytes.
    const CONV3D_WORKSPACE_LIMIT: usize = 32 * 1024 * 1024;

    pub(in crate::cuda_cosmos_policy::model::blocks::bf16) struct ConvWorkspace {
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
                    .checked_add(padding_before[axis])
                    .and_then(|v| v.checked_add(padding_after[axis]))
                    .and_then(|v| {
                        v.checked_add(if axis == 0 {
                            history.map_or(0, |h| h.shape().dims()[0])
                        } else {
                            0
                        })
                    })
                    .ok_or_else(|| Error::Other("Cosmos VAE convolution extent overflow".into()))?;
                if stride[axis] == 0 || padded < weights.kernel[axis] {
                    return Err(Error::Other(
                        "Cosmos VAE convolution padding/stride mismatch".into(),
                    ));
                }
                output_shape[axis] = (padded - weights.kernel[axis]) / stride[axis] + 1;
            }
            output_shape[3] = weights.output_channels;
            let bytes = output_shape
                .iter()
                .try_fold(2usize, |a, &b| a.checked_mul(b))
                .ok_or_else(|| Error::Other("Cosmos VAE convolution output overflow".into()))?;
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
        /// Materialize the biased result and the next channel normalization together.
        pub fn forward_normalized(
            &mut self,
            ctx: &CudaContext,
            weights: &Conv,
            norm_weight: &Tensor,
            normalized: &Tensor,
        ) -> Result<()> {
            self.convolution.enqueue()?;
            let shape = vec![
                self.output.numel() / weights.output_channels,
                weights.output_channels,
            ];
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

        /// Preserve the staged residual output while producing its next normalization.
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
            let shape = vec![
                self.output.numel() / weights.output_channels,
                weights.output_channels,
            ];
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

        /// Publish a residual result without materializing a biased convolution intermediate.
        /// `self.output` remains convolution scratch; temporal history uses normalized inputs.
        pub fn forward_residual(
            &mut self,
            ctx: &CudaContext,
            weights: &Conv,
            residual: &Tensor,
            output: &Tensor,
        ) -> Result<()> {
            self.convolution.enqueue()?;
            let shape = vec![
                self.output.numel() / weights.output_channels,
                weights.output_channels,
            ];
            residual_add(
                ctx,
                ResidualAddArgs {
                    input: &self.output.reshape(shape.clone())?,
                    bias: weights.linear.bias.as_ref(),
                    residual: &residual.reshape(shape.clone())?,
                    hidden: &output.reshape(shape)?,
                    gate: None,
                    // Conv output rounds first, bias addition rounds before residual addition.
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
    use crate::cuda_cosmos_policy::backend::ops::embedding_gather;
    use crate::cuda_cosmos_policy::backend::{
        attention, channel_l2_norm, channel_l2_norm_silu, pointwise, AttentionArgs,
        AttentionPolicy, ChannelL2NormArgs, Conv3dPolicy, CudaBuffer, CudaContext, PointwiseArgs,
        PointwiseSemantic,
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

    /// A chronological activation allocation shared by the two causal chunks.
    /// The two leading frames stay zero; producers write only their current view.
    struct CausalActivation {
        storage: CudaBuffer,
        current: Tensor,
        causal: Tensor,
        end: usize,
        capacity: usize,
    }

    impl CausalActivation {
        fn prepare(
            ctx: &CudaContext,
            shape: &[usize],
            total_frames: usize,
            previous: Option<&Self>,
        ) -> Result<Self> {
            if shape.len() != 4 || shape.contains(&0) || total_frames < shape[0] {
                return Err(Error::Other("invalid causal activation extent".into()));
            }
            let capacity = total_frames
                .checked_add(2)
                .ok_or_else(|| Error::Other("causal activation capacity overflow".into()))?;
            let frame_bytes = shape[1..]
                .iter()
                .try_fold(2usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| Error::Other("causal activation frame overflow".into()))?;
            let (storage, start) = if let Some(previous) = previous {
                if previous.capacity != capacity
                    || previous.current.shape().dims()[1..] != shape[1..]
                {
                    return Err(Error::Other(
                        "causal activation history geometry mismatch".into(),
                    ));
                }
                (previous.storage.clone(), previous.end)
            } else {
                let bytes = capacity
                    .checked_mul(frame_bytes)
                    .ok_or_else(|| Error::Other("causal activation allocation overflow".into()))?;
                (
                    CudaBuffer::alloc_zeros(bytes, ctx.device_id()).map_err(Error::Cuda)?,
                    2,
                )
            };
            let end = start
                .checked_add(shape[0])
                .filter(|&end| end <= capacity)
                .ok_or_else(|| {
                    Error::Other("causal activation exceeds prepared timeline".into())
                })?;
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
            })
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

    fn causal(
        policy: &Conv3dPolicy,
        ctx: &CudaContext,
        weights: &Conv,
        input: &Tensor,
        previous: Option<&Tensor>,
    ) -> Result<ConvWorkspace> {
        let history = previous.map(|p| tail(p, 2)).transpose()?;
        let past = history.as_ref().map_or(0, |h| h.shape().dims()[0]);
        ConvWorkspace::prepare(
            policy,
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
        channel_l2_norm_silu(
            ctx,
            ChannelL2NormArgs {
                input: &input,
                weight,
                out: &mut output,
                eps: 1e-12,
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

        // The producer has already populated norm1. The optional destination
        // composes this residual update with the following block's normalization.
        fn forward_normalized(
            &mut self,
            ctx: &CudaContext,
            w: &VaeResidual,
            next_normalization: Option<(&Tensor, &Tensor)>,
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
            if let Some((weight, normalized)) = next_normalization {
                self.conv2.forward_residual_normalized(
                    ctx,
                    &w.conv2,
                    residual,
                    &self.output,
                    weight,
                    normalized,
                )
            } else {
                self.conv2
                    .forward_residual(ctx, &w.conv2, residual, &self.output)
            }
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
            channel_l2_norm(
                ctx,
                ChannelL2NormArgs {
                    input: &matrix(&self.input)?,
                    weight: &w.attention_norm,
                    out: &mut matrix(&self.normalized)?,
                    eps: 1e-12,
                },
            )?;
            self.qkv.forward(ctx, &w.attention_qkv)?;
            let rows = self.q.numel() / 384;
            let table = self.qkv.output.reshape(vec![rows * 3, 384])?;
            for (ids, output) in self.ids.iter().zip([&self.q, &self.k, &self.v]) {
                embedding_gather(ctx, &table, ids, &output.reshape(vec![rows, 384])?)?;
            }
            let mut output = self.attended.reshape(self.q.shape().clone())?;
            let mut args = AttentionArgs::new(&self.q, &self.k, &self.v, &mut output);
            args.policy = attention_policy.clone();
            attention(ctx, args)?;
            self.projection
                .forward_residual(ctx, &w.attention_output, &self.input, &self.output)
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
        head_norm: CausalActivation,
        head_conv: ConvWorkspace,
        mean_projection: ConvWorkspace,
        mean_indices: Tensor,
        negative_mean: Tensor,
        inverse_std: Tensor,
        pub output: Tensor,
    }

    impl VaeChunk {
        pub fn prepare(
            policy: &Conv3dPolicy,
            ctx: &CudaContext,
            w: &VaeWeights,
            input: &Tensor,
            total_frames: usize,
            previous: Option<&Self>,
        ) -> Result<Self> {
            let input_conv = causal(policy, ctx, &w.input, input, previous.map(|p| &p.input))?;
            let mut current = input_conv.output.clone();
            let mut total_frames = total_frames;
            let mut blocks = Vec::new();
            let mut downsample = Vec::new();
            for (level, weights) in w.down_blocks.iter().enumerate() {
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
                    current = temporal
                        .as_ref()
                        .map_or(&spatial.output, |t| &t.output)
                        .clone();
                    downsample.push(Downsample { spatial, temporal });
                    if w.downsample[level].temporal.is_some() {
                        total_frames = (total_frames - 1) / 2 + 1;
                    }
                }
            }
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
            // The head statistics run over the head-normalized activation, whose
            // last dimension is the VAE latent width (384) regardless of frames.
            let rows = head_norm.current.numel() / 384;
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
            attention: &AttentionPolicy,
        ) -> Result<()> {
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
                second[0].forward_normalized(ctx, &w.down_blocks[level][1], None)?;
                if level < 3 {
                    let down = &mut self.downsample[level];
                    let norm_weight = &w.down_blocks[level + 1][0].norm1;
                    let normalized = &self.blocks[level + 1][0].norm1.current;
                    if let (Some(temporal), Some(weights)) =
                        (&mut down.temporal, &w.downsample[level].temporal)
                    {
                        down.spatial
                            .forward(ctx, &w.downsample[level].spatial)?;
                        temporal.forward_normalized(ctx, weights, norm_weight, normalized)?;
                    } else {
                        down.spatial.forward_normalized(
                            ctx,
                            &w.downsample[level].spatial,
                            norm_weight,
                            normalized,
                        )?;
                    }
                }
            }
            self.middle_before.forward(ctx, &w.middle_before)?;
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
            let mut first = VaeChunk::prepare(
                &Conv3dPolicy {
                    online_tune: true,
                    ..Default::default()
                },
                &ctx,
                &weights,
                &first,
                17,
                None,
            )?;
            let mut rest = VaeChunk::prepare(
                &Conv3dPolicy {
                    online_tune: true,
                    ..Default::default()
                },
                &ctx,
                &weights,
                &rest,
                17,
                Some(&first),
            )?;
            let session = ExecutionSession::with_capacity(256 * 1024 * 1024, 0)?;
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
                first.forward(&ctx, &weights, &attention)?;
                rest.forward(&ctx, &weights, &attention)
            })?;
            ctx.synchronize().map_err(Error::Cuda)?;
            let eager = collect(&first, &rest)?;
            let graph = crate::cuda_cosmos_policy::backend::capture(&ctx, || {
                session.run(|| {
                    first.forward(&ctx, &weights, &attention)?;
                    rest.forward(&ctx, &weights, &attention)
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
        attention, norm,
        ops::{
            fused_add_norm, Affine, BiasOrder, FusedAddNormArgs, NormKind, NormNumerics,
            ResidualGate,
        },
        AttentionArgs, AttentionPolicy, CudaContext, NormArgs, PointwiseActivation,
    };
    use crate::cuda_cosmos_policy::weights::{upload, Attention, DitWeights, Modulation};
    use apxinf_core::{DType, Result, Shape, Tensor};

    const TOKENS: usize = 9 * 14 * 14;
    const WIDTH: usize = 2048;

    /// Schedule-fixed BF16 styles, prepared with the same shared GEMM scratch
    /// as the original forward path. Request values never enter this cache.
    pub(super) struct StepModulation {
        time_global: Tensor,
        blocks: Vec<[Tensor; 3]>,
        final_style: Tensor,
    }
    impl StepModulation {
        pub(super) fn allocate(ctx: &CudaContext, w: &DitWeights) -> Result<Self> {
            Ok(Self {
                time_global: bf(ctx, 9, 6144)?,
                blocks: w
                    .blocks
                    .iter()
                    .map(|_| Ok([bf(ctx, 9, 6144)?, bf(ctx, 9, 6144)?, bf(ctx, 9, 6144)?]))
                    .collect::<Result<_>>()?,
                final_style: bf(ctx, 9, 4096)?,
            })
        }
    }

    struct Rotary {
        table: Tensor,
    }
    impl Rotary {
        fn new(ctx: &CudaContext, w: &DitWeights, ntk: [f32; 3]) -> Result<Self> {
            // Checkpoint ranges were rounded to BF16 at load. Preserve the
            // original FP32 angle/trigonometry; coefficients broadcast over heads.
            let temporal = w.temporal_range.as_bf16()?;
            let spatial = w.spatial_range.as_bf16()?;
            let seq = w.position_seq.as_bf16()?;
            let mut table = Vec::with_capacity(TOKENS * 128);
            for token in 0..TOKENS {
                let coordinates = [token / 196, token / 14 % 14, token % 14];
                let mut sin = Vec::with_capacity(64);
                for c in 0..64 {
                    let (axis, exponent) = if c < 22 {
                        (0, temporal[c])
                    } else if c < 43 {
                        (1, spatial[c - 22])
                    } else {
                        (2, spatial[c - 43])
                    };
                    let angle = seq[coordinates[axis]].to_f32()
                        * (1. / (10000. * ntk[axis]).powf(exponent.to_f32()));
                    table.push(angle.cos());
                    sin.push(angle.sin());
                }
                table.extend(sin);
            }
            Ok(Self {
                table: upload(
                    &Tensor::from_f32(vec![TOKENS, 2, 64], &table)?,
                    ctx.device_id(),
                )?,
            })
        }
        fn apply_normalized(
            &self,
            ctx: &CudaContext,
            input: &Tensor,
            weight: &Tensor,
        ) -> Result<()> {
            let heads = input.reshape(vec![TOKENS, 16, 128])?;
            crate::cuda_cosmos_policy::backend::ops::rope::rms_norm_apply_precomputed(
                ctx,
                &heads,
                weight,
                &self.table,
                1e-6,
                &heads,
            )
        }
    }

    pub(in crate::cuda_cosmos_policy::model::blocks::bf16) struct Dit {
        pub patches: Tensor,
        pub output: Tensor,
        final_global_ids: Tensor,
        rotary: Rotary,
    }

    /// Per-request intermediate activations. Allocated from the session arena
    /// at the start of a request traversal and dropped when it completes, so
    /// these are request-scoped scratch rather than model state.
    pub(in crate::cuda_cosmos_policy::model::blocks::bf16) struct DitScratch {
        hidden: Tensor,
        normalized: Tensor,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attended: Tensor,
        projected: Tensor,
        mlp: Tensor,
        pub(in crate::cuda_cosmos_policy::model::blocks::bf16) text_kv: Vec<[Tensor; 2]>,
    }

    impl Dit {
        pub fn prepare(ctx: &CudaContext, w: &DitWeights, ntk: [f32; 3]) -> Result<Self> {
            let final_global_ids = indices(
                ctx,
                &(0..9)
                    .flat_map(|t| [t * 3, t * 3 + 1])
                    .collect::<Vec<i32>>(),
            )?;
            Ok(Self {
                patches: bf(ctx, TOKENS, 72)?,
                output: bf(ctx, TOKENS, 64)?,
                final_global_ids,
                rotary: Rotary::new(ctx, w, ntk)?,
            })
        }
        /// Arena bytes required by one DiT request traversal (all DitScratch
        /// tensors coexist for the whole forward pass).
        pub(super) fn scratch_bytes(w: &DitWeights) -> usize {
            let core = TOKENS * WIDTH * 2 * 7 + TOKENS * 8192 * 2;
            let text_kv = 2 * 512 * WIDTH * 2 * w.blocks.len();
            core + text_kv
        }
        pub(super) fn scratch(&self, ctx: &CudaContext, w: &DitWeights) -> Result<DitScratch> {
            let a = |rows: usize, cols: usize| ctx.allocate_output(Shape::new(vec![rows, cols]), DType::BF16);
            Ok(DitScratch {
                hidden: a(TOKENS, WIDTH)?,
                normalized: a(TOKENS, WIDTH)?,
                q: a(TOKENS, WIDTH)?,
                k: a(TOKENS, WIDTH)?,
                v: a(TOKENS, WIDTH)?,
                attended: a(TOKENS, WIDTH)?,
                projected: a(TOKENS, WIDTH)?,
                mlp: a(TOKENS, 8192)?,
                text_kv: w
                    .blocks
                    .iter()
                    .map(|_| Ok([a(512, WIDTH)?, a(512, WIDTH)?]))
                    .collect::<Result<_>>()?,
            })
        }
        fn compute_modulation(
            &self,
            ctx: &CudaContext,
            w: &Modulation,
            final_layer: bool,
            time_activated: &Tensor,
            modulation_hidden: &Tensor,
            modulation: &Tensor,
            time_global: &Tensor,
            final_modulation: &Tensor,
            final_global: &Tensor,
            policy: &GemmPolicy,
        ) -> Result<()> {
            linear(ctx, time_activated, &w.down, modulation_hidden, policy)?;
            let (out, global) = if final_layer {
                (final_modulation, final_global)
            } else {
                (modulation, time_global)
            };
            linear(ctx, modulation_hidden, &w.up, out, policy)?;
            add(ctx, out, global, out)?;
            Ok(())
        }
        fn modulate(
            &self,
            ctx: &CudaContext,
            scratch: &DitScratch,
            modulation: &Tensor,
        ) -> Result<()> {
            let parts = modulation.numel() / (9 * WIDTH);
            let style = modulation.reshape(vec![9, parts, WIDTH])?;
            norm(
                ctx,
                NormArgs {
                    input: &scratch.hidden,
                    output: Some(&scratch.normalized),
                    kind: NormKind::Layer,
                    affine: Affine::GroupedAdaptive {
                        style: &style,
                        shift_first: true,
                    },
                    quantized: None,
                    numerics: NormNumerics::StagedBf16,
                    eps: 1e-6,
                },
            )
        }

        /// Refresh all text projections once per request, before the five
        /// denoising steps. This remains in the request graph so task updates
        /// cannot reuse projections from an earlier observation.
        pub(super) fn project_text(
            &self,
            ctx: &CudaContext,
            w: &DitWeights,
            text: &Tensor,
            text_kv: &[[Tensor; 2]],
            gemm: &GemmPolicy,
        ) -> Result<()> {
            for (block, [k, v]) in w.blocks.iter().zip(text_kv) {
                let w = &block.cross_attention;
                linear(ctx, text, &w.k, k, gemm)?;
                linear(ctx, text, &w.v, v, gemm)?;
                norm(
                    ctx,
                    NormArgs::rms(&matrix(k, 128)?, &w.k_norm, &mut matrix(k, 128)?, 1e-6),
                )?;
            }
            Ok(())
        }
        fn attend(
            &self,
            ctx: &CudaContext,
            scratch: &DitScratch,
            w: &Attention,
            text: Option<&[Tensor; 2]>,
            gemm: &GemmPolicy,
            policy: &AttentionPolicy,
        ) -> Result<()> {
            let (k, v) = match text {
                Some([k, v]) => (k, v),
                None => (&scratch.k, &scratch.v),
            };
            if text.is_none() {
                // Independent projections share the normalized input. Prepare V
                // first, then consume each Q/K projection immediately.
                linear(ctx, &scratch.normalized, &w.v, v, gemm)?;
                linear(ctx, &scratch.normalized, &w.q, &scratch.q, gemm)?;
                self.rotary.apply_normalized(ctx, &scratch.q, &w.q_norm)?;
                linear(ctx, &scratch.normalized, &w.k, k, gemm)?;
                self.rotary.apply_normalized(ctx, k, &w.k_norm)?;
            } else {
                linear(ctx, &scratch.normalized, &w.q, &scratch.q, gemm)?;
                norm(
                    ctx,
                    NormArgs::rms(
                        &matrix(&scratch.q, 128)?,
                        &w.q_norm,
                        &mut matrix(&scratch.q, 128)?,
                        1e-6,
                    ),
                )?;
            }
            let q = scratch.q.reshape(vec![1, TOKENS, 16, 128])?;
            let k = k.reshape(vec![1, k.numel() / WIDTH, 16, 128])?;
            let v = v.reshape(k.shape().clone())?;
            let mut out = scratch.attended.reshape(q.shape().clone())?;
            let mut args = AttentionArgs::new(&q, &k, &v, &mut out);
            args.policy = policy.clone();
            attention(ctx, args)?;
            linear(ctx, &scratch.attended, &w.output, &scratch.projected, gemm)
        }
        fn residual_modulate(
            &self,
            ctx: &CudaContext,
            scratch: &DitScratch,
            modulation: &Tensor,
            next: &Tensor,
        ) -> Result<()> {
            let gate = modulation.reshape(vec![9, 3, WIDTH])?;
            let style = next.reshape(vec![9, next.numel() / (9 * WIDTH), WIDTH])?;
            fused_add_norm(
                ctx,
                FusedAddNormArgs {
                    input: &scratch.projected,
                    bias: None,
                    residual: &scratch.hidden,
                    gate: Some(ResidualGate::GroupedStyle {
                        style: &gate,
                        numerics: NormNumerics::StagedBf16,
                    }),
                    hidden: &scratch.hidden,
                    normalized: Some(&scratch.normalized),
                    kind: NormKind::Layer,
                    affine: Affine::GroupedAdaptive {
                        style: &style,
                        shift_first: true,
                    },
                    quantized: None,
                    numerics: NormNumerics::StagedBf16,
                    bias_order: BiasOrder::SingleRound,
                    eps: 1e-6,
                },
            )
        }
        pub(super) fn prepare_modulation(
            &self,
            ctx: &CudaContext,
            w: &DitWeights,
            time: &Tensor,
            prepared: &StepModulation,
            gemm: &GemmPolicy,
        ) -> Result<()> {
            let time_hidden = bf(ctx, 9, WIDTH)?;
            let time_global = bf(ctx, 9, 6144)?;
            let time_normalized = bf(ctx, 9, WIDTH)?;
            let time_activated = bf(ctx, 9, WIDTH)?;
            let modulation_hidden = bf(ctx, 9, 256)?;
            let modulation = bf(ctx, 9, 6144)?;
            let final_modulation = bf(ctx, 9, 4096)?;
            let final_global = bf(ctx, 9, 4096)?;
            linear(ctx, time, &w.time_in, &time_hidden, gemm)?;
            activate(
                ctx,
                &time_hidden,
                None,
                PointwiseActivation::Silu,
                &time_hidden,
            )?;
            linear(ctx, &time_hidden, &w.time_out, &time_global, gemm)?;
            let mut time_normalized_view = time_normalized.clone();
            norm(ctx, NormArgs::rms(time, &w.time_norm, &mut time_normalized_view, 1e-6))?;
            activate(
                ctx,
                &time_normalized,
                None,
                PointwiseActivation::Silu,
                &time_activated,
            )?;
            gather(ctx, &time_global, &self.final_global_ids, WIDTH, &final_global)?;
            super::copy(ctx, &time_global, &prepared.time_global)?;
            for (block, styles) in w.blocks.iter().zip(&prepared.blocks) {
                for (weights, style) in [
                    &block.self_modulation,
                    &block.cross_modulation,
                    &block.mlp_modulation,
                ]
                .into_iter()
                .zip(styles)
                {
                    self.compute_modulation(
                        ctx,
                        weights,
                        false,
                        &time_activated,
                        &modulation_hidden,
                        &modulation,
                        &time_global,
                        &final_modulation,
                        &final_global,
                        gemm,
                    )?;
                    super::copy(ctx, &modulation, style)?;
                }
            }
            self.compute_modulation(
                ctx,
                &w.final_modulation,
                true,
                &time_activated,
                &modulation_hidden,
                &modulation,
                &time_global,
                &final_modulation,
                &final_global,
                gemm,
            )?;
            super::copy(ctx, &final_modulation, &prepared.final_style)
        }
        pub fn forward(
            &self,
            ctx: &CudaContext,
            w: &DitWeights,
            prepared: &StepModulation,
            scratch: &DitScratch,
            gemm: &GemmPolicy,
            attention: &AttentionPolicy,
        ) -> Result<()> {
            self.forward_observed(ctx, w, prepared, scratch, gemm, attention, |_, _| Ok(()))
        }
        pub fn forward_observed(
            &self,
            ctx: &CudaContext,
            w: &DitWeights,
            prepared: &StepModulation,
            scratch: &DitScratch,
            gemm: &GemmPolicy,
            attention: &AttentionPolicy,
            mut observe: impl FnMut(&str, &Tensor) -> Result<()>,
        ) -> Result<()> {
            linear(ctx, &self.patches, &w.patch, &scratch.hidden, gemm)?;
            observe("patch", &scratch.hidden)?;
            observe("time_lora", &prepared.time_global)?;
            self.modulate(ctx, &scratch, &prepared.blocks[0][0])?;
            for (index, block) in w.blocks.iter().enumerate() {
                self.attend(ctx, &scratch, &block.self_attention, None, gemm, attention)?;
                self.residual_modulate(
                    ctx,
                    &scratch,
                    &prepared.blocks[index][0],
                    &prepared.blocks[index][1],
                )?;
                self.attend(
                    ctx,
                    &scratch,
                    &block.cross_attention,
                    Some(&scratch.text_kv[index]),
                    gemm,
                    attention,
                )?;
                self.residual_modulate(
                    ctx,
                    &scratch,
                    &prepared.blocks[index][1],
                    &prepared.blocks[index][2],
                )?;
                linear_gelu_exact(ctx, &scratch.normalized, &block.mlp_up, &scratch.mlp, gemm)?;
                linear(ctx, &scratch.mlp, &block.mlp_down, &scratch.projected, gemm)?;
                let next = prepared
                    .blocks
                    .get(index + 1)
                    .map(|styles| &styles[0])
                    .unwrap_or(&prepared.final_style);
                self.residual_modulate(ctx, &scratch, &prepared.blocks[index][2], next)?;
                match index {
                    0 => observe("block0", &scratch.hidden)?,
                    13 => observe("block13", &scratch.hidden)?,
                    27 => observe("block27", &scratch.hidden)?,
                    _ => {}
                }
            }
            linear(
                ctx,
                &scratch.normalized,
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
            let prepared = StepModulation::allocate(&ctx, &weights)?;
            session.prepare(|| dit.prepare_modulation(&ctx, &weights, &time, &prepared, &gemm))?;
            let mut failures = Vec::new();
            session.prepare(|| {
            let scratch = dit.scratch(&ctx, &weights)?;
            dit.project_text(&ctx, &weights, &text, &scratch.text_kv, &gemm)?;
            dit.forward_observed(
                &ctx,
                &weights,
                &prepared,
                &scratch,
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
                session.run(|| {
                    let scratch = dit.scratch(&ctx, &weights)?;
                    dit.project_text(&ctx, &weights, &text, &scratch.text_kv, &gemm)?;
                    dit.forward(&ctx, &weights, &prepared, &scratch, &gemm, &attention)
                })
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
use self::{
    dit::{Dit, DitScratch, StepModulation},
    tensor_ops::*,
    vae::VaeChunk,
};
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
    modulation: StepModulation,
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
    /// Solver state. `joint` is the per-step joint latent (read by the action
    /// readout after the final step); `state`/`scaled_noise` are the FP32 views
    /// bound by the FP64 recurrence reset/update executions. They accumulate
    /// across the five denoising steps, so they are request-scoped state rather
    /// than per-step scratch and stay as fields.
    pub joint: Tensor,
    images: Tensor,
    video: Tensor,
    first: VaeChunk,
    prefix: VaeChunk,
    rest: VaeChunk,
    condition_source: Tensor,
    condition_map: Tensor,
    latent_input: Tensor,
    latent_view: Tensor,
    patch_map: Tensor,
    unpatch_map: Tensor,
    text: Tensor,
    dit: Dit,
    state: Tensor,
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
        convolution: &crate::cuda_cosmos_policy::backend::Conv3dPolicy,
        gemm: &GemmPolicy,
        attention: &AttentionPolicy,
    ) -> Result<Self> {
        let rgb = inputs.rgb.clone();
        let images = bf(&ctx, 2 * PIXELS, 3)?;
        let video = upload(
            &Tensor::from_bf16(
                vec![13, 224, 224, 3],
                &vec![half::bf16::NEG_ONE; 13 * PIXELS * 3],
            )?,
            ctx.device_id(),
        )?;
        // Frames 0..4 are the constant -1 padding the model never overwrites, so
        // their causal activations are computed once at construction. Only
        // frames 5..12 (the two camera views) are encoded per request.
        let first_input = slice(&video, 0, PIXELS * 3, vec![1, 224, 224, 3])?;
        let prefix_input = slice(&video, PIXELS * 3, 4 * PIXELS * 3, vec![4, 224, 224, 3])?;
        let rest_input = slice(&video, 5 * PIXELS * 3, 8 * PIXELS * 3, vec![8, 224, 224, 3])?;
        let mut first = VaeChunk::prepare(convolution, &ctx, &weights.vae, &first_input, 13, None)?;
        let mut prefix = VaeChunk::prepare(
            convolution,
            &ctx,
            &weights.vae,
            &prefix_input,
            13,
            Some(&first),
        )?;
        let rest = VaeChunk::prepare(
            convolution,
            &ctx,
            &weights.vae,
            &rest_input,
            13,
            Some(&prefix),
        )?;
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
        let text = bf(&ctx, 512, 1024)?;
        // Pinned LIBERO runtime: spatial extrapolation 3, temporal 1.
        let spatial_ntk = 3f64.powf(42. / 40.) as f32;
        let dit = Dit::prepare(&ctx, &weights.dit, [1., spatial_ntk, spatial_ntk])?;
        let state = alloc(&ctx, vec![1, LATENT], DType::F32)?;
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
                modulation: StepModulation::allocate(&ctx, &weights.dit)?,
                update,
            });
        }
        let actions = alloc(&ctx, vec![16, 7], DType::F32)?;
        let action_map: Vec<u32> = (0..FRAME)
            .map(|i| ((i / 784) * 9 * 784 + 4 * 784 + i % 784) as u32)
            .collect();
        let readout = PreparedIndexedMean::new(&ctx, &joint, &actions, &action_map)?;
        // Checkpoint-fixed constants: the -1 padding frames and the fixed
        // 2ab-schedule modulation never change per request, so they are
        // computed once here, outside the request graph, and reused by every
        // request traversal. This keeps plan preparation to a single
        // session.prepare(forward).
        first.forward(&ctx, &weights.vae, attention)?;
        prefix.forward(&ctx, &weights.vae, attention)?;
        for step in &steps {
            dit.prepare_modulation(&ctx, &weights.dit, &step.time, &step.modulation, gemm)?;
        }
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
            prefix,
            rest,
            condition_source,
            condition_map,
            latent_input,
            latent_view,
            patch_map,
            unpatch_map,
            text,
            dit,
            state,
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
        scratch: &DitScratch,
        condition: &Tensor,
        condition_bf16: &Tensor,
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
        self.rest.forward(ctx, &self.weights.vae, attention)?;
        copy(
            ctx,
            &self.first.output,
            &slice(&self.condition_source, 0, FRAME, vec![FRAME, 1])?,
        )?;
        // Latent 4 (frames 13..16) is never read by condition_map, so the encoder
        // stops after frame 12. Size the copy from the produced latents instead of
        // assuming how many latent frames causal downsampling yields.
        // Slots [FRAME, 2*FRAME) hold latent 1 from the constant prefix. Even
        // though `condition_map` substitutes proprioception at t == 1, the
        // chunk must still publish its latent: skipping the tail changes the
        // operator sequence the shared session sees, which shifts the buffer
        // offsets `rest` later observes and selects different vectorized
        // kernels. That is numerically visible (joint frame 2 relative L2
        // 0.46 versus 0.005), so the copy stays.
        let prefix_latents = self.prefix.output.numel();
        let rest_latents = self.rest.output.numel();
        assert!(
            prefix_latents == FRAME && rest_latents == 2 * FRAME,
            "unexpected latent geometry"
        );
        copy(
            ctx,
            &self.prefix.output,
            &slice(&self.condition_source, FRAME, prefix_latents, vec![prefix_latents, 1])?,
        )?;
        copy(
            ctx,
            &self.rest.output,
            &slice(&self.condition_source, 2 * FRAME, rest_latents, vec![rest_latents, 1])?,
        )?;
        cast(
            ctx,
            &self.proprio,
            &slice(&self.condition_source, 5 * FRAME, 9, vec![1, 9])?,
        )?;
        gather(ctx, &self.condition_source, &self.condition_map, 1, condition_bf16)?;
        cast(ctx, condition_bf16, condition)?;
        gather(ctx, &self.weights.text, &self.task, 512 * 1024, &self.text)?;
        self.dit.project_text(
            ctx,
            &self.weights.dit,
            &self.text,
            &scratch.text_kv,
            gemm,
        )
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
        scratch: &DitScratch,
        condition: &Tensor,
    ) -> Result<()> {
        let ctx = &self.ctx;
        let step = &self.steps[index];
        // Per-step temporaries from the session arena; dropped at step end.
        let step_scratch = ctx.allocate_output(Shape::new(vec![1, LATENT]), DType::F32)?;
        let network_output = ctx.allocate_output(Shape::new(vec![1, LATENT]), DType::BF16)?;
        multiply(ctx, &self.state, &step.input_scale, &step_scratch)?;
        add(ctx, &step_scratch, condition, &step_scratch)?;
        cast(ctx, &step_scratch, &self.latent_view)?;
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
            &step.modulation,
            scratch,
            gemm,
            attention,
        )?;
        gather(ctx, &self.dit.output, &self.unpatch_map, 1, &network_output)?;
        cast(ctx, &network_output, &step_scratch)?;
        multiply(ctx, &step_scratch, &step.output_scale, &step_scratch)?;
        multiply(ctx, &self.state, &step.input_scale, &self.joint)?;
        add(ctx, &self.joint, &step_scratch, &self.joint)?;
        add(ctx, &self.joint, condition, &self.joint)?;
        if let Some(update) = &step.update {
            update.enqueue()?;
        }
        Ok(())
    }
    pub fn finish(&self) -> Result<()> {
        self.readout.enqueue()
    }
    /// One request traversal: allocate the per-request DiT scratch from the
    /// session arena, then encode, denoise and read out.
    pub fn forward(&mut self, gemm: &GemmPolicy, attention: &AttentionPolicy) -> Result<()> {
        let scratch = self.dit.scratch(&self.ctx, &self.weights.dit)?;
        let condition = self
            .ctx
            .allocate_output(Shape::new(vec![1, LATENT]), DType::F32)?;
        let condition_bf16 = self
            .ctx
            .allocate_output(Shape::new(vec![1, LATENT]), DType::BF16)?;
        self.encode_observation(gemm, attention, &scratch, &condition, &condition_bf16)?;
        self.initial_state()?;
        for step in 0..self.steps.len() {
            self.denoise_step(step, gemm, attention, &scratch, &condition)?;
        }
        self.finish()
    }
    pub fn actions(&self) -> &Tensor {
        &self.actions
    }
    #[cfg(test)]
    pub fn joint(&self) -> &Tensor {
        &self.joint
    }
    pub fn workspace_bytes(&self) -> usize {
        // The session arena holds everything allocate_output hands out during
        // one request traversal: the whole-forward DiT scratch, the
        // request-scoped condition buffers, and the per-step scratch/network
        // temporaries.
        let dit = dit::Dit::scratch_bytes(&self.weights.dit);
        let condition = LATENT * 4 + LATENT * 2; // condition (F32) + condition_bf16 (BF16)
        let step = LATENT * 4 + LATENT * 2; // step_scratch (F32) + network_output (BF16)
        dit + condition + step
    }
}
