//! BF16 model blocks, including explicitly F32 residual and projection paths.
mod ops {
    //! Precision-specific projection, activation and rounding composition.
    use crate::cuda_fastwam::backend::{ops, Context};
    use crate::cuda_fastwam::weights::LinearWeights;
    use apxinf_core::{DType, Error, Result, Shape, Tensor};
    pub(in crate::cuda_fastwam::model::blocks::bf16) fn cast(
        ctx: &Context,
        x: &Tensor,
        dtype: DType,
    ) -> Result<Tensor> {
        if x.dtype() == dtype {
            return Ok(x.clone());
        }
        let semantic = match (x.dtype(), dtype) {
            (DType::BF16, DType::F32) => ops::QuantizationSemantic::CastBf16ToF32,
            (DType::F32, DType::BF16) => ops::QuantizationSemantic::CastF32ToBf16,
            _ => {
                return Err(Error::Other(
                    "Fast-WAM supports BF16/F32 conversion only".into(),
                ))
            }
        };
        let mut out = ctx.allocate_output(x.shape().clone(), dtype)?;
        ops::quantization(ctx, ops::QuantizationArgs::new(semantic, x, &mut out))?;
        Ok(out)
    }

    pub(in crate::cuda_fastwam::model::blocks::bf16) fn reshape(
        x: &Tensor,
        dims: Vec<usize>,
    ) -> Result<Tensor> {
        x.reshape(Shape::new(dims))
    }

    pub(in crate::cuda_fastwam::model::blocks::bf16) fn linear(
        ctx: &Context,
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        policy: &ops::GemmPolicy,
    ) -> Result<Tensor> {
        let x = cast(ctx, x, weight.dtype())?;
        let dims = x.shape().dims();
        if dims.len() != 2
            || weight.shape().dims().len() != 2
            || dims[1] != weight.shape().dims()[0]
        {
            return Err(Error::Other(format!(
                "Fast-WAM linear mismatch {:?} @ {:?}",
                x.shape(),
                weight.shape()
            )));
        }
        let mut out = ctx.allocate_output(
            Shape::new(vec![dims[0], weight.shape().dims()[1]]),
            weight.dtype(),
        )?;
        let mut args = ops::GemmArgs::new(&x, weight, &mut out);
        args.policy = policy.clone();
        if let Some(bias) = bias {
            ops::gemm_bias(ctx, ops::GemmBiasArgs { gemm: args, bias })?;
        } else {
            ops::gemm(ctx, args)?;
        }
        Ok(out)
    }

    /// PyTorch's cuDNN convolution materializes the projection in its output
    /// dtype before adding bias. Keep this rounding explicit in convolution composition.
    pub(in crate::cuda_fastwam::model::blocks::bf16) fn convolution_projection(
        ctx: &Context,
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        policy: &ops::GemmPolicy,
    ) -> Result<Tensor> {
        let projection = linear(ctx, x, weight, None, policy)?;
        let Some(bias) = bias else {
            return Ok(projection);
        };
        let mut out = ctx.allocate_output(projection.shape().clone(), projection.dtype())?;
        let mut args = ops::PointwiseArgs::new(
            ops::PointwiseSemantic::BiasActivation,
            &projection,
            &mut out,
        );
        args.bias = Some(bias);
        ops::pointwise(ctx, args)?;
        Ok(out)
    }

    pub(in crate::cuda_fastwam::model::blocks::bf16) fn activation(
        ctx: &Context,
        x: &Tensor,
        activation: ops::PointwiseActivation,
    ) -> Result<Tensor> {
        let mut out = ctx.allocate_output(x.shape().clone(), x.dtype())?;
        let mut args = ops::PointwiseArgs::new(ops::PointwiseSemantic::BiasActivation, x, &mut out);
        args.activation = activation;
        ops::pointwise(ctx, args)?;
        Ok(out)
    }

    pub(in crate::cuda_fastwam::model::blocks::bf16) fn add(
        ctx: &Context,
        x: &Tensor,
        y: &Tensor,
    ) -> Result<Tensor> {
        let mut out = ctx.allocate_output(x.shape().clone(), x.dtype())?;
        let mut args = ops::PointwiseArgs::new(ops::PointwiseSemantic::EulerUpdate, x, &mut out);
        args.secondary = Some(y);
        args.dt = 1.0;
        ops::pointwise(ctx, args)?;
        Ok(out)
    }

    pub(in crate::cuda_fastwam::model::blocks::bf16) fn bias_add(
        ctx: &Context,
        x: &Tensor,
        bias: &Tensor,
    ) -> Result<Tensor> {
        let mut out = ctx.allocate_output(x.shape().clone(), x.dtype())?;
        let mut args = ops::PointwiseArgs::new(ops::PointwiseSemantic::BiasActivation, x, &mut out);
        args.bias = Some(bias);
        ops::pointwise(ctx, args)?;
        Ok(out)
    }

    pub(in crate::cuda_fastwam::model::blocks::bf16) fn binary(
        ctx: &Context,
        x: &Tensor,
        y: &Tensor,
        semantic: ops::PointwiseSemantic,
    ) -> Result<Tensor> {
        let mut out = ctx.allocate_output(x.shape().clone(), x.dtype())?;
        let mut args = ops::PointwiseArgs::new(semantic, x, &mut out);
        args.secondary = Some(y);
        ops::pointwise(ctx, args)?;
        Ok(out)
    }

    pub(in crate::cuda_fastwam::model::blocks::bf16) fn project_conv(
        weights: &LinearWeights,
        ctx: &Context,
        x: &Tensor,
        policy: &ops::GemmPolicy,
    ) -> Result<Tensor> {
        convolution_projection(ctx, x, &weights.weight, weights.bias.as_ref(), policy)
    }

    pub(in crate::cuda_fastwam::model::blocks::bf16) fn project(
        weights: &LinearWeights,
        ctx: &Context,
        x: &Tensor,
        policy: &ops::GemmPolicy,
    ) -> Result<Tensor> {
        linear(ctx, x, &weights.weight, weights.bias.as_ref(), policy)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::cuda_fastwam::backend::{transfers, upload};
        use crate::cuda_fastwam::weights::host_tensor;
        use apxinf_core::DType;

        #[test]
        #[ignore = "requires CUDA"]
        fn fastwam_convolution_matches_baseline_bias_rounding() {
            let ctx = Context::new(0).unwrap();
            let upload_values = |shape, values: &[f32]| {
                upload(&ctx, &host_tensor(shape, values, DType::BF16).unwrap()).unwrap()
            };
            let input = upload_values(vec![1, 2], &[1., 1.]);
            let weight = upload_values(vec![2, 1], &[1., 0.00390625]);
            let bias = upload_values(vec![1], &[-1.]);
            let policy = ops::GemmPolicy::default();
            // 1 + 2^-8 is the BF16 halfway point: convolution rounds it to 1
            // before cancellation, matching the baseline GemmBias path.
            let conv = convolution_projection(&ctx, &input, &weight, Some(&bias), &policy).unwrap();
            let linear = linear(&ctx, &input, &weight, Some(&bias), &policy).unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(
                transfers::to_cpu(&conv).unwrap().to_f32_vec().unwrap(),
                vec![0.]
            );
            assert_eq!(
                transfers::to_cpu(&linear).unwrap().to_f32_vec().unwrap(),
                vec![0.]
            );
        }
    }
}
mod dit {
    //! Wan video prefill and action denoising. Weights and cached text projections
    //! are family local; backend calls express only mathematical operations.
    use super::ops as compute;
    use super::ops::cast;
    use super::ops::{bias_add, binary};
    use crate::cuda_fastwam::backend::{ops, Context};
    use crate::cuda_fastwam::weights::{ExpertWeights, ProjectionWeights};
    pub(in crate::cuda_fastwam::model::blocks::bf16) struct Expert {
        pub weights: ExpertWeights,
        /// [step][block][shift/scale/gate], computed once during construction.
        pub modulations: Vec<Vec<Vec<Tensor>>>,
    }

    use apxinf_core::{DType, Result, Shape, Tensor};

    // Preserve residual state across blocks without repeated BF16 rounding. Branch
    // projections and gates retain their checkpoint dtype and round before the add.
    fn residual(ctx: &Context, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        compute::add(ctx, x, &cast(ctx, y, DType::F32)?)
    }
    fn norm(ctx: &Context, x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let mut out = ctx.allocate_output(x.shape().clone(), x.dtype())?;
        ops::layer_norm(
            ctx,
            ops::LayerNormArgs::new(x, weight, bias, &mut out, 1e-6),
        )?;
        cast(ctx, &out, DType::BF16)
    }
    fn rms(ctx: &Context, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let mut out = ctx.allocate_output(x.shape().clone(), x.dtype())?;
        ops::rms_norm(ctx, ops::RmsNormArgs::new(x, weight, &mut out, 1e-6))?;
        Ok(out)
    }
    fn attend(
        ctx: &Context,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        policy: &ops::AttentionPolicy,
    ) -> Result<Tensor> {
        let q = cast(ctx, q, DType::F32)?;
        let k = cast(ctx, k, DType::F32)?;
        let v = cast(ctx, v, DType::F32)?;
        let (t, s) = (q.shape().dims()[0], k.shape().dims()[0]);
        let q = q.reshape(vec![1, t, 24, 128])?;
        let k = k.reshape(vec![1, s, 24, 128])?;
        let v = v.reshape(vec![1, s, 24, 128])?;
        let mut out = ctx.allocate_output(q.shape().clone(), DType::F32)?;
        let mut args = ops::AttentionArgs::new(&q, &k, &v, &mut out);
        args.policy = policy.clone();
        ops::attention(ctx, args)?;
        out.reshape(vec![t, 3072])
    }

    fn project_kv(
        weights: &ProjectionWeights,
        ctx: &Context,
        x: &Tensor,
        policy: &ops::GemmPolicy,
    ) -> Result<(Tensor, Tensor)> {
        Ok((
            rms(
                ctx,
                &compute::project(&weights.k, ctx, x, policy)?,
                &weights.nk,
            )?,
            compute::project(&weights.v, ctx, x, policy)?,
        ))
    }

    pub(in crate::cuda_fastwam::model::blocks::bf16) struct Kv {
        key: Tensor,
        value: Tensor,
    }
    pub(in crate::cuda_fastwam::model::blocks::bf16) struct ContextCache {
        // Request-local F32 copies of the rounded BF16 projections. All denoising
        // steps consume the same exact promotion; attention need not recast them.
        kv: Vec<Kv>,
    }

    impl Expert {
        /// Conservative sum of transient tensors for this fixed expert profile.
        /// Repeated action-step budgets also reserve context space, a safe overestimate.
        /// Native execution-owned scratch is separate from the tensor arena.
        pub(in crate::cuda_fastwam::model::blocks::bf16) fn workspace_bytes(&self) -> usize {
            let tokens = if self.weights.video { 98 } else { 32 };
            let prefix = if self.weights.video { 0 } else { 98 };
            let hidden = self.weights.input.weight.shape().dims()[1];
            let attention_width = 24 * 128;
            let context = 129;
            // Text projection/activation and per-layer K, normalized K and V.
            let context_bytes = context * hidden * 6
                + self.weights.blocks.len() * (context * attention_width * 6 + 3 * 256);
            let layer_bytes: usize = self
                .weights
                .blocks
                .iter()
                .map(|b| {
                    let ff = b.ff0.weight.shape().dims()[1];
                    // Residual/normalization/modulation outputs: <= 16 BF16 hidden
                    // tensors; FFN projection and activation: two BF16 FF tensors.
                    let hidden_bytes = tokens * (32 * hidden + 4 * ff);
                    // Q/K/V, rotary, attention outputs and BF16/F32 conversions.
                    // Includes concatenated video prefix and context FP32 K/V.
                    let attention_bytes =
                        attention_width * (tokens * 48 + (tokens + prefix) * 12 + context * 8);
                    // Reserve F32 residual/normalization storage and dtype conversions.
                    2 * (hidden_bytes + attention_bytes + 48 * 256)
                })
                .sum();
            let input_bytes = tokens * hidden * self.weights.input.weight.dtype().size_in_bytes()
                + 392 * 48 * 2
                + 98 * 192 * 2
                + if self.weights.video {
                    tokens * hidden * 2 + 256
                } else {
                    0
                }; // convolution bias add
            let head_bytes = self.weights.head.as_ref().map_or(0, |h| {
                tokens * h.weight.shape().dims()[1] * h.weight.dtype().size_in_bytes()
            });
            context_bytes + layer_bytes + input_bytes + head_bytes + 16 * 256
        }

        pub(in crate::cuda_fastwam::model::blocks::bf16) fn context(
            &self,
            ctx: &Context,
            context: &Tensor,
            policy: &ops::GemmPolicy,
        ) -> Result<ContextCache> {
            let x = compute::project(&self.weights.text0, ctx, context, policy)?;
            let x = compute::activation(ctx, &x, ops::PointwiseActivation::Gelu)?;
            let x = compute::project(&self.weights.text2, ctx, &x, policy)?;
            let kv = self
                .weights
                .blocks
                .iter()
                .map(|b| {
                    let (key, value) = project_kv(&b.ca, ctx, &x, policy)?;
                    Ok(Kv {
                        key: cast(ctx, &key, DType::F32)?,
                        value: cast(ctx, &value, DType::F32)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(ContextCache { kv })
        }
        fn rotate(&self, ctx: &Context, x: &Tensor) -> Result<Tensor> {
            let mut out = ctx.allocate_output(
                x.shape().clone(),
                if self.weights.video {
                    DType::F32
                } else {
                    DType::BF16
                },
            )?;
            ops::pairwise_rope(
                ctx,
                ops::PairwiseRopeArgs {
                    input: x,
                    table: &self.weights.rope,
                    output: &mut out,
                    heads: 24,
                },
            )?;
            Ok(out)
        }
        pub(in crate::cuda_fastwam::model::blocks::bf16) fn forward(
            &self,
            ctx: &Context,
            input: &Tensor,
            context: &ContextCache,
            video_cache: Option<&[Kv]>,
            step: usize,
            gemm: &ops::GemmPolicy,
            attention: &ops::AttentionPolicy,
        ) -> Result<(Tensor, Vec<Kv>)> {
            let input = if self.weights.video {
                let input = cast(ctx, input, DType::BF16)?;
                let mut patches = ctx.allocate_output(Shape::new(vec![98, 192]), DType::BF16)?;
                ops::image_im2col(ctx, &input, &mut patches, 14, 28, 2, 2, 0, 7, 14)?;
                patches
            } else {
                input.clone()
            };
            let x = if self.weights.video {
                compute::project_conv(&self.weights.input, ctx, &input, gemm)?
            } else {
                compute::project(&self.weights.input, ctx, &input, gemm)?
            };
            let mut x = cast(ctx, &x, DType::F32)?;
            let mut cache = Vec::new();
            for (i, b) in self.weights.blocks.iter().enumerate() {
                let m = &self.modulations[step][i];
                let u = norm(ctx, &x, &self.weights.one, &self.weights.zero)?;
                let u = binary(ctx, &u, &m[1], ops::PointwiseSemantic::BroadcastMultiply)?;
                let u = bias_add(ctx, &u, &m[0])?;
                let q = self.rotate(
                    ctx,
                    &rms(ctx, &compute::project(&b.sa.q, ctx, &u, gemm)?, &b.sa.nq)?,
                )?;
                let k = self.rotate(
                    ctx,
                    &rms(ctx, &compute::project(&b.sa.k, ctx, &u, gemm)?, &b.sa.nk)?,
                )?;
                let v = compute::project(&b.sa.v, ctx, &u, gemm)?;
                let (k, v) = if let Some(prefix) = video_cache {
                    (
                        ops::concat_rows(ctx, &prefix[i].key, &cast(ctx, &k, DType::F32)?)?,
                        ops::concat_rows(ctx, &prefix[i].value, &v)?,
                    )
                } else {
                    cache.push(Kv {
                        key: k.clone(),
                        value: v.clone(),
                    });
                    (k, v)
                };
                let y = attend(ctx, &q, &k, &v, attention)?;
                let y = compute::project(&b.sa.o, ctx, &y, gemm)?;
                let y = binary(ctx, &y, &m[2], ops::PointwiseSemantic::BroadcastMultiply)?;
                x = residual(ctx, &x, &y)?;
                let u = norm(ctx, &x, &b.norm3w, &b.norm3b)?;
                let q = rms(ctx, &compute::project(&b.ca.q, ctx, &u, gemm)?, &b.ca.nq)?;
                let y = attend(ctx, &q, &context.kv[i].key, &context.kv[i].value, attention)?;
                x = residual(ctx, &x, &compute::project(&b.ca.o, ctx, &y, gemm)?)?;
                let u = norm(ctx, &x, &self.weights.one, &self.weights.zero)?;
                let u = binary(ctx, &u, &m[4], ops::PointwiseSemantic::BroadcastMultiply)?;
                let u = bias_add(ctx, &u, &m[3])?;
                let u = compute::project(&b.ff0, ctx, &u, gemm)?;
                let u = compute::activation(ctx, &u, ops::PointwiseActivation::Gelu)?;
                let u = compute::project(&b.ff2, ctx, &u, gemm)?;
                let u = binary(ctx, &u, &m[5], ops::PointwiseSemantic::BroadcastMultiply)?;
                x = residual(ctx, &x, &u)?;
            }
            if let Some(head) = &self.weights.head {
                x = compute::project(head, ctx, &x, gemm)?;
            }
            Ok((x, cache))
        }
    }
}
mod text {
    //! UMT5 encoder: bidirectional relative bias, rounded RMS normalization and
    //! gated GELU. Zero padding is applied after encoding, as in Fast-WAM.
    use super::ops as compute;
    use crate::cuda_fastwam::backend::{ops, Context};
    use crate::cuda_fastwam::weights::TextEncoderWeights as TextWeights;
    use apxinf_core::{DType, Result, Shape, Tensor};

    fn rms(ctx: &Context, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let mut out = ctx.allocate_output(x.shape().clone(), x.dtype())?;
        let mut args = ops::RmsNormArgs::new(x, w, &mut out, 1e-6);
        args.intermediate_dtype = DType::BF16;
        ops::rms_norm(ctx, args)?;
        Ok(out)
    }
    pub(in crate::cuda_fastwam::model::blocks::bf16) struct TextEncoder {
        weights: TextWeights,
    }
    impl TextEncoder {
        pub(in crate::cuda_fastwam::model::blocks::bf16) fn new(weights: TextWeights) -> Self {
            Self { weights }
        }

        /// Monotonic graph arena: all layer outputs remain allocated until replay ends.
        pub(in crate::cuda_fastwam::model::blocks::bf16) fn workspace_bytes(&self) -> usize {
            let tokens = 128;
            let hidden = self.weights.embedding.shape().dims()[1];
            let layers: usize = self
                .weights
                .blocks
                .iter()
                .map(|b| {
                    let ff = b.wi0.weight.shape().dims()[1];
                    // Ten hidden tensors, four FFN tensors and one F32 attention bias.
                    tokens * (10 * hidden + 4 * ff) * 2 + 64 * tokens * tokens * 4 + 16 * 256
                })
                .sum();
            // Embedding, final RMS and row mask outputs.
            layers + 3 * (tokens * hidden * 2 + 256)
        }

        pub(in crate::cuda_fastwam::model::blocks::bf16) fn forward(
            &self,
            ctx: &Context,
            ids: &Tensor,
            padding_bias: &Tensor,
            row_mask: &Tensor,
            gemm: &ops::GemmPolicy,
            attention: &ops::AttentionPolicy,
        ) -> Result<Tensor> {
            self.forward_observed(
                ctx,
                ids,
                padding_bias,
                row_mask,
                gemm,
                attention,
                |_, _, _| Ok(()),
            )
        }

        fn forward_observed(
            &self,
            ctx: &Context,
            ids: &Tensor,
            padding_bias: &Tensor,
            row_mask: &Tensor,
            gemm: &ops::GemmPolicy,
            attention: &ops::AttentionPolicy,
            mut observe: impl FnMut(usize, &str, &Tensor) -> Result<()>,
        ) -> Result<Tensor> {
            let mut x = ctx.allocate_output(Shape::new(vec![128, 4096]), DType::BF16)?;
            ops::embedding_gather(ctx, &self.weights.embedding, ids, &x)?;
            for (index, b) in self.weights.blocks.iter().enumerate() {
                observe(index, "layer.0.layer_norm.in", &x)?;
                let u = rms(ctx, &x, &b.norm0)?;
                observe(index, "layer.0.layer_norm.out", &u)?;
                let q = compute::project(&b.q, ctx, &u, gemm)?.reshape(vec![1, 128, 64, 64])?;
                let k = compute::project(&b.k, ctx, &u, gemm)?.reshape(vec![1, 128, 64, 64])?;
                let v = compute::project(&b.v, ctx, &u, gemm)?.reshape(vec![1, 128, 64, 64])?;
                observe(index, "layer.0.SelfAttention.q.out", &q)?;
                observe(index, "layer.0.SelfAttention.k.out", &k)?;
                observe(index, "layer.0.SelfAttention.v.out", &v)?;
                let bias = compute::bias_add(ctx, &b.bias, padding_bias)?
                    .reshape(vec![1, 64, 128, 128])?;
                let mut y = ctx.allocate_output(q.shape().clone(), DType::BF16)?;
                let mut args = ops::AttentionArgs::new(&q, &k, &v, &mut y);
                args.scale = 1.;
                args.policy = attention.clone();
                ops::additive_attention(
                    ctx,
                    ops::AttentionBiasArgs {
                        intermediate_dtype: DType::BF16,
                        attention: args,
                        bias: &bias,
                    },
                )?;
                observe(index, "layer.0.SelfAttention.o.in", &y)?;
                let y = compute::project(&b.o, ctx, &y.reshape(vec![128, 4096])?, gemm)?;
                observe(index, "layer.0.SelfAttention.o.out", &y)?;
                x = compute::add(ctx, &x, &y)?;
                observe(index, "layer.1.layer_norm.in", &x)?;
                let u = rms(ctx, &x, &b.norm1)?;
                observe(index, "layer.1.layer_norm.out", &u)?;
                let gate = compute::project(&b.wi0, ctx, &u, gemm)?;
                observe(index, "layer.1.DenseReluDense.act.in", &gate)?;
                let gate = compute::activation(ctx, &gate, ops::PointwiseActivation::Gelu)?;
                observe(index, "layer.1.DenseReluDense.act.out", &gate)?;
                let up = compute::project(&b.wi1, ctx, &u, gemm)?;
                observe(index, "layer.1.DenseReluDense.wi_1.out", &up)?;
                let y = compute::binary(ctx, &gate, &up, ops::PointwiseSemantic::Multiply)?;
                observe(index, "layer.1.DenseReluDense.wo.in", &y)?;
                let y = compute::project(&b.wo, ctx, &y, gemm)?;
                observe(index, "layer.1.DenseReluDense.wo.out", &y)?;
                x = compute::add(ctx, &x, &y)?;
            }
            observe(self.weights.blocks.len(), "final_layer_norm.in", &x)?;
            let x = rms(ctx, &x, &self.weights.norm)?;
            observe(self.weights.blocks.len(), "final_layer_norm.out", &x)?;
            compute::binary(ctx, &x, row_mask, ops::PointwiseSemantic::RowMultiply)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::cuda_fastwam::backend;
        use crate::cuda_fastwam::weights::host_tensor;
        use std::collections::HashMap;

        // Use a captured input for each operation so an earlier approximation cannot
        // hide which operation first diverges. These diagnostics do not relax or
        // replace the end-to-end assertion below.
        fn captured_input(ctx: &Context, golden: &HashMap<String, Tensor>, key: &str) -> Tensor {
            let value = &golden[key];
            assert_eq!(value.shape().numel() % 128, 0, "{key}");
            backend::upload(ctx, value)
                .unwrap()
                .reshape(vec![128, value.shape().numel() / 128])
                .unwrap()
        }

        fn print_operator_error(name: &str, actual: &Tensor, expected: &Tensor) {
            let actual = backend::transfers::to_cpu(actual)
                .unwrap()
                .to_f32_vec()
                .unwrap();
            let expected = expected.to_f32_vec().unwrap();
            assert_eq!(actual.len(), expected.len(), "{name}");
            let mut square = 0f64;
            let mut norm = 0f64;
            let mut max = 0f32;
            for (&a, &b) in actual.iter().zip(&expected) {
                assert!(a.is_finite() && b.is_finite(), "nonfinite {name}");
                square += (f64::from(a) - f64::from(b)).powi(2);
                norm += f64::from(b).powi(2);
                max = max.max((a - b).abs());
            }
            let relative = (square / norm.max(1e-30)).sqrt();
            eprintln!("UMT5 independent {name}: max_abs={max} relative_l2={relative}");
        }

        fn replay_captured_operators(
            ctx: &Context,
            model: &TextEncoder,
            golden: &HashMap<String, Tensor>,
            padding: &Tensor,
            gemm: &ops::GemmPolicy,
            attention: &ops::AttentionPolicy,
        ) {
            for index in [0, 23] {
                let block = &model.weights.blocks[index];
                let prefix = format!("text.model.encoder.block.{index}");
                let attn_prefix = format!("{prefix}.layer.0.SelfAttention");
                let ff_prefix = format!("{prefix}.layer.1.DenseReluDense");
                for (layer, weight) in [(0, &block.norm0), (1, &block.norm1)] {
                    let name = format!("{prefix}.layer.{layer}.layer_norm");
                    let input = captured_input(ctx, golden, &format!("{name}.in"));
                    let actual = rms(ctx, &input, weight).unwrap();
                    print_operator_error(&name, &actual, &golden[&format!("{name}.out")]);
                }
                for (name, linear) in [
                    (format!("{attn_prefix}.q"), &block.q),
                    (format!("{attn_prefix}.k"), &block.k),
                    (format!("{attn_prefix}.v"), &block.v),
                    (format!("{attn_prefix}.o"), &block.o),
                    (format!("{ff_prefix}.wi_0"), &block.wi0),
                    (format!("{ff_prefix}.wi_1"), &block.wi1),
                    (format!("{ff_prefix}.wo"), &block.wo),
                ] {
                    let input = captured_input(ctx, golden, &format!("{name}.in"));
                    let actual = compute::project(linear, ctx, &input, gemm).unwrap();
                    print_operator_error(&name, &actual, &golden[&format!("{name}.out")]);
                }
                let name = format!("{ff_prefix}.act");
                let input = captured_input(ctx, golden, &format!("{name}.in"));
                let actual =
                    compute::activation(ctx, &input, ops::PointwiseActivation::Gelu).unwrap();
                print_operator_error(&name, &actual, &golden[&format!("{name}.out")]);

                let gate = captured_input(ctx, golden, &format!("{name}.out"));
                let up = captured_input(ctx, golden, &format!("{ff_prefix}.wi_1.out"));
                let actual =
                    compute::binary(ctx, &gate, &up, ops::PointwiseSemantic::Multiply).unwrap();
                print_operator_error(
                    &format!("{ff_prefix}.gated_multiply"),
                    &actual,
                    &golden[&format!("{ff_prefix}.wo.in")],
                );

                let qkv = ["q", "k", "v"].map(|projection| {
                    captured_input(ctx, golden, &format!("{attn_prefix}.{projection}.out"))
                        .reshape(vec![1, 128, 64, 64])
                        .unwrap()
                });
                let bias = compute::bias_add(ctx, &block.bias, padding)
                    .unwrap()
                    .reshape(vec![1, 64, 128, 128])
                    .unwrap();
                let mut actual = ctx
                    .allocate_output(qkv[0].shape().clone(), DType::BF16)
                    .unwrap();
                let mut args = ops::AttentionArgs::new(&qkv[0], &qkv[1], &qkv[2], &mut actual);
                args.scale = 1.;
                args.policy = attention.clone();
                ops::additive_attention(
                    ctx,
                    ops::AttentionBiasArgs {
                        intermediate_dtype: DType::BF16,
                        attention: args,
                        bias: &bias,
                    },
                )
                .unwrap();
                print_operator_error(
                    &format!("{attn_prefix}.attention"),
                    &actual,
                    &golden[&format!("{attn_prefix}.o.in")],
                );
            }
        }

        #[test]
        #[ignore = "requires UMT5 checkpoint, pinned Fast-WAM capture and CUDA"]
        fn fastwam_text_checkpoint_precision() {
            let path = std::path::PathBuf::from(std::env::var("APXINF_FASTWAM_TEXT").unwrap());
            let fixture =
                std::path::PathBuf::from(std::env::var("APXINF_FASTWAM_CAPTURE").unwrap());
            let (inputs, _) =
                apxinf_loader::safetensors::load_native_path(&fixture.join("inputs.safetensors"))
                    .unwrap();
            let (golden, _) =
                apxinf_loader::safetensors::load_native_path(&fixture.join("text.safetensors"))
                    .unwrap();
            let ctx = Context::new(0).unwrap();
            let model = TextEncoder::new(TextWeights::load(&ctx, &path).unwrap());
            let ids = Tensor::from_i32(
                vec![128],
                &inputs["inputs.ids"]
                    .to_f32_vec()
                    .unwrap()
                    .iter()
                    .map(|&x| x as i32)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let ids = backend::upload(&ctx, &ids).unwrap();
            let mask = inputs["inputs.mask"].to_f32_vec().unwrap();
            let padding = backend::upload(
                &ctx,
                &Tensor::from_f32(
                    vec![128],
                    &mask
                        .iter()
                        .map(|&x| if x == 0. { -3.3895314e38 } else { 0. })
                        .collect::<Vec<_>>(),
                )
                .unwrap(),
            )
            .unwrap();
            let rowmask =
                backend::upload(&ctx, &host_tensor(vec![128], &mask, DType::BF16).unwrap())
                    .unwrap();
            let mut gemm = ops::GemmPolicy::default();
            gemm.online_tune = false;
            gemm.cache_dir = Some(String::new());
            let mut attention = ops::AttentionPolicy::default();
            attention.online_tune = false;
            attention.cache_dir = Some(String::new());
            replay_captured_operators(&ctx, &model, &golden, &padding, &gemm, &attention);
            let actual = model
                .forward_observed(
                    &ctx,
                    &ids,
                    &padding,
                    &rowmask,
                    &gemm,
                    &attention,
                    |index, suffix, tensor| {
                        let key = if index == 24 {
                            format!("text.model.encoder.{suffix}")
                        } else {
                            format!("text.model.encoder.block.{index}.{suffix}")
                        };
                        if let Some(expected) = golden.get(&key) {
                            print_operator_error(&format!("chain {key}"), tensor, expected);
                        }
                        Ok(())
                    },
                )
                .unwrap();
            let actual = backend::transfers::to_cpu(&actual)
                .unwrap()
                .to_f32_vec()
                .unwrap();
            let mut expected = golden["text.model.encoder.final_layer_norm.out"]
                .to_f32_vec()
                .unwrap();
            for (row, &keep) in expected.chunks_exact_mut(4096).zip(&mask) {
                if keep == 0. {
                    row.fill(0.);
                }
            }
            let mut square = 0f64;
            let mut norm = 0f64;
            let mut actual_norm = 0f64;
            let mut dot = 0f64;
            let mut max = 0f32;
            for (&a, &b) in actual.iter().zip(&expected) {
                assert!(a.is_finite());
                square += ((a - b) as f64).powi(2);
                norm += (b as f64).powi(2);
                actual_norm += (a as f64).powi(2);
                dot += a as f64 * b as f64;
                max = max.max((a - b).abs());
            }
            let relative = (square / norm).sqrt();
            let cosine = dot / (actual_norm * norm).sqrt();
            eprintln!("UMT5 final max_abs={max} relative_l2={relative} cosine_similarity={cosine}");
            assert!(relative < 0.01, "UMT5 native/reference mismatch");
        }
    }
}
mod vae {
    //! Wan 2.2 encoder for a single observed frame. Temporal causal convolutions
    //! use their final temporal tap because the cache is empty for every request.
    //! This module never keeps a video cache or invokes a reference framework.
    use super::ops as compute;
    use super::ops::cast;
    use crate::cuda_fastwam::backend::{ops, Context};
    use crate::cuda_fastwam::weights::{
        AttentionWeights, ConvWeights, DownBlockWeights, NormWeights, ResidualWeights, VaeWeights,
    };
    use apxinf_core::{DType, Error, Result, Shape, Tensor};

    // Each arena allocation can consume at most 255 leading alignment bytes.
    // Keeping 256 per allocation also covers an aligned final capacity boundary.
    fn allocation_bytes(elements: usize, dtype: DType) -> usize {
        elements * dtype.size_in_bytes() + 256
    }

    fn conv_workspace_bytes(weights: &ConvWeights, h: usize, w: usize) -> usize {
        let rows = (h / weights.stride) * (w / weights.stride);
        let weight = weights.projection.weight.shape().dims();
        let projection = allocation_bytes(rows * weight[1], weights.projection.weight.dtype());
        2 * projection // dtype-rounded GEMM result plus separately rounded bias add
            + if weights.kernel == 1 && weights.stride == 1 {
                0
            } else {
                allocation_bytes(rows * weight[0], weights.projection.weight.dtype())
            }
    }
    fn conv_forward(
        weights: &ConvWeights,
        ctx: &Context,
        x: &Tensor,
        h: usize,
        w: usize,
        policy: &ops::GemmPolicy,
    ) -> Result<Tensor> {
        if weights.kernel == 1 && weights.stride == 1 {
            return compute::project_conv(&weights.projection, ctx, x, policy);
        }
        let (oh, ow) = (h / weights.stride, w / weights.stride);
        let mut columns = ctx.allocate_output(
            Shape::new(vec![oh * ow, weights.projection.weight.shape().dims()[0]]),
            x.dtype(),
        )?;
        ops::image_im2col(
            ctx,
            x,
            &mut columns,
            h,
            w,
            weights.kernel,
            weights.stride,
            weights.padding,
            oh,
            ow,
        )?;
        compute::project_conv(&weights.projection, ctx, &columns, policy)
    }

    fn norm_forward(weights: &NormWeights, ctx: &Context, x: &Tensor) -> Result<Tensor> {
        let mut out = ctx.allocate_output(x.shape().clone(), x.dtype())?;
        ops::channel_l2_normalize(
            ctx,
            x,
            &weights.gamma,
            &mut out,
            (x.shape().dims()[1] as f32).sqrt(),
            1e-12,
        )?;
        Ok(out)
    }
    fn norm_activated(weights: &NormWeights, ctx: &Context, x: &Tensor) -> Result<Tensor> {
        compute::activation(
            ctx,
            &norm_forward(weights, ctx, x)?,
            ops::PointwiseActivation::Silu,
        )
    }

    fn residual_workspace_bytes(weights: &ResidualWeights, h: usize, w: usize) -> usize {
        let input_channels = weights.conv1.projection.weight.shape().dims()[0]
            / (weights.conv1.kernel * weights.conv1.kernel);
        let output_channels = weights.conv1.projection.weight.shape().dims()[1];
        let dtype = weights.conv1.projection.weight.dtype();
        weights
            .shortcut
            .as_ref()
            .map_or(0, |c| conv_workspace_bytes(c, h, w))
            + 2 * allocation_bytes(h * w * input_channels, dtype)
            + conv_workspace_bytes(&weights.conv1, h, w)
            + 2 * allocation_bytes(h * w * output_channels, dtype)
            + conv_workspace_bytes(&weights.conv2, h, w)
            + allocation_bytes(h * w * output_channels, dtype)
    }
    fn residual_forward(
        weights: &ResidualWeights,
        ctx: &Context,
        x: &Tensor,
        h: usize,
        w: usize,
        policy: &ops::GemmPolicy,
    ) -> Result<Tensor> {
        let residual = match &weights.shortcut {
            Some(conv) => conv_forward(conv, ctx, x, h, w, policy)?,
            None => x.clone(),
        };
        let x = conv_forward(
            &weights.conv1,
            ctx,
            &norm_activated(&weights.norm1, ctx, x)?,
            h,
            w,
            policy,
        )?;
        let x = conv_forward(
            &weights.conv2,
            ctx,
            &norm_activated(&weights.norm2, ctx, &x)?,
            h,
            w,
            policy,
        )?;
        compute::add(ctx, &x, &residual)
    }

    fn down_forward(
        weights: &DownBlockWeights,
        ctx: &Context,
        x: &Tensor,
        h: usize,
        w: usize,
        policy: &ops::GemmPolicy,
    ) -> Result<Tensor> {
        let mut y = x.clone();
        for block in &weights.residuals {
            y = residual_forward(block, ctx, &y, h, w, policy)?;
        }
        let fs = if weights.downsample.is_some() { 2 } else { 1 };
        if let Some(conv) = &weights.downsample {
            y = conv_forward(conv, ctx, &y, h, w, policy)?;
        }
        let mut residual = ctx.allocate_output(y.shape().clone(), y.dtype())?;
        ops::image_grouped_average(
            ctx,
            x,
            &mut residual,
            h,
            w,
            fs,
            weights.temporal_factor,
            y.shape().dims()[1],
        )?;
        compute::add(ctx, &y, &residual)
    }

    fn attention_forward(
        weights: &AttentionWeights,
        ctx: &Context,
        x: &Tensor,
        h: usize,
        w: usize,
        policy: &ops::GemmPolicy,
    ) -> Result<Tensor> {
        let normalized = norm_forward(&weights.norm, ctx, x)?;
        let channels = x.shape().dims()[1];
        let q = compute::reshape(
            &compute::project_conv(&weights.q, ctx, &normalized, policy)?,
            vec![1, h * w, 1, channels],
        )?;
        let k = compute::reshape(
            &compute::project_conv(&weights.k, ctx, &normalized, policy)?,
            vec![1, h * w, 1, channels],
        )?;
        let v = compute::reshape(
            &compute::project_conv(&weights.v, ctx, &normalized, policy)?,
            vec![1, h * w, 1, channels],
        )?;
        let mut output = ctx.allocate_output(q.shape().clone(), q.dtype())?;
        let mut args = ops::AttentionArgs::new(&q, &k, &v, &mut output);
        args.policy.online_tune = policy.online_tune;
        args.policy.cache_dir = policy.cache_dir.clone();
        ops::attention(ctx, args)?;
        let output = compute::reshape(&output, vec![h * w, channels])?;
        compute::add(
            ctx,
            &conv_forward(&weights.projection, ctx, &output, h, w, policy)?,
            x,
        )
    }

    pub(in crate::cuda_fastwam::model::blocks::bf16) struct Vae {
        weights: VaeWeights,
    }
    impl Vae {
        pub(in crate::cuda_fastwam::model::blocks::bf16) fn new(weights: VaeWeights) -> Self {
            Self { weights }
        }

        /// Conservative arena capacity derived from this encoder's allocation
        /// schedule. Native provider workspaces are separately owned by executions.
        pub(in crate::cuda_fastwam::model::blocks::bf16) fn workspace_bytes(&self) -> usize {
            let mut bytes = allocation_bytes(224*448*3,self.weights.dtype) // possible input cast
            +allocation_bytes(112*224*12,self.weights.dtype); // width-first patchify
            let (mut h, mut w) = (112, 224);
            bytes += conv_workspace_bytes(&self.weights.input, h, w);
            for block in &self.weights.down {
                for residual in &block.residuals {
                    bytes += residual_workspace_bytes(residual, h, w);
                }
                if let Some(conv) = &block.downsample {
                    bytes += conv_workspace_bytes(conv, h, w);
                    h /= 2;
                    w /= 2;
                }
                let channels = block
                    .residuals
                    .last()
                    .expect("validated two-residual VAE block")
                    .conv2
                    .projection
                    .weight
                    .shape()
                    .dims()[1];
                bytes += 2 * allocation_bytes(h * w * channels, self.weights.dtype);
                // average shortcut + add
            }
            bytes += residual_workspace_bytes(&self.weights.mid0, h, w);
            let channels = self.weights.attention.q.weight.shape().dims()[1];
            bytes += 11 * allocation_bytes(h * w * channels, self.weights.dtype); // norm,Q/K/V GEMM+bias,attention,projection GEMM+bias,add
            bytes += residual_workspace_bytes(&self.weights.mid1, h, w);
            bytes += 2 * allocation_bytes(h * w * channels, self.weights.dtype); // final norm + SiLU
            bytes += conv_workspace_bytes(&self.weights.output, h, w)
                + conv_workspace_bytes(&self.weights.quant, h, w);
            bytes + allocation_bytes(h * w * 48, DType::F32)
        }
        /// One stitched HWC RGB frame in [-1,1], flattened `[224*448,3]`.
        /// Returns standardized posterior mean `[14*28,48]` in F32.
        pub(in crate::cuda_fastwam::model::blocks::bf16) fn forward(
            &self,
            ctx: &Context,
            input: &Tensor,
            policy: &ops::GemmPolicy,
        ) -> Result<Tensor> {
            if input.shape().dims() != [224 * 448, 3] {
                return Err(Error::Other(
                    "Wan VAE requires one 224x448 RGB frame".into(),
                ));
            }
            let input = cast(ctx, input, self.weights.dtype)?;
            let mut packed =
                ctx.allocate_output(Shape::new(vec![112 * 224, 12]), self.weights.dtype)?;
            ops::image_space_to_depth(ctx, &input, &mut packed, 224, 448, 2)?;
            let (mut h, mut w) = (112, 224);
            let mut x = conv_forward(&self.weights.input, ctx, &packed, h, w, policy)?;
            for block in &self.weights.down {
                x = down_forward(block, ctx, &x, h, w, policy)?;
                if block.downsample.is_some() {
                    h /= 2;
                    w /= 2;
                }
            }
            x = residual_forward(&self.weights.mid0, ctx, &x, h, w, policy)?;
            x = attention_forward(&self.weights.attention, ctx, &x, h, w, policy)?;
            x = residual_forward(&self.weights.mid1, ctx, &x, h, w, policy)?;
            x = conv_forward(
                &self.weights.output,
                ctx,
                &norm_activated(&self.weights.norm, ctx, &x)?,
                h,
                w,
                policy,
            )?;
            x = conv_forward(&self.weights.quant, ctx, &x, h, w, policy)?;
            let mut out = ctx.allocate_output(Shape::new(vec![h * w, 48]), DType::F32)?;
            ops::channel_standardize(ctx, &x, &self.weights.mean, &self.weights.std, &mut out)?;
            Ok(out)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::cuda_fastwam::backend;
        use std::collections::HashMap;
        use std::path::Path;

        fn compare(name: &str, actual: &Tensor, fixtures: &HashMap<String, Tensor>, dtype: DType) {
            let a = backend::transfers::to_cpu(actual)
                .unwrap()
                .to_f32_vec()
                .unwrap();
            let reference = fixtures.get(name).unwrap();
            assert_eq!(actual.shape(), reference.shape(), "{name}");
            let b = reference.to_f32_vec().unwrap();
            let max_abs = a
                .iter()
                .zip(&b)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let relative_l2 = (a
                .iter()
                .zip(&b)
                .map(|(a, b)| f64::from(a - b).powi(2))
                .sum::<f64>()
                / b.iter()
                    .map(|b| f64::from(*b).powi(2))
                    .sum::<f64>()
                    .max(1e-30))
            .sqrt();
            eprintln!("VAE {dtype:?} {name}: max_abs={max_abs:.9} relative_l2={relative_l2:.9}");
            assert!(a.iter().all(|x| x.is_finite()), "nonfinite {name}");
            // Component parity is stricter than one BF16 ULP in relative norm;
            // full-model acceptance separately checks the FP32 quality floor.
            let tolerance = if dtype == DType::F32 { 1e-4 } else { 0.008 };
            assert!(
                relative_l2 <= tolerance,
                "{name} relative error {relative_l2} exceeds {tolerance}"
            );
        }

        #[test]
        #[ignore = "requires Wan VAE assets and independent reference captures on CUDA"]
        fn fastwam_vae_checkpoint_precision() {
            let assets = std::env::var("APXINF_FASTWAM_VAE")
                .expect("APXINF_FASTWAM_VAE points to vae directory");
            let captures = std::env::var("APXINF_FASTWAM_VAE_FIXTURES")
                .expect("APXINF_FASTWAM_VAE_FIXTURES points to capture directory");
            let ctx = Context::new_with_autotune(0, false).unwrap();
            let policy = ops::GemmPolicy {
                online_tune: false,
                ..Default::default()
            };
            for (dtype, name) in [(DType::F32, "float32-strict"), (DType::BF16, "bfloat16")] {
                let (fixtures, _) = apxinf_loader::safetensors::load_native_path(
                    &Path::new(&captures).join(format!("{name}.safetensors")),
                )
                .unwrap();
                let model = Vae::new(VaeWeights::load(&ctx, Path::new(&assets), dtype).unwrap());
                let input = backend::upload(&ctx, fixtures.get("input").unwrap()).unwrap();
                let input = cast(&ctx, &input, dtype).unwrap();
                let mut packed = ctx
                    .allocate_output(Shape::new(vec![112 * 224, 12]), dtype)
                    .unwrap();
                ops::image_space_to_depth(&ctx, &input, &mut packed, 224, 448, 2).unwrap();
                let (mut h, mut w) = (112, 224);
                let mut x =
                    conv_forward(&model.weights.input, &ctx, &packed, h, w, &policy).unwrap();
                compare("encoder.conv_in", &x, &fixtures, dtype);
                for (i, block) in model.weights.down.iter().enumerate() {
                    x = down_forward(block, &ctx, &x, h, w, &policy).unwrap();
                    if block.downsample.is_some() {
                        h /= 2;
                        w /= 2;
                    }
                    compare(&format!("encoder.down_blocks.{i}"), &x, &fixtures, dtype);
                }
                x = residual_forward(&model.weights.mid0, &ctx, &x, h, w, &policy).unwrap();
                compare("encoder.mid_block.resnets.0", &x, &fixtures, dtype);
                x = attention_forward(&model.weights.attention, &ctx, &x, h, w, &policy).unwrap();
                compare("encoder.mid_block.attentions.0", &x, &fixtures, dtype);
                x = residual_forward(&model.weights.mid1, &ctx, &x, h, w, &policy).unwrap();
                compare("encoder.mid_block.resnets.1", &x, &fixtures, dtype);
                x = conv_forward(
                    &model.weights.output,
                    &ctx,
                    &norm_activated(&model.weights.norm, &ctx, &x).unwrap(),
                    h,
                    w,
                    &policy,
                )
                .unwrap();
                compare("encoder.conv_out", &x, &fixtures, dtype);
                x = conv_forward(&model.weights.quant, &ctx, &x, h, w, &policy).unwrap();
                compare("quant_conv", &x, &fixtures, dtype);
                let output = model.forward(&ctx, &input, &policy).unwrap();
                compare("output", &output, &fixtures, dtype);
                if dtype == DType::BF16 {
                    let mut captured = None;
                    let phase = apxinf_cuda_new::PreparedPhase::prepare_and_capture(
                        &ctx,
                        ops::ExecutionSession::with_capacity(
                            model.workspace_bytes(),
                            ctx.device_id(),
                        )
                        .unwrap(),
                        || {
                            captured = Some(model.forward(&ctx, &input, &policy)?);
                            Ok(())
                        },
                    )
                    .unwrap();
                    phase.replay().unwrap();
                    ctx.synchronize().unwrap();
                    compare("output", captured.as_ref().unwrap(), &fixtures, dtype);
                    let baseline = backend::transfers::to_cpu(captured.as_ref().unwrap())
                        .unwrap()
                        .to_f32_vec()
                        .unwrap();
                    for _ in 0..3 {
                        phase.replay().unwrap();
                    }
                    ctx.synchronize().unwrap();
                    assert_eq!(
                        baseline,
                        backend::transfers::to_cpu(captured.as_ref().unwrap())
                            .unwrap()
                            .to_f32_vec()
                            .unwrap()
                    );
                    let input_buffer = backend::DeviceBuffer::from_tensor(&input).unwrap();
                    input_buffer
                        .copy_from_host(&vec![0; input.shape().numel() * 2])
                        .unwrap();
                    phase
                        .run(|| {
                            captured = Some(model.forward(&ctx, &input, &policy)?);
                            Ok(())
                        })
                        .unwrap();
                    let changed = backend::transfers::to_cpu(captured.as_ref().unwrap())
                        .unwrap()
                        .to_f32_vec()
                        .unwrap();
                    assert!(
                        changed
                            .iter()
                            .zip(&baseline)
                            .any(|(x, y)| (x - y).abs() > 0.01),
                        "changed frame did not propagate"
                    );
                    drop(model);
                    drop(input);
                    drop(input_buffer);
                    phase.replay().unwrap();
                    ctx.synchronize().unwrap();
                    assert_eq!(
                        changed,
                        backend::transfers::to_cpu(captured.as_ref().unwrap())
                            .unwrap()
                            .to_f32_vec()
                            .unwrap()
                    );
                    let started = std::time::Instant::now();
                    for _ in 0..10 {
                        phase.replay().unwrap();
                    }
                    ctx.synchronize().unwrap();
                    eprintln!(
                        "VAE BF16 captured latency_ms={} workspace_used={}",
                        started.elapsed().as_secs_f64() * 100.,
                        phase.workspace_used()
                    );
                }
            }
        }
    }
}
mod prepare {
    //! Prepare fixed time modulation through the same BF16 computation helpers.
    use super::{
        dit::Expert,
        ops::{self as compute, bias_add, cast},
    };
    use crate::cuda_fastwam::{
        backend::{self, ops, Context},
        weights::{host_tensor, ExpertWeights, TimeWeights},
    };
    use apxinf_core::{DType, Error, Result, Shape, Tensor};
    impl Expert {
        pub(in crate::cuda_fastwam::model::blocks::bf16) fn new(
            ctx: &Context,
            loaded: (ExpertWeights, TimeWeights),
            gemm: &ops::GemmPolicy,
            timesteps: &[f32],
        ) -> Result<Self> {
            let (
                weights,
                TimeWeights {
                    time0,
                    time2,
                    timeproj,
                },
            ) = loaded;
            let video = weights.video;
            let width = if video { 3072 } else { 1024 };
            let blocks = &weights.blocks;
            let modulation_one = vector(ctx, &vec![1.; width], DType::BF16)?;
            let mut modulations = Vec::new();
            for step in 0..if video { 1 } else { 10 } {
                let t = if video {
                    0.
                } else {
                    half::bf16::from_f32(timesteps[step]).to_f32() as f64
                };
                let mut embedding = Vec::with_capacity(256);
                for trig in 0..2 {
                    for i in 0..128 {
                        let phase = t * 10000f64.powf(-(i as f64) / 128.);
                        embedding.push(if trig == 0 {
                            phase.cos() as f32
                        } else {
                            phase.sin() as f32
                        });
                    }
                }
                let emb = backend::upload(
                    ctx,
                    &host_tensor(vec![1, 256], &embedding, time0.weight.dtype())?,
                )?;
                let t = compute::project(&time0, ctx, &emb, gemm)?;
                let t = compute::activation(ctx, &t, ops::PointwiseActivation::Silu)?;
                let t = compute::project(&time2, ctx, &t, gemm)?;
                let t = compute::activation(ctx, &t, ops::PointwiseActivation::Silu)?;
                let t = cast(
                    ctx,
                    &compute::project(&timeproj, ctx, &t, gemm)?,
                    DType::BF16,
                )?;
                let mut layers = Vec::new();
                for b in blocks {
                    let sum = compute::add(ctx, &b.modulation, &t)?;
                    let mut parts = Vec::new();
                    for j in 0..6 {
                        let part = slice(&sum, j * width, width)?;
                        parts.push(if j == 1 || j == 4 {
                            bias_add(ctx, &part.reshape(vec![1, width])?, &modulation_one)?
                                .reshape(vec![width])?
                        } else {
                            part
                        });
                    }
                    layers.push(parts);
                }
                modulations.push(layers);
            }
            Ok(Self {
                weights,
                modulations,
            })
        }
    }
    fn vector(ctx: &Context, values: &[f32], dtype: DType) -> Result<Tensor> {
        backend::upload(ctx, &host_tensor(vec![values.len()], values, dtype)?)
    }

    fn slice(x: &Tensor, start: usize, count: usize) -> Result<Tensor> {
        backend::DeviceBuffer::from_tensor(x)
            .and_then(|b| {
                b.view(
                    start * x.dtype().size_in_bytes(),
                    count * x.dtype().size_in_bytes(),
                )
            })
            .and_then(|b| b.as_tensor(Shape::new(vec![count]), x.dtype()))
            .map_err(Error::Cuda)
    }
}

use self::ops::cast;
use super::super::Inputs;
use crate::cuda_fastwam::{
    backend::{ops as cuda_ops, Context},
    weights::{Bf16Weights, LinearWeights},
};
use apxinf_core::{DType, Result, Shape, Tensor};
use dit::{ContextCache, Expert, Kv};
use text::TextEncoder;
use vae::Vae;

pub(in crate::cuda_fastwam::model) struct Policies {
    gemm: cuda_ops::GemmPolicy,
    attention: cuda_ops::AttentionPolicy,
}

/// Physical conditioning representation is private to the BF16 implementation.
pub(in crate::cuda_fastwam::model) struct Bf16Conditioning {
    video: Vec<Kv>,
    action: ContextCache,
}

/// Owns device weights, precision-specific preparation, layouts and operator composition.
pub(in crate::cuda_fastwam::model) struct Bf16Blocks {
    text: TextEncoder,
    vae: Vae,
    video: Expert,
    action: Expert,
    proprio: LinearWeights,
    rgb_table: Tensor,
    gemm: cuda_ops::GemmPolicy,
    attention: cuda_ops::AttentionPolicy,
}
impl Bf16Blocks {
    pub(in crate::cuda_fastwam::model) fn policies(&self, autotune: bool) -> Policies {
        let mut gemm = self.gemm.clone();
        gemm.online_tune &= autotune;
        let mut attention = self.attention.clone();
        attention.online_tune &= autotune;
        Policies { gemm, attention }
    }
    pub(in crate::cuda_fastwam::model) fn workspace_bytes(&self) -> usize {
        let preprocessing = 224 * 448 * 3 * (4 + 2) + 129 * 4096 * 2 + 4096 * 2;
        let solver_outputs = 10 * (32 * 7 * 8 + 512) + 32 * 7 * 4;
        self.vae.workspace_bytes()
            + self.text.workspace_bytes()
            + self.video.workspace_bytes()
            + 10 * self.action.workspace_bytes()
            + preprocessing
            + solver_outputs
            + 16 * 256
    }

    pub(in crate::cuda_fastwam::model) fn encode_vision(
        &self,
        ctx: &Context,
        inputs: &Inputs,
        policies: &Policies,
    ) -> Result<Tensor> {
        let mut indices = ctx.allocate_output(Shape::new(vec![224 * 448 * 3]), DType::I32)?;
        cuda_ops::image_rgb_indices(
            ctx,
            &inputs.rgb,
            &mut indices,
            2,
            224,
            224,
            3,
            inputs.image_layout,
        )?;
        let rgb = ctx.allocate_output(Shape::new(vec![224 * 448 * 3, 1]), DType::BF16)?;
        cuda_ops::embedding_gather(ctx, &self.rgb_table, &indices, &rgb)?;
        self.vae
            .forward(ctx, &rgb.reshape(vec![224 * 448, 3])?, &policies.gemm)
    }
    pub(in crate::cuda_fastwam::model) fn encode_text(
        &self,
        ctx: &Context,
        inputs: &Inputs,
        policies: &Policies,
    ) -> Result<Tensor> {
        self.text.forward(
            ctx,
            &inputs.ids,
            &inputs.padding,
            &inputs.mask,
            &policies.gemm,
            &policies.attention,
        )
    }
    pub(in crate::cuda_fastwam::model) fn condition(
        &self,
        ctx: &Context,
        image: &Tensor,
        text: &Tensor,
        state: &Tensor,
        policies: &Policies,
    ) -> Result<Bf16Conditioning> {
        let proprio = ops::project(&self.proprio, ctx, state, &policies.gemm)?;
        let context = cuda_ops::concat_rows(ctx, text, &proprio)?;
        let video_context = self.video.context(ctx, &context, &policies.gemm)?;
        let (_, video) = self.video.forward(
            ctx,
            image,
            &video_context,
            None,
            0,
            &policies.gemm,
            &policies.attention,
        )?;
        let action = self.action.context(ctx, &context, &policies.gemm)?;
        Ok(Bf16Conditioning { video, action })
    }
    pub(in crate::cuda_fastwam::model) fn initial_state(
        &self,
        ctx: &Context,
        noise: &Tensor,
    ) -> Result<Tensor> {
        cast(ctx, noise, DType::F32)
    }
    pub(in crate::cuda_fastwam::model) fn step(
        &self,
        ctx: &Context,
        x: &Tensor,
        conditioning: &Bf16Conditioning,
        step: usize,
        dt: f32,
        policies: &Policies,
    ) -> Result<Tensor> {
        let (velocity, _) = self.action.forward(
            ctx,
            x,
            &conditioning.action,
            Some(&conditioning.video),
            step,
            &policies.gemm,
            &policies.attention,
        )?;
        let velocity = cast(ctx, &velocity, x.dtype())?;
        let mut next = ctx.allocate_output(x.shape().clone(), x.dtype())?;
        let mut args =
            cuda_ops::PointwiseArgs::new(cuda_ops::PointwiseSemantic::EulerUpdate, x, &mut next);
        args.secondary = Some(&velocity);
        // The BF16 checkpoint profile's scheduler materializes deltas in BF16,
        // then promotes them when integrating the F32 state.
        args.dt = half::bf16::from_f32(dt).to_f32();
        cuda_ops::pointwise(ctx, args)?;
        Ok(next)
    }
    pub(in crate::cuda_fastwam::model) fn finish(
        &self,
        ctx: &Context,
        x: &Tensor,
    ) -> Result<Tensor> {
        cast(ctx, x, DType::F32)
    }
    pub(in crate::cuda_fastwam::model) fn new(
        ctx: &Context,
        weights: Bf16Weights,
        gemm: cuda_ops::GemmPolicy,
        attention: cuda_ops::AttentionPolicy,
        timesteps: &[f32],
    ) -> Result<Self> {
        let video = Expert::new(ctx, weights.video, &gemm, &[0.])?;
        let action = Expert::new(ctx, weights.action, &gemm, timesteps)?;
        Ok(Self {
            video,
            action,
            proprio: weights.proprio,
            text: TextEncoder::new(weights.text),
            vae: Vae::new(weights.vae),
            rgb_table: weights.rgb_table,
            gemm,
            attention,
        })
    }
}
