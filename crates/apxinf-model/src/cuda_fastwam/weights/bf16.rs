//! Device weight representation for the BF16 model, including explicit F32 tensors.
use super::host::{host_tensor, validate_tensor, Bias, Store};
use crate::cuda_fastwam::backend::{self, ops, Context};
use apxinf_core::{DType, Error, Result, Tensor};
mod dit {
    //! Immutable checkpoint mapping and prepared weight data.
    use super::cast_weight;
    use super::{host_tensor, validate_tensor, Bias, LinearWeights, Store};
    use super::{load_linear, upload_tensor};
    use crate::cuda_fastwam::backend::{self, ops, Context};
    use apxinf_core::{DType, Error, Result, Tensor};
    pub(in crate::cuda_fastwam) struct ProjectionWeights {
        pub(in crate::cuda_fastwam) q: LinearWeights,
        pub(in crate::cuda_fastwam) k: LinearWeights,
        pub(in crate::cuda_fastwam) v: LinearWeights,
        pub(in crate::cuda_fastwam) o: LinearWeights,
        pub(in crate::cuda_fastwam) nq: Tensor,
        pub(in crate::cuda_fastwam) nk: Tensor,
    }

    impl ProjectionWeights {
        fn load(s: &mut Store, ctx: &Context, p: &str) -> Result<Self> {
            Ok(Self {
                q: load_linear(&mut *s, ctx, &format!("{p}.q"), Bias::Required)?,
                k: load_linear(&mut *s, ctx, &format!("{p}.k"), Bias::Required)?,
                v: load_linear(&mut *s, ctx, &format!("{p}.v"), Bias::Required)?,
                o: load_linear(&mut *s, ctx, &format!("{p}.o"), Bias::Required)?,
                nq: upload_tensor(&mut *s, ctx, &format!("{p}.norm_q.weight"))?,
                nk: upload_tensor(&mut *s, ctx, &format!("{p}.norm_k.weight"))?,
            })
        }
    }

    pub(in crate::cuda_fastwam) struct BlockWeights {
        pub(in crate::cuda_fastwam) sa: ProjectionWeights,
        pub(in crate::cuda_fastwam) ca: ProjectionWeights,
        pub(in crate::cuda_fastwam) norm3w: Tensor,
        pub(in crate::cuda_fastwam) norm3b: Tensor,
        pub(in crate::cuda_fastwam) ff0: LinearWeights,
        pub(in crate::cuda_fastwam) ff2: LinearWeights,
        pub(in crate::cuda_fastwam) modulation: Tensor,
    }

    pub(in crate::cuda_fastwam) struct ExpertWeights {
        pub(in crate::cuda_fastwam) blocks: Vec<BlockWeights>,
        pub(in crate::cuda_fastwam) text0: LinearWeights,
        pub(in crate::cuda_fastwam) text2: LinearWeights,
        pub(in crate::cuda_fastwam) input: LinearWeights,
        pub(in crate::cuda_fastwam) head: Option<LinearWeights>,
        pub(in crate::cuda_fastwam) one: Tensor,
        pub(in crate::cuda_fastwam) zero: Tensor,
        pub(in crate::cuda_fastwam) rope: ops::PairwiseRopeTable,
        pub(in crate::cuda_fastwam) video: bool,
    }

    impl ExpertWeights {
        pub(in crate::cuda_fastwam) fn load(
            s: &mut Store,
            ctx: &Context,
            video: bool,
        ) -> Result<(Self, TimeWeights)> {
            let name = if video { "video" } else { "action" };
            let p = format!("model.mot.mixtures.{name}");
            let width = if video { 3072 } else { 1024 };
            let mut blocks = Vec::new();
            for i in 0..30 {
                let b = format!("{p}.blocks.{i}");
                blocks.push(BlockWeights {
                    sa: ProjectionWeights::load(s, ctx, &format!("{b}.self_attn"))?,
                    ca: ProjectionWeights::load(s, ctx, &format!("{b}.cross_attn"))?,
                    norm3w: upload_tensor(&mut *s, ctx, &format!("{b}.norm3.weight"))?,
                    norm3b: upload_tensor(&mut *s, ctx, &format!("{b}.norm3.bias"))?,
                    ff0: load_linear(&mut *s, ctx, &format!("{b}.ffn.0"), Bias::Required)?,
                    ff2: load_linear(&mut *s, ctx, &format!("{b}.ffn.2"), Bias::Required)?,
                    modulation: upload_tensor(&mut *s, ctx, &format!("{b}.modulation"))?
                        .reshape(vec![1, 6 * width])?,
                });
            }
            for b in &blocks {
                for projections in [&b.sa, &b.ca] {
                    for projection in [&projections.q, &projections.k, &projections.v] {
                        projection.validate("Wan QKV", width, 3072)?;
                    }
                    projections
                        .o
                        .validate("Wan attention output", 3072, width)?;
                    validate_tensor("Wan Q norm", &projections.nq, &[3072], DType::BF16)?;
                    validate_tensor("Wan K norm", &projections.nk, &[3072], DType::BF16)?;
                }
                let ff = if video { 14336 } else { 4096 };
                b.ff0.validate("Wan FFN up", width, ff)?;
                b.ff2.validate("Wan FFN down", ff, width)?;
                validate_tensor("Wan cross norm", &b.norm3w, &[width], DType::BF16)?;
                validate_tensor("Wan cross norm bias", &b.norm3b, &[width], DType::BF16)?;
            }
            // Keep immutable affine data in the residual state's dtype. This avoids
            // repeating weight conversions inside every captured denoising step.
            for b in &mut blocks {
                b.norm3w = cast_weight(ctx, &b.norm3w, DType::F32)?;
                b.norm3b = cast_weight(ctx, &b.norm3b, DType::F32)?;
            }
            let mut input = if video {
                let raw = s.take(&format!("{p}.patch_embedding.weight"))?;
                if raw.shape().dims() != [3072, 48, 1, 2, 2] {
                    return Err(Error::Other(
                        "Fast-WAM video patch weight shape mismatch".into(),
                    ));
                }
                let raw = raw.to_f32_vec()?;
                let mut transposed = vec![0.; raw.len()];
                for n in 0..width {
                    for c in 0..48 {
                        for y in 0..2 {
                            for x in 0..2 {
                                transposed[((y * 2 + x) * 48 + c) * width + n] =
                                    raw[((n * 48 + c) * 2 + y) * 2 + x];
                            }
                        }
                    }
                }
                LinearWeights {
                    weight: backend::upload(
                        ctx,
                        &host_tensor(vec![192, width], &transposed, DType::BF16)?,
                    )?,
                    bias: Some(upload_tensor(
                        &mut *s,
                        ctx,
                        &format!("{p}.patch_embedding.bias"),
                    )?),
                }
            } else {
                load_linear(&mut *s, ctx, &format!("{p}.action_encoder"), Bias::Required)?
            };
            let mut time0 = load_linear(
                &mut *s,
                ctx,
                &format!("{p}.time_embedding.0"),
                Bias::Required,
            )?;
            let mut time2 = load_linear(
                &mut *s,
                ctx,
                &format!("{p}.time_embedding.2"),
                Bias::Required,
            )?;
            let mut timeproj = load_linear(
                &mut *s,
                ctx,
                &format!("{p}.time_projection.1"),
                Bias::Required,
            )?;
            input.validate("Wan input", if video { 192 } else { 7 }, width)?;
            if !video {
                // Keep Euler state in F32 through its small input projection.
                // Validate the original BF16 checkpoint before exact promotion.
                input.weight = cast_weight(ctx, &input.weight, DType::F32)?;
                input.bias = input
                    .bias
                    .as_ref()
                    .map(|bias| cast_weight(ctx, bias, DType::F32))
                    .transpose()?;
            }
            time0.validate("Wan time input", 256, width)?;
            time2.validate("Wan time hidden", width, width)?;
            timeproj.validate("Wan time modulation", width, 6 * width)?;
            if video {
                for projection in [&mut time0, &mut time2, &mut timeproj] {
                    projection.weight = cast_weight(ctx, &projection.weight, DType::F32)?;
                    projection.bias = projection
                        .bias
                        .as_ref()
                        .map(|b| cast_weight(ctx, b, DType::F32))
                        .transpose()?;
                }
            }
            let one = vector(ctx, &vec![1.; width], DType::F32)?;
            let zero = vector(ctx, &vec![0.; width], DType::F32)?;
            let tokens = if video { 98 } else { 32 };
            let mut coefficients = Vec::new();
            for pos in 0..tokens {
                if video {
                    for (coordinate, pairs) in [(0, 22), (pos / 14, 21), (pos % 14, 21)] {
                        for i in 0..pairs {
                            let phase =
                                coordinate as f64 * 10000f64.powf(-(i as f64) / pairs as f64);
                            coefficients.push((phase.cos(), phase.sin()));
                        }
                    }
                } else {
                    for i in 0..64 {
                        let phase = pos as f64 * 10000f64.powf(-(i as f64) / 64.);
                        coefficients.push((phase.cos(), phase.sin()));
                    }
                }
            }
            let text0 = load_linear(
                &mut *s,
                ctx,
                &format!("{p}.text_embedding.0"),
                Bias::Required,
            )?;
            let text2 = load_linear(
                &mut *s,
                ctx,
                &format!("{p}.text_embedding.2"),
                Bias::Required,
            )?;
            text0.validate("Wan text input", 4096, width)?;
            text2.validate("Wan text hidden", width, width)?;
            let mut head = if video {
                None
            } else {
                Some(load_linear(
                    &mut *s,
                    ctx,
                    &format!("{p}.head"),
                    Bias::Required,
                )?)
            };
            if let Some(head) = &mut head {
                head.validate("Wan action output", width, 7)?;
                // Preserve the F32 residual through the small velocity projection.
                // Checkpoint values are BF16; their F32 promotion is exact.
                head.weight = cast_weight(ctx, &head.weight, DType::F32)?;
                head.bias = head
                    .bias
                    .as_ref()
                    .map(|bias| cast_weight(ctx, bias, DType::F32))
                    .transpose()?;
            }
            Ok((
                Self {
                    blocks,
                    text0,
                    text2,
                    input,
                    head,
                    one,
                    zero,
                    rope: ops::PairwiseRopeTable::new(ctx, tokens, 128, &coefficients)?,
                    video,
                },
                TimeWeights {
                    time0,
                    time2,
                    timeproj,
                },
            ))
        }
    }

    fn vector(ctx: &Context, values: &[f32], dtype: DType) -> Result<Tensor> {
        backend::upload(ctx, &host_tensor(vec![values.len()], values, dtype)?)
    }

    pub(in crate::cuda_fastwam) struct TimeWeights {
        pub time0: LinearWeights,
        pub time2: LinearWeights,
        pub timeproj: LinearWeights,
    }
}

mod text {
    //! Immutable checkpoint mapping and prepared weight data.
    use super::{host_tensor, validate_tensor, Bias, LinearWeights, Store};
    use super::{load_linear, upload_tensor};
    use crate::cuda_fastwam::backend::{self, Context};
    use apxinf_core::{DType, Error, Result, Tensor};
    use std::path::Path;
    pub(in crate::cuda_fastwam) struct BlockWeights {
        pub(in crate::cuda_fastwam) norm0: Tensor,
        pub(in crate::cuda_fastwam) q: LinearWeights,
        pub(in crate::cuda_fastwam) k: LinearWeights,
        pub(in crate::cuda_fastwam) v: LinearWeights,
        pub(in crate::cuda_fastwam) o: LinearWeights,
        pub(in crate::cuda_fastwam) bias: Tensor,
        pub(in crate::cuda_fastwam) norm1: Tensor,
        pub(in crate::cuda_fastwam) wi0: LinearWeights,
        pub(in crate::cuda_fastwam) wi1: LinearWeights,
        pub(in crate::cuda_fastwam) wo: LinearWeights,
    }

    pub(in crate::cuda_fastwam) struct TextEncoderWeights {
        pub(in crate::cuda_fastwam) embedding: Tensor,
        pub(in crate::cuda_fastwam) blocks: Vec<BlockWeights>,
        pub(in crate::cuda_fastwam) norm: Tensor,
    }

    impl TextEncoderWeights {
        pub(in crate::cuda_fastwam) fn load(ctx: &Context, path: &Path) -> Result<Self> {
            let v: serde_json::Value = serde_json::from_slice(
                &std::fs::read(path.join("config.json"))
                    .map_err(|e| Error::Other(e.to_string()))?,
            )
            .map_err(|e| Error::Other(e.to_string()))?;
            for (key, value) in [
                ("d_model", 4096),
                ("d_ff", 10240),
                ("d_kv", 64),
                ("num_heads", 64),
                ("num_layers", 24),
                ("relative_attention_num_buckets", 32),
                ("relative_attention_max_distance", 128),
                ("vocab_size", 256384),
            ] {
                if v[key].as_u64() != Some(value) {
                    return Err(Error::Other(format!("unsupported Fast-WAM UMT5 {key}")));
                }
            }
            if v["dense_act_fn"] != "gelu_new"
                || v["feed_forward_proj"] != "gated-gelu"
                || v["layer_norm_epsilon"].as_f64() != Some(1e-6)
            {
                return Err(Error::Other(
                    "unsupported UMT5 activation or normalization".into(),
                ));
            }
            let mut store = Store::load(path)?;
            let embedding = upload_tensor(&mut store, ctx, "shared.weight")?;
            validate_tensor("UMT5 embedding", &embedding, &[256384, 4096], DType::BF16)?;
            let mut blocks = Vec::new();
            for i in 0..24 {
                let p = format!("encoder.block.{i}");
                let a = format!("{p}.layer.0.SelfAttention");
                let f = format!("{p}.layer.1.DenseReluDense");
                let relative = store.take(&format!("{a}.relative_attention_bias.weight"))?;
                if relative.shape().dims() != [32, 64] {
                    return Err(Error::Other("UMT5 relative bias must be [32,64]".into()));
                }
                let relative = relative.to_f32_vec()?;
                let mut expanded = vec![0.; 64 * 128 * 128];
                for h in 0..64 {
                    for q in 0..128 {
                        for k in 0..128 {
                            let distance = (k as i32 - q as i32).unsigned_abs() as usize;
                            let bucket = if distance < 8 {
                                distance
                            } else {
                                8 + ((distance as f32 / 8.).ln() / 16f32.ln() * 8.) as usize
                            }
                            .min(15)
                                + if k > q { 16 } else { 0 };
                            expanded[(h * 128 + q) * 128 + k] = relative[bucket * 64 + h];
                        }
                    }
                }
                blocks.push(BlockWeights {
                    norm0: upload_tensor(
                        &mut store,
                        ctx,
                        &format!("{p}.layer.0.layer_norm.weight"),
                    )?,
                    q: load_linear(&mut store, ctx, &format!("{a}.q"), Bias::Forbidden)?,
                    k: load_linear(&mut store, ctx, &format!("{a}.k"), Bias::Forbidden)?,
                    v: load_linear(&mut store, ctx, &format!("{a}.v"), Bias::Forbidden)?,
                    o: load_linear(&mut store, ctx, &format!("{a}.o"), Bias::Forbidden)?,
                    bias: backend::upload(
                        ctx,
                        &host_tensor(vec![64 * 128, 128], &expanded, DType::F32)?,
                    )?,
                    norm1: upload_tensor(
                        &mut store,
                        ctx,
                        &format!("{p}.layer.1.layer_norm.weight"),
                    )?,
                    wi0: load_linear(&mut store, ctx, &format!("{f}.wi_0"), Bias::Forbidden)?,
                    wi1: load_linear(&mut store, ctx, &format!("{f}.wi_1"), Bias::Forbidden)?,
                    wo: load_linear(&mut store, ctx, &format!("{f}.wo"), Bias::Forbidden)?,
                });
            }
            let norm = upload_tensor(&mut store, ctx, "encoder.final_layer_norm.weight")?;
            validate_tensor("UMT5 final norm", &norm, &[4096], DType::BF16)?;
            for b in &blocks {
                for projection in [&b.q, &b.k, &b.v, &b.o] {
                    projection.validate("UMT5 attention projection", 4096, 4096)?;
                }
                b.wi0.validate("UMT5 gate", 4096, 10240)?;
                b.wi1.validate("UMT5 up", 4096, 10240)?;
                b.wo.validate("UMT5 down", 10240, 4096)?;
                validate_tensor("UMT5 norm", &b.norm0, &[4096], DType::BF16)?;
                validate_tensor("UMT5 norm", &b.norm1, &[4096], DType::BF16)?;
            }
            Ok(Self {
                embedding,
                blocks,
                norm,
            })
        }
    }
}

mod vae {
    //! Immutable checkpoint mapping and prepared weight data.
    use super::{host_tensor, LinearWeights, Store};
    use crate::cuda_fastwam::backend::{self, Context};
    use apxinf_core::{DType, Error, Result, Tensor};
    use std::path::Path;
    pub(in crate::cuda_fastwam) struct ConvWeights {
        pub(in crate::cuda_fastwam) projection: LinearWeights,
        pub(in crate::cuda_fastwam) kernel: usize,
        pub(in crate::cuda_fastwam) stride: usize,
        pub(in crate::cuda_fastwam) padding: usize,
    }

    impl ConvWeights {
        fn load(
            store: &mut Store,
            ctx: &Context,
            prefix: &str,
            dtype: DType,
            expected: &[usize],
            stride: usize,
            padding: usize,
        ) -> Result<Self> {
            let raw = store.take(&format!("{prefix}.weight"))?;
            let raw_bias = store.take(&format!("{prefix}.bias"))?;
            validate_conv_shapes(
                prefix,
                raw.shape().dims(),
                raw_bias.shape().dims(),
                expected,
            )?;
            let (n, c, k) = (expected[0], expected[1], expected[expected.len() - 1]);
            let temporal = if expected.len() == 5 { expected[2] } else { 1 };
            let src = raw.to_f32_vec()?;
            let mut dst = vec![0.; k * k * c * n];
            for o in 0..n {
                for i in 0..c {
                    for y in 0..k {
                        for x in 0..k {
                            dst[((y * k + x) * c + i) * n + o] =
                                src[(((o * c + i) * temporal + temporal - 1) * k + y) * k + x];
                        }
                    }
                }
            }
            let weight = backend::upload(ctx, &host_tensor(vec![k * k * c, n], &dst, dtype)?)?;
            let bias =
                backend::upload(ctx, &host_tensor(vec![n], &raw_bias.to_f32_vec()?, dtype)?)?;
            Ok(Self {
                projection: LinearWeights {
                    weight,
                    bias: Some(bias),
                },
                kernel: k,
                stride,
                padding,
            })
        }
    }

    // The profile fixes each convolution's rank, channels and temporal/spatial
    // kernels. A kernel whitelist alone admits shape-preserving architecture changes.
    fn validate_conv_shapes(
        prefix: &str,
        weight: &[usize],
        bias: &[usize],
        expected: &[usize],
    ) -> Result<()> {
        if weight != expected || bias != [expected[0]] {
            return Err(Error::Other(format!(
                "invalid VAE convolution {prefix}: weight {weight:?}, bias {bias:?}; expected {expected:?} and [{}]",
                expected[0]
            )));
        }
        Ok(())
    }

    pub(in crate::cuda_fastwam) struct NormWeights {
        pub(in crate::cuda_fastwam) gamma: Tensor,
    }

    impl NormWeights {
        fn load(
            store: &mut Store,
            ctx: &Context,
            prefix: &str,
            dtype: DType,
            expected: &[usize],
        ) -> Result<Self> {
            let raw = store.take(&format!("{prefix}.gamma"))?;
            let shape = raw.shape().dims();
            if shape != expected {
                return Err(Error::Other(format!(
                    "invalid VAE norm gamma {prefix}: {shape:?}"
                )));
            }
            Ok(Self {
                gamma: backend::upload(
                    ctx,
                    &host_tensor(vec![raw.shape().numel()], &raw.to_f32_vec()?, dtype)?,
                )?,
            })
        }
    }

    pub(in crate::cuda_fastwam) struct ResidualWeights {
        pub(in crate::cuda_fastwam) norm1: NormWeights,
        pub(in crate::cuda_fastwam) conv1: ConvWeights,
        pub(in crate::cuda_fastwam) norm2: NormWeights,
        pub(in crate::cuda_fastwam) conv2: ConvWeights,
        pub(in crate::cuda_fastwam) shortcut: Option<ConvWeights>,
    }

    impl ResidualWeights {
        fn load(
            store: &mut Store,
            ctx: &Context,
            prefix: &str,
            dtype: DType,
            input: usize,
            output: usize,
        ) -> Result<Self> {
            let has_shortcut = store.contains(&format!("{prefix}.conv_shortcut.weight"));
            if has_shortcut != (input != output)
                || store.contains(&format!("{prefix}.conv_shortcut.bias")) != has_shortcut
            {
                return Err(Error::Other(format!(
                    "invalid VAE residual shortcut {prefix}"
                )));
            }
            let shortcut = if has_shortcut {
                Some(ConvWeights::load(
                    store,
                    ctx,
                    &format!("{prefix}.conv_shortcut"),
                    dtype,
                    &[output, input, 1, 1, 1],
                    1,
                    0,
                )?)
            } else {
                None
            };
            Ok(Self {
                norm1: NormWeights::load(
                    store,
                    ctx,
                    &format!("{prefix}.norm1"),
                    dtype,
                    &[input, 1, 1, 1],
                )?,
                conv1: ConvWeights::load(
                    store,
                    ctx,
                    &format!("{prefix}.conv1"),
                    dtype,
                    &[output, input, 3, 3, 3],
                    1,
                    1,
                )?,
                norm2: NormWeights::load(
                    store,
                    ctx,
                    &format!("{prefix}.norm2"),
                    dtype,
                    &[output, 1, 1, 1],
                )?,
                conv2: ConvWeights::load(
                    store,
                    ctx,
                    &format!("{prefix}.conv2"),
                    dtype,
                    &[output, output, 3, 3, 3],
                    1,
                    1,
                )?,
                shortcut,
            })
        }
    }

    pub(in crate::cuda_fastwam) struct DownBlockWeights {
        pub(in crate::cuda_fastwam) residuals: Vec<ResidualWeights>,
        pub(in crate::cuda_fastwam) downsample: Option<ConvWeights>,
        pub(in crate::cuda_fastwam) temporal_factor: usize,
    }

    pub(in crate::cuda_fastwam) struct AttentionWeights {
        pub(in crate::cuda_fastwam) norm: NormWeights,
        pub(in crate::cuda_fastwam) q: LinearWeights,
        pub(in crate::cuda_fastwam) k: LinearWeights,
        pub(in crate::cuda_fastwam) v: LinearWeights,
        pub(in crate::cuda_fastwam) projection: ConvWeights,
    }

    impl AttentionWeights {
        fn load(store: &mut Store, ctx: &Context, prefix: &str, dtype: DType) -> Result<Self> {
            let raw = store.take(&format!("{prefix}.to_qkv.weight"))?;
            let raw_bias = store.take(&format!("{prefix}.to_qkv.bias"))?;
            let c = validate_qkv_shapes(raw.shape().dims(), raw_bias.shape().dims())?;
            let values = raw.to_f32_vec()?;
            let bias = raw_bias.to_f32_vec()?;
            let mut parts = Vec::new();
            for part in 0..3 {
                let mut transposed = vec![0.; c * c];
                for n in 0..c {
                    for k in 0..c {
                        transposed[k * c + n] = values[(part * c + n) * c + k];
                    }
                }
                parts.push(LinearWeights {
                    weight: backend::upload(ctx, &host_tensor(vec![c, c], &transposed, dtype)?)?,
                    bias: Some(backend::upload(
                        ctx,
                        &host_tensor(vec![c], &bias[part * c..(part + 1) * c], dtype)?,
                    )?),
                });
            }
            let mut parts = parts.into_iter();
            Ok(Self {
                norm: NormWeights::load(
                    store,
                    ctx,
                    &format!("{prefix}.norm"),
                    dtype,
                    &[640, 1, 1],
                )?,
                q: parts.next().unwrap(),
                k: parts.next().unwrap(),
                v: parts.next().unwrap(),
                projection: ConvWeights::load(
                    store,
                    ctx,
                    &format!("{prefix}.proj"),
                    dtype,
                    &[640, 640, 1, 1],
                    1,
                    0,
                )?,
            })
        }
    }

    fn validate_qkv_shapes(weight: &[usize], bias: &[usize]) -> Result<usize> {
        if weight != [1920, 640, 1, 1] || bias != [1920] {
            return Err(Error::Other(format!(
                "invalid VAE attention weight/bias shapes: {weight:?}, {bias:?}"
            )));
        }
        Ok(640)
    }

    #[cfg(test)]
    mod tests {
        use super::{validate_conv_shapes, validate_qkv_shapes};

        #[test]
        fn vae_convolutions_reject_architecture_substitutions() {
            for expected in [
                &[160, 12, 3, 3, 3][..], // encoder input
                &[320, 160, 3, 3, 3],    // channel-changing residual
                &[640, 640, 3, 3, 3],    // constant-width residual
                &[320, 160, 1, 1, 1],    // residual shortcut
                &[320, 320, 3, 3],       // spatial downsampler
                &[640, 640, 1, 1],       // attention output
                &[96, 640, 3, 3, 3],     // encoder output
                &[96, 96, 1, 1, 1],      // latent quantization
            ] {
                let bias = [expected[0]];
                assert!(validate_conv_shapes("test", expected, &bias, expected).is_ok());
                for axis in 0..expected.len() {
                    let mut changed = expected.to_vec();
                    changed[axis] = if expected[axis] == 1 { 3 } else { 1 };
                    assert!(validate_conv_shapes("test", &changed, &bias, expected).is_err());
                }
                assert!(validate_conv_shapes("test", &[], &bias, expected).is_err());
                assert!(validate_conv_shapes("test", &expected[1..], &bias, expected).is_err());
                assert!(validate_conv_shapes("test", expected, &[1], expected).is_err());
                assert!(
                    validate_conv_shapes("test", expected, &[1, expected[0]], expected).is_err()
                );
            }
        }

        #[test]
        fn vae_qkv_rejects_malformed_shapes_before_indexing() {
            assert_eq!(
                validate_qkv_shapes(&[1920, 640, 1, 1], &[1920]).unwrap(),
                640
            );
            for shape in [
                &[][..],
                &[1920][..],
                &[1920, 640][..],
                &[1920, 640, 3, 3][..],
                &[usize::MAX, usize::MAX, 1, 1][..],
            ] {
                assert!(validate_qkv_shapes(shape, &[1920]).is_err());
            }
            assert!(validate_qkv_shapes(&[1920, 640, 1, 1], &[640]).is_err());
            assert!(validate_qkv_shapes(&[1920, 640, 1, 1], &[3, 640]).is_err());
        }
    }

    pub(in crate::cuda_fastwam) struct VaeWeights {
        pub(in crate::cuda_fastwam) input: ConvWeights,
        pub(in crate::cuda_fastwam) down: Vec<DownBlockWeights>,
        pub(in crate::cuda_fastwam) mid0: ResidualWeights,
        pub(in crate::cuda_fastwam) attention: AttentionWeights,
        pub(in crate::cuda_fastwam) mid1: ResidualWeights,
        pub(in crate::cuda_fastwam) norm: NormWeights,
        pub(in crate::cuda_fastwam) output: ConvWeights,
        pub(in crate::cuda_fastwam) quant: ConvWeights,
        pub(in crate::cuda_fastwam) mean: Tensor,
        pub(in crate::cuda_fastwam) std: Tensor,
        pub(in crate::cuda_fastwam) dtype: DType,
    }

    impl VaeWeights {
        pub(in crate::cuda_fastwam) fn load(
            ctx: &Context,
            path: &Path,
            dtype: DType,
        ) -> Result<Self> {
            let config: serde_json::Value = serde_json::from_slice(
                &std::fs::read(path.join("config.json"))
                    .map_err(|e| Error::Other(e.to_string()))?,
            )
            .map_err(|e| Error::Other(e.to_string()))?;
            for (name, expected) in [
                ("base_dim", 160),
                ("z_dim", 48),
                ("patch_size", 2),
                ("num_res_blocks", 2),
                ("in_channels", 12),
            ] {
                if config[name].as_u64() != Some(expected) {
                    return Err(Error::Other(format!("unsupported Wan VAE {name}")));
                }
            }
            if config["dim_mult"] != serde_json::json!([1, 2, 4, 4])
                || config["temperal_downsample"] != serde_json::json!([false, true, true])
                || config["is_residual"] != true
            {
                return Err(Error::Other("unsupported Wan VAE block geometry".into()));
            }
            let mut store = Store::load(&path.join("diffusion_pytorch_model.safetensors"))?;
            let mut down = Vec::new();
            let mut channels = 160;
            for (i, output) in [160, 320, 640, 640].into_iter().enumerate() {
                let prefix = format!("encoder.down_blocks.{i}");
                let mut residuals = Vec::new();
                for j in 0..2 {
                    residuals.push(ResidualWeights::load(
                        &mut store,
                        ctx,
                        &format!("{prefix}.resnets.{j}"),
                        dtype,
                        channels,
                        output,
                    )?);
                    channels = output;
                }
                let downsample = if i < 3 {
                    Some(ConvWeights::load(
                        &mut store,
                        ctx,
                        &format!("{prefix}.downsampler.resample.1"),
                        dtype,
                        &[channels, channels, 3, 3],
                        2,
                        0,
                    )?)
                } else {
                    None
                };
                down.push(DownBlockWeights {
                    residuals,
                    downsample,
                    temporal_factor: if i == 1 || i == 2 { 2 } else { 1 },
                });
            }
            let stat = |name: &str| -> Result<Tensor> {
                let values = config[name]
                    .as_array()
                    .ok_or_else(|| Error::Other(format!("VAE missing {name}")))?;
                let values = values
                    .iter()
                    .map(|v| {
                        v.as_f64()
                            .map(|f| f as f32)
                            .filter(|f| f.is_finite())
                            .ok_or_else(|| Error::Other("invalid VAE latent statistic".into()))
                    })
                    .collect::<Result<Vec<_>>>()?;
                if values.len() != 48 || (name == "latents_std" && values.iter().any(|&x| x <= 0.))
                {
                    return Err(Error::Other("invalid VAE latent statistics".into()));
                }
                // The adapter creates these buffers in F32 after loading the VAE;
                // FastWAM moves them to the device without changing their dtype.
                backend::upload(ctx, &host_tensor(vec![48], &values, DType::F32)?)
            };
            Ok(Self {
                input: ConvWeights::load(
                    &mut store,
                    ctx,
                    "encoder.conv_in",
                    dtype,
                    &[160, 12, 3, 3, 3],
                    1,
                    1,
                )?,
                down,
                mid0: ResidualWeights::load(
                    &mut store,
                    ctx,
                    "encoder.mid_block.resnets.0",
                    dtype,
                    640,
                    640,
                )?,
                attention: AttentionWeights::load(
                    &mut store,
                    ctx,
                    "encoder.mid_block.attentions.0",
                    dtype,
                )?,
                mid1: ResidualWeights::load(
                    &mut store,
                    ctx,
                    "encoder.mid_block.resnets.1",
                    dtype,
                    640,
                    640,
                )?,
                norm: NormWeights::load(
                    &mut store,
                    ctx,
                    "encoder.norm_out",
                    dtype,
                    &[640, 1, 1, 1],
                )?,
                output: ConvWeights::load(
                    &mut store,
                    ctx,
                    "encoder.conv_out",
                    dtype,
                    &[96, 640, 3, 3, 3],
                    1,
                    1,
                )?,
                quant: ConvWeights::load(
                    &mut store,
                    ctx,
                    "quant_conv",
                    dtype,
                    &[96, 96, 1, 1, 1],
                    1,
                    0,
                )?,
                mean: stat("latents_mean")?,
                std: stat("latents_std")?,
                dtype,
            })
        }
    }
}

pub(in crate::cuda_fastwam) struct LinearWeights {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}
impl LinearWeights {
    pub(in crate::cuda_fastwam) fn validate(
        &self,
        name: &str,
        input: usize,
        output: usize,
    ) -> Result<()> {
        validate_tensor(name, &self.weight, &[input, output], DType::BF16)?;
        if let Some(bias) = &self.bias {
            validate_tensor(name, bias, &[output], DType::BF16)?;
        }
        Ok(())
    }
}

pub(in crate::cuda_fastwam) fn upload_tensor(
    store: &mut Store,
    ctx: &Context,
    name: &str,
) -> Result<Tensor> {
    backend::upload(ctx, &store.take(name)?)
}

pub(in crate::cuda_fastwam) fn load_linear(
    store: &mut Store,
    ctx: &Context,
    prefix: &str,
    bias: Bias,
) -> Result<LinearWeights> {
    // Validate checkpoint semantics before allocating device weights.
    let (raw, bias) = store.take_linear(prefix, bias)?;
    let (n, k) = (raw.shape().dims()[0], raw.shape().dims()[1]);
    let src = raw.to_f32_vec()?;
    let mut dst = vec![0.; src.len()];
    for j in 0..n {
        for i in 0..k {
            dst[i * n + j] = src[j * k + i];
        }
    }
    let weight = backend::upload(ctx, &host_tensor(vec![k, n], &dst, raw.dtype())?)?;
    let bias = bias
        .as_ref()
        .map(|bias| backend::upload(ctx, bias))
        .transpose()?;
    Ok(LinearWeights { weight, bias })
}

fn cast_weight(ctx: &Context, x: &Tensor, dtype: DType) -> Result<Tensor> {
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

/// Loaded device representations; fixed timestep computation belongs to Blocks.
pub(in crate::cuda_fastwam) struct Bf16Weights {
    pub video: (dit::ExpertWeights, dit::TimeWeights),
    pub action: (dit::ExpertWeights, dit::TimeWeights),
    pub proprio: LinearWeights,
    pub text: text::TextEncoderWeights,
    pub vae: vae::VaeWeights,
    pub rgb_table: Tensor,
}

impl Bf16Weights {
    pub(in crate::cuda_fastwam) fn load(
        ctx: &Context,
        path: &std::path::Path,
        text_path: &std::path::Path,
        vae_path: &std::path::Path,
    ) -> Result<Self> {
        let mut store = Store::load(path)?;
        let video = dit::ExpertWeights::load(&mut store, ctx, true)?;
        let action = dit::ExpertWeights::load(&mut store, ctx, false)?;
        let proprio = load_linear(&mut store, ctx, "model.proprio_encoder", Bias::Required)?;
        proprio.validate("proprio encoder", 8, 4096)?;
        drop(store);
        let rgb = (0..256)
            .map(|v| {
                let unit = half::bf16::from_f32(v as f32 / 255.).to_f32();
                half::bf16::from_f32(half::bf16::from_f32(unit * 2.).to_f32() - 1.).to_f32()
            })
            .collect::<Vec<_>>();
        Ok(Self {
            video,
            action,
            proprio,
            text: text::TextEncoderWeights::load(ctx, text_path)?,
            vae: vae::VaeWeights::load(ctx, vae_path, DType::BF16)?,
            rgb_table: backend::upload(ctx, &host_tensor(vec![256, 1], &rgb, DType::BF16)?)?,
        })
    }
}

pub(in crate::cuda_fastwam) use dit::{ExpertWeights, ProjectionWeights, TimeWeights};
pub(in crate::cuda_fastwam) use text::TextEncoderWeights;
pub(in crate::cuda_fastwam) use vae::{
    AttentionWeights, ConvWeights, DownBlockWeights, NormWeights, ResidualWeights, VaeWeights,
};
