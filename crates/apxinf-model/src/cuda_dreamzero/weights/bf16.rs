//! BF16 device representations and one-time checkpoint layout transforms.

pub(in crate::cuda_dreamzero) mod text {
    use crate::cuda_dreamzero::{
        backend::{transfers, Context},
        DreamZeroConfig as C,
    };
    use apxinf_core::{Error, Result, Tensor};
    use std::collections::HashMap;

    pub(in crate::cuda_dreamzero) struct TextBlockWeights {
        pub norm1: Tensor,
        pub q: Tensor,
        pub k: Tensor,
        pub v: Tensor,
        pub o: Tensor,
        pub norm2: Tensor,
        pub gate: Tensor,
        pub up: Tensor,
        pub down: Tensor,
        /// Constant learned relative positions, expanded once at load time.
        pub position: Tensor,
    }
    pub(in crate::cuda_dreamzero) struct TextWeights {
        pub embedding: Tensor,
        pub blocks: Vec<TextBlockWeights>,
        pub norm: Tensor,
    }

    /// A layout transformation only: BF16 bits are preserved, with no F32 copy.
    fn transpose(tensor: Tensor) -> Result<Tensor> {
        let dims = tensor.shape().dims();
        if dims.len() != 2 {
            return Err(Error::Other("DreamZero linear must be rank two".into()));
        }
        let (rows, cols) = (dims[0], dims[1]);
        let src = tensor.as_bf16()?;
        let mut dst = vec![half::bf16::ZERO; src.len()];
        // Tile the host-only transpose so large BF16 matrices do not touch a new
        // cache line/page for every scalar store during checkpoint loading.
        for r0 in (0..rows).step_by(32) {
            for c0 in (0..cols).step_by(32) {
                for r in r0..(r0 + 32).min(rows) {
                    for c in c0..(c0 + 32).min(cols) {
                        dst[c * rows + r] = src[r * cols + c];
                    }
                }
            }
        }
        Tensor::from_bf16(vec![cols, rows], &dst)
    }

    fn take(map: &mut HashMap<String, Tensor>, name: &str, shape: &[usize]) -> Result<Tensor> {
        super::super::host::take(map, &format!("action_head.text_encoder.{name}"), shape)
    }

    /// Bidirectional T5 buckets: 16 per direction, exact for distances below 8.
    fn bucket(query: usize, key: usize) -> usize {
        let distance = query.abs_diff(key);
        let magnitude = if distance < 8 {
            distance
        } else {
            (8 + ((distance as f32 / 8.).ln() / 16f32.ln() * 8.) as usize).min(15)
        };
        magnitude + if key > query { 16 } else { 0 }
    }

    impl TextWeights {
        pub fn from_map(ctx: &Context, map: &mut HashMap<String, Tensor>) -> Result<Self> {
            let upload = |tensor: Tensor| transfers::to_cuda(&tensor, ctx.device_id());
            let embedding = upload(take(
                map,
                "token_embedding.weight",
                &[C::TEXT_VOCAB, C::TEXT_DIM],
            )?)?;
            let mut blocks = Vec::with_capacity(C::TEXT_LAYERS);
            for i in 0..C::TEXT_LAYERS {
                let p = format!("blocks.{i}");
                let norm1 = upload(take(map, &format!("{p}.norm1.weight"), &[C::TEXT_DIM])?)?;
                let norm2 = upload(take(map, &format!("{p}.norm2.weight"), &[C::TEXT_DIM])?)?;
                let table = take(
                    map,
                    &format!("{p}.pos_embedding.embedding.weight"),
                    &[32, C::TEXT_HEADS],
                )?;
                let table = table.as_bf16()?;
                let n = C::TEXT_LEN;
                let mut position = vec![half::bf16::ZERO; C::TEXT_HEADS * n * n];
                for h in 0..C::TEXT_HEADS {
                    for q in 0..n {
                        for k in 0..n {
                            position[(h * n + q) * n + k] = table[bucket(q, k) * C::TEXT_HEADS + h];
                        }
                    }
                }
                let position = upload(Tensor::from_bf16(vec![1, C::TEXT_HEADS, n, n], &position)?)?;
                let mut projection = |name: &str, input: usize, output: usize| {
                    upload(transpose(take(
                        map,
                        &format!("{p}.{name}.weight"),
                        &[output, input],
                    )?)?)
                };
                blocks.push(TextBlockWeights {
                    norm1,
                    norm2,
                    position,
                    q: projection("attn.q", C::TEXT_DIM, C::TEXT_DIM)?,
                    k: projection("attn.k", C::TEXT_DIM, C::TEXT_DIM)?,
                    v: projection("attn.v", C::TEXT_DIM, C::TEXT_DIM)?,
                    o: projection("attn.o", C::TEXT_DIM, C::TEXT_DIM)?,
                    gate: projection("ffn.gate.0", C::TEXT_DIM, C::TEXT_FFN)?,
                    up: projection("ffn.fc1", C::TEXT_DIM, C::TEXT_FFN)?,
                    down: projection("ffn.fc2", C::TEXT_FFN, C::TEXT_DIM)?,
                });
            }
            let norm = upload(take(map, "norm.weight", &[C::TEXT_DIM])?)?;
            Ok(Self {
                embedding,
                blocks,
                norm,
            })
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn t5_bucket_boundaries_and_direction() {
            assert_eq!((bucket(0, 0), bucket(0, 1), bucket(1, 0)), (0, 17, 1));
            assert_eq!((bucket(0, 7), bucket(0, 8), bucket(0, 16)), (23, 24, 26));
            assert_eq!((bucket(0, 128), bucket(511, 0)), (31, 15));
        }
        #[test]
        fn projection_transpose_preserves_bf16_bits() {
            let data = [0x8000, 0x3f80, 0x7fc1, 0xbf80, 0x3f81, 0x0001].map(half::bf16::from_bits);
            let x = Tensor::from_bf16(vec![2, 3], &data).unwrap();
            let y = transpose(x).unwrap();
            assert_eq!(y.shape().dims(), &[3, 2]);
            assert_eq!(
                y.as_bf16()
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                [data[0], data[3], data[1], data[4], data[2], data[5]]
                    .map(|v| v.to_bits())
                    .to_vec()
            );
        }
    }
}
pub(in crate::cuda_dreamzero) mod clip {
    use crate::cuda_dreamzero::backend::{transfers, Context};
    use apxinf_core::{Result, Tensor};
    use std::collections::HashMap;

    pub(in crate::cuda_dreamzero) const DIM: usize = 1280;
    pub(in crate::cuda_dreamzero) const TOKENS: usize = 257;
    pub(in crate::cuda_dreamzero) const HEADS: usize = 16;

    pub(in crate::cuda_dreamzero) struct Linear {
        pub weight: Tensor,
        pub bias: Tensor,
    }
    pub(in crate::cuda_dreamzero) struct Norm {
        pub weight: Tensor,
        pub bias: Tensor,
    }
    pub(in crate::cuda_dreamzero) struct ClipBlock {
        pub norm1: Norm,
        pub qkv: Linear,
        pub o: Linear,
        pub norm2: Norm,
        pub up: Linear,
        pub down: Linear,
    }
    pub(in crate::cuda_dreamzero) struct ClipWeights {
        pub patch: Tensor,
        pub class: Tensor,
        pub position: Tensor,
        pub pre_norm: Norm,
        pub blocks: Vec<ClipBlock>,
    }

    fn take(map: &mut HashMap<String, Tensor>, name: &str, shape: &[usize]) -> Result<Tensor> {
        super::super::host::take(
            map,
            &format!("action_head.image_encoder.model.{name}"),
            shape,
        )
    }
    fn upload(ctx: &Context, x: Tensor) -> Result<Tensor> {
        transfers::to_cuda(&x, ctx.device_id())
    }
    fn norm(ctx: &Context, map: &mut HashMap<String, Tensor>, prefix: &str) -> Result<Norm> {
        Ok(Norm {
            weight: upload(ctx, take(map, &format!("{prefix}.weight"), &[DIM])?)?,
            bias: upload(ctx, take(map, &format!("{prefix}.bias"), &[DIM])?)?,
        })
    }
    fn linear(
        ctx: &Context,
        map: &mut HashMap<String, Tensor>,
        prefix: &str,
        input: usize,
        output: usize,
    ) -> Result<Linear> {
        let weight = take(map, &format!("{prefix}.weight"), &[output, input])?;
        Ok(Linear {
            weight: upload(ctx, weight)?,
            bias: upload(ctx, take(map, &format!("{prefix}.bias"), &[output])?)?,
        })
    }
    impl ClipWeights {
        pub fn from_map(ctx: &Context, map: &mut HashMap<String, Tensor>) -> Result<Self> {
            let p = "visual";
            let kernel = take(map, "visual.patch_embedding.weight", &[DIM, 3, 14, 14])?;
            let src = kernel.as_bf16()?;
            // im2col columns are (height,width,channel), checkpoint OIHW.
            // Align channels to 16 to preserve the cuDNN reduction grouping.
            let mut patch = vec![half::bf16::ZERO; 16 * 14 * 14 * DIM];
            for o in 0..DIM {
                for h in 0..14 {
                    for w in 0..14 {
                        for c in 0..3 {
                            patch[((h * 14 + w) * 16 + c) * DIM + o] =
                                src[((o * 3 + c) * 14 + h) * 14 + w];
                        }
                    }
                }
            }
            let patch = upload(ctx, Tensor::from_bf16(vec![14 * 14 * 16, DIM], &patch)?)?;
            let class = upload(
                ctx,
                take(map, "visual.cls_embedding", &[1, 1, DIM])?.reshape(vec![1, DIM])?,
            )?;
            let position = upload(
                ctx,
                take(map, "visual.pos_embedding", &[1, TOKENS, DIM])?.reshape(vec![TOKENS, DIM])?,
            )?;
            let pre_norm = norm(ctx, map, &format!("{p}.pre_norm"))?;
            let mut blocks = Vec::with_capacity(31);
            for i in 0..32 {
                let prefix = format!("{p}.transformer.{i}");
                // Official use_31_block=True deliberately omits the final block.
                // Validate and consume its weights without uploading unused state.
                if i == 31 {
                    for n in ["norm1", "norm2"] {
                        for part in ["weight", "bias"] {
                            take(map, &format!("{prefix}.{n}.{part}"), &[DIM])?;
                        }
                    }
                    for (n, input, output) in [
                        ("attn.to_qkv", DIM, 3 * DIM),
                        ("attn.proj", DIM, DIM),
                        ("mlp.0", DIM, 4 * DIM),
                        ("mlp.2", 4 * DIM, DIM),
                    ] {
                        take(map, &format!("{prefix}.{n}.weight"), &[output, input])?;
                        take(map, &format!("{prefix}.{n}.bias"), &[output])?;
                    }
                    continue;
                }
                blocks.push(ClipBlock {
                    norm1: norm(ctx, map, &format!("{prefix}.norm1"))?,
                    qkv: linear(ctx, map, &format!("{prefix}.attn.to_qkv"), DIM, 3 * DIM)?,
                    o: linear(ctx, map, &format!("{prefix}.attn.proj"), DIM, DIM)?,
                    norm2: norm(ctx, map, &format!("{prefix}.norm2"))?,
                    up: linear(ctx, map, &format!("{prefix}.mlp.0"), DIM, 4 * DIM)?,
                    down: linear(ctx, map, &format!("{prefix}.mlp.2"), 4 * DIM, DIM)?,
                });
            }
            take(map, "log_scale", &[])?;
            take(map, "visual.head", &[DIM, 1024])?;
            take(map, "visual.post_norm.weight", &[DIM])?;
            take(map, "visual.post_norm.bias", &[DIM])?;
            Ok(Self {
                patch,
                class,
                position,
                pre_norm,
                blocks,
            })
        }
    }
}
pub(in crate::cuda_dreamzero) mod dit {
    //! Joint video/action transformer weights. Sampler profiles share this network.
    use crate::cuda_dreamzero::backend::{transfers, Context};
    use apxinf_core::{Result, Tensor};
    use std::collections::HashMap;

    pub(in crate::cuda_dreamzero) const DIM: usize = 5120;
    pub(in crate::cuda_dreamzero) const FFN: usize = 13824;
    pub(in crate::cuda_dreamzero) const HEADS: usize = 40;

    pub(in crate::cuda_dreamzero) struct Linear {
        pub weight: Tensor,
        pub bias: Tensor,
    }
    pub(in crate::cuda_dreamzero) struct Norm {
        pub weight: Tensor,
        pub bias: Tensor,
    }
    pub(in crate::cuda_dreamzero) struct Attention {
        pub q: Linear,
        pub k: Linear,
        pub v: Linear,
        pub o: Linear,
        pub norm_q: Tensor,
        pub norm_k: Tensor,
    }
    pub(in crate::cuda_dreamzero) struct Block {
        pub modulation: Tensor,
        pub self_attention: Attention,
        pub cross_attention: Attention,
        pub image_k: Linear,
        pub image_v: Linear,
        pub image_norm_k: Tensor,
        pub norm3: Norm,
        pub up: Linear,
        pub down: Linear,
    }
    pub(in crate::cuda_dreamzero) struct DitWeights {
        pub state: [Linear; 2],
        pub action: [Linear; 3],
        pub action_decoder: [Linear; 2],
        pub patch: Linear,
        pub text: [Linear; 2],
        pub time: [Linear; 2],
        pub time_projection: Linear,
        pub image_norm1: Norm,
        pub image: [Linear; 2],
        pub image_norm2: Norm,
        pub blocks: Vec<Block>,
        pub head_modulation: Tensor,
        pub head: Linear,
    }

    struct Reader<'a> {
        ctx: &'a Context,
        map: &'a mut HashMap<String, Tensor>,
    }
    impl Reader<'_> {
        fn take(&mut self, name: &str, shape: &[usize]) -> Result<Tensor> {
            super::super::host::take(self.map, &format!("action_head.model.{name}"), shape)
        }
        fn upload(&mut self, name: &str, shape: &[usize]) -> Result<Tensor> {
            let x = self.take(name, shape)?;
            transfers::to_cuda(&x, self.ctx.device_id())
        }
        fn linear(&mut self, name: &str, input: usize, output: usize) -> Result<Linear> {
            let source = self.take(&format!("{name}.weight"), &[output, input])?;
            Ok(Linear {
                weight: transfers::to_cuda(&source, self.ctx.device_id())?,
                bias: self.upload(&format!("{name}.bias"), &[output])?,
            })
        }
        fn category(&mut self, name: &str, input: usize, output: usize) -> Result<Linear> {
            // The released DROID constructor fixes category count to one. These
            // matrices already use IO order; they are not nn.Linear OI weights.
            Ok(Linear {
                weight: self
                    .upload(&format!("{name}.W"), &[1, input, output])?
                    .reshape(vec![input, output])?,
                bias: self
                    .upload(&format!("{name}.b"), &[1, output])?
                    .reshape(vec![output])?,
            })
        }
        fn norm(&mut self, name: &str, dim: usize) -> Result<Norm> {
            Ok(Norm {
                weight: self.upload(&format!("{name}.weight"), &[dim])?,
                bias: self.upload(&format!("{name}.bias"), &[dim])?,
            })
        }
        fn attention(&mut self, name: &str) -> Result<Attention> {
            Ok(Attention {
                q: self.linear(&format!("{name}.q"), DIM, DIM)?,
                k: self.linear(&format!("{name}.k"), DIM, DIM)?,
                v: self.linear(&format!("{name}.v"), DIM, DIM)?,
                o: self.linear(&format!("{name}.o"), DIM, DIM)?,
                norm_q: self.upload(&format!("{name}.norm_q.weight"), &[DIM])?,
                norm_k: self.upload(&format!("{name}.norm_k.weight"), &[DIM])?,
            })
        }
        fn block(&mut self, index: usize) -> Result<Block> {
            let p = format!("blocks.{index}");
            Ok(Block {
                modulation: self
                    .upload(&format!("{p}.modulation"), &[1, 6, DIM])?
                    .reshape(vec![6, DIM])?,
                self_attention: self.attention(&format!("{p}.self_attn"))?,
                cross_attention: self.attention(&format!("{p}.cross_attn"))?,
                image_k: self.linear(&format!("{p}.cross_attn.k_img"), DIM, DIM)?,
                image_v: self.linear(&format!("{p}.cross_attn.v_img"), DIM, DIM)?,
                image_norm_k: self.upload(&format!("{p}.cross_attn.norm_k_img.weight"), &[DIM])?,
                norm3: self.norm(&format!("{p}.norm3"), DIM)?,
                up: self.linear(&format!("{p}.ffn.0"), DIM, FFN)?,
                down: self.linear(&format!("{p}.ffn.2"), FFN, DIM)?,
            })
        }
    }
    impl DitWeights {
        pub fn from_map(ctx: &Context, map: &mut HashMap<String, Tensor>) -> Result<Self> {
            let mut r = Reader { ctx, map };
            let kernel = r.take("patch_embedding.weight", &[DIM, 36, 1, 2, 2])?;
            let src = kernel.as_bf16()?;
            let mut values = vec![half::bf16::ZERO; 4 * 48 * DIM];
            // Unfold columns THWC, checkpoint OITHW. Pad channels to 48 so
            // tensor-core reduction grouping matches the reference convolution.
            for o in 0..DIM {
                for c in 0..36 {
                    for xy in 0..4 {
                        values[(xy * 48 + c) * DIM + o] = src[(o * 36 + c) * 4 + xy];
                    }
                }
            }
            let patch = Linear {
                weight: transfers::to_cuda(
                    &Tensor::from_bf16(vec![4 * 48, DIM], &values)?,
                    ctx.device_id(),
                )?,
                bias: r.upload("patch_embedding.bias", &[DIM])?,
            };
            let mut blocks = Vec::with_capacity(40);
            for i in 0..40 {
                blocks.push(r.block(i)?);
            }
            Ok(Self {
                state: [
                    r.category("state_encoder.layer1", 64, 1024)?,
                    r.category("state_encoder.layer2", 1024, DIM)?,
                ],
                action: [
                    r.category("action_encoder.W1", 32, DIM)?,
                    r.category("action_encoder.W2", 2 * DIM, DIM)?,
                    r.category("action_encoder.W3", DIM, DIM)?,
                ],
                action_decoder: [
                    r.category("action_decoder.layer1", DIM, 1024)?,
                    r.category("action_decoder.layer2", 1024, 32)?,
                ],
                patch,
                text: [
                    r.linear("text_embedding.0", 4096, DIM)?,
                    r.linear("text_embedding.2", DIM, DIM)?,
                ],
                time: [
                    r.linear("time_embedding.0", 256, DIM)?,
                    r.linear("time_embedding.2", DIM, DIM)?,
                ],
                time_projection: r.linear("time_projection.1", DIM, 6 * DIM)?,
                image_norm1: r.norm("img_emb.proj.0", 1280)?,
                image: [
                    r.linear("img_emb.proj.1", 1280, 1280)?,
                    r.linear("img_emb.proj.3", 1280, DIM)?,
                ],
                image_norm2: r.norm("img_emb.proj.4", DIM)?,
                blocks,
                head_modulation: r
                    .upload("head.modulation", &[1, 2, DIM])?
                    .reshape(vec![2, DIM])?,
                head: r.linear("head.head", DIM, 64)?,
            })
        }
    }

    #[cfg(test)]
    impl Block {
        pub(in crate::cuda_dreamzero) fn from_map(
            ctx: &Context,
            map: &mut HashMap<String, Tensor>,
            index: usize,
        ) -> Result<Self> {
            Reader { ctx, map }.block(index)
        }
    }
}
pub(in crate::cuda_dreamzero) mod vae {
    use crate::cuda_dreamzero::backend::{transfers, Context};
    use apxinf_core::{Result, Tensor};
    use std::collections::HashMap;

    pub(in crate::cuda_dreamzero) struct Conv {
        pub input: usize,
        pub padded_input: usize,
        pub channel_block: usize,
        pub output: usize,
        pub kernel: [usize; 3],
        pub weight: Tensor,
        pub native_weight: Option<Tensor>,
        pub bias: Tensor,
    }
    pub(in crate::cuda_dreamzero) struct Residual {
        pub norm1: Tensor,
        pub conv1: Conv,
        pub norm2: Tensor,
        pub conv2: Conv,
        pub shortcut: Option<Conv>,
    }
    pub(in crate::cuda_dreamzero) enum Down {
        Residual(Residual),
        Spatial { conv: Conv, temporal: Option<Conv> },
    }
    pub(in crate::cuda_dreamzero) struct Attention {
        pub norm: Tensor,
        pub q: Conv,
        pub k: Conv,
        pub v: Conv,
        pub out: Conv,
    }
    pub(in crate::cuda_dreamzero) struct VaeWeights {
        pub input: Conv,
        pub down: Vec<Down>,
        pub middle1: Residual,
        pub attention: Attention,
        pub middle2: Residual,
        pub final_norm: Tensor,
        pub final_conv: Conv,
        pub moments: Conv,
        pub mean: Tensor,
        pub inverse_std: Tensor,
    }
    fn take(map: &mut HashMap<String, Tensor>, name: &str, shape: &[usize]) -> Result<Tensor> {
        super::super::host::take(map, &format!("action_head.vae.model.{name}"), shape)
    }
    fn upload(ctx: &Context, x: Tensor) -> Result<Tensor> {
        transfers::to_cuda(&x, ctx.device_id())
    }
    fn norm(
        ctx: &Context,
        map: &mut HashMap<String, Tensor>,
        name: &str,
        dim: usize,
        images: bool,
    ) -> Result<Tensor> {
        let shape = if images {
            vec![dim, 1, 1]
        } else {
            vec![dim, 1, 1, 1]
        };
        upload(ctx, take(map, name, &shape)?.reshape(vec![1, dim])?)
    }
    fn pack(
        ctx: &Context,
        data: &[half::bf16],
        input: usize,
        output: usize,
        kernel: [usize; 3],
        bias: Tensor,
        channel_block: usize,
    ) -> Result<Conv> {
        let taps: usize = kernel.iter().product();
        let padded_input = input.div_ceil(channel_block) * channel_block;
        let columns = padded_input * taps;
        let mut matrix = vec![half::bf16::ZERO; output * columns];
        for o in 0..output {
            for c in 0..input {
                for tap in 0..taps {
                    let k = ((c / channel_block) * taps + tap) * channel_block + c % channel_block;
                    matrix[k * output + o] = data[(o * input + c) * taps + tap];
                }
            }
        }
        Ok(Conv {
            native_weight: None,
            input,
            padded_input,
            channel_block,
            output,
            kernel,
            weight: upload(ctx, Tensor::from_bf16(vec![columns, output], &matrix)?)?,
            bias: upload(ctx, bias.reshape(vec![1, output])?)?,
        })
    }
    fn conv(
        ctx: &Context,
        map: &mut HashMap<String, Tensor>,
        name: &str,
        input: usize,
        output: usize,
        kernel: [usize; 3],
        is_2d: bool,
    ) -> Result<Conv> {
        let shape = if is_2d {
            vec![output, input, kernel[1], kernel[2]]
        } else {
            vec![output, input, kernel[0], kernel[1], kernel[2]]
        };
        let weight = take(map, &format!("{name}.weight"), &shape)?;
        let bias = take(map, &format!("{name}.bias"), &[output])?;
        let mut result = pack(
            ctx,
            weight.as_bf16()?,
            input,
            output,
            kernel,
            bias,
            input.min(64),
        )?;
        // The RGB stem and four 88x160 convolutions require the reference's
        // convolution reduction order at BF16 rounding boundaries.
        // Preserve OIDHW weights for the native convolution provider; remaining
        // layers use the aligned unfold/GEMM representation.
        if kernel == [3; 3]
            && (name == "encoder.conv1"
                || name.starts_with("encoder.downsamples.6.")
                || name.starts_with("encoder.downsamples.7."))
        {
            result.native_weight = Some(upload(ctx, weight)?);
        }
        Ok(result)
    }
    fn residual(
        ctx: &Context,
        map: &mut HashMap<String, Tensor>,
        name: &str,
        input: usize,
        output: usize,
    ) -> Result<Residual> {
        Ok(Residual {
            norm1: norm(ctx, map, &format!("{name}.residual.0.gamma"), input, false)?,
            conv1: conv(
                ctx,
                map,
                &format!("{name}.residual.2"),
                input,
                output,
                [3; 3],
                false,
            )?,
            norm2: norm(ctx, map, &format!("{name}.residual.3.gamma"), output, false)?,
            conv2: conv(
                ctx,
                map,
                &format!("{name}.residual.6"),
                output,
                output,
                [3; 3],
                false,
            )?,
            shortcut: if input != output {
                Some(conv(
                    ctx,
                    map,
                    &format!("{name}.shortcut"),
                    input,
                    output,
                    [1; 3],
                    false,
                )?)
            } else {
                None
            },
        })
    }
    impl VaeWeights {
        pub fn from_map(ctx: &Context, map: &mut HashMap<String, Tensor>) -> Result<Self> {
            let input = conv(ctx, map, "encoder.conv1", 3, 96, [3; 3], false)?;
            let mut down = Vec::new();
            let mut index = 0;
            let mut input_dim = 96;
            for (stage, dim) in [96, 192, 384, 384].into_iter().enumerate() {
                for _ in 0..2 {
                    down.push(Down::Residual(residual(
                        ctx,
                        map,
                        &format!("encoder.downsamples.{index}"),
                        input_dim,
                        dim,
                    )?));
                    input_dim = dim;
                    index += 1;
                }
                if stage < 3 {
                    let prefix = format!("encoder.downsamples.{index}");
                    down.push(Down::Spatial {
                        conv: conv(
                            ctx,
                            map,
                            &format!("{prefix}.resample.1"),
                            dim,
                            dim,
                            [1, 3, 3],
                            true,
                        )?,
                        temporal: if stage > 0 {
                            Some(conv(
                                ctx,
                                map,
                                &format!("{prefix}.time_conv"),
                                dim,
                                dim,
                                [3, 1, 1],
                                false,
                            )?)
                        } else {
                            None
                        },
                    });
                    index += 1;
                }
            }
            let middle1 = residual(ctx, map, "encoder.middle.0", 384, 384)?;
            let qkv = take(map, "encoder.middle.1.to_qkv.weight", &[1152, 384, 1, 1])?;
            let bias = take(map, "encoder.middle.1.to_qkv.bias", &[1152])?;
            let split = |i: usize| {
                pack(
                    ctx,
                    &qkv.as_bf16()?[i * 384 * 384..(i + 1) * 384 * 384],
                    384,
                    384,
                    [1; 3],
                    Tensor::from_bf16(vec![384], &bias.as_bf16()?[i * 384..(i + 1) * 384])?,
                    64,
                )
            };
            let attention = Attention {
                norm: norm(ctx, map, "encoder.middle.1.norm.gamma", 384, true)?,
                q: split(0)?,
                k: split(1)?,
                v: split(2)?,
                out: conv(ctx, map, "encoder.middle.1.proj", 384, 384, [1; 3], true)?,
            };
            let middle2 = residual(ctx, map, "encoder.middle.2", 384, 384)?;
            let final_norm = norm(ctx, map, "encoder.head.0.gamma", 384, false)?;
            let final_conv = conv(ctx, map, "encoder.head.2", 384, 32, [3; 3], false)?;
            let moments = conv(ctx, map, "conv1", 32, 32, [1; 3], false)?;
            let mean = [
                -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134,
                -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
            ];
            let std = [
                2.8184f32, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526,
                2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
            ];
            let constant =
                |values: Vec<half::bf16>| upload(ctx, Tensor::from_bf16(vec![1, 16], &values)?);
            let mean = constant(mean.into_iter().map(half::bf16::from_f32).collect())?;
            let inverse_std = constant(
                std.into_iter()
                    .map(|s| half::bf16::from_f32(1. / s))
                    .collect(),
            )?;
            Ok(Self {
                input,
                down,
                middle1,
                attention,
                middle2,
                final_norm,
                final_conv,
                moments,
                mean,
                inverse_std,
            })
        }
    }
}
use self::text::TextWeights;
use crate::cuda_dreamzero::backend::Context;
use apxinf_core::Result;
use std::path::Path;

pub(in crate::cuda_dreamzero) struct Bf16Weights {
    pub text: TextWeights,
    pub clip: clip::ClipWeights,
    pub vae: vae::VaeWeights,
    pub dit: dit::DitWeights,
}
impl Bf16Weights {
    pub fn from_safetensors(ctx: &Context, path: &Path) -> Result<Self> {
        let mut map = super::host::load(path)?;
        let weights = Self {
            text: TextWeights::from_map(ctx, &mut map)?,
            clip: clip::ClipWeights::from_map(ctx, &mut map)?,
            vae: vae::VaeWeights::from_map(ctx, &mut map)?,
            dit: dit::DitWeights::from_map(ctx, &mut map)?,
        };
        super::host::finish(&map)?;
        Ok(weights)
    }
}
