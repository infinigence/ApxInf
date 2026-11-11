//! Strict checkpoint loading. The Talker, code predictor and waveform decoder
//! use BF16 execution weights, as the reference casts every floating tensor;
//! the unused waveform encoder is never read.
use super::config::{codec, *};
use apxinf_core::{Error, Result};
use apxinf_mlx::{Array, MlxDType, Stream};
use std::collections::HashMap;
use std::path::Path;

pub struct Linear {
    pub weight: Array,
    pub bias: Option<Array>,
}

impl Linear {
    /// `nn.Linear`: `addmm(bias, x, W.T)` or `x @ W.T`.
    pub fn apply(&self, x: &Array) -> Result<Array> {
        let w = self.weight.transpose(&[1, 0])?;
        match &self.bias {
            Some(b) => b.addmm(x, &w, 1.0, 1.0),
            None => x.matmul(&w),
        }
    }
}

pub struct Attention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    /// Per-head Q/K RMSNorm weights; the waveform decoder has none.
    pub qk_norm: Option<(Array, Array)>,
}

pub struct Mlp {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

pub struct Layer {
    pub input_norm: Array,
    pub attention: Attention,
    pub post_norm: Array,
    pub mlp: Mlp,
}

pub struct Talker {
    pub text_embedding: Array,
    pub codec_embedding: Array,
    pub text_fc1: Linear,
    pub text_fc2: Linear,
    pub layers: Vec<Layer>,
    pub norm: Array,
    pub codec_head: Linear,
}

pub struct Predictor {
    /// `small_to_mtp_projection` as FP32 operands of the reference's FP32 addmm.
    pub to_mtp_weight: Array,
    pub to_mtp_bias: Array,
    pub codec_embedding: Vec<Array>,
    pub layers: Vec<Layer>,
    pub norm: Array,
    pub heads: Vec<Linear>,
}

/// `nn.Conv1d` or `nn.ConvTranspose1d` in MLX layout `[O, K, I]`, plus bias.
pub struct Conv {
    pub weight: Array,
    pub bias: Array,
}

pub struct Snake {
    pub alpha: Array,
    pub beta: Array,
}

pub struct CodecLayer {
    pub input_norm: Array,
    pub attention: Attention,
    pub attention_scale: Array,
    pub post_norm: Array,
    pub mlp: Mlp,
    pub mlp_scale: Array,
}

pub struct ConvNext {
    pub dwconv: Conv,
    pub norm_weight: Array,
    pub norm_bias: Array,
    pub pwconv1: Linear,
    pub pwconv2: Linear,
    pub gamma: Array,
}

pub struct Residual {
    pub act1: Snake,
    pub conv1: Conv,
    pub act2: Snake,
    pub conv2: Conv,
    pub dilation: usize,
}

pub struct DecoderBlock {
    pub snake: Snake,
    pub upsample: Conv,
    pub rate: usize,
    pub residuals: Vec<Residual>,
}

pub struct Codec {
    /// Codebook rows `[2048, 256]` for the semantic and 15 acoustic quantizers.
    pub codebooks: Vec<Array>,
    pub first_output: Array,
    pub rest_output: Array,
    /// `pre_conv` as FP32 operands of the reference's FP32 convolution.
    pub pre_conv_weight: Array,
    pub pre_conv_bias: Array,
    pub input_proj: Linear,
    pub layers: Vec<CodecLayer>,
    pub norm: Array,
    pub output_proj: Linear,
    pub upsample: Vec<(Conv, ConvNext)>,
    pub initial: Conv,
    pub blocks: Vec<DecoderBlock>,
    pub final_snake: Snake,
    pub final_conv: Conv,
}

struct Take {
    map: HashMap<String, Array>,
    what: &'static str,
    /// Execution weights, evaluated together when loading finishes.
    produced: Vec<Array>,
}

impl Take {
    fn raw(&mut self, name: &str, shape: &[usize]) -> Result<Array> {
        let array = self
            .map
            .remove(name)
            .ok_or_else(|| Error::Other(format!("{} is missing `{name}`", self.what)))?;
        if array.shape() != shape {
            return Err(Error::Other(format!(
                "{} `{name}`: expected {shape:?}, got {:?}",
                self.what,
                array.shape()
            )));
        }
        Ok(array)
    }

    /// A floating tensor as its BF16 execution value.
    fn bf16(&mut self, name: &str, shape: &[usize]) -> Result<Array> {
        let array = self.raw(name, shape)?;
        if !matches!(
            array.dtype(),
            MlxDType::BF16 | MlxDType::F32 | MlxDType::F16
        ) {
            return Err(Error::Other(format!(
                "{} `{name}` is not floating point",
                self.what
            )));
        }
        let array = array.cast(MlxDType::BF16)?;
        self.produced.push(array.clone());
        Ok(array)
    }

    fn linear(&mut self, name: &str, out: usize, input: usize, bias: bool) -> Result<Linear> {
        Ok(Linear {
            weight: self.bf16(&format!("{name}.weight"), &[out, input])?,
            bias: if bias {
                Some(self.bf16(&format!("{name}.bias"), &[out])?)
            } else {
                None
            },
        })
    }

    /// PyTorch `Conv1d` `[O, I, K]` as MLX `[O, K, I]`.
    fn conv(&mut self, name: &str, out: usize, input: usize, kernel: usize) -> Result<Conv> {
        let weight = self.bf16(&format!("{name}.weight"), &[out, input, kernel])?;
        let weight = weight.transpose(&[0, 2, 1])?.contiguous()?;
        self.produced.push(weight.clone());
        Ok(Conv {
            weight,
            bias: self.bf16(&format!("{name}.bias"), &[out])?,
        })
    }

    /// PyTorch `ConvTranspose1d` `[I, O, K]` as MLX `[O, K, I]`.
    fn conv_transpose(
        &mut self,
        name: &str,
        input: usize,
        out: usize,
        kernel: usize,
    ) -> Result<Conv> {
        let weight = self.bf16(&format!("{name}.weight"), &[input, out, kernel])?;
        let weight = weight.transpose(&[1, 2, 0])?.contiguous()?;
        self.produced.push(weight.clone());
        Ok(Conv {
            weight,
            bias: self.bf16(&format!("{name}.bias"), &[out])?,
        })
    }

    fn snake(&mut self, name: &str, channels: usize) -> Result<Snake> {
        Ok(Snake {
            alpha: self.bf16(&format!("{name}.alpha"), &[channels])?,
            beta: self.bf16(&format!("{name}.beta"), &[channels])?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &mut self,
        p: &str,
        hidden: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        norms: bool,
    ) -> Result<Attention> {
        let qk_norm = if norms {
            Some((
                self.bf16(&format!("{p}.q_norm.weight"), &[head_dim])?,
                self.bf16(&format!("{p}.k_norm.weight"), &[head_dim])?,
            ))
        } else {
            None
        };
        Ok(Attention {
            q: self.linear(&format!("{p}.q_proj"), heads * head_dim, hidden, false)?,
            k: self.linear(&format!("{p}.k_proj"), kv_heads * head_dim, hidden, false)?,
            v: self.linear(&format!("{p}.v_proj"), kv_heads * head_dim, hidden, false)?,
            o: self.linear(&format!("{p}.o_proj"), hidden, heads * head_dim, false)?,
            qk_norm,
        })
    }

    fn mlp(&mut self, p: &str, hidden: usize, inner: usize) -> Result<Mlp> {
        Ok(Mlp {
            gate: self.linear(&format!("{p}.gate_proj"), inner, hidden, false)?,
            up: self.linear(&format!("{p}.up_proj"), inner, hidden, false)?,
            down: self.linear(&format!("{p}.down_proj"), hidden, inner, false)?,
        })
    }

    fn layer(&mut self, p: &str, hidden: usize, mlp: usize) -> Result<Layer> {
        Ok(Layer {
            input_norm: self.bf16(&format!("{p}.input_layernorm.weight"), &[hidden])?,
            attention: self.attention(
                &format!("{p}.self_attn"),
                hidden,
                HEADS,
                KV_HEADS,
                HEAD_DIM,
                true,
            )?,
            post_norm: self.bf16(&format!("{p}.post_attention_layernorm.weight"), &[hidden])?,
            mlp: self.mlp(&format!("{p}.mlp"), hidden, mlp)?,
        })
    }

    /// Fail on unrecognized tensors, then evaluate every execution weight
    /// with `extra` derived arrays.
    fn finish(self, stream: &Stream, extra_arrays: &[Array]) -> Result<()> {
        let mut extra: Vec<_> = self.map.into_keys().collect();
        if extra.is_empty() {
            let mut arrays = self.produced;
            arrays.extend_from_slice(extra_arrays);
            return stream.eval(&arrays);
        }
        extra.sort();
        extra.truncate(8);
        Err(Error::Other(format!(
            "{} has unrecognized tensors: {}",
            self.what,
            extra.join(", ")
        )))
    }
}

fn load(
    stream: &Stream,
    file: &Path,
    keep: impl Fn(&str) -> bool,
    what: &'static str,
) -> Result<Take> {
    Ok(Take {
        map: apxinf_mlx::load_safetensors(stream, &[file], keep)?,
        what,
        produced: Vec::new(),
    })
}

impl Talker {
    pub fn load(stream: &Stream, dir: &Path) -> Result<(Self, Predictor)> {
        let mut t = load(
            stream,
            &dir.join("model.safetensors"),
            |_| true,
            "Qwen3-TTS model",
        )?;
        let layers = (0..LAYERS)
            .map(|i| t.layer(&format!("talker.model.layers.{i}"), HIDDEN, MLP))
            .collect::<Result<Vec<_>>>()?;
        let talker = Self {
            text_embedding: t.bf16("talker.model.text_embedding.weight", &[TEXT_VOCAB, HIDDEN])?,
            codec_embedding: t.bf16("talker.model.codec_embedding.weight", &[VOCAB, HIDDEN])?,
            text_fc1: t.linear("talker.text_projection.linear_fc1", HIDDEN, HIDDEN, true)?,
            text_fc2: t.linear("talker.text_projection.linear_fc2", HIDDEN, HIDDEN, true)?,
            layers,
            norm: t.bf16("talker.model.norm.weight", &[HIDDEN])?,
            codec_head: t.linear("talker.codec_head", VOCAB, HIDDEN, false)?,
        };
        let p = "talker.code_predictor";
        let to_mtp = t.linear(
            &format!("{p}.small_to_mtp_projection"),
            PREDICTOR_HIDDEN,
            HIDDEN,
            true,
        )?;
        let predictor = Predictor {
            to_mtp_weight: to_mtp.weight.cast(MlxDType::F32)?,
            to_mtp_bias: to_mtp
                .bias
                .expect("biased projection")
                .cast(MlxDType::F32)?,
            codec_embedding: (0..CODE_GROUPS - 1)
                .map(|i| {
                    t.bf16(
                        &format!("{p}.model.codec_embedding.{i}.weight"),
                        &[PREDICTOR_VOCAB, HIDDEN],
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            layers: (0..PREDICTOR_LAYERS)
                .map(|i| {
                    t.layer(
                        &format!("{p}.model.layers.{i}"),
                        PREDICTOR_HIDDEN,
                        PREDICTOR_MLP,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            norm: t.bf16(&format!("{p}.model.norm.weight"), &[PREDICTOR_HIDDEN])?,
            heads: (0..CODE_GROUPS - 1)
                .map(|i| {
                    t.linear(
                        &format!("{p}.lm_head.{i}"),
                        PREDICTOR_VOCAB,
                        PREDICTOR_HIDDEN,
                        false,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
        };
        t.finish(
            stream,
            &[
                predictor.to_mtp_weight.clone(),
                predictor.to_mtp_bias.clone(),
            ],
        )?;
        Ok((talker, predictor))
    }
}

impl Codec {
    pub fn load(stream: &Stream, dir: &Path) -> Result<Self> {
        let mut t = load(
            stream,
            &dir.join("speech_tokenizer/model.safetensors"),
            |n| n.starts_with("decoder."),
            "Qwen3-TTS waveform decoder",
        )?;
        let q = "decoder.quantizer";
        let half = codec::CODEBOOK_DIM / 2;
        // embedding_sum / clip(cluster_usage, 1e-5) in BF16, as the reference sanitizes it.
        let codebook = |t: &mut Take, p: &str| -> Result<Array> {
            let usage = t.bf16(
                &format!("{p}._codebook.cluster_usage"),
                &[codec::CODEBOOK_SIZE],
            )?;
            let sum = t.bf16(
                &format!("{p}._codebook.embedding_sum"),
                &[codec::CODEBOOK_SIZE, half],
            )?;
            let floor = Array::scalar(usage.stream(), (1e-5f64) as f32, MlxDType::BF16)?;
            sum.div(&usage.reshape(&[codec::CODEBOOK_SIZE, 1])?.maximum(&floor)?)
        };
        let mut codebooks = vec![codebook(&mut t, &format!("{q}.rvq_first.vq.layers.0"))?];
        for i in 0..CODE_GROUPS - 1 {
            codebooks.push(codebook(&mut t, &format!("{q}.rvq_rest.vq.layers.{i}"))?);
        }
        let output = |t: &mut Take, part: &str| -> Result<Array> {
            // The encoder-side input projection is stored but decoding never reads it.
            t.raw(
                &format!("{q}.{part}.input_proj.weight"),
                &[half, codec::CODEBOOK_DIM, 1],
            )?;
            let w = t.bf16(
                &format!("{q}.{part}.output_proj.weight"),
                &[codec::CODEBOOK_DIM, half, 1],
            )?;
            w.transpose(&[0, 2, 1])?.contiguous()
        };
        let first_output = output(&mut t, "rvq_first")?;
        let rest_output = output(&mut t, "rvq_rest")?;
        let pre_conv = t.conv(
            "decoder.pre_conv.conv",
            codec::LATENT,
            codec::CODEBOOK_DIM,
            3,
        )?;
        let p = "decoder.pre_transformer";
        let layers = (0..codec::LAYERS)
            .map(|i| {
                let l = format!("{p}.layers.{i}");
                Ok(CodecLayer {
                    input_norm: t.bf16(&format!("{l}.input_layernorm.weight"), &[codec::HIDDEN])?,
                    attention: t.attention(
                        &format!("{l}.self_attn"),
                        codec::HIDDEN,
                        codec::HEADS,
                        codec::HEADS,
                        codec::HEAD_DIM,
                        false,
                    )?,
                    attention_scale: t.bf16(
                        &format!("{l}.self_attn_layer_scale.scale"),
                        &[codec::HIDDEN],
                    )?,
                    post_norm: t.bf16(
                        &format!("{l}.post_attention_layernorm.weight"),
                        &[codec::HIDDEN],
                    )?,
                    mlp: t.mlp(&format!("{l}.mlp"), codec::HIDDEN, codec::MLP)?,
                    mlp_scale: t.bf16(&format!("{l}.mlp_layer_scale.scale"), &[codec::HIDDEN])?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let input_proj = t.linear(
            &format!("{p}.input_proj"),
            codec::HIDDEN,
            codec::LATENT,
            true,
        )?;
        let norm = t.bf16(&format!("{p}.norm.weight"), &[codec::HIDDEN])?;
        let output_proj = t.linear(
            &format!("{p}.output_proj"),
            codec::LATENT,
            codec::HIDDEN,
            true,
        )?;
        let upsample = (0..codec::UPSAMPLING_RATIOS.len())
            .map(|i| {
                let u = format!("decoder.upsample.{i}");
                let factor = codec::UPSAMPLING_RATIOS[i];
                let n = codec::LATENT;
                Ok((
                    t.conv_transpose(&format!("{u}.0.conv"), n, n, factor)?,
                    ConvNext {
                        dwconv: t.conv(&format!("{u}.1.dwconv.conv"), n, 1, 7)?,
                        norm_weight: t.bf16(&format!("{u}.1.norm.weight"), &[n])?,
                        norm_bias: t.bf16(&format!("{u}.1.norm.bias"), &[n])?,
                        pwconv1: t.linear(&format!("{u}.1.pwconv1"), 4 * n, n, true)?,
                        pwconv2: t.linear(&format!("{u}.1.pwconv2"), n, 4 * n, true)?,
                        gamma: t.bf16(&format!("{u}.1.gamma"), &[n])?,
                    },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let initial = t.conv(
            "decoder.decoder.0.conv",
            codec::DECODER_DIM,
            codec::LATENT,
            7,
        )?;
        let mut blocks = Vec::new();
        for (i, &rate) in codec::UPSAMPLE_RATES.iter().enumerate() {
            let (input, out) = (codec::DECODER_DIM >> i, codec::DECODER_DIM >> (i + 1));
            let b = format!("decoder.decoder.{}.block", i + 1);
            let residuals = [1, 3, 9]
                .iter()
                .enumerate()
                .map(|(j, &dilation)| {
                    let r = format!("{b}.{}", j + 2);
                    Ok(Residual {
                        act1: t.snake(&format!("{r}.act1"), out)?,
                        conv1: t.conv(&format!("{r}.conv1.conv"), out, out, 7)?,
                        act2: t.snake(&format!("{r}.act2"), out)?,
                        conv2: t.conv(&format!("{r}.conv2.conv"), out, out, 1)?,
                        dilation,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            blocks.push(DecoderBlock {
                snake: t.snake(&format!("{b}.0"), input)?,
                upsample: t.conv_transpose(&format!("{b}.1.conv"), input, out, 2 * rate)?,
                rate,
                residuals,
            });
        }
        let last = codec::DECODER_DIM >> codec::UPSAMPLE_RATES.len();
        let codec = Self {
            codebooks,
            first_output,
            rest_output,
            pre_conv_weight: pre_conv.weight.cast(MlxDType::F32)?,
            pre_conv_bias: pre_conv.bias.cast(MlxDType::F32)?,
            input_proj,
            layers,
            norm,
            output_proj,
            upsample,
            initial,
            blocks,
            final_snake: t.snake("decoder.decoder.5", last)?,
            final_conv: t.conv("decoder.decoder.6.conv", 1, last, 7)?,
        };
        let mut derived = codec.codebooks.clone();
        derived.extend([
            codec.first_output.clone(),
            codec.rest_output.clone(),
            codec.pre_conv_weight.clone(),
            codec.pre_conv_bias.clone(),
        ]);
        t.finish(stream, &derived)?;
        Ok(codec)
    }
}
