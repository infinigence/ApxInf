//! LFM3 checkpoint mapping and canonical weight geometry.
//!
//! Every transformation from the Hugging Face checkpoint happens once here:
//! linear weights become canonical `[in, out]` views of the immutable `[out, in]`
//! rows, and the depthwise conv table is flattened from `[C, 1, K]` to `[C, K]`
//! and widened to FP32, which is the dtype the reference convolution multiplies
//! in. The convolution weight layout is the one adaptation this family needs, so
//! it is recorded here and replayed against captured reference tensors.
//!
//! QUEUE #8 adds a second representation: with `APXINF_LFM3_4BIT=1` and the
//! quantized triple present, a projection runs MLX's affine group-64
//! `quantized_matmul` instead of BF16 matmul. The default (unset) arm never
//! inspects quantized keys and is byte-for-byte the original path.

use super::config::Lfm3Config;
use apxinf_core::{DType, Device, Error, Result, Tensor};
use apxinf_mlx::{Array, MlxDType, Stream};
use std::collections::HashMap;

/// One HF linear projection. The BF16 path keeps a canonical `[in, out]`
/// view; the QUEUE #8 4-bit path (when armed and the quantized triple is
/// present) instead holds MLX affine group-64 weights and runs
/// `quantized_matmul` — the same op the Python lane's packed4 engine uses.
#[derive(Clone)]
pub(crate) enum Linear {
    Bf16 { canonical: Array },
    Q4(super::q4::Q4),
}

impl Linear {
    pub fn call(&self, x: &Array) -> Result<Array> {
        match self {
            Linear::Bf16 { canonical } => x.matmul(canonical),
            Linear::Q4(q4) => q4.apply(x),
        }
    }

    /// Canonical BF16 `[in, out]` weight, or an error for a 4-bit projection.
    /// The ANE column split needs the dense checkpoint weights; combining it
    /// with `APXINF_LFM3_4BIT` is not a supported configuration.
    #[cfg(feature = "coreml")]
    pub(crate) fn bf16(&self) -> Result<&Array> {
        match self {
            Linear::Bf16 { canonical } => Ok(canonical),
            Linear::Q4(_) => Err(Error::Other(
                "LFM3 ANE needs BF16 projections; APXINF_LFM3_4BIT is not supported with APXINF_LFM3_ANE"
                    .into(),
            )),
        }
    }

    /// This projection restricted to output columns `[start, end)`.
    #[cfg(feature = "coreml")]
    pub(crate) fn output_slice(&self, start: usize, end: usize) -> Result<Self> {
        Ok(Self::Bf16 {
            canonical: self.bf16()?.slice_axis(1, start, end)?.contiguous()?,
        })
    }

    /// This projection restricted to input columns `[start, end)`.
    #[cfg(feature = "coreml")]
    pub(crate) fn input_slice(&self, start: usize, end: usize) -> Result<Self> {
        Ok(Self::Bf16 {
            canonical: self.bf16()?.slice_axis(0, start, end)?.contiguous()?,
        })
    }

    /// All arrays to warm at load: the BF16 operand, or the Q4 triple.
    pub fn warm_arrays(&self) -> Vec<Array> {
        match self {
            Linear::Bf16 { canonical } => vec![canonical.clone()],
            Linear::Q4(q4) => q4.arrays().to_vec(),
        }
    }
}

/// The layer's non-residual operator: exactly one of the two branches exists.
#[derive(Clone)]
pub(crate) enum Mixer {
    Conv {
        in_proj: Linear,
        out_proj: Linear,
        /// `[hidden, kernel]` FP32 depthwise tap table (checkpoint stores `[C, 1, K]`).
        taps: Array,
        /// The same table as BF16 for the fused decode kernel. The checkpoint
        /// stores BF16 taps and `taps` above is their exact F32 widening, so this
        /// cast restores the checkpoint's bytes: it quantizes nothing new.
        taps_bf16: Array,
    },
    Attention {
        q: Linear,
        k: Linear,
        v: Linear,
        o: Linear,
        /// Per-head RMSNorm weight over `head_dim`.
        q_norm: Array,
        k_norm: Array,
        /// `hidden -> num_attention_heads` per-head output gate.
        gate: Linear,
    },
}

#[derive(Clone)]
pub(crate) struct LayerWeights {
    pub operator_norm: Array,
    pub ffn_norm: Array,
    pub post_operator_norm: Array,
    pub post_ffn_norm: Array,
    pub w1: Linear,
    pub w3: Linear,
    pub w2: Linear,
    pub mixer: Mixer,
}

/// Tied embedding/lm-head: BF16 gather table plus its precomputed `[H, V]`
/// logit view (HEAD's `head` field), or QUEUE #8's quantized `embed_tokens`
/// triple. `take` / the logits dispatch select the representation.
#[derive(Clone)]
enum Embedding {
    Bf16 { table: Array, head: Array },
    Q4(super::q4::Q4),
}

impl Embedding {
    fn take(&self, ids: &Array) -> Result<Array> {
        match self {
            Embedding::Bf16 { table, .. } => table.take(ids, 0),
            // Gather the packed rows first, then dequantize just those — the
            // input side of the tied table. Packed rows are `[in/8]` U32
            // words, so a plain `take` on the packed weight alone would drop
            // the matching scales/biases; `take_rows` keeps all three in
            // lockstep.
            Embedding::Q4(q4) => q4.take_rows(ids),
        }
    }
    fn warm_arrays(&self) -> Vec<Array> {
        match self {
            Embedding::Bf16 { table, head } => vec![table.clone(), head.clone()],
            Embedding::Q4(q4) => q4.arrays().to_vec(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct Weights {
    /// Tied embedding/lm-head table: gather rows for input, transpose for logits.
    embedding: Embedding,
    pub final_norm: Array,
    pub layers: Vec<LayerWeights>,
    /// True when at least one projection was loaded as Q4 (gate-0 census).
    pub quantized_layers: usize,
}

fn take(
    map: &mut HashMap<String, Tensor>,
    name: &str,
    shape: &[usize],
    stream: &Stream,
) -> Result<Array> {
    let tensor = map
        .remove(name)
        .ok_or_else(|| Error::Other(format!("LFM3 missing weight `{name}`")))?;
    if tensor.shape().dims() != shape {
        return Err(Error::ShapeMismatch {
            expected: format!("{name} {shape:?}"),
            got: format!("{:?}", tensor.shape()),
        });
    }
    if tensor.device() != Device::Cpu {
        return Err(Error::UnsupportedDevice(tensor.device()));
    }
    if tensor.dtype() != DType::BF16 {
        return Err(Error::UnsupportedDType {
            got: tensor.dtype(),
            allowed: "LFM3 checkpoint BF16",
        });
    }
    let bytes = tensor
        .storage()
        .as_cpu()
        .ok_or_else(|| Error::Other("weight has no CPU storage".into()))?;
    Array::from_bytes(stream, shape, MlxDType::BF16, bytes)
}

/// Canonical `[in, out]` view of the checkpoint's immutable `[out, in]` rows,
/// or — when the QUEUE #8 switch is armed and the quantized triple is present
/// in `map` — the MLX affine group-64 4-bit weights. `use_4bit` is `false` on
/// the default (unset) arm, so that path never even inspects quantized keys
/// (empty-arm gate).
fn linear(
    map: &mut HashMap<String, Tensor>,
    name: &str,
    input: usize,
    output: usize,
    stream: &Stream,
    use_4bit: bool,
) -> Result<Linear> {
    if use_4bit {
        if let Some(q4) = super::q4::take(map, name, input, output, stream)? {
            return Ok(Linear::Q4(q4));
        }
    }
    Ok(Linear::Bf16 {
        canonical: take(map, name, &[output, input], stream)?.transpose(&[1, 0])?,
    })
}

/// Per-head RMSNorm weights stay BF16 so the family uses the same weighted
/// native RMSNorm as the verified reference path.
fn norm(
    map: &mut HashMap<String, Tensor>,
    name: &str,
    size: usize,
    stream: &Stream,
) -> Result<Array> {
    take(map, name, &[size], stream)
}

impl Weights {
    pub fn load(
        config: &Lfm3Config,
        mut map: HashMap<String, Tensor>,
        stream: &Stream,
    ) -> Result<Self> {
        let h = config.hidden_size;
        let q = config.num_attention_heads * config.head_dim;
        let kv = config.num_key_value_heads * config.head_dim;
        // The default (unset) arm never looks for quantized keys.
        let use_4bit = super::q4::enabled();
        let mut quantized_layers = 0usize;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let kind = config.layer_kind(i)?;
            let p = format!("model.layers.{i}");
            let mut n = |suffix: &str, size: usize| {
                norm(&mut map, &format!("{p}.{suffix}.weight"), size, stream)
            };
            let operator_norm = n("operator_norm", h)?;
            let ffn_norm = n("ffn_norm", h)?;
            let post_operator_norm = n("post_operator_norm", h)?;
            let post_ffn_norm = n("post_ffn_norm", h)?;
            let w1 = linear(
                &mut map,
                &format!("{p}.feed_forward.w1.weight"),
                h,
                config.intermediate_size,
                stream,
                use_4bit,
            )?;
            let w3 = linear(
                &mut map,
                &format!("{p}.feed_forward.w3.weight"),
                h,
                config.intermediate_size,
                stream,
                use_4bit,
            )?;
            let w2 = linear(
                &mut map,
                &format!("{p}.feed_forward.w2.weight"),
                config.intermediate_size,
                h,
                stream,
                use_4bit,
            )?;
            let mixer = if kind.is_conv() {
                // The depthwise tap table is stored as `[C, 1, K]` in the
                // BF16 checkpoint and `[C, K, 1]` in the Hugging Face MLX
                // quantized export (the quantizer reorders the singleton
                // axis). The flattened `[C, K]` result is identical; the
                // default (unset) arm keeps the original `[C, 1, K]` lookup
                // so its path is unchanged byte-for-byte.
                let conv_key = format!("{p}.conv.conv.weight");
                let conv_shape: [usize; 3] = if use_4bit
                    && map
                        .get(&conv_key)
                        .is_some_and(|t| t.shape().dims() == [h, config.conv_l_cache, 1])
                {
                    [h, config.conv_l_cache, 1]
                } else {
                    [h, 1, config.conv_l_cache]
                };
                let taps = take(&mut map, &conv_key, &conv_shape, stream)?
                    .reshape(&[h, config.conv_l_cache])?
                    .cast(MlxDType::F32)?;
                // BF16 -> F32 is exact, so this round trip reproduces the
                // checkpoint's tap values for the fused kernel's BF16 input.
                let taps_bf16 = taps.cast(MlxDType::BF16)?;
                let in_proj = linear(
                    &mut map,
                    &format!("{p}.conv.in_proj.weight"),
                    h,
                    3 * h,
                    stream,
                    use_4bit,
                )?;
                let out_proj = linear(
                    &mut map,
                    &format!("{p}.conv.out_proj.weight"),
                    h,
                    h,
                    stream,
                    use_4bit,
                )?;
                for linear in [&w1, &w3, &w2, &in_proj, &out_proj] {
                    quantized_layers += usize::from(matches!(linear, Linear::Q4(_)));
                }
                Mixer::Conv {
                    in_proj,
                    out_proj,
                    taps,
                    taps_bf16,
                }
            } else {
                // `q` above is the projection width `heads × head_dim`; the
                // per-projection bindings are hoisted so their Q4 census can
                // run before the mixer is built (HEAD inlined these in the
                // struct literal and never shadowed it).
                let q_in = q;
                let q = linear(
                    &mut map,
                    &format!("{p}.self_attn.q_proj.weight"),
                    h,
                    q_in,
                    stream,
                    use_4bit,
                )?;
                let k = linear(
                    &mut map,
                    &format!("{p}.self_attn.k_proj.weight"),
                    h,
                    kv,
                    stream,
                    use_4bit,
                )?;
                let v = linear(
                    &mut map,
                    &format!("{p}.self_attn.v_proj.weight"),
                    h,
                    kv,
                    stream,
                    use_4bit,
                )?;
                let o = linear(
                    &mut map,
                    &format!("{p}.self_attn.o_proj.weight"),
                    q_in,
                    h,
                    stream,
                    use_4bit,
                )?;
                let gate = linear(
                    &mut map,
                    &format!("{p}.self_attn.gate_proj.weight"),
                    h,
                    config.num_attention_heads,
                    stream,
                    use_4bit,
                )?;
                for linear in [&w1, &w3, &w2, &q, &k, &v, &o, &gate] {
                    quantized_layers += usize::from(matches!(linear, Linear::Q4(_)));
                }
                Mixer::Attention {
                    q,
                    k,
                    v,
                    o,
                    q_norm: norm(
                        &mut map,
                        &format!("{p}.self_attn.q_norm.weight"),
                        config.head_dim,
                        stream,
                    )?,
                    k_norm: norm(
                        &mut map,
                        &format!("{p}.self_attn.k_norm.weight"),
                        config.head_dim,
                        stream,
                    )?,
                    gate,
                }
            };
            layers.push(LayerWeights {
                operator_norm,
                ffn_norm,
                post_operator_norm,
                post_ffn_norm,
                w1,
                w3,
                w2,
                mixer,
            });
        }
        // The checkpoint ships no `lm_head`: the token table is tied, and the
        // logits projection is its transpose. A present head must be identical.
        if let Some(head) = map.remove("lm_head.weight") {
            let embed = map
                .get("model.embed_tokens.weight")
                .ok_or_else(|| Error::Other("LFM3 missing tied embedding".into()))?;
            if head.shape() != embed.shape()
                || head.dtype() != embed.dtype()
                || head.storage().as_cpu() != embed.storage().as_cpu()
            {
                return Err(Error::Other(
                    "LFM3 tied lm_head differs from embedding checkpoint bytes".into(),
                ));
            }
        }
        let embedding = if use_4bit {
            if let Some(q4) = super::q4::take(
                &mut map,
                "model.embed_tokens.weight",
                h,
                config.vocab_size,
                stream,
            )? {
                Embedding::Q4(q4)
            } else {
                let table = take(
                    &mut map,
                    "model.embed_tokens.weight",
                    &[config.vocab_size, h],
                    stream,
                )?;
                let head = table.transpose(&[1, 0])?;
                Embedding::Bf16 { table, head }
            }
        } else {
            // Default path: identical to HEAD (table + one-time head view).
            let table = take(
                &mut map,
                "model.embed_tokens.weight",
                &[config.vocab_size, h],
                stream,
            )?;
            let head = table.transpose(&[1, 0])?;
            Embedding::Bf16 { table, head }
        };
        // The tied table is the 154th quantized triple in the official
        // export (153 layer projections + this one).
        quantized_layers += usize::from(matches!(embedding, Embedding::Q4(_)));
        let final_norm = norm(&mut map, "model.embedding_norm.weight", h, stream)?;
        // Any leftover tensor means this is not the architecture analyzed here.
        if !map.is_empty() {
            let mut names: Vec<_> = map.keys().cloned().collect();
            names.sort();
            names.truncate(6);
            return Err(Error::Other(format!(
                "unconsumed LFM3 checkpoint tensors: {}",
                names.join(", ")
            )));
        }
        let mut arrays = embedding.warm_arrays();
        arrays.push(final_norm.clone());
        for layer in &layers {
            arrays.extend([
                layer.operator_norm.clone(),
                layer.ffn_norm.clone(),
                layer.post_operator_norm.clone(),
                layer.post_ffn_norm.clone(),
            ]);
            arrays.extend(layer.w1.warm_arrays());
            arrays.extend(layer.w2.warm_arrays());
            arrays.extend(layer.w3.warm_arrays());
            match &layer.mixer {
                Mixer::Conv {
                    in_proj,
                    out_proj,
                    taps,
                    taps_bf16,
                } => {
                    arrays.extend(in_proj.warm_arrays());
                    arrays.extend(out_proj.warm_arrays());
                    arrays.extend([taps.clone(), taps_bf16.clone()]);
                }
                Mixer::Attention {
                    q,
                    k,
                    v,
                    o,
                    q_norm,
                    k_norm,
                    gate,
                } => {
                    arrays.extend(q.warm_arrays());
                    arrays.extend(k.warm_arrays());
                    arrays.extend(v.warm_arrays());
                    arrays.extend(o.warm_arrays());
                    arrays.extend(gate.warm_arrays());
                    arrays.extend([q_norm.clone(), k_norm.clone()]);
                }
            }
        }
        let mut arrays = arrays;
        arrays.shrink_to_fit();
        stream.eval(&arrays)?;
        Ok(Self {
            embedding,
            final_norm,
            layers,
            quantized_layers,
        })
    }

    pub fn embed(&self, ids: &Array) -> Result<Array> {
        self.embedding.take(ids)
    }

    pub fn project(&self, x: &Array) -> Result<Array> {
        match &self.embedding {
            Embedding::Bf16 { head, .. } => x.matmul(head),
            Embedding::Q4(q4) => q4.apply(x),
        }
    }
}
