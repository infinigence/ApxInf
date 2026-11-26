use super::config::{Config, Variant};
use apxinf_core::{DType, Error, Result, Tensor};
use apxinf_mlx::{Array, MlxDType, Stream};
use std::collections::HashMap;

/// MLX affine group size for the decode copies.
const GROUP: i32 = 64;

/// Private MLX affine packing of one projection, derived once from the
/// canonical weight. `APXINF_MINICPM5_W8_BITS` selects the decode width
/// (8 default; 4 and 6 accepted).
#[derive(Clone)]
struct Quantized {
    packed: Array,
    scales: Array,
    biases: Array,
    bits: i32,
}

impl Quantized {
    fn new(hf_weight: &Array) -> Result<Self> {
        let bits: i32 = match std::env::var("APXINF_MINICPM5_W8_BITS").ok().as_deref() {
            Some("4") => 4,
            Some("6") => 6,
            _ => 8,
        };
        let [packed, scales, biases] = hf_weight.quantize(GROUP, bits)?;
        Ok(Self {
            packed,
            scales,
            biases,
            bits,
        })
    }

    fn project(&self, x: &Array) -> Result<Array> {
        x.quantized_matmul(
            &self.packed,
            &self.scales,
            &self.biases,
            true,
            GROUP,
            self.bits,
        )
    }

    fn arrays(&self) -> [Array; 3] {
        [
            self.packed.clone(),
            self.scales.clone(),
            self.biases.clone(),
        ]
    }
}

/// One projection: the canonical BF16 lane for prompt GEMMs plus, in the
/// `mixed-w8` variant, an affine decode copy used when the lane is one row.
#[derive(Clone)]
pub struct Linear {
    /// Canonical [in, out] view of the checkpoint's immutable [out, in]
    /// backing. Keep the private view: forcing physical contiguity can select
    /// a different MLX M=1 reduction.
    pub canonical: Array,
    packed: Option<Quantized>,
}

impl Linear {
    fn new(canonical: Array, mixed: bool) -> Result<Self> {
        let packed = if mixed {
            Some(Quantized::new(&canonical.transpose(&[1, 0])?)?)
        } else {
            None
        };
        Ok(Self { canonical, packed })
    }

    /// Single-row lanes use the decode copy; prompt GEMMs keep BF16.
    /// `preserve_m1` keeps the tapped-verification reduction shape on the
    /// BF16 lane.
    pub fn project(&self, x: &Array, preserve_m1: bool) -> Result<Array> {
        if x.shape()[..x.shape().len() - 1].iter().product::<usize>() == 1 {
            if let Some(packed) = &self.packed {
                return packed.project(x);
            }
        }
        super::math::project(x, &self.canonical, preserve_m1)
    }

    fn arrays(&self) -> Vec<Array> {
        let mut arrays = vec![self.canonical.clone()];
        if let Some(packed) = &self.packed {
            arrays.extend(packed.arrays());
        }
        arrays
    }
}

#[derive(Clone)]
pub struct LayerWeights {
    pub input_norm: Array,
    pub post_norm: Array,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
}

impl LayerWeights {
    fn arrays(&self) -> Vec<Array> {
        let mut arrays = vec![self.input_norm.clone(), self.post_norm.clone()];
        for linear in [
            &self.q, &self.k, &self.v, &self.o, &self.gate, &self.up, &self.down,
        ] {
            arrays.extend(linear.arrays());
        }
        arrays
    }
}

pub struct Weights {
    pub embedding: Array,
    pub norm: Array,
    pub head: Linear,
    pub layers: Vec<LayerWeights>,
}

fn take(
    map: &mut HashMap<String, Tensor>,
    stream: &Stream,
    name: &str,
    shape: &[usize],
) -> Result<Array> {
    let t = map
        .remove(name)
        .ok_or_else(|| Error::Other(format!("missing MiniCPM5 weight {name}")))?;
    if t.dtype() != DType::BF16 || t.shape().dims() != shape {
        return Err(Error::Other(format!(
            "MiniCPM5 weight {name}: expected BF16 {shape:?}, got {} {}",
            t.dtype(),
            t.shape()
        )));
    }
    Array::from_bytes(
        stream,
        shape,
        MlxDType::BF16,
        t.storage()
            .as_cpu()
            .ok_or(Error::Contract("checkpoint weights must be host tensors"))?,
    )
}

impl Weights {
    pub fn load(
        config: &Config,
        variant: Variant,
        stream: &Stream,
        mut map: HashMap<String, Tensor>,
    ) -> Result<Self> {
        let mixed = variant == Variant::MixedW8;
        let embedding = take(
            &mut map,
            stream,
            "model.embed_tokens.weight",
            &[config.vocab_size, config.hidden_size],
        )?;
        let norm = take(&mut map, stream, "model.norm.weight", &[config.hidden_size])?;
        let head = Linear::new(
            take(
                &mut map,
                stream,
                "lm_head.weight",
                &[config.vocab_size, config.hidden_size],
            )?
            .transpose(&[1, 0])?,
            mixed,
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            let mut matrix = |suffix: &str, out: usize, input: usize| -> Result<Linear> {
                Linear::new(
                    take(
                        &mut map,
                        stream,
                        &format!("{prefix}.{suffix}.weight"),
                        &[out, input],
                    )?
                    .transpose(&[1, 0])?,
                    mixed,
                )
            };
            let q = matrix("self_attn.q_proj", 2048, 2048)?;
            let k = matrix("self_attn.k_proj", 256, 2048)?;
            let v = matrix("self_attn.v_proj", 256, 2048)?;
            let o = matrix("self_attn.o_proj", 2048, 2048)?;
            let gate = matrix("mlp.gate_proj", 6144, 2048)?;
            let up = matrix("mlp.up_proj", 6144, 2048)?;
            let down = matrix("mlp.down_proj", 2048, 6144)?;
            let input_norm = take(
                &mut map,
                stream,
                &format!("{prefix}.input_layernorm.weight"),
                &[2048],
            )?;
            let post_norm = take(
                &mut map,
                stream,
                &format!("{prefix}.post_attention_layernorm.weight"),
                &[2048],
            )?;
            let layer = LayerWeights {
                input_norm,
                post_norm,
                q,
                k,
                v,
                o,
                gate,
                up,
                down,
            };
            // Quantizing a matrix is lazy; one evaluation per layer keeps the
            // decode copies from materializing one projection at a time.
            stream.eval(&layer.arrays())?;
            layers.push(layer);
        }
        if !map.is_empty() {
            let mut unexpected = map.keys().cloned().collect::<Vec<_>>();
            unexpected.sort();
            return Err(Error::Other(format!(
                "unconsumed MiniCPM5 checkpoint tensors: {unexpected:?}"
            )));
        }
        let mut remaining = vec![embedding.clone(), norm.clone()];
        remaining.extend(head.arrays());
        stream.eval(&remaining)?;
        Ok(Self {
            embedding,
            norm,
            head,
            layers,
        })
    }
}
