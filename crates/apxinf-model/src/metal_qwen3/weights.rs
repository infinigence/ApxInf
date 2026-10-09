//! Checkpoint loading and canonical weight geometry, independent of session state.

use super::config::{Qwen3Config, Variant};
use apxinf_core::{DType, Device, Error, Result, Tensor};
use apxinf_mlx::{Array, MlxDType, Stream};
use std::collections::HashMap;

/// Private MLX affine packing, derived once from the canonical weight.
struct Quantized {
    packed: Array,
    scales: Array,
    biases: Array,
}

impl Quantized {
    fn new(hf_weight: &Array) -> Result<Self> {
        let [packed, scales, biases] = hf_weight.quantize(64, 8)?;
        hf_weight
            .stream()
            .eval(&[packed.clone(), scales.clone(), biases.clone()])?;
        Ok(Self {
            packed,
            scales,
            biases,
        })
    }

    fn project(&self, x: &Array) -> Result<Array> {
        x.quantized_matmul(&self.packed, &self.scales, &self.biases, true, 64, 8)
    }

    fn gather(&self, ids: &Array) -> Result<Array> {
        self.packed.take(ids, 0)?.dequantize(
            &self.scales.take(ids, 0)?,
            &self.biases.take(ids, 0)?,
            64,
            8,
        )
    }

    fn arrays(&self) -> Vec<Array> {
        vec![
            self.packed.clone(),
            self.scales.clone(),
            self.biases.clone(),
        ]
    }
}

pub(crate) struct Linear {
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

    pub fn call(&self, x: &Array) -> Result<Array> {
        // Accepted scoped lane: only a single row uses W8; prompt GEMMs keep BF16.
        if x.shape()[..x.shape().len() - 1].iter().product::<usize>() == 1 {
            if let Some(packed) = &self.packed {
                return packed.project(x);
            }
        }
        x.matmul(&self.canonical)
    }

    fn arrays(&self) -> Vec<Array> {
        let mut arrays = vec![self.canonical.clone()];
        if let Some(packed) = &self.packed {
            arrays.extend(packed.arrays());
        }
        arrays
    }
}

enum GateUp {
    Separate { gate: Array, up: Array },
    Fused(Linear),
}

pub(crate) struct LayerWeights {
    pub input_norm: Array,
    pub q_norm: Array,
    pub k_norm: Array,
    pub post_norm: Array,
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
    gate_up: GateUp,
    pub down: Linear,
}

enum Table {
    Bf16 { embedding: Array, output: Array },
    Quantized(Quantized),
}

pub(crate) struct Weights {
    table: Table,
    pub final_norm: Array,
    pub layers: Vec<LayerWeights>,
}

fn take(
    map: &mut HashMap<String, Tensor>,
    name: &str,
    shape: &[usize],
    stream: &Stream,
) -> Result<Array> {
    let tensor = map
        .remove(name)
        .ok_or_else(|| Error::Other(format!("Qwen3 missing weight `{name}`")))?;
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
            allowed: "Qwen3 checkpoint BF16",
        });
    }
    let bytes = tensor
        .storage()
        .as_cpu()
        .ok_or_else(|| Error::Other("weight has no CPU storage".into()))?;
    Array::from_bytes(stream, shape, MlxDType::BF16, bytes)
}

fn linear(
    map: &mut HashMap<String, Tensor>,
    name: &str,
    input: usize,
    output: usize,
    stream: &Stream,
) -> Result<Array> {
    // Canonical [in,out] view of immutable HF [out,in]. Keep the private view:
    // forcing physical contiguity can select a different MLX M=1 reduction.
    take(map, name, &[output, input], stream)?.transpose(&[1, 0])
}

impl Weights {
    pub fn load(
        config: &Qwen3Config,
        mut map: HashMap<String, Tensor>,
        stream: &Stream,
        variant: Variant,
    ) -> Result<Self> {
        let h = config.hidden_size;
        let q = config.num_attention_heads * config.head_dim;
        let kv = config.num_key_value_heads * config.head_dim;
        let mixed = variant == Variant::MixedW8;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let mut norm = |suffix: &str, size: usize| -> Result<Array> {
                take(&mut map, &format!("{p}.{suffix}.weight"), &[size], stream)?
                    .cast(MlxDType::F32)
            };
            let input_norm = norm("input_layernorm", h)?;
            let post_norm = norm("post_attention_layernorm", h)?;
            let q_norm = norm("self_attn.q_norm", config.head_dim)?;
            let k_norm = norm("self_attn.k_norm", config.head_dim)?;
            let mut proj = |suffix: &str, input, output| {
                linear(
                    &mut map,
                    &format!("{p}.{suffix}.weight"),
                    input,
                    output,
                    stream,
                )
            };
            let gate = proj("mlp.gate_proj", h, config.intermediate_size)?;
            let up = proj("mlp.up_proj", h, config.intermediate_size)?;
            let gate_up = if mixed {
                let fused =
                    Array::concat(&[&gate.transpose(&[1, 0])?, &up.transpose(&[1, 0])?], 0)?
                        .transpose(&[1, 0])?;
                // Quantized and BF16 fused weights own their storage; the two
                // original halves can be released after materialization.
                let linear = Linear::new(fused, true)?;
                stream.eval(&linear.arrays())?;
                GateUp::Fused(linear)
            } else {
                GateUp::Separate { gate, up }
            };
            let layer = LayerWeights {
                input_norm,
                q_norm,
                k_norm,
                post_norm,
                q: Linear::new(proj("self_attn.q_proj", h, q)?, mixed)?,
                k: Linear::new(proj("self_attn.k_proj", h, kv)?, mixed)?,
                v: Linear::new(proj("self_attn.v_proj", h, kv)?, mixed)?,
                o: Linear::new(proj("self_attn.o_proj", q, h)?, mixed)?,
                gate_up,
                down: Linear::new(proj("mlp.down_proj", config.intermediate_size, h)?, mixed)?,
            };
            stream.eval(&layer.arrays())?;
            layers.push(layer);
        }
        // Validate duplicated tied storage before consuming either host tensor.
        if config.tie_word_embeddings {
            if let Some(head) = map.remove("lm_head.weight") {
                let embed = map
                    .get("model.embed_tokens.weight")
                    .ok_or_else(|| Error::Other("missing tied embedding".into()))?;
                if head.shape() != embed.shape()
                    || head.dtype() != embed.dtype()
                    || head.storage().as_cpu() != embed.storage().as_cpu()
                {
                    return Err(Error::Other(
                        "Qwen3 tied lm_head differs from embedding checkpoint bytes".into(),
                    ));
                }
            }
        }
        let embedding = take(
            &mut map,
            "model.embed_tokens.weight",
            &[config.vocab_size, h],
            stream,
        )?;
        let table = if mixed {
            if !config.tie_word_embeddings {
                return Err(Error::Other(
                    "mixed-w8 requires the tied embedding/head table".into(),
                ));
            }
            // No BF16 table remains resident: both consumers use this same pack.
            Table::Quantized(Quantized::new(&embedding)?)
        } else {
            let output = if config.tie_word_embeddings {
                embedding.transpose(&[1, 0])?
            } else {
                linear(&mut map, "lm_head.weight", h, config.vocab_size, stream)?
            };
            stream.eval(&[embedding.clone(), output.clone()])?;
            Table::Bf16 { embedding, output }
        };
        let final_norm = take(&mut map, "model.norm.weight", &[h], stream)?.cast(MlxDType::F32)?;
        // An unexpected tensor can indicate a biased, hybrid or quantized model.
        if !map.is_empty() {
            let mut names: Vec<_> = map.keys().cloned().collect();
            names.sort();
            names.truncate(5);
            return Err(Error::Other(format!(
                "unconsumed Qwen3 checkpoint tensors: {}",
                names.join(", ")
            )));
        }
        stream.eval(&[final_norm.clone()])?;
        Ok(Self {
            table,
            final_norm,
            layers,
        })
    }

    pub fn embed(&self, ids: &Array) -> Result<Array> {
        match &self.table {
            Table::Bf16 { embedding, .. } => embedding.take(ids, 0),
            Table::Quantized(table) => table.gather(ids),
        }
    }

    pub fn project(&self, x: &Array) -> Result<Array> {
        match &self.table {
            Table::Bf16 { output, .. } => x.matmul(output),
            Table::Quantized(table) => table.project(x),
        }
    }
}

impl LayerWeights {
    pub fn arrays(&self) -> Vec<Array> {
        let mut arrays = vec![
            self.input_norm.clone(),
            self.q_norm.clone(),
            self.k_norm.clone(),
            self.post_norm.clone(),
        ];
        for projection in [&self.q, &self.k, &self.v, &self.o, &self.down] {
            arrays.extend(projection.arrays());
        }
        match &self.gate_up {
            GateUp::Separate { gate, up } => arrays.extend([gate.clone(), up.clone()]),
            GateUp::Fused(projection) => arrays.extend(projection.arrays()),
        }
        arrays
    }

    pub fn block_arrays(&self) -> Result<Vec<Array>> {
        let GateUp::Separate { gate, up } = &self.gate_up else {
            return Err(Error::Other(
                "W8 has no inherited compiled decoder block".into(),
            ));
        };
        Ok(vec![
            self.input_norm.clone(),
            self.q_norm.clone(),
            self.k_norm.clone(),
            self.post_norm.clone(),
            self.q.canonical.clone(),
            self.k.canonical.clone(),
            self.v.canonical.clone(),
            self.o.canonical.clone(),
            gate.clone(),
            up.clone(),
            self.down.canonical.clone(),
        ])
    }

    pub fn gate_up(&self, x: &Array) -> Result<(Array, Array)> {
        match &self.gate_up {
            GateUp::Separate { gate, up } => Ok((x.matmul(gate)?, x.matmul(up)?)),
            GateUp::Fused(projection) => {
                let projected = projection.call(x)?;
                let axis = projected.shape().len() - 1;
                let width = projected.shape()[axis] / 2;
                Ok((
                    projected.slice_axis(axis, 0, width)?,
                    projected.slice_axis(axis, width, width * 2)?,
                ))
            }
        }
    }
}
