use super::config::Config;
use apxinf_core::{DType, Error, Result, Tensor};
use apxinf_mlx::{Array, MlxDType, Stream};
use std::collections::HashMap;

#[derive(Clone)]
pub struct LayerWeights {
    pub input_norm: Array,
    pub post_norm: Array,
    /// Canonical [in, out] matrices. Private MLX views retain the checkpoint's
    /// physical [out, in] backing, preserving the reference GEMV selection.
    pub q: Array,
    pub k: Array,
    pub v: Array,
    pub o: Array,
    pub gate: Array,
    pub up: Array,
    pub down: Array,
}

pub struct Weights {
    pub embedding: Array,
    pub norm: Array,
    pub head: Array,
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
        stream: &Stream,
        mut map: HashMap<String, Tensor>,
    ) -> Result<Self> {
        let embedding = take(
            &mut map,
            stream,
            "model.embed_tokens.weight",
            &[config.vocab_size, config.hidden_size],
        )?;
        let norm = take(&mut map, stream, "model.norm.weight", &[config.hidden_size])?;
        let head = take(
            &mut map,
            stream,
            "lm_head.weight",
            &[config.vocab_size, config.hidden_size],
        )?
        .transpose(&[1, 0])?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            let mut matrix = |suffix: &str, out: usize, input: usize| -> Result<Array> {
                take(
                    &mut map,
                    stream,
                    &format!("{prefix}.{suffix}.weight"),
                    &[out, input],
                )?
                .transpose(&[1, 0])
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
            stream.eval(&[
                q.clone(),
                k.clone(),
                v.clone(),
                o.clone(),
                gate.clone(),
                up.clone(),
                down.clone(),
                input_norm.clone(),
                post_norm.clone(),
            ])?;
            layers.push(LayerWeights {
                input_norm,
                post_norm,
                q,
                k,
                v,
                o,
                gate,
                up,
                down,
            });
        }
        if !map.is_empty() {
            let mut unexpected = map.keys().cloned().collect::<Vec<_>>();
            unexpected.sort();
            return Err(Error::Other(format!(
                "unconsumed MiniCPM5 checkpoint tensors: {unexpected:?}"
            )));
        }
        stream.eval(&[embedding.clone(), norm.clone(), head.clone()])?;
        Ok(Self {
            embedding,
            norm,
            head,
            layers,
        })
    }
}
