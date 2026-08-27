//! Qwen2.5-Omni Thinker text weights.

use std::collections::HashMap;

use apxinf_core::{Backend, DType, Error, Result, Tensor};

use super::config::Qwen25OmniConfig;

pub struct Qwen25OmniTextWeights {
    pub token_embedding: Tensor,
    pub layers: Vec<Qwen25OmniTextLayer>,
    pub output_norm: Tensor,
    /// Separate checkpoint LM head, transposed to `[hidden, vocab]`.
    pub lm_head: Tensor,
}

pub enum Qwen25OmniQkvWeights {
    Separate {
        wq: Tensor,
        bq: Tensor,
        wk: Tensor,
        bk: Tensor,
        wv: Tensor,
        bv: Tensor,
    },
    Packed {
        weight: Tensor,
        bias: Tensor,
    },
}

pub struct Qwen25OmniTextLayer {
    pub attn_norm: Tensor,
    pub qkv: Qwen25OmniQkvWeights,
    pub wo: Tensor,
    pub ffn_norm: Tensor,
    pub w_gate: Tensor,
    pub w_up: Tensor,
    pub gate_up_packed: Option<Tensor>,
    pub w_down: Tensor,
}

impl Qwen25OmniTextWeights {
    pub fn from_map(
        config: &Qwen25OmniConfig,
        tensors: &mut HashMap<String, Tensor>,
    ) -> Result<Self> {
        let take = |name: &str, map: &mut HashMap<String, Tensor>| {
            map.remove(name)
                .ok_or_else(|| Error::Other(format!("missing {name}")))
        };
        let mut layers = Vec::with_capacity(config.text.n_layers);
        for index in 0..config.text.n_layers {
            let prefix = format!("thinker.model.layers.{index}");
            layers.push(Qwen25OmniTextLayer {
                attn_norm: take(&format!("{prefix}.input_layernorm.weight"), tensors)?,
                qkv: Qwen25OmniQkvWeights::Separate {
                    wq: transpose_2d(&take(
                        &format!("{prefix}.self_attn.q_proj.weight"),
                        tensors,
                    )?)?,
                    bq: take(&format!("{prefix}.self_attn.q_proj.bias"), tensors)?,
                    wk: transpose_2d(&take(
                        &format!("{prefix}.self_attn.k_proj.weight"),
                        tensors,
                    )?)?,
                    bk: take(&format!("{prefix}.self_attn.k_proj.bias"), tensors)?,
                    wv: transpose_2d(&take(
                        &format!("{prefix}.self_attn.v_proj.weight"),
                        tensors,
                    )?)?,
                    bv: take(&format!("{prefix}.self_attn.v_proj.bias"), tensors)?,
                },
                wo: transpose_2d(&take(
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    tensors,
                )?)?,
                ffn_norm: take(
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    tensors,
                )?,
                w_gate: transpose_2d(&take(&format!("{prefix}.mlp.gate_proj.weight"), tensors)?)?,
                w_up: transpose_2d(&take(&format!("{prefix}.mlp.up_proj.weight"), tensors)?)?,
                gate_up_packed: None,
                w_down: transpose_2d(&take(&format!("{prefix}.mlp.down_proj.weight"), tensors)?)?,
            });
        }
        Ok(Self {
            token_embedding: take("thinker.model.embed_tokens.weight", tensors)?,
            layers,
            output_norm: take("thinker.model.norm.weight", tensors)?,
            lm_head: transpose_2d(&take("thinker.lm_head.weight", tensors)?)?,
        })
    }

    pub fn to_device(self, backend: &dyn Backend) -> Result<Self> {
        let layers = self
            .layers
            .into_iter()
            .map(|layer| {
                Ok(Qwen25OmniTextLayer {
                    attn_norm: backend.to_device(&layer.attn_norm)?,
                    qkv: match layer.qkv {
                        Qwen25OmniQkvWeights::Separate {
                            wq,
                            bq,
                            wk,
                            bk,
                            wv,
                            bv,
                        } => Qwen25OmniQkvWeights::Separate {
                            wq: backend.to_device(&wq)?,
                            bq: backend.to_device(&bq)?,
                            wk: backend.to_device(&wk)?,
                            bk: backend.to_device(&bk)?,
                            wv: backend.to_device(&wv)?,
                            bv: backend.to_device(&bv)?,
                        },
                        Qwen25OmniQkvWeights::Packed { weight, bias } => {
                            Qwen25OmniQkvWeights::Packed {
                                weight: backend.to_device(&weight)?,
                                bias: backend.to_device(&bias)?,
                            }
                        }
                    },
                    wo: backend.to_device(&layer.wo)?,
                    ffn_norm: backend.to_device(&layer.ffn_norm)?,
                    w_gate: backend.to_device(&layer.w_gate)?,
                    w_up: backend.to_device(&layer.w_up)?,
                    gate_up_packed: layer
                        .gate_up_packed
                        .as_ref()
                        .map(|weight| backend.to_device(weight))
                        .transpose()?,
                    w_down: backend.to_device(&layer.w_down)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            token_embedding: backend.to_device(&self.token_embedding)?,
            layers,
            output_norm: backend.to_device(&self.output_norm)?,
            lm_head: backend.to_device(&self.lm_head)?,
        })
    }

    pub fn into_packed_qkv(self, backend: &dyn Backend) -> Result<Self> {
        let layers = self
            .layers
            .into_iter()
            .map(|layer| {
                let qkv = match layer.qkv {
                    Qwen25OmniQkvWeights::Separate {
                        wq,
                        bq,
                        wk,
                        bk,
                        wv,
                        bv,
                    } => {
                        let weight = backend.concat_2d(&[&wq, &wk, &wv])?;
                        let bq = bq.reshape(vec![1, bq.numel()])?;
                        let bk = bk.reshape(vec![1, bk.numel()])?;
                        let bv = bv.reshape(vec![1, bv.numel()])?;
                        let bias = backend.concat_2d(&[&bq, &bk, &bv])?;
                        backend.synchronize()?;
                        Qwen25OmniQkvWeights::Packed {
                            weight,
                            bias: bias.reshape(vec![bias.numel()])?,
                        }
                    }
                    Qwen25OmniQkvWeights::Packed { .. } => {
                        return Err(Error::Other(
                            "Qwen2.5-Omni QKV weights were packed twice".into(),
                        ))
                    }
                };
                Ok(Qwen25OmniTextLayer {
                    attn_norm: layer.attn_norm,
                    qkv,
                    wo: layer.wo,
                    ffn_norm: layer.ffn_norm,
                    w_gate: layer.w_gate,
                    w_up: layer.w_up,
                    gate_up_packed: layer.gate_up_packed,
                    w_down: layer.w_down,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            token_embedding: self.token_embedding,
            layers,
            output_norm: self.output_norm,
            lm_head: self.lm_head,
        })
    }

    pub fn into_packed_gate_up(self, backend: &dyn Backend) -> Result<Self> {
        let layers = self
            .layers
            .into_iter()
            .map(|layer| {
                if layer.gate_up_packed.is_some() {
                    return Err(Error::Other(
                        "Qwen2.5-Omni Gate/Up weights were packed twice".into(),
                    ));
                }
                let gate_up_packed = Some(backend.concat_2d(&[&layer.w_gate, &layer.w_up])?);
                Ok(Qwen25OmniTextLayer {
                    attn_norm: layer.attn_norm,
                    qkv: layer.qkv,
                    wo: layer.wo,
                    ffn_norm: layer.ffn_norm,
                    w_gate: layer.w_gate,
                    w_up: layer.w_up,
                    gate_up_packed,
                    w_down: layer.w_down,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        backend.synchronize()?;
        Ok(Self {
            token_embedding: self.token_embedding,
            layers,
            output_norm: self.output_norm,
            lm_head: self.lm_head,
        })
    }
}

pub(crate) fn reshape_first_axis(tensor: &Tensor, rows: usize, cols: usize) -> Result<Tensor> {
    let elements = tensor
        .shape()
        .dims()
        .iter()
        .try_fold(1usize, |total, value| {
            total
                .checked_mul(*value)
                .ok_or_else(|| Error::Other("tensor shape product overflow".into()))
        })?;
    if elements != rows * cols {
        return Err(Error::Other(format!(
            "reshape expected {} elements for [{rows}, {cols}], got {elements}",
            rows * cols
        )));
    }
    tensor.reshape(vec![rows, cols])
}

pub(crate) fn flatten_and_transpose(tensor: &Tensor, rows: usize, cols: usize) -> Result<Tensor> {
    transpose_2d(&reshape_first_axis(tensor, rows, cols)?)
}

pub(crate) fn transpose_2d(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!(
            "transpose_2d expected rank 2, got rank {}",
            dims.len()
        )));
    }
    let (rows, cols) = (dims[0], dims[1]);
    match tensor.dtype() {
        DType::F32 => {
            let input = tensor.as_f32()?;
            let mut output = vec![0.0_f32; input.len()];
            for row in 0..rows {
                for col in 0..cols {
                    output[col * rows + row] = input[row * cols + col];
                }
            }
            Tensor::from_f32(vec![cols, rows], &output)
        }
        DType::BF16 => {
            let input = tensor.as_bf16()?;
            let mut output = vec![half::bf16::from_f32(0.0); input.len()];
            for row in 0..rows {
                for col in 0..cols {
                    output[col * rows + row] = input[row * cols + col];
                }
            }
            Tensor::from_bf16(vec![cols, rows], &output)
        }
        dtype => Err(Error::Other(format!(
            "qwen2.5-omni transpose does not support {dtype}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transposes_bf16_linear_weight_without_upcast() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map(half::bf16::from_f32);
        let input = Tensor::from_bf16(vec![2, 3], &values).unwrap();
        let output = transpose_2d(&input).unwrap();
        assert_eq!(output.dtype(), DType::BF16);
        assert_eq!(output.shape().dims(), [3, 2]);
        assert_eq!(
            output.to_f32_vec().unwrap(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }
}
