//! Llama weight structs + HF weight-map loader.

use std::collections::HashMap;

use apxinf_core::{Error, Result, Tensor};
use apxinf_loader::ModelConfig;

pub struct LlamaWeights {
    pub token_embedding: Tensor,
    pub layers: Vec<TransformerLayer>,
    pub output_norm_weight: Tensor,
    pub output_weight: Tensor,
}

impl LlamaWeights {
    /// Construct from a HuggingFace-style weight map.
    pub fn from_map(config: &ModelConfig, mut tensors: HashMap<String, Tensor>) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.n_layers);

        for i in 0..config.n_layers {
            let prefix = format!("model.layers.{i}");
            layers.push(TransformerLayer {
                attn_norm_weight: tensors
                    .remove(&format!("{prefix}.input_layernorm.weight"))
                    .ok_or_else(|| {
                        Error::Other(format!("missing {prefix}.input_layernorm.weight"))
                    })?,
                wq: transpose_weight(
                    &tensors
                        .remove(&format!("{prefix}.self_attn.q_proj.weight"))
                        .ok_or_else(|| {
                            Error::Other(format!("missing {prefix}.self_attn.q_proj.weight"))
                        })?,
                )?,
                wk: transpose_weight(
                    &tensors
                        .remove(&format!("{prefix}.self_attn.k_proj.weight"))
                        .ok_or_else(|| {
                            Error::Other(format!("missing {prefix}.self_attn.k_proj.weight"))
                        })?,
                )?,
                wv: transpose_weight(
                    &tensors
                        .remove(&format!("{prefix}.self_attn.v_proj.weight"))
                        .ok_or_else(|| {
                            Error::Other(format!("missing {prefix}.self_attn.v_proj.weight"))
                        })?,
                )?,
                wo: transpose_weight(
                    &tensors
                        .remove(&format!("{prefix}.self_attn.o_proj.weight"))
                        .ok_or_else(|| {
                            Error::Other(format!("missing {prefix}.self_attn.o_proj.weight"))
                        })?,
                )?,
                ffn_norm_weight: tensors
                    .remove(&format!("{prefix}.post_attention_layernorm.weight"))
                    .ok_or_else(|| {
                        Error::Other(format!("missing {prefix}.post_attention_layernorm.weight"))
                    })?,
                w_gate: transpose_weight(
                    &tensors
                        .remove(&format!("{prefix}.mlp.gate_proj.weight"))
                        .ok_or_else(|| {
                            Error::Other(format!("missing {prefix}.mlp.gate_proj.weight"))
                        })?,
                )?,
                w_up: transpose_weight(
                    &tensors
                        .remove(&format!("{prefix}.mlp.up_proj.weight"))
                        .ok_or_else(|| {
                            Error::Other(format!("missing {prefix}.mlp.up_proj.weight"))
                        })?,
                )?,
                w_down: transpose_weight(
                    &tensors
                        .remove(&format!("{prefix}.mlp.down_proj.weight"))
                        .ok_or_else(|| {
                            Error::Other(format!("missing {prefix}.mlp.down_proj.weight"))
                        })?,
                )?,
                qkv_packed: None,
                gate_up_packed: None,
            });
        }

        let lm_head = tensors
            .remove("lm_head.weight")
            .ok_or_else(|| Error::Other("missing lm_head.weight".into()))?;

        let embed_raw = tensors
            .remove("model.embed_tokens.weight")
            .ok_or_else(|| Error::Other("missing model.embed_tokens.weight".into()))?;
        let embed_is_zeros = embed_raw
            .to_f32_vec()
            .map(|d| d.iter().all(|&v| v == 0.0))
            .unwrap_or(false);
        let token_embedding = if embed_is_zeros && embed_raw.shape() == lm_head.shape() {
            lm_head.clone()
        } else {
            embed_raw
        };

        Ok(LlamaWeights {
            token_embedding,
            layers,
            output_norm_weight: tensors
                .remove("model.norm.weight")
                .ok_or_else(|| Error::Other("missing model.norm.weight".into()))?,
            output_weight: transpose_weight(&lm_head)?,
        })
    }
}

/// Weights for a single transformer layer.
pub struct TransformerLayer {
    pub attn_norm_weight: Tensor,
    pub wq: Tensor,
    pub wk: Tensor,
    pub wv: Tensor,
    pub wo: Tensor,
    pub ffn_norm_weight: Tensor,
    pub w_gate: Tensor,
    pub w_up: Tensor,
    pub w_down: Tensor,
    /// Fused `[hidden, hidden + 2*kv_proj]` weight = concat(wq, wk, wv)
    /// along the output axis. `None` until `pack_fused_weights` is called.
    /// Used by the fused-QKV GEMM path (Phase 2 of kernel-fusion plan).
    pub qkv_packed: Option<Tensor>,
    /// Fused `[hidden, 2*intermediate]` weight = concat(w_gate, w_up)
    /// along the output axis. Used by the fused Gate/Up GEMM + SwiGLU
    /// path (Phase 3 of kernel-fusion plan).
    pub gate_up_packed: Option<Tensor>,
}

/// Transpose a 2D weight tensor, or return as-is if 1D.
/// PyTorch stores linear layer weights as [out_features, in_features],
/// but we need [in_features, out_features] for matmul.
fn transpose_weight(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() == 1 {
        // 1D tensor (norm weights, biases) - no transpose needed
        return Ok(tensor.clone());
    }
    if dims.len() != 2 {
        return Err(Error::Other(format!(
            "transpose_weight expected 1D or 2D tensor, got {}D",
            dims.len()
        )));
    }
    let [rows, cols] = [dims[0], dims[1]];
    match tensor.dtype() {
        apxinf_core::DType::F32 => {
            let data = tensor.as_f32()?;
            let mut out = vec![0.0f32; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    out[j * rows + i] = data[i * cols + j];
                }
            }
            Tensor::from_f32(vec![cols, rows], &out)
        }
        apxinf_core::DType::BF16 => {
            let data = tensor.as_bf16()?;
            let mut out = vec![half::bf16::from_f32(0.0); rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    out[j * rows + i] = data[i * cols + j];
                }
            }
            Tensor::from_bf16(vec![cols, rows], &out)
        }
        dtype => Err(Error::Other(format!(
            "Llama weight transpose does not support {dtype}"
        ))),
    }
}
