//! Qwen3-VL text-stack weights loader.
//!
//! Consumes a HF-style weight map (from `apxinf_loader::safetensors::load_native`)
//! and produces the per-layer tensor slots the decode graph expects. The
//! vision + merger weights are ignored for now — Phase 4 will add a
//! `Qwen3VLVisionWeights` type alongside this one.

use std::collections::HashMap;

use apxinf_core::{Error, Result, Tensor};

use super::config::Qwen3VLConfig;

/// All text-stack weights for Qwen3-VL, in the layout the decode graph
/// wants (2D projections transposed to `[in, out]` for cuBLAS row-major).
pub struct Qwen3VLTextWeights {
    /// `[vocab_size, hidden_size]`. Doubles as the lm_head weight because
    /// Qwen3-VL uses tied embeddings.
    pub token_embedding: Tensor,
    pub layers: Vec<Qwen3VLLayer>,
    pub output_norm_weight: Tensor,
}

pub struct Qwen3VLLayer {
    pub attn_norm_weight: Tensor,
    /// `[hidden, n_heads * head_dim]` — transposed q_proj.
    pub wq: Tensor,
    /// `[hidden, n_kv_heads * head_dim]` — transposed k_proj.
    pub wk: Tensor,
    /// `[hidden, n_kv_heads * head_dim]` — transposed v_proj.
    pub wv: Tensor,
    /// `[n_heads * head_dim, hidden]` — transposed o_proj.
    pub wo: Tensor,
    /// `[head_dim]` — per-head RMSNorm applied to Q after projection.
    pub q_norm_weight: Tensor,
    /// `[head_dim]` — per-head RMSNorm applied to K after projection.
    pub k_norm_weight: Tensor,
    pub ffn_norm_weight: Tensor,
    /// `[hidden, intermediate]` — transposed gate_proj.
    pub w_gate: Tensor,
    /// `[hidden, intermediate]` — transposed up_proj.
    pub w_up: Tensor,
    /// `[intermediate, hidden]` — transposed down_proj.
    pub w_down: Tensor,
    /// Fused `[hidden, hidden + 2*kv_proj]` weight = concat(wq, wk, wv).
    /// `None` until `pack_fused_weights` runs. Enables the fused QKV GEMM
    /// in the decode fast path.
    pub qkv_packed: Option<Tensor>,
    /// Fused `[hidden, 2*intermediate]` weight = concat(w_gate, w_up).
    pub gate_up_packed: Option<Tensor>,
}

impl Qwen3VLTextWeights {
    pub fn from_map(cfg: &Qwen3VLConfig, mut tensors: HashMap<String, Tensor>) -> Result<Self> {
        let n_layers = cfg.text.n_layers;
        let mut layers = Vec::with_capacity(n_layers);

        for i in 0..n_layers {
            let p = format!("model.language_model.layers.{i}");
            let take = |name: &str, map: &mut HashMap<String, Tensor>| -> Result<Tensor> {
                map.remove(name)
                    .ok_or_else(|| Error::Other(format!("missing {name}")))
            };
            layers.push(Qwen3VLLayer {
                attn_norm_weight: take(&format!("{p}.input_layernorm.weight"), &mut tensors)?,
                wq: transpose_2d(&take(
                    &format!("{p}.self_attn.q_proj.weight"),
                    &mut tensors,
                )?)?,
                wk: transpose_2d(&take(
                    &format!("{p}.self_attn.k_proj.weight"),
                    &mut tensors,
                )?)?,
                wv: transpose_2d(&take(
                    &format!("{p}.self_attn.v_proj.weight"),
                    &mut tensors,
                )?)?,
                wo: transpose_2d(&take(
                    &format!("{p}.self_attn.o_proj.weight"),
                    &mut tensors,
                )?)?,
                q_norm_weight: take(&format!("{p}.self_attn.q_norm.weight"), &mut tensors)?,
                k_norm_weight: take(&format!("{p}.self_attn.k_norm.weight"), &mut tensors)?,
                ffn_norm_weight: take(
                    &format!("{p}.post_attention_layernorm.weight"),
                    &mut tensors,
                )?,
                w_gate: transpose_2d(&take(&format!("{p}.mlp.gate_proj.weight"), &mut tensors)?)?,
                w_up: transpose_2d(&take(&format!("{p}.mlp.up_proj.weight"), &mut tensors)?)?,
                w_down: transpose_2d(&take(&format!("{p}.mlp.down_proj.weight"), &mut tensors)?)?,
                qkv_packed: None,
                gate_up_packed: None,
            });
        }

        let token_embedding = tensors
            .remove("model.language_model.embed_tokens.weight")
            .ok_or_else(|| {
                Error::Other("missing model.language_model.embed_tokens.weight".into())
            })?;
        let output_norm_weight = tensors
            .remove("model.language_model.norm.weight")
            .ok_or_else(|| Error::Other("missing model.language_model.norm.weight".into()))?;

        Ok(Self {
            token_embedding,
            layers,
            output_norm_weight,
        })
    }
}

/// Transpose a 2D HF projection weight `[out, in]` → `[in, out]` so the
/// decode graph can use a row-major GEMM. Mirrors `llama::transpose_weight`
/// which we don't reuse because it lives in a different module and the
/// Qwen3-VL loader path stays self-contained.
fn transpose_2d(tensor: &Tensor) -> Result<Tensor> {
    let dims = tensor.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!(
            "transpose_2d expected 2D tensor, got {}D",
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
            "Qwen3-VL weight transpose does not support {dtype}"
        ))),
    }
}
