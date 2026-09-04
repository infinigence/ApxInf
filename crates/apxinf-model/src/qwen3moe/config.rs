//! Qwen3-MoE (`model_type = "qwen3_moe"`) configuration parsed from the
//! Hugging Face `config.json`, including the AutoAWQ quantization block.
//!
//! The model owns its schema (`adding-a-new-model.md`): `ModelConfig` from the
//! loader crate is Llama-shaped and has no notion of experts or quantization.

use std::path::Path;

use apxinf_core::{Error, Result};

/// Weight quantization scheme accepted by this model family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen3MoeQuantization {
    /// AutoAWQ `version = "gemm"`, 4-bit, asymmetric zero points, group-wise.
    AwqInt4 { group_size: usize },
}

#[derive(Clone, Debug)]
pub struct Qwen3MoeConfig {
    pub hidden_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub tie_word_embeddings: bool,
    pub eos_token_id: u32,
    pub quantization: Qwen3MoeQuantization,
}

impl Qwen3MoeConfig {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| Error::Other(format!("read {}: {e}", path.display())))?;
        Self::from_json_str(&raw)
    }

    pub fn from_json_str(text: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| Error::Other(format!("qwen3_moe config json: {e}")))?;
        let model_type = v["model_type"].as_str().unwrap_or("");
        if model_type != "qwen3_moe" {
            return Err(Error::Other(format!(
                "qwen3moe: expected model_type \"qwen3_moe\", found {model_type:?}"
            )));
        }
        let usize_field = |name: &str| -> Result<usize> {
            v[name]
                .as_u64()
                .map(|value| value as usize)
                .ok_or_else(|| Error::Other(format!("qwen3_moe config: missing `{name}`")))
        };
        let f32_field = |name: &str, default: f64| v[name].as_f64().unwrap_or(default) as f32;

        let hidden_size = usize_field("hidden_size")?;
        let n_heads = usize_field("num_attention_heads")?;
        let head_dim = v["head_dim"]
            .as_u64()
            .map(|value| value as usize)
            .unwrap_or(hidden_size / n_heads);
        if !v["mlp_only_layers"].as_array().map_or(true, |a| a.is_empty()) {
            return Err(Error::Other(
                "qwen3moe: `mlp_only_layers` is not supported (all layers must be MoE)".into(),
            ));
        }
        if v["decoder_sparse_step"].as_u64().unwrap_or(1) != 1 {
            return Err(Error::Other(
                "qwen3moe: `decoder_sparse_step` != 1 is not supported".into(),
            ));
        }
        if v["use_sliding_window"].as_bool().unwrap_or(false) {
            return Err(Error::Other("qwen3moe: sliding window is not supported".into()));
        }
        if !v["rope_scaling"].is_null() {
            return Err(Error::Other("qwen3moe: rope_scaling is not supported".into()));
        }
        if v["attention_bias"].as_bool().unwrap_or(false) {
            return Err(Error::Other("qwen3moe: attention_bias is not supported".into()));
        }

        let q = &v["quantization_config"];
        let quantization = if q.is_object() {
            let method = q["quant_method"].as_str().unwrap_or("");
            let bits = q["bits"].as_u64().unwrap_or(0);
            let version = q["version"].as_str().unwrap_or("gemm");
            let zero_point = q["zero_point"].as_bool().unwrap_or(true);
            let group_size = q["group_size"].as_u64().unwrap_or(0) as usize;
            if method != "awq" || bits != 4 || version != "gemm" || !zero_point {
                return Err(Error::Other(format!(
                    "qwen3moe: only AutoAWQ 4-bit gemm checkpoints are supported, \
                     found method={method:?} bits={bits} version={version:?} zero_point={zero_point}"
                )));
            }
            if group_size == 0 || hidden_size % group_size != 0 {
                return Err(Error::Other(format!(
                    "qwen3moe: unsupported AWQ group_size {group_size}"
                )));
            }
            let skipped = q["modules_to_not_convert"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|m| m.as_str())
                        .map(str::to_string)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            for module in &skipped {
                if module != "mlp.gate" && module != "lm_head" {
                    return Err(Error::Other(format!(
                        "qwen3moe: unexpected unquantized module {module:?}"
                    )));
                }
            }
            Qwen3MoeQuantization::AwqInt4 { group_size }
        } else {
            return Err(Error::Other(
                "qwen3moe: only quantized (AutoAWQ INT4) checkpoints are supported; \
                 the BF16 checkpoint does not fit the device memory budget"
                    .into(),
            ));
        };

        let eos_token_id = match &v["eos_token_id"] {
            serde_json::Value::Number(n) => n.as_u64().unwrap_or(151645) as u32,
            serde_json::Value::Array(a) => a
                .first()
                .and_then(|x| x.as_u64())
                .unwrap_or(151645) as u32,
            _ => 151645,
        };

        Ok(Self {
            hidden_size,
            n_layers: usize_field("num_hidden_layers")?,
            n_heads,
            n_kv_heads: usize_field("num_key_value_heads")?,
            head_dim,
            vocab_size: usize_field("vocab_size")?,
            max_position_embeddings: usize_field("max_position_embeddings")?,
            rms_norm_eps: f32_field("rms_norm_eps", 1e-6),
            rope_theta: f32_field("rope_theta", 10_000_000.0),
            num_experts: usize_field("num_experts")?,
            num_experts_per_tok: usize_field("num_experts_per_tok")?,
            moe_intermediate_size: usize_field("moe_intermediate_size")?,
            norm_topk_prob: v["norm_topk_prob"].as_bool().unwrap_or(true),
            tie_word_embeddings: v["tie_word_embeddings"].as_bool().unwrap_or(false),
            eos_token_id,
            quantization,
        })
    }

    pub fn group_size(&self) -> usize {
        match self.quantization {
            Qwen3MoeQuantization::AwqInt4 { group_size } => group_size,
        }
    }

    /// Width of the packed `[q | k | v]` projection output.
    pub fn qkv_dim(&self) -> usize {
        (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"{
        "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
        "hidden_size": 2048, "num_hidden_layers": 48, "num_attention_heads": 32,
        "num_key_value_heads": 4, "head_dim": 128, "vocab_size": 151936,
        "max_position_embeddings": 262144, "rms_norm_eps": 1e-06, "rope_theta": 10000000,
        "num_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 768,
        "norm_topk_prob": true, "tie_word_embeddings": false, "eos_token_id": 151645,
        "mlp_only_layers": [], "decoder_sparse_step": 1, "rope_scaling": null,
        "quantization_config": {"bits": 4, "group_size": 128, "quant_method": "awq",
            "version": "gemm", "zero_point": true, "modules_to_not_convert": ["mlp.gate", "lm_head"]}
    }"#;

    #[test]
    fn parses_qwen3_30b_a3b_awq() {
        let cfg = Qwen3MoeConfig::from_json_str(CONFIG).unwrap();
        assert_eq!(cfg.n_layers, 48);
        assert_eq!(cfg.qkv_dim(), 4096 + 2 * 512);
        assert_eq!(cfg.group_size(), 128);
        assert_eq!(cfg.num_experts, 128);
        assert!(cfg.norm_topk_prob);
    }

    #[test]
    fn rejects_unquantized_and_wrong_type() {
        let no_quant = CONFIG.replace(
            r#""quantization_config": {"bits": 4, "group_size": 128, "quant_method": "awq",
            "version": "gemm", "zero_point": true, "modules_to_not_convert": ["mlp.gate", "lm_head"]}"#,
            r#""torch_dtype": "bfloat16""#,
        );
        assert!(Qwen3MoeConfig::from_json_str(&no_quant).is_err());
        assert!(Qwen3MoeConfig::from_json_str(&CONFIG.replace("qwen3_moe", "qwen3")).is_err());
    }
}
