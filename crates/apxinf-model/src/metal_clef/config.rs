//! The Clef Flash geometry and checkpoint format admitted by this port.
use apxinf_core::{Error, Result};
use serde::Deserialize;

/// Longest admitted request. Longer requests fail; they are never truncated.
pub const MAX_INPUT_TOKENS: usize = 4096;

/// Backbone geometry of `mlx-community/clef-flash-4bit` (Qwen3.5 hybrid).
#[derive(Clone, Debug, Deserialize)]
pub struct TextConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub full_attention_interval: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub attn_output_gate: bool,
    pub attention_bias: bool,
    pub tie_word_embeddings: bool,
    pub hidden_act: String,
    pub rope_parameters: RopeParameters,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub rope_type: String,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
pub struct Quantization {
    pub group_size: usize,
    pub bits: usize,
    pub mode: Mode,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    Affine,
}

#[derive(Clone, Debug, Deserialize)]
struct CheckpointConfig {
    model_type: String,
    text_config: TextConfig,
    quantization: Quantization,
}

/// Joint schema head geometry (`joint_head_config.json`).
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct HeadConfig {
    pub hidden_size: usize,
    pub width: usize,
    pub routing_layers: usize,
    pub layers: usize,
    pub heads: usize,
    pub feedforward: usize,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub text: TextConfig,
    pub quantization: Quantization,
    pub head: HeadConfig,
}

impl Config {
    pub fn from_json(checkpoint: &str, head: &str) -> Result<Self> {
        let parsed: CheckpointConfig = serde_json::from_str(checkpoint)
            .map_err(|e| Error::Other(format!("Clef config: {e}")))?;
        let head: HeadConfig = serde_json::from_str(head)
            .map_err(|e| Error::Other(format!("Clef joint head config: {e}")))?;
        if parsed.model_type != "qwen3_5" {
            return Err(Error::Other(format!(
                "Clef expects a qwen3_5 checkpoint, got {}",
                parsed.model_type
            )));
        }
        let config = Self {
            text: parsed.text_config,
            quantization: parsed.quantization,
            head,
        };
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        let t = &self.text;
        let r = &t.rope_parameters;
        if t.model_type != "qwen3_5_text"
            || (
                t.hidden_size,
                t.intermediate_size,
                t.num_hidden_layers,
                t.full_attention_interval,
                t.num_attention_heads,
                t.num_key_value_heads,
                t.head_dim,
            ) != (4096, 12288, 32, 4, 16, 4, 256)
            || (
                t.linear_num_key_heads,
                t.linear_num_value_heads,
                t.linear_key_head_dim,
                t.linear_value_head_dim,
                t.linear_conv_kernel_dim,
            ) != (16, 32, 128, 128, 4)
            || t.vocab_size != 248_320
            || t.rms_norm_eps != 1e-6
            || !t.attn_output_gate
            || t.attention_bias
            || t.tie_word_embeddings
            || t.hidden_act != "silu"
            || r.rope_type != "default"
            || r.rope_theta != 10_000_000.0
            || r.partial_rotary_factor != 0.25
        {
            return Err(Error::Other(
                "Clef MLX supports the clef-flash geometry: Qwen3.5 hybrid, 32 layers, \
                 width 4096, full attention every fourth layer, untied head"
                    .into(),
            ));
        }
        if self.quantization
            != (Quantization {
                group_size: 64,
                bits: 4,
                mode: Mode::Affine,
            })
        {
            return Err(Error::Other(
                "Clef MLX requires affine 4-bit group-64 backbone weights".into(),
            ));
        }
        if self.head
            != (HeadConfig {
                hidden_size: 4096,
                width: 1024,
                routing_layers: 2,
                layers: 4,
                heads: 16,
                feedforward: 4096,
            })
        {
            return Err(Error::Other(
                "Clef MLX supports the clef-flash joint head".into(),
            ));
        }
        Ok(())
    }

    /// Layer `i` is full attention; the others are gated delta layers.
    pub fn is_attention(&self, layer: usize) -> bool {
        (layer + 1) % self.text.full_attention_interval == 0
    }

    pub fn rotary_dim(&self) -> usize {
        (self.text.head_dim as f32 * self.text.rope_parameters.partial_rotary_factor) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint() -> serde_json::Value {
        serde_json::json!({
            "model_type": "qwen3_5",
            "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
            "text_config": {
                "model_type": "qwen3_5_text", "hidden_size": 4096, "intermediate_size": 12288,
                "num_hidden_layers": 32, "full_attention_interval": 4, "num_attention_heads": 16,
                "num_key_value_heads": 4, "head_dim": 256, "linear_num_key_heads": 16,
                "linear_num_value_heads": 32, "linear_key_head_dim": 128,
                "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "vocab_size": 248320,
                "rms_norm_eps": 1e-6, "attn_output_gate": true, "attention_bias": false,
                "tie_word_embeddings": false, "hidden_act": "silu",
                "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25,
                                    "rope_type": "default", "mrope_section": [11, 11, 10]}
            }
        })
    }
    const HEAD: &str = r#"{"hidden_size": 4096, "width": 1024, "routing_layers": 2,
        "layers": 4, "heads": 16, "feedforward": 4096}"#;

    #[test]
    fn admits_only_the_clef_flash_geometry_and_quantization() {
        let config = Config::from_json(&checkpoint().to_string(), HEAD).unwrap();
        assert_eq!(config.rotary_dim(), 64);
        let attention: Vec<_> = (0..32).filter(|&i| config.is_attention(i)).collect();
        assert_eq!(attention, [3, 7, 11, 15, 19, 23, 27, 31]);
        for (pointer, value) in [
            ("/model_type", serde_json::json!("qwen3_5_moe")),
            ("/text_config/hidden_size", serde_json::json!(2560)),
            ("/text_config/tie_word_embeddings", serde_json::json!(true)),
            (
                "/text_config/rope_parameters/partial_rotary_factor",
                serde_json::json!(0.5),
            ),
            ("/quantization/bits", serde_json::json!(8)),
            ("/quantization/group_size", serde_json::json!(32)),
            ("/quantization/mode", serde_json::json!("mxfp4")),
        ] {
            let mut c = checkpoint();
            *c.pointer_mut(pointer).unwrap() = value;
            assert!(
                Config::from_json(&c.to_string(), HEAD).is_err(),
                "accepted {pointer}"
            );
        }
        assert!(
            Config::from_json(&checkpoint().to_string(), &HEAD.replace("1024", "512")).is_err()
        );
        assert!(Config::from_json(
            &checkpoint().to_string(),
            &HEAD.replace("}", r#", "x": 1}"#)
        )
        .is_err());
    }
}
