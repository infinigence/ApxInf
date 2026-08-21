//! Qwen3.5 text-model configuration parsed from `config.json` → `text_config`.

use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct RopeParameters {
    pub rope_type: String,
    pub rope_theta: f64,
    #[serde(default)]
    pub partial_rotary_factor: f64,
    #[serde(default)]
    pub mrope_interleaved: bool,
    #[serde(default)]
    pub mrope_section: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Qwen35TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
    pub layer_types: Vec<LayerType>,
    #[serde(default)]
    pub full_attention_interval: Option<usize>,
    #[serde(default)]
    pub attn_output_gate: bool,
    #[serde(default)]
    pub output_gate_type: Option<String>,
    #[serde(default)]
    pub hidden_act: String,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub bos_token_id: Option<usize>,
    #[serde(default)]
    pub eos_token_id: Option<usize>,
    #[serde(default)]
    pub max_position_embeddings: usize,
}

impl Qwen35TextConfig {
    /// Read the nested `text_config` object out of a Qwen3.5 `config.json`.
    pub fn from_json_file(path: &Path) -> Result<Self, String> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        Self::from_json(&raw)
    }

    pub fn from_json(raw: &str) -> Result<Self, String> {
        let parsed: serde_json::Value = serde_json::from_str(raw)
            .map_err(|e| format!("parse config.json: {e}"))?;
        let text = parsed
            .get("text_config")
            .cloned()
            .unwrap_or_else(|| parsed.clone());
        serde_json::from_value(text).map_err(|e| format!("parse text_config: {e}"))
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .map(|r| r.rope_theta)
            .or(self.rope_theta)
            .unwrap_or(10000.0)
    }

    pub fn partial_rotary_factor(&self) -> f64 {
        self.rope_parameters
            .as_ref()
            .and_then(|r| {
                if r.partial_rotary_factor > 0.0 {
                    Some(r.partial_rotary_factor)
                } else {
                    None
                }
            })
            .or(self.partial_rotary_factor)
            .unwrap_or(1.0)
    }

    /// Number of `head_dim` dimensions that receive RoPE.
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.partial_rotary_factor()).floor() as usize
    }

    pub fn is_full_attention(&self, layer_idx: usize) -> bool {
        match self.layer_types.get(layer_idx) {
            Some(LayerType::FullAttention) => true,
            Some(LayerType::LinearAttention) => false,
            None => self
                .full_attention_interval
                .map(|interval| (layer_idx + 1) % interval == 0)
                .unwrap_or(false),
        }
    }
}
