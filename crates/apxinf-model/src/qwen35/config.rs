use std::path::Path;

use apxinf_core::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    LinearAttention,
    FullAttention,
}

#[derive(Clone, Debug)]
pub struct Qwen35TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub mrope_section: [usize; 3],
    pub mrope_interleaved: bool,
    pub full_attention_interval: usize,
    pub attn_output_gate: bool,
    pub output_gate_type: String,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub layer_types: Vec<LayerKind>,
    pub tie_word_embeddings: bool,
}

#[derive(Clone, Debug)]
pub struct Qwen35Config {
    pub model_type: String,
    pub text: Qwen35TextConfig,
    pub image_token_id: Option<u32>,
    pub video_token_id: Option<u32>,
    pub vision_start_token_id: Option<u32>,
    pub vision_end_token_id: Option<u32>,
}

impl Qwen35Config {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
        Self::from_json_str(&raw)
    }

    pub fn from_json_str(raw: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(raw)
            .map_err(|error| Error::Other(format!("qwen3_5 config json: {error}")))?;
        let model_type = value
            .get("model_type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| Error::Other("qwen3_5 config: missing model_type".into()))?;
        if model_type != "qwen3_5" {
            return Err(Error::Other(format!(
                "qwen3_5 config: expected model_type `qwen3_5`, got `{model_type}`"
            )));
        }
        let text_value = value
            .get("text_config")
            .ok_or_else(|| Error::Other("qwen3_5 config: missing text_config".into()))?;
        let text = parse_text_config(text_value)?;
        Ok(Self {
            model_type: model_type.to_owned(),
            text,
            image_token_id: optional_u32(&value, "image_token_id"),
            video_token_id: optional_u32(&value, "video_token_id"),
            vision_start_token_id: optional_u32(&value, "vision_start_token_id"),
            vision_end_token_id: optional_u32(&value, "vision_end_token_id"),
        })
    }
}

fn parse_text_config(value: &serde_json::Value) -> Result<Qwen35TextConfig> {
    let layer_values = value
        .get("layer_types")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::Other("qwen3_5 text_config: missing layer_types".into()))?;
    let layer_types = layer_values
        .iter()
        .map(|item| match item.as_str() {
            Some("linear_attention") => Ok(LayerKind::LinearAttention),
            Some("full_attention") => Ok(LayerKind::FullAttention),
            Some(other) => Err(Error::Other(format!(
                "qwen3_5 text_config: unsupported layer type `{other}`"
            ))),
            None => Err(Error::Other(
                "qwen3_5 text_config: layer_types entries must be strings".into(),
            )),
        })
        .collect::<Result<Vec<_>>>()?;
    let n_layers = required_usize(value, "num_hidden_layers")?;
    if layer_types.len() != n_layers {
        return Err(Error::Other(format!(
            "qwen3_5 text_config: layer_types length {} != num_hidden_layers {n_layers}",
            layer_types.len()
        )));
    }

    let rope = value
        .get("rope_parameters")
        .ok_or_else(|| Error::Other("qwen3_5 text_config: missing rope_parameters".into()))?;
    let section = rope
        .get("mrope_section")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::Other("qwen3_5 text_config: missing rope_parameters.mrope_section".into()))?;
    if section.len() != 3 {
        return Err(Error::Other(format!(
            "qwen3_5 text_config: mrope_section must have 3 entries, got {}",
            section.len()
        )));
    }

    Ok(Qwen35TextConfig {
        hidden_size: required_usize(value, "hidden_size")?,
        intermediate_size: required_usize(value, "intermediate_size")?,
        n_layers,
        n_heads: required_usize(value, "num_attention_heads")?,
        n_kv_heads: required_usize(value, "num_key_value_heads")?,
        head_dim: required_usize(value, "head_dim")?,
        vocab_size: required_usize(value, "vocab_size")?,
        max_position_embeddings: required_usize(value, "max_position_embeddings")?,
        rms_norm_eps: required_f32(value, "rms_norm_eps")?,
        rope_theta: required_f32(rope, "rope_theta")?,
        partial_rotary_factor: required_f32(value, "partial_rotary_factor")?,
        mrope_section: [
            json_u64(&section[0], "mrope_section[0]")? as usize,
            json_u64(&section[1], "mrope_section[1]")? as usize,
            json_u64(&section[2], "mrope_section[2]")? as usize,
        ],
        mrope_interleaved: rope
            .get("mrope_interleaved")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true),
        full_attention_interval: required_usize(value, "full_attention_interval")?,
        attn_output_gate: value
            .get("attn_output_gate")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        output_gate_type: value
            .get("output_gate_type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("none")
            .to_owned(),
        linear_conv_kernel_dim: required_usize(value, "linear_conv_kernel_dim")?,
        linear_key_head_dim: required_usize(value, "linear_key_head_dim")?,
        linear_num_key_heads: required_usize(value, "linear_num_key_heads")?,
        linear_num_value_heads: required_usize(value, "linear_num_value_heads")?,
        linear_value_head_dim: required_usize(value, "linear_value_head_dim")?,
        layer_types,
        tie_word_embeddings: value
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
    })
}

fn required_usize(value: &serde_json::Value, key: &str) -> Result<usize> {
    Ok(value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| Error::Other(format!("qwen3_5 config: missing integer `{key}`")))?
        as usize)
}

fn required_f32(value: &serde_json::Value, key: &str) -> Result<f32> {
    Ok(value
        .get(key)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| Error::Other(format!("qwen3_5 config: missing number `{key}`")))?
        as f32)
}

fn json_u64(value: &serde_json::Value, name: &str) -> Result<u64> {
    value
        .as_u64()
        .ok_or_else(|| Error::Other(format!("qwen3_5 config: `{name}` must be an integer")))
}

fn optional_u32(value: &serde_json::Value, key: &str) -> Option<u32> {
    value
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .map(|value| value as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN_CONFIG: &str = r#"{
        "model_type": "qwen3_5",
        "image_token_id": 248056,
        "text_config": {
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 4,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 248320,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "partial_rotary_factor": 0.25,
            "full_attention_interval": 4,
            "attn_output_gate": true,
            "output_gate_type": "swish",
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
            "tie_word_embeddings": false,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10],
                "rope_theta": 10000000
            }
        }
    }"#;

    #[test]
    fn parses_qwen35_config() {
        let config = Qwen35Config::from_json_str(MIN_CONFIG).unwrap();
        assert_eq!(config.model_type, "qwen3_5");
        assert_eq!(config.text.n_layers, 4);
        assert_eq!(config.text.layer_types[3], LayerKind::FullAttention);
        assert_eq!(config.text.mrope_section, [11, 11, 10]);
        assert_eq!(config.text.linear_num_value_heads, 48);
        assert!(config.text.attn_output_gate);
    }
}
