pub const MODEL_VOCAB_SIZE: usize = 248_320;
pub const IMAGE_TOKEN_ID: u32 = 248_056;
pub const MODEL_MAX_POSITION: usize = 262_144;
pub const EOS_TOKEN_IDS: [u32; 2] = [248_046, 248_044];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    Gdn,
    FullAttention,
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen35ConfigError {
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("missing required field: {0}")]
    MissingField(String),
    #[error("invalid type for field: {0}")]
    InvalidType(String),
    #[error("invalid configuration: {0}")]
    InvalidValue(String),
    #[error("unsupported architecture: {0}")]
    Architecture(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen35ModelConfig {
    pub vocab_size: usize,
    pub image_token_id: u32,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub layer_types: Vec<LayerType>,
    pub linear_key_heads: usize,
    pub linear_value_heads: usize,
    pub linear_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub full_attention_heads: usize,
    pub full_attention_kv_heads: usize,
    pub full_attention_head_dim: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub partial_rotary_factor: f32,
    pub attn_output_gate: bool,
    pub eos_token_ids: [u32; 2],
}

impl Qwen35ModelConfig {
    pub fn from_json_file(path: impl AsRef<std::path::Path>) -> Result<Self, Qwen35ConfigError> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| Qwen35ConfigError::InvalidValue(e.to_string()))?;
        Self::from_json_str(&raw)
    }

    pub fn from_json_str(raw: &str) -> Result<Self, Qwen35ConfigError> {
        let root: serde_json::Value = serde_json::from_str(raw)?;
        let architecture = root
            .get("architectures")
            .and_then(serde_json::Value::as_array)
            .and_then(|a| a.first())
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| Qwen35ConfigError::MissingField("architectures".into()))?;
        if architecture != "Qwen3_5ForConditionalGeneration" {
            return Err(Qwen35ConfigError::Architecture(architecture.into()));
        }
        let model_type = required_str(&root, "model_type")?;
        if model_type != "qwen3_5" {
            return Err(Qwen35ConfigError::Architecture(model_type.into()));
        }
        let image_token_id = required_u64(&root, "image_token_id")?;
        if image_token_id != IMAGE_TOKEN_ID as u64 {
            return Err(Qwen35ConfigError::InvalidValue("image_token_id".into()));
        }
        validate_quantization_config(&root)?;
        let text = root
            .get("text_config")
            .ok_or_else(|| Qwen35ConfigError::MissingField("text_config".into()))?;
        let text_type = required_str(text, "model_type")?;
        if text_type != "qwen3_5_text" {
            return Err(Qwen35ConfigError::Architecture(text_type.into()));
        }

        let vocab_size = required_usize(text, "vocab_size")?;
        let hidden_size = required_usize(text, "hidden_size")?;
        let intermediate_size = required_usize(text, "intermediate_size")?;
        let num_hidden_layers = required_usize(text, "num_hidden_layers")?;
        let layer_values = text
            .get("layer_types")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| Qwen35ConfigError::MissingField("text_config.layer_types".into()))?;
        if layer_values.len() != num_hidden_layers {
            return Err(Qwen35ConfigError::InvalidValue("layer_types length".into()));
        }
        let mut layer_types = Vec::with_capacity(layer_values.len());
        for value in layer_values {
            let value = value
                .as_str()
                .ok_or_else(|| Qwen35ConfigError::InvalidType("layer_types".into()))?;
            layer_types.push(match value {
                "linear_attention" => LayerType::Gdn,
                "full_attention" => LayerType::FullAttention,
                _ => {
                    return Err(Qwen35ConfigError::InvalidValue(format!(
                        "layer_types={value}"
                    )))
                }
            });
        }
        let linear_key_heads = required_usize(text, "linear_num_key_heads")?;
        let linear_value_heads = required_usize(text, "linear_num_value_heads")?;
        let linear_key_dim = required_usize(text, "linear_key_head_dim")?;
        let linear_value_dim = required_usize(text, "linear_value_head_dim")?;
        let linear_conv_kernel_dim = required_usize(text, "linear_conv_kernel_dim")?;
        if linear_key_dim != linear_value_dim {
            return Err(Qwen35ConfigError::InvalidValue(
                "linear head dimensions".into(),
            ));
        }
        let full_attention_heads = required_usize(text, "num_attention_heads")?;
        let full_attention_kv_heads = required_usize(text, "num_key_value_heads")?;
        let full_attention_head_dim = required_usize(text, "head_dim")?;
        let max_position_embeddings = required_usize(text, "max_position_embeddings")?;
        let rms_norm_eps = required_f32(text, "rms_norm_eps")?;
        let partial_rotary_factor = required_f32(text, "partial_rotary_factor")?;
        let attn_output_gate = required_bool(text, "attn_output_gate")?;
        let eos_token_id = required_u64(text, "eos_token_id")?;
        if vocab_size != MODEL_VOCAB_SIZE
            || eos_token_id != EOS_TOKEN_IDS[1] as u64
            || hidden_size == 0
            || num_hidden_layers == 0
            || linear_conv_kernel_dim == 0
        {
            return Err(Qwen35ConfigError::InvalidValue("model contract".into()));
        }
        if partial_rotary_factor <= 0.0 || partial_rotary_factor > 1.0 {
            return Err(Qwen35ConfigError::InvalidValue(
                "partial_rotary_factor".into(),
            ));
        }
        Ok(Self {
            vocab_size,
            image_token_id: image_token_id as u32,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            layer_types,
            linear_key_heads,
            linear_value_heads,
            linear_head_dim: linear_key_dim,
            linear_conv_kernel_dim,
            full_attention_heads,
            full_attention_kv_heads,
            full_attention_head_dim,
            max_position_embeddings,
            rms_norm_eps,
            partial_rotary_factor,
            attn_output_gate,
            eos_token_ids: EOS_TOKEN_IDS,
        })
    }

    pub fn gdn_layer_count(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|t| **t == LayerType::Gdn)
            .count()
    }

    pub fn full_attention_layer_count(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|t| **t == LayerType::FullAttention)
            .count()
    }

    pub fn partial_rotary_dim(&self) -> usize {
        (self.full_attention_head_dim as f32 * self.partial_rotary_factor) as usize
    }
}

impl Qwen35ModelConfig {
    pub fn validate_frozen_contract(&self) -> Result<(), Qwen35ConfigError> {
        let exact = [
            (self.vocab_size == MODEL_VOCAB_SIZE, "vocab_size"),
            (self.eos_token_ids == EOS_TOKEN_IDS, "eos_token_id"),
            (self.hidden_size == 5_120, "hidden_size"),
            (self.intermediate_size == 17_408, "intermediate_size"),
            (self.num_hidden_layers == 64, "num_hidden_layers"),
            (self.linear_key_heads == 16, "linear_num_key_heads"),
            (self.linear_value_heads == 48, "linear_num_value_heads"),
            (self.linear_head_dim == 128, "linear_head_dim"),
            (self.linear_conv_kernel_dim == 4, "linear_conv_kernel_dim"),
            (self.full_attention_heads == 24, "num_attention_heads"),
            (self.full_attention_kv_heads == 4, "num_key_value_heads"),
            (self.full_attention_head_dim == 256, "head_dim"),
            (
                self.max_position_embeddings == MODEL_MAX_POSITION,
                "max_position_embeddings",
            ),
            (
                self.rms_norm_eps.to_bits() == 1e-6f32.to_bits(),
                "rms_norm_eps",
            ),
            (
                self.partial_rotary_factor.to_bits() == 0.25f32.to_bits(),
                "partial_rotary_factor",
            ),
            (self.attn_output_gate, "attn_output_gate"),
        ];
        if let Some((false, field)) = exact.into_iter().find(|(valid, _)| !valid) {
            return Err(Qwen35ConfigError::InvalidValue(format!(
                "model contract.{field}"
            )));
        }
        if self.layer_types.len() != 64
            || self.layer_types.iter().enumerate().any(|(index, layer)| {
                *layer
                    != if index % 4 == 3 {
                        LayerType::FullAttention
                    } else {
                        LayerType::Gdn
                    }
            })
        {
            return Err(Qwen35ConfigError::InvalidValue(
                "model contract.layer_types".into(),
            ));
        }
        Ok(())
    }
}

fn required<'a>(
    root: &'a serde_json::Value,
    name: &str,
) -> Result<&'a serde_json::Value, Qwen35ConfigError> {
    root.get(name)
        .ok_or_else(|| Qwen35ConfigError::MissingField(name.into()))
}
fn required_str<'a>(root: &'a serde_json::Value, name: &str) -> Result<&'a str, Qwen35ConfigError> {
    required(root, name)?
        .as_str()
        .ok_or_else(|| Qwen35ConfigError::InvalidType(name.into()))
}
fn required_u64(root: &serde_json::Value, name: &str) -> Result<u64, Qwen35ConfigError> {
    required(root, name)?
        .as_u64()
        .ok_or_else(|| Qwen35ConfigError::InvalidType(name.into()))
}
fn required_usize(root: &serde_json::Value, name: &str) -> Result<usize, Qwen35ConfigError> {
    usize::try_from(required_u64(root, name)?)
        .map_err(|_| Qwen35ConfigError::InvalidValue(name.into()))
}
fn required_f32(root: &serde_json::Value, name: &str) -> Result<f32, Qwen35ConfigError> {
    required(root, name)?
        .as_f64()
        .map(|v| v as f32)
        .ok_or_else(|| Qwen35ConfigError::InvalidType(name.into()))
}
fn required_bool(root: &serde_json::Value, name: &str) -> Result<bool, Qwen35ConfigError> {
    required(root, name)?
        .as_bool()
        .ok_or_else(|| Qwen35ConfigError::InvalidType(name.into()))
}

fn validate_quantization_config(root: &serde_json::Value) -> Result<(), Qwen35ConfigError> {
    let quant = required(root, "quantization_config")?;
    let format = required_str(quant, "format")?;
    if format != "pack-quantized" {
        return Err(Qwen35ConfigError::InvalidValue(
            "quantization_config.format".into(),
        ));
    }
    let group = required(quant, "config_groups")?
        .get("group_0")
        .ok_or_else(|| {
            Qwen35ConfigError::MissingField("quantization_config.config_groups.group_0".into())
        })?;
    if required_str(group, "format")? != "pack-quantized" {
        return Err(Qwen35ConfigError::InvalidValue(
            "quantization_config.group_0.format".into(),
        ));
    }
    let weight = required(group, "weights")?;
    if required_usize(weight, "group_size")? != 32
        || required_usize(weight, "num_bits")? != 4
        || required_bool(weight, "symmetric")?
        || required_str(weight, "strategy")? != "group"
        || required_str(weight, "type")? != "int"
    {
        return Err(Qwen35ConfigError::InvalidValue(
            "quantization_config.weights".into(),
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen35Config {
    pub vocab_size: usize,
    pub image_token_id: u32,
    pub max_position_embeddings: usize,
}

impl Qwen35Config {
    pub const fn frozen() -> Self {
        Self {
            vocab_size: MODEL_VOCAB_SIZE,
            image_token_id: IMAGE_TOKEN_ID,
            max_position_embeddings: MODEL_MAX_POSITION,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    #[test]
    fn frozen_model_constants_match_checkpoint_contract() {
        assert_eq!(super::MODEL_VOCAB_SIZE, 248_320);
        assert_eq!(super::IMAGE_TOKEN_ID, 248_056);
    }

    #[test]
    fn parses_pinned_nested_text_config_without_defaults() {
        let raw = std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../fixtures/qwen35-metadata/config.json"),
        )
        .unwrap();
        let config = super::Qwen35ModelConfig::from_json_str(&raw).unwrap();
        assert_eq!(config.vocab_size, 248_320);
        assert_eq!(config.hidden_size, 5_120);
        assert_eq!(config.intermediate_size, 17_408);
        assert_eq!(config.num_hidden_layers, 64);
        assert_eq!(config.gdn_layer_count(), 48);
        assert_eq!(config.full_attention_layer_count(), 16);
        assert_eq!(config.layer_types[3], super::LayerType::FullAttention);
        assert_eq!(config.layer_types[0], super::LayerType::Gdn);
        assert_eq!(config.linear_key_heads, 16);
        assert_eq!(config.linear_value_heads, 48);
        assert_eq!(config.linear_head_dim, 128);
        assert_eq!(config.linear_conv_kernel_dim, 4);
        assert_eq!(config.partial_rotary_dim(), 64);
        assert!(config.attn_output_gate);
        assert_eq!(config.eos_token_ids, [248_046, 248_044]);
    }

    #[test]
    fn rejects_wrong_architecture_and_missing_required_field() {
        let raw = r#"{
          "architectures":["Qwen3_5ForConditionalGeneration"],
          "model_type":"qwen3_5",
          "image_token_id":248056,
          "text_config":{"model_type":"qwen3_5_text","vocab_size":248320}
        }"#;
        assert!(matches!(
            super::Qwen35ModelConfig::from_json_str(raw),
            Err(super::Qwen35ConfigError::MissingField(_))
        ));

        let wrong = raw.replace("qwen3_5", "llama");
        assert!(matches!(
            super::Qwen35ModelConfig::from_json_str(&wrong),
            Err(super::Qwen35ConfigError::Architecture(_))
        ));
    }

    #[test]
    fn rejects_non_frozen_dimensions_and_layer_schedule() {
        let raw = std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../fixtures/qwen35-metadata/config.json"),
        )
        .unwrap();

        let mut wrong_dimension: serde_json::Value = serde_json::from_str(&raw).unwrap();
        wrong_dimension["text_config"]["hidden_size"] = serde_json::json!(5121);
        assert!(matches!(
            super::Qwen35ModelConfig::from_json_str(&wrong_dimension.to_string())
                .and_then(|config| config.validate_frozen_contract()),
            Err(super::Qwen35ConfigError::InvalidValue(field)) if field == "model contract.hidden_size"
        ));

        let mut wrong_schedule: serde_json::Value = serde_json::from_str(&raw).unwrap();
        wrong_schedule["text_config"]["layer_types"][2] = serde_json::json!("full_attention");
        assert!(matches!(
            super::Qwen35ModelConfig::from_json_str(&wrong_schedule.to_string())
                .and_then(|config| config.validate_frozen_contract()),
            Err(super::Qwen35ConfigError::InvalidValue(field)) if field == "model contract.layer_types"
        ));
    }
}
