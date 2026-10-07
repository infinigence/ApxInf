//! The qualified MiniCPM5-2B geometry, independent of shared Llama defaults.
use apxinf_core::{Error, Result};
use serde::Deserialize;

pub const CONTEXT_CAPACITY: usize = 4096;

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub hidden_act: String,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub attention_bias: bool,
}

impl Config {
    pub fn from_json(raw: &str) -> Result<Self> {
        let config: Self =
            serde_json::from_str(raw).map_err(|e| Error::Other(format!("MiniCPM5 config: {e}")))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != "llama"
            || (
                self.hidden_size,
                self.intermediate_size,
                self.num_hidden_layers,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                self.vocab_size,
            ) != (2048, 6144, 42, 16, 2, 128, 130560)
            || self.max_position_embeddings < CONTEXT_CAPACITY
            || self.rms_norm_eps != 1e-6
            || self.rope_theta != 5_000_000.0
            || self.tie_word_embeddings
            || self.hidden_act != "silu"
            || self.rope_scaling.is_some()
            || self.attention_bias
        {
            return Err(Error::Other(
                "MiniCPM5 MLX supports the official 2B BF16 geometry, unscaled NeoX RoPE and untied head".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Public,
    Compiled,
    DSpark,
}

impl Variant {
    pub fn parse(name: Option<&str>) -> Result<Self> {
        match name.unwrap_or("bf16-compiled") {
            "bf16-public" => Ok(Self::Public),
            "bf16-compiled" => Ok(Self::Compiled),
            "dspark" => Ok(Self::DSpark),
            name => Err(Error::Other(format!(
                "unsupported MiniCPM5 MLX variant: {name}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        Config {
            model_type: "llama".into(),
            hidden_size: 2048,
            intermediate_size: 6144,
            num_hidden_layers: 42,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            head_dim: 128,
            vocab_size: 130560,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-6,
            rope_theta: 5_000_000.0,
            tie_word_embeddings: false,
            hidden_act: "silu".into(),
            rope_scaling: None,
            attention_bias: false,
        }
    }

    #[test]
    fn refuses_nearby_llama_shapes_and_unsupported_attention_semantics() {
        assert!(config().validate().is_ok());
        let mut c = config();
        c.num_hidden_layers = 32;
        assert!(c.validate().is_err());
        let mut c = config();
        c.rope_scaling = Some(serde_json::json!({"factor": 2}));
        assert!(c.validate().is_err());
        let mut c = config();
        c.tie_word_embeddings = true;
        assert!(c.validate().is_err());
        assert_eq!(Variant::parse(None).unwrap(), Variant::Compiled);
        assert!(Variant::parse(Some("mixed-w8")).is_err());
    }
}
