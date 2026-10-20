//! Dense Qwen3 checkpoint configuration. Head width is independent of hidden size.

use apxinf_core::{Error, Result};
use serde::Deserialize;

/// The source-bound migration profile; larger contexts need separate evidence.
pub const MAX_CONTEXT: usize = 2048;

#[derive(Clone, Debug, Deserialize)]
pub struct Qwen3Config {
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
    pub attention_bias: bool,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
}

impl Qwen3Config {
    pub fn validate_checkpoint_scope(&self) -> Result<()> {
        if (
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.vocab_size,
        ) != (1024, 3072, 28, 16, 8, 128, 151936)
            || !self.tie_word_embeddings
            || self.rms_norm_eps != 1e-6
            || self.rope_theta != 1_000_000.
        {
            return Err(Error::Other("Qwen3 MLX migration currently supports the Qwen3-0.6B geometry, tied embeddings, epsilon=1e-6 and RoPE base=1000000 only".into()));
        }
        Ok(())
    }
    pub fn from_json(json: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(json)
            .map_err(|e| Error::Other(format!("invalid Qwen3 config: {e}")))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        let dimensions = [
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.vocab_size,
            self.max_position_embeddings,
        ];
        if dimensions.iter().any(|&n| n == 0 || n > i32::MAX as usize)
            || self.head_dim % 2 != 0
            || self.num_attention_heads % self.num_key_value_heads != 0
            || self.max_position_embeddings > 16_777_216
            || self
                .num_attention_heads
                .checked_mul(self.head_dim)
                .is_none()
            || self
                .num_key_value_heads
                .checked_mul(self.head_dim)
                .is_none()
        {
            return Err(Error::Other(
                "invalid Qwen3 dimensions/head grouping".into(),
            ));
        }
        if self.model_type != "qwen3"
            || self.hidden_act != "silu"
            || self.attention_bias
            || self.use_sliding_window
            || self.rope_scaling.is_some()
        {
            return Err(Error::Other(
                "native Qwen3 supports dense SiLU, full attention, bias-free projections and unscaled RoPE".into(),
            ));
        }
        if !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.
        {
            return Err(Error::Other(
                "Qwen3 norm epsilon and RoPE base must be positive and finite".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Bf16Public,
    Bf16Compiled,
    MixedW8,
}

impl Variant {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("bf16-compiled") {
            "bf16-public" => Ok(Self::Bf16Public),
            "auto" | "bf16-compiled" => Ok(Self::Bf16Compiled),
            "mixed-w8" => Ok(Self::MixedW8),
            other => Err(Error::Other(format!(
                "unsupported Qwen3 MLX variant `{other}`; implemented: bf16-public, bf16-compiled, mixed-w8"
            ))),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Bf16Public => "bf16-public",
            Self::Bf16Compiled => "bf16-compiled",
            Self::MixedW8 => "mixed-w8",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn valid() -> serde_json::Value {
        serde_json::json!({"model_type":"qwen3", "hidden_size":1024,
            "intermediate_size":3072, "num_hidden_layers":28, "num_attention_heads":16,
            "num_key_value_heads":8, "head_dim":128, "vocab_size":151936,
            "max_position_embeddings":40960, "rms_norm_eps":0.000001,
            "rope_theta":1000000, "tie_word_embeddings":true, "hidden_act":"silu"})
    }
    #[test]
    fn explicit_head_width_is_not_hidden_div_heads() {
        let c = Qwen3Config::from_json(&valid().to_string()).unwrap();
        assert_eq!(c.head_dim, 128);
        assert_ne!(c.head_dim, c.hidden_size / c.num_attention_heads);
        let mut missing = valid();
        missing.as_object_mut().unwrap().remove("head_dim");
        assert!(Qwen3Config::from_json(&missing.to_string()).is_err());
    }
    #[test]
    fn rejects_unimplemented_semantics() {
        for (key, value) in [
            ("attention_bias", serde_json::json!(true)),
            ("use_sliding_window", serde_json::json!(true)),
            ("rope_scaling", serde_json::json!({"factor":2})),
            ("head_dim", serde_json::json!(127)),
            ("num_key_value_heads", serde_json::json!(3)),
        ] {
            let mut c = valid();
            c[key] = value;
            assert!(
                Qwen3Config::from_json(&c.to_string()).is_err(),
                "accepted {key}"
            );
        }
        assert_eq!(Variant::parse(Some("mixed-w8")).unwrap(), Variant::MixedW8);
    }

    #[test]
    fn checkpoint_scope_is_explicit() {
        let mut c = Qwen3Config::from_json(&valid().to_string()).unwrap();
        c.validate_checkpoint_scope().unwrap();
        c.hidden_size = 2048;
        assert!(c.validate_checkpoint_scope().is_err());
        assert_eq!(MAX_CONTEXT, 2048);
    }
}
