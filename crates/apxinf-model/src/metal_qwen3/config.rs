//! Dense Qwen3 checkpoint configuration. Head width is independent of hidden size.

use apxinf_core::{Error, Result};
use serde::Deserialize;

/// Maximum total positions in the Qwen3-0.6B checkpoint.
pub const MAX_CONTEXT: usize = 40960;

#[derive(Clone, Copy)]
pub(super) enum PrefillChunk {
    Tokens512 = 512,
    Tokens1024 = 1024,
    Tokens2048 = 2048,
}

impl PrefillChunk {
    pub(super) fn from_env() -> Result<Option<Self>> {
        match std::env::var("APXINF_QWEN3_PREFILL_CHUNK") {
            Err(std::env::VarError::NotPresent) => Ok(None),
            Ok(value) => match value.as_str() {
                "512" => Ok(Some(Self::Tokens512)),
                "1024" => Ok(Some(Self::Tokens1024)),
                "2048" => Ok(Some(Self::Tokens2048)),
                _ => Err(Error::Contract(
                    "APXINF_QWEN3_PREFILL_CHUNK must be 512, 1024 or 2048",
                )),
            },
            Err(_) => Err(Error::Contract("APXINF_QWEN3_PREFILL_CHUNK must be UTF-8")),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct PrefillPlan {
    pub(super) prefix_len: usize,
    chunk: usize,
}

impl PrefillPlan {
    pub(super) fn new(prompt: usize, chunk: PrefillChunk) -> Result<Self> {
        Ok(Self {
            prefix_len: prompt
                .checked_sub(1)
                .ok_or(Error::Contract("Qwen3 prompt must not be empty"))?,
            chunk: chunk as usize,
        })
    }

    pub(super) fn chunks(self) -> impl Iterator<Item = std::ops::Range<usize>> {
        (0..self.prefix_len)
            .step_by(self.chunk)
            .map(move |start| start..(start + self.chunk).min(self.prefix_len))
    }

    pub(super) fn required_rows(self) -> Vec<usize> {
        let mut rows: Vec<_> = self
            .chunks()
            .map(|r| r.len())
            .chain(std::iter::once(1))
            .collect();
        rows.sort_unstable();
        rows.dedup();
        rows
    }
}

pub(super) fn attention_width(end: usize, capacity: usize) -> usize {
    if capacity <= 2048 {
        capacity
    } else {
        (end.div_ceil(256) * 256).min(capacity)
    }
}

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
    pub(super) fn request_capacity(&self, prompt: usize, output: usize) -> Result<usize> {
        let total = prompt
            .checked_add(output)
            .ok_or_else(|| Error::Other("Qwen3 context overflow".into()))?;
        let limit = self.max_position_embeddings.min(MAX_CONTEXT);
        if prompt == 0 || total > limit {
            return Err(Error::Other(format!(
                "Qwen3 requires a nonempty prompt and prompt + output <= {limit}"
            )));
        }
        Ok(total.div_ceil(256).saturating_mul(256).min(limit))
    }

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

    #[test]
    fn prefill_plan_preserves_final_token_and_nonzero_tails() {
        for chunk in [
            PrefillChunk::Tokens512,
            PrefillChunk::Tokens1024,
            PrefillChunk::Tokens2048,
        ] {
            let width = chunk as usize;
            assert!(PrefillPlan::new(0, chunk).is_err());
            let one = PrefillPlan::new(1, chunk).unwrap();
            assert_eq!(one.chunks().count(), 0);
            assert_eq!(one.required_rows(), vec![1]);
            let full = PrefillPlan::new(width + 1, chunk).unwrap();
            assert_eq!(full.chunks().collect::<Vec<_>>(), vec![0..width]);
            assert_eq!(full.required_rows(), vec![1, width]);
            let tail = PrefillPlan::new(width + 3, chunk).unwrap();
            assert_eq!(
                tail.chunks().collect::<Vec<_>>(),
                vec![0..width, width..width + 2]
            );
            assert_eq!(tail.required_rows(), vec![1, 2, width]);
        }
    }

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
        assert_eq!(MAX_CONTEXT, 40960);
    }

    #[test]
    fn request_capacity_grows_with_demand_and_respects_checkpoint() {
        let mut c = Qwen3Config::from_json(&valid().to_string()).unwrap();
        assert_eq!(c.request_capacity(1, 0).unwrap(), 256);
        assert_eq!(c.request_capacity(255, 1).unwrap(), 256);
        assert_eq!(c.request_capacity(256, 1).unwrap(), 512);
        assert_eq!(c.request_capacity(40832, 128).unwrap(), 40960);
        assert!(c.request_capacity(40833, 128).is_err());
        assert!(c.request_capacity(0, 1).is_err());
        assert!(c.request_capacity(usize::MAX, 1).is_err());
        c.max_position_embeddings = 300;
        assert_eq!(c.request_capacity(300, 0).unwrap(), 300);
        assert!(c.request_capacity(300, 1).is_err());
    }
}
