//! The qualified MiniCPM5-2B geometry, independent of shared Llama defaults.
use apxinf_core::{Error, Result};
use serde::Deserialize;

pub const CONTEXT_CAPACITY: usize = 4096;
pub const DSPARK_CONTEXT_CAPACITY: usize = 4096;
pub const PREFILL_CHUNK_TOKENS: usize = 256;

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
    pub const fn context_capacity(self) -> usize {
        match self {
            Self::Public | Self::Compiled => CONTEXT_CAPACITY,
            Self::DSpark => DSPARK_CONTEXT_CAPACITY,
        }
    }

    pub(crate) fn request_capacity(self, prompt: usize, output: usize) -> Result<usize> {
        let total = prompt
            .checked_add(output)
            .ok_or(Error::Contract("MiniCPM context overflow"))?;
        if prompt == 0 || total > self.context_capacity() {
            return Err(Error::Other(format!(
                "MiniCPM5 supports B1 with at most {} consumed positions for {self:?}",
                self.context_capacity()
            )));
        }
        Ok(total.div_ceil(PREFILL_CHUNK_TOKENS) * PREFILL_CHUNK_TOKENS)
    }

    pub(crate) fn prefill_rows(self, prompt: usize, position: usize) -> Option<usize> {
        if position >= prompt {
            return None;
        }
        if self == Self::DSpark {
            return (position == 0).then_some(prompt);
        }
        (position % PREFILL_CHUNK_TOKENS == 0)
            .then_some((prompt - position).min(PREFILL_CHUNK_TOKENS))
    }

    pub(crate) fn prepared_lengths(self, prompt: usize) -> Vec<usize> {
        let max_decode_rows = if self == Self::DSpark { 8 } else { 1 };
        let mut lengths = (1..=max_decode_rows).collect::<Vec<_>>();
        lengths.push(prompt);
        if self != Self::DSpark {
            lengths.push(prompt.min(PREFILL_CHUNK_TOKENS));
            let tail = prompt % PREFILL_CHUNK_TOKENS;
            if tail > 0 {
                lengths.push(tail);
            }
        }
        lengths.sort_unstable();
        lengths.dedup();
        lengths
    }

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

    #[test]
    fn request_capacity_checks_empty_edges_and_overflow() {
        for variant in [Variant::Public, Variant::Compiled, Variant::DSpark] {
            let limit = variant.context_capacity();
            assert!(variant.request_capacity(0, 1).is_err());
            assert_eq!(variant.request_capacity(limit - 1, 1).unwrap(), limit);
            assert!(variant.request_capacity(limit, 1).is_err());
            assert!(variant.request_capacity(usize::MAX, 1).is_err());
            assert_eq!(variant.request_capacity(256, 1).unwrap(), 512);
        }
        assert_eq!(Variant::DSpark.context_capacity(), 4096);
    }

    #[test]
    fn prefill_admits_only_exact_chunk_offsets_and_shapes() {
        for variant in [Variant::Public, Variant::Compiled] {
            for (prompt, expected) in [
                (1, vec![(0, 1)]),
                (255, vec![(0, 255)]),
                (256, vec![(0, 256)]),
                (257, vec![(0, 256), (256, 1)]),
                (513, vec![(0, 256), (256, 256), (512, 1)]),
            ] {
                let actual = (0..=prompt)
                    .filter_map(|position| {
                        variant
                            .prefill_rows(prompt, position)
                            .map(|rows| (position, rows))
                    })
                    .collect::<Vec<_>>();
                assert_eq!(actual, expected);
            }
            assert_eq!(variant.prefill_rows(0, 0), None);
            assert_eq!(variant.prefill_rows(257, usize::MAX), None);
            assert_eq!(variant.prepared_lengths(513), vec![1, 256, 513]);
            assert_eq!(variant.prepared_lengths(511), vec![1, 255, 256, 511]);
        }
        assert_eq!(Variant::DSpark.prefill_rows(513, 0), Some(513));
        assert_eq!(Variant::DSpark.prefill_rows(513, 256), None);
        assert_eq!(
            Variant::DSpark.prepared_lengths(513),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 513]
        );
    }
}
