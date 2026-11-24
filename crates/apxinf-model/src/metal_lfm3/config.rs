//! LFM3-1.7B checkpoint configuration, layer plan and rotary geometry.
//!
//! The checkpoint is not Llama-shaped: 24 layers alternate three operator types
//! (width-7 gated short conv, full attention, 256-window sliding attention) and
//! use two independent RoPE parameter sets. Everything the forward paths need is
//! derived and validated here, once, so no execution path guesses a default.

use apxinf_core::{Error, Result};
use serde::Deserialize;

/// Largest total position count this family admits.
///
/// The checkpoint declares `max_position_embeddings = 262144`, but full-attention
/// layers keep an unbounded dense KV row (12 KiB per token over six layers), so
/// this port admits a smaller envelope instead of promising the checkpoint bound.
pub const MAX_CONTEXT: usize = 8192;

/// Capacity rounding so a growing request set does not reallocate every prompt.
const CAPACITY_GRAIN: usize = 256;

/// Operator type of one decoder layer, from `config.json: layer_types`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Width-`conv_L_cache` gated depthwise short convolution over 2048 channels.
    Conv,
    /// Unbounded causal attention, RoPE theta 1e6 with partial rotation.
    FullAttention,
    /// Causal attention restricted to a 256-token window, RoPE theta 1e4.
    SlidingAttention,
}

impl LayerKind {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "conv" => Ok(Self::Conv),
            "full_attention" => Ok(Self::FullAttention),
            "sliding_attention" => Ok(Self::SlidingAttention),
            other => Err(Error::Other(format!(
                "LFM3 layer type `{other}` is not implemented (conv, full_attention, sliding_attention only)"
            ))),
        }
    }

    pub fn is_conv(self) -> bool {
        matches!(self, Self::Conv)
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Conv => "conv",
            Self::FullAttention => "full_attention",
            Self::SlidingAttention => "sliding_attention",
        }
    }
}

/// One `rope_parameters.<layer_type>` entry.
#[derive(Clone, Debug, Deserialize)]
struct RopeEntry {
    rope_theta: f32,
    #[serde(default)]
    partial_rotary_factor: Option<f32>,
    #[serde(default)]
    rope_type: Option<String>,
}

/// `config.json: rope_parameters`, one independent set per attention type.
#[derive(Clone, Debug, Deserialize)]
struct RopeParameters {
    full_attention: Option<RopeEntry>,
    sliding_attention: Option<RopeEntry>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Lfm3Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub norm_eps: f32,
    #[serde(rename = "conv_L_cache")]
    pub conv_l_cache: usize,
    #[serde(default)]
    pub conv_bias: bool,
    pub sliding_window: usize,
    pub layer_types: Vec<String>,
    #[serde(default)]
    pub use_sandwich_norm: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    rope_parameters: Option<RopeParameters>,
}

/// One oriented RoPE program: precomputed cos/sin coverage for every position.
#[derive(Clone, Copy, Debug)]
pub struct RopeScheme {
    pub theta: f32,
    /// Rotated prefix width: `(partial_rotary_factor * head_dim) as usize / 2 * 2`.
    pub rotary_dim: usize,
    /// Number of angle frequencies that are actually non-zero (`rotary_dim / 2`).
    pub angles: usize,
}

impl RopeScheme {
    /// Inverse frequencies with the zero tail the reference implements: index
    /// pair `i` uses `theta^(-2i/head_dim)`; dimensions `>= rotary_dim` carry
    /// `cos=1, sin=0` and are therefore identity rotations.
    pub fn inv_freq(&self, head_dim: usize) -> Vec<f32> {
        let mut inv = Vec::with_capacity(head_dim / 2);
        for i in 0..self.angles {
            let exponent = (2 * i) as f64 / head_dim as f64;
            inv.push((1.0 / (self.theta as f64).powf(exponent)) as f32);
        }
        inv.resize(head_dim / 2, 0.0);
        inv
    }
}

impl Lfm3Config {
    pub fn from_json(json: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(json)
            .map_err(|e| Error::Other(format!("invalid LFM3 config: {e}")))?;
        config.validate()?;
        Ok(config)
    }

    /// Reject anything this port does not implement instead of guessing.
    pub fn validate(&self) -> Result<()> {
        if self.model_type != "lfm3" {
            return Err(Error::Other(format!(
                "metal_lfm3 requires model_type `lfm3`, got `{}`",
                self.model_type
            )));
        }
        let dimensions = [
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.vocab_size,
            self.max_position_embeddings,
            self.sliding_window,
            self.conv_l_cache,
        ];
        // The attention projections are square over `heads * head_dim` and the
        // residual stream is `hidden_size`, so the two must agree exactly.
        let head_width = self
            .num_attention_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| Error::Other("LFM3 head geometry overflows".into()))?;
        if dimensions.iter().any(|&n| n == 0 || n > i32::MAX as usize)
            || self.head_dim % 2 != 0
            || self.num_attention_heads % self.num_key_value_heads != 0
            || head_width != self.hidden_size
            || self.conv_l_cache < 2
            || self.layer_types.len() != self.num_hidden_layers
        {
            return Err(Error::Other(
                "invalid LFM3 dimensions, head grouping or layer plan".into(),
            ));
        }
        if self.conv_bias || !self.use_sandwich_norm || !self.tie_word_embeddings {
            return Err(Error::Other(
                "metal_lfm3 implements the bias-free conv, sandwich-norm, tied-embedding LFM3 checkpoint only"
                    .into(),
            ));
        }
        if !self.norm_eps.is_finite() || self.norm_eps <= 0.0 {
            return Err(Error::Other(
                "LFM3 norm epsilon must be positive and finite".into(),
            ));
        }
        // Validate every schedule entry and the two rotary programs now, so a
        // malformed checkpoint fails at load rather than inside a layer.
        for spec in &self.layer_types {
            LayerKind::parse(spec)?;
        }
        self.rope(LayerKind::FullAttention)?;
        self.rope(LayerKind::SlidingAttention)?;
        Ok(())
    }

    /// The pinned LFM3-1.7B geometry and layer composition of this port.
    pub fn validate_checkpoint_scope(&self) -> Result<()> {
        let conv = self
            .layer_types
            .iter()
            .filter(|spec| spec.as_str() == "conv")
            .count();
        let full = self
            .layer_types
            .iter()
            .filter(|spec| spec.as_str() == "full_attention")
            .count();
        let sliding = self
            .layer_types
            .iter()
            .filter(|spec| spec.as_str() == "sliding_attention")
            .count();
        if (
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.vocab_size,
            self.sliding_window,
            self.conv_l_cache,
        ) != (2048, 7680, 24, 16, 4, 128, 128000, 256, 7)
            || (conv, full, sliding) != (13, 6, 5)
            || self.norm_eps != 1e-5
        {
            return Err(Error::Other(
                "metal_lfm3 supports the LFM3-1.7B geometry only: hidden 2048, intermediate 7680, 24 layers (13 conv / 6 full / 5 sliding), 16/4 heads of 128, vocab 128000, window 256, conv width 7, epsilon 1e-5"
                    .into(),
            ));
        }
        Ok(())
    }

    pub fn layer_kinds(&self) -> Result<Vec<LayerKind>> {
        self.layer_types
            .iter()
            .map(|spec| LayerKind::parse(spec))
            .collect()
    }

    pub fn layer_kind(&self, layer: usize) -> Result<LayerKind> {
        let spec = self
            .layer_types
            .get(layer)
            .ok_or_else(|| Error::Other(format!("LFM3 layer {layer} is outside the plan")))?;
        LayerKind::parse(spec)
    }

    /// Rotary program for one attention type; conv layers have none.
    pub fn rope(&self, kind: LayerKind) -> Result<RopeScheme> {
        let params = self
            .rope_parameters
            .as_ref()
            .ok_or_else(|| Error::Other("LFM3 config is missing rope_parameters".into()))?;
        let (entry, label) = match kind {
            LayerKind::FullAttention => (params.full_attention.as_ref(), "full_attention"),
            LayerKind::SlidingAttention => (params.sliding_attention.as_ref(), "sliding_attention"),
            LayerKind::Conv => {
                return Err(Error::Other(
                    "LFM3 conv layers have no rotary embedding".into(),
                ))
            }
        };
        let entry = entry.ok_or_else(|| {
            Error::Other(format!("LFM3 config is missing rope_parameters.{label}"))
        })?;
        if entry
            .rope_type
            .as_deref()
            .is_some_and(|kind| kind != "default")
        {
            return Err(Error::Other(format!(
                "LFM3 rope_parameters.{label}.rope_type must be `default`, got `{:?}`",
                entry.rope_type
            )));
        }
        if !entry.rope_theta.is_finite() || entry.rope_theta <= 0.0 {
            return Err(Error::Other(format!(
                "LFM3 rope_parameters.{label}.rope_theta must be positive"
            )));
        }
        let partial = entry.partial_rotary_factor.unwrap_or(1.0);
        if !partial.is_finite() || partial <= 0.0 || partial > 1.0 {
            return Err(Error::Other(format!(
                "LFM3 rope_parameters.{label}.partial_rotary_factor must be in (0, 1]"
            )));
        }
        // The reference derives the rotated prefix in whole four-angle groups:
        // `int(partial * head_dim) // 2`. Keep that exact truncation.
        let rotary_dim = ((partial as f64 * self.head_dim as f64) as usize) / 2 * 2;
        if rotary_dim == 0 || rotary_dim > self.head_dim {
            return Err(Error::Other(format!(
                "LFM3 rope_parameters.{label} yields an empty or oversized rotary prefix"
            )));
        }
        Ok(RopeScheme {
            theta: entry.rope_theta,
            rotary_dim,
            angles: rotary_dim / 2,
        })
    }

    /// Tokens kept between steps by the short convolution (`conv_L_cache - 1`).
    pub fn conv_state_len(&self) -> usize {
        self.conv_l_cache - 1
    }

    /// Number of positions the sliding layers may attend (checkpoint window).
    pub fn window(&self) -> usize {
        self.sliding_window
    }

    /// Prepare a total position capacity for a prompt plus its output budget.
    pub fn request_capacity(&self, prompt: usize, output: usize) -> Result<usize> {
        let total = prompt
            .checked_add(output)
            .ok_or_else(|| Error::Other("LFM3 context overflow".into()))?;
        let limit = self.max_position_embeddings.min(MAX_CONTEXT);
        if prompt == 0 || total > limit {
            return Err(Error::Other(format!(
                "LFM3 requires a nonempty prompt and prompt + output <= {limit}"
            )));
        }
        let grain = total.div_ceil(CAPACITY_GRAIN) * CAPACITY_GRAIN;
        Ok(grain.min(limit))
    }

    /// End-of-sequence ids, which the checkpoint may store as a list.
    pub fn eos_ids(&self) -> Vec<u32> {
        match &self.eos_token_id {
            Some(serde_json::Value::Array(values)) => values
                .iter()
                .filter_map(|value| value.as_u64().map(|id| id as u32))
                .collect(),
            Some(serde_json::Value::Number(value)) => {
                value.as_u64().map(|id| id as u32).into_iter().collect()
            }
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint_json() -> String {
        // An inline copy of the pinned checkpoint's relevant fields: maintained
        // tests must not depend on a private capture artifact.
        serde_json::json!({
                "model_type": "lfm3",
                "hidden_size": 2048,
                "intermediate_size": 7680,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "head_dim": 128,
                "vocab_size": 128000,
                "max_position_embeddings": 262144,
                "norm_eps": 1e-5,
                "conv_L_cache": 7,
                "conv_bias": false,
                "sliding_window": 256,
                "layer_types": ["conv","conv","full_attention","conv","sliding_attention",
                                "conv","full_attention","conv","sliding_attention","conv",
                                "full_attention","conv","sliding_attention","conv",
                                "full_attention","conv","sliding_attention","conv",
                                "full_attention","conv","sliding_attention","conv",
                                "full_attention","conv"],
                "use_sandwich_norm": true,
                "tie_word_embeddings": true,
                "eos_token_id": 124900,
                "pad_token_id": 124893,
                "rope_parameters": {
                    "full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1000000.0, "rope_type": "default"},
                    "sliding_attention": {"rope_theta": 10000, "rope_type": "default"}
                }
            })
            .to_string()
    }

    #[test]
    fn parses_the_pinned_checkpoint_and_derives_both_rotary_programs() {
        let config = Lfm3Config::from_json(&checkpoint_json()).unwrap();
        config.validate_checkpoint_scope().unwrap();
        assert_eq!(config.layer_types.len(), 24);
        assert_eq!(config.conv_state_len(), 6);
        assert_eq!(config.layer_kind(0).unwrap(), LayerKind::Conv);
        assert_eq!(config.layer_kind(2).unwrap(), LayerKind::FullAttention);
        assert_eq!(config.layer_kind(4).unwrap(), LayerKind::SlidingAttention);
        let full = config.rope(LayerKind::FullAttention).unwrap();
        assert_eq!(
            (full.theta, full.rotary_dim, full.angles),
            (1_000_000.0, 64, 32)
        );
        let sliding = config.rope(LayerKind::SlidingAttention).unwrap();
        assert_eq!(
            (sliding.theta, sliding.rotary_dim, sliding.angles),
            (10_000.0, 128, 64)
        );
        // Zero tail: half the angle table is identity for the full layers.
        let inv = full.inv_freq(config.head_dim);
        assert_eq!(inv.len(), 64);
        assert!(inv[32..].iter().all(|&value| value == 0.0));
        assert!(inv[..32].iter().all(|&value| value > 0.0));
        assert_eq!(config.eos_ids(), vec![124900]);
    }

    #[test]
    fn rejects_unimplemented_semantics() {
        for (key, value) in [
            ("conv_bias", serde_json::json!(true)),
            ("use_sandwich_norm", serde_json::json!(false)),
            ("tie_word_embeddings", serde_json::json!(false)),
            ("head_dim", serde_json::json!(126)),
            ("layer_types", serde_json::json!(["conv"])),
            ("conv_L_cache", serde_json::json!(1)),
        ] {
            let mut config: serde_json::Value = serde_json::from_str(&checkpoint_json()).unwrap();
            config[key] = value;
            assert!(
                Lfm3Config::from_json(&config.to_string()).is_err(),
                "accepted {key}"
            );
        }
        let mut config: serde_json::Value = serde_json::from_str(&checkpoint_json()).unwrap();
        config["layer_types"][0] = serde_json::json!("gated_deltanet");
        assert!(Lfm3Config::from_json(&config.to_string()).is_err());
        let mut config: serde_json::Value = serde_json::from_str(&checkpoint_json()).unwrap();
        config["rope_parameters"]["full_attention"]["rope_type"] = serde_json::json!("linear");
        assert!(Lfm3Config::from_json(&config.to_string()).is_err());
    }

    #[test]
    fn checkpoint_scope_is_explicit() {
        let mut config = Lfm3Config::from_json(&checkpoint_json()).unwrap();
        config.validate_checkpoint_scope().unwrap();
        config.hidden_size = 1024;
        assert!(config.validate_checkpoint_scope().is_err());
        let mut config = Lfm3Config::from_json(&checkpoint_json()).unwrap();
        config.sliding_window = 512;
        assert!(config.validate_checkpoint_scope().is_err());
    }

    #[test]
    fn capacity_grows_in_grain_and_respects_the_admitted_envelope() {
        let config = Lfm3Config::from_json(&checkpoint_json()).unwrap();
        assert_eq!(config.request_capacity(1, 0).unwrap(), 256);
        assert_eq!(config.request_capacity(255, 1).unwrap(), 256);
        assert_eq!(config.request_capacity(256, 1).unwrap(), 512);
        assert_eq!(config.request_capacity(1024, 64).unwrap(), 1280);
        assert_eq!(
            config.request_capacity(MAX_CONTEXT - 1, 1).unwrap(),
            MAX_CONTEXT
        );
        assert!(config.request_capacity(MAX_CONTEXT, 1).is_err());
        assert!(config.request_capacity(0, 1).is_err());
        assert!(config.request_capacity(usize::MAX, 1).is_err());
    }
}
