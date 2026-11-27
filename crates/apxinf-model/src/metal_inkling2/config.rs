//! Inkling-2-Mini (`inkling2_loop`) checkpoint configuration and layer schedule.

use apxinf_core::{Error, Result};
use apxinf_mlx::QuantMode;
use serde::Deserialize;
use serde_json::Value;

/// Maximum context capacity advertised for Inkling-2-Mini on Apple Silicon.
pub const MAX_CONTEXT: usize = 1_048_576;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantSpec {
    pub mode: QuantMode,
    pub bits: i32,
    pub group_size: i32,
}

impl QuantSpec {
    pub const NVFP4: Self = Self {
        mode: QuantMode::Nvfp4,
        bits: 4,
        group_size: 16,
    };
    pub const MXFP8: Self = Self {
        mode: QuantMode::Mxfp8,
        bits: 8,
        group_size: 32,
    };
    pub const AFFINE_Q8_G64: Self = Self {
        mode: QuantMode::Affine,
        bits: 8,
        group_size: 64,
    };
    pub const AFFINE_Q4_G64: Self = Self {
        mode: QuantMode::Affine,
        bits: 4,
        group_size: 64,
    };

    pub fn from_json(v: &Value, default: Option<Self>) -> Result<Self> {
        let obj = v
            .as_object()
            .ok_or(Error::Contract("quantization spec must be a JSON object"))?;
        let mode_str = obj
            .get("mode")
            .and_then(Value::as_str)
            .or_else(|| default.map(|d| match d.mode {
                QuantMode::Affine => "affine",
                QuantMode::Nvfp4 => "nvfp4",
                QuantMode::Mxfp8 => "mxfp8",
            }))
            .unwrap_or("affine");
        let mode = match mode_str {
            "nvfp4" => return Ok(Self::NVFP4),
            "mxfp8" => return Ok(Self::MXFP8),
            "affine" => QuantMode::Affine,
            _ => return Err(Error::Contract("unsupported Inkling2 quantization mode")),
        };
        let bits = obj
            .get("bits")
            .and_then(Value::as_i64)
            .map(|x| x as i32)
            .or_else(|| default.map(|d| d.bits))
            .unwrap_or(4);
        let group_size = obj
            .get("group_size")
            .and_then(Value::as_i64)
            .map(|x| x as i32)
            .or_else(|| default.map(|d| d.group_size))
            .unwrap_or(64);
        Ok(Self {
            mode,
            bits,
            group_size,
        })
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct LoopConfig {
    pub core_start: usize,
    pub core_end: usize,
    pub num_passes: usize,
    #[serde(default)]
    pub gate_rank: Option<usize>,
    #[serde(default)]
    pub is_varloop: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct TextConfig {
    pub model_type: String,
    #[serde(default = "default_max_length")]
    pub model_max_length: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    #[serde(default)]
    pub unpadded_vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    #[serde(default)]
    pub swa_num_attention_heads: Option<usize>,
    #[serde(default)]
    pub swa_num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub swa_head_dim: Option<usize>,
    #[serde(default = "default_sliding_window")]
    pub sliding_window_size: usize,
    #[serde(default = "default_d_rel")]
    pub d_rel: usize,
    #[serde(default = "default_rel_extent")]
    pub rel_extent: usize,
    #[serde(default = "default_log_floor")]
    pub log_scaling_n_floor: usize,
    #[serde(default = "default_log_alpha")]
    pub log_scaling_alpha: f32,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default)]
    pub use_value_norm_no_gamma: bool,
    #[serde(default = "default_qk_clip")]
    pub qk_l2_clip_threshold: f32,
    #[serde(default = "default_rel_clip")]
    pub rel_l2_clip_threshold: f32,
    #[serde(default)]
    pub local_layer_ids: Option<Vec<usize>>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default = "default_sconv_kernel")]
    pub sconv_kernel_size: usize,
    #[serde(default = "default_attn_res_block")]
    pub n_layers_per_attn_res_block: usize,
    #[serde(default)]
    pub attn_res_normalize_values: bool,
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub n_shared_experts: usize,
    pub intermediate_size: usize,
    #[serde(default = "default_gate_softcap")]
    pub mlp_gate_softcap: f32,
    #[serde(default = "default_up_softcap")]
    pub mlp_up_softcap: f32,
    #[serde(default = "default_route_scale")]
    pub route_scale: f32,
    #[serde(default)]
    pub use_gate_bias: bool,
    #[serde(default)]
    pub final_logit_softcapping: Option<f32>,
    #[serde(default)]
    pub loop_config: Option<LoopConfig>,
}

fn default_max_length() -> usize {
    MAX_CONTEXT
}
fn default_sliding_window() -> usize {
    1024
}
fn default_d_rel() -> usize {
    16
}
fn default_rel_extent() -> usize {
    1024
}
fn default_log_floor() -> usize {
    8192
}
fn default_log_alpha() -> f32 {
    0.1
}
fn default_rms_eps() -> f32 {
    1e-6
}
fn default_qk_clip() -> f32 {
    22.627_417
}
fn default_rel_clip() -> f32 {
    8.0
}
fn default_sconv_kernel() -> usize {
    4
}
fn default_attn_res_block() -> usize {
    6
}
fn default_gate_softcap() -> f32 {
    4.0
}
fn default_up_softcap() -> f32 {
    25.0
}
fn default_route_scale() -> f32 {
    8.0
}
fn default_rope_theta() -> f32 {
    1_000_000.0
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct MtpConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default)]
    pub local_layer_ids: Vec<usize>,
    #[serde(default = "default_sliding_window")]
    pub sliding_window_size: usize,
    #[serde(default)]
    pub block_size: Option<usize>,
    #[serde(default)]
    pub markov_rank: Option<usize>,
    #[serde(default)]
    pub markov_head_type: Option<String>,
    #[serde(default)]
    pub q_bias: bool,
    #[serde(default)]
    pub o_bias: bool,
    #[serde(default)]
    pub target_layer_ids: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Config {
    pub model_type: String,
    #[serde(default = "default_eos")]
    pub eos_token_id: u32,
    pub text_config: TextConfig,
    #[serde(default)]
    pub mtp_config: Option<MtpConfig>,
    /// Either ModelOpt `{quant_method, quant_algo}` or MLX per-layer map.
    #[serde(default)]
    pub quantization_config: Option<Value>,
    /// MLX-format per-layer quantization map (present on `oQ4e-mtp` checkpoints).
    #[serde(default)]
    pub quantization: Option<Value>,
}

fn default_eos() -> u32 {
    200_006
}

impl Config {
    pub fn from_json(raw: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(raw)
            .map_err(|e| Error::Other(format!("invalid Inkling2 config.json: {e}")))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != "inkling2_loop" && self.model_type != "inkling2" {
            return Err(Error::Contract(
                "Inkling2 requires model_type 'inkling2_loop' or 'inkling2'",
            ));
        }
        let t = &self.text_config;
        if t.hidden_size == 0 || t.num_hidden_layers == 0 || t.vocab_size == 0 {
            return Err(Error::Contract(
                "Inkling2 text_config dimensions must be positive",
            ));
        }
        if t.num_attention_heads == 0
            || t.num_key_value_heads == 0
            || t.num_attention_heads % t.num_key_value_heads != 0
        {
            return Err(Error::Contract(
                "Inkling2 attention heads must be divisible by key/value heads",
            ));
        }
        if let Some(loop_cfg) = &t.loop_config {
            let block = t.n_layers_per_attn_res_block;
            if block == 0
                || loop_cfg.core_start == 0
                || loop_cfg.core_start >= loop_cfg.core_end
                || loop_cfg.core_end > t.num_hidden_layers
                || loop_cfg.core_start % block != 0
                || loop_cfg.core_end % block != 0
                || loop_cfg.num_passes < 2
            {
                return Err(Error::Contract(
                    "Inkling2 loop_config must align on attention-residual block boundaries",
                ));
            }
        }
        Ok(())
    }

    /// Returns the MLX per-layer quantization table if this is a converted `oQ` checkpoint.
    pub fn mlx_quantization_map(&self) -> Option<&serde_json::Map<String, Value>> {
        if let Some(obj) = self.quantization.as_ref().and_then(Value::as_object) {
            return Some(obj);
        }
        let obj = self.quantization_config.as_ref().and_then(Value::as_object)?;
        if obj.get("quant_method").and_then(Value::as_str) == Some("modelopt") {
            return None;
        }
        if obj.contains_key("bits") || obj.contains_key("group_size") {
            Some(obj)
        } else {
            None
        }
    }

    /// True if this checkpoint uses MLX per-layer affine quantization (e.g. `oQ4e-mtp`).
    pub fn is_mlx_quantized(&self) -> bool {
        self.mlx_quantization_map().is_some()
    }

    /// Lookup quantization spec for a sanitized module path (e.g. `"lm_head"`,
    /// `"model.layers.0.attn.in_proj"`, `"model.layers.20.mlp.experts.w13"`).
    pub fn quant_spec_for(&self, path: &str) -> Result<Option<QuantSpec>> {
        let Some(map) = self.mlx_quantization_map() else {
            return Ok(None);
        };
        let default = QuantSpec::from_json(&Value::Object(map.clone()), None)?;
        if let Some(entry) = map.get(path) {
            if entry.is_object() {
                return Ok(Some(QuantSpec::from_json(entry, Some(default))?));
            }
            if entry.as_bool() == Some(false) {
                return Ok(None);
            }
        }
        Ok(Some(default))
    }

    /// Returns `(layer_idx, pass_idx)` in execution order, expanding looped middle layers.
    pub fn layer_schedule(&self) -> Vec<(usize, usize)> {
        let n = self.text_config.num_hidden_layers;
        let Some(loop_cfg) = &self.text_config.loop_config else {
            return (0..n).map(|i| (i, 0)).collect();
        };
        let mut schedule = Vec::with_capacity(
            loop_cfg.core_start
                + loop_cfg.num_passes * (loop_cfg.core_end - loop_cfg.core_start)
                + (n - loop_cfg.core_end),
        );
        for i in 0..loop_cfg.core_start {
            schedule.push((i, 0));
        }
        for pass in 0..loop_cfg.num_passes {
            for i in loop_cfg.core_start..loop_cfg.core_end {
                schedule.push((i, pass));
            }
        }
        for i in loop_cfg.core_end..n {
            schedule.push((i, 0));
        }
        schedule
    }

    pub fn is_sliding(&self, layer_idx: usize) -> bool {
        let t = &self.text_config;
        if let Some(types) = &t.layer_types {
            return types.get(layer_idx).is_some_and(|s| s == "hybrid_sliding");
        }
        if let Some(local) = &t.local_layer_ids {
            return local.contains(&layer_idx);
        }
        (layer_idx + 1) % 6 != 0
    }

    pub fn unpadded_vocab_size(&self) -> usize {
        if self.text_config.unpadded_vocab_size > 0 {
            self.text_config.unpadded_vocab_size
        } else {
            self.text_config.vocab_size
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_CONFIG: &str = r#"{
        "model_type": "inkling2_loop",
        "eos_token_id": 200006,
        "text_config": {
            "model_type": "inkling2_loop_text",
            "model_max_length": 1048576,
            "hidden_size": 2304,
            "num_hidden_layers": 24,
            "vocab_size": 201024,
            "unpadded_vocab_size": 200058,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "local_layer_ids": [0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22],
            "n_layers_per_attn_res_block": 6,
            "n_routed_experts": 128,
            "num_experts_per_tok": 8,
            "intermediate_size": 1152,
            "loop_config": {
                "core_start": 6,
                "core_end": 18,
                "num_passes": 3,
                "gate_rank": 64,
                "is_varloop": false
            }
        }
    }"#;

    #[test]
    fn parses_inkling2_loop_and_expands_schedule() {
        let cfg = Config::from_json(SAMPLE_CONFIG).unwrap();
        let sched = cfg.layer_schedule();
        // 6 prefix + 3 * 12 looped + 6 suffix = 48 layer executions
        assert_eq!(sched.len(), 48);
        assert_eq!(sched[0], (0, 0));
        assert_eq!(sched[6], (6, 0));
        assert_eq!(sched[18], (6, 1));
        assert_eq!(sched[30], (6, 2));
        assert_eq!(sched[42], (18, 0));
        assert!(cfg.is_sliding(0));
        assert!(!cfg.is_sliding(5));
        assert!(!cfg.is_sliding(23));
    }
}
