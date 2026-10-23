//! π0.5 architecture and checkpoint config, as this HIP port reads it.
//!
//! A copy owned by `hip_pi05`, not shared with the CUDA implementation in
//! `pi05/`: it keeps only what a BF16 inference on HIP needs. The JSON
//! aliases and defaults are the checkpoint contract and match `pi05/config.rs`
//! by intent; agreement is checked by the shared accuracy fixtures, not by
//! sharing code.

use std::path::Path;

use apxinf_core::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmaConfig {
    pub width: usize,
    pub depth: usize,
    pub mlp_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl GemmaConfig {
    pub const GEMMA_2B: Self = Self {
        width: 2048,
        depth: 18,
        mlp_dim: 16_384,
        num_heads: 8,
        num_kv_heads: 1,
        head_dim: 256,
    };

    pub const GEMMA_300M: Self = Self {
        width: 1024,
        depth: 18,
        mlp_dim: 4096,
        num_heads: 8,
        num_kv_heads: 1,
        head_dim: 256,
    };

    /// Width of the packed `[q | k | v]` projection.
    pub fn qkv_width(&self) -> usize {
        (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub action_dim: usize,
    pub action_horizon: usize,
    pub max_token_len: usize,
    pub num_flow_steps: usize,
    pub flow_start_time: f32,
    pub num_views: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub vision_width: usize,
    pub vision_depth: usize,
    pub vision_mlp_dim: usize,
    pub vision_heads: usize,
    pub vision_head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub layer_norm_eps: f32,
    pub rope_theta: f32,
    pub time_min_period: f32,
    pub time_max_period: f32,
    pub language: GemmaConfig,
    pub action_expert: GemmaConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            action_dim: 32,
            action_horizon: 50,
            max_token_len: 200,
            num_flow_steps: 10,
            flow_start_time: 1.0,
            num_views: 3,
            image_size: 224,
            patch_size: 14,
            vision_width: 1152,
            vision_depth: 27,
            vision_mlp_dim: 4304,
            vision_heads: 16,
            vision_head_dim: 72,
            vocab_size: 257_152,
            rms_norm_eps: 1e-6,
            layer_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            time_min_period: 4e-3,
            time_max_period: 4.0,
            language: GemmaConfig::GEMMA_2B,
            action_expert: GemmaConfig::GEMMA_300M,
        }
    }
}

impl Config {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| Error::Other(format!("read {}: {e}", path.display())))?;
        Self::from_json_str(&raw)
    }

    /// Parse LeRobot's policy config, accepting the OpenPI aliases. Fields an
    /// older checkpoint omits keep the validated π0.5 values.
    pub fn from_json_str(raw: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(raw)
            .map_err(|e| Error::Other(format!("pi05 config json: {e}")))?;
        let mut cfg = Self::default();
        cfg.action_dim = usize_field(&v, &["max_action_dim", "action_dim"], cfg.action_dim);
        cfg.action_horizon = usize_field(&v, &["chunk_size", "action_horizon"], cfg.action_horizon);
        cfg.max_token_len = usize_field(
            &v,
            &["tokenizer_max_length", "max_token_len"],
            cfg.max_token_len,
        );
        cfg.num_flow_steps = usize_field(
            &v,
            &["num_inference_steps", "num_flow_steps"],
            cfg.num_flow_steps,
        );
        cfg.flow_start_time = f32_field(&v, "flow_start_time", cfg.flow_start_time);
        cfg.num_views = resolve_num_views(&v, cfg.num_views);
        cfg.time_min_period = f32_field(&v, "min_period", cfg.time_min_period);
        cfg.time_max_period = f32_field(&v, "max_period", cfg.time_max_period);
        if let Some(size) = v
            .get("image_resolution")
            .and_then(|x| x.as_array())
            .and_then(|r| r.first())
            .and_then(|x| x.as_u64())
        {
            cfg.image_size = size as usize;
        }
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn patches_per_view(&self) -> usize {
        let side = self.image_size / self.patch_size;
        side * side
    }

    /// Patch rows across all views.
    pub fn patch_rows(&self) -> usize {
        self.num_views * self.patches_per_view()
    }

    /// Flattened patch width, `3 * patch_size²`.
    pub fn patch_width(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }

    pub fn validate(&self) -> Result<()> {
        if self.patch_size == 0 || !self.image_size.is_multiple_of(self.patch_size) {
            return Err(Error::Other(format!(
                "pi05 image size {} is not divisible by patch size {}",
                self.image_size, self.patch_size
            )));
        }
        if self.action_horizon == 0 || self.action_horizon > 50 {
            return Err(Error::Other(format!(
                "pi05 action_horizon must be in 1..=50, got {}",
                self.action_horizon
            )));
        }
        if self.num_flow_steps == 0 {
            return Err(Error::Other("pi05 num_flow_steps must be non-zero".into()));
        }
        if !(self.flow_start_time.is_finite()
            && self.flow_start_time > 0.0
            && self.flow_start_time <= 1.0)
        {
            return Err(Error::Other(format!(
                "pi05 flow_start_time must be in (0, 1], got {}",
                self.flow_start_time
            )));
        }
        if self.language.depth != self.action_expert.depth
            || self.language.num_heads != self.action_expert.num_heads
            || self.language.num_kv_heads != self.action_expert.num_kv_heads
            || self.language.head_dim != self.action_expert.head_dim
        {
            return Err(Error::Other(
                "pi05 language and action experts must share depth/head geometry".into(),
            ));
        }
        Ok(())
    }
}

fn usize_field(v: &serde_json::Value, names: &[&str], default: usize) -> usize {
    names
        .iter()
        .find_map(|name| v.get(*name).and_then(|x| x.as_u64()))
        .map(|x| x as usize)
        .unwrap_or(default)
}

fn f32_field(v: &serde_json::Value, name: &str, default: f32) -> f32 {
    v.get(name)
        .and_then(|x| x.as_f64())
        .map(|x| x as f32)
        .unwrap_or(default)
}

/// An explicit `num_views` wins; otherwise count the `VISUAL` entries in
/// LeRobot's `input_features`. `empty_cameras` is ignored: absent cameras are
/// never sent, which is equivalent to OpenPI's masked trailing padding.
fn resolve_num_views(v: &serde_json::Value, default: usize) -> usize {
    if let Some(explicit) = v.get("num_views").and_then(|x| x.as_u64()) {
        return explicit as usize;
    }
    let visual = v
        .get("input_features")
        .and_then(|x| x.as_object())
        .map(|features| {
            features
                .values()
                .filter(|f| f.get("type").and_then(|t| t.as_str()) == Some("VISUAL"))
                .count()
        })
        .unwrap_or(0);
    if visual > 0 {
        visual
    } else {
        default
    }
}

/// OpenPI's float64-generated sinusoidal flow-timestep embedding: sines then
/// cosines over periods geometric from `min_period` to `max_period`.
pub fn sinusoidal_time_embedding(
    time: f32,
    dimension: usize,
    min_period: f32,
    max_period: f32,
) -> Vec<f32> {
    let half = dimension / 2;
    let mut sin = Vec::with_capacity(half);
    let mut cos = Vec::with_capacity(half);
    for i in 0..half {
        let fraction = if half == 1 {
            0.0
        } else {
            i as f64 / (half - 1) as f64
        };
        let period = min_period as f64 * (max_period as f64 / min_period as f64).powf(fraction);
        let phase = time as f64 * std::f64::consts::TAU / period;
        sin.push(phase.sin() as f32);
        cos.push(phase.cos() as f32);
    }
    sin.extend(cos);
    sin
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_lerobot_aliases() {
        let cfg = Config::from_json_str(
            r#"{"max_action_dim": 32, "chunk_size": 10, "tokenizer_max_length": 200,
                "num_inference_steps": 10, "flow_start_time": 0.5, "num_views": 2,
                "image_resolution": [224, 224]}"#,
        )
        .unwrap();
        assert_eq!(
            cfg,
            Config {
                action_horizon: 10,
                num_views: 2,
                flow_start_time: 0.5,
                ..Config::default()
            }
        );
        assert_eq!(cfg.patch_rows(), 512);
    }

    #[test]
    fn counts_real_cameras_and_ignores_empty_ones() {
        let cfg = Config::from_json_str(
            r#"{"input_features": {
                "observation.images.image": {"type": "VISUAL"},
                "observation.images.image2": {"type": "VISUAL"},
                "observation.state": {"type": "STATE"}},
              "empty_cameras": 1}"#,
        )
        .unwrap();
        assert_eq!(cfg.num_views, 2);
    }

    #[test]
    fn rejects_invalid_horizon() {
        assert!(Config::from_json_str(r#"{"chunk_size": 0}"#)
            .unwrap_err()
            .to_string()
            .contains("action_horizon"));
    }

    #[test]
    fn time_embedding_matches_its_endpoints() {
        let e = sinusoidal_time_embedding(0.0, 8, 4e-3, 4.0);
        assert_eq!(&e[..4], &[0.0; 4]);
        assert_eq!(&e[4..], &[1.0; 4]);
    }
}
