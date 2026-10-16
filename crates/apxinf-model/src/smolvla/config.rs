use std::path::Path;

use apxinf_core::{Error, Result};

#[derive(Clone, Debug)]
pub struct SmolVlaConfig {
    pub num_views: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub vision_depth: usize,
    pub vision_width: usize,
    pub vision_heads: usize,
    pub vision_mlp_dim: usize,
    pub language_depth: usize,
    pub language_width: usize,
    pub language_heads: usize,
    pub language_kv_heads: usize,
    pub language_mlp_dim: usize,
    pub expert_width: usize,
    pub expert_mlp_dim: usize,
    pub max_token_len: usize,
    pub max_state_dim: usize,
    pub max_action_dim: usize,
    pub action_dim: usize,
    pub action_horizon: usize,
    pub num_flow_steps: usize,
    pub rms_norm_eps: f32,
    pub layer_norm_eps: f32,
    pub rope_theta: f32,
    pub time_min_period: f32,
    pub time_max_period: f32,
}

impl Default for SmolVlaConfig {
    fn default() -> Self {
        Self {
            num_views: 2,
            image_size: 512,
            patch_size: 16,
            vision_depth: 12,
            vision_width: 768,
            vision_heads: 12,
            vision_mlp_dim: 3072,
            language_depth: 16,
            language_width: 960,
            language_heads: 15,
            language_kv_heads: 5,
            language_mlp_dim: 2560,
            expert_width: 720,
            expert_mlp_dim: 2048,
            max_token_len: 48,
            max_state_dim: 32,
            max_action_dim: 32,
            action_dim: 7,
            action_horizon: 50,
            num_flow_steps: 10,
            rms_norm_eps: 1e-5,
            layer_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            time_min_period: 0.004,
            time_max_period: 4.0,
        }
    }
}

impl SmolVlaConfig {
    pub fn patches_per_view(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }

    pub fn pixel_tokens_per_view(&self) -> usize {
        self.patches_per_view() / 16
    }

    pub fn validate(&self) -> Result<()> {
        if self.num_views == 0
            || self.image_size % self.patch_size != 0
            || self.vision_width % self.vision_heads != 0
            || self.language_width % self.language_heads != 0
            || self.language_heads % self.language_kv_heads != 0
            || self.expert_width == 0
            || self.expert_mlp_dim == 0
            || self.action_dim > self.max_action_dim
            || self.num_flow_steps == 0
        {
            return Err(Error::Other("invalid SmolVLA configuration".into()));
        }
        Ok(())
    }

    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
        let value: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|error| Error::Other(format!("parse {}: {error}", path.display())))?;
        let mut config = Self::default();
        config.num_views = usize_field(&value, "n_camera", config.num_views);
        if let Some(inputs) = value.get("input_features").and_then(|x| x.as_object()) {
            let visual_inputs = inputs
                .values()
                .filter_map(|feature| feature.get("type"))
                .filter_map(|kind| kind.as_str())
                .filter(|kind| *kind == "VISUAL")
                .count();
            if visual_inputs != 0 {
                config.num_views = visual_inputs;
            }
        }
        config.max_token_len = usize_field(&value, "tokenizer_max_length", config.max_token_len);
        config.max_state_dim = usize_field(&value, "max_state_dim", config.max_state_dim);
        config.max_action_dim = usize_field(&value, "max_action_dim", config.max_action_dim);
        config.action_horizon = usize_field(&value, "chunk_size", config.action_horizon);
        config.num_flow_steps = usize_field(&value, "num_steps", config.num_flow_steps);
        if let Some(num_vlm_layers) = value.get("num_vlm_layers").and_then(|field| field.as_u64()) {
            if num_vlm_layers == 0 {
                config.language_depth = 32;
            } else {
                config.language_depth = num_vlm_layers as usize;
            }
        }
        let num_expert_layers = value
            .get("num_expert_layers")
            .and_then(|field| field.as_i64())
            .unwrap_or(-1);
        if num_expert_layers > 0 && num_expert_layers != config.language_depth as i64 {
            return Err(Error::Other(
                "SmolVLA currently requires num_expert_layers to match num_vlm_layers".into(),
            ));
        }
        config.time_min_period = f32_field(&value, "min_period", config.time_min_period);
        config.time_max_period = f32_field(&value, "max_period", config.time_max_period);
        let expert_width_multiplier =
            f32_field(&value, "expert_width_multiplier", 1.0);
        if !expert_width_multiplier.is_finite() || expert_width_multiplier <= 0.0 {
            return Err(Error::Other(
                "SmolVLA expert_width_multiplier must be positive and finite".into(),
            ));
        }
        if value.get("expert_width_multiplier").is_some() {
            config.expert_width = (config.language_width as f32
                * expert_width_multiplier)
                .round() as usize;
            config.expert_mlp_dim = (config.language_mlp_dim as f32
                * expert_width_multiplier)
                .round() as usize;
        }
        if let Some(outputs) = value.pointer("/output_features/action/shape").and_then(|x| x.as_array()) {
            if let Some(dim) = outputs.first().and_then(|x| x.as_u64()) {
                config.action_dim = dim as usize;
            }
        }
        if let Some(resize) = value.get("resize_imgs_with_padding").and_then(|x| x.as_array()) {
            if let Some(size) = resize.first().and_then(|x| x.as_u64()) {
                config.image_size = size as usize;
            }
        }
        if let Some(cameras) = value.get("n_camera").and_then(|x| x.as_u64()) {
            config.num_views = cameras as usize;
        }
        config.validate()?;
        Ok(config)
    }
}

fn usize_field(value: &serde_json::Value, key: &str, fallback: usize) -> usize {
    value.get(key).and_then(|x| x.as_u64()).map_or(fallback, |x| x as usize)
}

fn f32_field(value: &serde_json::Value, key: &str, fallback: f32) -> f32 {
    value.get(key).and_then(|x| x.as_f64()).map_or(fallback, |x| x as f32)
}
