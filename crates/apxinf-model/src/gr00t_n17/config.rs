use std::path::Path;

use apxinf_core::{Error, Result};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct GrootN17DiffusionConfig {
    pub attention_head_dim: usize,
    pub num_attention_heads: usize,
    pub num_layers: usize,
    pub output_dim: usize,
    pub interleave_self_attention: bool,
    pub norm_type: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct GrootN17VlSelfAttentionConfig {
    pub attention_head_dim: usize,
    pub num_attention_heads: usize,
    pub num_layers: usize,
}

/// Frozen architecture fields used by the native GR00T N1.7 executor.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct GrootN17Config {
    pub model_type: String,
    pub model_name: String,
    pub dtype: String,
    pub action_horizon: usize,
    pub max_action_dim: usize,
    pub max_state_dim: usize,
    pub max_num_embodiments: usize,
    pub max_seq_len: usize,
    pub hidden_size: usize,
    pub backbone_embedding_dim: usize,
    pub select_layer: usize,
    pub num_inference_timesteps: usize,
    pub num_timestep_buckets: usize,
    pub add_pos_embed: bool,
    pub use_alternate_vl_dit: bool,
    pub use_vlln: bool,
    pub diffusion_model_cfg: GrootN17DiffusionConfig,
    pub vl_self_attention_cfg: GrootN17VlSelfAttentionConfig,
}

impl GrootN17Config {
    pub const LIBERO_EMBODIMENT_ID: usize = 2;
    pub const LIBERO_ACTION_HORIZON: usize = 16;
    pub const LIBERO_ACTION_DIM: usize = 7;

    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
        let config: Self = serde_json::from_str(&raw)
            .map_err(|error| Error::Other(format!("parse {}: {error}", path.display())))?;
        config.validate()?;
        Ok(config)
    }

    pub fn action_width(&self) -> usize {
        self.diffusion_model_cfg.num_attention_heads
            * self.diffusion_model_cfg.attention_head_dim
    }

    pub fn vl_attention_width(&self) -> usize {
        self.vl_self_attention_cfg.num_attention_heads
            * self.vl_self_attention_cfg.attention_head_dim
    }

    pub fn timestep_values(&self) -> Vec<u32> {
        (0..self.num_inference_timesteps)
            .map(|step| step * self.num_timestep_buckets / self.num_inference_timesteps)
            .map(|value| value as u32)
            .collect()
    }

    pub fn validate(&self) -> Result<()> {
        let expected = [
            (self.model_type == "Gr00tN1d7", "model_type must be Gr00tN1d7"),
            (self.dtype == "bfloat16", "only the BF16 checkpoint is supported"),
            (self.action_horizon == 40, "action_horizon must be 40"),
            (self.max_action_dim == 132, "max_action_dim must be 132"),
            (self.max_state_dim == 132, "max_state_dim must be 132"),
            (self.select_layer == 16, "select_layer must be 16"),
            (self.num_inference_timesteps == 4, "denoising steps must be 4"),
            (self.num_timestep_buckets == 1000, "timestep buckets must be 1000"),
            (self.add_pos_embed, "action position embedding is required"),
            (self.use_alternate_vl_dit, "alternate VL DiT is required"),
            (self.use_vlln, "VL LayerNorm is required"),
            (self.action_width() == 1536, "DiT width must be 1536"),
            (self.diffusion_model_cfg.num_layers == 32, "DiT depth must be 32"),
            (self.diffusion_model_cfg.output_dim == 1024, "DiT output must be 1024"),
            (self.diffusion_model_cfg.interleave_self_attention, "DiT must interleave self attention"),
            (self.diffusion_model_cfg.norm_type == "ada_norm", "DiT must use ada_norm"),
            (self.vl_attention_width() == 2048, "VL attention width must be 2048"),
            (self.vl_self_attention_cfg.num_layers == 4, "VL attention depth must be 4"),
        ];
        if let Some((_, message)) = expected.into_iter().find(|(ok, _)| !ok) {
            return Err(Error::Other(format!("unsupported GR00T N1.7 config: {message}")));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::GrootN17Config;

    #[test]
    fn parses_frozen_libero_checkpoint() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../experiment/gr00t-n1d7-workflow-validation/checkpoint_metadata/config.json");
        if !path.is_file() {
            return;
        }
        let config = GrootN17Config::from_json_file(&path).unwrap();
        assert_eq!(config.action_width(), 1536);
        assert_eq!(config.vl_attention_width(), 2048);
        assert_eq!(config.timestep_values(), [0, 250, 500, 750]);
    }
}
