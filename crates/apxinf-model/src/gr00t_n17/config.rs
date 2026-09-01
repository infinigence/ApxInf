use std::path::Path;

use apxinf_core::{Error, Result};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct DiffusionConfig {
    pub attention_head_dim: usize,
    pub num_attention_heads: usize,
    pub num_layers: usize,
    pub output_dim: usize,
    pub norm_type: String,
    pub interleave_self_attention: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct VlSelfAttentionConfig {
    pub attention_head_dim: usize,
    pub num_attention_heads: usize,
    pub num_layers: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Gr00tN17Config {
    pub model_type: String,
    pub model_name: String,
    pub action_horizon: usize,
    pub max_action_dim: usize,
    pub max_state_dim: usize,
    pub max_num_embodiments: usize,
    #[serde(default = "default_state_history_length")]
    pub state_history_length: usize,
    pub hidden_size: usize,
    #[serde(default = "default_input_embedding_dim")]
    pub input_embedding_dim: usize,
    pub backbone_embedding_dim: usize,
    pub num_inference_timesteps: usize,
    pub num_timestep_buckets: usize,
    pub max_seq_len: usize,
    pub select_layer: usize,
    pub add_pos_embed: bool,
    pub use_alternate_vl_dit: bool,
    pub use_vlln: bool,
    pub diffusion_model_cfg: DiffusionConfig,
    pub vl_self_attention_cfg: VlSelfAttentionConfig,
}

const fn default_state_history_length() -> usize {
    1
}

const fn default_input_embedding_dim() -> usize {
    1536
}

impl Gr00tN17Config {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
        Self::from_json_str(&raw)
    }

    pub fn from_json_str(raw: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(raw)
            .map_err(|error| Error::Other(format!("GR00T N1.7 config: {error}")))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != "Gr00tN1d7" {
            return Err(Error::Other(format!(
                "expected model_type Gr00tN1d7, got {}",
                self.model_type
            )));
        }
        if self.model_name != "nvidia/Cosmos-Reason2-2B" {
            return Err(Error::Other(format!(
                "GR00T N1.7 requires Cosmos-Reason2-2B, got {}",
                self.model_name
            )));
        }
        if self.action_horizon == 0
            || self.max_action_dim == 0
            || self.max_state_dim == 0
            || self.state_history_length == 0
            || self.num_inference_timesteps == 0
        {
            return Err(Error::Other(
                "GR00T N1.7 dimensions and denoising steps must be non-zero".into(),
            ));
        }
        if self.input_embedding_dim != 1536
            || self.backbone_embedding_dim != 2048
            || self.hidden_size != 1024
        {
            return Err(Error::Other(format!(
                "unsupported GR00T N1.7 hidden contract: action={}, backbone={}, output={}",
                self.input_embedding_dim, self.backbone_embedding_dim, self.hidden_size
            )));
        }
        if !self.use_alternate_vl_dit || !self.use_vlln {
            return Err(Error::Other(
                "ApxInf GR00T N1.7 supports the released AlternateVLDiT + VLLN checkpoint".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RELEASED: &str = r#"{
      "model_type":"Gr00tN1d7", "model_name":"nvidia/Cosmos-Reason2-2B",
      "action_horizon":40, "max_action_dim":132, "max_state_dim":132,
      "max_num_embodiments":32, "state_history_length":1,
      "hidden_size":1024, "input_embedding_dim":1536,
      "backbone_embedding_dim":2048, "num_inference_timesteps":4,
      "num_timestep_buckets":1000, "max_seq_len":1024, "select_layer":16,
      "add_pos_embed":true, "use_alternate_vl_dit":true, "use_vlln":true,
      "diffusion_model_cfg":{"attention_head_dim":48,"num_attention_heads":32,
        "num_layers":32,"output_dim":1024,"norm_type":"ada_norm",
        "interleave_self_attention":true},
      "vl_self_attention_cfg":{"attention_head_dim":64,"num_attention_heads":32,
        "num_layers":4}
    }"#;

    #[test]
    fn parses_released_libero_contract() {
        let config = Gr00tN17Config::from_json_str(RELEASED).unwrap();
        assert_eq!(config.action_horizon, 40);
        assert_eq!(config.diffusion_model_cfg.num_layers, 32);
        assert_eq!(config.vl_self_attention_cfg.num_layers, 4);
        assert_eq!(config.select_layer, 16);
    }
}
