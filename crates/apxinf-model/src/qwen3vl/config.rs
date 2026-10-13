//! Qwen3-VL config parsed from HF `config.json`.
//!
//! We do NOT reuse `apxinf_loader::ModelConfig` because Qwen3-VL's schema is
//! structurally different (nested `text_config` / `vision_config`, mRoPE
//! sections, tied embeddings). Parsing here keeps the loader crate model-
//! agnostic and matches the plan's "each model owns its config" rule.

use std::path::Path;

use apxinf_core::{Error, Result};

/// Text-stack config for Qwen3-VL. Vision fields live under
/// `Qwen3VLVisionConfig` (added when the vision tower lands in Phase 4).
#[derive(Clone, Debug)]
pub struct Qwen3VLTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// `[T, H, W]` split of the 64 (= head_dim/2) frequency pairs.
    pub mrope_section: [usize; 3],
    pub mrope_interleaved: bool,
    pub tie_word_embeddings: bool,
}

/// Vision-tower config for Qwen3-VL.
#[derive(Clone, Debug)]
pub struct Qwen3VLVisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub in_channels: usize,
    pub spatial_merge_size: usize,
    pub num_position_embeddings: usize,
    pub out_hidden_size: usize,
    /// Which 3 of the `depth` vision blocks feed the deepstack mergers.
    pub deepstack_visual_indexes: Vec<usize>,
}

impl Qwen3VLVisionConfig {
    pub fn head_dim(&self) -> usize {
        // HF: head_dim = hidden_size // num_heads (= 64 for Qwen3-VL-2B).
        if self.head_dim != 0 {
            self.head_dim
        } else {
            self.hidden_size / self.num_heads
        }
    }
}

/// Full Qwen3-VL config. `vision` is `None` until Phase 4.
#[derive(Clone, Debug)]
pub struct Qwen3VLConfig {
    pub text: Qwen3VLTextConfig,
    pub vision: Qwen3VLVisionConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
}

impl Qwen3VLConfig {
    /// Parse from a HuggingFace `config.json` file.
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| Error::Other(format!("read {}: {e}", path.display())))?;
        Self::from_json_str(&raw)
    }

    pub fn from_json_str(s: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s)
            .map_err(|e| Error::Other(format!("qwen3vl config json: {e}")))?;
        let tc = &v["text_config"];
        if !tc.is_object() {
            return Err(Error::Other("qwen3vl config: missing text_config".into()));
        }

        let rope_scaling = &tc["rope_scaling"];
        let section = rope_scaling["mrope_section"].as_array().ok_or_else(|| {
            Error::Other("qwen3vl config: missing rope_scaling.mrope_section".into())
        })?;
        if section.len() != 3 {
            return Err(Error::Other(format!(
                "qwen3vl config: mrope_section must be [T,H,W] (3 entries), got {}",
                section.len()
            )));
        }
        let mrope_section = [
            section[0].as_u64().unwrap_or(24) as usize,
            section[1].as_u64().unwrap_or(20) as usize,
            section[2].as_u64().unwrap_or(20) as usize,
        ];
        let mrope_interleaved = rope_scaling["mrope_interleaved"].as_bool().unwrap_or(true);

        let text = Qwen3VLTextConfig {
            hidden_size: tc["hidden_size"].as_u64().unwrap_or(2048) as usize,
            intermediate_size: tc["intermediate_size"].as_u64().unwrap_or(6144) as usize,
            n_layers: tc["num_hidden_layers"].as_u64().unwrap_or(28) as usize,
            n_heads: tc["num_attention_heads"].as_u64().unwrap_or(16) as usize,
            n_kv_heads: tc["num_key_value_heads"].as_u64().unwrap_or(8) as usize,
            head_dim: tc["head_dim"].as_u64().unwrap_or(128) as usize,
            vocab_size: tc["vocab_size"].as_u64().unwrap_or(151936) as usize,
            max_position_embeddings: tc["max_position_embeddings"].as_u64().unwrap_or(262144)
                as usize,
            rms_norm_eps: tc["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            rope_theta: tc["rope_theta"].as_f64().unwrap_or(5_000_000.0) as f32,
            mrope_section,
            mrope_interleaved,
            tie_word_embeddings: tc["tie_word_embeddings"].as_bool().unwrap_or(true),
        };

        let vc = &v["vision_config"];
        let dvi = vc["deepstack_visual_indexes"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| x as usize)
                    .collect()
            })
            .unwrap_or_else(|| vec![5, 11, 17]);
        let vision = Qwen3VLVisionConfig {
            depth: vc["depth"].as_u64().unwrap_or(24) as usize,
            hidden_size: vc["hidden_size"].as_u64().unwrap_or(1024) as usize,
            intermediate_size: vc["intermediate_size"].as_u64().unwrap_or(4096) as usize,
            num_heads: vc["num_heads"].as_u64().unwrap_or(16) as usize,
            head_dim: vc
                .get("head_dim")
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(0),
            patch_size: vc["patch_size"].as_u64().unwrap_or(16) as usize,
            temporal_patch_size: vc["temporal_patch_size"].as_u64().unwrap_or(2) as usize,
            in_channels: vc["in_channels"].as_u64().unwrap_or(3) as usize,
            spatial_merge_size: vc["spatial_merge_size"].as_u64().unwrap_or(2) as usize,
            num_position_embeddings: vc["num_position_embeddings"].as_u64().unwrap_or(2304)
                as usize,
            out_hidden_size: vc["out_hidden_size"].as_u64().unwrap_or(2048) as usize,
            deepstack_visual_indexes: dvi,
        };

        Ok(Qwen3VLConfig {
            text,
            vision,
            image_token_id: v["image_token_id"].as_u64().unwrap_or(151655) as u32,
            video_token_id: v["video_token_id"].as_u64().unwrap_or(151656) as u32,
            vision_start_token_id: v["vision_start_token_id"].as_u64().unwrap_or(151652) as u32,
            vision_end_token_id: v["vision_end_token_id"].as_u64().unwrap_or(151653) as u32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal HF-shaped config, matching Qwen3-VL-2B-Instruct.
    const MIN_CONFIG: &str = r#"{
        "image_token_id": 151655,
        "video_token_id": 151656,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "text_config": {
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "rope_theta": 5000000,
            "tie_word_embeddings": true,
            "rope_scaling": {
                "mrope_interleaved": true,
                "mrope_section": [24, 20, 20]
            }
        }
    }"#;

    #[test]
    fn parses_qwen3vl_2b_config() {
        let cfg = Qwen3VLConfig::from_json_str(MIN_CONFIG).unwrap();
        assert_eq!(cfg.text.n_layers, 28);
        assert_eq!(cfg.text.head_dim, 128);
        assert_eq!(cfg.text.n_heads, 16);
        assert_eq!(cfg.text.n_kv_heads, 8);
        assert_eq!(cfg.text.mrope_section, [24, 20, 20]);
        assert!(cfg.text.mrope_interleaved);
        assert!(cfg.text.tie_word_embeddings);
        assert_eq!(cfg.text.rope_theta, 5_000_000.0);
        assert_eq!(cfg.image_token_id, 151655);
    }
}
