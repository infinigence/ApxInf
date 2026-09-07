//! Qwen-Drive-1.0 configuration, parsed from the checkpoint `config.json`.
//!
//! The schema mirrors `configuration_qwen_drive.py` in the pinned reference
//! (revision 28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0): a Qwen3.5 hybrid VLM
//! (`vlm_config`) plus a flow-matching planning expert (`expert_config`), with
//! trajectory and sampler constants at the top level. Parsing here keeps the
//! loader crate model-agnostic and matches "each model owns its config".

use std::path::Path;

use apxinf_core::{Error, Result};

/// Text-decoder geometry of the Qwen3.5 VLM. The decoder is hybrid: layers
/// listed as `full_attention` carry a standard GQA KV cache (these are the
/// caches the planning expert reads), while `linear_attention` layers are
/// gated delta net (GDN) linear-attention layers with a recurrent state.
#[derive(Clone, Debug)]
pub struct QwenDriveTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    /// Qwen3.5 emits a per-head sigmoid gate on the attention output.
    pub attn_output_gate: bool,
    /// Per-layer type names: `full_attention` or `linear_attention`.
    pub layer_types: Vec<String>,
    // GDN geometry (only meaningful for `linear_attention` layers).
    pub linear_num_key_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    // Partial interleaved mRoPE.
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub mrope_section: [usize; 3],
    pub mrope_interleaved: bool,
    pub tie_word_embeddings: bool,
    pub eos_token_id: u32,
}

impl QwenDriveTextConfig {
    /// Rotary channel count of the partial RoPE (64 of 256 for Qwen3.5-4B).
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f32 * self.partial_rotary_factor).round() as usize
    }

    /// Indices of the full-attention layers, in layer order. These are exactly
    /// the caches the planning expert consumes (config.full_attention_layers).
    pub fn full_attention_layers(&self) -> Vec<usize> {
        self.layer_types
            .iter()
            .enumerate()
            .filter(|(_, kind)| kind.as_str() == "full_attention")
            .map(|(index, _)| index)
            .collect()
    }

    pub fn is_full_attention(&self, layer_index: usize) -> bool {
        self.layer_types
            .get(layer_index)
            .map(|kind| kind.as_str() == "full_attention")
            .unwrap_or(false)
    }

    /// Validate cross-field invariants the executor relies on.
    pub fn validate(&self) -> Result<()> {
        if self.layer_types.len() != self.n_layers {
            return Err(Error::Other(format!(
                "qwen_drive text config: layer_types has {} entries but num_hidden_layers is {}",
                self.layer_types.len(),
                self.n_layers
            )));
        }
        for (index, kind) in self.layer_types.iter().enumerate() {
            if kind != "full_attention" && kind != "linear_attention" {
                return Err(Error::Other(format!(
                    "qwen_drive text config: unknown layer type {kind:?} at index {index}"
                )));
            }
        }
        let rotary = self.rotary_dim();
        if rotary == 0 || rotary % 2 != 0 || rotary > self.head_dim {
            return Err(Error::Other(format!(
                "qwen_drive text config: invalid rotary dim {rotary} for head dim {}",
                self.head_dim
            )));
        }
        let pairs: usize = self.mrope_section.iter().sum();
        if pairs != rotary / 2 {
            return Err(Error::Other(format!(
                "qwen_drive text config: mrope_section {:?} must sum to {} frequency pairs",
                self.mrope_section,
                rotary / 2
            )));
        }
        Ok(())
    }
}

/// Vision tower geometry (ViT, 24 blocks, hidden 1024, merger to 2560).
#[derive(Clone, Debug)]
pub struct QwenDriveVisionConfig {
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
}

impl QwenDriveVisionConfig {
    pub fn head_dim(&self) -> usize {
        if self.head_dim != 0 {
            self.head_dim
        } else {
            self.hidden_size / self.num_heads
        }
    }
}

/// Planning expert geometry (`PlanningExpertConfig` in the reference).
#[derive(Clone, Debug)]
pub struct PlanningExpertConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Consecutive expert layers sharing one VLM scene cache.
    pub layers_per_kv: usize,
    pub rms_norm_eps: f32,
    pub time_embed_dim: usize,
    pub time_embed_scale: f32,
    pub fourier_num_features: usize,
    pub fourier_max_frequency: f32,
    pub nav_command_classes: usize,
    pub ego_status_dim: usize,
    pub history_dynamics_dim: usize,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
    pub mrope_section: [usize; 3],
}

impl PlanningExpertConfig {
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f32 * self.partial_rotary_factor).round() as usize
    }

    /// Number of VLM scene caches consumed (one per `layers_per_kv` layers).
    pub fn num_kv_sources(&self) -> usize {
        self.n_layers / self.layers_per_kv
    }

    pub fn validate(&self) -> Result<()> {
        if self.n_layers % self.layers_per_kv != 0 {
            return Err(Error::Other(format!(
                "qwen_drive expert config: {} layers not divisible by layers_per_kv {}",
                self.n_layers,
                self.layers_per_kv
            )));
        }
        let rotary = self.rotary_dim();
        if rotary % 2 != 0 || rotary > self.head_dim {
            return Err(Error::Other(format!(
                "qwen_drive expert config: invalid rotary dim {rotary}"
            )));
        }
        let pairs: usize = self.mrope_section.iter().sum();
        if pairs != rotary / 2 {
            return Err(Error::Other(format!(
                "qwen_drive expert config: mrope_section {:?} must sum to {} frequency pairs",
                self.mrope_section,
                rotary / 2
            )));
        }
        Ok(())
    }
}

/// Full Qwen-Drive-1.0 configuration (`QwenDriveConfig` in the reference).
#[derive(Clone, Debug)]
pub struct QwenDriveConfig {
    pub text: QwenDriveTextConfig,
    pub vision: QwenDriveVisionConfig,
    pub expert: PlanningExpertConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    // Trajectory geometry.
    pub num_future_points: usize,
    pub num_history_points: usize,
    pub trajectory_point_dim: usize,
    pub trajectory_hz: f32,
    /// Per-channel normalization scale; the heading entry is the bfloat16
    /// rounding of pi/2 (1.5703125), which is the value training saw.
    pub trajectory_scale: [f32; 3],
    // Flow-matching sampler.
    pub num_inference_steps: usize,
    pub noise_init_std: f32,
    pub noise_seed: u64,
    pub min_one_minus_t: f32,
    // Image preprocessing budgets.
    pub history_image_pixels: usize,
    pub current_image_pixels: usize,
    pub image_patch_size: usize,
    pub image_temporal_patch_size: usize,
    pub image_spatial_merge_size: usize,
    // Reasoning-stage generation bounds.
    pub max_reasoning_tokens: usize,
    pub min_reasoning_tokens: usize,
}

impl QwenDriveConfig {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| Error::Other(format!("read {}: {e}", path.display())))?;
        Self::from_json_str(&raw)
    }

    pub fn from_json_str(s: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s)
            .map_err(|e| Error::Other(format!("qwen_drive config json: {e}")))?;
        let vlm = &v["vlm_config"];
        if !vlm.is_object() {
            return Err(Error::Other("qwen_drive config: missing vlm_config".into()));
        }
        let tc = &vlm["text_config"];
        if !tc.is_object() {
            return Err(Error::Other(
                "qwen_drive config: missing vlm_config.text_config".into(),
            ));
        }
        let layer_types: Vec<String> = tc["layer_types"]
            .as_array()
            .ok_or_else(|| Error::Other("qwen_drive config: missing text_config.layer_types".into()))?
            .iter()
            .filter_map(|entry| entry.as_str().map(str::to_owned))
            .collect();
        let rope = &tc["rope_parameters"];
        let section = rope["mrope_section"]
            .as_array()
            .ok_or_else(|| {
                Error::Other("qwen_drive config: missing rope_parameters.mrope_section".into())
            })?;
        if section.len() != 3 {
            return Err(Error::Other(format!(
                "qwen_drive config: mrope_section must have 3 entries, got {}",
                section.len()
            )));
        }
        let mrope_section = [
            section[0].as_u64().unwrap_or(11) as usize,
            section[1].as_u64().unwrap_or(11) as usize,
            section[2].as_u64().unwrap_or(10) as usize,
        ];
        let text = QwenDriveTextConfig {
            hidden_size: tc["hidden_size"].as_u64().unwrap_or(2560) as usize,
            intermediate_size: tc["intermediate_size"].as_u64().unwrap_or(9216) as usize,
            n_layers: tc["num_hidden_layers"].as_u64().unwrap_or(32) as usize,
            n_heads: tc["num_attention_heads"].as_u64().unwrap_or(16) as usize,
            n_kv_heads: tc["num_key_value_heads"].as_u64().unwrap_or(4) as usize,
            head_dim: tc["head_dim"].as_u64().unwrap_or(256) as usize,
            vocab_size: tc["vocab_size"].as_u64().unwrap_or(248320) as usize,
            max_position_embeddings: tc["max_position_embeddings"].as_u64().unwrap_or(32768) as usize,
            rms_norm_eps: tc["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            attn_output_gate: tc["attn_output_gate"].as_bool().unwrap_or(true),
            layer_types,
            linear_num_key_heads: tc["linear_num_key_heads"].as_u64().unwrap_or(16) as usize,
            linear_key_head_dim: tc["linear_key_head_dim"].as_u64().unwrap_or(128) as usize,
            linear_num_value_heads: tc["linear_num_value_heads"].as_u64().unwrap_or(32) as usize,
            linear_value_head_dim: tc["linear_value_head_dim"].as_u64().unwrap_or(128) as usize,
            linear_conv_kernel_dim: tc["linear_conv_kernel_dim"].as_u64().unwrap_or(4) as usize,
            rope_theta: rope["rope_theta"].as_f64().unwrap_or(10_000_000.0) as f32,
            partial_rotary_factor: rope["partial_rotary_factor"].as_f64().unwrap_or(0.25) as f32,
            mrope_section,
            mrope_interleaved: rope["mrope_interleaved"].as_bool().unwrap_or(true),
            tie_word_embeddings: tc["tie_word_embeddings"].as_bool().unwrap_or(true),
            eos_token_id: tc["eos_token_id"].as_u64().unwrap_or(248044) as u32,
        };
        text.validate()?;

        let vc = &vlm["vision_config"];
        if !vc.is_object() {
            return Err(Error::Other(
                "qwen_drive config: missing vlm_config.vision_config".into(),
            ));
        }
        let vision = QwenDriveVisionConfig {
            depth: vc["depth"].as_u64().unwrap_or(24) as usize,
            hidden_size: vc["hidden_size"].as_u64().unwrap_or(1024) as usize,
            intermediate_size: vc["intermediate_size"].as_u64().unwrap_or(4096) as usize,
            num_heads: vc["num_heads"].as_u64().unwrap_or(16) as usize,
            head_dim: vc.get("head_dim").and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(0),
            patch_size: vc["patch_size"].as_u64().unwrap_or(16) as usize,
            temporal_patch_size: vc["temporal_patch_size"].as_u64().unwrap_or(2) as usize,
            in_channels: vc["in_channels"].as_u64().unwrap_or(3) as usize,
            spatial_merge_size: vc["spatial_merge_size"].as_u64().unwrap_or(2) as usize,
            num_position_embeddings: vc["num_position_embeddings"].as_u64().unwrap_or(2304) as usize,
            out_hidden_size: vc["out_hidden_size"].as_u64().unwrap_or(2560) as usize,
        };

        let ec = &v["expert_config"];
        if !ec.is_object() {
            return Err(Error::Other("qwen_drive config: missing expert_config".into()));
        }
        let esection = ec["mrope_section"]
            .as_array()
            .ok_or_else(|| Error::Other("qwen_drive config: missing expert mrope_section".into()))?;
        if esection.len() != 3 {
            return Err(Error::Other(format!(
                "qwen_drive config: expert mrope_section must have 3 entries, got {}",
                esection.len()
            )));
        }
        let expert = PlanningExpertConfig {
            hidden_size: ec["hidden_size"].as_u64().unwrap_or(1024) as usize,
            intermediate_size: ec["intermediate_size"].as_u64().unwrap_or(3584) as usize,
            n_layers: ec["num_hidden_layers"].as_u64().unwrap_or(32) as usize,
            n_heads: ec["num_attention_heads"].as_u64().unwrap_or(16) as usize,
            n_kv_heads: ec["num_key_value_heads"].as_u64().unwrap_or(4) as usize,
            head_dim: ec["head_dim"].as_u64().unwrap_or(256) as usize,
            layers_per_kv: ec["layers_per_kv"].as_u64().unwrap_or(4) as usize,
            rms_norm_eps: ec["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
            time_embed_dim: ec["time_embed_dim"].as_u64().unwrap_or(128) as usize,
            time_embed_scale: ec["time_embed_scale"].as_f64().unwrap_or(1000.0) as f32,
            fourier_num_features: ec["fourier_num_features"].as_u64().unwrap_or(16) as usize,
            fourier_max_frequency: ec["fourier_max_frequency"].as_f64().unwrap_or(16.0) as f32,
            nav_command_classes: ec["nav_command_classes"].as_u64().unwrap_or(3) as usize,
            ego_status_dim: ec["ego_status_dim"].as_u64().unwrap_or(8) as usize,
            history_dynamics_dim: ec["history_dynamics_dim"].as_u64().unwrap_or(2) as usize,
            rope_theta: ec["rope_theta"].as_f64().unwrap_or(10_000_000.0) as f32,
            partial_rotary_factor: ec["partial_rotary_factor"].as_f64().unwrap_or(0.25) as f32,
            mrope_section: [
                esection[0].as_u64().unwrap_or(11) as usize,
                esection[1].as_u64().unwrap_or(11) as usize,
                esection[2].as_u64().unwrap_or(10) as usize,
            ],
        };
        expert.validate()?;

        let scale = v["trajectory_scale"]
            .as_array()
            .ok_or_else(|| Error::Other("qwen_drive config: missing trajectory_scale".into()))?;
        if scale.len() != 3 {
            return Err(Error::Other(format!(
                "qwen_drive config: trajectory_scale must have 3 entries, got {}",
                scale.len()
            )));
        }
        let config = QwenDriveConfig {
            text,
            vision,
            expert,
            image_token_id: vlm["image_token_id"].as_u64().unwrap_or(248056) as u32,
            video_token_id: vlm["video_token_id"].as_u64().unwrap_or(248057) as u32,
            vision_start_token_id: vlm["vision_start_token_id"].as_u64().unwrap_or(248053) as u32,
            vision_end_token_id: vlm["vision_end_token_id"].as_u64().unwrap_or(248054) as u32,
            num_future_points: v["num_future_points"].as_u64().unwrap_or(50) as usize,
            num_history_points: v["num_history_points"].as_u64().unwrap_or(16) as usize,
            trajectory_point_dim: v["trajectory_point_dim"].as_u64().unwrap_or(3) as usize,
            trajectory_hz: v["trajectory_hz"].as_f64().unwrap_or(10.0) as f32,
            trajectory_scale: [
                scale[0].as_f64().unwrap_or(165.0) as f32,
                scale[1].as_f64().unwrap_or(25.0) as f32,
                scale[2].as_f64().unwrap_or(1.5703125) as f32,
            ],
            num_inference_steps: v["num_inference_steps"].as_u64().unwrap_or(10) as usize,
            noise_init_std: v["noise_init_std"].as_f64().unwrap_or(1.0) as f32,
            noise_seed: v["noise_seed"].as_u64().unwrap_or(42),
            min_one_minus_t: v["min_one_minus_t"].as_f64().unwrap_or(0.1) as f32,
            history_image_pixels: v["history_image_pixels"].as_u64().unwrap_or(174080) as usize,
            current_image_pixels: v["current_image_pixels"].as_u64().unwrap_or(921600) as usize,
            image_patch_size: v["image_patch_size"].as_u64().unwrap_or(16) as usize,
            image_temporal_patch_size: v["image_temporal_patch_size"].as_u64().unwrap_or(2) as usize,
            image_spatial_merge_size: v["image_spatial_merge_size"].as_u64().unwrap_or(2) as usize,
            max_reasoning_tokens: v["max_reasoning_tokens"].as_u64().unwrap_or(256) as usize,
            min_reasoning_tokens: v["min_reasoning_tokens"].as_u64().unwrap_or(10) as usize,
        };
        config.validate()?;
        Ok(config)
    }

    /// Cross-validate the VLM and the expert: the expert reads the VLM's
    /// post-rotary K/V directly, so their attention geometry must match and
    /// the number of exported caches must equal the expert's KV sources.
    pub fn validate(&self) -> Result<()> {
        let full = self.text.full_attention_layers();
        if full.len() != self.expert.num_kv_sources() {
            return Err(Error::Other(format!(
                "qwen_drive config: {} VLM full-attention layers but the expert expects {} scene caches",
                full.len(),
                self.expert.num_kv_sources()
            )));
        }
        if self.text.n_kv_heads != self.expert.n_kv_heads || self.text.head_dim != self.expert.head_dim {
            return Err(Error::Other(format!(
                "qwen_drive config: expert KV geometry ({} heads x dim {}) must match the VLM ({} x {})",
                self.expert.n_kv_heads,
                self.expert.head_dim,
                self.text.n_kv_heads,
                self.text.head_dim
            )));
        }
        if self.num_history_points < 2 {
            return Err(Error::Other(
                "qwen_drive config: num_history_points must be at least 2".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Abbreviated but schema-faithful excerpt of the released checkpoint
    // config (experiment/qwen-drive-k3/checkpoint-configs/config.json).
    const CHECKPOINT_CONFIG: &str = r#"{
        "model_type": "qwen_drive",
        "num_future_points": 50,
        "num_history_points": 16,
        "trajectory_point_dim": 3,
        "trajectory_hz": 10.0,
        "trajectory_scale": [165.0, 25.0, 1.5703125],
        "num_inference_steps": 10,
        "noise_init_std": 1.0,
        "noise_seed": 42,
        "min_one_minus_t": 0.1,
        "max_reasoning_tokens": 256,
        "history_image_pixels": 174080,
        "current_image_pixels": 921600,
        "image_patch_size": 16,
        "image_temporal_patch_size": 2,
        "image_spatial_merge_size": 2,
        "expert_config": {
            "hidden_size": 1024,
            "intermediate_size": 3584,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "layers_per_kv": 4,
            "rms_norm_eps": 1e-05,
            "time_embed_dim": 128,
            "time_embed_scale": 1000.0,
            "fourier_num_features": 16,
            "fourier_max_frequency": 16.0,
            "nav_command_classes": 3,
            "ego_status_dim": 8,
            "history_dynamics_dim": 2,
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10]
        },
        "vlm_config": {
            "model_type": "qwen3_5",
            "image_token_id": 248056,
            "video_token_id": 248057,
            "vision_start_token_id": 248053,
            "vision_end_token_id": 248054,
            "tie_word_embeddings": true,
            "text_config": {
                "hidden_size": 2560,
                "intermediate_size": 9216,
                "num_hidden_layers": 32,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "vocab_size": 248320,
                "max_position_embeddings": 32768,
                "rms_norm_eps": 1e-06,
                "attn_output_gate": true,
                "eos_token_id": 248044,
                "tie_word_embeddings": true,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention"],
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128,
                "linear_num_value_heads": 32,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "rope_parameters": {
                    "mrope_interleaved": true,
                    "mrope_section": [11, 11, 10],
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10000000
                }
            },
            "vision_config": {
                "depth": 24,
                "hidden_size": 1024,
                "intermediate_size": 4096,
                "num_heads": 16,
                "in_channels": 3,
                "num_position_embeddings": 2304,
                "out_hidden_size": 2560,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2
            }
        }
    }"#;

    #[test]
    fn parses_checkpoint_schema() {
        let cfg = QwenDriveConfig::from_json_str(CHECKPOINT_CONFIG).unwrap();
        assert_eq!(cfg.text.hidden_size, 2560);
        assert_eq!(cfg.text.n_layers, 32);
        assert_eq!(cfg.text.head_dim, 256);
        assert_eq!(cfg.text.rotary_dim(), 64);
        assert_eq!(cfg.text.mrope_section, [11, 11, 10]);
        assert!(cfg.text.attn_output_gate);
        assert!(cfg.text.is_full_attention(3));
        assert!(!cfg.text.is_full_attention(0));
        assert_eq!(
            cfg.text.full_attention_layers(),
            vec![3, 7, 11, 15, 19, 23, 27, 31]
        );
        assert_eq!(cfg.vision.depth, 24);
        assert_eq!(cfg.vision.head_dim(), 64);
        assert_eq!(cfg.expert.n_layers, 32);
        assert_eq!(cfg.expert.num_kv_sources(), 8);
        assert_eq!(cfg.expert.rotary_dim(), 64);
        assert_eq!(cfg.trajectory_scale, [165.0, 25.0, 1.5703125]);
        assert_eq!(cfg.noise_seed, 42);
        assert_eq!(cfg.num_future_points, 50);
        assert_eq!(cfg.min_reasoning_tokens, 10);
    }

    #[test]
    fn rejects_expert_vlm_cache_mismatch() {
        // Growing the expert to 36 layers makes it expect 9 scene caches
        // while the VLM only has 8 full-attention layers.
        let broken = CHECKPOINT_CONFIG.replacen("\"num_hidden_layers\": 32,", "\"num_hidden_layers\": 36,", 1);
        let err = QwenDriveConfig::from_json_str(&broken);
        assert!(err.is_err());
        let message = format!("{}", err.err().unwrap());
        assert!(message.contains("full-attention"), "unexpected error: {message}");
    }
}
