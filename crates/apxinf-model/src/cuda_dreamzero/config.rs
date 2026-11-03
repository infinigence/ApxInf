use apxinf_core::{Error, Result};
use serde_json::Value;
use std::path::Path;

/// Geometry admitted by the released DreamZero DROID joint model.
/// Sampler steps are executable reference behavior; the upstream training
/// config's `num_inference_timesteps=4` does not select its inference schedule.
#[derive(Clone, Debug, PartialEq)]
pub struct DreamZeroConfig {
    pub(super) dim: usize,
    pub(super) ffn_dim: usize,
    pub(super) heads: usize,
    pub(super) layers: usize,
    pub(super) action_horizon: usize,
    pub(super) action_dim: usize,
    pub(super) state_dim: usize,
    pub(super) latent_frames: usize,
    pub(super) max_chunk_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    // Curated architecture fields from GEAR-Dreams/DreamZero-DROID revision
    // 96ad344138c66e82536422432ad742f015784942; no training paths or state.
    const CONFIG: &str = include_str!("config_fixture.json");
    #[test]
    fn released_standard_has_joint_latent_contract() {
        let c = DreamZeroConfig::from_json(CONFIG).unwrap();
        assert_eq!(c.video_noise_elements(), 112640);
        assert_eq!(c.joint_noise_elements(), 113408);
    }
    #[test]
    fn incompatible_geometry_and_sampling_fail_closed() {
        for (path, value) in [
            ("/action_head_cfg/config/action_dim", Value::from(48)),
            (
                "/action_head_cfg/config/decouple_inference_noise",
                Value::Bool(true),
            ),
            (
                "/action_head_cfg/config/decouple_video_action_noise",
                Value::Bool(true),
            ),
            (
                "/action_head_cfg/config/train_architecture",
                Value::from("lora"),
            ),
            (
                "/action_head_cfg/config/diffusion_model_cfg/num_layers",
                Value::from(39),
            ),
            (
                "/action_head_cfg/config/text_encoder_cfg/_target_",
                Value::from("another.Encoder"),
            ),
        ] {
            let mut config: Value = serde_json::from_str(CONFIG).unwrap();
            *config.pointer_mut(path).unwrap() = value;
            assert!(
                DreamZeroConfig::from_json(&config.to_string()).is_err(),
                "{path}"
            );
        }
        let mut config: Value = serde_json::from_str(CONFIG).unwrap();
        config["action_head_cfg"]["config"]["vae_cfg"]["z_dim"] = Value::from(32);
        assert!(DreamZeroConfig::from_json(&config.to_string()).is_err());
    }
}

impl DreamZeroConfig {
    pub const TEXT_DIM: usize = 4096;
    pub const TEXT_FFN: usize = 10240;
    pub const TEXT_HEADS: usize = 64;
    pub const TEXT_LAYERS: usize = 24;
    pub const TEXT_VOCAB: usize = 256384;
    pub const TEXT_LEN: usize = 512;
    pub const LATENT_HEIGHT: usize = 44;
    pub const LATENT_WIDTH: usize = 80;
    pub const LATENT_CHANNELS: usize = 16;
    pub const STEPS: usize = 16;

    pub fn from_json_file(path: &Path) -> Result<Self> {
        let source = std::fs::read_to_string(path)
            .map_err(|e| Error::Other(format!("{}: {e}", path.display())))?;
        Self::from_json(&source)
    }

    pub fn from_json(source: &str) -> Result<Self> {
        let root: Value = serde_json::from_str(source)
            .map_err(|e| Error::Other(format!("DreamZero config: {e}")))?;
        let head = &root["action_head_cfg"]["config"];
        let dit = &head["diffusion_model_cfg"];
        let require = |value: &Value, key: &str, expected: Value| -> Result<()> {
            if value[key] != expected {
                return Err(Error::Other(format!(
                    "unsupported DreamZero {key}: expected {expected}, got {}",
                    value[key]
                )));
            }
            Ok(())
        };
        require(
            &root["action_head_cfg"],
            "_target_",
            Value::from(
                "groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf.WANPolicyHead",
            ),
        )?;
        require(dit, "_target_", Value::from("groot.vla.model.dreamzero.modules.wan_video_dit_action_casual_chunk.CausalWanModel"))?;
        for (key, expected) in [
            ("dim", 5120),
            ("ffn_dim", 13824),
            ("num_heads", 40),
            ("num_layers", 40),
            ("in_dim", 36),
            ("out_dim", 16),
            ("freq_dim", 256),
            ("frame_seqlen", 880),
            ("num_action_per_block", 24),
            ("num_frame_per_block", 2),
            ("num_state_per_block", 1),
            ("max_chunk_size", 4),
        ] {
            require(dit, key, Value::from(expected))?;
        }
        for (key, expected) in [
            ("action_horizon", 24),
            ("action_dim", 32),
            ("max_action_dim", 32),
            ("max_state_dim", 64),
            ("num_frames", 33),
            ("num_frame_per_block", 2),
        ] {
            require(head, key, Value::from(expected))?;
        }
        require(dit, "model_type", Value::from("i2v"))?;
        require(&root, "action_dim", Value::from(32))?;
        require(&root, "action_horizon", Value::from(24))?;
        require(
            &root["backbone_cfg"],
            "_target_",
            Value::from("groot.vla.model.dreamzero.backbone.identity.IdentityBackbone"),
        )?;
        require(head, "train_architecture", Value::from("full"))?;
        require(head, "num_timestep_buckets", Value::from(1000))?;
        for (section, class, path) in [
            (
                "text_encoder_cfg",
                "wan_video_text_encoder.WanTextEncoder",
                "text_encoder_pretrained_path",
            ),
            (
                "image_encoder_cfg",
                "wan_video_image_encoder.WanImageEncoder",
                "image_encoder_pretrained_path",
            ),
            (
                "vae_cfg",
                "wan_video_vae.WanVideoVAE",
                "vae_pretrained_path",
            ),
        ] {
            let config = &head[section];
            require(
                config,
                "_target_",
                Value::from(format!("groot.vla.model.dreamzero.modules.{class}")),
            )?;
            let object = config
                .as_object()
                .ok_or_else(|| Error::Other(format!("DreamZero {section} must be an object")))?;
            for key in object.keys() {
                if !["_target_", "_convert_", "_recursive_", path].contains(&key.as_str()) {
                    return Err(Error::Other(format!(
                        "unsupported DreamZero {section}.{key} override"
                    )));
                }
            }
            if config.get(path).is_some_and(|value| !value.is_null()) {
                return Err(Error::Other(format!("DreamZero loads {section} from the joint checkpoint, not an external pretrained path")));
            }
        }
        for (key, default) in [
            ("patch_size", serde_json::json!([1, 2, 2])),
            ("qk_norm", Value::Bool(true)),
            ("cross_attn_norm", Value::Bool(true)),
        ] {
            if dit.get(key).is_some_and(|value| value != &default) {
                return Err(Error::Other(format!("unsupported DreamZero DiT {key}")));
            }
        }
        require(head, "tiled", Value::Bool(false))?;
        for key in ["decouple_video_action_noise", "decouple_inference_noise"] {
            if head.get(key).is_some_and(|v| v != &Value::Bool(false)) {
                return Err(Error::Other(format!("DreamZero {key} requires an independently verified sampling profile; reduced steps or decoupling do not identify Flash weights")));
            }
        }
        if dit["eps"].as_f64() != Some(1e-6) {
            return Err(Error::Other(
                "unsupported DreamZero normalization epsilon".into(),
            ));
        }
        Ok(Self {
            dim: 5120,
            ffn_dim: 13824,
            heads: 40,
            layers: 40,
            action_horizon: 24,
            action_dim: 32,
            state_dim: 64,
            latent_frames: 2,
            max_chunk_size: 4,
        })
    }

    pub fn video_noise_elements(&self) -> usize {
        Self::LATENT_CHANNELS * self.latent_frames * Self::LATENT_HEIGHT * Self::LATENT_WIDTH
    }
    pub fn joint_noise_elements(&self) -> usize {
        self.video_noise_elements() + self.action_horizon * self.action_dim
    }
}

#[cfg(feature = "cuda")]
mod execution {
    use super::super::backend::ops;
    #[derive(Clone)]
    pub(in crate::cuda_dreamzero) struct Policies {
        pub gemm: ops::GemmPolicy,
        pub attention: ops::AttentionPolicy,
    }

    impl Default for Policies {
        fn default() -> Self {
            Self {
                gemm: ops::GemmPolicy {
                    deterministic: true,
                    algorithm: ops::GemmAlgorithmPolicy::ColdBaseline,
                    // Bound per-projection retained scratch across all DiT blocks.
                    workspace_limit: 16 * 1024 * 1024,
                    ..Default::default()
                },
                attention: ops::AttentionPolicy {
                    deterministic: true,
                    // UMT5's materialized rounded attention needs 40 MiB. Bound
                    // each prepared execution so joint DiT attention chooses the
                    // existing linear-memory FA2 fallback on a cold recipe store.
                    workspace_limit: 64 * 1024 * 1024,
                    ..Default::default()
                },
            }
        }
    }
}
#[cfg(feature = "cuda")]
pub(super) use execution::Policies;
