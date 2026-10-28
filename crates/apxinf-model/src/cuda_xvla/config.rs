//! X-VLA configuration (X-VLA-Libero validated profile + checkpoint parsing).
//!
//! Reference: 2toINF/X-VLA @ 6bc2513, checkpoint `X-VLA-Libero`. The validated
//! profile below is read from the checkpoint `config.json`; `from_checkpoint`
//! parses the nested Florence-2 layout and rejects values the port does not
//! implement yet.

use std::path::Path;

use apxinf_core::{Error, Result};

/// Static vision-tower description (DaViT).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DavitConfig {
    pub depths: [usize; 4],
    pub dims: [usize; 4],
    pub num_heads: [usize; 4],
    pub num_groups: [usize; 4],
    pub window_size: usize,
    pub patch_size: [usize; 4],
    pub patch_stride: [usize; 4],
    pub patch_padding: [usize; 4],
    pub patch_prenorm: [bool; 4],
    pub mlp_ratio: usize,
    pub projection_dim: usize,
    pub image_pos_max: usize,
    pub temporal_max: usize,
}

impl DavitConfig {
    pub const XVLA_LIBERO: Self = Self {
        depths: [1, 1, 9, 1],
        dims: [256, 512, 1024, 2048],
        num_heads: [8, 16, 32, 64],
        num_groups: [8, 16, 32, 64],
        window_size: 12,
        patch_size: [7, 3, 3, 3],
        patch_stride: [4, 2, 2, 2],
        patch_padding: [3, 1, 1, 1],
        patch_prenorm: [false, true, true, true],
        mlp_ratio: 4,
        projection_dim: 1024,
        image_pos_max: 50,
        temporal_max: 100,
    };

    /// Feature-map side at each stage for a `side × side` input (224 → 7).
    pub fn feature_sides(&self, input_side: usize) -> [usize; 4] {
        let mut sides = [0usize; 4];
        let mut side = input_side;
        for stage in 0..4 {
            side = (side + 2 * self.patch_padding[stage] - self.patch_size[stage])
                / self.patch_stride[stage]
                + 1;
            sides[stage] = side;
        }
        sides
    }
}

/// BART encoder description (Florence-2 text tower).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BartConfig {
    pub layers: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub ffn_dim: usize,
    pub max_positions: usize,
    pub position_offset: usize,
    pub vocab_size: usize,
    pub pad_token_id: usize,
    pub embed_scale_denominator: Option<usize>,
}

impl BartConfig {
    pub const XVLA_LIBERO: Self = Self {
        layers: 12,
        d_model: 1024,
        num_heads: 16,
        ffn_dim: 4096,
        max_positions: 4096,
        position_offset: 2,
        vocab_size: 51_289,
        pad_token_id: 1,
        embed_scale_denominator: None,
    };
}

/// Action-space hook description (ee6d).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionSpaceConfig {
    pub action_mode: String,
    pub dim_action: usize,
    pub dim_proprio: usize,
    pub gripper_indices: [usize; 2],
}

impl ActionSpaceConfig {
    pub const EE6D: Self = Self {
        action_mode: String::new(),
        dim_action: 20,
        dim_proprio: 20,
        gripper_indices: [9, 19],
    };
}

/// Action head description (SoftPromptedTransformer).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActionHeadConfig {
    pub hidden_size: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub mlp_ratio: usize,
    pub num_domains: usize,
    pub len_soft_prompts: usize,
    pub dim_time: usize,
    pub max_len_seq: usize,
    pub use_hetero_proj: bool,
    pub time_max_period: usize,
}

impl ActionHeadConfig {
    pub const XVLA_LIBERO: Self = Self {
        hidden_size: 1024,
        depth: 24,
        num_heads: 16,
        mlp_ratio: 4,
        num_domains: 30,
        len_soft_prompts: 32,
        dim_time: 32,
        max_len_seq: 512,
        use_hetero_proj: false,
        time_max_period: 100,
    };
}

/// Shape-stable inference profile (CUDA graph capture / plan cache key).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XvlaPerformanceProfile {
    /// Input resolution fed to the vision tower.
    pub image_size: usize,
    /// Number of image views (first view → text merge, rest → aux stream).
    pub num_views: usize,
    /// Text token count after padding.
    pub tokens: usize,
    /// Action horizon (denoised tokens).
    pub action_horizon: usize,
    /// Denoising iterations.
    pub num_flow_steps: usize,
}

impl XvlaPerformanceProfile {
    /// Validated Libero profile: 3 views, 50 tokens, horizon 30, 10 flow steps.
    pub const XVLA_LIBERO: Self = Self {
        image_size: 224,
        num_views: 3,
        tokens: 50,
        action_horizon: 30,
        num_flow_steps: 10,
    };
}

/// Complete X-VLA description.
#[derive(Clone, Debug, PartialEq)]
pub struct XvlaConfig {
    pub num_actions: usize,
    pub use_proprio: bool,
    pub action_space: ActionSpaceConfig,
    pub vision: DavitConfig,
    pub text: BartConfig,
    pub head: ActionHeadConfig,
    pub profile: XvlaPerformanceProfile,
    pub layer_norm_eps: f32,
}

impl XvlaConfig {
    /// Validated X-VLA-Libero profile.
    pub fn xvla_libero() -> Self {
        Self {
            num_actions: 30,
            use_proprio: true,
            action_space: ActionSpaceConfig::EE6D,
            vision: DavitConfig::XVLA_LIBERO,
            text: BartConfig::XVLA_LIBERO,
            head: ActionHeadConfig::XVLA_LIBERO,
            profile: XvlaPerformanceProfile::XVLA_LIBERO,
            // BART text config uses the Transformers default (1e-5).
            layer_norm_eps: 1e-5,
        }
    }

    /// Parse a checkpoint `config.json` and require the implemented profile.
    pub fn from_checkpoint(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let raw = std::fs::read_to_string(&path)
            .map_err(|error| Error::Other(format!("failed to read {}: {error}", path.display())))?;
        let root: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|error| Error::Other(format!("invalid {}: {error}", path.display())))?;
        Self::from_json(&root)
    }

    pub fn from_json(root: &serde_json::Value) -> Result<Self> {
        let florence = field(root, "florence_config")?;
        let vision = field(florence, "vision_config")?;
        let text = field(florence, "text_config")?;

        let action_mode = string(root, "action_mode")?.to_ascii_lowercase();
        if action_mode != "ee6d" {
            return Err(Error::Other(format!(
                "xvla: action_mode {action_mode:?} is not implemented (ee6d only)"
            )));
        }
        let use_hetero_proj = bool_at(root, "use_hetero_proj")?;
        if use_hetero_proj {
            return Err(Error::Other(
                "xvla: use_hetero_proj=true is not implemented".into(),
            ));
        }

        let dims_vec = usize_array(vision, "dim_embed")?;
        let depths_vec = usize_array(vision, "depths")?;
        if dims_vec.as_slice() != [256, 512, 1024, 2048] || depths_vec.as_slice() != [1, 1, 9, 1] {
            return Err(Error::Other(format!(
                "xvla: vision dims/depths {dims_vec:?}/{depths_vec:?} not implemented"
            )));
        }
        let heads = usize_array(vision, "num_heads")?;
        let groups = usize_array(vision, "num_groups")?;
        let patch_size = usize_array(vision, "patch_size")?;
        let patch_stride = usize_array(vision, "patch_stride")?;
        let patch_padding = usize_array(vision, "patch_padding")?;
        let prenorm = bool_array(vision, "patch_prenorm")?;

        let profile = XvlaPerformanceProfile::XVLA_LIBERO;
        let image_size = 224usize;
        let config = Self {
            num_actions: usize_at(root, "num_actions")?,
            use_proprio: bool_at(root, "use_proprio")?,
            action_space: ActionSpaceConfig {
                action_mode,
                dim_action: 20,
                dim_proprio: 20,
                gripper_indices: [9, 19],
            },
            vision: DavitConfig {
                depths: fixed4(&depths_vec)?,
                dims: fixed4(&dims_vec)?,
                num_heads: fixed4(&heads)?,
                num_groups: fixed4(&groups)?,
                window_size: usize_at(vision, "window_size")?,
                patch_size: fixed4(&patch_size)?,
                patch_stride: fixed4(&patch_stride)?,
                patch_padding: fixed4(&patch_padding)?,
                patch_prenorm: fixed4(&prenorm)?,
                mlp_ratio: 4,
                projection_dim: usize_at(vision, "projection_dim")?,
                image_pos_max: vision
                    .get("image_pos_embed")
                    .and_then(|value| value.get("max_pos_embeddings"))
                    .and_then(serde_json::Value::as_u64)
                    .map(|value| value as usize)
                    .unwrap_or(50),
                temporal_max: vision
                    .get("visual_temporal_embedding")
                    .and_then(|value| value.get("max_temporal_embeddings"))
                    .and_then(serde_json::Value::as_u64)
                    .map(|value| value as usize)
                    .unwrap_or(100),
            },
            text: BartConfig {
                layers: usize_at(text, "encoder_layers")?,
                d_model: usize_at(text, "d_model")?,
                num_heads: usize_at(text, "encoder_attention_heads")?,
                ffn_dim: usize_at(text, "encoder_ffn_dim")?,
                max_positions: usize_at(text, "max_position_embeddings")?,
                position_offset: 2,
                vocab_size: usize_at(text, "vocab_size")?,
                pad_token_id: usize_at(text, "pad_token_id")?,
                embed_scale_denominator: if bool_at_or(text, "scale_embedding", false)? {
                    Some(usize_at(text, "d_model")?.isqrt())
                } else {
                    None
                },
            },
            head: ActionHeadConfig {
                hidden_size: usize_at(root, "hidden_size")?,
                depth: usize_at(root, "depth")?,
                num_heads: usize_at(root, "num_heads")?,
                mlp_ratio: number_at(root, "mlp_ratio")? as usize,
                num_domains: usize_at(root, "num_domains")?,
                len_soft_prompts: usize_at(root, "len_soft_prompts")?,
                dim_time: usize_at(root, "dim_time")?,
                max_len_seq: usize_at(root, "max_len_seq")?,
                use_hetero_proj,
                time_max_period: 100,
            },
            profile: XvlaPerformanceProfile {
                image_size,
                ..profile
            },
            layer_norm_eps: text
                .get("layer_norm_eps")
                .and_then(serde_json::Value::as_f64)
                .map(|value| value as f32)
                .unwrap_or(1e-5),
        };

        config.validate()?;
        Ok(config)
    }

    /// Shapes the port hard-codes in the fixed profile.
    pub fn validate(&self) -> Result<()> {
        let expected = Self::xvla_libero();
        let mut problems = Vec::new();
        if self.num_actions != 30 {
            problems.push(format!("num_actions={} (want 30)", self.num_actions));
        }
        if !self.use_proprio {
            problems.push("use_proprio=false".into());
        }
        if self.vision != expected.vision {
            problems.push(format!("vision config {:?} != validated", self.vision));
        }
        if self.text.layers != 12
            || self.text.d_model != 1024
            || self.text.num_heads != 16
            || self.text.ffn_dim != 4096
            || self.text.vocab_size != 51_289
            || self.text.position_offset != 2
        {
            problems.push(format!("text config {:?} not supported", self.text));
        }
        if self.head != expected.head {
            problems.push(format!("action head {:?} not supported", self.head));
        }
        if self.profile != expected.profile {
            problems.push(format!(
                "profile {:?} is not the validated one",
                self.profile
            ));
        }
        if problems.is_empty() {
            Ok(())
        } else {
            Err(Error::Other(format!(
                "xvla: checkpoint config outside the ported profile: {}",
                problems.join("; ")
            )))
        }
    }
}

fn field<'a>(value: &'a serde_json::Value, key: &str) -> Result<&'a serde_json::Value> {
    value
        .get(key)
        .ok_or_else(|| Error::Other(format!("xvla config: missing `{key}`")))
}

fn usize_at(value: &serde_json::Value, key: &str) -> Result<usize> {
    field(value, key)?
        .as_u64()
        .map(|number| number as usize)
        .ok_or_else(|| {
            Error::Other(format!(
                "xvla config: `{key}` must be a non-negative integer"
            ))
        })
}

fn number_at(value: &serde_json::Value, key: &str) -> Result<f64> {
    field(value, key)?
        .as_f64()
        .ok_or_else(|| Error::Other(format!("xvla config: `{key}` must be a number")))
}

fn bool_at(value: &serde_json::Value, key: &str) -> Result<bool> {
    field(value, key)?
        .as_bool()
        .ok_or_else(|| Error::Other(format!("xvla config: `{key}` must be a boolean")))
}

fn bool_at_or(value: &serde_json::Value, key: &str, default: bool) -> Result<bool> {
    match value.get(key) {
        None => Ok(default),
        Some(inner) => inner
            .as_bool()
            .ok_or_else(|| Error::Other(format!("xvla config: `{key}` must be a boolean"))),
    }
}

fn string<'a>(value: &'a serde_json::Value, key: &str) -> Result<&'a str> {
    field(value, key)?
        .as_str()
        .ok_or_else(|| Error::Other(format!("xvla config: `{key}` must be a string")))
}

fn usize_array(value: &serde_json::Value, key: &str) -> Result<Vec<usize>> {
    let array = field(value, key)?
        .as_array()
        .ok_or_else(|| Error::Other(format!("xvla config: `{key}` must be an array")))?;
    array
        .iter()
        .map(|entry| {
            entry.as_u64().map(|number| number as usize).ok_or_else(|| {
                Error::Other(format!("xvla config: `{key}` entries must be integers"))
            })
        })
        .collect()
}

fn bool_array(value: &serde_json::Value, key: &str) -> Result<Vec<bool>> {
    let array = field(value, key)?
        .as_array()
        .ok_or_else(|| Error::Other(format!("xvla config: `{key}` must be an array")))?;
    array
        .iter()
        .map(|entry| {
            entry.as_bool().ok_or_else(|| {
                Error::Other(format!("xvla config: `{key}` entries must be booleans"))
            })
        })
        .collect()
}

fn fixed4<T: Copy>(values: &[T]) -> Result<[T; 4]> {
    values.try_into().map_err(|_| {
        Error::Other(format!(
            "xvla config: expected 4 entries, got {}",
            values.len()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validated_profile_roundtrips() {
        let config = XvlaConfig::xvla_libero();
        config.validate().unwrap();
        assert_eq!(config.profile.image_size, 224);
        assert_eq!(config.vision.feature_sides(224), [56, 28, 14, 7]);
    }
}
