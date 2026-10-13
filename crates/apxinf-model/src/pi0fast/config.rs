//! π0-FAST checkpoint and execution configuration.
//!
//! The checkpoint is a LeRobot `pi0_fast` policy: a PaliGemma (SigLIP So400m/14
//! vision + Gemma-2B text) backbone whose action tokens are produced
//! autoregressively by the same LM head and then decoded by the FAST action
//! tokenizer. Configuration is parsed from the checkpoint's own `config.json`
//! without forcing it into a π0.5-shaped layout.

use std::path::Path;

use apxinf_core::{Error, Result};

/// Column alignment of the pruned LM head.
///
/// E4M3 tensor-core GEMMs in cuBLASLt reject a leading dimension that is not
/// vector-width aligned (`CUBLAS_STATUS_NOT_SUPPORTED`), so the pruned head is
/// padded out to this multiple with repeats of its last column. A repeated
/// column can only ever win the argmax where the original would have won too,
/// and it remaps to the same token id, so the padding is behaviour-neutral.
pub const ACTION_HEAD_COLUMN_ALIGN: usize = 64;

/// Per-variant Gemma dimensions used by the PaliGemma text tower.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pi0FastLanguageConfig {
    pub width: usize,
    pub depth: usize,
    pub mlp_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl Pi0FastLanguageConfig {
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

    pub fn from_variant(variant: &str) -> Result<Self> {
        match variant {
            "gemma_2b" => Ok(Self::GEMMA_2B),
            "gemma_300m" => Ok(Self::GEMMA_300M),
            other => Err(Error::Other(format!(
                "pi0fast: unknown paligemma_variant {other:?}; expected gemma_2b or gemma_300m"
            ))),
        }
    }
}

/// Checkpoint-fixed shape profile for CUDA graph capture and workspace sizing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pi0FastPerformanceProfile {
    pub num_views: usize,
    pub max_action_tokens: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pi0FastConfig {
    /// Deployable action width decoded by the FAST tokenizer (7 for LIBERO).
    pub action_dim: usize,
    /// Number of timesteps in one decoded action chunk (`n_action_steps`).
    pub action_horizon: usize,
    /// Padded state width written into the prompt (`max_state_dim`).
    pub max_state_dim: usize,
    /// Padded action width the checkpoint trained against (`max_action_dim`).
    pub max_action_dim: usize,
    /// Upper bound on autoregressive FAST decoding steps.
    pub max_action_tokens: usize,
    /// PaliGemma token-space offset between language and action tokens.
    pub fast_skip_tokens: usize,
    /// Width of the contiguous action-token window at the tail of the
    /// vocabulary. FAST action ids live in
    /// `[vocab_size - fast_skip_tokens - 1024, vocab_size - fast_skip_tokens)`;
    /// 2048 is the conservative superset the reference implementation prunes to.
    pub action_vocab_size: usize,
    /// Token ids outside the action window that the decode must still be able
    /// to emit. π0-FAST writes the deterministic prefix `"Action: "` before the
    /// action tokens and terminates on `"|"`, and none of those ids sit near the
    /// vocabulary tail, so a naive tail-only window would change both the first
    /// three steps and the terminator.
    pub action_head_extra_tokens: Vec<u32>,
    /// Prompt token budget (padded/truncated, `tokenizer_max_length`).
    pub max_token_len: usize,
    pub num_views: usize,
    /// Views declared in `input_features` but intentionally fed as -1 padding.
    pub empty_cameras: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub vision_width: usize,
    pub vision_depth: usize,
    pub vision_mlp_dim: usize,
    pub vision_heads: usize,
    pub vision_head_dim: usize,
    pub vocab_size: usize,
    pub image_token_index: usize,
    pub rms_norm_eps: f32,
    pub layer_norm_eps: f32,
    pub rope_theta: f32,
    pub language: Pi0FastLanguageConfig,
}

impl Default for Pi0FastConfig {
    fn default() -> Self {
        Self {
            action_dim: 7,
            action_horizon: 10,
            max_state_dim: 32,
            max_action_dim: 32,
            max_action_tokens: 256,
            fast_skip_tokens: 128,
            action_vocab_size: 2048,
            // PaliGemma ids, fixed by the π0-FAST prompt protocol:
            // 1 = <eos>, 4022 = "Action", 235248 = " ", 235292 = ":", 235371 = "|".
            action_head_extra_tokens: vec![1, 4022, 235_248, 235_292, 235_371],
            max_token_len: 200,
            num_views: 3,
            empty_cameras: 1,
            image_size: 224,
            patch_size: 14,
            vision_width: 1152,
            vision_depth: 27,
            vision_mlp_dim: 4304,
            vision_heads: 16,
            vision_head_dim: 72,
            vocab_size: 257_152,
            image_token_index: 257_152,
            rms_norm_eps: 1e-6,
            layer_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            language: Pi0FastLanguageConfig::GEMMA_2B,
        }
    }
}

impl Pi0FastConfig {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| Error::Other(format!("read {}: {e}", path.display())))?;
        Self::from_json_str(&raw)
    }

    pub fn from_json_str(raw: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(raw)
            .map_err(|e| Error::Other(format!("pi0fast config json: {e}")))?;
        let mut cfg = Self::default();

        if let Some(variant) = v.get("paligemma_variant").and_then(|x| x.as_str()) {
            cfg.language = Pi0FastLanguageConfig::from_variant(variant)?;
        }

        cfg.action_dim = action_dim(&v).unwrap_or(cfg.action_dim);
        cfg.action_horizon = usize_field(&v, &["n_action_steps", "chunk_size"], cfg.action_horizon);
        cfg.max_state_dim = usize_field(&v, &["max_state_dim"], cfg.max_state_dim);
        cfg.max_action_dim = usize_field(&v, &["max_action_dim"], cfg.max_action_dim);
        cfg.max_action_tokens = usize_field(&v, &["max_action_tokens"], cfg.max_action_tokens);
        cfg.fast_skip_tokens = usize_field(&v, &["fast_skip_tokens"], cfg.fast_skip_tokens);
        cfg.action_vocab_size =
            usize_field(&v, &["action_vocab_size"], cfg.action_vocab_size);
        if let Some(tokens) = v.get("action_head_extra_tokens").and_then(|x| x.as_array()) {
            cfg.action_head_extra_tokens = tokens
                .iter()
                .map(|token| {
                    token.as_u64().map(|value| value as u32).ok_or_else(|| {
                        Error::Other(format!(
                            "pi0fast action_head_extra_tokens entries must be integers, got {token}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
        }
        cfg.max_token_len = usize_field(
            &v,
            &["tokenizer_max_length", "max_token_len"],
            cfg.max_token_len,
        );
        cfg.empty_cameras = usize_field(&v, &["empty_cameras"], cfg.empty_cameras);

        if let Some(resolution) = v.get("image_resolution").and_then(|x| x.as_array()) {
            if let Some(size) = resolution.first().and_then(|x| x.as_u64()) {
                cfg.image_size = size as usize;
            }
        }

        cfg.num_views = resolve_num_views(&v, &cfg);
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn patches_per_view(&self) -> usize {
        let side = self.image_size / self.patch_size;
        side * side
    }

    /// Camera views the runtime actually consumes.
    ///
    /// LeRobot appends one all-`-1` view per missing camera and masks it out.
    /// The reference positions tokens with `cumsum(pad_mask) - 1`, so a masked
    /// view neither shifts any later position nor contributes a key: dropping it
    /// is numerically identical to carrying the padding, and it keeps the fixed
    /// patch shape as small as the sensor set.
    pub fn effective_views(&self) -> usize {
        self.num_views.saturating_sub(self.empty_cameras)
    }

    pub fn patch_tokens(&self) -> usize {
        self.effective_views() * self.patches_per_view()
    }

    /// Upper bound on the number of vocabulary columns the pruned LM head keeps,
    /// including the alignment padding `action_head_columns` appends.
    ///
    /// `validate` rejects configurations whose extras overlap the window, so at
    /// construction time this is exact; using the bound elsewhere keeps arena
    /// sizing infallible and only ever over-reserves.
    pub fn action_head_width(&self) -> usize {
        (self.action_vocab_size + self.action_head_extra_tokens.len())
            .next_multiple_of(ACTION_HEAD_COLUMN_ALIGN)
    }

    /// First vocabulary index of the contiguous action-token window.
    pub fn action_token_start(&self) -> usize {
        self.vocab_size - self.fast_skip_tokens - self.action_vocab_size
    }

    /// One past the last vocabulary index of the action-token window.
    pub fn action_token_end(&self) -> usize {
        self.vocab_size - self.fast_skip_tokens
    }

    /// Vocabulary columns the LM head keeps, in the order the pruned head emits
    /// them. Column `i` of the pruned head corresponds to token
    /// `action_head_columns()[i]`, which is exactly the table the argmax kernel
    /// remaps through.
    pub fn action_head_columns(&self) -> Result<Vec<usize>> {
        let (start, end) = (self.action_token_start(), self.action_token_end());
        let mut columns = Vec::with_capacity(self.action_vocab_size + self.action_head_extra_tokens.len());
        columns.extend(start..end);
        for &token in &self.action_head_extra_tokens {
            let index = token as usize;
            if index >= self.vocab_size {
                return Err(Error::Other(format!(
                    "pi0fast action_head_extra_tokens entry {token} is outside the \
                     {}-wide vocabulary",
                    self.vocab_size
                )));
            }
            // The action window is a contiguous range, so a duplicate can only
            // hide inside it or earlier in this (tiny) extra list.
            if (start..end).contains(&index) || columns[end - start..].contains(&index) {
                continue;
            }
            columns.push(index);
        }
        let padded = columns.len().next_multiple_of(ACTION_HEAD_COLUMN_ALIGN);
        let last = *columns.last().expect("the action window is never empty");
        columns.resize(padded, last);
        Ok(columns)
    }

    pub fn max_prefix_len(&self) -> usize {
        self.patch_tokens() + self.max_token_len + 1
    }

    pub fn performance_profile(&self) -> Pi0FastPerformanceProfile {
        Pi0FastPerformanceProfile {
            num_views: self.num_views,
            max_action_tokens: self.max_action_tokens,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.action_dim == 0 || self.action_dim > self.max_action_dim {
            return Err(Error::Other(format!(
                "pi0fast action_dim must be in 1..={}, got {}",
                self.max_action_dim, self.action_dim
            )));
        }
        if self.action_horizon == 0 {
            return Err(Error::Other("pi0fast action_horizon must be > 0".into()));
        }
        if self.max_action_tokens == 0 {
            return Err(Error::Other(
                "pi0fast max_action_tokens must be > 0".into(),
            ));
        }
        if self.action_vocab_size == 0 {
            return Err(Error::Other(
                "pi0fast action_vocab_size must be > 0".into(),
            ));
        }
        if self.fast_skip_tokens + self.action_vocab_size > self.vocab_size {
            return Err(Error::Other(format!(
                "pi0fast fast_skip_tokens {} + action_vocab_size {} exceeds the \
                 {}-wide vocabulary",
                self.fast_skip_tokens, self.action_vocab_size, self.vocab_size
            )));
        }
        self.action_head_columns()?;
        if self.num_views == 0 {
            return Err(Error::Other("pi0fast requires at least one view".into()));
        }
        if self.effective_views() == 0 {
            return Err(Error::Other(format!(
                "pi0fast declares only empty cameras ({}, {} padding)",
                self.num_views, self.empty_cameras
            )));
        }
        if self.image_size % self.patch_size != 0 {
            return Err(Error::Other(format!(
                "pi0fast image_size {} is not a multiple of patch_size {}",
                self.image_size, self.patch_size
            )));
        }
        if self.empty_cameras >= self.num_views {
            return Err(Error::Other(format!(
                "pi0fast empty_cameras {} must be less than num_views {}",
                self.empty_cameras, self.num_views
            )));
        }
        Ok(())
    }
}

fn usize_field(v: &serde_json::Value, keys: &[&str], fallback: usize) -> usize {
    keys.iter()
        .find_map(|key| v.get(*key).and_then(|x| x.as_u64()))
        .map(|x| x as usize)
        .unwrap_or(fallback)
}

/// The deployable action width lives in `output_features.action.shape[0]`; the
/// padded `max_action_dim` is not the FAST-token width.
fn action_dim(v: &serde_json::Value) -> Option<usize> {
    v.get("output_features")
        .and_then(|x| x.get("action"))
        .and_then(|x| x.get("shape"))
        .and_then(|x| x.as_array())
        .and_then(|shape| shape.first())
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
}

/// Count real camera views from `input_features` VISUAL entries. Empty cameras
/// are declared there too but are still consumed as padded views, so they are
/// counted; `empty_cameras` records how many of them are padding.
fn resolve_num_views(v: &serde_json::Value, cfg: &Pi0FastConfig) -> usize {
    let declared = v
        .get("input_features")
        .and_then(|x| x.as_object())
        .map(|features| {
            features
                .iter()
                .filter(|(key, value)| {
                    key.starts_with("observation.images.")
                        && value.get("type").and_then(|t| t.as_str()) == Some("VISUAL")
                })
                .count()
        })
        .unwrap_or(0);
    if declared > 0 {
        declared
    } else {
        cfg.num_views
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHECKPOINT: &str = r#"{
        "type": "pi0_fast",
        "input_features": {
            "observation.images.base_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
            "observation.images.left_wrist_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
            "observation.images.empty_camera_0": {"type": "VISUAL", "shape": [3, 224, 224]},
            "observation.state": {"type": "STATE", "shape": [32]}
        },
        "output_features": {"action": {"type": "ACTION", "shape": [7]}},
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "image_resolution": [224, 224],
        "empty_cameras": 1,
        "chunk_size": 10,
        "n_action_steps": 10,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "max_action_tokens": 256,
        "tokenizer_max_length": 200,
        "fast_skip_tokens": 128
    }"#;

    #[test]
    fn parses_lerobot_pi0_fast_config() {
        let cfg = Pi0FastConfig::from_json_str(CHECKPOINT).expect("parse");
        assert_eq!(cfg.action_dim, 7);
        assert_eq!(cfg.action_horizon, 10);
        assert_eq!(cfg.num_views, 3);
        assert_eq!(cfg.empty_cameras, 1);
        assert_eq!(cfg.max_action_tokens, 256);
        assert_eq!(cfg.fast_skip_tokens, 128);
        assert_eq!(cfg.language, Pi0FastLanguageConfig::GEMMA_2B);
        assert_eq!(cfg.patches_per_view(), 256);
        assert_eq!(cfg.effective_views(), 2);
        assert_eq!(cfg.patch_tokens(), 512);
        cfg.validate().expect("valid");
    }

    #[test]
    fn action_head_columns_keep_the_reachable_ids() {
        let cfg = Pi0FastConfig::from_json_str(CHECKPOINT).expect("parse");
        let columns = cfg.action_head_columns().expect("columns");
        let window = cfg.action_vocab_size;

        assert_eq!(columns.len(), cfg.action_head_width());
        assert_eq!(columns.len() % ACTION_HEAD_COLUMN_ALIGN, 0);
        assert_eq!(columns.len(), 2112);

        // The action window stays contiguous and in place.
        assert_eq!(columns[0], 257_152 - 128 - 2048);
        assert_eq!(columns[window - 1], 257_023);

        // Ids pi0-FAST emits outside the window must survive pruning: the
        // "Action: " prefix (4022, 235292, 235248), the "|" terminator
        // (235371) and <eos> (1).
        for token in [1usize, 4022, 235_248, 235_292, 235_371] {
            assert!(columns.contains(&token), "token {token} was pruned");
        }

        // Padding repeats the last retained id so it can never change an argmax.
        let last = columns[window + 4];
        assert!(columns[window + 5..].iter().all(|column| *column == last));
    }

    #[test]
    fn action_window_must_fit_the_vocabulary() {
        let raw = CHECKPOINT.replace("\"max_action_tokens\": 256", "\"max_action_tokens\": 256, \"action_vocab_size\": 257152");
        assert!(Pi0FastConfig::from_json_str(&raw).is_err());
    }

    #[test]
    fn empty_camera_must_leave_a_real_view() {
        let raw = CHECKPOINT.replace("\"empty_cameras\": 1", "\"empty_cameras\": 3");
        assert!(Pi0FastConfig::from_json_str(&raw).is_err());
    }
}
