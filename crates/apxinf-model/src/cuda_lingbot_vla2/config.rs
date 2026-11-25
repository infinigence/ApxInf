//! LingBot-VLA 2.0 configuration.
//!
//! The published checkpoint ships `config.json = {"vlm_family": "qwen3_vl"}` and no architecture
//! fields: the geometry lives in the training config (`lingbotvla_cli.yaml`), which the
//! pre-trained repo does not include. The values below are therefore the architecture of
//! `robbyant/lingbot-vla-v2-6b`, and `validate_checkpoint` proves them against the tensor shapes
//! of whatever checkpoint is loaded instead of trusting them.

use apxinf_core::{Error, Result};

/// Qwen3-VL-4B text decoder, shared by the vision-language prefix.
pub const VLM_LAYERS: usize = 36;
pub const VLM_HIDDEN: usize = 2560;
pub const VLM_FFN: usize = 9728;
pub const VLM_HEADS: usize = 32;
pub const VLM_KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const VLM_VOCAB: usize = 151_936;
pub const VLM_EPS: f32 = 1e-6;
pub const VLM_ROPE_THETA: f64 = 5_000_000.0;
pub const MROPE_SECTIONS: [usize; 3] = [24, 20, 20];

/// Qwen3-VL ViT.
pub const VIT_LAYERS: usize = 24;
pub const VIT_HIDDEN: usize = 1024;
pub const VIT_FFN: usize = 4096;
pub const VIT_HEADS: usize = 16;
pub const VIT_PATCH: usize = 16;
pub const VIT_TEMPORAL_PATCH: usize = 2;
pub const VIT_MERGE: usize = 2;
pub const VIT_PATCH_DIM: usize = 3 * VIT_TEMPORAL_PATCH * VIT_PATCH * VIT_PATCH; // 1536
pub const VIT_POS_GRID: usize = 48;
pub const VIT_EPS: f32 = 1e-6;
pub const VIT_ROPE_THETA: f32 = 10_000.0;
pub const DEEPSTACK_TAPS: [usize; 3] = [5, 11, 17];

/// Action expert (Qwen2-style, joint attention with the VLM at every layer).
pub const EXPERT_HIDDEN: usize = 768;
pub const EXPERT_HEADS: usize = 32;
pub const EXPERT_KV_HEADS: usize = 8;
pub const EXPERT_EPS: f32 = 1e-6;
// The expert has no rotary embedding of its own: upstream applies the VLM's rotary embedding (theta 5e6) to the
// expert's Q/K with the suffix positions, so the expert uses `VLM_ROPE_THETA`.

/// Token-choice MoE inside every expert layer.
pub const MOE_EXPERTS: usize = 32;
pub const MOE_TOP_K: usize = 4;
pub const MOE_FFN: usize = 512;
pub const MOE_SHARED_FFN: usize = 704;
pub const MOE_ROUTED_SCALE: f32 = 4.0;

/// Unified action space and flow schedule.
pub const STATE_DIM: usize = 55;
pub const ACTION_DIM: usize = 55;
pub const ACTION_HORIZON: usize = 50;
pub const FLOW_STEPS: usize = 10;
pub const TIME_MIN_PERIOD: f32 = 4e-3;
pub const TIME_MAX_PERIOD: f32 = 4.0;

/// Learned query tokens appended to the prefix: current queries then future queries.
pub const QUERY_TABLE_ROWS: usize = 256;
pub const TASK_TOKENS: usize = 8;
pub const PREFIX_QUERY_TOKENS: usize = 2 * TASK_TOKENS;

/// Prompt cap used by training (`tokenizer_max_length`). Right padding is removed before
/// execution, so this is only an upper bound on valid tokens.
pub const MAX_PROMPT_TOKENS: usize = 72;

/// Special token ids of the Qwen3-VL tokenizer.
pub const VISION_START_ID: u32 = 151_652;
pub const VISION_END_ID: u32 = 151_653;

/// Per-view geometry of the published checkpoint: 256x256 images, patch 16 -> 16x16 grid.
pub const NUM_VIEWS: usize = 3;
pub const IMAGE_SIZE: usize = 256;
pub const GRID: u32 = (IMAGE_SIZE / VIT_PATCH) as u32;
pub const PATCHES_PER_VIEW: usize = (GRID * GRID) as usize; // 256 ViT rows
pub const MERGED_PER_VIEW: usize = PATCHES_PER_VIEW / (VIT_MERGE * VIT_MERGE); // 64 LLM tokens
pub const IMAGE_TOKENS_PER_VIEW: usize = MERGED_PER_VIEW + 2; // + vision_start / vision_end

/// State and action vectors are zero-padded from 55 to 64 so every GEMM sees an 8-aligned K or N.
/// Padded input columns meet zero weight rows and padded output columns come from zero weight rows and
/// biases, so the padding never changes a result.
pub const PADDED_DIM: usize = 64;

/// Suffix: one state token followed by the action tokens.
pub const SUFFIX_TOKENS: usize = 1 + ACTION_HORIZON;

/// Total prefix length for a prompt of `prompt_tokens` valid tokens.
pub const fn prefix_len(prompt_tokens: usize) -> usize {
    NUM_VIEWS * IMAGE_TOKENS_PER_VIEW + prompt_tokens + PREFIX_QUERY_TOKENS
}

#[derive(Clone, Debug)]
pub struct LingbotVla2Config {
    pub flow_steps: usize,
    pub views: usize,
}

impl Default for LingbotVla2Config {
    fn default() -> Self {
        Self {
            flow_steps: FLOW_STEPS,
            views: NUM_VIEWS,
        }
    }
}

impl LingbotVla2Config {
    pub fn validate(&self) -> Result<()> {
        if self.flow_steps == 0 || self.flow_steps > 64 {
            return Err(Error::Other(format!(
                "LingBot-VLA 2.0 flow_steps must be in 1..=64, got {}",
                self.flow_steps
            )));
        }
        if self.views != NUM_VIEWS {
            return Err(Error::Other(format!(
                "LingBot-VLA 2.0 supports exactly {NUM_VIEWS} camera views, got {}",
                self.views
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_geometry_matches_the_reference_trace() {
        // golden_a: 3 views x 66 tokens + 16 prompt tokens + 16 queries = 230 valid prefix rows
        assert_eq!(IMAGE_TOKENS_PER_VIEW, 66);
        assert_eq!(prefix_len(16), 230);
        // padded reference prefix is 286 = 198 + 72 + 16
        assert_eq!(
            NUM_VIEWS * IMAGE_TOKENS_PER_VIEW + MAX_PROMPT_TOKENS + PREFIX_QUERY_TOKENS,
            286
        );
        assert_eq!(SUFFIX_TOKENS, 51);
    }

    #[test]
    fn rejects_unsupported_view_counts_and_steps() {
        assert!(LingbotVla2Config::default().validate().is_ok());
        assert!(LingbotVla2Config {
            views: 2,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(LingbotVla2Config {
            flow_steps: 0,
            ..Default::default()
        }
        .validate()
        .is_err());
    }
}
