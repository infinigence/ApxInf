use apxinf_core::{Error, Result};
use serde::Deserialize;
use std::path::Path;

pub const PROFILE: &str = "droid-policy-v1";
/// Edge-LLM source the export/architecture contract was read from.
pub const SOURCE_REVISION: &str = "a16b2d162bb12245f98ebef4227a1d7fe9c67757";
/// HF checkpoint revision of `nvidia/Cosmos3-Edge-Policy-DROID`.
pub const CHECKPOINT_REVISION: &str = "0110e64ca982589f1a3d856cbdd7d465dccb79bc";

/// Explicit supported geometry, read from the verified asset conversion.
/// Unknown profiles are rejected rather than interpreted using defaults.
///
/// Cosmos3-Edge is a Mixture-of-Transformers: each of the 28 layers carries an
/// UND (language/understanding) tower and a GEN (diffusion) tower. The policy
/// path runs the UND tower once as a K/V prefill, then the GEN tower for the
/// UniPC flow-matching denoise steps.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cosmos3EdgeConfig {
    pub model_type: String,
    pub profile: String,
    pub format_version: u32,
    pub source_revision: String,
    pub checkpoint_revision: String,
    pub precision: String,
    pub num_tasks: usize,

    // Transformer geometry.
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub mrope_section: [usize; 3],

    // Latent / action geometry.
    pub latent_channel: usize,
    pub latent_patch_size: usize,
    /// `[C, T, H, W]` of the condition latent (frame-0 observation).
    pub latent_shape: [usize; 4],
    pub action_chunk: usize,
    pub raw_action_dim: usize,
    pub max_action_dim: usize,

    // Flow-matching / UniPC scheduler.
    pub num_inference_steps: usize,
    pub flow_shift: f64,
    pub num_train_timesteps: usize,

    // Embodiment / text / pixels.
    pub num_embodiment_domains: usize,
    pub domain_id: usize,
    pub vocab_size: usize,
    pub max_text_tokens: usize,
    pub image_size: [usize; 2],
    pub pixel_frames: usize,
}

impl Cosmos3EdgeConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| Error::Other(format!("read {}: {e}", path.display())))?;
        let config: Self = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Other(format!("parse {}: {e}", path.display())))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != "cosmos3-edge"
            || self.profile != PROFILE
            || self.format_version != 1
            || self.source_revision != SOURCE_REVISION
            || self.checkpoint_revision != CHECKPOINT_REVISION
            || self.precision != "bf16"
            || self.num_tasks != 1
            || self.hidden_size != 2048
            || self.num_layers != 28
            || self.num_heads != 16
            || self.num_kv_heads != 8
            || self.head_dim != 128
            || self.intermediate_size != 9216
            || self.mrope_section != [24, 20, 20]
            || self.latent_channel != 48
            || self.latent_patch_size != 2
            || self.latent_shape != [48, 5, 34, 46]
            || self.action_chunk != 16
            || self.max_action_dim != 64
            || self.raw_action_dim != 10
            || self.num_inference_steps != 4
            || (self.flow_shift - 5.0).abs() > f64::EPSILON
            || self.num_train_timesteps != 1000
            || self.num_embodiment_domains != 32
            || self.image_size != [544, 736]
            || !matches!(self.pixel_frames, 1 | 17)
            || self.domain_id != 8
            || self.vocab_size != 131072
            || self.max_text_tokens != 126
            || self.rms_norm_eps != 1e-5
            || self.rope_theta != 1e8
        {
            return Err(Error::Other(
                "unsupported Cosmos3-Edge checkpoint/profile; prepare the pinned DROID policy assets".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Cosmos3EdgeConfig {
        Cosmos3EdgeConfig {
            model_type: "cosmos3-edge".into(),
            profile: PROFILE.into(),
            format_version: 1,
            source_revision: SOURCE_REVISION.into(),
            checkpoint_revision: CHECKPOINT_REVISION.into(),
            precision: "bf16".into(),
            num_tasks: 1,
            hidden_size: 2048,
            num_layers: 28,
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 9216,
            rms_norm_eps: 1e-5,
            rope_theta: 1e8,
            mrope_section: [24, 20, 20],
            latent_channel: 48,
            latent_patch_size: 2,
            latent_shape: [48, 5, 34, 46],
            action_chunk: 16,
            raw_action_dim: 10,
            max_action_dim: 64,
            num_inference_steps: 4,
            flow_shift: 5.0,
            num_train_timesteps: 1000,
            num_embodiment_domains: 32,
            domain_id: 8,
            vocab_size: 131072,
            max_text_tokens: 126,
            image_size: [544, 736],
            pixel_frames: 17,
        }
    }

    #[test]
    fn accepts_the_canonical_profile() {
        config().validate().unwrap();
    }

    #[test]
    fn rejects_altered_geometry_and_sampling() {
        for mutate in [
            |c: &mut Cosmos3EdgeConfig| c.num_layers = 27,
            |c: &mut Cosmos3EdgeConfig| c.num_kv_heads = 16,
            |c: &mut Cosmos3EdgeConfig| c.mrope_section = [24, 24, 16],
            |c: &mut Cosmos3EdgeConfig| c.latent_shape = [48, 5, 17, 23],
            |c: &mut Cosmos3EdgeConfig| c.num_inference_steps = 8,
            |c: &mut Cosmos3EdgeConfig| c.flow_shift = 1.0,
            |c: &mut Cosmos3EdgeConfig| c.checkpoint_revision = "unverified".into(),
            |c: &mut Cosmos3EdgeConfig| c.precision = "fp8".into(),
            |c: &mut Cosmos3EdgeConfig| c.profile = "libero".into(),
            |c: &mut Cosmos3EdgeConfig| c.domain_id = 0,
            |c: &mut Cosmos3EdgeConfig| c.max_text_tokens = 125,
            |c: &mut Cosmos3EdgeConfig| c.rms_norm_eps = f64::NAN,
            |c: &mut Cosmos3EdgeConfig| c.precision = "fp16".into(),
        ] {
            let mut c = config();
            mutate(&mut c);
            assert!(c.validate().is_err());
        }
    }
}
