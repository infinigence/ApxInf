use apxinf_core::{Error, Result};
use serde::Deserialize;
use std::path::Path;

/// The single source of truth for the pinned LIBERO direct profile geometry.
/// `config.rs` validates the prepared assets against these values, the runner
/// builds its public `VlaContract` from them, and the blocks allocate from
/// them, so the numbers are not restated per module.
pub(crate) const NUM_VIEWS: usize = 2;
pub(crate) const IMAGE_SIZE: usize = 224;
pub(crate) const STATE_DIM: usize = 9;
pub(crate) const DENOISER_CALLS: usize = 5;
pub(crate) const ACTION_HORIZON: usize = 16;
pub(crate) const ACTION_DIM: usize = 7;
pub(crate) const LATENT_FRAMES: usize = 9;
pub(crate) const LATENT_SIDE: usize = 28;
pub(crate) const LATENT_CHANNELS: usize = 16;
pub(crate) const TEXT_TOKENS: usize = 512;
pub(crate) const TEXT_WIDTH: usize = 1024;
pub(crate) const NUM_TASKS: usize = 40;

/// `[1, 16, 9, 28, 28]`: the complete joint latent the sampler consumes.
pub(crate) const INITIAL_LATENT_SHAPE: [usize; 5] =
    [1, LATENT_CHANNELS, LATENT_FRAMES, LATENT_SIDE, LATENT_SIDE];
/// `[16, 7]`: the public action chunk, independent of the latent geometry.
pub(crate) const ACTION_SHAPE: [usize; 2] = [ACTION_HORIZON, ACTION_DIM];
pub const PROFILE: &str = "libero-2b-direct-v1";
pub const SOURCE_REVISION: &str = "18a2accadf4e7a3531e56754102af5a24d2316da";
pub const CHECKPOINT_REVISION: &str = "cb689ec0e3347c13667d70a78a3447388f5c3bb8";

/// Explicit supported geometry, read from the verified asset conversion.
/// Unknown profiles are rejected rather than interpreted using LIBERO defaults.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CosmosPolicyConfig {
    pub model_type: String,
    pub profile: String,
    pub format_version: u32,
    pub source_revision: String,
    pub checkpoint_revision: String,
    pub precision: String,
    pub num_tasks: usize,
    pub text_shape: [usize; 2],
    pub initial_latent_shape: [usize; 5],
    pub action_shape: [usize; 2],
    pub num_views: usize,
    pub image_size: usize,
    pub state_dim: usize,
    pub denoiser_calls: usize,
    pub scaling: String,
    pub solver: String,
}

impl CosmosPolicyConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| Error::Other(format!("read {}: {e}", path.display())))?;
        let config: Self = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Other(format!("parse {}: {e}", path.display())))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != "cosmos-policy"
            || self.profile != PROFILE
            || self.format_version != 1
            || self.source_revision != SOURCE_REVISION
            || self.checkpoint_revision != CHECKPOINT_REVISION
            || self.precision != "bf16"
            || self.num_tasks != NUM_TASKS
            || self.text_shape != [TEXT_TOKENS, TEXT_WIDTH]
            || self.initial_latent_shape != INITIAL_LATENT_SHAPE
            || self.action_shape != ACTION_SHAPE
            || self.num_views != NUM_VIEWS
            || self.image_size != IMAGE_SIZE
            || self.state_dim != STATE_DIM
            || self.denoiser_calls != DENOISER_CALLS
            || self.scaling != "rectified_flow"
            || self.solver != "2ab"
        {
            return Err(Error::Other("unsupported Cosmos Policy checkpoint/profile; prepare the pinned LIBERO direct assets".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> CosmosPolicyConfig {
        CosmosPolicyConfig {
            model_type: "cosmos-policy".into(),
            profile: PROFILE.into(),
            format_version: 1,
            source_revision: SOURCE_REVISION.into(),
            checkpoint_revision: CHECKPOINT_REVISION.into(),
            precision: "bf16".into(),
            num_tasks: 40,
            text_shape: [512, 1024],
            initial_latent_shape: [1, 16, 9, 28, 28],
            action_shape: [16, 7],
            num_views: 2,
            image_size: 224,
            state_dim: 9,
            denoiser_calls: 5,
            scaling: "rectified_flow".into(),
            solver: "2ab".into(),
        }
    }

    #[test]
    fn profile_rejects_action_only_noise_and_altered_sampling() {
        config().validate().unwrap();
        let mut c = config();
        c.initial_latent_shape = [1, 1, 1, 16, 7];
        assert!(c.validate().is_err());
        c = config();
        c.scaling = "edm".into();
        assert!(c.validate().is_err());
        c = config();
        c.denoiser_calls = 10;
        assert!(c.validate().is_err());
        c = config();
        c.checkpoint_revision = "unverified".into();
        assert!(c.validate().is_err());
        c = config();
        c.profile = "aloha".into();
        assert!(c.validate().is_err());
    }
}
