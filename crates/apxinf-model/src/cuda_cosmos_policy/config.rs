use apxinf_core::{Error, Result};
use serde::Deserialize;
use std::path::Path;

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
            || self.num_tasks != 40
            || self.text_shape != [512, 1024]
            || self.initial_latent_shape != [1, 16, 9, 28, 28]
            || self.action_shape != [16, 7]
            || self.num_views != 2
            || self.image_size != 224
            || self.state_dim != 9
            || self.denoiser_calls != 5
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
