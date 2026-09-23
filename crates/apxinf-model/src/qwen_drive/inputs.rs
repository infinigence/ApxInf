//! Canonical planning conditioning, independent of model execution.
use super::config::QwenDriveConfig;
use apxinf_core::{Error, Result};

/// Conditioning that the processor produces once per planning request.
pub struct ExpertConditioning {
    /// Normalized history poses re-referenced to the oldest pose, flattened
    /// `[num_history_points - 1, 3]` (the dropped origin row is implied).
    pub history: Vec<f32>,
    /// Raw history velocity `[num_history_points, 2]`, flattened.
    pub history_velocity: Vec<f32>,
    /// Raw history acceleration `[num_history_points, 2]`, flattened.
    pub history_acceleration: Vec<f32>,
    /// Navigation command index; out-of-range maps to an all-zero one-hot,
    /// exactly like the reference `_one_hot`.
    pub nav_command: i64,
    /// `[ego_status_dim]` velocity + acceleration + one-hot driving command.
    pub ego_status: Vec<f32>,
}

impl ExpertConditioning {
    pub fn validate(&self, config: &QwenDriveConfig) -> Result<()> {
        let poses = (config.num_history_points - 1) * config.trajectory_point_dim;
        if self.history.len() != poses {
            return Err(Error::Other(format!(
                "qwen_drive planner: history has {} values, expected {poses}",
                self.history.len()
            )));
        }
        let dynamics = config.num_history_points * config.expert.history_dynamics_dim;
        if self.history_velocity.len() != dynamics {
            return Err(Error::Other(format!(
                "qwen_drive planner: history_velocity has {} values, expected {dynamics}",
                self.history_velocity.len()
            )));
        }
        if self.history_acceleration.len() != dynamics {
            return Err(Error::Other(format!(
                "qwen_drive planner: history_acceleration has {} values, expected {dynamics}",
                self.history_acceleration.len()
            )));
        }
        if self.ego_status.len() != config.expert.ego_status_dim {
            return Err(Error::Other(format!(
                "qwen_drive planner: ego_status has {} values, expected {}",
                self.ego_status.len(),
                config.expert.ego_status_dim
            )));
        }
        Ok(())
    }
}

impl ExpertConditioning {
    /// Policy wire layout: history poses, velocity, acceleration, ego status,
    /// then one exactly represented navigation-command integer.
    pub fn from_packed(config: &QwenDriveConfig, values: &[f32]) -> Result<Self> {
        let poses = (config.num_history_points - 1) * config.trajectory_point_dim;
        let dynamics = config.num_history_points * config.expert.history_dynamics_dim;
        let ego = config.expert.ego_status_dim;
        let expected = poses + 2 * dynamics + ego + 1;
        if values.len() != expected || values.iter().any(|x| !x.is_finite()) {
            return Err(Error::Other(format!(
                "qwen_drive conditioning requires {expected} finite values"
            )));
        }
        let nav = values[expected - 1];
        if nav.fract() != 0.0 || nav.abs() > 16777216.0 {
            return Err(Error::Other(
                "qwen_drive navigation command must be an exact integer".into(),
            ));
        }
        let result = Self {
            history: values[..poses].to_vec(),
            history_velocity: values[poses..poses + dynamics].to_vec(),
            history_acceleration: values[poses + dynamics..poses + 2 * dynamics].to_vec(),
            ego_status: values[poses + 2 * dynamics..expected - 1].to_vec(),
            nav_command: nav as i64,
        };
        result.validate(config)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_conditioning_has_exact_sections_and_rejects_invalid_values() {
        let config =
            QwenDriveConfig::from_json_str(super::super::config::tests::CHECKPOINT_CONFIG).unwrap();
        let poses = (config.num_history_points - 1) * config.trajectory_point_dim;
        let dynamics = config.num_history_points * config.expert.history_dynamics_dim;
        let count = poses + 2 * dynamics + config.expert.ego_status_dim + 1;
        let mut values: Vec<f32> = (0..count).map(|n| n as f32).collect();
        values[count - 1] = 2.0;
        let input = ExpertConditioning::from_packed(&config, &values).unwrap();
        assert_eq!(input.history.len(), poses);
        assert_eq!(input.history_velocity[0], poses as f32);
        assert_eq!(input.history_acceleration[0], (poses + dynamics) as f32);
        assert_eq!(input.ego_status[0], (poses + 2 * dynamics) as f32);
        assert_eq!(input.nav_command, 2);
        assert!(ExpertConditioning::from_packed(&config, &values[..count - 1]).is_err());
        values[count - 1] = 0.5;
        assert!(ExpertConditioning::from_packed(&config, &values).is_err());
        values[count - 1] = 2.0;
        values[0] = f32::NAN;
        assert!(ExpertConditioning::from_packed(&config, &values).is_err());
    }
}
