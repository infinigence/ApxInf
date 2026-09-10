use apxinf_core::{DType, Error, Result, Tensor};

use super::Gr00tConfig;

/// Fully preprocessed input for one GR00T N1.7 inference.
///
/// This deliberately remains separate from [`crate::vla::Observation`]. GR00T
/// requires proprioception, an embodiment selector, and Qwen3-VL patch-grid
/// metadata that the existing PI0.5-oriented contract cannot represent.
#[derive(Clone, Debug)]
pub struct Gr00tObservation {
    /// Qwen3-VL visual input produced by the checkpoint processor.
    pub pixel_values: Tensor,
    /// One `(temporal, height, width)` triple for every input image.
    pub image_grid_thw: Vec<[u32; 3]>,
    /// Tokenized language and image-placeholder sequence, shaped `[batch, seq]`
    /// logically. The first implementation supports batch size one.
    pub token_ids: Vec<u32>,
    /// One entry per token. Non-zero entries participate in backbone attention.
    pub attention_mask: Vec<u8>,
    /// Normalized, zero-padded state tensor `[1, history, max_state_dim]`.
    pub state: Tensor,
    /// Selects one of the checkpoint's category-specific state/action MLPs.
    pub embodiment_id: usize,
    /// Initial Gaussian flow noise `[1, action_horizon, max_action_dim]`.
    pub noise: Tensor,
}

impl Gr00tObservation {
    pub fn inference_spec(&self, config: &Gr00tConfig) -> Result<Gr00tInferenceSpec> {
        self.validate(config)?;
        Ok(Gr00tInferenceSpec {
            batch_size: 1,
            token_count: self.token_ids.len(),
            image_grid_thw: self.image_grid_thw.clone(),
            state_history_length: config.state_history_length,
            state_dim: config.max_state_dim,
            action_horizon: config.action_horizon,
            action_dim: config.max_action_dim,
        })
    }

    pub fn validate(&self, config: &Gr00tConfig) -> Result<()> {
        config.validate()?;
        if self.token_ids.is_empty() {
            return Err(Error::Other(
                "GR00T observation requires at least one token".into(),
            ));
        }
        if self.token_ids.len() != self.attention_mask.len() {
            return Err(Error::Other(format!(
                "GR00T token/mask length mismatch: {} tokens, {} mask entries",
                self.token_ids.len(),
                self.attention_mask.len()
            )));
        }
        if let Some((index, value)) = self
            .attention_mask
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| *value > 1)
        {
            return Err(Error::Other(format!(
                "GR00T attention_mask[{index}] must be 0 or 1, got {value}"
            )));
        }
        if self.image_grid_thw.is_empty() {
            return Err(Error::Other(
                "GR00T observation requires at least one image grid".into(),
            ));
        }
        for (index, grid) in self.image_grid_thw.iter().enumerate() {
            if grid.contains(&0) {
                return Err(Error::Other(format!(
                    "GR00T image_grid_thw[{index}] must contain non-zero dimensions, got {grid:?}"
                )));
            }
        }
        if self.embodiment_id >= config.max_num_embodiments {
            return Err(Error::Other(format!(
                "GR00T embodiment_id {} is outside 0..{}",
                self.embodiment_id, config.max_num_embodiments
            )));
        }
        expect_bf16_shape(
            "state",
            &self.state,
            &[1, config.state_history_length, config.max_state_dim],
        )?;
        expect_bf16_shape(
            "noise",
            &self.noise,
            &[1, config.action_horizon, config.max_action_dim],
        )?;
        if self.pixel_values.dtype() != DType::BF16 {
            return Err(Error::Other(format!(
                "GR00T pixel_values must be BF16 after preprocessing, got {}",
                self.pixel_values.dtype()
            )));
        }
        if self.pixel_values.numel() == 0 {
            return Err(Error::Other("GR00T pixel_values must not be empty".into()));
        }
        Ok(())
    }
}

/// Fixed-shape contract for a prepared GR00T inference.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Gr00tInferenceSpec {
    pub batch_size: usize,
    pub token_count: usize,
    pub image_grid_thw: Vec<[u32; 3]>,
    pub state_history_length: usize,
    pub state_dim: usize,
    pub action_horizon: usize,
    pub action_dim: usize,
}

impl Gr00tInferenceSpec {
    pub fn validate(&self, config: &Gr00tConfig) -> Result<()> {
        if self.batch_size != 1 {
            return Err(Error::Other(format!(
                "GR00T N1.7 first-stage runtime supports batch size one, got {}",
                self.batch_size
            )));
        }
        if self.token_count == 0 {
            return Err(Error::Other(
                "GR00T inference spec requires at least one token".into(),
            ));
        }
        if self.image_grid_thw.is_empty() {
            return Err(Error::Other(
                "GR00T inference spec requires at least one image grid".into(),
            ));
        }
        let expected = (
            config.state_history_length,
            config.max_state_dim,
            config.action_horizon,
            config.max_action_dim,
        );
        let actual = (
            self.state_history_length,
            self.state_dim,
            self.action_horizon,
            self.action_dim,
        );
        if actual != expected {
            return Err(Error::Other(format!(
                "GR00T inference shape mismatch: expected {expected:?}, got {actual:?}"
            )));
        }
        Ok(())
    }

    pub fn matches(&self, observation: &Gr00tObservation, config: &Gr00tConfig) -> bool {
        observation
            .inference_spec(config)
            .is_ok_and(|spec| spec == *self)
    }
}

fn expect_bf16_shape(name: &str, tensor: &Tensor, expected: &[usize]) -> Result<()> {
    if tensor.dtype() != DType::BF16 || tensor.shape().dims() != expected {
        return Err(Error::Other(format!(
            "GR00T {name} must be BF16 {expected:?}, got {} {:?}",
            tensor.dtype(),
            tensor.shape().dims()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;

    fn valid_observation(config: &Gr00tConfig) -> Gr00tObservation {
        Gr00tObservation {
            pixel_values: Tensor::from_bf16(vec![2, 4], &[bf16::ZERO; 8]).unwrap(),
            image_grid_thw: vec![[1, 16, 16]],
            token_ids: vec![1, 2, 3],
            attention_mask: vec![1, 1, 1],
            state: Tensor::zeros(
                vec![1, config.state_history_length, config.max_state_dim],
                DType::BF16,
            ),
            embodiment_id: 0,
            noise: Tensor::zeros(
                vec![1, config.action_horizon, config.max_action_dim],
                DType::BF16,
            ),
        }
    }

    #[test]
    fn validates_released_n1d7_input_contract() {
        let config = Gr00tConfig::default();
        let observation = valid_observation(&config);
        let spec = observation.inference_spec(&config).unwrap();
        assert_eq!(spec.batch_size, 1);
        assert_eq!(spec.token_count, 3);
        assert_eq!(spec.action_horizon, 40);
        assert!(spec.matches(&observation, &config));
    }

    #[test]
    fn rejects_invalid_embodiment_and_noise_shape() {
        let config = Gr00tConfig::default();
        let mut observation = valid_observation(&config);
        observation.embodiment_id = config.max_num_embodiments;
        assert!(observation.validate(&config).is_err());

        observation.embodiment_id = 0;
        observation.noise = Tensor::zeros(vec![40, 132], DType::BF16);
        assert!(observation.validate(&config).is_err());
    }
}
