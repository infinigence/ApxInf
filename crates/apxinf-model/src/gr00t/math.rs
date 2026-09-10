use std::f64::consts::LN_10;

use apxinf_core::{Error, Result};

/// One inference point in GR00T's forward-time Euler flow schedule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gr00tFlowStep {
    pub index: usize,
    pub continuous_time: f32,
    pub discrete_timestep: usize,
    pub delta: f32,
}

/// Match GR00T N1.7 inference: `t = step / steps`, then truncate
/// `t * timestep_buckets` to an integer.
pub fn flow_schedule(steps: usize, timestep_buckets: usize) -> Result<Vec<Gr00tFlowStep>> {
    if steps == 0 {
        return Err(Error::Other(
            "GR00T flow schedule requires at least one step".into(),
        ));
    }
    if timestep_buckets == 0 {
        return Err(Error::Other(
            "GR00T flow schedule requires timestep buckets".into(),
        ));
    }
    let delta = 1.0 / steps as f32;
    Ok((0..steps)
        .map(|index| {
            let continuous_time = index as f32 / steps as f32;
            Gr00tFlowStep {
                index,
                continuous_time,
                // Python computes this using its f64 `float` before `int`
                // truncation. Do the same so non-power-of-two step counts do
                // not drift by one bucket through an f32 intermediate.
                discrete_timestep: ((index as f64 / steps as f64) * timestep_buckets as f64)
                    as usize,
                delta,
            }
        })
        .collect())
}

/// Sin/cos embedding used by the multi-embodiment action encoder.
///
/// Frequencies are `exp(-i * ln(10000) / half_dim)`, followed by all sine
/// channels and then all cosine channels.
pub fn action_timestep_embedding(timestep: usize, dimension: usize) -> Result<Vec<f32>> {
    sinusoidal_embedding(timestep, dimension, false, 0)
}

/// Diffusers `Timesteps(256, flip_sin_to_cos=true,
/// downscale_freq_shift=1)` projection used before GR00T's learned timestep
/// MLP. The returned vector is FP32, matching the reference projection.
pub fn dit_timestep_projection(timestep: usize, dimension: usize) -> Result<Vec<f32>> {
    sinusoidal_embedding(timestep, dimension, true, 1)
}

fn sinusoidal_embedding(
    timestep: usize,
    dimension: usize,
    flip_sin_to_cos: bool,
    downscale_freq_shift: usize,
) -> Result<Vec<f32>> {
    if dimension == 0 || dimension % 2 != 0 {
        return Err(Error::Other(format!(
            "GR00T sinusoidal embedding dimension must be positive and even, got {dimension}"
        )));
    }
    let half = dimension / 2;
    let denominator = half.checked_sub(downscale_freq_shift).ok_or_else(|| {
        Error::Other(format!(
            "GR00T timestep embedding dimension {dimension} is too small"
        ))
    })?;
    if denominator == 0 {
        return Err(Error::Other(format!(
            "GR00T timestep embedding dimension {dimension} is too small"
        )));
    }

    let mut sin = Vec::with_capacity(half);
    let mut cos = Vec::with_capacity(half);
    let log_max_period = 4.0 * LN_10;
    for index in 0..half {
        let frequency = (-(index as f64) * log_max_period / denominator as f64).exp();
        let phase = timestep as f64 * frequency;
        sin.push(phase.sin() as f32);
        cos.push(phase.cos() as f32);
    }
    if flip_sin_to_cos {
        cos.extend(sin);
        Ok(cos)
    } else {
        sin.extend(cos);
        Ok(sin)
    }
}

/// Apply one GR00T forward-time Euler update: `actions += dt * velocity`.
pub fn euler_flow_step(actions: &mut [f32], velocity: &[f32], delta: f32) -> Result<()> {
    if actions.len() != velocity.len() {
        return Err(Error::Other(format!(
            "GR00T Euler update length mismatch: {} actions, {} velocities",
            actions.len(),
            velocity.len()
        )));
    }
    if !delta.is_finite() || delta <= 0.0 {
        return Err(Error::Other(format!(
            "GR00T Euler delta must be positive and finite, got {delta}"
        )));
    }
    for (action, &velocity) in actions.iter_mut().zip(velocity) {
        *action += delta * velocity;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn four_step_schedule_matches_n1d7_inference() {
        let schedule = flow_schedule(4, 1000).unwrap();
        assert_eq!(
            schedule
                .iter()
                .map(|step| step.discrete_timestep)
                .collect::<Vec<_>>(),
            vec![0, 250, 500, 750]
        );
        assert!(schedule.iter().all(|step| step.delta == 0.25));
    }

    #[test]
    fn action_embedding_uses_sin_then_cos() {
        let embedding = action_timestep_embedding(0, 8).unwrap();
        assert_eq!(&embedding[..4], &[0.0; 4]);
        assert_eq!(&embedding[4..], &[1.0; 4]);

        let embedding = action_timestep_embedding(1, 8).unwrap();
        assert_close(embedding[0], 1.0f32.sin());
        assert_close(embedding[4], 1.0f32.cos());
    }

    #[test]
    fn dit_projection_flips_cosine_before_sine() {
        let projection = dit_timestep_projection(0, 8).unwrap();
        assert_eq!(&projection[..4], &[1.0; 4]);
        assert_eq!(&projection[4..], &[0.0; 4]);
    }

    #[test]
    fn euler_flow_uses_forward_time_sign() {
        let mut actions = [1.0, -1.0];
        euler_flow_step(&mut actions, &[2.0, -4.0], 0.25).unwrap();
        assert_close(actions[0], 1.5);
        assert_close(actions[1], -2.0);
    }

    #[test]
    fn invalid_math_inputs_return_errors() {
        assert!(flow_schedule(0, 1000).is_err());
        assert!(action_timestep_embedding(0, 7).is_err());
        assert!(dit_timestep_projection(0, 2).is_err());
        assert!(euler_flow_step(&mut [0.0], &[0.0, 1.0], 0.25).is_err());
    }
}
