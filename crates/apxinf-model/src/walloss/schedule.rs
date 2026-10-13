//! Fixed-step action solver schedule and timestep encoding.

use apxinf_core::{Error, Result};

/// Build the inclusive solver timeline used by inference.
pub fn solver_times(steps: usize, scheduler_s: f32, time_shift: f32) -> Result<Vec<f32>> {
    if steps == 0 {
        return Err(Error::Other("walloss solver requires at least one step".into()));
    }
    if !(scheduler_s.is_finite() && scheduler_s > 0.0 && scheduler_s <= 1.0) {
        return Err(Error::Other(format!(
            "walloss scheduler_s must be in (0, 1], got {scheduler_s}"
        )));
    }
    if !(time_shift.is_finite() && time_shift > 0.0) {
        return Err(Error::Other(format!(
            "walloss time_shift must be positive, got {time_shift}"
        )));
    }
    let mut times = Vec::with_capacity(steps + 1);
    for index in 0..=steps {
        let time = index as f32 / steps as f32;
        let shifted = (time_shift * time) / (1.0 + (time_shift - 1.0) * time);
        times.push(shifted * scheduler_s);
    }
    Ok(times)
}

/// Encode one scalar timestep as half sine and half cosine features.
pub fn sinusoidal_time_embedding(time: f32, width: usize) -> Result<Vec<f32>> {
    if width < 4 || width % 2 != 0 {
        return Err(Error::Other(format!(
            "walloss timestep width must be even and at least four, got {width}"
        )));
    }
    let half = width / 2;
    let exponent = 10_000.0_f32.ln() / (half - 1) as f32;
    let mut output = vec![0.0; width];
    for index in 0..half {
        let phase = time * (-(index as f32) * exponent).exp();
        output[index] = phase.sin();
        output[half + index] = phase.cos();
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ten_step_schedule_is_inclusive_and_scaled() {
        let times = solver_times(10, 0.999, 1.0).unwrap();
        assert_eq!(times.len(), 11);
        assert_eq!(times[0], 0.0);
        assert!((times[5] - 0.4995).abs() < 1e-7);
        assert!((times[10] - 0.999).abs() < 1e-7);
    }

    #[test]
    fn embedding_uses_sine_then_cosine() {
        let embedding = sinusoidal_time_embedding(0.0, 8).unwrap();
        assert_eq!(&embedding[..4], &[0.0; 4]);
        assert_eq!(&embedding[4..], &[1.0; 4]);
    }
}
