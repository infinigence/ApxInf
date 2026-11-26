//! HIP-independent π0.5 helpers, owned by this port.
//!
//! Model loading uses [`sinusoidal_time_embedding`] to prepare the fixed
//! timestep conditioning; like the rest of the port this is a copy of the
//! family's math, not shared code.

/// OpenPI's float64-generated sinusoidal flow-timestep embedding: sines then
/// cosines over periods geometric from `min_period` to `max_period`.
pub fn sinusoidal_time_embedding(
    time: f32,
    dimension: usize,
    min_period: f32,
    max_period: f32,
) -> Vec<f32> {
    let half = dimension / 2;
    let mut sin = Vec::with_capacity(half);
    let mut cos = Vec::with_capacity(half);
    for i in 0..half {
        let fraction = if half == 1 {
            0.0
        } else {
            i as f64 / (half - 1) as f64
        };
        let period = min_period as f64 * (max_period as f64 / min_period as f64).powf(fraction);
        let phase = time as f64 * std::f64::consts::TAU / period;
        sin.push(phase.sin() as f32);
        cos.push(phase.cos() as f32);
    }
    sin.extend(cos);
    sin
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_embedding_matches_its_endpoints() {
        let e = sinusoidal_time_embedding(0.0, 8, 4e-3, 4.0);
        assert_eq!(&e[..4], &[0.0; 4]);
        assert_eq!(&e[4..], &[1.0; 4]);
    }
}
