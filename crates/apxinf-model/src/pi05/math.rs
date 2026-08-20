use std::f64::consts::TAU;

/// OpenPI's float64-generated sinusoidal flow timestep embedding.
pub fn sinusoidal_time_embedding(
    time: f32,
    dimension: usize,
    min_period: f32,
    max_period: f32,
) -> Vec<f32> {
    assert_eq!(dimension % 2, 0);
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
        let phase = time as f64 * TAU / period;
        sin.push(phase.sin() as f32);
        cos.push(phase.cos() as f32);
    }
    sin.extend(cos);
    sin
}

/// Match NumPy `digitize(state, linspace(-1, 1, 257)[:-1]) - 1`, with
/// saturation for out-of-range sensor values.
pub fn discretize_state(state: &[f32]) -> Vec<u8> {
    state
        .iter()
        .map(|&value| (((value + 1.0) * 128.0).floor() as i32).clamp(0, 255) as u8)
        .collect()
}

pub fn pi05_prompt(task: &str, normalized_state: &[f32], discrete_state_input: bool) -> String {
    let task = task.trim().replace('_', " ").replace('\n', " ");
    if !discrete_state_input {
        return format!("{task}\n");
    }
    let state = discretize_state(normalized_state)
        .iter()
        .map(u8::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    format!("Task: {task}, State: {state};\nAction: ")
}

/// One explicit Euler step for OpenPI's reverse-time flow convention.
/// `x <- x - velocity / num_steps`.
pub fn euler_flow_step(x: &mut [f32], velocity: &[f32], num_steps: usize) {
    assert_eq!(x.len(), velocity.len());
    assert!(num_steps > 0);
    let dt = 1.0 / num_steps as f32;
    for (x, &v) in x.iter_mut().zip(velocity) {
        *x -= dt * v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestep_embedding_matches_endpoints() {
        let emb = sinusoidal_time_embedding(0.0, 8, 4e-3, 4.0);
        assert_eq!(&emb[..4], &[0.0; 4]);
        assert_eq!(&emb[4..], &[1.0; 4]);
    }

    #[test]
    fn state_discretization_and_prompt_match_pi05() {
        assert_eq!(
            discretize_state(&[-2.0, -1.0, 0.0, 0.999, 1.0, 2.0]),
            vec![0, 0, 128, 255, 255, 255]
        );
        assert_eq!(
            pi05_prompt(" pick_up\ncup ", &[-1.0, 0.0, 1.0], true),
            "Task: pick up cup, State: 0 128 255;\nAction: "
        );
        assert_eq!(pi05_prompt("pick_up", &[], false), "pick up\n");
    }

    #[test]
    fn euler_flow_uses_reverse_time_sign() {
        let mut x = [1.0, -1.0];
        euler_flow_step(&mut x, &[2.0, -4.0], 10);
        assert!((x[0] - 0.8).abs() < 1e-6);
        assert!((x[1] + 0.6).abs() < 1e-6);
    }
}
