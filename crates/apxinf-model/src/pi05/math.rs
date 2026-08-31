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

/// Match NumPy `digitize(state, linspace(-1, 1, 257)[:-1]) - 1`.
///
/// `digitize` returns `0` for `v < -1`, so the bin is `-1`, and openpi writes
/// that straight into the prompt (`models/tokenizer.py`) — it is a value the
/// model saw during training, so it must not be clamped to `0`. The return type
/// is therefore signed. Values `>= 1` do saturate, to `255`.
///
/// Implemented as a search over the edges, which is what `digitize` does, rather
/// than the arithmetic `floor((v + 1.0) * 128.0)`. The two are not equal: adding
/// `1.0` to a value just below an edge in `[-0.5, -0.25)` moves it into a binade
/// with twice the ulp, the sum rounds *up* onto the edge, and the floor lands one
/// bin high. `-0.4921875` (edge 65) is the one f32 value where this happens, and
/// one bin is a different prompt string, different token ids, different rollout.
///
/// Every edge is `-1 + i/128` with `i <= 255`, i.e. `(i - 128)/128` — at most
/// eight significant bits — so it is exact in f32 and each comparison below is
/// exact. NaN falls out as `-1` here (all comparisons false); NaN state is
/// rejected upstream and is not a supported input on either side.
pub fn discretize_state(state: &[f32]) -> Vec<i16> {
    state
        .iter()
        .map(|&value| {
            // Binary search for the number of edges <= value, i.e.
            // `searchsorted(edges, value, side="right")`; the bin is that minus one.
            let (mut lo, mut hi) = (0i32, 256i32);
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if -1.0 + mid as f32 / 128.0 <= value {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            (lo - 1) as i16
        })
        .collect()
}

pub fn pi05_prompt(task: &str, normalized_state: &[f32], discrete_state_input: bool) -> String {
    let task = task.trim().replace('_', " ").replace('\n', " ");
    if !discrete_state_input {
        return format!("{task}\n");
    }
    let state = discretize_state(normalized_state)
        .iter()
        .map(i16::to_string)
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
            discretize_state(&[-2.0, -1.000_001, -1.0, 0.0, 0.999, 1.0, 2.0]),
            vec![-1, -1, 0, 128, 255, 255, 255]
        );
        assert_eq!(
            pi05_prompt(" pick_up\ncup ", &[-2.0, -1.0, 0.0, 1.0], true),
            "Task: pick up cup, State: -1 0 128 255;\nAction: "
        );
        assert_eq!(pi05_prompt("pick_up", &[], false), "pick up\n");
    }

    /// The next representable f32 below `x`. `0.0` steps to the smallest
    /// negative subnormal, which is what makes edge 128 (exactly `0.0`) testable.
    fn next_below(x: f32) -> f32 {
        if x > 0.0 {
            f32::from_bits(x.to_bits() - 1)
        } else if x < 0.0 {
            f32::from_bits(x.to_bits() + 1)
        } else {
            -f32::from_bits(1)
        }
    }

    fn next_above(x: f32) -> f32 {
        -next_below(-x)
    }

    #[test]
    fn discretization_matches_digitize_at_every_bin_edge() {
        // Each edge and both of its f32 neighbours: the edge itself opens its own
        // bin, one ulp below stays in the previous one. The arithmetic form
        // `floor((v + 1.0) * 128.0)` gets exactly one of these 768 cases wrong.
        for i in 0..256i32 {
            let edge = -1.0 + i as f32 / 128.0;
            assert_eq!(discretize_state(&[edge]), vec![i as i16], "edge {i}");
            assert_eq!(
                discretize_state(&[next_below(edge)]),
                vec![(i - 1) as i16],
                "one ulp below edge {i}"
            );
            assert_eq!(
                discretize_state(&[next_above(edge)]),
                vec![i as i16],
                "one ulp above edge {i}"
            );
        }
    }

    #[test]
    fn discretization_does_not_round_up_onto_an_edge() {
        // One ulp below edge 65 (-0.4921875). Adding 1.0 lands the value in a
        // binade with twice the ulp, so the sum rounds up to exactly 0.5078125
        // and a floor-based index returns 65 instead of 64 -- a different prompt
        // string for a state that is, by NumPy's reckoning, in bin 64.
        let v = f32::from_bits((-0.4921875f32).to_bits() + 1);
        assert!(v < -0.4921875, "one ulp below the edge");
        assert_eq!(((v + 1.0) * 128.0).floor() as i32, 65, "the trap this guards");
        assert_eq!(discretize_state(&[v]), vec![64]);
    }

    #[test]
    fn euler_flow_uses_reverse_time_sign() {
        let mut x = [1.0, -1.0];
        euler_flow_step(&mut x, &[2.0, -4.0], 10);
        assert!((x[0] - 0.8).abs() < 1e-6);
        assert!((x[1] + 0.6).abs() < 1e-6);
    }
}
