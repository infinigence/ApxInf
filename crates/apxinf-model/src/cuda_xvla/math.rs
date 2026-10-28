//! Host-side constant tables for X-VLA.
//!
//! Everything here is checkpoint- or schedule-determined, so it is computed once
//! at load time in f32 and uploaded as a device BF16 constant.

/// Cosine temporal embedding row 0 for `T = 1`.
///
/// `PositionalEmbeddingCosine1D` evaluates `sin(0) = 0` / `cos(0) = 1` at
/// position zero, i.e. an even-zero / odd-one vector. For a single frame this is
/// a constant added to every spatial token, so it is folded into the 2D image
/// position table.
pub fn temporal_position_zero(dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|index| if index % 2 == 0 { 0.0 } else { 1.0 })
        .collect()
}

/// `LearnedAbsolutePositionEmbedding2D` table plus the temporal constant.
///
/// `x[i, j] += cat(column[j], row[i]) + temporal[row]` where the temporal term is
/// broadcast over tokens. Output is `[side * side, 2 * half]`.
pub fn image_position_table(
    row_embeddings: &[f32],
    column_embeddings: &[f32],
    side: usize,
    half: usize,
) -> Vec<f32> {
    assert_eq!(row_embeddings.len(), 50 * half);
    assert_eq!(column_embeddings.len(), 50 * half);
    let temporal = temporal_position_zero(2 * half);
    let mut table = vec![0.0f32; side * side * 2 * half];
    for y in 0..side {
        for x in 0..side {
            let token = y * side + x;
            let row = &mut table[token * 2 * half..(token + 1) * 2 * half];
            row[..half].copy_from_slice(&column_embeddings[x * half..(x + 1) * half]);
            row[half..].copy_from_slice(&row_embeddings[y * half..(y + 1) * half]);
            for (value, extra) in row.iter_mut().zip(&temporal) {
                *value += extra;
            }
        }
    }
    table
}

/// `timestep_embedding(t, dim, max_period)` from `transformer.py`.
///
/// `freqs = exp(-ln(max_period) * arange(half) / half)`, output is
/// `[cos(t·freqs), sin(t·freqs)]` (cos block first).
pub fn timestep_embedding(t: f32, dim: usize, max_period: usize) -> Vec<f32> {
    let half = dim / 2;
    let log_period = (max_period as f32).ln();
    let mut out = vec![0.0f32; dim];
    for index in 0..half {
        let frequency = (-log_period * index as f32 / half as f32).exp();
        let angle = t * frequency;
        out[index] = angle.cos();
        out[half + index] = angle.sin();
    }
    out
}

/// The inference schedule: `i / steps` for `i = steps ..= 1` (1.0, 0.9, …, 0.1).
pub fn flow_timesteps(steps: usize) -> Vec<f32> {
    (0..steps)
        .map(|index| (steps - index) as f32 / steps as f32)
        .collect()
}

/// Window partition / reverse tables for one DaViT stage.
///
/// Both directions are row gathers over a token-major `[tokens, C]` buffer:
///
/// * `partition_rows` has `windows * window²` entries; each entry indexes the
///   source token for that window slot, or `sentinel` for zero padding.
///   The source buffer must hold `tokens + 1` rows with a zero last row.
/// * `reverse_rows` has `tokens` entries mapping an output token back to its
///   window-buffer row (padding already cropped away).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowPlan {
    pub height: usize,
    pub width: usize,
    pub window: usize,
    pub windows_h: usize,
    pub windows_w: usize,
    pub sentinel: u32,
    pub partition_rows: Vec<u32>,
    pub reverse_rows: Vec<u32>,
}

impl WindowPlan {
    pub fn windows(&self) -> usize {
        self.windows_h * self.windows_w
    }
}

pub fn window_plan(height: usize, width: usize, window: usize) -> WindowPlan {
    let windows_h = height.div_ceil(window);
    let windows_w = width.div_ceil(window);
    let windows = windows_h * windows_w;
    let sentinel = (height * width) as u32;

    let mut partition_rows = vec![sentinel; windows * window * window];
    for window_y in 0..windows_h {
        for window_x in 0..windows_w {
            let window_index = window_y * windows_w + window_x;
            for inner_y in 0..window {
                for inner_x in 0..window {
                    let out_row = ((window_index * window + inner_y) * window) + inner_x;
                    let y = window_y * window + inner_y;
                    let x = window_x * window + inner_x;
                    if y < height && x < width {
                        partition_rows[out_row] = (y * width + x) as u32;
                    }
                }
            }
        }
    }

    let mut reverse_rows = vec![0u32; height * width];
    for y in 0..height {
        for x in 0..width {
            let window_y = y / window;
            let window_x = x / window;
            let window_index = window_y * windows_w + window_x;
            let inner_y = y % window;
            let inner_x = x % window;
            reverse_rows[y * width + x] =
                ((window_index * window + inner_y) * window + inner_x) as u32;
        }
    }

    WindowPlan {
        height,
        width,
        window,
        windows_h,
        windows_w,
        sentinel,
        partition_rows,
        reverse_rows,
    }
}

/// Row gather table for `spatial_avg_pool`: a single all-ones row of `tokens`.
pub fn mean_row(tokens: usize) -> Vec<f32> {
    vec![1.0; tokens]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestep_matches_reference_schedule() {
        let zero = timestep_embedding(0.0, 32, 100);
        assert!((zero[0] - 1.0).abs() < 1e-6);
        assert!(zero[16].abs() < 1e-6);
        let one = timestep_embedding(1.0, 32, 100);
        assert!((one[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((one[16] - 1.0f32.sin()).abs() < 1e-6);
        let last = timestep_embedding(0.1, 32, 100);
        assert!(last[15] > 0.99);
        assert!(last[31] > 0.0);
    }

    #[test]
    fn flow_schedule_descends() {
        assert_eq!(
            flow_timesteps(10),
            vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        );
    }

    #[test]
    fn window_plan_roundtrips_without_padding() {
        let plan = window_plan(12, 12, 12);
        assert_eq!(plan.windows(), 1);
        assert_eq!(plan.partition_rows.len(), 144);
        for (out, src) in plan.reverse_rows.iter().enumerate() {
            assert_eq!(plan.partition_rows[*src as usize] as usize, out);
        }
    }

    #[test]
    fn window_plan_handles_padding_and_crop() {
        let plan = window_plan(7, 7, 12);
        assert_eq!(plan.windows_h, 1);
        assert_eq!(plan.windows(), 1);
        assert_eq!(plan.partition_rows.len(), 144);
        assert_eq!(plan.sentinel, 49);
        assert_eq!(
            plan.partition_rows.iter().filter(|row| **row == 49).count(),
            144 - 49
        );
        let plan = window_plan(14, 14, 12);
        assert_eq!((plan.windows_h, plan.windows_w, plan.windows()), (2, 2, 4));
        for (out, src) in plan.reverse_rows.iter().enumerate() {
            let src = *src as usize;
            assert_eq!(plan.partition_rows[src] as usize, out);
        }
        let plan = window_plan(56, 56, 12);
        assert_eq!((plan.windows_h, plan.windows_w, plan.windows()), (5, 5, 25));
    }
}
