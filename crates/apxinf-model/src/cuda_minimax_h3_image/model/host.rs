//! Host arithmetic that depends only on the checkpoint and the request geometry: AdaLN curve
//! evaluation, per-step modulation rows, folded PDD heads and the rotary table. Nothing here touches the device.
use crate::cuda_minimax_h3_image::config::*;
use crate::cuda_minimax_h3_image::weights::{AdalnCurve, HostHead};
use half::bf16;

/// `torch.lerp` for one element, including its two-sided formulation.
fn lerp(start: f32, end: f32, weight: f32) -> f32 {
    if weight < 0.5 {
        start + weight * (end - start)
    } else {
        end - (end - start) * (1.0 - weight)
    }
}

/// Point of the AdaLN time-embedding curve at `t` (ComfyUI `_time_embedding`, curve form): `t` clamps to `[0, 1]`
/// and maps to a fractional row of the `[ADALN_GRID, ADALN_RANK]` table, then two rows are blended.
pub(crate) fn curve_embedding(table: &[f32], t: f32) -> [f32; ADALN_RANK] {
    debug_assert_eq!(table.len(), ADALN_GRID * ADALN_RANK);
    let pos = t.clamp(0.0, 1.0) * (ADALN_GRID - 1) as f32;
    let i0 = (pos.floor() as usize).min(ADALN_GRID - 2);
    let weight = pos - i0 as f32;
    let mut out = [0.0; ADALN_RANK];
    for (j, value) in out.iter_mut().enumerate() {
        *value = lerp(
            table[i0 * ADALN_RANK + j],
            table[(i0 + 1) * ADALN_RANK + j],
            weight,
        );
    }
    out
}

/// `embedding @ weight^T + bias` for output rows `rows`, in FP32 as the curve checkpoints specify (`adaln_dtype` FP32).
pub(crate) fn project(
    curve: &AdalnCurve,
    embedding: &[f32; ADALN_RANK],
    rows: std::ops::Range<usize>,
) -> Vec<f32> {
    rows.map(|r| {
        let w = &curve.weight[r * ADALN_RANK..(r + 1) * ADALN_RANK];
        let sum: f32 = w.iter().zip(embedding).map(|(w, e)| w.to_f32() * e).sum();
        sum + curve.bias[r].to_f32()
    })
    .collect()
}

/// Modulation of one DiT block at one timestep: BF16 `[modality, 6, HIDDEN]` for the image (0) and text (1) rows,
/// parts ordered `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`. Every row is rounded to BF16
/// once, as the reference does when it applies a row to a BF16 stream.
pub(crate) fn block_modulation(curve: &AdalnCurve, embedding: &[f32; ADALN_RANK]) -> Vec<bf16> {
    let used = (MODALITY_TEXT + 1) * 6 * HIDDEN;
    project(curve, embedding, 0..used)
        .into_iter()
        .map(bf16::from_f32)
        .collect()
}

/// Final-layer shift and scale in FP32 (the reference keeps them FP32 and promotes the normalized rows).
pub(crate) fn final_modulation(
    curve: &AdalnCurve,
    embedding: &[f32; ADALN_RANK],
) -> (Vec<f32>, Vec<f32>) {
    let all = project(curve, embedding, 0..2 * HIDDEN);
    (all[..HIDDEN].to_vec(), all[HIDDEN..].to_vec())
}

/// Fold `norm * (1 + scale) + shift` into the output head: `head(norm * (1 + s) + t)` equals
/// `norm @ (W (1 + s)) + (W t + b)`. Returns the weight as row-major `[HIDDEN, PATCH_DIM]` (K by N) and the bias.
pub(crate) fn fold_head(head: &HostHead, shift: &[f32], scale: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut weight = vec![0.0f32; HIDDEN * PATCH_DIM];
    let mut bias = vec![0.0f32; PATCH_DIM];
    for o in 0..PATCH_DIM {
        let row = &head.weight[o * HIDDEN..(o + 1) * HIDDEN];
        let mut acc = head.bias[o] as f64;
        for k in 0..HIDDEN {
            weight[k * PATCH_DIM + o] = row[k] * (1.0 + scale[k]);
            acc += row[k] as f64 * shift[k] as f64;
        }
        bias[o] = acc as f32;
    }
    (weight, bias)
}

/// `[LATENT_CHANNELS, h, w]` latent to `[(h / 2) * (w / 2), PATCH_DIM]` patch rows, feature order `(channel, py, px)`
/// (ComfyUI `patchify_video` with patch `(1, 2, 2)`).
pub(crate) fn patchify(latent: &[f32], h: usize, w: usize) -> Vec<f32> {
    assert_eq!(latent.len(), LATENT_CHANNELS * h * w);
    assert!(h % PATCH == 0 && w % PATCH == 0);
    let (gh, gw) = (h / PATCH, w / PATCH);
    let mut out = vec![0.0; gh * gw * PATCH_DIM];
    for gy in 0..gh {
        for gx in 0..gw {
            let row = &mut out[(gy * gw + gx) * PATCH_DIM..][..PATCH_DIM];
            for c in 0..LATENT_CHANNELS {
                for py in 0..PATCH {
                    for px in 0..PATCH {
                        row[(c * PATCH + py) * PATCH + px] =
                            latent[(c * h + gy * PATCH + py) * w + gx * PATCH + px];
                    }
                }
            }
        }
    }
    out
}

/// Inverse of [`patchify`].
pub(crate) fn unpatchify(rows: &[f32], h: usize, w: usize) -> Vec<f32> {
    let (gh, gw) = (h / PATCH, w / PATCH);
    assert_eq!(rows.len(), gh * gw * PATCH_DIM);
    let mut out = vec![0.0; LATENT_CHANNELS * h * w];
    for gy in 0..gh {
        for gx in 0..gw {
            let row = &rows[(gy * gw + gx) * PATCH_DIM..][..PATCH_DIM];
            for c in 0..LATENT_CHANNELS {
                for py in 0..PATCH {
                    for px in 0..PATCH {
                        out[(c * h + gy * PATCH + py) * w + gx * PATCH + px] =
                            row[(c * PATCH + py) * PATCH + px];
                    }
                }
            }
        }
    }
    out
}

/// Axis coordinates of the image patch grid: `(arange(n) * (ratio / n) + (1 - ratio) / 2) * 32` with
/// `ratio = latent_side / sqrt(latent_area)` (ComfyUI `_axis_from_sqrt_area`), in float64.
fn axis(side: usize, area_root: f64) -> Vec<f64> {
    let ratio = side as f64 / area_root;
    let n = side / PATCH;
    (0..n)
        .map(|i| (i as f64 * (ratio / n as f64) + (1.0 - ratio) / 2.0) * 32.0)
        .collect()
}

/// `(t, h, w)` position of every packed row: text rows count along `t`; every image row sits at `t = text_rows`
/// with the area-normalised patch coordinates in `h`, `w`.
pub(crate) fn positions(layout: Layout, canvas: Canvas) -> Vec<[f64; 3]> {
    let (lat_h, lat_w) = (canvas.latent_height(), canvas.latent_width());
    let area_root = ((lat_h * lat_w) as f64).sqrt();
    let (hs, ws) = (axis(lat_h, area_root), axis(lat_w, area_root));
    let mut out = Vec::with_capacity(layout.total_rows());
    for i in 0..layout.text_rows {
        out.push([i as f64, 0.0, 0.0]);
    }
    for &h in &hs {
        for &w in &ws {
            out.push([layout.text_rows as f64, h, w]);
        }
    }
    out
}

/// Cosine and sine for the half-split rotation over all `HEAD_DIM` channels after the RoPE channel permutation.
/// FP32 `[rows, 2, HEAD_DIM / 2]`, cosine then sine. Channels `0..48` rotate with the model's pair angles, channels
/// `48..64` are the pass-through pairs (cos 1, sin 0). Both are rounded to BF16 like the reference's rotation table.
pub(crate) fn rope_table(positions: &[[f64; 3]], inv_freq: &[f32]) -> Vec<f32> {
    const HALF: usize = HEAD_DIM / 2;
    let mut out = vec![0.0f32; positions.len() * 2 * HALF];
    for (row, p) in positions.iter().enumerate() {
        let (cos, sin) = out[row * 2 * HALF..(row + 1) * 2 * HALF].split_at_mut(HALF);
        for c in 0..HALF {
            let angle = if c < 3 * ROPE_FREQS {
                (p[c / ROPE_FREQS] as f32) * inv_freq[c % ROPE_FREQS]
            } else {
                0.0
            };
            let (s, co) = if c < 3 * ROPE_FREQS {
                (angle.sin(), angle.cos())
            } else {
                (0.0, 1.0)
            };
            cos[c] = bf16::from_f32(co).to_f32();
            sin[c] = bf16::from_f32(s).to_f32();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp_table() -> Vec<f32> {
        (0..ADALN_GRID * ADALN_RANK)
            .map(|i| (i / ADALN_RANK) as f32 + 0.01 * (i % ADALN_RANK) as f32)
            .collect()
    }

    #[test]
    fn curve_embedding_clamps_and_blends() {
        let table = ramp_table();
        assert_eq!(curve_embedding(&table, 0.0)[0], 0.0);
        assert_eq!(curve_embedding(&table, -3.0)[0], 0.0);
        // t = 1 reads the last interval (rows 1023 and 1024) at weight 1, so the last row exactly
        assert_eq!(curve_embedding(&table, 1.0)[0], 1024.0);
        assert_eq!(curve_embedding(&table, 7.0)[0], 1024.0);
        // t = 0.5 is row 512 exactly
        assert!((curve_embedding(&table, 0.5)[3] - 512.03).abs() < 1e-3);
        // a quarter of the way between rows 10 and 11: t * 1024 = 10.25
        assert!((curve_embedding(&table, 10.25 / 1024.0)[0] - 10.25).abs() < 1e-4);
    }

    #[test]
    fn projection_adds_bias_and_rounds_rows_to_bf16() {
        let rows = 6 * HIDDEN * MODALITIES;
        let curve = AdalnCurve {
            weight: (0..rows * ADALN_RANK)
                .map(|i| half::f16::from_f32(((i % 7) as f32 - 3.0) * 0.25))
                .collect(),
            bias: (0..rows).map(|i| bf16::from_f32((i % 5) as f32)).collect(),
        };
        let e = [1.0, 0.5, 0.0, -1.0, 2.0, 0.0, 0.0, 0.25];
        let modulation = block_modulation(&curve, &e);
        assert_eq!(modulation.len(), 2 * 6 * HIDDEN);
        let expected: f32 = (0..ADALN_RANK)
            .map(|j| e[j] * curve.weight[j].to_f32())
            .sum::<f32>()
            + curve.bias[0].to_f32();
        assert_eq!(modulation[0], bf16::from_f32(expected));
    }

    #[test]
    fn folded_head_matches_modulate_then_project() {
        let head = HostHead {
            weight: (0..PATCH_DIM * HIDDEN)
                .map(|i| ((i * 31 % 17) as f32 - 8.0) * 0.01)
                .collect(),
            bias: (0..PATCH_DIM).map(|i| i as f32 * 0.1).collect(),
        };
        let shift: Vec<f32> = (0..HIDDEN).map(|i| ((i % 9) as f32 - 4.0) * 0.05).collect();
        let scale: Vec<f32> = (0..HIDDEN).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
        let normalized: Vec<f32> = (0..HIDDEN)
            .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.2)
            .collect();
        let (w, b) = fold_head(&head, &shift, &scale);
        for o in [0, 17, PATCH_DIM - 1] {
            let direct: f64 = (0..HIDDEN)
                .map(|k| {
                    head.weight[o * HIDDEN + k] as f64
                        * (normalized[k] as f64 * (1.0 + scale[k] as f64) + shift[k] as f64)
                })
                .sum::<f64>()
                + head.bias[o] as f64;
            let folded: f64 = (0..HIDDEN)
                .map(|k| normalized[k] as f64 * w[k * PATCH_DIM + o] as f64)
                .sum::<f64>()
                + b[o] as f64;
            assert!(
                (direct - folded).abs() < 1e-3 * direct.abs().max(1.0),
                "o={o} {direct} vs {folded}"
            );
        }
    }

    #[test]
    fn patchify_orders_features_channel_then_pixel_and_round_trips() {
        let (h, w) = (4, 6);
        let latent: Vec<f32> = (0..LATENT_CHANNELS * h * w).map(|i| i as f32).collect();
        let rows = patchify(&latent, h, w);
        assert_eq!(rows.len(), 2 * 3 * PATCH_DIM);
        // patch (gy=1, gx=2), channel 3, pixel (1, 0) is latent[3, 2 + 1, 4 + 0]
        let row = &rows[(1 * 3 + 2) * PATCH_DIM..][..PATCH_DIM];
        assert_eq!(row[(3 * 2 + 1) * 2], latent[(3 * h + 3) * w + 4]);
        assert_eq!(unpatchify(&rows, h, w), latent);
    }

    #[test]
    fn square_canvas_positions_are_the_patch_grid() {
        let canvas = Canvas::new(1024, 1024).unwrap();
        let layout = Layout::new(11, canvas).unwrap();
        let p = positions(layout, canvas);
        assert_eq!(p.len(), 11 + 32 * 32);
        assert_eq!(p[10], [10.0, 0.0, 0.0]);
        assert_eq!(p[11], [11.0, 0.0, 0.0]);
        assert_eq!(p[11 + 33], [11.0, 1.0, 1.0]);
        assert_eq!(p[11 + 32 * 32 - 1], [11.0, 31.0, 31.0]);
    }

    #[test]
    fn wide_canvas_is_centred_by_area() {
        // 2048 x 1024 pixels is a 64 x 128 latent: area root sqrt(8192) = 90.51, so the short side spans
        // 64 / 90.51 = 0.7071 of the 32-unit grid and the long side 1.4142 of it, both centred on the grid middle.
        let canvas = Canvas::new(2048, 1024).unwrap();
        let layout = Layout::new(4, canvas).unwrap();
        let p = positions(layout, canvas);
        assert_eq!(p.len(), 4 + 32 * 64);
        let (first, last) = (p[4], p[p.len() - 1]);
        let close = |a: f64, b: f64| (a - b).abs() < 1e-3;
        assert!(
            close(first[1], 4.6863) && close(first[2], -6.6274),
            "{first:?}"
        );
        assert!(
            close(last[1], 26.6066) && close(last[2], 37.9203),
            "{last:?}"
        );
        assert_eq!(first[0], 4.0);
    }

    #[test]
    fn rope_table_rotates_only_the_model_pairs() {
        let canvas = Canvas::new(1024, 1024).unwrap();
        let layout = Layout::new(3, canvas).unwrap();
        let inv: Vec<f32> = (0..ROPE_FREQS)
            .map(|i| 1.0 / 10000f32.powf(i as f32 / ROPE_FREQS as f32))
            .collect();
        let table = rope_table(&positions(layout, canvas), &inv);
        let row = |r: usize| &table[r * 128..(r + 1) * 128];
        // row 0 is text position 0: every angle 0
        assert!(row(0)[..64].iter().all(|c| *c == 1.0) && row(0)[64..].iter().all(|s| *s == 0.0));
        // a text row at t = 2 rotates the 16 time channels only; h and w axes are 0
        let r2 = row(2);
        assert_eq!(r2[0], bf16::from_f32((2.0f32 * inv[0]).cos()).to_f32());
        assert_eq!(r2[16..48].iter().filter(|c| **c != 1.0).count(), 0);
        // pass-through pairs keep cos 1 and sin 0 for every row
        for r in 0..layout.total_rows() {
            assert!(row(r)[48..64].iter().all(|c| *c == 1.0));
            assert!(row(r)[64 + 48..].iter().all(|s| *s == 0.0));
        }
    }
}

/// Rotary table of the text encoder for positions `0..rows`: FP32 `[rows, 2, TE_HEAD_DIM / 2]`, cosine then sine, plain
/// half-split rotation over all 128 channels of a head (text-only input takes ComfyUI's non-interleaved branch).
/// `inv_freq[c] = 1 / theta^(2c / head_dim)` in FP32 like `precompute_freqs_cis`.
pub(crate) fn text_rope_table(rows: usize) -> Vec<f32> {
    const HALF: usize = TE_HEAD_DIM / 2;
    let inv_freq: Vec<f32> = (0..HALF)
        .map(|c| 1.0 / TE_ROPE_THETA.powf((2 * c) as f32 / TE_HEAD_DIM as f32))
        .collect();
    let mut table = vec![0.0f32; rows * 2 * HALF];
    for row in 0..rows {
        let (cos, sin) = table[row * 2 * HALF..(row + 1) * 2 * HALF].split_at_mut(HALF);
        for c in 0..HALF {
            let angle = row as f32 * inv_freq[c];
            cos[c] = angle.cos();
            sin[c] = angle.sin();
        }
    }
    table
}

#[cfg(test)]
mod text_tests {
    use super::*;

    #[test]
    fn text_rope_table_is_the_half_split_rotation_at_theta_5e6() {
        let table = text_rope_table(5);
        assert_eq!(table.len(), 5 * 128);
        let row = |r: usize| &table[r * 128..(r + 1) * 128];
        // position 0: identity
        assert!(row(0)[..64].iter().all(|c| *c == 1.0) && row(0)[64..].iter().all(|s| *s == 0.0));
        // channel 0 has inverse frequency 1, so position 3 rotates by exactly 3 radians
        assert!((row(3)[0] - 3f32.cos()).abs() < 1e-6 && (row(3)[64] - 3f32.sin()).abs() < 1e-6);
        // the last pair's inverse frequency is theta^(-126/128), about 2e-7 rad per position, so it is nearly identity at position 4
        let last = 5_000_000f32.powf(-126.0 / 128.0);
        assert!((row(4)[63] - (4.0 * last).cos()).abs() < 1e-6 && last < 3e-7);
    }
}
