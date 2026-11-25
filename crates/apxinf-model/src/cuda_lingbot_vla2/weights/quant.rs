//! Offline ConvRot W8A8 weight preparation, run once at load time.
//!
//! The published checkpoint is F32. A linear `W [N, K]` (K a multiple of 256) is rotated by the
//! normalised Hadamard matrix `H = (H4 x H4 x H4 x H4) / 16` on every 256-wide group of K and then
//! quantised symmetrically to INT8 with one scale per output channel. At run time the activation
//! gets the same rotation (`QuantizationSemantic::RotateRowwiseI8`), and because `H` is orthogonal
//! `(x H)(W H)^T = x W^T`, so the product is unchanged while outlier channels are spread.
//!
//! The rotation here is the same radix-4 butterfly the CUDA kernel runs (`rotate_rows.cuh`), applied
//! in FP32 and never rounded to BF16, so weight and activation see the identical transform.

use rayon_free::parallel_rows;

pub const GROUP: usize = 256;

/// In-place unscaled Hadamard rotation of one 256-wide group (result is 16x the normalised one).
pub fn rotate_group_unscaled(tile: &mut [f32; GROUP]) {
    let mut stride = 1;
    while stride < GROUP {
        for quad in 0..GROUP / 4 {
            let base = (quad / stride) * 4 * stride + (quad % stride);
            let (a, b, c, d) = (
                tile[base],
                tile[base + stride],
                tile[base + 2 * stride],
                tile[base + 3 * stride],
            );
            tile[base] = a + b + c - d;
            tile[base + stride] = a + b - c + d;
            tile[base + 2 * stride] = a - b + c + d;
            tile[base + 3 * stride] = -a + b + c + d;
        }
        stride *= 4;
    }
}

/// One quantised linear: output-major INT8 bytes `[n, k]` and one FP32 scale per output channel.
pub struct QuantLinear {
    pub weight: Vec<u8>,
    pub scales: Vec<f32>,
    pub n: usize,
    pub k: usize,
}

/// Rotate and quantise `rows` consecutive rows of an F32 matrix with `k` columns.
/// `scale = max(amax / 127, 1e-12)`, `q = clamp(round_half_away(y / scale), -128, 127)`, matching the
/// activation operator's rule so both sides of the product use one definition.
pub fn quantize_rows(src: &[f32], rows: usize, k: usize) -> QuantLinear {
    assert!(
        k.is_multiple_of(GROUP),
        "ConvRot needs K % 256 == 0, got {k}"
    );
    assert_eq!(src.len(), rows * k);
    let mut weight = vec![0u8; rows * k];
    let mut scales = vec![0f32; rows];
    parallel_rows(&mut weight, &mut scales, k, |row, q_out, scale_out| {
        let x = &src[row * k..(row + 1) * k];
        let mut rotated = vec![0f32; k];
        let mut tile = [0f32; GROUP];
        for g in 0..k / GROUP {
            tile.copy_from_slice(&x[g * GROUP..(g + 1) * GROUP]);
            rotate_group_unscaled(&mut tile);
            for (dst, v) in rotated[g * GROUP..(g + 1) * GROUP].iter_mut().zip(tile) {
                *dst = v * (1.0 / 16.0);
            }
        }
        let amax = rotated.iter().fold(0f32, |m, v| m.max(v.abs()));
        let scale = (amax / 127.0).max(1e-12);
        *scale_out = scale;
        for (dst, v) in q_out.iter_mut().zip(&rotated) {
            *dst = (v / scale).round().clamp(-128.0, 127.0) as i8 as u8;
        }
    });
    QuantLinear {
        weight,
        scales,
        n: rows,
        k,
    }
}

/// Stack several `[rows_i, k]` matrices along the output dimension and quantise them as one linear.
/// Used for the dense-masked MoE: gate/up of all 32 experts form one `[2 * E * I, 768]` GEMM, and
/// the per-expert down projections form one `[768, E * I]` GEMM with K = E * I.
pub fn concat_rows(parts: &[&[f32]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
    for p in parts {
        out.extend_from_slice(p);
    }
    out
}

/// Minimal scoped row parallelism on std threads (no new dependency).
mod rayon_free {
    pub fn parallel_rows<F>(weight: &mut [u8], scales: &mut [f32], k: usize, f: F)
    where
        F: Fn(usize, &mut [u8], &mut f32) + Sync,
    {
        let rows = scales.len();
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(rows.max(1));
        let per = rows.div_ceil(threads.max(1)).max(1);
        std::thread::scope(|scope| {
            let mut w = weight;
            let mut s = scales;
            let mut first = 0usize;
            while !s.is_empty() {
                let take = per.min(s.len());
                let (wh, wt) = w.split_at_mut(take * k);
                let (sh, st) = s.split_at_mut(take);
                let base = first;
                let f = &f;
                scope.spawn(move || {
                    for (i, (q, sc)) in wh.chunks_mut(k).zip(sh.iter_mut()).enumerate() {
                        f(base + i, q, sc);
                    }
                });
                w = wt;
                s = st;
                first += take;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hadamard() -> Vec<f32> {
        let h4 = [
            [1f32, 1., 1., -1.],
            [1., 1., -1., 1.],
            [1., -1., 1., 1.],
            [-1., 1., 1., 1.],
        ];
        let mut h = vec![vec![1f32]];
        for _ in 0..4 {
            let n = h.len();
            let mut next = vec![vec![0f32; n * 4]; n * 4];
            for (i, row) in h.iter().enumerate() {
                for (j, v) in row.iter().enumerate() {
                    for a in 0..4 {
                        for b in 0..4 {
                            next[i * 4 + a][j * 4 + b] = v * h4[a][b];
                        }
                    }
                }
            }
            h = next;
        }
        h.into_iter().flatten().collect()
    }

    #[test]
    fn butterfly_equals_the_kronecker_matrix() {
        let h = hadamard();
        let x: Vec<f32> = (0..GROUP).map(|i| ((i * 37 % 31) as f32) - 15.0).collect();
        let mut tile = [0f32; GROUP];
        tile.copy_from_slice(&x);
        rotate_group_unscaled(&mut tile);
        for j in 0..GROUP {
            let expect: f32 = (0..GROUP).map(|i| x[i] * h[i * GROUP + j]).sum();
            assert_eq!(tile[j], expect, "column {j}");
        }
    }

    #[test]
    fn rotation_is_orthogonal_up_to_the_16x_scale() {
        let x: Vec<f32> = (0..GROUP)
            .map(|i| ((i * 13 % 17) as f32) * 0.25 - 2.0)
            .collect();
        let mut tile = [0f32; GROUP];
        tile.copy_from_slice(&x);
        rotate_group_unscaled(&mut tile);
        let before: f32 = x.iter().map(|v| v * v).sum();
        let after: f32 = tile.iter().map(|v| v * v).sum::<f32>() / 256.0;
        assert!(
            (before - after).abs() <= 1e-3 * before,
            "{before} vs {after}"
        );
    }

    #[test]
    fn quantised_product_tracks_the_float_product() {
        let (n, k) = (8usize, 512usize);
        let w: Vec<f32> = (0..n * k)
            .map(|i| (((i * 2654435761usize) >> 7) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        // an activation with a few huge channels, the case ConvRot exists for
        let mut x: Vec<f32> = (0..k)
            .map(|i| (((i * 40503) % 997) as f32) / 500.0 - 1.0)
            .collect();
        x[3] = 60.0;
        x[300] = -45.0;
        let q = quantize_rows(&w, n, k);
        let xq = quantize_rows(&x, 1, k);
        for row in 0..n {
            let exact: f32 = (0..k).map(|i| x[i] * w[row * k + i]).sum();
            let int: i32 = (0..k)
                .map(|i| (xq.weight[i] as i8 as i32) * (q.weight[row * k + i] as i8 as i32))
                .sum();
            let approx = int as f32 * xq.scales[0] * q.scales[row];
            let scale = (0..k).map(|i| (x[i] * w[row * k + i]).abs()).sum::<f32>();
            assert!(
                (approx - exact).abs() <= 0.02 * scale,
                "row {row}: {approx} vs {exact}"
            );
        }
    }

    #[test]
    fn zero_rows_keep_a_positive_scale() {
        let q = quantize_rows(&vec![0f32; 256], 1, 256);
        assert!(q.scales[0] > 0.0);
        assert!(q.weight.iter().all(|b| *b == 0));
    }
}
