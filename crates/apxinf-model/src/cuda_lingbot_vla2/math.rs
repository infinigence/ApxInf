//! Host-side math for the LingBot-VLA 2.0 port: mRoPE positions and tables, ViT position embeddings,
//! the sinusoidal time embedding and the bf16 denoise schedule.
//!
//! Everything here is a pure CPU function over `std` + `half`, written to reproduce the upstream PyTorch reference
//! (`lingbotvla/models/vla/lingbot_vla/`, HF `modeling_qwen3_vl.py` 4.57.3) value for value. Where torch's rounding
//! behaviour matters, the formula it uses is replicated rather than the mathematically ideal one:
//!
//! * `torch.linspace` is `start + step * i` for the first half and `end - step * (n - i - 1)` for the second half,
//!   both fused (FMA) in f32, and per-operation rounded in bf16.
//! * `torch.pow` / `sin` / `cos` are Sleef (<= 1 ulp). Rust has no Sleef, so `powf`/`sin`/`cos` are evaluated in f64
//!   and rounded once to f32. They agree with torch to 1 f32 ulp, and are bit-identical for most inputs.
//!
//! Provenance: `vit_pos_embeds` and `vit_axial_positions` keep the block-major layout of
//! `qwen_drive::model::blocks::bf16::{compute_pos_embeds, compute_vision_pos_ids}` (themselves copied from the
//! Qwen3-VL tower); the interpolation arithmetic follows HF `Qwen3VLVisionModel.fast_pos_embed_interpolate`.

use half::bf16;

/// Learned position-embedding table side (`num_position_embeddings = 48 * 48`).
pub const VIT_POS_GRID: usize = 48;
/// Spatial merge size of the Qwen3-VL vision tower.
pub const VIT_MERGE: usize = 2;

fn round_bf16(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

// ---------------------------------------------------------------------------------------------------------------------
// mRoPE
// ---------------------------------------------------------------------------------------------------------------------

/// Per-token `(t, h, w)` mRoPE positions of the compact prefix (no padding slots).
///
/// Layout, in order: for every image `[vision_start, t*(h/2)*(w/2) image tokens, vision_end]`, then `lang_tokens`
/// ordinary text tokens, then `task_tokens` ordinary text tokens (the learned query tokens, which upstream feeds as
/// EOS ids). This is HF `get_rope_index` for the image branch with `spatial_merge_size = 2`: text positions are equal
/// on all three axes and continue from the previous maximum + 1; image tokens use
/// `(t_idx, h_idx, w_idx) + text_len + start`.
///
/// `vision_end` of image *i* and `vision_start` of image *i + 1* are consecutive text tokens. Grid heights and widths
/// must be even.
pub fn rope_index(grids: &[[u32; 3]], lang_tokens: usize, task_tokens: usize) -> Vec<[i64; 3]> {
    let image_tokens: usize = grids
        .iter()
        .map(|g| g[0] as usize * (g[1] as usize / VIT_MERGE) * (g[2] as usize / VIT_MERGE) + 2)
        .sum();
    let mut out = Vec::with_capacity(image_tokens + lang_tokens + task_tokens);
    let mut start = 0i64;
    for (index, grid) in grids.iter().enumerate() {
        debug_assert!(
            grid[1] % 2 == 0 && grid[2] % 2 == 0,
            "grid {grid:?} is not divisible by the merge size"
        );
        let (t, h, w) = (grid[0] as i64, (grid[1] / 2) as i64, (grid[2] / 2) as i64);
        // vision_start (and, from the second image on, the previous vision_end)
        let text_len = if index == 0 { 1 } else { 2 };
        for k in 0..text_len {
            let p = start + k;
            out.push([p, p, p]);
        }
        let offset = start + text_len;
        for ti in 0..t {
            for hi in 0..h {
                for wi in 0..w {
                    out.push([offset + ti, offset + hi, offset + wi]);
                }
            }
        }
        start = if t * h * w == 0 {
            offset
        } else {
            offset + t.max(h).max(w)
        };
    }
    // the last vision_end, the language tokens and the learned query tokens
    let tail = usize::from(!grids.is_empty()) + lang_tokens + task_tokens;
    for k in 0..tail as i64 {
        let p = start + k;
        out.push([p, p, p]);
    }
    out
}

/// First suffix position: the maximum over every row and axis, plus one (upstream `prefix_offsets`).
pub fn suffix_position_offset(positions: &[[i64; 3]]) -> i64 {
    positions
        .iter()
        .flat_map(|p| p.iter().copied())
        .max()
        .map_or(0, |max| max + 1)
}

// ---------------------------------------------------------------------------------------------------------------------
// Vision tower
// ---------------------------------------------------------------------------------------------------------------------

/// Arithmetic precision of `fast_pos_embed_interpolate`'s index math (`torch.linspace`, `dh`, `1 - dh`, products).
///
/// The weights are always rounded to bf16 afterwards (`torch.tensor(weight_list, dtype=pos_embed.weight.dtype)`), and
/// the weighted rows and their 4-way sum are bf16 operations in both modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpArithmetic {
    /// The process default dtype was bf16 (what the golden tensors were produced with: `ref_run.py` calls
    /// `torch.set_default_dtype(torch.bfloat16)`), so `linspace` and the interpolation weights are bf16 values.
    Bf16,
    /// The process default dtype was f32 (upstream `deploy/` leaves it at f32 and casts the model afterwards).
    /// Measured on the real table, 16x16 grid: image embeds differ from [`InterpArithmetic::Bf16`] by 1.8% rel L2.
    /// Not selected by the runtime: the parity target was produced with the bf16 default dtype.
    #[cfg_attr(not(test), allow(dead_code))]
    F32,
}

/// `torch.linspace(0, end, n)` in the default dtype `precision`, returned as f32 values.
fn linspace_grid(end: f32, n: usize, precision: InterpArithmetic) -> Vec<f32> {
    match n {
        0 => return Vec::new(),
        1 => return vec![0.0],
        _ => {}
    }
    let half = n / 2;
    match precision {
        InterpArithmetic::F32 => {
            let step = end / (n - 1) as f32;
            (0..n)
                .map(|i| {
                    if i < half {
                        step.mul_add(i as f32, 0.0)
                    } else {
                        (-step).mul_add((n - i - 1) as f32, end)
                    }
                })
                .collect()
        }
        InterpArithmetic::Bf16 => {
            let r = round_bf16;
            let step = r(r(end) / r((n - 1) as f32));
            (0..n)
                .map(|i| {
                    if i < half {
                        r(r(0.0) + r(step * r(i as f32)))
                    } else {
                        r(r(end) - r(step * r((n - i - 1) as f32)))
                    }
                })
                .collect()
        }
    }
}

/// Bilinear interpolation of the learned `48 x 48` position table to `grid = [t, h, w]` followed by the merge-size-2
/// block-major permutation. `table` is `[2304, hidden]` row-major bf16; the result is `[t * h * w, hidden]` row-major.
///
/// Mirrors HF `fast_pos_embed_interpolate` with the golden arithmetic ([`InterpArithmetic::Bf16`]); use
/// [`vit_pos_embeds_with`] for the f32-default variant.
pub fn vit_pos_embeds(table: &[bf16], grid: [u32; 3]) -> Vec<bf16> {
    vit_pos_embeds_with(table, grid, InterpArithmetic::Bf16)
}

/// [`vit_pos_embeds`] with an explicit index-arithmetic precision.
pub fn vit_pos_embeds_with(
    table: &[bf16],
    grid: [u32; 3],
    arithmetic: InterpArithmetic,
) -> Vec<bf16> {
    let side = VIT_POS_GRID;
    assert!(
        table.len().is_multiple_of(side * side) && !table.is_empty(),
        "position table length {} is not 2304 * hidden",
        table.len()
    );
    let hidden = table.len() / (side * side);
    let (t, h, w) = (grid[0] as usize, grid[1] as usize, grid[2] as usize);
    assert!(
        h % VIT_MERGE == 0 && w % VIT_MERGE == 0,
        "grid {grid:?} is not divisible by the merge size"
    );
    let rd = |x: f32| match arithmetic {
        InterpArithmetic::Bf16 => round_bf16(x),
        InterpArithmetic::F32 => x,
    };

    // per-axis: floor and ceil row/column of the 48-grid, and the fractional offset in the default dtype
    let axis = |n: usize| -> (Vec<usize>, Vec<usize>, Vec<f32>) {
        let idx = linspace_grid((side - 1) as f32, n, arithmetic);
        let floor: Vec<usize> = idx.iter().map(|v| *v as usize).collect(); // `.int()` truncates
        let ceil: Vec<usize> = floor.iter().map(|f| (f + 1).min(side - 1)).collect();
        let frac: Vec<f32> = idx
            .iter()
            .zip(&floor)
            .map(|(v, f)| rd(v - *f as f32))
            .collect();
        (floor, ceil, frac)
    };
    let (h_floor, h_ceil, dh) = axis(h);
    let (w_floor, w_ceil, dw) = axis(w);

    // HF flattens (h, w) row-major for each of the four corners
    let mut corner_index = vec![[0usize; 4]; h * w];
    let mut corner_weight = vec![[bf16::ZERO; 4]; h * w];
    for hi in 0..h {
        for wi in 0..w {
            let k = hi * w + wi;
            corner_index[k] = [
                h_floor[hi] * side + w_floor[wi],
                h_floor[hi] * side + w_ceil[wi],
                h_ceil[hi] * side + w_floor[wi],
                h_ceil[hi] * side + w_ceil[wi],
            ];
            let (a, b) = (rd(1.0 - dh[hi]), rd(1.0 - dw[wi]));
            let weights = [
                rd(a * b),
                rd(a * dw[wi]),
                rd(dh[hi] * b),
                rd(dh[hi] * dw[wi]),
            ];
            corner_weight[k] = weights.map(bf16::from_f32);
        }
    }

    let interpolate = |k: usize, c: usize| -> bf16 {
        // `pos_embed(idx) * weight[:, :, None]` is a bf16 product; the four rows are then summed left to right in bf16
        let mut acc = 0.0f32;
        for corner in 0..4 {
            let row = table[corner_index[k][corner] * hidden + c].to_f32();
            let product = round_bf16(row * corner_weight[k][corner].to_f32());
            acc = if corner == 0 {
                product
            } else {
                round_bf16(acc + product)
            };
        }
        bf16::from_f32(acc)
    };

    let merged_h = h / VIT_MERGE;
    let merged_w = w / VIT_MERGE;
    let mut out = Vec::with_capacity(t * h * w * hidden);
    for _ti in 0..t {
        for mh in 0..merged_h {
            for mw in 0..merged_w {
                for ih in 0..VIT_MERGE {
                    for iw in 0..VIT_MERGE {
                        let k = (mh * VIT_MERGE + ih) * w + (mw * VIT_MERGE + iw);
                        out.extend((0..hidden).map(|c| interpolate(k, c)));
                    }
                }
            }
        }
    }
    out
}

/// `(row, col)` position pairs of every patch in merge-block-major order, `[t * h * w, 2]` flattened (HF
/// `rot_pos_emb` ids; they feed `split_qkv_axial_rope` with theta 10000).
pub fn vit_axial_positions(grid: [u32; 3]) -> Vec<u32> {
    let (t, h, w) = (grid[0] as usize, grid[1] as usize, grid[2] as usize);
    assert!(
        h % VIT_MERGE == 0 && w % VIT_MERGE == 0,
        "grid {grid:?} is not divisible by the merge size"
    );
    let mut ids = Vec::with_capacity(t * h * w * 2);
    for _ti in 0..t {
        for mh in 0..h / VIT_MERGE {
            for mw in 0..w / VIT_MERGE {
                for ih in 0..VIT_MERGE {
                    for iw in 0..VIT_MERGE {
                        ids.push((mh * VIT_MERGE + ih) as u32);
                        ids.push((mw * VIT_MERGE + iw) as u32);
                    }
                }
            }
        }
    }
    ids
}

// ---------------------------------------------------------------------------------------------------------------------
// Flow-matching time conditioning
// ---------------------------------------------------------------------------------------------------------------------

/// `torch.linspace(0, 1, n)` in f32.
fn unit_linspace(n: usize) -> Vec<f32> {
    match n {
        0 => Vec::new(),
        1 => vec![0.0],
        _ => {
            let step = 1.0f32 / (n - 1) as f32;
            let half = n / 2;
            (0..n)
                .map(|i| {
                    if i < half {
                        step.mul_add(i as f32, 0.0)
                    } else {
                        (-step).mul_add((n - i - 1) as f32, 1.0)
                    }
                })
                .collect()
        }
    }
}

/// Upstream `create_sinusoidal_pos_embedding` for one scalar time: `[sin(half), cos(half)]`, f32, `dim` even.
///
/// `period_i = min * (max / min) ** linspace(0, 1, dim / 2)[i]`, `x_i = (1 / period_i * 2 * pi) * t`. The reference
/// computes `max / min` on Python doubles, so `min_period` / `max_period` are read back as the decimal literals they
/// were written as (`4e-3`, `4.0`) rather than as their f32 approximations; taking the f32 quotient instead moves the
/// ratio by one ulp and shifts every large argument of sin/cos. See [`sinusoidal_time_embedding_f64`].
pub fn sinusoidal_time_embedding(t: f32, dim: usize, min_period: f32, max_period: f32) -> Vec<f32> {
    let literal = |v: f32| format!("{v}").parse::<f64>().unwrap_or(f64::from(v));
    sinusoidal_time_embedding_f64(t, dim, literal(min_period), literal(max_period))
}

/// [`sinusoidal_time_embedding`] with the periods given as doubles.
pub fn sinusoidal_time_embedding_f64(
    t: f32,
    dim: usize,
    min_period: f64,
    max_period: f64,
) -> Vec<f32> {
    assert!(
        dim >= 2 && dim.is_multiple_of(2),
        "dimension ({dim}) must be divisible by 2"
    );
    let half = dim / 2;
    let ratio = (max_period / min_period) as f32;
    let min = min_period as f32;
    let mut out = vec![0.0f32; dim];
    for (i, fraction) in unit_linspace(half).into_iter().enumerate() {
        let period = min * f64::from(ratio).powf(f64::from(fraction)) as f32;
        let scale = (1.0f32 / period * 2.0) * std::f32::consts::PI;
        let x = f64::from(scale * t);
        out[i] = x.sin() as f32;
        out[half + i] = x.cos() as f32;
    }
    out
}

/// Round every value to bf16 and back (the reference casts the time embedding to the state dtype).
pub fn bf16_round(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| round_bf16(x)).collect()
}

/// The `(times, dt)` the reference sampling loop runs with bf16 scalar tensors:
/// `dt = bf16(-1 / num_steps)`, `t = bf16(1)`, `while t >= -dt / 2 { push(t); t = bf16(t + dt) }`.
///
/// For 10 steps this is 1.0, 0.8984375, 0.796875, 0.6953125, 0.59375, 0.494140625, 0.39453125, 0.294921875, 0.1953125,
/// 0.09521484375 with `dt = -0.10009765625`: bf16 drift makes the grid slightly non-uniform, and the model must use
/// these exact values to reproduce the reference. `num_steps == 0` returns no steps. Where bf16 stalls (`t + dt == t`,
/// possible for very large `num_steps`) the reference would never terminate; this stops there.
pub fn denoise_schedule(num_steps: usize) -> (Vec<f32>, f32) {
    if num_steps == 0 {
        return (Vec::new(), 0.0);
    }
    let dt = bf16::from_f32(-1.0 / num_steps as f32);
    let threshold = -dt.to_f32() / 2.0; // exact in bf16: halving only changes the exponent
    let mut t = bf16::from_f32(1.0);
    let mut times = Vec::with_capacity(num_steps + 1);
    while t.to_f32() >= threshold && times.len() <= 2 * num_steps + 2 {
        times.push(t.to_f32());
        let next = bf16::from_f32(t.to_f32() + dt.to_f32());
        if next == t {
            break;
        }
        t = next;
    }
    (times, dt.to_f32())
}

/// `y = w @ x (+ bias)` with `w` row-major `[rows, cols]`, sequential f32 accumulation. For load-time folding of small
/// linears (AdaRMS gamma/beta, the time half of `action_time_mlp_in`); not tuned for large shapes.
pub fn matvec_f32(
    w: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
    bias: Option<&[f32]>,
) -> Vec<f32> {
    assert_eq!(w.len(), rows * cols, "weight is not [{rows}, {cols}]");
    assert_eq!(x.len(), cols, "input length {} != {cols}", x.len());
    if let Some(bias) = bias {
        assert_eq!(bias.len(), rows, "bias length {} != {rows}", bias.len());
    }
    (0..rows)
        .map(|r| {
            let row = &w[r * cols..(r + 1) * cols];
            let mut acc = bias.map_or(0.0, |b| b[r]);
            for (a, b) in row.iter().zip(x) {
                acc += a * b;
            }
            acc
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_index_single_merged_token_grid() {
        assert_eq!(
            rope_index(&[[1, 2, 2]], 0, 0),
            vec![[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        );
    }

    #[test]
    fn rope_index_two_by_two_merged_grid() {
        let got = rope_index(&[[1, 4, 4]], 1, 0);
        assert_eq!(
            got,
            vec![
                [0, 0, 0],
                [1, 1, 1],
                [1, 1, 2],
                [1, 2, 1],
                [1, 2, 2],
                [3, 3, 3],
                [4, 4, 4]
            ]
        );
    }

    #[test]
    fn rope_index_two_images_share_the_text_between_them() {
        let got = rope_index(&[[1, 2, 2], [1, 2, 2]], 0, 0);
        // start, img, end, start, img, end
        assert_eq!(
            got,
            vec![
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [4, 4, 4],
                [5, 5, 5]
            ]
        );
    }

    #[test]
    fn rope_index_without_images_is_plain_text() {
        assert_eq!(rope_index(&[], 2, 1), vec![[0, 0, 0], [1, 1, 1], [2, 2, 2]]);
        assert!(rope_index(&[], 0, 0).is_empty());
    }

    #[test]
    fn rope_index_length_matches_layout() {
        let grids = [[1, 16, 16]; 3];
        assert_eq!(rope_index(&grids, 16, 16).len(), 3 * 66 + 16 + 16);
        assert_eq!(rope_index(&grids, 0, 16).len(), 3 * 66 + 16);
    }

    #[test]
    fn suffix_offset_is_max_plus_one() {
        assert_eq!(suffix_position_offset(&[[0, 5, 2], [9, 1, 1]]), 10);
        assert_eq!(suffix_position_offset(&[]), 0);
    }

    #[test]
    fn schedule_ten_steps_is_the_bf16_grid() {
        let (times, dt) = denoise_schedule(10);
        assert_eq!(
            times,
            vec![
                1.0,
                0.8984375,
                0.796875,
                0.6953125,
                0.59375,
                0.494140625,
                0.39453125,
                0.294921875,
                0.1953125,
                0.09521484375
            ]
        );
        assert_eq!(dt, -0.10009765625);
    }

    #[test]
    fn schedule_small_and_degenerate_counts() {
        assert_eq!(denoise_schedule(1), (vec![1.0], -1.0));
        assert_eq!(denoise_schedule(2), (vec![1.0, 0.5], -0.5));
        assert_eq!(denoise_schedule(0), (vec![], 0.0));
        // bf16 cannot resolve 1 - 2^-10: terminates instead of looping forever
        let (times, _) = denoise_schedule(1024);
        assert!(!times.is_empty() && times.len() < 4);
    }

    #[test]
    fn linspace_endpoints_and_single_point() {
        for precision in [InterpArithmetic::F32, InterpArithmetic::Bf16] {
            let v = linspace_grid(47.0, 16, precision);
            assert_eq!((v[0], v[15]), (0.0, 47.0));
            assert_eq!(linspace_grid(47.0, 1, precision), vec![0.0]);
            assert!(linspace_grid(47.0, 0, precision).is_empty());
        }
    }

    #[test]
    fn pos_embed_identity_grid_returns_table_rows() {
        // a 48x48 request interpolates at integer positions: every output row is exactly a table row, permuted
        let hidden = 2;
        let table: Vec<bf16> = (0..2304 * hidden)
            .map(|i| bf16::from_f32((i % 251) as f32 / 16.0))
            .collect();
        let out = vit_pos_embeds(&table, [1, 48, 48]);
        let ids = vit_axial_positions([1, 48, 48]);
        for token in 0..48 * 48 {
            let (row, col) = (ids[2 * token] as usize, ids[2 * token + 1] as usize);
            for c in 0..hidden {
                assert_eq!(
                    out[token * hidden + c],
                    table[(row * 48 + col) * hidden + c],
                    "token {token}"
                );
            }
        }
    }

    #[test]
    fn pos_embed_constant_table_is_constant() {
        let table = vec![bf16::from_f32(0.5); 2304 * 4];
        for arithmetic in [InterpArithmetic::F32, InterpArithmetic::Bf16] {
            let out = vit_pos_embeds_with(&table, [1, 12, 20], arithmetic);
            assert_eq!(out.len(), 12 * 20 * 4);
            // weights sum to 1 up to bf16 rounding of the weights
            assert!(
                out.iter().all(|v| (v.to_f32() - 0.5).abs() <= 0.5 * 0.02),
                "{arithmetic:?}"
            );
        }
    }

    #[test]
    fn axial_positions_small_grid() {
        assert_eq!(vit_axial_positions([1, 2, 2]), vec![0, 0, 0, 1, 1, 0, 1, 1]);
        assert_eq!(vit_axial_positions([2, 2, 2]).len(), 16);
    }

    #[test]
    fn time_embedding_is_unit_sin_cos_pairs() {
        let e = sinusoidal_time_embedding(0.5, 768, 4e-3, 4.0);
        assert_eq!(e.len(), 768);
        for i in 0..384 {
            assert!((e[i] * e[i] + e[384 + i] * e[384 + i] - 1.0).abs() < 1e-5);
        }
        let zero = sinusoidal_time_embedding(0.0, 768, 4e-3, 4.0);
        assert!(zero[..384].iter().all(|&v| v == 0.0) && zero[384..].iter().all(|&v| v == 1.0));
    }

    #[test]
    fn matvec_with_and_without_bias() {
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(
            matvec_f32(&w, 2, 3, &[1.0, 0.0, -1.0], None),
            vec![-2.0, -2.0]
        );
        assert_eq!(
            matvec_f32(&w, 2, 3, &[1.0, 0.0, -1.0], Some(&[10.0, 20.0])),
            vec![8.0, 18.0]
        );
    }

    #[test]
    fn bf16_round_rounds_to_nearest_even() {
        assert_eq!(
            bf16_round(&[1.0, 1.00390625, 1.00585938, 3.1]),
            vec![1.0, 1.0, 1.0078125, 3.09375]
        );
    }
}
