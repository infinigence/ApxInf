//! Host arithmetic of the VAE decode that is not tensor math: the tile plan, the blend of overlapping tiles into
//! one canvas, the latent-to-token rearrangement and the final pixel conversion. Ported from ComfyUI's
//! `MiniMaxH3VideoVAE.split_tiles` / `blend` / `tiled_decode` (`comfy/ldm/minimax/vae.py`), which defines the released images.
use crate::cuda_minimax_h3_image::config::*;
use apxinf_core::{Error, Result};

/// Start, length and the overlap with the next tile of every tile along one axis, in pixels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TilePlan {
    pub starts: Vec<usize>,
    pub lens: Vec<usize>,
    pub overlaps: Vec<usize>,
}

impl TilePlan {
    /// `split_tiles(input_len)`: tiles of `VAE_TILE_PIXELS` overlapping by at least `VAE_TILE_OVERLAP`, widened in
    /// multiples of the VAE ratio so the tiles cover the axis exactly.
    pub fn new(input_len: usize) -> Self {
        let tile = VAE_TILE_PIXELS;
        if tile >= input_len {
            return Self {
                starts: vec![0],
                lens: vec![input_len],
                overlaps: vec![],
            };
        }
        let mut n = input_len.div_ceil(tile);
        let remaining = loop {
            let covered = tile * n;
            let minimum = VAE_TILE_OVERLAP * (n - 1);
            if covered >= minimum + input_len {
                break covered - minimum - input_len;
            }
            n += 1;
        };
        let mut overlaps = vec![VAE_TILE_OVERLAP; n - 1];
        for i in 0..remaining / VAE_RATIO {
            overlaps[i % (n - 1)] += VAE_RATIO;
        }
        let mut starts = vec![0];
        for overlap in &overlaps {
            starts.push(starts.last().unwrap() + tile - overlap);
        }
        Self {
            lens: vec![tile; n],
            starts,
            overlaps,
        }
    }

    pub fn count(&self) -> usize {
        self.starts.len()
    }
}

/// A `[channels, height, width]` FP32 image.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Chw {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Chw {
    pub fn zeros(c: usize, h: usize, w: usize) -> Self {
        Self {
            c,
            h,
            w,
            data: vec![0.0; c * h * w],
        }
    }

    fn at(&self, c: usize, y: usize, x: usize) -> f32 {
        self.data[(c * self.h + y) * self.w + x]
    }

    /// Rows `y0..y1`, columns `x0..x1`.
    fn crop(&self, y0: usize, y1: usize, x0: usize, x1: usize) -> Self {
        let mut out = Self::zeros(self.c, y1 - y0, x1 - x0);
        for c in 0..self.c {
            for y in y0..y1 {
                let src = (c * self.h + y) * self.w;
                let dst = (c * out.h + (y - y0)) * out.w;
                out.data[dst..dst + out.w].copy_from_slice(&self.data[src + x0..src + x1]);
            }
        }
        out
    }
}

/// Which axis a blend runs along.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Axis {
    Rows,
    Cols,
}

/// `blend(a, b, extent, dim)`: the last `extent` entries of `a` fade into the first `extent` entries of `b`
/// with weights `1 - p / extent` and `p / extent`; the rest of `b` follows unchanged.
fn blend(a: &Chw, b: &Chw, extent: usize, axis: Axis) -> Chw {
    let (a_len, b_len) = if axis == Axis::Rows {
        (a.h, b.h)
    } else {
        (a.w, b.w)
    };
    let extent = extent.min(a_len).min(b_len);
    let mut out = b.clone();
    for c in 0..b.c {
        for y in 0..b.h {
            for x in 0..b.w {
                let position = if axis == Axis::Rows { y } else { x };
                if position >= extent {
                    continue;
                }
                let (wa, wb) = (
                    1.0 - position as f32 / extent as f32,
                    position as f32 / extent as f32,
                );
                let (ay, ax) = if axis == Axis::Rows {
                    (a.h - extent + y, x)
                } else {
                    (y, a.w - extent + x)
                };
                out.data[(c * b.h + y) * b.w + x] = a.at(c, ay, ax) * wa + b.at(c, y, x) * wb;
            }
        }
    }
    out
}

/// Blend decoded tiles into one `[channels, height, width]` canvas exactly as `tiled_decode` does: every tile is blended
/// against its upper neighbour's strip and its left neighbour's tail as already blended, then cropped to the part
/// it owns. `decode_row(row, x_starts, x_lens)` returns the decoded tiles of one tile row in column order.
pub(crate) fn assemble(
    height: usize,
    width: usize,
    mut decode_row: impl FnMut(usize, &TilePlan) -> Result<Vec<Chw>>,
) -> Result<Chw> {
    let (ys, xs) = (TilePlan::new(height), TilePlan::new(width));
    let mut canvas: Option<Chw> = None;
    let mut strip: Option<Chw> = None;
    let mut out_y = 0;
    for i in 0..ys.count() {
        let tiles = decode_row(i, &xs)?;
        if tiles.len() != xs.count() {
            return Err(Error::Other(format!(
                "VAE tile row {i} returned {} tiles, expected {}",
                tiles.len(),
                xs.count()
            )));
        }
        let mut new_strip: Option<Chw> = None;
        let mut left_tail: Option<Chw> = None;
        let (mut out_x, mut tile_height) = (0, 0);
        for (j, mut tile) in tiles.into_iter().enumerate() {
            if tile.h != ys.lens[i] || tile.w != xs.lens[j] {
                return Err(Error::Other(format!(
                    "VAE tile ({i}, {j}) is {}x{}, expected {}x{}",
                    tile.h, tile.w, ys.lens[i], xs.lens[j]
                )));
            }
            if i > 0 {
                let above = strip.as_ref().unwrap();
                let a = above.crop(0, above.h, xs.starts[j], xs.starts[j] + xs.lens[j]);
                tile = blend(&a, &tile, ys.overlaps[i - 1], Axis::Rows);
            }
            if j > 0 {
                tile = blend(
                    left_tail.as_ref().unwrap(),
                    &tile,
                    xs.overlaps[j - 1],
                    Axis::Cols,
                );
            }
            left_tail =
                (j + 1 < xs.count()).then(|| tile.crop(0, tile.h, tile.w - xs.overlaps[j], tile.w));
            if j + 1 < xs.count() {
                tile = tile.crop(0, tile.h, 0, tile.w - xs.overlaps[j]);
            }
            let canvas = canvas.get_or_insert_with(|| Chw::zeros(tile.c, height, width));
            if i + 1 < ys.count() {
                let overlap = ys.overlaps[i];
                let strip_rows =
                    new_strip.get_or_insert_with(|| Chw::zeros(tile.c, overlap, width));
                for c in 0..tile.c {
                    for y in 0..overlap {
                        let src = (c * tile.h + (tile.h - overlap + y)) * tile.w;
                        let dst = (c * overlap + y) * width + out_x;
                        strip_rows.data[dst..dst + tile.w]
                            .copy_from_slice(&tile.data[src..src + tile.w]);
                    }
                }
                tile = tile.crop(0, tile.h - overlap, 0, tile.w);
            }
            for c in 0..tile.c {
                for y in 0..tile.h {
                    let src = (c * tile.h + y) * tile.w;
                    let dst = (c * height + out_y + y) * width + out_x;
                    canvas.data[dst..dst + tile.w].copy_from_slice(&tile.data[src..src + tile.w]);
                }
            }
            tile_height = tile.h;
            out_x += tile.w;
        }
        strip = new_strip;
        out_y += tile_height;
    }
    canvas.ok_or_else(|| Error::Other("VAE decode produced no tiles".into()))
}

/// Tokens of one decoded tile: the latent pixels plus the register tokens and one zero token.
pub(crate) const TILE_TOKENS: usize = VAE_TILE_LATENT * VAE_TILE_LATENT + VAE_SUFFIX;

/// Rotary table of one tile for `rms_norm_apply_precomputed`: FP32 `[TILE_TOKENS, 2, VAE_HEAD_DIM / 2]`, cosine then sine,
/// for the half-split rotation over the channel-permuted head. Pair `c < 16` rotates by
/// `2 pi * id[c / 8] * inv_freq[c % 8]` (ComfyUI `RotaryEmbeddingND`, theta 100, ids `(arange(0.5, n) / n) * 2 - 1`
/// per axis, zero for the suffix tokens); pairs `16..32` are pass-through (cos 1, sin 0). Values are rounded to FP16 as the reference
/// rounds its table to the decoder dtype.
pub(crate) fn vae_rope_table() -> Vec<f32> {
    const PAIRS: usize = VAE_HEAD_DIM / 2;
    const AXES: usize = 2;
    const FREQS: usize = VAE_ROPE_DIM / 2 / AXES;
    let inv_freq: Vec<f32> = (0..FREQS)
        .map(|f| 1.0 / VAE_ROPE_THETA.powf(f as f32 * (2.0 * AXES as f32 / VAE_ROPE_DIM as f32)))
        .collect();
    let n = VAE_TILE_LATENT;
    let coord = |k: usize| ((k as f32 + 0.5) / n as f32) * 2.0 - 1.0;
    let angle_scale = std::f64::consts::TAU as f32;
    let mut table = vec![0.0f32; TILE_TOKENS * 2 * PAIRS];
    for token in 0..TILE_TOKENS {
        let ids = if token < n * n {
            [coord(token / n), coord(token % n)]
        } else {
            [0.0, 0.0]
        };
        let (cos, sin) = table[token * 2 * PAIRS..(token + 1) * 2 * PAIRS].split_at_mut(PAIRS);
        for c in 0..PAIRS {
            let (co, si) = if c < AXES * FREQS {
                let angle = angle_scale * ids[c / FREQS] * inv_freq[c % FREQS];
                (angle.cos(), angle.sin())
            } else {
                (1.0, 0.0)
            };
            cos[c] = half::f16::from_f32(co).to_f32();
            sin[c] = half::f16::from_f32(si).to_f32();
        }
    }
    table
}

/// Latent `[LATENT_CHANNELS, h, w]` rows of one tile, token-major `[tile_h * tile_w, LATENT_CHANNELS]`:
/// token `(ty, tx)` is the channel vector at latent pixel `(y0 + ty, x0 + tx)`.
pub(crate) fn latent_tile_tokens(
    latent: &Chw,
    y0: usize,
    x0: usize,
    tile_h: usize,
    tile_w: usize,
    out: &mut Vec<f32>,
) {
    for ty in 0..tile_h {
        for tx in 0..tile_w {
            for c in 0..latent.c {
                out.push(latent.at(c, y0 + ty, x0 + tx));
            }
        }
    }
}

/// Decoder output tokens `[tile_h * tile_w, OUT]` (`OUT = PIXEL_CHANNELS * VAE_RATIO^2`, features `(c, py, px)`) to a
/// `[PIXEL_CHANNELS, tile_h * VAE_RATIO, tile_w * VAE_RATIO]` tile (`unpatchify` of `comfy/ldm/minimax/vae.py`).
pub(crate) fn unpatchify_pixels(tokens: &[f32], tile_h: usize, tile_w: usize) -> Chw {
    assert_eq!(tokens.len(), tile_h * tile_w * VAE_OUT_DIM);
    let mut tile = Chw::zeros(PIXEL_CHANNELS, tile_h * VAE_RATIO, tile_w * VAE_RATIO);
    for ty in 0..tile_h {
        for tx in 0..tile_w {
            let row = &tokens[(ty * tile_w + tx) * VAE_OUT_DIM..][..VAE_OUT_DIM];
            for c in 0..PIXEL_CHANNELS {
                for py in 0..VAE_RATIO {
                    for px in 0..VAE_RATIO {
                        let (y, x) = (ty * VAE_RATIO + py, tx * VAE_RATIO + px);
                        tile.data[(c * tile.h + y) * tile.w + x] =
                            row[(c * VAE_RATIO + py) * VAE_RATIO + px];
                    }
                }
            }
        }
    }
    tile
}

/// Decoder-space canvas to RGB8: `v * std + mean` per channel, clamped to `[0, 1]`, scaled and rounded half to even
/// like `torch.round`. Output is row-major `[height, width, 3]`.
pub(crate) fn to_rgb8(canvas: &Chw) -> Vec<u8> {
    assert_eq!(canvas.c, PIXEL_CHANNELS);
    let mut rgb = vec![0u8; canvas.h * canvas.w * PIXEL_CHANNELS];
    for c in 0..PIXEL_CHANNELS {
        for y in 0..canvas.h {
            for x in 0..canvas.w {
                let v = (canvas.at(c, y, x) * PIXEL_STD[c] + PIXEL_MEAN[c]).clamp(0.0, 1.0);
                rgb[(y * canvas.w + x) * PIXEL_CHANNELS + c] = (v * 255.0).round_ties_even() as u8;
            }
        }
    }
    rgb
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `scripts/vae_tile_reference.py` (ComfyUI `split_tiles`) for a few axis lengths.
    #[test]
    fn tile_plans_match_comfyui() {
        let p = TilePlan::new(1024);
        assert_eq!(
            (p.starts.clone(), p.overlaps.clone()),
            (vec![0, 192, 384, 576, 768], vec![64; 4])
        );
        let p = TilePlan::new(2048);
        assert_eq!(
            p.starts,
            vec![0, 176, 352, 528, 704, 880, 1056, 1232, 1408, 1600, 1792]
        );
        assert_eq!(p.overlaps, vec![80, 80, 80, 80, 80, 80, 80, 80, 64, 64]);
        let p = TilePlan::new(288);
        assert_eq!((p.starts, p.overlaps), (vec![0, 32], vec![224]));
        let p = TilePlan::new(256);
        assert_eq!((p.starts, p.lens, p.overlaps), (vec![0], vec![256], vec![]));
        for len in (256..=4096).step_by(32) {
            let p = TilePlan::new(len);
            assert_eq!(
                p.starts.last().unwrap() + p.lens.last().unwrap(),
                len,
                "tiles must end exactly at {len}"
            );
            assert!(p
                .overlaps
                .iter()
                .all(|o| o % VAE_RATIO == 0 && *o >= VAE_TILE_OVERLAP.min(len - 1)));
        }
    }

    /// Synthetic tile of `scripts/vae_assemble_reference.py`.
    fn tile_value(i: usize, j: usize, c: usize, y: usize, x: usize) -> f32 {
        (0.3 * i as f64 - 0.2 * j as f64 + 0.1 * c as f64) as f32 + 0.002 * y as f32
            - 0.0015 * x as f32
            + 0.5
    }

    fn synthetic(height: usize, width: usize) -> Chw {
        assemble(height, width, |i, xs| {
            Ok((0..xs.count())
                .map(|j| {
                    let mut t = Chw::zeros(PIXEL_CHANNELS, VAE_TILE_PIXELS.min(height), xs.lens[j]);
                    for c in 0..PIXEL_CHANNELS {
                        for y in 0..t.h {
                            for x in 0..t.w {
                                t.data[(c * t.h + y) * t.w + x] = tile_value(i, j, c, y, x);
                            }
                        }
                    }
                    t
                })
                .collect())
        })
        .unwrap()
    }

    fn check(canvas: &Chw, probes: &[[f64; 4]]) {
        for p in probes {
            let (c, y, x) = (p[0] as usize, p[1] as usize, p[2] as usize);
            let got = canvas.at(c, y, x) as f64;
            assert!(
                (got - p[3]).abs() < 5e-5,
                "c={c} y={y} x={x}: got {got}, ComfyUI {}",
                p[3]
            );
        }
    }

    // Probe values produced by ComfyUI's own `tiled_decode` on the same synthetic tiles (`scripts/vae_assemble_reference.py`).
    #[test]
    fn assembly_matches_comfyui_on_a_two_by_two_grid_with_large_overlaps() {
        let canvas = synthetic(288, 416);
        assert_eq!((canvas.h, canvas.w), (288, 416));
        check(
            &canvas,
            &[
                [0., 0., 0., 0.5],
                [1., 0., 0., 0.6],
                [2., 0., 0., 0.7],
                [0., 30., 138., 0.353],
                [1., 31., 138., 0.455],
                [0., 31., 159., 0.3235],
                [2., 32., 0., 0.764],
                [0., 32., 138., 0.357],
                [0., 33., 138., 0.360054],
                [1., 33., 161., 0.42597],
                [2., 34., 138., 0.563107],
                [0., 96., 158., 0.522429],
                [1., 96., 159., 0.620929],
                [2., 96., 160., 0.719429],
                [0., 96., 161., 0.518345],
                [1., 96., 162., 0.617262],
                [0., 144., 208., 0.614],
                [2., 287., 415., 0.9275],
            ],
        );
    }

    #[test]
    fn assembly_matches_comfyui_on_a_three_by_three_grid() {
        let canvas = synthetic(608, 608);
        check(
            &canvas,
            &[
                [0., 0., 0., 0.5],
                [1., 174., 202., 0.6658],
                [2., 175., 175., 0.7875],
                [0., 175., 351., 0.3875],
                [1., 176., 0., 0.952],
                [2., 176., 202., 0.7698],
                [0., 177., 202., 0.57115],
                [1., 177., 353., 0.48865],
                [2., 178., 202., 0.7725],
                [0., 202., 175., 0.6246],
                [1., 202., 176., 0.7231],
                [2., 202., 177., 0.8224],
                [0., 202., 350., 0.4261],
                [1., 202., 351., 0.5246],
                [2., 202., 352., 0.6231],
                [0., 202., 354., 0.4217],
                [1., 304., 304., 0.764],
                [2., 350., 202., 1.0658],
                [0., 351., 202., 0.8678],
                [1., 351., 351., 0.7875],
                [2., 352., 0., 1.352],
                [0., 353., 177., 0.88865],
                [1., 353., 202., 0.97115],
                [2., 353., 353., 0.88865],
                [0., 607., 607., 0.8275],
            ],
        );
    }

    #[test]
    fn single_tile_axis_passes_through() {
        let canvas = synthetic(256, 256);
        assert_eq!(canvas.data[0], 0.5);
        assert!(
            (canvas.at(2, 255, 255) as f64 - (0.2 + 0.002 * 255.0 - 0.0015 * 255.0 + 0.5)).abs()
                < 1e-5
        );
    }

    #[test]
    fn vae_rope_table_rotates_two_axes_and_passes_the_rest_through() {
        let table = vae_rope_table();
        assert_eq!(table.len(), TILE_TOKENS * 2 * 32);
        let row = |t: usize| &table[t * 64..(t + 1) * 64];
        // suffix tokens have id 0 on both axes: every angle is 0
        for t in VAE_TILE_LATENT * VAE_TILE_LATENT..TILE_TOKENS {
            assert!(
                row(t)[..32].iter().all(|c| *c == 1.0) && row(t)[32..].iter().all(|s| *s == 0.0)
            );
        }
        // pass-through pairs for every token
        for t in 0..TILE_TOKENS {
            assert!(
                row(t)[16..32].iter().all(|c| *c == 1.0) && row(t)[48..].iter().all(|s| *s == 0.0)
            );
        }
        // token (0, 0): ids are -1 + 1/16 on both axes, frequency 0 is 1: angle = 2 pi * (-0.9375)
        let angle = std::f64::consts::TAU * -0.9375;
        assert!(
            (row(0)[0] as f64 - angle.cos()).abs() < 1e-3
                && (row(0)[32] as f64 - angle.sin()).abs() < 1e-3
        );
        // the second axis (pairs 8..16) of token (0, 1) uses id -1 + 3/16 and frequency 0 at pair 8
        let angle = std::f64::consts::TAU * (-1.0 + 3.0 / 16.0);
        assert!(
            (row(1)[8] as f64 - angle.cos()).abs() < 1e-3
                && (row(1)[40] as f64 - angle.sin()).abs() < 1e-3
        );
        // and the first axis of token (0, 1) is unchanged from token (0, 0)
        assert_eq!(row(1)[..8], row(0)[..8]);
    }

    #[test]
    fn latent_tokens_are_channel_vectors_in_raster_order() {
        let mut latent = Chw::zeros(LATENT_CHANNELS, 4, 5);
        for (i, v) in latent.data.iter_mut().enumerate() {
            *v = i as f32;
        }
        let mut tokens = Vec::new();
        latent_tile_tokens(&latent, 1, 2, 2, 3, &mut tokens);
        assert_eq!(tokens.len(), 6 * LATENT_CHANNELS);
        // token 0 is latent pixel (1, 2); its channel 5 is element (5 * 4 + 1) * 5 + 2
        assert_eq!(tokens[5], ((5 * 4 + 1) * 5 + 2) as f32);
        // token 4 is (ty=1, tx=1) = latent pixel (2, 3)
        assert_eq!(tokens[4 * LATENT_CHANNELS], (2 * 5 + 3) as f32);
    }

    #[test]
    fn pixel_unpatchify_orders_features_channel_then_py_then_px() {
        let (th, tw) = (1, 2);
        let tokens: Vec<f32> = (0..th * tw * VAE_OUT_DIM).map(|i| i as f32).collect();
        let tile = unpatchify_pixels(&tokens, th, tw);
        assert_eq!((tile.c, tile.h, tile.w), (3, 16, 32));
        // token 1 (tx = 1), channel 2, py 3, px 4
        assert_eq!(
            tile.at(2, 3, 16 + 4),
            (VAE_OUT_DIM + (2 * VAE_RATIO + 3) * VAE_RATIO + 4) as f32
        );
    }

    #[test]
    fn rgb_conversion_denormalises_clamps_and_rounds_half_to_even() {
        let mut canvas = Chw::zeros(3, 1, 4);
        // channel 0: values chosen so the denormalised result is exactly 0, 0.5 (127.5 -> 128), 1.5 and 1
        for (x, target) in [0.0f32, 0.5, 1.5, 1.0].into_iter().enumerate() {
            canvas.data[x] = (target - PIXEL_MEAN[0]) / PIXEL_STD[0];
        }
        let rgb = to_rgb8(&canvas);
        assert_eq!(rgb[0], 0);
        assert!(
            rgb[3] == 128 || rgb[3] == 127,
            "half-way value must round to an adjacent integer, got {}",
            rgb[3]
        );
        assert_eq!(rgb[6], 255, "out-of-range values clamp");
        assert_eq!(rgb[9], 255);
        // channel 1 and 2 were 0 -> denormalised to the mean
        assert_eq!(rgb[1], (PIXEL_MEAN[1] * 255.0).round_ties_even() as u8);
    }
}
