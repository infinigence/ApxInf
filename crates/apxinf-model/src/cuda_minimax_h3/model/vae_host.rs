//! Host arithmetic of the video VAE decode that is not tensor math: the spatial tile plan, the temporal chunk plan, the blends of
//! overlapping tiles and chunks, the rotary table, and the rearrangements between latents, tokens and pixels. Ported from ComfyUI's
//! `MiniMaxH3VideoVAE` (`comfy/ldm/minimax/vae.py`: `split_tiles`, `blend`, `tiled_decode`, `decode_temporal`), which defines the released
//! videos.
use crate::cuda_minimax_h3::config::*;
use apxinf_core::{Error, Result};

/// Start, length and the overlap with the next tile of every tile along one spatial axis, in pixels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TilePlan {
    pub starts: Vec<usize>,
    pub lens: Vec<usize>,
    pub overlaps: Vec<usize>,
}

impl TilePlan {
    /// `split_tiles(input_len)`: tiles of `VV_TILE_PIXELS` overlapping by at least `VV_TILE_OVERLAP`, widened in multiples of the VAE
    /// ratio so the tiles cover the axis exactly.
    pub fn new(input_len: usize) -> Self {
        let tile = VV_TILE_PIXELS;
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
            let minimum = VV_TILE_OVERLAP * (n - 1);
            if covered >= minimum + input_len {
                break covered - minimum - input_len;
            }
            n += 1;
        };
        let mut overlaps = vec![VV_TILE_OVERLAP; n - 1];
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

/// How the latent steps of a clip are cut into decoder chunks, and how many pixel frames come out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TemporalPlan {
    /// Latent steps appended (copies of the last) so the chunks tile the clip.
    pub pad_tokens: usize,
    pub chunks: usize,
    /// Pixel frames of the decoded clip.
    pub frames: usize,
}

impl TemporalPlan {
    /// `_decode_temporal_chunks` and `_decode_temporal_frame_plan` for `z_len >= 2` latent steps. A single latent step decodes as
    /// a still and is not supported here.
    pub fn new(z_len: usize) -> Result<Self> {
        if z_len < 2 {
            return Err(Error::Other(
                "MiniMax-H3 video decode needs at least two latent steps".into(),
            ));
        }
        let mut pad_tokens =
            (VV_CHUNK_TOKENS - (z_len + VV_TOKEN_DROP) % VV_CHUNK_TOKENS) % VV_CHUNK_TOKENS;
        let pseudo_total = z_len + VV_TOKEN_DROP + pad_tokens;
        let mut chunks = pseudo_total / VV_CHUNK_TOKENS - 1;
        if chunks < 1 {
            pad_tokens += VV_CHUNK_TOKENS;
            chunks += 1;
        }
        let padded = z_len + pad_tokens;
        let chunk_frames = VV_CHUNK_TOKENS * VV_RATIO_T;
        let (mut total, mut final_overlap) = (0usize, 0usize);
        for i in 0..chunks {
            let start = i * VV_CHUNK_TOKENS;
            let end = start + VV_CHUNK_TOKENS + VV_TOKEN_OVERLAP;
            let clip_frames = (end.min(padded) - start.min(padded)) * VV_RATIO_T;
            for j in 0..2 {
                let (f0, f1) = (
                    j * chunk_frames,
                    (j * chunk_frames + chunk_frames).min(clip_frames),
                );
                let kept = (f1.saturating_sub(f0)).saturating_sub(VV_FRAME_PRE_PADDING);
                if j == 0 {
                    total += kept;
                } else {
                    final_overlap = kept;
                }
            }
        }
        total += final_overlap;
        Ok(Self {
            pad_tokens,
            chunks,
            frames: total - pad_frames(z_len, pad_tokens),
        })
    }

    /// Latent steps `[start, end)` of chunk `i` in the padded clip (the last chunk may be shorter than `VV_CHUNK_LATENT`).
    pub fn chunk(&self, i: usize, z_len: usize) -> (usize, usize) {
        let padded = z_len + self.pad_tokens;
        let start = i * VV_CHUNK_TOKENS;
        (start, (start + VV_CHUNK_LATENT).min(padded))
    }
}

/// `_decode_temporal_pad_frames`: pixel frames the appended latent steps contributed, dropped from the end.
fn pad_frames(z_len: usize, pad_tokens: usize) -> usize {
    if pad_tokens == 0 {
        return 0;
    }
    let intra_tail = VV_CLIP_FRAMES % VV_RATIO_T;
    if intra_tail == 0 {
        return pad_tokens * VV_RATIO_T;
    }
    (0..pad_tokens)
        .map(|k| {
            if (z_len + k) % VV_CHUNK_TOKENS == 0 {
                intra_tail
            } else {
                VV_RATIO_T
            }
        })
        .sum()
}

/// A `[channels, frames, height, width]` FP32 volume.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Vid {
    pub c: usize,
    pub t: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Axis {
    Frames,
    Rows,
    Cols,
}

impl Vid {
    pub fn zeros(c: usize, t: usize, h: usize, w: usize) -> Self {
        Self {
            c,
            t,
            h,
            w,
            data: vec![0.0; c * t * h * w],
        }
    }

    fn row(&self, c: usize, t: usize, y: usize) -> &[f32] {
        let start = ((c * self.t + t) * self.h + y) * self.w;
        &self.data[start..start + self.w]
    }

    pub fn at(&self, c: usize, t: usize, y: usize, x: usize) -> f32 {
        self.data[((c * self.t + t) * self.h + y) * self.w + x]
    }

    /// Frames `t0..t1`, rows `y0..y1`, columns `x0..x1`.
    pub fn crop(&self, t0: usize, t1: usize, y0: usize, y1: usize, x0: usize, x1: usize) -> Self {
        let mut out = Self::zeros(self.c, t1 - t0, y1 - y0, x1 - x0);
        for c in 0..self.c {
            for t in t0..t1 {
                for y in y0..y1 {
                    let dst = (((c * out.t) + (t - t0)) * out.h + (y - y0)) * out.w;
                    out.data[dst..dst + out.w].copy_from_slice(&self.row(c, t, y)[x0..x1]);
                }
            }
        }
        out
    }

    fn len_along(&self, axis: Axis) -> usize {
        match axis {
            Axis::Frames => self.t,
            Axis::Rows => self.h,
            Axis::Cols => self.w,
        }
    }
}

/// `blend(a, b, extent, dim)`: the last `extent` entries of `a` along `axis` fade into the first `extent` entries of `b` with weights
/// `1 - p / extent` and `p / extent`; the rest of `b` follows unchanged.
pub(crate) fn blend(a: &Vid, b: &Vid, extent: usize, axis: Axis) -> Vid {
    let extent = extent.min(a.len_along(axis)).min(b.len_along(axis));
    let mut out = b.clone();
    for c in 0..b.c {
        for t in 0..b.t {
            for y in 0..b.h {
                for x in 0..b.w {
                    let position = match axis {
                        Axis::Frames => t,
                        Axis::Rows => y,
                        Axis::Cols => x,
                    };
                    if position >= extent {
                        continue;
                    }
                    let (wa, wb) = (
                        1.0 - position as f32 / extent as f32,
                        position as f32 / extent as f32,
                    );
                    let (at, ay, ax) = match axis {
                        Axis::Frames => (a.t - extent + t, y, x),
                        Axis::Rows => (t, a.h - extent + y, x),
                        Axis::Cols => (t, y, a.w - extent + x),
                    };
                    out.data[((c * b.t + t) * b.h + y) * b.w + x] =
                        a.at(c, at, ay, ax) * wa + b.at(c, t, y, x) * wb;
                }
            }
        }
    }
    out
}

/// Blend decoded tiles of one chunk into one `[channels, frames, height, width]` canvas exactly as `tiled_decode` does: every tile is
/// blended against its upper neighbour's strip and its left neighbour's tail as already blended, then cropped to the part it owns.
/// `decode_row(row, columns)` returns the decoded tiles of one tile row in column order.
pub(crate) fn assemble(
    height: usize,
    width: usize,
    mut decode_row: impl FnMut(usize, &TilePlan) -> Result<Vec<Vid>>,
) -> Result<Vid> {
    let (ys, xs) = (TilePlan::new(height), TilePlan::new(width));
    let mut canvas: Option<Vid> = None;
    let mut strip: Option<Vid> = None;
    let mut out_y = 0;
    for i in 0..ys.count() {
        let tiles = decode_row(i, &xs)?;
        if tiles.len() != xs.count() {
            return Err(Error::Other(format!(
                "video tile row {i} returned {} tiles, expected {}",
                tiles.len(),
                xs.count()
            )));
        }
        let mut new_strip: Option<Vid> = None;
        let mut left_tail: Option<Vid> = None;
        let (mut out_x, mut tile_height) = (0, 0);
        for (j, mut tile) in tiles.into_iter().enumerate() {
            if tile.h != ys.lens[i] || tile.w != xs.lens[j] {
                return Err(Error::Other(format!(
                    "video tile ({i}, {j}) is {}x{}, expected {}x{}",
                    tile.h, tile.w, ys.lens[i], xs.lens[j]
                )));
            }
            if i > 0 {
                let above = strip.as_ref().unwrap();
                let a = above.crop(
                    0,
                    above.t,
                    0,
                    above.h,
                    xs.starts[j],
                    xs.starts[j] + xs.lens[j],
                );
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
            left_tail = (j + 1 < xs.count())
                .then(|| tile.crop(0, tile.t, 0, tile.h, tile.w - xs.overlaps[j], tile.w));
            if j + 1 < xs.count() {
                tile = tile.crop(0, tile.t, 0, tile.h, 0, tile.w - xs.overlaps[j]);
            }
            let canvas = canvas.get_or_insert_with(|| Vid::zeros(tile.c, tile.t, height, width));
            if i + 1 < ys.count() {
                let overlap = ys.overlaps[i];
                let strip_rows =
                    new_strip.get_or_insert_with(|| Vid::zeros(tile.c, tile.t, overlap, width));
                for c in 0..tile.c {
                    for t in 0..tile.t {
                        for y in 0..overlap {
                            let src = ((c * tile.t + t) * tile.h + (tile.h - overlap + y)) * tile.w;
                            let dst = ((c * tile.t + t) * overlap + y) * width + out_x;
                            strip_rows.data[dst..dst + tile.w]
                                .copy_from_slice(&tile.data[src..src + tile.w]);
                        }
                    }
                }
                tile = tile.crop(0, tile.t, 0, tile.h - overlap, 0, tile.w);
            }
            for c in 0..tile.c {
                for t in 0..tile.t {
                    for y in 0..tile.h {
                        let src = ((c * tile.t + t) * tile.h + y) * tile.w;
                        let dst = ((c * tile.t + t) * height + out_y + y) * width + out_x;
                        canvas.data[dst..dst + tile.w]
                            .copy_from_slice(&tile.data[src..src + tile.w]);
                    }
                }
            }
            tile_height = tile.h;
            out_x += tile.w;
        }
        strip = new_strip;
        out_y += tile_height;
    }
    canvas.ok_or_else(|| Error::Other("video decode produced no tiles".into()))
}

/// Join the decoded chunks of a clip into `[3, frames, h, w]` pixels in `[0, 1]`, as `decode_temporal` does: each chunk's decoded
/// frames are split at 20 (the first 17 kept after dropping 3 frames of pre-padding, the next 5 held back and blended into the following
/// chunk's first 5), and the pixel mean and standard deviation are applied last.
/// `decode_chunk(start, end)` returns the decoded `[3, 4 * (end - start), h, w]` volume of latent steps `start..end` of the padded clip.
pub(crate) fn splice_chunks(
    z_len: usize,
    plan: &TemporalPlan,
    mut decode_chunk: impl FnMut(usize, usize) -> Result<Vid>,
) -> Result<Vid> {
    let chunk_frames = VV_CHUNK_TOKENS * VV_RATIO_T;
    let mut out: Option<Vid> = None;
    let mut written = 0usize;
    let mut held: Option<Vid> = None;
    let write = |part: &Vid, out: &mut Option<Vid>, written: &mut usize| {
        let out = out.get_or_insert_with(|| Vid::zeros(part.c, plan.frames, part.h, part.w));
        let take = part.t.min(plan.frames.saturating_sub(*written));
        for c in 0..part.c {
            for t in 0..take {
                let src = (c * part.t + t) * part.h * part.w;
                let dst = (c * plan.frames + *written + t) * part.h * part.w;
                for (d, v) in out.data[dst..dst + part.h * part.w]
                    .iter_mut()
                    .zip(&part.data[src..src + part.h * part.w])
                {
                    *d = (v * PIXEL_STD[c] + PIXEL_MEAN[c]).clamp(0.0, 1.0);
                }
            }
        }
        *written += take;
    };
    for i in 0..plan.chunks {
        let (start, end) = plan.chunk(i, z_len);
        let decoded = decode_chunk(start, end)?;
        for j in 0..2 {
            let (f0, f1) = (
                j * chunk_frames,
                (j * chunk_frames + chunk_frames).min(decoded.t),
            );
            if f1 <= f0 + VV_FRAME_PRE_PADDING {
                continue;
            }
            let piece = decoded.crop(f0 + VV_FRAME_PRE_PADDING, f1, 0, decoded.h, 0, decoded.w);
            if j == 0 {
                let piece = match held.take() {
                    Some(previous) => blend(&previous, &piece, VV_FRAME_OVERLAP, Axis::Frames),
                    None => piece,
                };
                write(&piece, &mut out, &mut written);
            } else {
                held = Some(piece);
            }
        }
        if i + 1 == plan.chunks {
            if let Some(tail) = held.take() {
                write(&tail, &mut out, &mut written);
            }
        }
    }
    out.ok_or_else(|| Error::Other("video decode produced no frames".into()))
}

/// Tokens of one tile of `t` latent steps: the latent pixels plus the register tokens and one zero token.
pub(crate) fn tile_tokens(t: usize) -> usize {
    t * VV_TILE_LATENT * VV_TILE_LATENT + VV_SUFFIX
}

/// Rotary table of one tile of `t` latent steps for `rms_norm_apply_precomputed`: FP32 `[tile_tokens(t), 2, VV_HEAD_DIM / 2]`, cosine
/// then sine, for the half-split rotation over the channel-permuted head (see `weights::vae::vae_rope_permutation`). Pair `p < 24`
/// rotates by `2 pi * id[p / 8] * inv_freq[p % 8]` (ComfyUI `RotaryEmbeddingND`, theta 100; ids `(arange(0.5, n) / n) * 2 - 1` per axis
/// computed in FP16, zero for the suffix tokens); pairs `24..32` are pass-through (cos 1, sin 0). The table is rounded to FP16 as the
/// reference rounds it to the decoder dtype.
pub(crate) fn vae_rope_table(t: usize) -> Vec<f32> {
    const PAIRS: usize = VV_HEAD_DIM / 2;
    const AXES: usize = 3;
    let inv_freq: Vec<f32> = (0..VV_ROPE_FREQS)
        .map(|f| 1.0 / VV_ROPE_THETA.powf(f as f32 * (2.0 * AXES as f32 / VV_ROPE_DIM as f32)))
        .collect();
    let f16 = |v: f32| half::f16::from_f32(v).to_f32();
    // coordinates are built in FP16: arange(0.5, n) / n, then 2 x - 1, each rounded to FP16
    let coord = |k: usize, n: usize| f16(f16(f16(k as f32 + 0.5) / n as f32) * 2.0 - 1.0);
    let n = VV_TILE_LATENT;
    let angle_scale = std::f64::consts::TAU as f32;
    let tokens = tile_tokens(t);
    let mut table = vec![0.0f32; tokens * 2 * PAIRS];
    for token in 0..tokens {
        let ids = if token < t * n * n {
            let (ft, rest) = (token / (n * n), token % (n * n));
            [coord(ft, t), coord(rest / n, n), coord(rest % n, n)]
        } else {
            [0.0; 3]
        };
        let (cos, sin) = table[token * 2 * PAIRS..(token + 1) * 2 * PAIRS].split_at_mut(PAIRS);
        for p in 0..PAIRS {
            let (co, si) = if p < AXES * VV_ROPE_FREQS {
                let angle = angle_scale * ids[p / VV_ROPE_FREQS] * inv_freq[p % VV_ROPE_FREQS];
                (angle.cos(), angle.sin())
            } else {
                (1.0, 0.0)
            };
            cos[p] = f16(co);
            sin[p] = f16(si);
        }
    }
    table
}

/// Latent `[LATENT_CHANNELS, t, h, w]` steps `t0..t1` and the `tile_h x tile_w` window at `(y0, x0)` as tokens
/// `[(t1 - t0) * tile_h * tile_w, LATENT_CHANNELS]`: token `(ft, ty, tx)` in raster order is the channel vector at that latent pixel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn latent_tile_tokens(
    latent: &[f32],
    dims: [usize; 3],
    t0: usize,
    t1: usize,
    y0: usize,
    x0: usize,
    tile: [usize; 2],
    out: &mut Vec<f32>,
) {
    let [t, h, w] = dims;
    for ft in t0..t1 {
        for ty in 0..tile[0] {
            for tx in 0..tile[1] {
                for c in 0..LATENT_CHANNELS {
                    out.push(latent[((c * t + ft) * h + y0 + ty) * w + x0 + tx]);
                }
            }
        }
    }
}

/// Decoder output tokens `[t * tile_h * tile_w, VV_OUT_DIM]` (features `(c, pt, py, px)`) to a
/// `[PIXEL_CHANNELS, t * VV_RATIO_T, tile_h * VAE_RATIO, tile_w * VAE_RATIO]` tile (`unpatchify` of `vae.py`).
pub(crate) fn unpatchify_pixels(tokens: &[f32], t: usize, tile_h: usize, tile_w: usize) -> Vid {
    assert_eq!(tokens.len(), t * tile_h * tile_w * VV_OUT_DIM);
    let mut tile = Vid::zeros(
        PIXEL_CHANNELS,
        t * VV_RATIO_T,
        tile_h * VAE_RATIO,
        tile_w * VAE_RATIO,
    );
    for ft in 0..t {
        for ty in 0..tile_h {
            for tx in 0..tile_w {
                let row = &tokens[((ft * tile_h + ty) * tile_w + tx) * VV_OUT_DIM..][..VV_OUT_DIM];
                for c in 0..PIXEL_CHANNELS {
                    for pt in 0..VV_RATIO_T {
                        for py in 0..VAE_RATIO {
                            let (f, y, x0) =
                                (ft * VV_RATIO_T + pt, ty * VAE_RATIO + py, tx * VAE_RATIO);
                            let dst = ((c * tile.t + f) * tile.h + y) * tile.w + x0;
                            let src = ((c * VV_RATIO_T + pt) * VAE_RATIO + py) * VAE_RATIO;
                            tile.data[dst..dst + VAE_RATIO]
                                .copy_from_slice(&row[src..src + VAE_RATIO]);
                        }
                    }
                }
            }
        }
    }
    tile
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ComfyUI's `_decode_temporal_chunks` / `_decode_temporal_frame_plan` for t = 1..14 (`scripts/comfy_h3_vae_plan.py`).
    #[test]
    fn temporal_plan_matches_comfyui() {
        // (t, chunks, pad_tokens, frames)
        let reference = [
            (2, 1, 5, 5),
            (3, 1, 4, 9),
            (4, 1, 3, 13),
            (5, 1, 2, 17),
            (6, 1, 1, 18),
            (7, 1, 0, 22),
            (8, 2, 4, 26),
            (9, 2, 3, 30),
            (10, 2, 2, 34),
            (11, 2, 1, 35),
            (12, 2, 0, 39),
            (13, 3, 4, 43),
            (14, 3, 3, 47),
        ];
        for (t, chunks, pad, frames) in reference {
            let plan = TemporalPlan::new(t).unwrap();
            assert_eq!(
                (plan.chunks, plan.pad_tokens, plan.frames),
                (chunks, pad, frames),
                "t = {t}"
            );
        }
        assert!(TemporalPlan::new(1).is_err());
        // a clip on the 17k + 5 grid has 5k + 2 latent steps and decodes to exactly 17k + 5 frames
        for k in 0..8 {
            assert_eq!(
                TemporalPlan::new(5 * k + 2).unwrap().frames,
                17 * k + 5,
                "k = {k}"
            );
        }
    }

    #[test]
    fn spatial_tile_plans_match_comfyui() {
        // (side, starts, overlaps) from `split_tiles` (`scripts/comfy_h3_vae_plan.py`)
        let p = TilePlan::new(768);
        assert_eq!(
            (p.starts, p.overlaps),
            (vec![0, 160, 336, 512], vec![96, 80, 80])
        );
        let p = TilePlan::new(448);
        assert_eq!((p.starts, p.overlaps), (vec![0, 192], vec![64]));
        let p = TilePlan::new(256);
        assert_eq!((p.starts, p.lens, p.overlaps), (vec![0], vec![256], vec![]));
        for len in (256..=2048).step_by(32) {
            let p = TilePlan::new(len);
            assert_eq!(p.starts.last().unwrap() + p.lens.last().unwrap(), len);
        }
    }

    #[test]
    fn chunks_of_a_whole_clip_are_seven_latent_steps_except_the_last() {
        // 12 latent steps need no padding; ComfyUI slices `z[t_start : t_start + 7]`, which Python clips at the end
        let plan = TemporalPlan::new(12).unwrap();
        assert_eq!((plan.chunk(0, 12), plan.chunk(1, 12)), ((0, 7), (5, 12)));
        let plan = TemporalPlan::new(7).unwrap();
        assert_eq!(plan.chunk(0, 7), (0, 7));
        // 8 steps pad 4 copies of the last: two chunks over 12 steps
        let plan = TemporalPlan::new(8).unwrap();
        assert_eq!(
            (plan.chunks, plan.pad_tokens, plan.chunk(1, 8)),
            (2, 4, (5, 12))
        );
    }

    #[test]
    fn splice_keeps_seventeen_frames_per_chunk_and_blends_the_overlap() {
        // a constant volume per chunk, a ramp across the 5 overlapping frames
        let z_len = 12;
        let plan = TemporalPlan::new(z_len).unwrap();
        let out = splice_chunks(z_len, &plan, |start, end| {
            let frames = (end - start) * VV_RATIO_T;
            let mut v = Vid::zeros(PIXEL_CHANNELS, frames, 1, 1);
            let value = if start == 0 { 0.0f32 } else { 1.0 };
            v.data
                .iter_mut()
                .for_each(|d| *d = (value - PIXEL_MEAN[0]) / PIXEL_STD[0]);
            Ok(v)
        })
        .unwrap();
        assert_eq!((out.t, out.h, out.w), (plan.frames, 1, 1));
        // frames 0..17 come from chunk 0 alone; blended frames 17..22 ramp from chunk 0's held tail toward chunk 1
        assert!(
            (out.at(0, 0, 0, 0) - 0.0).abs() < 1e-3,
            "first frame: {}",
            out.at(0, 0, 0, 0)
        );
        let first = out.at(0, 17, 0, 0);
        let last = out.at(0, 21, 0, 0);
        assert!(
            first <= last,
            "blend must move toward the second chunk: {first} -> {last}"
        );
    }

    /// The table ComfyUI's `RotaryEmbeddingND` built for one 7-step tile (`scripts/comfy_h3_vae_tile_taps.py`), as FP16-rounded
    /// `[c, -s; s, c]` per pair. `APXINF_H3_VAE_TILE=/path/vae_tile` with `rope_table.f32` (1797 tokens x 24 pairs x 4).
    #[test]
    #[ignore = "requires the captured ComfyUI rope table"]
    fn rope_table_equals_comfyui() {
        let dir = std::path::PathBuf::from(
            std::env::var("APXINF_H3_VAE_TILE").expect("APXINF_H3_VAE_TILE"),
        );
        let bytes = std::fs::read(dir.join("rope_table.f32")).unwrap();
        let reference: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let table = vae_rope_table(7);
        let tokens = tile_tokens(7);
        assert_eq!(reference.len(), tokens * 24 * 4);
        let mut worst = 0f32;
        for token in 0..tokens {
            for p in 0..24 {
                let r = &reference[(token * 24 + p) * 4..][..4];
                let (c, s) = (table[token * 64 + p], table[token * 64 + 32 + p]);
                for (got, want) in [(c, r[0]), (-s, r[1]), (s, r[2]), (c, r[3])] {
                    worst = worst.max((got - want).abs());
                }
            }
        }
        eprintln!("rope table: worst abs difference to ComfyUI {worst:.2e}");
        assert!(worst < 2e-3, "rope table differs from ComfyUI's by {worst}");
    }

    #[test]
    fn patchify_round_trip_orders_features_channel_time_row_column() {
        let (t, th, tw) = (2, 1, 2);
        let tokens: Vec<f32> = (0..t * th * tw * VV_OUT_DIM).map(|i| i as f32).collect();
        let tile = unpatchify_pixels(&tokens, t, th, tw);
        assert_eq!((tile.c, tile.t, tile.h, tile.w), (3, 8, 16, 32));
        // token (ft 1, tx 1), channel 2, pt 3, py 4, px 5
        let token = (th * tw + 1) * VV_OUT_DIM;
        assert_eq!(
            tile.at(2, 4 + 3, 4, 16 + 5),
            (token + ((2 * VV_RATIO_T + 3) * VAE_RATIO + 4) * VAE_RATIO + 5) as f32
        );
    }

    #[test]
    fn latent_tokens_are_channel_vectors_in_time_row_column_order() {
        let (t, h, w) = (3, 4, 5);
        let latent: Vec<f32> = (0..LATENT_CHANNELS * t * h * w).map(|i| i as f32).collect();
        let mut tokens = Vec::new();
        latent_tile_tokens(&latent, [t, h, w], 1, 3, 1, 2, [2, 3], &mut tokens);
        assert_eq!(tokens.len(), 2 * 6 * LATENT_CHANNELS);
        // token 0 is (ft 1, y 1, x 2); channel 5 sits at ((5 * 3 + 1) * 4 + 1) * 5 + 2
        assert_eq!(tokens[5], (((5 * 3 + 1) * 4 + 1) * 5 + 2) as f32);
        // token 6 is (ft 2, y 1, x 2)
        assert_eq!(tokens[6 * LATENT_CHANNELS], ((2 * 4 + 1) * 5 + 2) as f32);
    }

    #[test]
    fn rope_table_has_unit_rotation_on_suffix_and_pass_through_pairs() {
        let table = vae_rope_table(2);
        let tokens = tile_tokens(2);
        assert_eq!(table.len(), tokens * 64);
        for token in 0..tokens {
            let row = &table[token * 64..(token + 1) * 64];
            assert!(row[24..32].iter().all(|c| *c == 1.0) && row[56..].iter().all(|s| *s == 0.0));
        }
        for token in 2 * VV_TILE_LATENT * VV_TILE_LATENT..tokens {
            let row = &table[token * 64..(token + 1) * 64];
            assert!(row[..32].iter().all(|c| *c == 1.0) && row[32..].iter().all(|s| *s == 0.0));
        }
    }
}
