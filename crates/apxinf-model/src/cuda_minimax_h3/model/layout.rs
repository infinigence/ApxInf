//! Packed-sequence layout of one text-to-video-and-audio request: which rows are text, target audio and target video, where each sits
//! on the three position axes, and which AdaLN modality row and timestep class each segment addresses. Pure host arithmetic.
//!
//! Matches ComfyUI `PackedLayout` (`comfy/ldm/minimax/model.py`) for the request this family implements: no keyframes and no reference
//! blocks. Segments are contiguous and ordered `text | audio | video`.
use crate::cuda_minimax_h3::config::*;
use apxinf_core::{Error, Result};

/// Frames a request may ask for are snapped up onto the `17 k + 5` grid the model was trained on.
pub(crate) fn align_frame_count(frames: usize) -> usize {
    let mut n = frames.max(5);
    while n % 17 != 5 {
        n += 1;
    }
    n
}

/// Latent time steps of a frame count already on the grid: two for the shortest clip, then five per 17 frames.
pub(crate) fn video_latent_steps(frames: usize) -> usize {
    if frames <= 5 {
        2
    } else {
        ((frames - 5) / 17) * 5 + 2
    }
}

/// Audio latent frames of a clip: `AUDIO_LATENT_FPS` per second of video.
pub(crate) fn audio_latent_steps(frames: usize) -> usize {
    // Python `round` (half to even) on `frames / FPS * AUDIO_LATENT_FPS`
    let exact = frames as f64 / FPS as f64 * AUDIO_LATENT_FPS as f64;
    let floor = exact.floor();
    let diff = exact - floor;
    let rounded = if diff > 0.5 || (diff == 0.5 && floor as i64 % 2 == 1) {
        floor + 1.0
    } else {
        floor
    };
    rounded as usize
}

/// Kind of one packed segment; the discriminant order is the packing order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Text,
    Audio,
    Video,
}

impl Kind {
    /// AdaLN modality row: visual 0, text 1, audio 2.
    pub fn modality(self) -> usize {
        match self {
            Kind::Video => MODALITY_VISUAL,
            Kind::Text => MODALITY_TEXT,
            Kind::Audio => MODALITY_AUDIO,
        }
    }
}

/// `[start, stop)` rows of one kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Segment {
    pub start: usize,
    pub stop: usize,
    pub kind: Kind,
}

impl Segment {
    pub fn rows(&self) -> usize {
        self.stop - self.start
    }
}

/// Everything the shape of a request fixes. A prepared plan is reused by every request with the same layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Layout {
    pub text_rows: usize,
    /// Video latent grid `(t, h, w)` before patching; `h` and `w` are even.
    pub latent_t: usize,
    pub latent_h: usize,
    pub latent_w: usize,
    pub audio_t: usize,
}

impl Layout {
    pub fn new(
        text_rows: usize,
        latent_t: usize,
        latent_h: usize,
        latent_w: usize,
        audio_t: usize,
    ) -> Result<Self> {
        if text_rows == 0 || latent_t == 0 || audio_t == 0 {
            return Err(Error::Other(
                "MiniMax-H3 needs text rows, video steps and audio steps".into(),
            ));
        }
        if latent_h == 0 || latent_w == 0 || latent_h % PATCH != 0 || latent_w % PATCH != 0 {
            return Err(Error::Other(format!(
                "MiniMax-H3 latent sides must be positive multiples of {PATCH}, got {latent_h}x{latent_w}"
            )));
        }
        Ok(Self {
            text_rows,
            latent_t,
            latent_h,
            latent_w,
            audio_t,
        })
    }

    /// Patch rows of one latent frame.
    pub fn frame_rows(&self) -> usize {
        (self.latent_h / PATCH) * (self.latent_w / PATCH)
    }

    pub fn audio_rows(&self) -> usize {
        self.audio_t * AUDIO_STEREO
    }

    pub fn video_rows(&self) -> usize {
        self.latent_t * self.frame_rows()
    }

    pub fn total_rows(&self) -> usize {
        self.text_rows + self.audio_rows() + self.video_rows()
    }

    /// Contiguous segments in packing order.
    pub fn segments(&self) -> [Segment; 3] {
        let (a, v) = (self.text_rows + self.audio_rows(), self.total_rows());
        [
            Segment {
                start: 0,
                stop: self.text_rows,
                kind: Kind::Text,
            },
            Segment {
                start: self.text_rows,
                stop: a,
                kind: Kind::Audio,
            },
            Segment {
                start: a,
                stop: v,
                kind: Kind::Video,
            },
        ]
    }

    /// `(t, h, w)` position of every row.
    ///
    /// Text rows count along `t` from 0. The target timeline starts at `text_rows`. Audio rows advance one unit of `t` per latent
    /// frame, channel-major, with `w` pinned to the first (left channel) and last (right channel) video column coordinate.
    /// Video rows advance `t` by the cumulative frame-per-token spans and carry the area-normalised patch coordinates.
    pub fn positions(&self) -> Vec<[f64; 3]> {
        let cursor = self.text_rows as f64;
        let mut out = Vec::with_capacity(self.total_rows());
        for i in 0..self.text_rows {
            out.push([i as f64, 0.0, 0.0]);
        }
        let (rows, cols) = (self.latent_h / PATCH, self.latent_w / PATCH);
        let area_root = ((self.latent_h * self.latent_w) as f64).sqrt();
        let hs = axis(self.latent_h, area_root);
        let ws = axis(self.latent_w, area_root);
        debug_assert_eq!((hs.len(), ws.len()), (rows, cols));
        let (w_low, w_high) = (ws[0], ws[cols - 1]);
        for channel in 0..AUDIO_STEREO {
            let w = if channel == 0 { w_low } else { w_high };
            for t in 0..self.audio_t {
                out.push([cursor + t as f64, 0.0, w]);
            }
        }
        let mut t_pos = cursor;
        for step in 0..self.latent_t {
            for &h in &hs {
                for &w in &ws {
                    out.push([t_pos, h, w]);
                }
            }
            t_pos += FRAME_RESCALE * FRAME_PER_TOKEN[step % 5];
        }
        out
    }
}

/// `(arange(n) * (ratio / n) + (1 - ratio) / 2) * 32` with `ratio = side / sqrt(latent_area)` in float64 (ComfyUI `_axis_from_sqrt_area`).
fn axis(side: usize, area_root: f64) -> Vec<f64> {
    let ratio = side as f64 / area_root;
    let n = side / PATCH;
    (0..n)
        .map(|i| (i as f64 * (ratio / n as f64) + (1.0 - ratio) / 2.0) * 32.0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_counts_snap_to_the_17k_plus_5_grid() {
        assert_eq!(align_frame_count(1), 5);
        assert_eq!(align_frame_count(5), 5);
        assert_eq!(align_frame_count(6), 22);
        assert_eq!(align_frame_count(22), 22);
        assert_eq!(
            align_frame_count(124),
            124,
            "124 = 17 * 7 + 5, the five-second default"
        );
        assert_eq!(align_frame_count(125), 141);
        assert_eq!(video_latent_steps(5), 2);
        assert_eq!(video_latent_steps(22), 7);
        assert_eq!(video_latent_steps(124), 37);
    }

    #[test]
    fn every_accepted_frame_count_snaps_inside_the_bound() {
        for frames in MIN_FRAMES..=MAX_FRAMES {
            let snapped = align_frame_count(frames);
            assert!(
                snapped <= MAX_FRAMES,
                "{frames} frames snap to {snapped}, beyond the {MAX_FRAMES} the request check promises"
            );
        }
        assert_eq!(
            align_frame_count(MAX_FRAMES),
            MAX_FRAMES,
            "the bound is on the grid"
        );
    }

    #[test]
    fn audio_steps_round_half_to_even_like_python() {
        assert_eq!(audio_latent_steps(22), 37, "22 / 24 * 40 = 36.67");
        assert_eq!(audio_latent_steps(5), 8, "5 / 24 * 40 = 8.33");
        assert_eq!(audio_latent_steps(124), 207, "124 / 24 * 40 = 206.67");
        assert_eq!(audio_latent_steps(3), 5, "3 / 24 * 40 = 5.0");
        assert_eq!(audio_latent_steps(27), 45, "27 / 24 * 40 = 45.0");
    }

    #[test]
    fn captured_request_has_the_segments_comfyui_built() {
        // 768 x 448, 22 frames: latent 28 x 48, 7 video steps, 37 audio steps, 14 text rows (captured trace, layout.json)
        let layout = Layout::new(14, 7, 28, 48, 37).unwrap();
        assert_eq!(layout.frame_rows(), 14 * 24);
        assert_eq!(layout.total_rows(), 2440);
        let s = layout.segments();
        assert_eq!((s[0].start, s[0].stop), (0, 14));
        assert_eq!((s[1].start, s[1].stop), (14, 88));
        assert_eq!((s[2].start, s[2].stop), (88, 2440));
        assert_eq!(s[1].kind.modality(), MODALITY_AUDIO);
        assert_eq!(s[2].kind.modality(), MODALITY_VISUAL);
    }

    #[test]
    fn positions_follow_the_axis_conventions() {
        let layout = Layout::new(3, 2, 4, 6, 4).unwrap();
        let p = layout.positions();
        assert_eq!(p.len(), layout.total_rows());
        assert_eq!(p[2], [2.0, 0.0, 0.0], "text rows count along t");
        // audio: left channel then right channel, t advances from the cursor (= text rows)
        assert_eq!(p[3][0], 3.0);
        assert_eq!(p[6][0], 6.0);
        assert_eq!(p[7][0], 3.0, "second channel restarts the timeline");
        assert!(
            p[3][2] < p[7][2],
            "left channel at the lowest column coordinate, right at the highest"
        );
        assert_eq!((p[3][1], p[7][1]), (0.0, 0.0), "audio rows have h = 0");
        // video: first step at the cursor, the next advances by 5/3 * 1 then 5/3 * 4
        let v0 = 3 + 8;
        assert_eq!(p[v0][0], 3.0);
        assert!((p[v0 + layout.frame_rows()][0] - (3.0 + FRAME_RESCALE * 1.0)).abs() < 1e-12);
    }

    /// `APXINF_H3_LAYOUT=/path/comfy_t2va/layout cargo test --release -p apxinf-model --features cuda --lib -- --ignored positions_equal_comfyui`
    /// where the directory holds `position_ids.f64` exported by `scripts/export_h3_layout.py --text 14 --t 7 --h 28 --w 48 --audio 37`.
    #[test]
    #[ignore = "requires a ComfyUI layout export"]
    fn positions_equal_comfyui() {
        let dir =
            std::path::PathBuf::from(std::env::var("APXINF_H3_LAYOUT").expect("APXINF_H3_LAYOUT"));
        let bytes = std::fs::read(dir.join("position_ids.f64")).unwrap();
        let reference: Vec<f64> = bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let layout = Layout::new(14, 7, 28, 48, 37).unwrap();
        let ours = layout.positions();
        assert_eq!(reference.len(), ours.len() * 3);
        let mut worst = 0f64;
        for (row, p) in ours.iter().enumerate() {
            for axis in 0..3 {
                worst = worst.max((p[axis] - reference[row * 3 + axis]).abs());
            }
        }
        eprintln!(
            "max |position difference| over {} rows: {worst:e}",
            ours.len()
        );
        // The two sides evaluate `i * (ratio / n) + (1 - ratio) / 2` with the same formula but different operation order (torch
        // `arange * scalar + scalar` against this loop), so float64 rounding differs in the last bits. 1e-12 is far below the FP32
        // rounding (6e-8 relative) the positions go through before they become angles, and far above a convention error (>= 1e-3).
        assert!(
            worst < 1e-12,
            "position ids differ from ComfyUI's float64 values by {worst:e}"
        );
    }
}
