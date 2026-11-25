//! Host arithmetic that depends only on the checkpoint and the request geometry: AdaLN curve evaluation, per-step modulation
//! rows for the three modalities, the flow-shift schedules of the two streams, packing of the video and audio latents and the
//! rotary table. Nothing here touches the device.
use crate::cuda_minimax_h3::config::*;
use crate::cuda_minimax_h3::weights::{AdalnCurve, HostHead};
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

/// Modulation of one DiT block at one timestep: BF16 `[modality, 6, HIDDEN]` for the visual (0), text (1) and audio (2) rows,
/// parts ordered `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`. Every row is rounded to BF16 once, as the
/// reference does when it applies a row to a BF16 stream.
pub(crate) fn block_modulation(curve: &AdalnCurve, embedding: &[f32; ADALN_RANK]) -> Vec<bf16> {
    project(curve, embedding, 0..MODALITIES * 6 * HIDDEN)
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

/// Fold `norm * (1 + scale) + shift` into an output head of `out` features: `head(norm * (1 + s) + t)` equals
/// `norm @ (W (1 + s)) + (W t + b)`. Returns the weight as row-major `[HIDDEN, out]` (K by N) and the bias.
pub(crate) fn fold_head(
    head: &HostHead,
    out: usize,
    shift: &[f32],
    scale: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut weight = vec![0.0f32; HIDDEN * out];
    let mut bias = vec![0.0f32; out];
    for o in 0..out {
        let row = &head.weight[o * HIDDEN..(o + 1) * HIDDEN];
        let mut acc = head.bias[o] as f64;
        for k in 0..HIDDEN {
            weight[k * out + o] = row[k] * (1.0 + scale[k]);
            acc += row[k] as f64 * shift[k] as f64;
        }
        bias[o] = acc as f32;
    }
    (weight, bias)
}

/// Base-grid sigma of the other stream: invert `sigma = s b / (1 + (s - 1) b)` for the `from` shift, re-apply the `to` shift
/// (ComfyUI `time_shift_sigma`, FP32).
pub(crate) fn time_shift_sigma(sigma: f32, from_shift: f32, to_shift: f32) -> f32 {
    let base = sigma / (from_shift + sigma * (1.0 - from_shift));
    to_shift * base / (1.0 + (to_shift - 1.0) * base)
}

/// Sigma of the video stream at normalised time `u` in `[0, 1]` (1 = pure noise): the flow-shifted uniform grid
/// `s u / (1 + (s - 1) u)` ComfyUI's `ModelSamplingDiscreteFlow` evaluates.
#[cfg(test)]
pub(crate) fn shifted_sigma(u: f32, shift: f32) -> f32 {
    shift * u / (1.0 + (shift - 1.0) * u)
}

/// Video sigmas of a `steps`-step request, `steps + 1` values ending in 0: ComfyUI's `simple` scheduler over the shifted flow grid.
/// The grid is `sigma((k + 1) / 1000)` for `k = 0..1000`, evaluated in FP32 exactly as `ModelSamplingDiscreteFlow` does, and the
/// scheduler takes entry `-(1 + int(i * 1000 / steps))` for step `i`.
pub(crate) fn simple_sigmas(steps: usize, shift: f32) -> Vec<f32> {
    const GRID: usize = 1000;
    let stride = GRID as f64 / steps as f64;
    let mut sigmas: Vec<f32> = (0..steps)
        .map(|i| {
            let index = GRID - 1 - (i as f64 * stride) as usize;
            // arange(1, 1001) / 1000 * 1000 / 1000, each division and product rounded to FP32
            let timestep = ((index + 1) as f32 / GRID as f32) * GRID as f32;
            let t = timestep / GRID as f32;
            shift * t / (1.0 + (shift - 1.0) * t)
        })
        .collect();
    sigmas.push(0.0);
    sigmas
}

/// `[AUDIO_CHANNELS, AUDIO_STEREO, t]` audio latent to `[AUDIO_STEREO * t, AUDIO_CHANNELS]` rows, channel-major: all steps of
/// stereo channel 0, then channel 1 (ComfyUI `pack_audio`).
pub(crate) fn pack_audio(latent: &[f32], t: usize) -> Vec<f32> {
    assert_eq!(latent.len(), AUDIO_CHANNELS * AUDIO_STEREO * t);
    let mut out = vec![0.0; AUDIO_STEREO * t * AUDIO_CHANNELS];
    for c in 0..AUDIO_CHANNELS {
        for ch in 0..AUDIO_STEREO {
            for i in 0..t {
                out[(ch * t + i) * AUDIO_CHANNELS + c] = latent[(c * AUDIO_STEREO + ch) * t + i];
            }
        }
    }
    out
}

/// Inverse of [`pack_audio`].
pub(crate) fn unpack_audio(rows: &[f32], t: usize) -> Vec<f32> {
    assert_eq!(rows.len(), AUDIO_STEREO * t * AUDIO_CHANNELS);
    let mut out = vec![0.0; AUDIO_CHANNELS * AUDIO_STEREO * t];
    for c in 0..AUDIO_CHANNELS {
        for ch in 0..AUDIO_STEREO {
            for i in 0..t {
                out[(c * AUDIO_STEREO + ch) * t + i] = rows[(ch * t + i) * AUDIO_CHANNELS + c];
            }
        }
    }
    out
}

/// `[LATENT_CHANNELS, t, h, w]` video latent to `[t * (h / 2) * (w / 2), PATCH_DIM]` patch rows, feature order `(channel, py, px)`
/// with patch `(1, 2, 2)` (ComfyUI `patchify_video`): frames are patched independently, raster order inside a frame.
pub(crate) fn patchify_video(latent: &[f32], t: usize, h: usize, w: usize) -> Vec<f32> {
    assert_eq!(latent.len(), LATENT_CHANNELS * t * h * w);
    assert!(h % PATCH == 0 && w % PATCH == 0);
    let (gh, gw) = (h / PATCH, w / PATCH);
    let mut out = vec![0.0; t * gh * gw * PATCH_DIM];
    for f in 0..t {
        for gy in 0..gh {
            for gx in 0..gw {
                let row = &mut out[((f * gh + gy) * gw + gx) * PATCH_DIM..][..PATCH_DIM];
                for c in 0..LATENT_CHANNELS {
                    for py in 0..PATCH {
                        for px in 0..PATCH {
                            row[(c * PATCH + py) * PATCH + px] =
                                latent[((c * t + f) * h + gy * PATCH + py) * w + gx * PATCH + px];
                        }
                    }
                }
            }
        }
    }
    out
}

/// Inverse of [`patchify_video`].
pub(crate) fn unpatchify_video(rows: &[f32], t: usize, h: usize, w: usize) -> Vec<f32> {
    let (gh, gw) = (h / PATCH, w / PATCH);
    assert_eq!(rows.len(), t * gh * gw * PATCH_DIM);
    let mut out = vec![0.0; LATENT_CHANNELS * t * h * w];
    for f in 0..t {
        for gy in 0..gh {
            for gx in 0..gw {
                let row = &rows[((f * gh + gy) * gw + gx) * PATCH_DIM..][..PATCH_DIM];
                for c in 0..LATENT_CHANNELS {
                    for py in 0..PATCH {
                        for px in 0..PATCH {
                            out[((c * t + f) * h + gy * PATCH + py) * w + gx * PATCH + px] =
                                row[(c * PATCH + py) * PATCH + px];
                        }
                    }
                }
            }
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
    #[test]
    fn simple_sigmas_match_the_captured_four_step_timesteps() {
        // ComfyUI handed the DiT timestep = sigma * 1000: 1000, 972.9729614257812, 923.076904296875, 800
        let sigmas = simple_sigmas(4, SIGMA_SHIFT_VIDEO);
        assert_eq!(sigmas.len(), 5);
        let timesteps: Vec<f32> = sigmas.iter().map(|s| s * 1000.0).collect();
        for (got, want) in timesteps
            .iter()
            .zip([1000.0f32, 972.972_96, 923.076_9, 800.0, 0.0])
        {
            assert!(
                (got - want).abs() <= want.abs() * 2e-7 + 1e-9,
                "{timesteps:?}"
            );
        }
        // strictly decreasing for any step count, and the first step is pure noise
        for steps in [1, 2, 3, 5, 8, 20, 50] {
            let s = simple_sigmas(steps, SIGMA_SHIFT_VIDEO);
            assert_eq!((s.len(), s[0], *s.last().unwrap()), (steps + 1, 1.0, 0.0));
            assert!(s.windows(2).all(|w| w[0] > w[1]), "{steps}: {s:?}");
        }
    }

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
        assert_eq!(curve_embedding(&table, 1.0)[0], 1024.0);
        assert_eq!(curve_embedding(&table, 7.0)[0], 1024.0);
        assert!((curve_embedding(&table, 0.5)[3] - 512.03).abs() < 1e-3);
        assert!((curve_embedding(&table, 10.25 / 1024.0)[0] - 10.25).abs() < 1e-4);
    }

    #[test]
    fn modulation_covers_three_modalities_and_rounds_to_bf16() {
        let rows = 6 * HIDDEN * MODALITIES;
        let curve = AdalnCurve {
            weight: (0..rows * ADALN_RANK)
                .map(|i| half::f16::from_f32(((i % 7) as f32 - 3.0) * 0.25))
                .collect(),
            bias: (0..rows)
                .map(|i| half::f16::from_f32((i % 5) as f32))
                .collect(),
        };
        let e = [1.0, 0.5, 0.0, -1.0, 2.0, 0.0, 0.0, 0.25];
        let modulation = block_modulation(&curve, &e);
        assert_eq!(modulation.len(), MODALITIES * 6 * HIDDEN);
        let expected: f32 = (0..ADALN_RANK)
            .map(|j| e[j] * curve.weight[j].to_f32())
            .sum::<f32>()
            + curve.bias[0].to_f32();
        assert_eq!(modulation[0], bf16::from_f32(expected));
        // the audio rows are the last third and are addressed independently
        let last = modulation.len() - 1;
        let w = &curve.weight[last * ADALN_RANK..(last + 1) * ADALN_RANK];
        let expected: f32 =
            w.iter().zip(&e).map(|(w, e)| w.to_f32() * e).sum::<f32>() + curve.bias[last].to_f32();
        assert_eq!(modulation[last], bf16::from_f32(expected));
    }

    #[test]
    fn folded_head_matches_modulate_then_project_for_both_stream_widths() {
        for out in [PATCH_DIM, AUDIO_CHANNELS] {
            let head = HostHead {
                weight: (0..out * HIDDEN)
                    .map(|i| ((i * 31 % 17) as f32 - 8.0) * 0.01)
                    .collect(),
                bias: (0..out).map(|i| i as f32 * 0.1).collect(),
            };
            let shift: Vec<f32> = (0..HIDDEN).map(|i| ((i % 9) as f32 - 4.0) * 0.05).collect();
            let scale: Vec<f32> = (0..HIDDEN).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
            let normalized: Vec<f32> = (0..HIDDEN)
                .map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.2)
                .collect();
            let (w, b) = fold_head(&head, out, &shift, &scale);
            for o in [0, out / 2, out - 1] {
                let direct: f64 = (0..HIDDEN)
                    .map(|k| {
                        head.weight[o * HIDDEN + k] as f64
                            * (normalized[k] as f64 * (1.0 + scale[k] as f64) + shift[k] as f64)
                    })
                    .sum::<f64>()
                    + head.bias[o] as f64;
                let folded: f64 = (0..HIDDEN)
                    .map(|k| normalized[k] as f64 * w[k * out + o] as f64)
                    .sum::<f64>()
                    + b[o] as f64;
                assert!(
                    (direct - folded).abs() < 1e-3 * direct.abs().max(1.0),
                    "out={out} o={o} {direct} vs {folded}"
                );
            }
        }
    }

    #[test]
    fn sigma_shift_round_trips_and_matches_the_flow_formula() {
        // the base grid point is invariant: shifting video to audio and back returns the video sigma
        for &s in &[1.0f32, 0.9, 0.5, 0.25, 0.01] {
            let a = time_shift_sigma(s, SIGMA_SHIFT_VIDEO, SIGMA_SHIFT_AUDIO);
            let back = time_shift_sigma(a, SIGMA_SHIFT_AUDIO, SIGMA_SHIFT_VIDEO);
            assert!((back - s).abs() < 1e-5, "{s} -> {a} -> {back}");
        }
        // endpoints stay put, and a smaller shift keeps more of the early steps: audio sigma is below video sigma
        assert!((time_shift_sigma(1.0, 12.0, 3.0) - 1.0).abs() < 1e-6);
        assert!(time_shift_sigma(0.5, 12.0, 3.0) < 0.5);
        assert!((shifted_sigma(0.5, 12.0) - 12.0 * 0.5 / (1.0 + 11.0 * 0.5)).abs() < 1e-7);
        assert_eq!(shifted_sigma(1.0, 12.0), 1.0);
        assert_eq!(shifted_sigma(0.0, 12.0), 0.0);
    }

    #[test]
    fn audio_packing_is_channel_major_and_round_trips() {
        let t = 5;
        let latent: Vec<f32> = (0..AUDIO_CHANNELS * AUDIO_STEREO * t)
            .map(|i| i as f32)
            .collect();
        let rows = pack_audio(&latent, t);
        // row (channel 1, step 3) carries feature c = 7 from latent[7, 1, 3]
        assert_eq!(
            rows[(t + 3) * AUDIO_CHANNELS + 7],
            latent[(7 * AUDIO_STEREO + 1) * t + 3]
        );
        assert_eq!(unpack_audio(&rows, t), latent);
    }

    #[test]
    fn video_patching_keeps_frames_apart_and_round_trips() {
        let (t, h, w) = (3, 4, 6);
        let latent: Vec<f32> = (0..LATENT_CHANNELS * t * h * w).map(|i| i as f32).collect();
        let rows = patchify_video(&latent, t, h, w);
        assert_eq!(rows.len(), t * 2 * 3 * PATCH_DIM);
        // frame 2, patch (gy 1, gx 2), channel 3, pixel (1, 0)
        let row = &rows[((2 * 2 + 1) * 3 + 2) * PATCH_DIM..][..PATCH_DIM];
        assert_eq!(row[(3 * 2 + 1) * 2], latent[((3 * t + 2) * h + 3) * w + 4]);
        assert_eq!(unpatchify_video(&rows, t, h, w), latent);
    }

    #[test]
    fn rope_table_rotates_only_the_model_pairs() {
        let layout = crate::cuda_minimax_h3::model::layout::Layout::new(3, 2, 8, 8, 4).unwrap();
        let inv: Vec<f32> = (0..ROPE_FREQS)
            .map(|i| 1.0 / 10000f32.powf(i as f32 / ROPE_FREQS as f32))
            .collect();
        let table = rope_table(&layout.positions(), &inv);
        let row = |r: usize| &table[r * 128..(r + 1) * 128];
        assert!(row(0)[..64].iter().all(|c| *c == 1.0) && row(0)[64..].iter().all(|s| *s == 0.0));
        let r2 = row(2);
        assert_eq!(r2[0], bf16::from_f32((2.0f32 * inv[0]).cos()).to_f32());
        assert_eq!(r2[16..48].iter().filter(|c| **c != 1.0).count(), 0);
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
