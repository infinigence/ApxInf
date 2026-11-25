//! Checkpoint-fixed image canonicalization: `smart_resize` → bicubic
//! antialiased resize → normalize → patchify, producing the `pixel_values`
//! `[N, C·T·P·P]` and `image_grid_thw` `[1, H, W]` contract the vision tower
//! consumes.
//!
//! Every step is determined by the checkpoint's `preprocessor_config.json`
//! plus `vision_config`, so the runtime owns it (see
//! `doc/model-layer-architecture.md`, "VLA input seam"). Decoding the image
//! container stays with the caller, which passes decoded RGB `u8`.
//!
//! The resize copies PyTorch's native uint8 bicubic, antialiased kernel
//! exactly (double-precision Keys weights with `a = -0.5`, separable
//! horizontal-then-vertical application, int16 fixed-point accumulation): the
//! checkpoint's own processor runs that kernel on the CPU, so an element-wise
//! comparison against the reference `pixel_values` comes back exact.

use super::config::{PreprocessorContract, VisionContract};
use apxinf_core::{Error, Result};

/// One image's canonical tensors, in the layout the vision tower reads.
pub struct PreparedImage {
    /// `[grid_h * grid_w, C · T · P · P]`, merge-block spatial order, F32.
    pub pixel_values: Vec<f32>,
    /// `[1, grid_h, grid_w]` for a still image.
    pub grid_thw: [u32; 3],
    /// Size after `smart_resize`, in pixels `(height, width)`.
    pub resized: (usize, usize),
    /// Image tokens after the spatial merger: `grid_h * grid_w / merge²`.
    pub image_tokens: usize,
}

/// Python's `round` (ties to even), which `smart_resize` uses and Rust's
/// `f32::round` does not implement.
fn round_half_even(value: f64) -> f64 {
    let floor = value.floor();
    let fraction = value - floor;
    if fraction > 0.5 {
        floor + 1.0
    } else if fraction < 0.5 {
        floor
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    }
}

/// `Qwen2VLImageProcessor.smart_resize`: both sides divisible by
/// `patch * merge`, pixel count inside `[min_pixels, max_pixels]`, aspect
/// ratio preserved as closely as possible (and rejected above 200:1).
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    if height == 0 || width == 0 || factor == 0 {
        return Err(Error::Other(
            "d3-lite image resize needs positive dimensions and factor".into(),
        ));
    }
    if height.max(width) as f64 / height.min(width) as f64 > 200.0 {
        return Err(Error::Other(format!(
            "d3-lite image aspect ratio {} exceeds the processor limit 200",
            height.max(width) as f64 / height.min(width) as f64
        )));
    }
    let mut resized_height = round_half_even(height as f64 / factor as f64) as usize * factor;
    let mut resized_width = round_half_even(width as f64 / factor as f64) as usize * factor;
    if resized_height * resized_width > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        resized_height = ((height as f64 / beta / factor as f64).floor() as usize).max(1) * factor;
        resized_width = ((width as f64 / beta / factor as f64).floor() as usize).max(1) * factor;
    } else if resized_height * resized_width < min_pixels {
        let beta = (min_pixels as f64 / (height * width) as f64).sqrt();
        resized_height = (height as f64 * beta / factor as f64).ceil() as usize * factor;
        resized_width = (width as f64 * beta / factor as f64).ceil() as usize * factor;
    }
    Ok((resized_height, resized_width))
}

/// Keys' cubic convolution with `a = -0.5`, the filter PyTorch uses for
/// bicubic with antialiasing enabled (PIL-compatible).
fn aa_filter(value: f64) -> f64 {
    const A: f64 = -0.5;
    let x = value.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}

/// One axis's antialiased taps: `(input offset, int16 weights)` per output
/// index plus the fixed-point precision the accumulation shifts by.
fn axis_taps(input_size: usize, output_size: usize) -> Result<(Vec<(usize, Vec<i16>)>, u32)> {
    let scale = input_size as f64 / output_size as f64;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let max_interp = (support.ceil() as usize) * 2 + 1;
    let mut rows = Vec::with_capacity(output_size);
    let mut weight_max = 0.0f64;
    for index in 0..output_size {
        let center = scale * (index as f64 + 0.5);
        let inverse_scale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
        let minimum = ((center - support + 0.5) as i64).max(0) as usize;
        let size = ((center + support + 0.5) as i64)
            .min(input_size as i64)
            .saturating_sub(minimum as i64)
            .clamp(0, max_interp as i64) as usize;
        let mut weights = vec![0.0f64; size];
        let mut total = 0.0f64;
        for (tap, weight) in weights.iter_mut().enumerate() {
            *weight = aa_filter((tap as f64 + minimum as f64 - center + 0.5) * inverse_scale);
            total += *weight;
        }
        if total != 0.0 {
            for weight in &mut weights {
                *weight /= total;
                weight_max = weight_max.max(*weight);
            }
        }
        rows.push((minimum, weights));
    }
    // Same precision search as PyTorch: the largest weight must fit in int16.
    let mut precision = 22u32;
    for candidate in 0..22u32 {
        let next = (0.5 + weight_max * (1i64 << (candidate + 1)) as f64) as i64;
        if next >= (1 << 15) {
            precision = candidate;
            break;
        }
    }
    // The accumulation rounds with `1 << (precision - 1)`, so precision must
    // stay positive exactly as PyTorch's own assert requires.
    let precision = precision.max(1);
    let scale_by = (1i64 << precision) as f64;
    let quantized = rows
        .into_iter()
        .map(|(minimum, weights)| {
            let weights = weights
                .into_iter()
                .map(|weight| {
                    let value = weight * scale_by;
                    (if value < 0.0 {
                        (-0.5 + value) as i64
                    } else {
                        (0.5 + value) as i64
                    }) as i16
                })
                .collect();
            (minimum, weights)
        })
        .collect();
    Ok((quantized, precision))
}

/// PyTorch's native uint8 bicubic antialiased resize, separable with the
/// horizontal pass first and `u8` intermediates (each pass rounds through the
/// fixed-point accumulation).
pub fn resize_bicubic(
    chw: &[u8],
    height: usize,
    width: usize,
    out_h: usize,
    out_w: usize,
) -> Result<Vec<u8>> {
    if chw.len() != 3 * height * width {
        return Err(Error::Other(format!(
            "d3-lite image RGB buffer holds {} bytes for 3 x {height} x {width}",
            chw.len()
        )));
    }
    if (out_h, out_w) == (height, width) {
        return Ok(chw.to_vec());
    }
    let mut intermediate = chw.to_vec();
    let mut current_w = width;
    if out_w != width {
        let (taps, precision) = axis_taps(width, out_w)?;
        let mut output = vec![0u8; 3 * height * out_w];
        for channel in 0..3 {
            for row in 0..height {
                let source = (channel * height + row) * width;
                let target = (channel * height + row) * out_w;
                for (column, (minimum, weights)) in taps.iter().enumerate() {
                    let mut accumulator = 1i64 << (precision - 1);
                    for (tap, &weight) in weights.iter().enumerate() {
                        accumulator += intermediate[source + minimum + tap] as i64 * weight as i64;
                    }
                    output[target + column] = (accumulator >> precision).clamp(0, 255) as u8;
                }
            }
        }
        intermediate = output;
        current_w = out_w;
    }
    if out_h != height {
        let (taps, precision) = axis_taps(height, out_h)?;
        let mut output = vec![0u8; 3 * out_h * current_w];
        for channel in 0..3 {
            for row in 0..out_h {
                let (minimum, weights) = &taps[row];
                for column in 0..current_w {
                    let mut accumulator = 1i64 << (precision - 1);
                    for (tap, &weight) in weights.iter().enumerate() {
                        let source = (channel * height + minimum + tap) * current_w + column;
                        accumulator += intermediate[source] as i64 * weight as i64;
                    }
                    output[(channel * out_h + row) * current_w + column] =
                        (accumulator >> precision).clamp(0, 255) as u8;
                }
            }
        }
        intermediate = output;
    }
    Ok(intermediate)
}

/// Decoded RGB `u8` → the vision tower's input tensors.
///
/// `rgb` is **interleaved HWC** — the layout every image decoder produces —
/// and is converted to the channel-first form the patchify layout and the
/// separable resize need.
pub fn preprocess(
    preprocessor: &PreprocessorContract,
    vision: &VisionContract,
    rgb: &[u8],
    width: usize,
    height: usize,
) -> Result<PreparedImage> {
    if width == 0 || height == 0 {
        return Err(Error::Other("d3-lite image must be nonempty".into()));
    }
    if rgb.len() != 3 * width * height {
        return Err(Error::Other(format!(
            "d3-lite RGB buffer holds {} bytes for 3 x {width} x {height}",
            rgb.len()
        )));
    }
    let (resized_height, resized_width) = smart_resize(
        height,
        width,
        preprocessor.factor,
        preprocessor.min_pixels,
        preprocessor.max_pixels,
    )?;
    let grid_h = resized_height / vision.patch;
    let grid_w = resized_width / vision.patch;
    if grid_h == 0 || grid_w == 0 {
        return Err(Error::Other(format!(
            "d3-lite resize produced {resized_height} x {resized_width}, below one patch"
        )));
    }
    // HWC (decoder layout) → CHW for the separable resize and patchify.
    let mut source = vec![0u8; 3 * height * width];
    for row in 0..height {
        for column in 0..width {
            for channel in 0..3 {
                source[channel * height * width + row * width + column] =
                    rgb[(row * width + column) * 3 + channel];
            }
        }
    }
    let resized = resize_bicubic(&source, height, width, resized_height, resized_width)?;
    // `rescale_factor` and the mean/std cancel into one fused normalization:
    // (u8 - 127.5) / 127.5, exactly what the checkpoint's fused
    // `rescale_and_normalize` computes.
    let half = 127.5f32;
    let patch_dim = 3 * vision.temporal_patch * vision.patch * vision.patch;
    let merge = vision.spatial_merge;
    let pixel_count = grid_h * grid_w;
    let mut pixel_values = vec![0.0f32; pixel_count * patch_dim];
    for block_row in 0..grid_h / merge {
        for block_column in 0..grid_w / merge {
            for inner_row in 0..merge {
                for inner_column in 0..merge {
                    let row = block_row * merge + inner_row;
                    let column = block_column * merge + inner_column;
                    let patch = ((block_row * (grid_w / merge) + block_column) * merge + inner_row)
                        * merge
                        + inner_column;
                    let target = &mut pixel_values[patch * patch_dim..(patch + 1) * patch_dim];
                    let mut feature = 0usize;
                    for channel in 0..3 {
                        for _temporal in 0..vision.temporal_patch {
                            for patch_row in 0..vision.patch {
                                let source_row = row * vision.patch + patch_row;
                                for patch_column in 0..vision.patch {
                                    let source_column = column * vision.patch + patch_column;
                                    let value = resized[(channel * resized_height + source_row)
                                        * resized_width
                                        + source_column]
                                        as f32;
                                    target[feature] = (value - half) / half;
                                    feature += 1;
                                }
                            }
                        }
                    }
                    debug_assert_eq!(feature, patch_dim);
                }
            }
        }
    }
    let image_tokens = pixel_count / (merge * merge);
    Ok(PreparedImage {
        pixel_values,
        grid_thw: [1, grid_h as u32, grid_w as u32],
        resized: (resized_height, resized_width),
        image_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract() -> (PreprocessorContract, VisionContract) {
        let vision = VisionContract {
            depth: 12,
            hidden: 768,
            intermediate: 3_072,
            heads: 12,
            head_dim: 64,
            out_hidden: 1_024,
            patch: 16,
            temporal_patch: 2,
            spatial_merge: 2,
            num_position_embeddings: 2_304,
            num_grid_per_side: 48,
            rope_theta: 10_000.0,
            tensors: 153,
        };
        let preprocessor = PreprocessorContract {
            min_pixels: 65_536,
            max_pixels: 1_638_400,
            factor: 32,
            image_mean: [0.5, 0.5, 0.5],
            image_std: [0.5, 0.5, 0.5],
        };
        (preprocessor, vision)
    }

    #[test]
    fn smart_resize_rounds_halves_to_even_and_obeys_the_budget() {
        // 400 / 32 = 12.5 rounds to 12 (even), so a 400-pixel edge resizes to 384.
        assert_eq!(
            smart_resize(400, 320, 32, 65_536, 1_638_400).unwrap(),
            (384, 320)
        );
        assert_eq!(
            smart_resize(512, 512, 32, 65_536, 1_638_400).unwrap(),
            (512, 512)
        );
        assert_eq!(
            smart_resize(64, 80, 32, 65_536, 1_638_400).unwrap(),
            (256, 288)
        );
        let (height, width) = smart_resize(1200, 1600, 32, 65_536, 1_638_400).unwrap();
        assert!(height % 32 == 0 && width % 32 == 0);
        assert!(height * width <= 1_638_400);
        assert!(smart_resize(10, 2500, 32, 65_536, 1_638_400).is_err());
    }

    #[test]
    fn patchify_duplicates_time_and_orders_merge_blocks() {
        let (preprocessor, vision) = contract();
        // 256 x 256 sits exactly on the 64 K minimum, so no resize happens
        // and source pixels land at provable positions.
        let width = 256usize;
        let height = 256usize;
        // Interleaved HWC, the decoder layout: channel varies fastest.
        let rgb = (0..3 * height * width)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        let prepared = preprocess(&preprocessor, &vision, &rgb, width, height).unwrap();
        assert_eq!(prepared.grid_thw, [1, 16, 16]);
        assert_eq!(prepared.image_tokens, 64);
        assert_eq!(prepared.pixel_values.len(), 256 * 1_536);
        // The fused rescale+normalize contract: (u8 - 127.5) / 127.5.
        let normalize = |value: u8| ((value as f32 - 127.5) / 127.5).to_bits();
        let pixel =
            |row: usize, column: usize, channel: usize| rgb[(row * width + column) * 3 + channel];
        // Patch 0 starts at source (0, 0); its feature 16 is source row 1.
        assert_eq!(
            prepared.pixel_values[0].to_bits(),
            normalize(pixel(0, 0, 0))
        );
        assert_eq!(
            prepared.pixel_values[16].to_bits(),
            normalize(pixel(1, 0, 0))
        );
        // Patch 1 is the next inner patch of the same merge block: source (0, 16).
        assert_eq!(
            prepared.pixel_values[1_536].to_bits(),
            normalize(pixel(0, 16, 0))
        );
        // Patch 4 starts the next merge block to the right: source (0, 32).
        assert_eq!(
            prepared.pixel_values[4 * 1_536].to_bits(),
            normalize(pixel(0, 32, 0))
        );
        // The two temporal slices of every patch hold the same pixels.
        for patch in 0..256 {
            let row = &prepared.pixel_values[patch * 1_536..(patch + 1) * 1_536];
            assert!(row[..256] == row[256..512]);
            assert!(row[512..768] == row[768..1_024]);
            assert!(row[1_024..1_280] == row[1_280..1_536]);
        }
    }

    /// The int16 weights of one output index must sum to the fixed-point
    /// unit, or a constant input region would drift (the failure the
    /// real-image acceptance test originally caught).
    #[test]
    fn axis_tap_weights_sum_to_the_fixed_point_unit() {
        let (taps, precision) = axis_taps(400, 384).unwrap();
        let expected = 1i64 << precision;
        for (index, (minimum, weights)) in taps.iter().enumerate().take(16) {
            let sum = weights.iter().map(|&weight| weight as i64).sum::<i64>();
            assert!(
                (sum - expected).abs() <= weights.len() as i64,
                "row {index} xmin {minimum}: sum {sum} vs {expected} (weights {weights:?})"
            );
        }
    }

    /// Oracle values from `tvF.resize` (PyTorch's native uint8 bicubic,
    /// antialiased), generated by `devlocal/d3-vision-metal/scripts/` for a
    /// deterministic ramp. They pin both the separable order and the
    /// fixed-point rounding of the resize.
    #[test]
    fn bicubic_resize_matches_the_torch_uint8_oracle() {
        let height = 400usize;
        let width = 320usize;
        let mut source = vec![0u8; 3 * height * width];
        for channel in 0..3 {
            for row in 0..height {
                for column in 0..width {
                    source[(channel * height + row) * width + column] =
                        ((row * 7 + column * 3 + channel * 31) % 256) as u8;
                }
            }
        }
        let value = |chw: &[u8], channel: usize, row: usize, column: usize, h: usize, w: usize| {
            chw[(channel * h + row) * w + column]
        };
        let down = resize_bicubic(&source, height, width, 384, 320).unwrap();
        assert_eq!(value(&down, 0, 0, 0, 384, 320), 0);
        assert_eq!(value(&down, 0, 0, 1, 384, 320), 3);
        assert_eq!(value(&down, 0, 1, 0, 384, 320), 7);
        assert_eq!(value(&down, 0, 100, 50, 384, 320), 111);
        assert_eq!(value(&down, 1, 0, 0, 384, 320), 31);
        assert_eq!(value(&down, 2, 200, 160, 384, 320), 208);
        assert_eq!(value(&down, 0, 383, 319, 384, 320), 166);
        assert_eq!(value(&down, 1, 100, 50, 384, 320), 142);

        let up = resize_bicubic(&source, height, width, 448, 352).unwrap();
        assert_eq!(value(&up, 0, 0, 0, 448, 352), 0);
        assert_eq!(value(&up, 0, 0, 1, 448, 352), 3);
        assert_eq!(value(&up, 0, 1, 0, 448, 352), 6);
        assert_eq!(value(&up, 0, 100, 50, 448, 352), 234);
        assert_eq!(value(&up, 1, 0, 0, 448, 352), 31);
        assert_eq!(value(&up, 2, 200, 160, 448, 352), 212);
        assert_eq!(value(&up, 0, 447, 351, 448, 352), 166);
        assert_eq!(value(&up, 1, 100, 50, 448, 352), 24);

        let same = resize_bicubic(&source, height, width, 400, 320).unwrap();
        assert_eq!(value(&same, 0, 100, 50, 400, 320), 82);
        assert_eq!(value(&same, 2, 200, 160, 400, 320), 150);
    }

    /// End-to-end preprocessor acceptance on a real image: the native
    /// resize + normalize + patchify must reproduce the checkpoint's
    /// PyTorch `pixel_values` element for element.
    #[test]
    #[ignore = "requires APXINF_D3_TEST_IMAGE and APXINF_D3_REFERENCE_PIXELS"]
    fn preprocessing_reproduces_the_reference_pixel_values() -> Result<()> {
        let image_path = std::env::var_os("APXINF_D3_TEST_IMAGE")
            .map(std::path::PathBuf::from)
            .expect("set APXINF_D3_TEST_IMAGE to a PNG/JPEG");
        let reference_path = std::env::var_os("APXINF_D3_REFERENCE_PIXELS")
            .map(std::path::PathBuf::from)
            .expect("set APXINF_D3_REFERENCE_PIXELS to the reference .f32 file");
        let decoded = image::open(&image_path)
            .map_err(|error| Error::Other(format!("decode {}: {error}", image_path.display())))?
            .to_rgb8();
        let (width, height) = (decoded.width() as usize, decoded.height() as usize);
        let (preprocessor, vision) = contract();
        let prepared = preprocess(&preprocessor, &vision, decoded.as_raw(), width, height)?;
        let bytes = std::fs::read(&reference_path)
            .map_err(|error| Error::Other(format!("read {}: {error}", reference_path.display())))?;
        let expected = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(prepared.pixel_values.len(), expected.len());
        let mut worst = 0.0f64;
        let mut first = None;
        for (index, (actual, expected)) in prepared.pixel_values.iter().zip(&expected).enumerate() {
            let delta = (*actual as f64 - *expected as f64).abs();
            if delta > worst {
                worst = delta;
                first = Some((index, *actual, *expected));
            }
        }
        assert!(
            worst < 1e-6,
            "native pixel_values differ from the reference by {worst} at {first:?}"
        );
        Ok(())
    }

    #[test]
    fn identity_resize_copies_and_validation_rejects_bad_buffers() {
        let (preprocessor, vision) = contract();
        let rgb = vec![7u8; 3 * 64 * 64];
        let resized = resize_bicubic(&rgb, 64, 64, 64, 64).unwrap();
        assert_eq!(resized, rgb);
        assert!(resize_bicubic(&rgb[..10], 64, 64, 64, 64).is_err());
        assert!(preprocess(&preprocessor, &vision, &[], 0, 64).is_err());
    }
}
