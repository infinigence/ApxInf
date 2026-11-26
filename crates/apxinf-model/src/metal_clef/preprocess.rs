//! PNG decoding and Qwen3-VL image preprocessing for the Clef media path.
//!
//! Numeric behaviour mirrors the MLX-VLM processor the reference checkpoint was
//! trained with (`mlx_vlm.models.qwen3_vl.processing_qwen3_vl`): a
//! `processor_config.json` smart resize, Pillow's 8-bit BICUBIC resample, uint8
//! rescale/normalize in FP32, temporal duplication and patchify. It runs on the
//! CPU inside the timed request: no Python and no pre-decoded tensor at the API.
use crate::decision::DecisionImage;
use apxinf_core::{Error, Result};
use sha2::{Digest, Sha256};

/// Vision patch geometry, from `config.json` `vision_config`.
pub const PATCH_SIZE: usize = 16;
pub const TEMPORAL_PATCH_SIZE: usize = 2;
pub const SPATIAL_MERGE_SIZE: usize = 2;
/// Flattened patch width `channels * temporal * patch * patch`.
pub const PATCH_WIDTH: usize = 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE;
/// One token's merge block, in pixels.
const MERGE: usize = SPATIAL_MERGE_SIZE;
/// Resize factor: `patch_size * merge_size`.
const FACTOR: usize = PATCH_SIZE * MERGE;
/// `processor_config.json` `image_processor.size`.
const MIN_PIXELS: usize = 65536;
const MAX_PIXELS: usize = 16777216;

/// Images are decoded and preprocessed without keeping the RGB buffers.
pub struct PreparedImages {
    /// Patch rows, each `PATCH_WIDTH` values in `(channel, temporal, row, column)`
    /// order: the order the checkpoint's Conv3d patch weight flattens to.
    pub pixels: Vec<f32>,
    /// `[temporal, height / patch, width / patch]` per image, in input order.
    pub grids: Vec<[u32; 3]>,
    /// SHA-256 of the prepared patch matrix and grids. Tier-2 uses this as the
    /// image identity so two different images with identical placeholder token
    /// ids can never share a cached prefix.
    pub image_key: [u8; 32],
}

impl PreparedImages {
    pub fn rows(&self) -> usize {
        self.grids
            .iter()
            .map(|grid| (grid[0] * grid[1] * grid[2]) as usize)
            .sum()
    }

    /// Merged rows one image contributes to the language prompt.
    pub fn merged_rows(&self) -> usize {
        self.grids
            .iter()
            .map(|grid| {
                let per_image = (grid[0] as usize) * (grid[1] as usize) * (grid[2] as usize);
                per_image / (MERGE * MERGE)
            })
            .sum()
    }
}

/// Stable identity of the exact vision input: the ordered patch matrix and
/// grids. The digest is computed on host f32 bit patterns, so it changes when
/// any preprocessed pixel or grid changes, but not when the same visual input
/// is supplied through a different PNG encoding.
fn prepared_image_key(pixels: &[f32], grids: &[[u32; 3]]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update((pixels.len() as u64).to_le_bytes());
    hasher.update((grids.len() as u64).to_le_bytes());
    for grid in grids {
        for value in grid {
            hasher.update(value.to_le_bytes());
        }
    }
    // Buffer the pixel bits first. Calling `update` once per f32 (hundreds of
    // thousands of times for a 480-row image) dominates the request on the
    // M1 Pro; one bulk update keeps the key's cost in the low single-digit ms.
    let mut bytes = Vec::with_capacity(pixels.len() * 4);
    for value in pixels {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    hasher.update(&bytes);
    hasher.finalize().into()
}

/// Decode and preprocess every image of one request, in request order.
pub fn prepare(images: &[DecisionImage<'_>]) -> Result<PreparedImages> {
    if images.is_empty() {
        return Err(Error::Contract(
            "Clef media request needs at least one image",
        ));
    }
    let mut pixels = Vec::new();
    let mut grids = Vec::with_capacity(images.len());
    for image in images {
        let bytes = match image {
            DecisionImage::Path(path) => std::fs::read(path)
                .map_err(|e| Error::Other(format!("Clef image {}: {e}", path.to_string_lossy())))?,
            DecisionImage::Png(bytes) => bytes.to_vec(),
        };
        let prepared = preprocess_png(&bytes)?;
        pixels.extend_from_slice(&prepared.pixels);
        grids.push(prepared.grid);
    }
    let rows = pixels.len() / PATCH_WIDTH;
    if rows == 0 || pixels.len() % PATCH_WIDTH != 0 {
        return Err(Error::Contract("Clef image patches are empty or ragged"));
    }
    let image_key = prepared_image_key(&pixels, &grids);
    let prepared = PreparedImages {
        pixels,
        grids,
        image_key,
    };
    dump_if_requested(&prepared)?;
    Ok(prepared)
}

/// `APXINF_CLEF_DUMP_PIXELS=<prefix>` writes `<prefix>.f32` (native-endian FP32
/// patch matrix) and `<prefix>.json` (rows, width, grids) for the stage-by-stage
/// parity harness. Off by default; one request per process keeps it simple.
fn dump_if_requested(prepared: &PreparedImages) -> Result<()> {
    let Some(prefix) = std::env::var_os("APXINF_CLEF_DUMP_PIXELS") else {
        return Ok(());
    };
    let prefix = std::path::PathBuf::from(prefix);
    let pixels = prefix.with_extension("f32");
    let mut bytes = Vec::with_capacity(prepared.pixels.len() * 4);
    for value in &prepared.pixels {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    std::fs::write(&pixels, bytes)
        .map_err(|e| Error::Other(format!("dump {}: {e}", pixels.display())))?;
    let meta = format!(
        "{{\"rows\": {}, \"patch_width\": {}, \"grids\": {}}}",
        prepared.rows(),
        PATCH_WIDTH,
        serde_json::json!(prepared.grids)
    );
    std::fs::write(prefix.with_extension("json"), meta)
        .map_err(|e| Error::Other(format!("dump {prefix:?}: {e}")))?;
    Ok(())
}

/// Patchify a normalized-free RGB8 resized image into the processor's patch
/// matrix.
///
/// Token order is `(merged_row, merged_column, intra_row, intra_column)`; each
/// row vector is `(channel, temporal, patch_row, patch_column)`, the flatten
/// order of the checkpoint's Conv3d patch weight. The two temporal frames are
/// the same image. `patchify_locks_the_processor_order` pins both orders.
fn patchify(resized: &[u8], width: usize, height: usize) -> Result<(Vec<f32>, usize)> {
    if width % (PATCH_SIZE * MERGE) != 0 || height % (PATCH_SIZE * MERGE) != 0 {
        return Err(Error::Other(format!(
            "Clef resized image {width}x{height} is not a multiple of patch {} and merge {MERGE}",
            PATCH_SIZE
        )));
    }
    if resized.len() != width * height * 3 {
        return Err(Error::Other(format!(
            "Clef decoded image has {} bytes, expected {}x{}x3",
            resized.len(),
            width,
            height
        )));
    }
    let merged_height = height / PATCH_SIZE / MERGE;
    let merged_width = width / PATCH_SIZE / MERGE;
    let rows = merged_height * merged_width * MERGE * MERGE;
    let mut pixels = Vec::with_capacity(rows * PATCH_WIDTH);
    for merged_row in 0..merged_height {
        for merged_column in 0..merged_width {
            for inner_row in 0..MERGE {
                for inner_column in 0..MERGE {
                    let y = (merged_row * MERGE + inner_row) * PATCH_SIZE;
                    let x = (merged_column * MERGE + inner_column) * PATCH_SIZE;
                    for channel in 0..3 {
                        for _ in 0..TEMPORAL_PATCH_SIZE {
                            for patch_row in 0..PATCH_SIZE {
                                for patch_column in 0..PATCH_SIZE {
                                    let source =
                                        ((y + patch_row) * width + x + patch_column) * 3 + channel;
                                    pixels.push(normalize(resized[source]));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((pixels, rows))
}

/// One image from in-memory PNG bytes.
pub struct PreparedImage {
    pub pixels: Vec<f32>,
    pub grid: [u32; 3],
}

pub fn preprocess_png(png: &[u8]) -> Result<PreparedImage> {
    let (width, height, rgb) = decode_png(png)?;
    let (resized_height, resized_width) = smart_resize(height, width)?;
    let resized = resize_bicubic(&rgb, width, height, resized_width, resized_height);
    let grid = [
        1,
        (resized_height / PATCH_SIZE) as u32,
        (resized_width / PATCH_SIZE) as u32,
    ];
    let (pixels, _) = patchify(&resized, resized_width, resized_height)?;
    Ok(PreparedImage { pixels, grid })
}

/// `(x * rescale_factor - mean) / std` in FP32, as the numpy processor does.
fn normalize(value: u8) -> f32 {
    let scaled = value as f32 * (1.0 / 255.0);
    (scaled - 0.5) / 0.5
}

/// Python `round(h / factor) * factor` (ties to even) and the min/max pixel
/// branches of `_smart_resize_image`.
pub fn smart_resize(height: usize, width: usize) -> Result<(usize, usize)> {
    if height == 0 || width == 0 {
        return Err(Error::Other("Clef image has a zero dimension".into()));
    }
    if height.max(width) / height.min(width) > 200 {
        return Err(Error::Other(format!(
            "Clef image aspect ratio {} exceeds 200",
            height.max(width) / height.min(width)
        )));
    }
    let mut height_bar = round_div(height, FACTOR) * FACTOR;
    let mut width_bar = round_div(width, FACTOR) * FACTOR;
    if height_bar * width_bar > MAX_PIXELS {
        let beta = ((height * width) as f64 / MAX_PIXELS as f64).sqrt();
        height_bar = ((height as f64 / beta / FACTOR as f64).floor() as usize * FACTOR).max(FACTOR);
        width_bar = ((width as f64 / beta / FACTOR as f64).floor() as usize * FACTOR).max(FACTOR);
    } else if height_bar * width_bar < MIN_PIXELS {
        let beta = (MIN_PIXELS as f64 / (height * width) as f64).sqrt();
        height_bar = (height as f64 * beta / FACTOR as f64).ceil() as usize * FACTOR;
        width_bar = (width as f64 * beta / FACTOR as f64).ceil() as usize * FACTOR;
    }
    Ok((height_bar, width_bar))
}

fn round_div(value: usize, divisor: usize) -> usize {
    let quotient = value / divisor;
    match (value % divisor) * 2 {
        remainder if remainder < divisor => quotient,
        remainder if remainder > divisor => quotient + 1,
        _ if quotient % 2 == 0 => quotient,
        _ => quotient + 1,
    }
}

/// Pillow BICUBIC (`a = -0.5`) resample, separable, horizontal pass first and
/// the intermediate rounded and clipped to uint8, exactly as `Image.resize`
/// does for an `RGB` uint8 image. The transform is identity in size.
pub fn resize_bicubic(
    source: &[u8],
    width: usize,
    height: usize,
    target_width: usize,
    target_height: usize,
) -> Vec<u8> {
    if (width, height) == (target_width, target_height) {
        return source.to_vec();
    }
    let horizontal = taps(width, target_width);
    let vertical = taps(height, target_height);
    let mut intermediate = vec![0u8; height * target_width * 3];
    for row in 0..height {
        let line = &source[row * width * 3..(row + 1) * width * 3];
        for (column, tap) in horizontal.iter().enumerate() {
            for channel in 0..3 {
                let sum = tap
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(index, weight)| line[(tap.start + index) * 3 + channel] as f64 * weight)
                    .sum();
                intermediate[row * target_width * 3 + column * 3 + channel] = clip_round(sum);
            }
        }
    }
    let mut output = vec![0u8; target_height * target_width * 3];
    for (row, tap) in vertical.iter().enumerate() {
        for column in 0..target_width {
            for channel in 0..3 {
                let sum = tap
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(index, weight)| {
                        intermediate[(tap.start + index) * target_width * 3 + column * 3 + channel]
                            as f64
                            * weight
                    })
                    .sum();
                output[row * target_width * 3 + column * 3 + channel] = clip_round(sum);
            }
        }
    }
    output
}

struct Tap {
    start: usize,
    weights: Vec<f64>,
}

/// Pillow `precompute_coeffs`: `support = 2 * max(scale, 1)`, centre at
/// `(x + 0.5) * scale`, weights normalised to one.
fn taps(input: usize, output: usize) -> Vec<Tap> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    let step = 1.0 / filter_scale;
    let mut result = Vec::with_capacity(output);
    for index in 0..output {
        let center = (index as f64 + 0.5) * scale;
        let start = ((center - support + 0.5) as isize).max(0) as usize;
        let end = (((center + support + 0.5) as isize).max(0) as usize).min(input);
        let count = end.max(start + 1) - start;
        let mut weights = (0..count)
            .map(|offset| bicubic((offset as f64 + start as f64 - center + 0.5) * step))
            .collect::<Vec<_>>();
        let total = weights.iter().sum::<f64>();
        if total != 0.0 {
            for weight in &mut weights {
                *weight /= total;
            }
        }
        result.push(Tap { start, weights });
    }
    result
}

/// Catmull-Rom kernel with `a = -0.5`, as Pillow's `BICUBIC` filter.
fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x <= 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

fn clip_round(value: f64) -> u8 {
    (value + 0.5).floor().clamp(0.0, 255.0) as u8
}

/// Decode an 8-bit, non-interlaced PNG to packed RGB. Uses the system zlib for
/// the DEFLATE stream; no new crate, no image library and no colour management.
pub fn decode_png(png: &[u8]) -> Result<(usize, usize, Vec<u8>)> {
    const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    if png.len() < 8 || png[..8] != SIGNATURE {
        return Err(Error::Other("Clef image is not a PNG".into()));
    }
    let mut offset = 8;
    let (mut width, mut height) = (0usize, 0usize);
    let mut color_type = 0u8;
    let mut bit_depth = 0u8;
    let mut interlace = 0u8;
    let mut compressed = Vec::new();
    let mut seen_header = false;
    while offset + 8 <= png.len() {
        let length = u32::from_be_bytes(png[offset..offset + 4].try_into().unwrap()) as usize;
        let kind = &png[offset + 4..offset + 8];
        let data_start = offset + 8;
        let data_end = data_start
            .checked_add(length)
            .filter(|end| end + 4 <= png.len())
            .ok_or_else(|| Error::Other("Clef PNG chunk overruns the file".into()))?;
        match kind {
            b"IHDR" => {
                if length != 13 {
                    return Err(Error::Other("Clef PNG has a malformed IHDR".into()));
                }
                let header = &png[data_start..data_end];
                width = u32::from_be_bytes(header[0..4].try_into().unwrap()) as usize;
                height = u32::from_be_bytes(header[4..8].try_into().unwrap()) as usize;
                bit_depth = header[8];
                color_type = header[9];
                interlace = header[12];
                seen_header = true;
            }
            b"IDAT" => compressed.extend_from_slice(&png[data_start..data_end]),
            b"IEND" => break,
            _ => {}
        }
        offset = data_end + 4;
    }
    if !seen_header || width == 0 || height == 0 {
        return Err(Error::Other("Clef PNG has no usable IHDR".into()));
    }
    if bit_depth != 8 || interlace != 0 {
        return Err(Error::Other(format!(
            "Clef PNG must be 8-bit and non-interlaced, got bit depth {bit_depth} interlace {interlace}"
        )));
    }
    let channels = match color_type {
        0 => 1,
        2 => 3,
        4 => 2,
        6 => 4,
        other => {
            return Err(Error::Other(format!(
                "Clef PNG colour type {other} is unsupported (palette and 16-bit images are not)"
            )))
        }
    };
    let stride = width * channels;
    let raw_length = (stride + 1) * height;
    let inflated = inflate(&compressed, raw_length)?;
    let mut rgb = vec![0u8; width * height * 3];
    let mut previous = vec![0u8; stride];
    let mut current = vec![0u8; stride];
    for row in 0..height {
        let line = &inflated[row * (stride + 1)..(row + 1) * (stride + 1)];
        let filter = line[0];
        for column in 0..stride {
            let left = if column >= channels {
                current[column - channels] as i32
            } else {
                0
            };
            let up = previous[column] as i32;
            let upper_left = if column >= channels {
                previous[column - channels] as i32
            } else {
                0
            };
            let value = line[1 + column] as i32
                + match filter {
                    0 => 0,
                    1 => left,
                    2 => up,
                    3 => (left + up) / 2,
                    4 => paeth(left, up, upper_left),
                    other => {
                        return Err(Error::Other(format!("Clef PNG uses filter type {other}")))
                    }
                };
            current[column] = value as u8;
        }
        for column in 0..width {
            let (r, g, b) = match channels {
                1 => (current[column], current[column], current[column]),
                2 => {
                    let grey = current[column * 2];
                    (grey, grey, grey)
                }
                3 => (
                    current[column * 3],
                    current[column * 3 + 1],
                    current[column * 3 + 2],
                ),
                _ => (
                    current[column * 4],
                    current[column * 4 + 1],
                    current[column * 4 + 2],
                ),
            };
            rgb[(row * width + column) * 3] = r;
            rgb[(row * width + column) * 3 + 1] = g;
            rgb[(row * width + column) * 3 + 2] = b;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    Ok((width, height, rgb))
}

fn paeth(left: i32, up: i32, upper_left: i32) -> i32 {
    let estimate = left + up - upper_left;
    let (da, db, dc) = (
        (estimate - left).abs(),
        (estimate - up).abs(),
        (estimate - upper_left).abs(),
    );
    if da <= db && da <= dc {
        left
    } else if db <= dc {
        up
    } else {
        upper_left
    }
}

#[link(name = "z")]
extern "C" {
    fn uncompress(
        destination: *mut u8,
        destination_length: *mut std::os::raw::c_ulong,
        source: *const u8,
        source_length: std::os::raw::c_ulong,
    ) -> i32;
}

/// zlib `uncompress` into an exactly-sized buffer; a size mismatch is an error,
/// since a PNG's IDAT inflates to `(stride + 1) * height` bytes.
fn inflate(compressed: &[u8], expected: usize) -> Result<Vec<u8>> {
    if expected == 0 || i32::try_from(expected).is_err() {
        return Err(Error::Other("Clef PNG is too large".into()));
    }
    let mut output = vec![0u8; expected];
    let mut length = expected as std::os::raw::c_ulong;
    let status = unsafe {
        uncompress(
            output.as_mut_ptr(),
            &mut length,
            compressed.as_ptr(),
            compressed.len() as std::os::raw::c_ulong,
        )
    };
    if status != 0 {
        return Err(Error::Other(format!(
            "Clef PNG IDAT does not inflate (zlib status {status})"
        )));
    }
    if length as usize != expected {
        return Err(Error::Other(format!(
            "Clef PNG inflates to {length} bytes, expected {expected}"
        )));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three hand-built 8-bit PNGs: opaque RGB (filter 0), RGBA (alpha
    /// dropped, as `PIL.Image.convert("RGB")` does), and per-row filter types
    /// 0 (None), 1 (Sub) and 2 (Up). Generated with zlib and manual CRCs; the
    /// expected pixels below are the unfiltered values.
    const RGB_PNG: &[u8] = &[
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x08, 0x02, 0x00, 0x00, 0x00, 0xfd,
        0xd4, 0x9a, 0x73, 0x00, 0x00, 0x00, 0x16, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0xe0,
        0x12, 0x91, 0xd3, 0x30, 0xb2, 0x61, 0xb0, 0x71, 0x0b, 0x88, 0x4a, 0xc9, 0x03, 0x00, 0x0e,
        0x46, 0x02, 0xd1, 0x61, 0x7a, 0xc5, 0xf5, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44,
        0xae, 0x42, 0x60, 0x82,
    ];
    const RGBA_PNG: &[u8] = &[
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x08, 0x06, 0x00, 0x00, 0x00, 0x72,
        0xb6, 0x0d, 0x24, 0x00, 0x00, 0x00, 0x1a, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0x60,
        0x64, 0x62, 0xfe, 0xcf, 0xc2, 0xca, 0xf6, 0x9f, 0x81, 0x83, 0x93, 0xeb, 0x3f, 0x37, 0x0f,
        0xef, 0x7f, 0x00, 0x20, 0x29, 0x04, 0x51, 0x01, 0x32, 0x7d, 0x33, 0x00, 0x00, 0x00, 0x00,
        0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
    ];
    const FILTERED_PNG: &[u8] = &[
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x08, 0x02, 0x00, 0x00, 0x00, 0x36,
        0x88, 0x49, 0xd6, 0x00, 0x00, 0x00, 0x19, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0xe0,
        0x12, 0x91, 0xd3, 0x30, 0xb2, 0x61, 0x74, 0x0b, 0x88, 0x92, 0x93, 0x93, 0x63, 0x62, 0x05,
        0x03, 0x00, 0x1c, 0x9c, 0x02, 0x3e, 0x37, 0x66, 0x6a, 0xa0, 0x00, 0x00, 0x00, 0x00, 0x49,
        0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
    ];

    #[test]
    fn patchify_locks_the_processor_order() {
        // 64x32 RGB: 2 merged rows, 1 merged column -> 8 patch rows. The red
        // channel encodes the source row, green the source column, so the patch
        // order and the row-vector order are both observable.
        let (width, height) = (32usize, 64usize);
        let mut rgb = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                rgb.push(y as u8);
                rgb.push(x as u8);
                rgb.push(7);
            }
        }
        let (pixels, rows) = patchify(&rgb, width, height).unwrap();
        assert_eq!(rows, 8);
        assert_eq!(pixels.len(), rows * PATCH_WIDTH);
        let token = |index: usize| &pixels[index * PATCH_WIDTH..(index + 1) * PATCH_WIDTH];
        // Row vector order: channel 0 (red = source row) over two temporal
        // frames of 16x16 patch values, then channel 1 (green = source column).
        let red = token(0);
        assert_eq!(red[0], normalize(0)); // patch row 0, column 0
        assert_eq!(red[1], normalize(0)); // patch column 1, same source row
        assert_eq!(red[PATCH_SIZE], normalize(1)); // patch row 1
        assert_eq!(red[PATCH_SIZE * PATCH_SIZE], normalize(0)); // temporal frame 2
        assert_eq!(red[2 * PATCH_SIZE * PATCH_SIZE], normalize(0)); // green, column 0
        assert_eq!(
            red[2 * PATCH_SIZE * PATCH_SIZE + 1],
            normalize(1),
            "green encodes the source column"
        );
        // Token order: (merged_row, merged_column, intra_row, intra_column).
        // Token 1 is intra column 1, i.e. source column 16.
        assert_eq!(
            token(1)[2 * PATCH_SIZE * PATCH_SIZE],
            normalize(16),
            "intra column is the fastest axis"
        );
        // Token 2 is intra row 1: source row 16.
        assert_eq!(token(2)[0], normalize(16), "intra row before intra column");
        // Token 4 is merged row 1: source row 32.
        assert_eq!(token(4)[0], normalize(32));
        // Patch dimensions must cover whole merge blocks.
        assert!(patchify(&rgb, 31, 64).is_err());
        assert!(patchify(&rgb, 32, 63).is_err());
    }

    #[test]
    fn png_decode_matches_the_hand_built_pixels() {
        let cases: [(&[u8], usize, usize, &[(u8, u8, u8)]); 3] = [
            (
                RGB_PNG,
                2,
                2,
                &[(10, 20, 30), (40, 50, 60), (60, 70, 80), (90, 100, 110)],
            ),
            (
                RGBA_PNG,
                2,
                2,
                &[(1, 2, 3), (4, 5, 6), (8, 9, 10), (11, 12, 13)],
            ),
            (
                FILTERED_PNG,
                2,
                3,
                &[
                    (10, 20, 30),
                    (40, 50, 60),
                    (70, 80, 90),
                    (100, 110, 120),
                    (75, 85, 95),
                    (105, 115, 125),
                ],
            ),
        ];
        for (png, width, height, pixels) in cases {
            let (decoded_width, decoded_height, rgb) = decode_png(png).unwrap();
            assert_eq!((decoded_width, decoded_height), (width, height));
            let observed = rgb
                .chunks_exact(3)
                .map(|pixel| (pixel[0], pixel[1], pixel[2]))
                .collect::<Vec<_>>();
            assert_eq!(observed, pixels, "{width}x{height}");
        }
        assert!(decode_png(b"not a png").is_err());
    }

    #[test]
    fn preprocess_resizes_patchifies_and_reports_processor_grids() {
        // A 2x2 crop is below `min_pixels`, so the processor upscales it to
        // 256x256: 16x16 patches, 8x8 merged rows, all patch units.
        let prepared = preprocess_png(RGB_PNG).unwrap();
        assert_eq!(prepared.grid, [1, 16, 16]);
        assert_eq!(prepared.pixels.len(), 16 * 16 * PATCH_WIDTH);
        let first = &prepared.pixels[..PATCH_WIDTH];
        // Patch order: channel, temporal frame, patch row, patch column. The
        // top-left patch is Pillow 12.3.0's BICUBIC upscale of the 2x2 crop to
        // 256x256 at pixel (0, 0), which is (0, 10, 20) - the Catmull-Rom
        // overshoot clips the red channel at the border.
        assert_eq!(first[0], normalize(0), "red channel");
        assert_eq!(
            first[2 * PATCH_SIZE * PATCH_SIZE],
            normalize(10),
            "green channel"
        );
        assert_eq!(
            first[4 * PATCH_SIZE * PATCH_SIZE],
            normalize(20),
            "blue channel"
        );
        assert!(first.iter().all(|value| value.is_finite()));
        let patch = PATCH_SIZE * PATCH_SIZE;
        for channel in 0..3 {
            let frame = channel * TEMPORAL_PATCH_SIZE * patch;
            assert_eq!(
                &first[frame..frame + patch],
                &first[frame + patch..frame + 2 * patch],
                "channel {channel}"
            );
        }
    }

    #[test]
    fn image_key_tracks_prepared_pixels_and_grids() {
        let base = prepared_image_key(&[1.0, 2.0], &[[1, 16, 16]]);
        assert_eq!(base, prepared_image_key(&[1.0, 2.0], &[[1, 16, 16]]));
        // Increment the f32 bit pattern so the perturbation cannot round away.
        let changed = f32::from_bits(2.0f32.to_bits() + 1);
        assert_ne!(base, prepared_image_key(&[1.0, changed], &[[1, 16, 16]]));
        assert_ne!(base, prepared_image_key(&[1.0], &[[1, 16, 16]]));
        assert_ne!(base, prepared_image_key(&[1.0, 2.0], &[[1, 16, 32]]));
        assert_ne!(
            base,
            prepared_image_key(&[1.0, 2.0], &[[1, 16, 16], [1, 16, 16]])
        );
    }

    #[test]
    fn prepare_publishes_the_key_of_its_prepared_input() {
        let prepared = prepare(&[DecisionImage::Png(RGB_PNG)]).unwrap();
        assert_eq!(
            prepared.image_key,
            prepared_image_key(&prepared.pixels, &prepared.grids)
        );
    }

    /// Pillow 12.3.0 `Image.resize(..., BICUBIC)` reference values for a
    /// non-integer downscale and a 3x upscale of the same 8x8 gradient.
    fn gradient() -> Vec<u8> {
        let mut source = Vec::new();
        for row in 0..8u32 {
            for column in 0..8u32 {
                let value = (row * 8 + column) * 4;
                source.extend_from_slice(&[value as u8, (255 - value) as u8, 128]);
            }
        }
        source
    }

    #[test]
    fn bicubic_downscale_matches_pillow_byte_for_byte() {
        let expected: [u8; 90] = [
            11, 244, 128, 16, 239, 128, 21, 234, 128, 27, 228, 128, 32, 223, 128, 37, 218, 128, 61,
            194, 128, 66, 189, 128, 71, 184, 128, 77, 178, 128, 82, 173, 128, 87, 168, 128, 113,
            142, 128, 118, 137, 128, 123, 132, 128, 129, 126, 128, 134, 121, 128, 139, 116, 128,
            165, 90, 128, 170, 85, 128, 175, 80, 128, 181, 74, 128, 186, 69, 128, 191, 64, 128,
            215, 40, 128, 220, 35, 128, 225, 30, 128, 231, 24, 128, 236, 19, 128, 241, 14, 128,
        ];
        assert_eq!(resize_bicubic(&gradient(), 8, 8, 6, 5), expected);
    }

    #[test]
    fn bicubic_upscale_matches_pillow_rows() {
        let up = resize_bicubic(&gradient(), 8, 8, 24, 25);
        let first: [u8; 72] = [
            0, 255, 128, 0, 255, 128, 0, 255, 128, 0, 255, 128, 1, 254, 128, 2, 253, 128, 4, 251,
            128, 5, 250, 128, 6, 249, 128, 8, 247, 128, 9, 246, 128, 10, 245, 128, 12, 243, 128,
            13, 242, 128, 14, 241, 128, 16, 239, 128, 17, 238, 128, 18, 237, 128, 20, 235, 128, 21,
            234, 128, 23, 232, 128, 24, 231, 128, 25, 230, 128, 25, 230, 128,
        ];
        let last: [u8; 72] = [
            227, 28, 128, 227, 28, 128, 228, 27, 128, 229, 26, 128, 231, 24, 128, 232, 23, 128,
            234, 21, 128, 235, 20, 128, 236, 19, 128, 238, 17, 128, 239, 16, 128, 240, 15, 128,
            242, 13, 128, 243, 12, 128, 244, 11, 128, 246, 9, 128, 247, 8, 128, 248, 7, 128, 250,
            5, 128, 251, 4, 128, 253, 2, 128, 254, 1, 128, 255, 0, 128, 255, 0, 128,
        ];
        assert_eq!(&up[..72], &first[..]);
        assert_eq!(&up[24 * 24 * 3..], &last[..]);
    }

    #[test]
    fn resize_is_identity_at_the_same_size() {
        let source = (0..8 * 8 * 3).map(|v| v as u8).collect::<Vec<_>>();
        assert_eq!(resize_bicubic(&source, 8, 8, 8, 8), source);
    }

    #[test]
    fn smart_resize_matches_the_reference_processor() {
        // Recorded from mlx-vlm `Qwen3VLImageProcessor` with the checkpoint's
        // `size = {shortest_edge: 65536, longest_edge: 16777216}`.
        for (height, width, expected) in [
            (400usize, 320usize, (384usize, 320usize)),
            (288, 384, (288, 384)),
            (320, 480, (320, 480)),
            (64, 80, (256, 288)),
            (280, 350, (288, 352)),
            (100, 100, (256, 256)),
        ] {
            assert_eq!(
                smart_resize(height, width).unwrap(),
                expected,
                "{height}x{width}"
            );
        }
    }

    #[test]
    fn round_div_uses_python_rounding() {
        assert_eq!(round_div(12, 32), 0);
        assert_eq!(round_div(16, 32), 0); // 0.5 -> 0, ties to even
        assert_eq!(round_div(200, 32), 6); // 6.25 -> 6
        assert_eq!(round_div(400, 32), 12); // 12.5 -> 12
        assert_eq!(round_div(48, 32), 2); // 1.5 -> 2
        assert_eq!(round_div(400, 32) * 32, 384);
    }

    #[test]
    fn normalize_matches_the_float32_processor() {
        for value in [0u8, 1, 127, 255] {
            let expected = ((value as f32) * (1.0 / 255.0) - 0.5) / 0.5;
            assert_eq!(normalize(value), expected, "{value}");
        }
    }
}
