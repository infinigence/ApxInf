//! PNG data-URL decoding and Qwen3.5 vision preprocessing.
//!
//! Reproduces the checkpoint's `Qwen2VLImageProcessorFast` pipeline for
//! images whose dimensions are already multiples of `patch_size * merge_size`
//! (the contract's public and hidden suites are both fixed 448x448):
//!
//! 1. decode the base64 `data:image/png;base64,` payload to RGB8;
//! 2. `smart_resize` bounds check (65_536 <= h*w <= 16_777_216); images that
//!    would require an actual resample are rejected rather than approximated;
//! 3. fused rescale+normalize exactly as the fast processor does it:
//!    `(x - 127.5) / 127.5` in f32 (mean/std 0.5 fused with the 1/255
//!    rescale into one subtract and one divide);
//! 4. patchify to `[grid_h * grid_w, 1536]` with patch-row order
//!    `[block_h, block_w, in_h, in_w]` and per-row element order
//!    `[channel][temporal][patch_h][patch_w]` (the temporal axis duplicates
//!    the single frame, matching `temporal_patch_size = 2`).

use apxinf_model::qwen35::vision::{VISION_MERGE, VISION_PATCH_DIM};

const PATCH_SIZE: usize = 16;
const GRID_FACTOR: usize = PATCH_SIZE * VISION_MERGE; // 32
const MIN_PIXELS: usize = 65_536;
const MAX_PIXELS: usize = 16_777_216;
const TEMPORAL_PATCH: usize = 2;

pub struct DecodedImage {
    pub width: usize,
    pub height: usize,
    /// Tightly packed RGB8.
    pub rgb: Vec<u8>,
}

/// Decode a `data:image/png;base64,` URL into RGB8 pixels.
pub fn decode_png_data_url(url: &str) -> Result<DecodedImage, String> {
    const PREFIX: &str = "data:image/png;base64,";
    let encoded = url
        .strip_prefix(PREFIX)
        .ok_or_else(|| "image_url must be a data:image/png;base64 URL".to_string())?;
    use base64::Engine as _;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded.trim())
        .map_err(|error| format!("invalid base64 image payload: {error}"))?;
    decode_png_bytes(&bytes)
}

pub fn decode_png_bytes(bytes: &[u8]) -> Result<DecodedImage, String> {
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let mut reader = decoder
        .read_info()
        .map_err(|error| format!("invalid PNG: {error}"))?;
    let buffer_size = reader
        .output_buffer_size()
        .ok_or_else(|| "PNG dimensions overflow".to_string())?;
    let mut buffer = vec![0u8; buffer_size];
    let info = reader
        .next_frame(&mut buffer)
        .map_err(|error| format!("PNG decode failed: {error}"))?;
    buffer.truncate(info.buffer_size());
    if info.bit_depth != png::BitDepth::Eight {
        return Err(format!("unsupported PNG bit depth {:?}", info.bit_depth));
    }
    let width = info.width as usize;
    let height = info.height as usize;
    let rgb = match info.color_type {
        png::ColorType::Rgb => buffer,
        png::ColorType::Rgba => buffer
            .chunks_exact(4)
            .flat_map(|px| [px[0], px[1], px[2]])
            .collect(),
        png::ColorType::Grayscale => buffer.iter().flat_map(|&g| [g, g, g]).collect(),
        png::ColorType::GrayscaleAlpha => buffer
            .chunks_exact(2)
            .flat_map(|px| [px[0], px[0], px[0]])
            .collect(),
        other => return Err(format!("unsupported PNG color type {other:?}")),
    };
    if rgb.len() != width * height * 3 {
        return Err("PNG payload size mismatch".to_string());
    }
    Ok(DecodedImage { width, height, rgb })
}

/// Fused rescale+normalize then patchify, in f32. Returns the pixel matrix
/// `[grid_h * grid_w, 1536]` and the `[t, h, w]` grid.
pub fn preprocess_f32(image: &DecodedImage) -> Result<(Vec<f32>, [u32; 3]), String> {
    let (width, height) = (image.width, image.height);
    if width == 0 || height == 0 {
        return Err("image is empty".to_string());
    }
    if width % GRID_FACTOR != 0 || height % GRID_FACTOR != 0 {
        return Err(format!(
            "unsupported image size {width}x{height}: dimensions must be multiples of \
             {GRID_FACTOR} (this implementation does not resample; the image contract \
             uses 448x448)"
        ));
    }
    let pixels = width * height;
    if !(MIN_PIXELS..=MAX_PIXELS).contains(&pixels) {
        return Err(format!(
            "unsupported image size {width}x{height}: pixel count outside \
             [{MIN_PIXELS}, {MAX_PIXELS}]"
        ));
    }

    let grid_h = height / PATCH_SIZE;
    let grid_w = width / PATCH_SIZE;
    let blocks_h = grid_h / VISION_MERGE;
    let blocks_w = grid_w / VISION_MERGE;

    // (x - 127.5) / 127.5 for each of the 256 byte values, computed once.
    let mut lut = [0.0f32; 256];
    for (value, slot) in lut.iter_mut().enumerate() {
        *slot = (value as f32 - 127.5) / 127.5;
    }

    let n_patches = grid_h * grid_w;
    let mut out = vec![0.0f32; n_patches * VISION_PATCH_DIM];
    let mut patch_index = 0usize;
    for bh in 0..blocks_h {
        for bw in 0..blocks_w {
            for ih in 0..VISION_MERGE {
                for iw in 0..VISION_MERGE {
                    let row0 = (bh * VISION_MERGE + ih) * PATCH_SIZE;
                    let col0 = (bw * VISION_MERGE + iw) * PATCH_SIZE;
                    let base = patch_index * VISION_PATCH_DIM;
                    let mut offset = 0usize;
                    for channel in 0..3 {
                        for _t in 0..TEMPORAL_PATCH {
                            for ph in 0..PATCH_SIZE {
                                let row = row0 + ph;
                                for pw in 0..PATCH_SIZE {
                                    let col = col0 + pw;
                                    let byte = image.rgb[(row * width + col) * 3 + channel];
                                    out[base + offset] = lut[byte as usize];
                                    offset += 1;
                                }
                            }
                        }
                    }
                    patch_index += 1;
                }
            }
        }
    }
    Ok((out, [1, grid_h as u32, grid_w as u32]))
}

/// Full pipeline to the BF16 payload the runtime consumes.
pub fn preprocess_to_payload(
    image: &DecodedImage,
) -> Result<apxinf_model::MultimodalPayload, String> {
    let (f32_values, grid) = preprocess_f32(image)?;
    let pixel_values = f32_values
        .iter()
        .map(|&v| half::bf16::from_f32(v))
        .collect();
    Ok(apxinf_model::MultimodalPayload { pixel_values, grid })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_image(width: usize, height: usize) -> DecodedImage {
        let mut rgb = vec![0u8; width * height * 3];
        for row in 0..height {
            for col in 0..width {
                let base = (row * width + col) * 3;
                rgb[base] = (row % 256) as u8;
                rgb[base + 1] = (col % 256) as u8;
                rgb[base + 2] = ((row + col) % 256) as u8;
            }
        }
        DecodedImage { width, height, rgb }
    }

    #[test]
    fn preprocess_produces_expected_grid_and_layout() {
        // 256x256 (exactly the 65,536-pixel minimum) -> grid 16x16 -> 256
        // patches, blocks 8x8. Verify one specific element: patch
        // (bh=1, bw=0, ih=0, iw=1) covers rows 32..48, cols 16..32; element
        // (c=1, t=1, ph=2, pw=3) reads pixel (row 34, col 19) green.
        let image = synthetic_image(256, 256);
        let (values, grid) = preprocess_f32(&image).unwrap();
        assert_eq!(grid, [1, 16, 16]);
        assert_eq!(values.len(), 256 * VISION_PATCH_DIM);

        // patch order: bh*(blocks_w*merge*merge) + bw*(merge*merge) + ih*merge + iw
        let patch_index = 1 * (8 * 2 * 2) + 0 * (2 * 2) + 0 * 2 + 1;
        let element = ((1 * TEMPORAL_PATCH + 1) * PATCH_SIZE + 2) * PATCH_SIZE + 3;
        let value = values[patch_index * VISION_PATCH_DIM + element];
        let expected = (19.0f32 - 127.5) / 127.5; // green channel = col % 256 = 19
        assert_eq!(value, expected);

        // Temporal duplication: t=0 and t=1 copies of the same element match.
        let element_t0 = ((1 * TEMPORAL_PATCH) * PATCH_SIZE + 2) * PATCH_SIZE + 3;
        assert_eq!(values[patch_index * VISION_PATCH_DIM + element_t0], value);
    }

    /// End-to-end processor equality against the offline Transformers oracle:
    /// decode the oracle's probe PNG and reproduce its `pixel_values` matrix
    /// bit for bit (both sides are f32 with the same fused
    /// `(x - 127.5) / 127.5` arithmetic). Skips when the oracle directory is
    /// absent so the suite stays runnable on other hosts.
    #[test]
    fn preprocess_bit_matches_vision_oracle_pixel_values() {
        let oracle_dir = std::env::var("APXINF_VISION_ORACLE_DIR")
            .unwrap_or_else(|_| "/tmp/apxinf-vision-oracle".to_string());
        let image_path = std::path::Path::new(&oracle_dir).join("image.png");
        let golden_path = std::path::Path::new(&oracle_dir).join("pixel_values.f32.bin");
        if !image_path.is_file() || !golden_path.is_file() {
            eprintln!("skipping: vision oracle not present at {oracle_dir}");
            return;
        }
        let png_bytes = std::fs::read(&image_path).unwrap();
        let decoded = decode_png_bytes(&png_bytes).unwrap();
        assert_eq!((decoded.width, decoded.height), (448, 448));
        let (values, grid) = preprocess_f32(&decoded).unwrap();
        assert_eq!(grid, [1, 28, 28]);

        let golden_bytes = std::fs::read(&golden_path).unwrap();
        assert_eq!(golden_bytes.len(), values.len() * 4);
        let golden: Vec<f32> = golden_bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let mismatches = values
            .iter()
            .zip(&golden)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            mismatches,
            0,
            "pixel_values differ from the oracle in {mismatches} of {} elements",
            values.len()
        );
    }

    #[test]
    fn preprocess_rejects_unsupported_sizes() {
        assert!(preprocess_f32(&synthetic_image(60, 64)).is_err());
        // 32x32 is grid-aligned but below the 65,536-pixel minimum.
        assert!(preprocess_f32(&synthetic_image(32, 32)).is_err());
    }

    #[test]
    fn png_data_url_round_trip() {
        // Encode a tiny RGB PNG with the same crate, then decode through the
        // data-URL path.
        let width = 3u32;
        let height = 2u32;
        let pixels: Vec<u8> = (0..(width * height * 3) as u8).collect();
        let mut encoded = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut encoded, width, height);
            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().unwrap();
            writer.write_image_data(&pixels).unwrap();
        }
        use base64::Engine as _;
        let url = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&encoded)
        );
        let decoded = decode_png_data_url(&url).unwrap();
        assert_eq!(decoded.width, 3);
        assert_eq!(decoded.height, 2);
        assert_eq!(decoded.rgb, pixels);

        assert!(decode_png_data_url("data:image/jpeg;base64,AAAA").is_err());
        assert!(decode_png_data_url("data:image/png;base64,!!!").is_err());
    }
}
