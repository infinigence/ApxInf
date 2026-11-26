//! Compatibility frontage for the d3 checkpoint contract. The canonicalization
//! algorithm is shared with other Qwen-style towers in vision_preprocess.
use super::config::{PreprocessorContract, VisionContract};
use apxinf_core::Result;
pub use crate::vision_preprocess::{smart_resize, PreparedImage};
#[cfg(test)]
use crate::vision_preprocess::{resize_bicubic,axis_taps};

pub fn preprocess(p: &PreprocessorContract, v: &VisionContract, rgb: &[u8], width: usize, height: usize) -> Result<PreparedImage> {
    let contract=crate::vision_preprocess::VisionContract {
        model:"metal_d3".into(),max_images:1,min_pixels:p.min_pixels,max_pixels:p.max_pixels,
        patch_size:v.patch,temporal_patch_size:v.temporal_patch,spatial_merge_size:v.spatial_merge,
        image_token_id:0,pixel_dtype:apxinf_core::DType::F32,
        image_mean:p.image_mean.map(f64::from),image_std:p.image_std.map(f64::from),
        rescale_factor:1.0/f64::from(u8::MAX),
    };
    crate::vision_preprocess::preprocess(&contract,rgb,width,height)
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
