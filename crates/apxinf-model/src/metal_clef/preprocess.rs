//! Clef's view of the shared Qwen-style image preprocessing.
//!
//! The canonicalization algorithm (smart resize, bicubic resample, FP32
//! normalize, temporal duplication and patchify) lives in
//! [`crate::vision_preprocess`]; this module only binds it to Clef's checkpoint
//! contract and packs several images into the row-major patch matrix the vision
//! tower consumes.
use crate::vision_preprocess::DecodedImage;
use apxinf_core::{DType, Error, Result};
use sha2::{Digest, Sha256};
use std::path::Path;

/// The checkpoint's patch geometry. Each row vector is
/// `(channel, temporal, patch_row, patch_column)`, the flatten order of the
/// Conv3d patch weight.
pub const PATCH_SIZE: usize = 16;
pub const TEMPORAL_PATCH_SIZE: usize = 2;
pub const SPATIAL_MERGE_SIZE: usize = 2;
pub const PATCH_WIDTH: usize = 3 * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE;

/// One request's worth of prepared images, in request order.
pub struct PreparedImages {
    /// FP32 patch rows, `[rows, PATCH_WIDTH]` in request order.
    pub pixels: Vec<f32>,
    /// `[temporal, height_patches, width_patches]` per image.
    pub grids: Vec<[u32; 3]>,
    /// Identity of the prepared pixels, so a caller can tell two requests apart.
    pub image_key: [u8; 32],
}

impl PreparedImages {
    /// Patch rows across every image.
    pub fn rows(&self) -> usize {
        self.pixels.len() / PATCH_WIDTH
    }

    /// Patch rows after the spatial merge, one per image token.
    pub fn merged_rows(&self) -> usize {
        self.grids
            .iter()
            .map(|[t, h, w]| {
                let merge = SPATIAL_MERGE_SIZE as u32;
                (*t as usize) * (*h as usize / merge as usize) * (*w as usize / merge as usize)
            })
            .sum()
    }
}

/// The checkpoint's preprocessing contract, resolved once at load.
pub struct Preprocessor {
    contract: crate::vision_preprocess::VisionContract,
}

impl Preprocessor {
    /// Reads `preprocessor_config.json` beside the weights.
    pub fn load(model_dir: &Path, max_images: usize) -> Result<Self> {
        Ok(Self {
            contract: crate::vision_preprocess::VisionContract::from_dir(
                model_dir,
                "metal_clef",
                max_images,
                DType::F32,
            )?,
        })
    }

    /// Decodes every image and patchifies it, in request order.
    pub fn prepare(&self, images: &[DecodedImage]) -> Result<PreparedImages> {
        if images.is_empty() {
            return Err(Error::Contract(
                "Clef media request needs at least one image",
            ));
        }
        let mut pixels = Vec::new();
        let mut grids = Vec::with_capacity(images.len());
        for image in images {
            let prepared = crate::vision_preprocess::preprocess(
                &self.contract,
                &image.rgb,
                image.width,
                image.height,
            )?;
            if prepared.pixel_values.len() % PATCH_WIDTH != 0 {
                return Err(Error::Contract(
                    "Clef preprocessing returned a ragged patch matrix",
                ));
            }
            pixels.extend_from_slice(&prepared.pixel_values);
            grids.push(prepared.grid_thw);
        }
        let rows = pixels.len() / PATCH_WIDTH;
        if rows == 0 {
            return Err(Error::Contract("Clef image patches are empty"));
        }
        let mut hasher = Sha256::new();
        for value in &pixels {
            hasher.update(value.to_bits().to_le_bytes());
        }
        for grid in &grids {
            hasher.update(grid[0].to_le_bytes());
            hasher.update(grid[1].to_le_bytes());
            hasher.update(grid[2].to_le_bytes());
        }
        Ok(PreparedImages {
            pixels,
            grids,
            image_key: hasher.finalize().into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merged_rows_follow_the_spatial_merge() {
        let prepared = PreparedImages {
            pixels: vec![0.0; PATCH_WIDTH * 4],
            grids: vec![[1, 4, 4]],
            image_key: [0; 32],
        };
        assert_eq!(prepared.rows(), 4);
        assert_eq!(prepared.merged_rows(), 4);
    }
}
