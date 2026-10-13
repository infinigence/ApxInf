//! Static vision layout prepared once before graph capture.

use apxinf_core::{Error, Result};

use super::WallossVisionConfig;

#[cfg(feature = "cuda")]
use super::backend::{Context, DeviceBuffer};

#[derive(Clone, Debug, PartialEq)]
pub struct VisionGeometry {
    /// Raw patch rows in window-major order.
    pub patch_order: Vec<u32>,
    /// `(height, width)` position IDs matching `patch_order`.
    pub position_ids: Vec<u32>,
    /// Packed local-window boundaries in raw patch tokens.
    pub window_offsets: Vec<u32>,
    /// Packed full-attention boundaries, one segment per temporal frame.
    pub full_offsets: Vec<u32>,
    /// Restores merged vision tokens to image-major order.
    pub reverse_indices: Vec<u32>,
    pub max_window_tokens: usize,
    pub max_full_tokens: usize,
}

impl VisionGeometry {
    pub fn new(config: &WallossVisionConfig, grids: &[[usize; 3]]) -> Result<Self> {
        if grids.is_empty() {
            return Err(Error::Other("walloss vision grid cannot be empty".into()));
        }
        let merge = config.spatial_merge_size;
        let merge_unit = merge * merge;
        let merger_window = config.window_size / merge / config.patch_size;
        if merge == 0 || merger_window == 0 {
            return Err(Error::Other("walloss vision merge geometry is invalid".into()));
        }

        let mut merged_order = Vec::new();
        let mut base_positions = Vec::new();
        let mut window_offsets = vec![0u32];
        let mut full_offsets = vec![0u32];
        let mut merged_base = 0usize;

        for &[t, h, w] in grids {
            if t == 0 || h == 0 || w == 0 || h % merge != 0 || w % merge != 0 {
                return Err(Error::Other(format!(
                    "walloss vision grid [{t}, {h}, {w}] is incompatible with merge size {merge}"
                )));
            }
            let llm_h = h / merge;
            let llm_w = w / merge;
            for _ in 0..t {
                for group_h in 0..llm_h {
                    for group_w in 0..llm_w {
                        for inner_h in 0..merge {
                            for inner_w in 0..merge {
                                base_positions.push(((group_h * merge + inner_h) as u32, (group_w * merge + inner_w) as u32));
                            }
                        }
                    }
                }
                let next = full_offsets.last().copied().unwrap() as usize + h * w;
                full_offsets.push(u32::try_from(next).map_err(|_| Error::Other("walloss full-attention offsets overflow u32".into()))?);
            }

            for frame in 0..t {
                for window_h in (0..llm_h).step_by(merger_window) {
                    for window_w in (0..llm_w).step_by(merger_window) {
                        let before = merged_order.len();
                        for local_h in 0..merger_window.min(llm_h - window_h) {
                            for local_w in 0..merger_window.min(llm_w - window_w) {
                                let index = merged_base
                                    + frame * llm_h * llm_w
                                    + (window_h + local_h) * llm_w
                                    + window_w
                                    + local_w;
                                merged_order.push(index);
                            }
                        }
                        let tokens = (merged_order.len() - before) * merge_unit;
                        let next = window_offsets.last().copied().unwrap() as usize + tokens;
                        window_offsets.push(u32::try_from(next).map_err(|_| Error::Other("walloss window offsets overflow u32".into()))?);
                    }
                }
            }
            merged_base += t * llm_h * llm_w;
        }

        let mut patch_order = Vec::with_capacity(merged_order.len() * merge_unit);
        let mut position_ids = Vec::with_capacity(merged_order.len() * merge_unit * 2);
        for &group in &merged_order {
            for inner in 0..merge_unit {
                let patch = group * merge_unit + inner;
                patch_order.push(u32::try_from(patch).map_err(|_| Error::Other("walloss patch index overflow u32".into()))?);
                let (height, width) = base_positions[patch];
                position_ids.extend([height, width]);
            }
        }
        let mut reverse_indices = vec![0u32; merged_order.len()];
        for (window_position, &original_position) in merged_order.iter().enumerate() {
            reverse_indices[original_position] = u32::try_from(window_position)
                .map_err(|_| Error::Other("walloss reverse index overflow u32".into()))?;
        }
        let max_window_tokens = offset_max_delta(&window_offsets);
        let max_full_tokens = offset_max_delta(&full_offsets);
        Ok(Self {
            patch_order,
            position_ids,
            window_offsets,
            full_offsets,
            reverse_indices,
            max_window_tokens,
            max_full_tokens,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn upload(&self, context: &Context) -> Result<DeviceVisionGeometry> {
        Ok(DeviceVisionGeometry {
            patch_order: upload_u32(context, &self.patch_order)?,
            position_ids: upload_u32(context, &self.position_ids)?,
            window_offsets: upload_u32(context, &self.window_offsets)?,
            full_offsets: upload_u32(context, &self.full_offsets)?,
            reverse_indices: upload_u32(context, &self.reverse_indices)?,
            window_segments: self.window_offsets.len() - 1,
            full_segments: self.full_offsets.len() - 1,
            max_window_tokens: self.max_window_tokens,
            max_full_tokens: self.max_full_tokens,
            host_window_offsets: self.window_offsets.clone(),
            host_full_offsets: self.full_offsets.clone(),
        })
    }
}

#[cfg(feature = "cuda")]
pub struct DeviceVisionGeometry {
    pub patch_order: DeviceBuffer,
    pub position_ids: DeviceBuffer,
    pub window_offsets: DeviceBuffer,
    pub full_offsets: DeviceBuffer,
    pub reverse_indices: DeviceBuffer,
    pub window_segments: usize,
    pub full_segments: usize,
    pub max_window_tokens: usize,
    pub max_full_tokens: usize,
    pub host_window_offsets: Vec<u32>,
    pub host_full_offsets: Vec<u32>,
}

#[cfg(feature = "cuda")]
fn upload_u32(context: &Context, values: &[u32]) -> Result<DeviceBuffer> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let buffer = DeviceBuffer::alloc_zeros(bytes.len(), context.device_id())
        .map_err(Error::Cuda)?;
    buffer.copy_from_host(&bytes).map_err(Error::Cuda)?;
    Ok(buffer)
}

fn offset_max_delta(offsets: &[u32]) -> usize {
    offsets
        .windows(2)
        .map(|pair| (pair[1] - pair[0]) as usize)
        .max()
        .unwrap_or(0)
}

/// Build token-major multimodal position IDs for one unpadded sequence.
pub fn multimodal_position_ids(
    token_ids: &[u32],
    grids: &[[usize; 3]],
    image_token_id: u32,
    spatial_merge: usize,
) -> Result<Vec<u32>> {
    let mut axes = [Vec::new(), Vec::new(), Vec::new()];
    let mut token_offset = 0usize;
    let mut next_position = 0u32;
    for &[t, h, w] in grids {
        let image_start = token_ids[token_offset..]
            .iter()
            .position(|&token| token == image_token_id)
            .map(|index| token_offset + index)
            .ok_or_else(|| Error::Other("walloss prompt has fewer image-token runs than configured grids".into()))?;
        for position in token_offset..image_start {
            let value = next_position + (position - token_offset) as u32;
            for axis in &mut axes {
                axis.push(value);
            }
        }
        next_position += (image_start - token_offset) as u32;
        let llm_h = h / spatial_merge;
        let llm_w = w / spatial_merge;
        let image_tokens = t * llm_h * llm_w;
        if token_ids.get(image_start..image_start + image_tokens).is_none()
            || token_ids[image_start..image_start + image_tokens]
                .iter()
                .any(|&token| token != image_token_id)
        {
            return Err(Error::Other(format!(
                "walloss prompt image run does not contain {image_tokens} tokens"
            )));
        }
        for time in 0..t {
            for height in 0..llm_h {
                for width in 0..llm_w {
                    axes[0].push(next_position + time as u32);
                    axes[1].push(next_position + height as u32);
                    axes[2].push(next_position + width as u32);
                }
            }
        }
        next_position += t.max(llm_h).max(llm_w) as u32;
        token_offset = image_start + image_tokens;
    }
    for position in token_offset..token_ids.len() {
        let value = next_position + (position - token_offset) as u32;
        for axis in &mut axes {
            axis.push(value);
        }
    }
    let mut output = Vec::with_capacity(token_ids.len() * 3);
    for token in 0..token_ids.len() {
        output.extend([axes[0][token], axes[1][token], axes[2][token]]);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> WallossVisionConfig {
        WallossVisionConfig {
            depth: 32,
            hidden_size: 1280,
            intermediate_size: 3420,
            num_heads: 16,
            patch_size: 14,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            out_hidden_size: 2048,
            window_size: 112,
            full_attention_blocks: vec![7, 15, 23, 31],
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        }
    }

    #[test]
    fn produces_static_window_and_full_attention_layouts() {
        let geometry = VisionGeometry::new(&config(), &[[1, 8, 12], [1, 8, 8]]).unwrap();
        assert_eq!(geometry.patch_order.len(), 160);
        assert_eq!(geometry.position_ids.len(), 320);
        assert_eq!(geometry.full_offsets, vec![0, 96, 160]);
        assert_eq!(*geometry.window_offsets.last().unwrap(), 160);
        assert_eq!(geometry.reverse_indices.len(), 40);
        assert_eq!(geometry.max_full_tokens, 96);
    }


    #[test]
    fn positions_advance_by_multimodal_extent() {
        let image = 9;
        let tokens = [1, 2, image, image, image, image, 3, 4];
        let positions = multimodal_position_ids(&tokens, &[[1, 4, 4]], image, 2).unwrap();
        let rows = positions.chunks_exact(3).collect::<Vec<_>>();
        assert_eq!(rows[0], &[0, 0, 0]);
        assert_eq!(rows[2], &[2, 2, 2]);
        assert_eq!(rows[5], &[2, 3, 3]);
        assert_eq!(rows[6], &[4, 4, 4]);
    }
}
