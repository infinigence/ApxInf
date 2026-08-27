use apxinf_core::{Error, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen35MropePositions {
    pub positions: Vec<[u32; 3]>,
    pub decode_delta: i64,
}

/// Reproduce Transformers Qwen3.5 `get_rope_index` for one unpadded sample.
/// Modality 0 is text and modality 1 is image. Video is deliberately outside
/// the first native ApxInf multimodal contract.
pub fn compute_mrope_positions(
    modality_types: &[u8],
    image_grids: &[[u32; 3]],
    spatial_merge_size: u32,
) -> Result<Qwen35MropePositions> {
    if modality_types.is_empty() || spatial_merge_size == 0 {
        return Err(Error::Other(
            "Qwen3.5 mRoPE requires non-empty modality types and non-zero merge size".into(),
        ));
    }
    let mut positions = Vec::with_capacity(modality_types.len());
    let mut current_position = 0u32;
    let mut image_index = 0usize;
    let mut start = 0usize;
    while start < modality_types.len() {
        let modality = modality_types[start];
        let mut end = start + 1;
        while end < modality_types.len() && modality_types[end] == modality {
            end += 1;
        }
        match modality {
            0 => {
                for offset in 0..end - start {
                    let position = current_position + offset as u32;
                    positions.push([position, position, position]);
                }
                current_position += (end - start) as u32;
            }
            1 => {
                let grid = image_grids.get(image_index).ok_or_else(|| {
                    Error::Other("Qwen3.5 mRoPE image group has no matching grid".into())
                })?;
                image_index += 1;
                let temporal = grid[0];
                let height = grid[1] / spatial_merge_size;
                let width = grid[2] / spatial_merge_size;
                if temporal == 0
                    || height == 0
                    || width == 0
                    || grid[1] % spatial_merge_size != 0
                    || grid[2] % spatial_merge_size != 0
                {
                    return Err(Error::Other(format!(
                        "Qwen3.5 mRoPE invalid image grid {grid:?} for merge {spatial_merge_size}"
                    )));
                }
                let expected = temporal as usize * height as usize * width as usize;
                if end - start != expected {
                    return Err(Error::Other(format!(
                        "Qwen3.5 mRoPE image group has {} tokens, expected {expected} from grid {grid:?}",
                        end - start,
                    )));
                }
                for temporal_position in 0..temporal {
                    for height_position in 0..height {
                        for width_position in 0..width {
                            positions.push([
                                current_position + temporal_position,
                                current_position + height_position,
                                current_position + width_position,
                            ]);
                        }
                    }
                }
                current_position += height.max(width);
            }
            other => {
                return Err(Error::Other(format!(
                    "Qwen3.5 mRoPE modality {other} is unsupported; native v1 supports text and one image"
                )))
            }
        }
        start = end;
    }
    if image_index != image_grids.len() {
        return Err(Error::Other(format!(
            "Qwen3.5 mRoPE consumed {image_index} image grids, got {}",
            image_grids.len()
        )));
    }
    let maximum = positions
        .iter()
        .flat_map(|position| position.iter())
        .copied()
        .max()
        .ok_or_else(|| Error::Other("Qwen3.5 mRoPE produced no positions".into()))?;
    Ok(Qwen35MropePositions {
        decode_delta: maximum as i64 + 1 - modality_types.len() as i64,
        positions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_transformers_single_28x28_image_contract() {
        let mut modality_types = vec![0u8; 46];
        modality_types.extend(std::iter::repeat_n(1u8, 196));
        modality_types.extend(std::iter::repeat_n(0u8, 28));
        let result = compute_mrope_positions(&modality_types, &[[1, 28, 28]], 2).unwrap();
        assert_eq!(result.positions.len(), 270);
        assert_eq!(result.positions[45], [45, 45, 45]);
        assert_eq!(result.positions[46], [46, 46, 46]);
        assert_eq!(result.positions[241], [46, 59, 59]);
        assert_eq!(result.positions[242], [60, 60, 60]);
        assert_eq!(result.positions[269], [87, 87, 87]);
        assert_eq!(result.decode_delta, -182);
    }

    #[test]
    fn rejects_image_token_count_drift() {
        let mut modality_types = vec![0u8; 4];
        modality_types.extend(std::iter::repeat_n(1u8, 195));
        let error = compute_mrope_positions(&modality_types, &[[1, 28, 28]], 2).unwrap_err();
        assert!(error.to_string().contains("expected 196"));
    }
}
