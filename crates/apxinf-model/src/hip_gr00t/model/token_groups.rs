//! Row indices retained for GR00T's alternating cross-attention.
//!
//! The reference passes a boolean key mask on every DiT block. With batch size
//! one, selecting the valid rows once is mathematically equivalent and avoids
//! a model-specific mask contract in the CUDA layer.

use apxinf_core::{Error, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gr00tBackboneTokenGroups {
    pub non_image: Vec<usize>,
    pub image: Vec<usize>,
}

pub fn build_backbone_token_groups(
    token_ids: &[u32],
    attention_mask: &[u8],
    image_token_id: u32,
) -> Result<Gr00tBackboneTokenGroups> {
    if token_ids.len() != attention_mask.len() {
        return Err(Error::Other(format!(
            "GR00T token/mask length mismatch: {} tokens, {} mask entries",
            token_ids.len(),
            attention_mask.len()
        )));
    }
    let mut groups = Gr00tBackboneTokenGroups {
        non_image: Vec::new(),
        image: Vec::new(),
    };
    for (index, (&token, &valid)) in token_ids.iter().zip(attention_mask).enumerate() {
        match valid {
            0 => {}
            1 if token == image_token_id => groups.image.push(index),
            1 => groups.non_image.push(index),
            value => {
                return Err(Error::Other(format!(
                    "GR00T attention_mask[{index}] must be 0 or 1, got {value}"
                )))
            }
        }
    }
    if groups.non_image.is_empty() || groups.image.is_empty() {
        return Err(Error::Other(format!(
            "GR00T AlternateVLDiT requires valid image and non-image tokens, got {} and {}",
            groups.image.len(),
            groups.non_image.len()
        )));
    }
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_only_valid_backbone_rows() {
        let groups = build_backbone_token_groups(&[10, 99, 11, 99], &[1, 1, 0, 1], 99).unwrap();
        assert_eq!(groups.non_image, vec![0]);
        assert_eq!(groups.image, vec![1, 3]);
    }

    #[test]
    fn rejects_non_boolean_masks_and_empty_groups() {
        assert!(build_backbone_token_groups(&[1, 2], &[1, 2], 2).is_err());
        assert!(build_backbone_token_groups(&[1, 2], &[1, 1], 9).is_err());
    }
}
