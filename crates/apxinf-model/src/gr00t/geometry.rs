use apxinf_core::{Error, Result};

use super::Gr00tConfig;

/// Encoder source used by one GR00T DiT transformer block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gr00tDitAttentionSource {
    FullBackbone,
    NonImageBackbone,
    ImageBackbone,
    StateActionSelf,
}

/// Return the exact AlternateVLDiT attention pattern for a layer.
pub fn dit_attention_source(config: &Gr00tConfig, layer: usize) -> Result<Gr00tDitAttentionSource> {
    if layer >= config.diffusion.num_layers {
        return Err(Error::Other(format!(
            "GR00T DiT layer {layer} is outside 0..{}",
            config.diffusion.num_layers
        )));
    }
    if layer % 2 == 1 && config.diffusion.interleave_self_attention {
        return Ok(Gr00tDitAttentionSource::StateActionSelf);
    }
    if !config.use_alternate_vl_dit {
        return Ok(Gr00tDitAttentionSource::FullBackbone);
    }
    let period = config
        .attend_text_every_n_blocks
        .checked_mul(2)
        .ok_or_else(|| Error::Other("GR00T attention alternation period overflow".into()))?;
    if layer % period == 0 {
        Ok(Gr00tDitAttentionSource::NonImageBackbone)
    } else {
        Ok(Gr00tDitAttentionSource::ImageBackbone)
    }
}

/// Row indices retained for GR00T's alternating cross-attention.
///
/// The reference passes a boolean key mask on every DiT block. With batch size
/// one, selecting the valid rows once is mathematically equivalent and avoids
/// a model-specific mask contract in the CUDA layer.
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
    fn released_alternate_vl_pattern_matches_python() {
        let mut config = Gr00tConfig::default();
        config.diffusion.num_layers = 8;
        let actual = (0..8)
            .map(|layer| dit_attention_source(&config, layer).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            vec![
                Gr00tDitAttentionSource::NonImageBackbone,
                Gr00tDitAttentionSource::StateActionSelf,
                Gr00tDitAttentionSource::ImageBackbone,
                Gr00tDitAttentionSource::StateActionSelf,
                Gr00tDitAttentionSource::NonImageBackbone,
                Gr00tDitAttentionSource::StateActionSelf,
                Gr00tDitAttentionSource::ImageBackbone,
                Gr00tDitAttentionSource::StateActionSelf,
            ]
        );
    }

    #[test]
    fn splits_only_valid_backbone_rows() {
        let groups = build_backbone_token_groups(&[10, 99, 11, 99], &[1, 1, 0, 1], 99).unwrap();
        assert_eq!(groups.non_image, vec![0]);
        assert_eq!(groups.image, vec![1, 3]);
    }

    #[test]
    fn non_alternating_dit_uses_the_complete_backbone() {
        let mut config = Gr00tConfig::default();
        config.use_alternate_vl_dit = false;
        assert_eq!(
            dit_attention_source(&config, 0).unwrap(),
            Gr00tDitAttentionSource::FullBackbone
        );
    }

    #[test]
    fn rejects_non_boolean_masks_and_empty_groups() {
        assert!(build_backbone_token_groups(&[1, 2], &[1, 2], 2).is_err());
        assert!(build_backbone_token_groups(&[1, 2], &[1, 1], 9).is_err());
    }
}
