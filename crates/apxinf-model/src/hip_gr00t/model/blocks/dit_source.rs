//! The AlternateVLDiT attention pattern: which encoder rows each DiT block
//! attends to. Config-derived, load-time knowledge.

use apxinf_core::{Error, Result};

use crate::hip_gr00t::Gr00tConfig;

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
    fn non_alternating_dit_uses_the_complete_backbone() {
        let mut config = Gr00tConfig::default();
        config.use_alternate_vl_dit = false;
        assert_eq!(
            dit_attention_source(&config, 0).unwrap(),
            Gr00tDitAttentionSource::FullBackbone
        );
    }
}
