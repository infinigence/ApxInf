//! GR00T N1.7 checkpoint assembly (HIP port).
//!
//! Copied from `gr00t/` rather than shared, per the port rule. The three
//! submodules keep their own helpers verbatim: `action_head`'s `transpose_2d`
//! is BF16-only while `text`'s also accepts F32, and their error messages name
//! different parts of the model. Folding them into one helper would be a
//! behaviour change dressed up as cleanup.
//!
//! NVIDIA stores the truncated Qwen3-VL backbone and the action head in one
//! sharded checkpoint. The Qwen architecture config remains in the separate
//! Cosmos-Reason2-2B directory. This module joins both sources without cloning
//! the multi-gigabyte tensor map.

use std::collections::HashMap;
use std::path::Path;

use apxinf_core::{DType, Error, Result, Tensor};

use self::action_head::Gr00tActionHeadWeights;
use self::text::Qwen3VLTextWeights;
use self::vision::Qwen3VLVisionWeights;
use crate::hip_gr00t::config::{Gr00tConfig, Qwen3VLConfig};

pub mod action_head;
pub mod text;
pub mod vision;

const BACKBONE_PREFIX: &str = "backbone.model.";

/// Complete host-side weights required by the GR00T N1.7 inference graph.
pub struct Gr00tWeights {
    pub backbone_text: Qwen3VLTextWeights,
    pub backbone_vision: Qwen3VLVisionWeights,
    pub action_head: Gr00tActionHeadWeights,
}

impl Gr00tWeights {
    /// Load the sharded GR00T checkpoint once and split it into typed backbone
    /// and action-head weights.
    pub fn from_safetensors(
        config: &Gr00tConfig,
        backbone_config_path: &Path,
        checkpoint_path: &Path,
    ) -> Result<(Qwen3VLConfig, Self)> {
        let backbone_config = load_backbone_config(config, backbone_config_path)?;
        let (tensors, _) = apxinf_loader::safetensors::load_native_path(checkpoint_path)
            .map_err(|error| Error::Other(format!("load GR00T SafeTensors: {error}")))?;
        let weights = Self::from_map(config, &backbone_config, tensors)?;
        Ok((backbone_config, weights))
    }

    /// Consume a complete NVIDIA GR00T N1.7 tensor map.
    pub fn from_map(
        config: &Gr00tConfig,
        backbone_config: &Qwen3VLConfig,
        mut tensors: HashMap<String, Tensor>,
    ) -> Result<Self> {
        config.validate()?;
        validate_backbone_config(config, backbone_config)?;
        validate_activations(config, backbone_config)?;
        let mut selected_backbone_config = backbone_config.clone();
        selected_backbone_config.text.n_layers = config.select_layer;

        let action_head = Gr00tActionHeadWeights::from_map(config, &mut tensors)?;
        normalize_backbone_prefix(&mut tensors)?;
        discard_unused_lm_head(&selected_backbone_config, &mut tensors)?;
        let backbone_text =
            Qwen3VLTextWeights::take_from_map(&selected_backbone_config, &mut tensors)
                .map_err(|error| Error::Other(format!("load GR00T text backbone: {error}")))?;
        let backbone_vision =
            Qwen3VLVisionWeights::take_from_map(&selected_backbone_config, &mut tensors)
                .map_err(|error| Error::Other(format!("load GR00T vision backbone: {error}")))?;
        validate_backbone_weights(&selected_backbone_config, &backbone_text, &backbone_vision)?;

        let mut unexpected = tensors.keys().cloned().collect::<Vec<_>>();
        unexpected.sort();
        if !unexpected.is_empty() {
            return Err(Error::Other(format!(
                "unconsumed GR00T checkpoint tensors: {}",
                unexpected.join(", ")
            )));
        }

        Ok(Self {
            backbone_text,
            backbone_vision,
            action_head,
        })
    }
}

/// GR00T consumes a selected hidden state and never computes token logits,
/// so the language-model output head is dead weight. When the checkpoint
/// carries one (the released N1.7 checkpoints do), validate it before
/// discarding, so strict checkpoint loading still detects mismatched files;
/// a tied-embedding checkpoint without `lm_head.weight` is equally valid --
/// the field is optional the way `qwen3vl` treats it, not a completeness
/// marker.
fn discard_unused_lm_head(
    config: &Qwen3VLConfig,
    tensors: &mut HashMap<String, Tensor>,
) -> Result<()> {
    let name = "lm_head.weight";
    match tensors.remove(name) {
        Some(tensor) => {
            expect_bf16_shape(
                "unused backbone language-model head",
                &tensor,
                &[config.text.vocab_size, config.text.hidden_size],
            )
        }
        None => Ok(()),
    }
}

/// Parse and validate Cosmos-Reason2-2B's Qwen3-VL config, then truncate the
/// language stack exactly as NVIDIA's `Qwen3Backbone` does.
pub fn load_backbone_config(
    config: &Gr00tConfig,
    backbone_config_path: &Path,
) -> Result<Qwen3VLConfig> {
    let path = if backbone_config_path.is_dir() {
        backbone_config_path.join("config.json")
    } else {
        backbone_config_path.to_path_buf()
    };
    let mut backbone = Qwen3VLConfig::from_json_file(&path)?;
    validate_backbone_config(config, &backbone)?;
    backbone.text.n_layers = config.select_layer;
    Ok(backbone)
}

/// Reject a checkpoint whose configured activations are not the ones this port
/// implements.
///
/// GR00T N1.7 asks for two different GELU variants, and the port hardcodes one
/// per position rather than dispatching at runtime:
///
/// | position | source | variant |
/// |---|---|---|
/// | Qwen3-VL vision MLP | `vision_config.hidden_act` | tanh |
/// | Qwen3-VL patch merger | `transformers` hardcodes `nn.GELU()` | exact erf |
/// | DiT feed-forward | `diffusion_model_cfg.activation_fn` | tanh |
/// | VL self-attention feed-forward | `vl_self_attention_cfg.activation_fn` | tanh |
///
/// Three of those four are config-driven, so a different checkpoint could ask
/// for the other variant. Computing the hardcoded one anyway would be silently
/// wrong by about 3e-4 per element -- small enough to pass an eyeball and an
/// end-to-end cosine gate, which is exactly why this fails loudly instead.
/// (`diffusion.activation_fn` is already checked in [`Gr00tConfig::validate`];
/// this covers the two that were not.)
fn validate_activations(config: &Gr00tConfig, backbone: &Qwen3VLConfig) -> Result<()> {
    const VISION_MLP: &str = "gelu_pytorch_tanh";
    const FEED_FORWARD: &str = "gelu-approximate";

    if backbone.vision.hidden_act != VISION_MLP {
        return Err(Error::Other(format!(
            "GR00T on HIP implements the Qwen3-VL vision MLP as {VISION_MLP}, but the backbone \
             config asks for {}",
            backbone.vision.hidden_act
        )));
    }
    if let Some(vl) = config.vl_self_attention.as_ref().filter(|_| config.use_vl_self_attention) {
        if vl.activation_fn != FEED_FORWARD {
            return Err(Error::Other(format!(
                "GR00T on HIP implements the VL self-attention feed-forward as {FEED_FORWARD}, \
                 but vl_self_attention_cfg asks for {}",
                vl.activation_fn
            )));
        }
    }
    Ok(())
}

fn validate_backbone_config(config: &Gr00tConfig, backbone: &Qwen3VLConfig) -> Result<()> {
    if backbone.text.n_layers < config.select_layer {
        return Err(Error::Other(format!(
            "GR00T selects {} Qwen layers but backbone config contains {}",
            config.select_layer, backbone.text.n_layers
        )));
    }
    if backbone.text.hidden_size != config.backbone_embedding_dim {
        return Err(Error::Other(format!(
            "GR00T backbone width {} does not match Qwen text width {}",
            config.backbone_embedding_dim, backbone.text.hidden_size
        )));
    }
    if backbone.vision.out_hidden_size != config.backbone_embedding_dim {
        return Err(Error::Other(format!(
            "GR00T backbone width {} does not match Qwen vision output width {}",
            config.backbone_embedding_dim, backbone.vision.out_hidden_size
        )));
    }
    if backbone.text.n_heads == 0
        || backbone.text.n_kv_heads == 0
        || backbone.text.head_dim == 0
        || backbone.vision.depth == 0
        || backbone.vision.num_heads == 0
        || backbone.vision.spatial_merge_size == 0
    {
        return Err(Error::Other(
            "GR00T Qwen3-VL backbone dimensions must be non-zero".into(),
        ));
    }
    if !backbone.text.mrope_interleaved {
        return Err(Error::Other(
            "GR00T N1.7 requires the interleaved Qwen3-VL mRoPE layout".into(),
        ));
    }
    Ok(())
}

fn normalize_backbone_prefix(tensors: &mut HashMap<String, Tensor>) -> Result<()> {
    let names = tensors
        .keys()
        .filter(|name| name.starts_with(BACKBONE_PREFIX))
        .cloned()
        .collect::<Vec<_>>();
    if names.is_empty() {
        return Err(Error::Other(format!(
            "GR00T checkpoint has no {BACKBONE_PREFIX} tensors"
        )));
    }
    for source_name in names {
        let destination_name = source_name
            .strip_prefix(BACKBONE_PREFIX)
            .ok_or_else(|| {
                Error::Other(format!(
                    "GR00T backbone key {source_name} lost its expected {BACKBONE_PREFIX} prefix"
                ))
            })?
            .to_owned();
        let tensor = tensors.remove(&source_name).ok_or_else(|| {
            Error::Other(format!(
                "GR00T checkpoint key {source_name} disappeared during prefix normalization"
            ))
        })?;
        if tensors.insert(destination_name.clone(), tensor).is_some() {
            return Err(Error::Other(format!(
                "GR00T checkpoint key collision after removing {BACKBONE_PREFIX}: {destination_name}"
            )));
        }
    }
    Ok(())
}

fn validate_backbone_weights(
    config: &Qwen3VLConfig,
    text: &Qwen3VLTextWeights,
    vision: &Qwen3VLVisionWeights,
) -> Result<()> {
    let tc = &config.text;
    expect_bf16_shape(
        "backbone token embedding",
        &text.token_embedding,
        &[tc.vocab_size, tc.hidden_size],
    )?;
    expect_bf16_shape(
        "backbone output norm",
        &text.output_norm_weight,
        &[tc.hidden_size],
    )?;
    if text.layers.len() != tc.n_layers {
        return Err(Error::Other(format!(
            "GR00T text layer count: expected {}, got {}",
            tc.n_layers,
            text.layers.len()
        )));
    }
    let query_width = checked_mul(tc.n_heads, tc.head_dim, "Qwen query width")?;
    let key_value_width = checked_mul(tc.n_kv_heads, tc.head_dim, "Qwen key/value width")?;
    for (index, layer) in text.layers.iter().enumerate() {
        let prefix = format!("backbone text layer {index}");
        expect_bf16_shape(
            &format!("{prefix} input norm"),
            &layer.attn_norm_weight,
            &[tc.hidden_size],
        )?;
        expect_bf16_shape(
            &format!("{prefix} query"),
            &layer.wq,
            &[tc.hidden_size, query_width],
        )?;
        expect_bf16_shape(
            &format!("{prefix} key"),
            &layer.wk,
            &[tc.hidden_size, key_value_width],
        )?;
        expect_bf16_shape(
            &format!("{prefix} value"),
            &layer.wv,
            &[tc.hidden_size, key_value_width],
        )?;
        expect_bf16_shape(
            &format!("{prefix} output"),
            &layer.wo,
            &[query_width, tc.hidden_size],
        )?;
        expect_bf16_shape(
            &format!("{prefix} query norm"),
            &layer.q_norm_weight,
            &[tc.head_dim],
        )?;
        expect_bf16_shape(
            &format!("{prefix} key norm"),
            &layer.k_norm_weight,
            &[tc.head_dim],
        )?;
        expect_bf16_shape(
            &format!("{prefix} FFN norm"),
            &layer.ffn_norm_weight,
            &[tc.hidden_size],
        )?;
        expect_bf16_shape(
            &format!("{prefix} gate"),
            &layer.w_gate,
            &[tc.hidden_size, tc.intermediate_size],
        )?;
        expect_bf16_shape(
            &format!("{prefix} up"),
            &layer.w_up,
            &[tc.hidden_size, tc.intermediate_size],
        )?;
        expect_bf16_shape(
            &format!("{prefix} down"),
            &layer.w_down,
            &[tc.intermediate_size, tc.hidden_size],
        )?;
    }

    let vc = &config.vision;
    let patch_width = vc
        .in_channels
        .checked_mul(vc.temporal_patch_size)
        .and_then(|value| value.checked_mul(vc.patch_size))
        .and_then(|value| value.checked_mul(vc.patch_size))
        .ok_or_else(|| Error::Other("Qwen3-VL patch width overflow".into()))?;
    expect_bf16_shape(
        "backbone vision patch projection",
        &vision.patch_embed_weight,
        &[patch_width, vc.hidden_size],
    )?;
    expect_bf16_shape(
        "backbone vision patch bias",
        &vision.patch_embed_bias,
        &[vc.hidden_size],
    )?;
    expect_bf16_shape(
        "backbone vision position embedding",
        &vision.pos_embed,
        &[vc.num_position_embeddings, vc.hidden_size],
    )?;
    if vision.blocks.len() != vc.depth {
        return Err(Error::Other(format!(
            "GR00T vision block count: expected {}, got {}",
            vc.depth,
            vision.blocks.len()
        )));
    }
    let qkv_width = checked_mul(3, vc.hidden_size, "vision QKV width")?;
    for (index, block) in vision.blocks.iter().enumerate() {
        let prefix = format!("backbone vision block {index}");
        expect_bf16_shape(
            &format!("{prefix} norm1 weight"),
            &block.norm1_w,
            &[vc.hidden_size],
        )?;
        expect_bf16_shape(
            &format!("{prefix} norm1 bias"),
            &block.norm1_b,
            &[vc.hidden_size],
        )?;
        expect_bf16_shape(
            &format!("{prefix} QKV weight"),
            &block.qkv_w,
            &[vc.hidden_size, qkv_width],
        )?;
        expect_bf16_shape(&format!("{prefix} QKV bias"), &block.qkv_b, &[qkv_width])?;
        expect_bf16_shape(
            &format!("{prefix} output weight"),
            &block.proj_w,
            &[vc.hidden_size, vc.hidden_size],
        )?;
        expect_bf16_shape(
            &format!("{prefix} output bias"),
            &block.proj_b,
            &[vc.hidden_size],
        )?;
        expect_bf16_shape(
            &format!("{prefix} norm2 weight"),
            &block.norm2_w,
            &[vc.hidden_size],
        )?;
        expect_bf16_shape(
            &format!("{prefix} norm2 bias"),
            &block.norm2_b,
            &[vc.hidden_size],
        )?;
        expect_bf16_shape(
            &format!("{prefix} FFN input"),
            &block.fc1_w,
            &[vc.hidden_size, vc.intermediate_size],
        )?;
        expect_bf16_shape(
            &format!("{prefix} FFN input bias"),
            &block.fc1_b,
            &[vc.intermediate_size],
        )?;
        expect_bf16_shape(
            &format!("{prefix} FFN output"),
            &block.fc2_w,
            &[vc.intermediate_size, vc.hidden_size],
        )?;
        expect_bf16_shape(
            &format!("{prefix} FFN output bias"),
            &block.fc2_b,
            &[vc.hidden_size],
        )?;
    }

    let merge_cells = checked_mul(
        vc.spatial_merge_size,
        vc.spatial_merge_size,
        "vision merge cells",
    )?;
    let merged_width = checked_mul(vc.hidden_size, merge_cells, "vision merger width")?;
    validate_merger(
        "backbone vision merger",
        &vision.merger,
        vc.hidden_size,
        merged_width,
        vc.out_hidden_size,
    )?;
    if vision.deepstack_mergers.len() != vc.deepstack_visual_indexes.len() {
        return Err(Error::Other(format!(
            "GR00T deepstack merger count: expected {}, got {}",
            vc.deepstack_visual_indexes.len(),
            vision.deepstack_mergers.len()
        )));
    }
    for (index, merger) in vision.deepstack_mergers.iter().enumerate() {
        validate_merger(
            &format!("backbone deepstack merger {index}"),
            merger,
            merged_width,
            merged_width,
            vc.out_hidden_size,
        )?;
    }
    Ok(())
}

fn validate_merger(
    name: &str,
    merger: &self::vision::Qwen3VLMerger,
    norm_width: usize,
    hidden_width: usize,
    output_width: usize,
) -> Result<()> {
    expect_bf16_shape(
        &format!("{name} norm weight"),
        &merger.norm_w,
        &[norm_width],
    )?;
    expect_bf16_shape(&format!("{name} norm bias"), &merger.norm_b, &[norm_width])?;
    expect_bf16_shape(
        &format!("{name} input projection"),
        &merger.fc1_w,
        &[hidden_width, hidden_width],
    )?;
    expect_bf16_shape(
        &format!("{name} input bias"),
        &merger.fc1_b,
        &[hidden_width],
    )?;
    expect_bf16_shape(
        &format!("{name} output projection"),
        &merger.fc2_w,
        &[hidden_width, output_width],
    )?;
    expect_bf16_shape(
        &format!("{name} output bias"),
        &merger.fc2_b,
        &[output_width],
    )?;
    Ok(())
}

fn expect_bf16_shape(name: &str, tensor: &Tensor, expected: &[usize]) -> Result<()> {
    if tensor.dtype() != DType::BF16 || tensor.shape().dims() != expected {
        return Err(Error::Other(format!(
            "GR00T {name}: expected BF16 {expected:?}, got {} {:?}",
            tensor.dtype(),
            tensor.shape().dims()
        )));
    }
    Ok(())
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| Error::Other(format!("GR00T {name} overflow")))
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;
    use crate::hip_gr00t::config::Gr00tVlSelfAttentionConfig;

    #[test]
    fn removes_only_the_gr00t_backbone_wrapper_prefix() {
        let mut tensors = HashMap::from([
            (
                "backbone.model.model.language_model.norm.weight".into(),
                Tensor::from_bf16(vec![1], &[bf16::ONE]).unwrap(),
            ),
            (
                "action_head.marker".into(),
                Tensor::from_bf16(vec![1], &[bf16::ZERO]).unwrap(),
            ),
        ]);
        normalize_backbone_prefix(&mut tensors).unwrap();
        assert!(tensors.contains_key("model.language_model.norm.weight"));
        assert!(tensors.contains_key("action_head.marker"));
        assert!(!tensors.contains_key("backbone.model.model.language_model.norm.weight"));
    }

    #[test]
    fn rejects_checkpoint_without_nvidia_backbone_prefix() {
        let mut tensors = HashMap::new();
        assert!(normalize_backbone_prefix(&mut tensors).is_err());
    }

    #[test]
    fn validates_and_consumes_the_unused_language_model_head() {
        let config = Qwen3VLConfig::from_json_str(
            r#"{
                "text_config": {
                    "hidden_size": 2,
                    "vocab_size": 3,
                    "rope_scaling": {"mrope_section": [1, 1, 1]}
                }
            }"#,
        )
        .unwrap();
        let mut tensors = HashMap::from([(
            "lm_head.weight".into(),
            Tensor::from_bf16(vec![3, 2], &[bf16::ZERO; 6]).unwrap(),
        )]);
        discard_unused_lm_head(&config, &mut tensors).unwrap();
        assert!(tensors.is_empty());
    }

    #[test]
    fn a_tied_checkpoint_without_a_language_model_head_is_accepted() {
        let config = Qwen3VLConfig::from_json_str(
            r#"{
                "text_config": {
                    "hidden_size": 2,
                    "vocab_size": 3,
                    "rope_scaling": {"mrope_section": [1, 1, 1]}
                }
            }"#,
        )
        .unwrap();
        let mut tensors = HashMap::new();
        discard_unused_lm_head(&config, &mut tensors).unwrap();
        assert!(tensors.is_empty());
    }

    fn backbone_with_vision_act(hidden_act: &str) -> Qwen3VLConfig {
        Qwen3VLConfig::from_json_str(&format!(
            r#"{{
                "text_config": {{
                    "hidden_size": 2048,
                    "vocab_size": 3,
                    "rope_scaling": {{"mrope_section": [1, 1, 1]}}
                }},
                "vision_config": {{"hidden_act": "{hidden_act}"}}
            }}"#
        ))
        .unwrap()
    }

    #[test]
    fn the_released_checkpoints_activations_are_the_ones_the_port_implements() {
        let config = Gr00tConfig::default();
        validate_activations(&config, &backbone_with_vision_act("gelu_pytorch_tanh")).unwrap();
        // The default is also what a config omitting the field gets, so the
        // guard must not fire on the common case.
        let omitted = Qwen3VLConfig::from_json_str(
            r#"{"text_config": {"hidden_size": 2048, "vocab_size": 3,
                 "rope_scaling": {"mrope_section": [1, 1, 1]}}}"#,
        )
        .unwrap();
        validate_activations(&config, &omitted).unwrap();
    }

    #[test]
    fn a_checkpoint_asking_for_a_different_activation_is_refused() {
        // Both of these differ from what the port computes by about 3e-4 per
        // element -- small enough to pass an end-to-end cosine gate, so the
        // load has to fail rather than the inference quietly drift.
        let config = Gr00tConfig::default();
        let error = validate_activations(&config, &backbone_with_vision_act("gelu"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("vision MLP"), "{error}");
        assert!(error.contains("asks for gelu"), "{error}");

        let mut config = Gr00tConfig::default();
        config.use_vl_self_attention = true;
        config.vl_self_attention = Some(Gr00tVlSelfAttentionConfig {
            activation_fn: "gelu".into(),
            ..Gr00tVlSelfAttentionConfig::default()
        });
        let error = validate_activations(&config, &backbone_with_vision_act("gelu_pytorch_tanh"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("VL self-attention"), "{error}");

        // Disabled VL self-attention must not be checked: its config is then
        // never read, and failing on it would reject a valid checkpoint.
        config.use_vl_self_attention = false;
        validate_activations(&config, &backbone_with_vision_act("gelu_pytorch_tanh")).unwrap();
    }

    #[test]
    fn the_activation_guard_is_wired_into_the_load_path() {
        // Testing `validate_activations` directly proves the function works;
        // this proves it is actually reached. The checks run before any tensor
        // is consumed, so an empty map is enough to see which error comes out
        // -- and a guard that were removed from `from_map` would surface here
        // as a missing-tensor error instead.
        let config = Gr00tConfig::default();
        let error = match Gr00tWeights::from_map(
            &config,
            &backbone_with_vision_act("gelu"),
            HashMap::new(),
        ) {
            Ok(_) => panic!("from_map accepted a config the port cannot serve"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("vision MLP"), "{error}");
        assert!(error.contains("asks for gelu"), "{error}");
    }
}
