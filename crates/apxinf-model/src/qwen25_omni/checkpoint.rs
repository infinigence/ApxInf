//! Header-first checkpoint contract and selective Thinker loading.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use apxinf_core::{DType, Error, Result, Tensor};
use apxinf_loader::safetensors::{self, CheckpointManifest};

use super::config::Qwen25OmniConfig;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen25OmniCheckpointReport {
    pub shard_count: usize,
    pub checkpoint_tensor_count: usize,
    pub checkpoint_tensor_bytes: u64,
    pub required_tensor_count: usize,
    pub required_tensor_bytes: u64,
    pub excluded_tensor_count: usize,
    pub excluded_tensor_bytes: u64,
    pub dtype_counts: BTreeMap<DType, usize>,
}

#[derive(Clone, Debug)]
struct TensorSpec {
    name: String,
    shape: Vec<usize>,
}

impl Qwen25OmniCheckpointReport {
    pub fn inspect(model_dir: &Path, config: &Qwen25OmniConfig) -> Result<Self> {
        let manifest = safetensors::inspect_path(model_dir)
            .map_err(|error| Error::Other(format!("inspect qwen2.5-omni checkpoint: {error}")))?;
        validate_manifest(&manifest, config)
    }
}

/// Validate every tensor owned by the deployed Thinker slice before reading
/// payload bytes. Talker and token-to-wave tensors are intentionally present in
/// the upstream checkpoint but excluded from this deployment.
pub fn validate_manifest(
    manifest: &CheckpointManifest,
    config: &Qwen25OmniConfig,
) -> Result<Qwen25OmniCheckpointReport> {
    config.validate_pinned_contract()?;
    let specs = required_specs(config)?;
    let mut required_tensor_bytes = 0_u64;
    for spec in &specs {
        let entry = manifest.tensor(&spec.name).ok_or_else(|| {
            Error::Other(format!("qwen2.5-omni checkpoint missing `{}`", spec.name))
        })?;
        if entry.dtype != DType::BF16 {
            return Err(Error::Other(format!(
                "qwen2.5-omni tensor `{}` is {}, expected bf16",
                entry.name, entry.dtype
            )));
        }
        if entry.shape != spec.shape {
            return Err(Error::Other(format!(
                "qwen2.5-omni tensor `{}` shape {:?}, expected {:?}",
                entry.name, entry.shape, spec.shape
            )));
        }
        required_tensor_bytes = required_tensor_bytes
            .checked_add(entry.byte_len)
            .ok_or_else(|| {
                Error::Other("qwen2.5-omni required tensor byte total overflow".into())
            })?;
    }

    let mut excluded_tensor_count = 0;
    let mut excluded_tensor_bytes = 0_u64;
    for entry in &manifest.tensors {
        if entry.name.starts_with("talker.") || entry.name.starts_with("token2wav.") {
            excluded_tensor_count += 1;
            excluded_tensor_bytes = excluded_tensor_bytes
                .checked_add(entry.byte_len)
                .ok_or_else(|| {
                    Error::Other("qwen2.5-omni excluded tensor byte total overflow".into())
                })?;
        }
    }
    if excluded_tensor_count == 0 {
        return Err(Error::Other(
            "qwen2.5-omni checkpoint has no excluded talker/token2wav tensors; wrong deployment artifact"
                .into(),
        ));
    }

    Ok(Qwen25OmniCheckpointReport {
        shard_count: manifest.shards.len(),
        checkpoint_tensor_count: manifest.tensors.len(),
        checkpoint_tensor_bytes: manifest.tensor_bytes,
        required_tensor_count: specs.len(),
        required_tensor_bytes,
        excluded_tensor_count,
        excluded_tensor_bytes,
        dtype_counts: manifest.dtype_counts(),
    })
}

/// Load only validated Thinker text, vision and audio tensors. This function
/// never materializes `talker.*` or `token2wav.*` payloads.
pub fn load_required_tensors(
    model_dir: &Path,
    config: &Qwen25OmniConfig,
) -> Result<(HashMap<String, Tensor>, Qwen25OmniCheckpointReport)> {
    let manifest = safetensors::inspect_path(model_dir)
        .map_err(|error| Error::Other(format!("inspect qwen2.5-omni checkpoint: {error}")))?;
    let report = validate_manifest(&manifest, config)?;
    let specs = required_specs(config)?;
    let mut tensors = HashMap::with_capacity(specs.len());
    for spec in specs {
        let entry = manifest
            .tensor(&spec.name)
            .expect("manifest validated above");
        let tensor = safetensors::load_manifest_tensor(entry)
            .map_err(|error| Error::Other(format!("load `{}`: {error}", spec.name)))?;
        if tensors.insert(spec.name.clone(), tensor).is_some() {
            return Err(Error::Other(format!(
                "qwen2.5-omni duplicate selected tensor `{}`",
                spec.name
            )));
        }
    }
    Ok((tensors, report))
}

fn required_specs(config: &Qwen25OmniConfig) -> Result<Vec<TensorSpec>> {
    let text = &config.text;
    let vision = &config.vision;
    let audio = &config.audio;
    let mut specs = Vec::new();
    let mut add = |name: String, shape: &[usize]| {
        specs.push(TensorSpec {
            name,
            shape: shape.to_vec(),
        });
    };

    add(
        "thinker.model.embed_tokens.weight".into(),
        &[text.vocab_size, text.hidden_size],
    );
    add("thinker.model.norm.weight".into(), &[text.hidden_size]);
    add(
        "thinker.lm_head.weight".into(),
        &[text.vocab_size, text.hidden_size],
    );
    let kv_width = text
        .n_kv_heads
        .checked_mul(text.head_dim)
        .ok_or_else(|| Error::Other("qwen2.5-omni KV width overflow".into()))?;
    for layer in 0..text.n_layers {
        let prefix = format!("thinker.model.layers.{layer}");
        add(
            format!("{prefix}.input_layernorm.weight"),
            &[text.hidden_size],
        );
        add(
            format!("{prefix}.post_attention_layernorm.weight"),
            &[text.hidden_size],
        );
        add(
            format!("{prefix}.self_attn.q_proj.weight"),
            &[text.n_heads * text.head_dim, text.hidden_size],
        );
        add(
            format!("{prefix}.self_attn.q_proj.bias"),
            &[text.n_heads * text.head_dim],
        );
        for projection in ["k_proj", "v_proj"] {
            add(
                format!("{prefix}.self_attn.{projection}.weight"),
                &[kv_width, text.hidden_size],
            );
            add(format!("{prefix}.self_attn.{projection}.bias"), &[kv_width]);
        }
        add(
            format!("{prefix}.self_attn.o_proj.weight"),
            &[text.hidden_size, text.n_heads * text.head_dim],
        );
        for projection in ["gate_proj", "up_proj"] {
            add(
                format!("{prefix}.mlp.{projection}.weight"),
                &[text.intermediate_size, text.hidden_size],
            );
        }
        add(
            format!("{prefix}.mlp.down_proj.weight"),
            &[text.hidden_size, text.intermediate_size],
        );
    }

    let patch_width = vision
        .in_channels
        .checked_mul(vision.temporal_patch_size)
        .and_then(|value| value.checked_mul(vision.patch_size))
        .and_then(|value| value.checked_mul(vision.patch_size))
        .ok_or_else(|| Error::Other("qwen2.5-omni vision patch width overflow".into()))?;
    add(
        "thinker.visual.patch_embed.proj.weight".into(),
        &[
            vision.hidden_size,
            vision.in_channels,
            vision.temporal_patch_size,
            vision.patch_size,
            vision.patch_size,
        ],
    );
    debug_assert_eq!(patch_width, 1176);
    for layer in 0..vision.depth {
        let prefix = format!("thinker.visual.blocks.{layer}");
        add(format!("{prefix}.norm1.weight"), &[vision.hidden_size]);
        add(format!("{prefix}.norm2.weight"), &[vision.hidden_size]);
        for projection in ["q", "k", "v"] {
            add(
                format!("{prefix}.attn.{projection}.weight"),
                &[vision.hidden_size, vision.hidden_size],
            );
            add(
                format!("{prefix}.attn.{projection}.bias"),
                &[vision.hidden_size],
            );
        }
        add(
            format!("{prefix}.attn.proj.weight"),
            &[vision.hidden_size, vision.hidden_size],
        );
        add(format!("{prefix}.attn.proj.bias"), &[vision.hidden_size]);
        for projection in ["gate_proj", "up_proj"] {
            add(
                format!("{prefix}.mlp.{projection}.weight"),
                &[vision.intermediate_size, vision.hidden_size],
            );
            add(
                format!("{prefix}.mlp.{projection}.bias"),
                &[vision.intermediate_size],
            );
        }
        add(
            format!("{prefix}.mlp.down_proj.weight"),
            &[vision.hidden_size, vision.intermediate_size],
        );
        add(
            format!("{prefix}.mlp.down_proj.bias"),
            &[vision.hidden_size],
        );
    }
    let merged_width = vision
        .hidden_size
        .checked_mul(vision.spatial_merge_size * vision.spatial_merge_size)
        .ok_or_else(|| Error::Other("qwen2.5-omni vision merger width overflow".into()))?;
    add(
        "thinker.visual.merger.ln_q.weight".into(),
        &[vision.hidden_size],
    );
    add(
        "thinker.visual.merger.mlp.0.weight".into(),
        &[merged_width, merged_width],
    );
    add("thinker.visual.merger.mlp.0.bias".into(), &[merged_width]);
    add(
        "thinker.visual.merger.mlp.2.weight".into(),
        &[vision.out_hidden_size, merged_width],
    );
    add(
        "thinker.visual.merger.mlp.2.bias".into(),
        &[vision.out_hidden_size],
    );

    add(
        "thinker.audio_tower.conv1.weight".into(),
        &[audio.hidden_size, audio.num_mel_bins, 3],
    );
    add(
        "thinker.audio_tower.conv1.bias".into(),
        &[audio.hidden_size],
    );
    add(
        "thinker.audio_tower.conv2.weight".into(),
        &[audio.hidden_size, audio.hidden_size, 3],
    );
    add(
        "thinker.audio_tower.conv2.bias".into(),
        &[audio.hidden_size],
    );
    add(
        "thinker.audio_tower.audio_bos_eos_token.weight".into(),
        &[2, audio.output_dim],
    );
    for layer in 0..audio.n_layers {
        let prefix = format!("thinker.audio_tower.layers.{layer}");
        for norm in ["self_attn_layer_norm", "final_layer_norm"] {
            add(format!("{prefix}.{norm}.weight"), &[audio.hidden_size]);
            add(format!("{prefix}.{norm}.bias"), &[audio.hidden_size]);
        }
        add(
            format!("{prefix}.self_attn.q_proj.weight"),
            &[audio.hidden_size, audio.hidden_size],
        );
        add(
            format!("{prefix}.self_attn.q_proj.bias"),
            &[audio.hidden_size],
        );
        add(
            format!("{prefix}.self_attn.k_proj.weight"),
            &[audio.hidden_size, audio.hidden_size],
        );
        add(
            format!("{prefix}.self_attn.v_proj.weight"),
            &[audio.hidden_size, audio.hidden_size],
        );
        add(
            format!("{prefix}.self_attn.v_proj.bias"),
            &[audio.hidden_size],
        );
        add(
            format!("{prefix}.self_attn.out_proj.weight"),
            &[audio.hidden_size, audio.hidden_size],
        );
        add(
            format!("{prefix}.self_attn.out_proj.bias"),
            &[audio.hidden_size],
        );
        add(
            format!("{prefix}.fc1.weight"),
            &[audio.intermediate_size, audio.hidden_size],
        );
        add(format!("{prefix}.fc1.bias"), &[audio.intermediate_size]);
        add(
            format!("{prefix}.fc2.weight"),
            &[audio.hidden_size, audio.intermediate_size],
        );
        add(format!("{prefix}.fc2.bias"), &[audio.hidden_size]);
    }
    add(
        "thinker.audio_tower.ln_post.weight".into(),
        &[audio.hidden_size],
    );
    add(
        "thinker.audio_tower.ln_post.bias".into(),
        &[audio.hidden_size],
    );
    add(
        "thinker.audio_tower.proj.weight".into(),
        &[audio.output_dim, audio.hidden_size],
    );
    add("thinker.audio_tower.proj.bias".into(), &[audio.output_dim]);

    Ok(specs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_loader::safetensors::TensorManifestEntry;
    use std::path::PathBuf;

    fn config() -> Qwen25OmniConfig {
        let raw = include_str!("../../tests/data/qwen25_omni_config_minimal.json");
        let mut config = Qwen25OmniConfig::from_json_str(raw).unwrap();
        config.processor = super::super::config::Qwen25OmniProcessorConfig {
            sampling_rate: 16000,
            n_fft: 400,
            hop_length: 160,
            feature_size: 128,
        };
        config
    }

    #[test]
    fn required_contract_contains_only_thinker_tensors() {
        let specs = required_specs(&config()).unwrap();
        assert!(specs.iter().all(|spec| spec.name.starts_with("thinker.")));
        assert!(specs
            .iter()
            .all(|spec| !spec.name.starts_with("thinker.talker")));
        assert!(specs
            .iter()
            .any(|spec| spec.name == "thinker.audio_tower.layers.31.fc2.weight"));
        assert!(specs
            .iter()
            .any(|spec| spec.name == "thinker.visual.blocks.31.attn.q.weight"));
        assert!(specs
            .iter()
            .any(|spec| spec.name == "thinker.model.layers.35.self_attn.q_proj.bias"));
    }

    #[test]
    fn rejects_shape_and_dtype_before_payload_loading() {
        let config = config();
        let specs = required_specs(&config).unwrap();
        let mut entries = specs
            .iter()
            .map(|spec| TensorManifestEntry {
                name: spec.name.clone(),
                dtype: DType::BF16,
                shape: spec.shape.clone(),
                data_offsets: [0, 2],
                file_offset: 0,
                byte_len: (spec.shape.iter().product::<usize>() * 2) as u64,
                shard: "model.safetensors".into(),
                file: PathBuf::from("model.safetensors"),
            })
            .collect::<Vec<_>>();
        entries.push(TensorManifestEntry {
            name: "talker.codec_head.weight".into(),
            dtype: DType::BF16,
            shape: vec![1],
            data_offsets: [0, 2],
            file_offset: 0,
            byte_len: 2,
            shard: "model.safetensors".into(),
            file: PathBuf::from("model.safetensors"),
        });
        entries.sort_by(|left, right| left.name.cmp(&right.name));
        let manifest = CheckpointManifest {
            tensor_bytes: entries.iter().map(|entry| entry.byte_len).sum(),
            tensors: entries,
            shards: vec!["model.safetensors".into()],
            indexed_total_size: None,
            metadata: HashMap::new(),
        };
        validate_manifest(&manifest, &config).unwrap();

        let mut wrong_shape = manifest.clone();
        wrong_shape
            .tensors
            .iter_mut()
            .find(|entry| entry.name == "thinker.model.norm.weight")
            .unwrap()
            .shape = vec![1];
        assert!(validate_manifest(&wrong_shape, &config)
            .unwrap_err()
            .to_string()
            .contains("shape"));

        let mut wrong_dtype = manifest;
        wrong_dtype
            .tensors
            .iter_mut()
            .find(|entry| entry.name == "thinker.lm_head.weight")
            .unwrap()
            .dtype = DType::F32;
        assert!(validate_manifest(&wrong_dtype, &config)
            .unwrap_err()
            .to_string()
            .contains("expected bf16"));
    }
}
