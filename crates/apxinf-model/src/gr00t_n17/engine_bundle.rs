use std::path::{Path, PathBuf};

use apxinf_core::{Error, Result};
use serde::Deserialize;

use super::Gr00tN17Config;

const REQUIRED_ENGINES: &[&str] = &[
    "vit.engine",
    "llm_bf16.engine",
    "vl_self_attention.engine",
    "state_encoder.engine",
    "action_encoder.engine",
    "dit_bf16.engine",
    "action_decoder.engine",
];

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct EngineMetadata {
    pub schema_version: u32,
    pub model_version: String,
    pub sa_seq_len: usize,
    pub vl_seq_len: usize,
    pub llm_seq_len: usize,
    pub llm_hidden_size: usize,
    pub num_deepstack: usize,
    pub num_vis_tokens: usize,
    pub num_patches: usize,
    pub num_merged_patches: usize,
    pub action_horizon: usize,
    pub max_action_dim: usize,
    pub max_state_dim: usize,
    pub state_history_length: usize,
    pub hidden_size: usize,
    pub input_embedding_dim: usize,
    pub backbone_embedding_dim: usize,
    pub precision: String,
    pub batch_size: usize,
    pub vit_grid_thw: Vec<[usize; 3]>,
}

#[derive(Clone, Debug)]
pub struct EngineBundle {
    root: PathBuf,
    pub metadata: EngineMetadata,
}

impl EngineBundle {
    pub fn load(checkpoint: &Path, config: &Gr00tN17Config) -> Result<Self> {
        let root = checkpoint.join("apxinf-engines");
        let metadata_path = root.join("export_metadata.json");
        let raw = std::fs::read_to_string(&metadata_path).map_err(|error| {
            Error::Other(format!(
                "read GR00T TensorRT metadata {}: {error}; install the target-built engine bundle under apxinf-engines/",
                metadata_path.display()
            ))
        })?;
        let metadata: EngineMetadata = serde_json::from_str(&raw).map_err(|error| {
            Error::Other(format!(
                "parse GR00T TensorRT metadata {}: {error}",
                metadata_path.display()
            ))
        })?;
        Self::validate_metadata(&metadata, config)?;
        for name in REQUIRED_ENGINES {
            let path = root.join(name);
            if !path.is_file() {
                return Err(Error::Other(format!(
                    "GR00T TensorRT engine is missing: {}",
                    path.display()
                )));
            }
        }
        Ok(Self { root, metadata })
    }

    fn validate_metadata(metadata: &EngineMetadata, config: &Gr00tN17Config) -> Result<()> {
        if metadata.schema_version != 1
            || metadata.model_version != "n1d7"
            || metadata.precision != "bf16"
            || metadata.batch_size != 1
        {
            return Err(Error::Other(format!(
                "unsupported GR00T engine bundle: schema={}, model={}, precision={}, batch={}",
                metadata.schema_version,
                metadata.model_version,
                metadata.precision,
                metadata.batch_size
            )));
        }
        let expected = (
            config.action_horizon,
            config.max_action_dim,
            config.max_state_dim,
            config.state_history_length,
            config.hidden_size,
            config.input_embedding_dim,
            config.backbone_embedding_dim,
        );
        let actual = (
            metadata.action_horizon,
            metadata.max_action_dim,
            metadata.max_state_dim,
            metadata.state_history_length,
            metadata.hidden_size,
            metadata.input_embedding_dim,
            metadata.backbone_embedding_dim,
        );
        if actual != expected {
            return Err(Error::Other(format!(
                "GR00T engine/checkpoint shape mismatch: engine={actual:?}, checkpoint={expected:?}"
            )));
        }
        if metadata.sa_seq_len != config.action_horizon + 1
            || metadata.llm_seq_len != metadata.vl_seq_len
            || metadata.llm_hidden_size != config.backbone_embedding_dim
            || metadata.num_patches != metadata.num_merged_patches * 4
        {
            return Err(Error::Other(
                "GR00T engine metadata has an inconsistent sequence contract".into(),
            ));
        }
        Ok(())
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }
}
