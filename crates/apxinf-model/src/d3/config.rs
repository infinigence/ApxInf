//! Official d3 package and Qwen3.5-4B geometry.

use std::collections::HashSet;
use std::path::Path;

use apxinf_core::{Error, Result};
use serde::Deserialize;

const D3_MINI_MODEL: &str = "d3-mini";
const D3_MINI_REPOSITORY: &str = "vllm-sr/d3-mini";
const D3_FORMAT: &str = "d3-code-readout-v1";
const D3_PROMPT: &str = "d3";
const D3_ATTENTION: &str = "noncausal_full_attention";
const D3_POOLING: &str = "last";
const D3_READOUT_DTYPE: &str = "float32";
const D3_BACKBONE_DTYPE: &str = "bfloat16";

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct D3Config {
    pub architectures: Vec<String>,
    pub dtype: String,
    pub image_token_id: u32,
    pub model_type: String,
    pub text_config: D3TextConfig,
    pub tie_word_embeddings: bool,
    pub video_token_id: u32,
    pub vision_config: D3VisionConfig,
    pub vision_end_token_id: u32,
    pub vision_start_token_id: u32,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct D3TextConfig {
    pub attention_bias: bool,
    pub attn_output_gate: bool,
    pub dtype: String,
    pub eos_token_id: u32,
    pub full_attention_interval: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub layer_types: Vec<String>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub partial_rotary_factor: f32,
    pub rms_norm_eps: f32,
    pub rope_parameters: D3TextRopeConfig,
    pub tie_word_embeddings: bool,
    pub vocab_size: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct D3TextRopeConfig {
    pub mrope_interleaved: bool,
    pub mrope_section: [usize; 3],
    pub partial_rotary_factor: f32,
    pub rope_theta: f32,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct D3VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_position_embeddings: usize,
    pub out_hidden_size: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct D3DecisionConfig {
    pub format_version: usize,
    pub format_id: String,
    pub prompt: String,
    pub base_model: String,
    pub revision: String,
    pub codes: Vec<String>,
    pub token_ids: Vec<u32>,
    pub temperature: f32,
    pub attention_mode: String,
    pub pooling: String,
    pub max_length: Option<usize>,
    pub readout_dtype: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct D3ManifestIdentity {
    pub model_sha256: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct D3ManifestParameters {
    pub text: u64,
    pub vision: u64,
    pub readout: u64,
    pub loaded: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct D3Manifest {
    pub schema: String,
    pub model_name: String,
    pub repo_id: String,
    pub format_id: String,
    pub prompt: String,
    pub attention_mode: String,
    pub identity: D3ManifestIdentity,
    pub parameters: D3ManifestParameters,
}

/// All package metadata required before device allocation begins.
#[derive(Clone, Debug, PartialEq)]
pub struct D3PackageConfig {
    pub model: D3Config,
    pub decision: D3DecisionConfig,
    pub manifest: D3Manifest,
}

impl D3PackageConfig {
    pub fn from_dir(model_dir: &Path) -> Result<Self> {
        let package = Self {
            model: read_json(&model_dir.join("config.json"))?,
            decision: read_json(&model_dir.join("decision_config.json"))?,
            manifest: read_json(&model_dir.join("MODEL_MANIFEST.json"))?,
        };
        package.validate()?;
        Ok(package)
    }

    pub fn validate(&self) -> Result<()> {
        self.model.validate()?;
        self.decision.validate()?;
        self.manifest.validate()?;
        if self.manifest.format_id != self.decision.format_id
            || self.manifest.prompt != self.decision.prompt
            || self.manifest.attention_mode != self.decision.attention_mode
        {
            return Err(invalid(
                "manifest and decision_config disagree on the d3 execution contract",
            ));
        }
        if self.manifest.parameters.readout
            != (self.decision.codes.len() * self.model.text_config.hidden_size) as u64
        {
            return Err(invalid(
                "manifest readout parameter count does not match codes x hidden_size",
            ));
        }
        Ok(())
    }
}

impl D3Config {
    pub fn validate(&self) -> Result<()> {
        if self.architectures.as_slice() != ["Qwen3_5Model"]
            || self.dtype != D3_BACKBONE_DTYPE
            || self.model_type != "qwen3_5"
            || !self.tie_word_embeddings
        {
            return Err(invalid("unsupported d3 backbone identity or dtype"));
        }
        self.text_config.validate()?;
        self.vision_config.validate()?;
        let mut special = HashSet::new();
        for token in [
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        ] {
            if token >= self.text_config.vocab_size as u32 || !special.insert(token) {
                return Err(invalid("invalid or duplicate d3 vision special token"));
            }
        }
        Ok(())
    }
}

impl D3TextConfig {
    pub fn validate(&self) -> Result<()> {
        let exact_geometry = self.dtype == D3_BACKBONE_DTYPE
            && !self.attention_bias
            && self.attn_output_gate
            && self.hidden_size == 2560
            && self.intermediate_size == 9216
            && self.num_hidden_layers == 32
            && self.num_attention_heads == 16
            && self.num_key_value_heads == 4
            && self.head_dim == 256
            && self.full_attention_interval == 4
            && self.linear_num_key_heads == 16
            && self.linear_num_value_heads == 32
            && self.linear_key_head_dim == 128
            && self.linear_value_head_dim == 128
            && self.linear_conv_kernel_dim == 4
            && self.vocab_size == 248_320
            && self.max_position_embeddings == 262_144
            && self.tie_word_embeddings
            && self.eos_token_id < self.vocab_size as u32;
        if !exact_geometry {
            return Err(invalid("unsupported d3-mini text geometry"));
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(invalid(
                "d3 layer_types length does not match num_hidden_layers",
            ));
        }
        for (index, kind) in self.layer_types.iter().enumerate() {
            let expected = if (index + 1) % self.full_attention_interval == 0 {
                "full_attention"
            } else {
                "linear_attention"
            };
            if kind != expected {
                return Err(invalid(
                    "d3 layer_types do not follow the 3 GDN + 1 full-attention schedule",
                ));
            }
        }
        if !self.partial_rotary_factor.is_finite()
            || self.partial_rotary_factor != 0.25
            || self.rms_norm_eps != 1e-6
            || self.rope_parameters.partial_rotary_factor != self.partial_rotary_factor
            || !self.rope_parameters.mrope_interleaved
            || self.rope_parameters.mrope_section != [11, 11, 10]
            || self.rope_parameters.rope_theta != 10_000_000.0
        {
            return Err(invalid(
                "unsupported d3-mini normalization or mRoPE contract",
            ));
        }
        Ok(())
    }
}

impl D3VisionConfig {
    pub fn validate(&self) -> Result<()> {
        if self.depth != 24
            || self.hidden_size != 1024
            || self.intermediate_size != 4096
            || self.num_heads != 16
            || self.num_position_embeddings != 2304
            || self.out_hidden_size != 2560
            || self.patch_size != 16
            || self.temporal_patch_size != 2
            || self.spatial_merge_size != 2
        {
            return Err(invalid("unsupported d3-mini vision geometry"));
        }
        Ok(())
    }
}

impl D3DecisionConfig {
    pub fn validate(&self) -> Result<()> {
        if self.format_version != 1
            || self.format_id != D3_FORMAT
            || self.prompt != D3_PROMPT
            || self.attention_mode != D3_ATTENTION
            || self.pooling != D3_POOLING
            || self.readout_dtype != D3_READOUT_DTYPE
            || self.max_length.is_some()
            || !self.temperature.is_finite()
            || self.temperature <= 0.0
        {
            return Err(invalid("unsupported d3 decision contract"));
        }
        if self.codes.is_empty()
            || self.codes.len() > 255
            || self.codes.len() != self.token_ids.len()
        {
            return Err(invalid(
                "d3 decision codes must have matching 1..=255 token IDs",
            ));
        }
        if self.codes.iter().collect::<HashSet<_>>().len() != self.codes.len()
            || self.token_ids.iter().collect::<HashSet<_>>().len() != self.token_ids.len()
        {
            return Err(invalid("d3 decision codes and token IDs must be unique"));
        }
        Ok(())
    }

    /// Code names in index order.
    pub fn code_names(&self) -> &[String] {
        &self.codes
    }

    /// Token IDs for each code in index order.
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }
}

impl D3Manifest {
    pub fn validate(&self) -> Result<()> {
        if self.schema != "d3-package-manifest/1"
            || self.model_name != D3_MINI_MODEL
            || self.repo_id != D3_MINI_REPOSITORY
            || self.format_id != D3_FORMAT
            || self.prompt != D3_PROMPT
            || self.attention_mode != D3_ATTENTION
            || self.identity.model_sha256.len() != 64
            || self.parameters.text != 4_205_751_296
            || self.parameters.vision != 333_514_240
            || self.parameters.readout != 652_800
            || self.parameters.loaded != 4_539_918_336
        {
            return Err(invalid("unsupported d3-mini package manifest"));
        }
        Ok(())
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    let raw = std::fs::read_to_string(path)
        .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
    serde_json::from_str(&raw)
        .map_err(|error| Error::Other(format!("parse {}: {error}", path.display())))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(format!("d3-mini config: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn official_text() -> D3TextConfig {
        D3TextConfig {
            attention_bias: false,
            attn_output_gate: true,
            dtype: D3_BACKBONE_DTYPE.into(),
            eos_token_id: 248_044,
            full_attention_interval: 4,
            head_dim: 256,
            hidden_size: 2560,
            intermediate_size: 9216,
            layer_types: (0..32)
                .map(|index| {
                    if (index + 1) % 4 == 0 {
                        "full_attention"
                    } else {
                        "linear_attention"
                    }
                    .into()
                })
                .collect(),
            linear_conv_kernel_dim: 4,
            linear_key_head_dim: 128,
            linear_num_key_heads: 16,
            linear_num_value_heads: 32,
            linear_value_head_dim: 128,
            max_position_embeddings: 262_144,
            num_attention_heads: 16,
            num_hidden_layers: 32,
            num_key_value_heads: 4,
            partial_rotary_factor: 0.25,
            rms_norm_eps: 1e-6,
            rope_parameters: D3TextRopeConfig {
                mrope_interleaved: true,
                mrope_section: [11, 11, 10],
                partial_rotary_factor: 0.25,
                rope_theta: 10_000_000.0,
            },
            tie_word_embeddings: true,
            vocab_size: 248_320,
        }
    }

    #[test]
    fn accepts_official_text_geometry() {
        official_text().validate().unwrap();
    }

    #[test]
    fn rejects_causal_full_attention_schedule_substitution() {
        let mut config = official_text();
        config.layer_types[3] = "linear_attention".into();
        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_non_fp32_readout() {
        let config = D3DecisionConfig {
            format_version: 1,
            format_id: D3_FORMAT.into(),
            prompt: D3_PROMPT.into(),
            base_model: "Qwen/Qwen3.5-4B".into(),
            revision: "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a".into(),
            codes: vec!["A".into(), "B".into()],
            token_ids: vec![32, 33],
            temperature: 1.0,
            attention_mode: D3_ATTENTION.into(),
            pooling: D3_POOLING.into(),
            max_length: None,
            readout_dtype: "bfloat16".into(),
        };
        assert!(config.validate().is_err());
    }

    #[test]
    #[ignore = "needs the pinned official d3-mini package"]
    fn official_package_files_pass_contract() {
        let path = std::env::var_os("APXINF_D3_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("set APXINF_D3_CHECKPOINT to the official d3-mini directory");
        let package = D3PackageConfig::from_dir(&path).unwrap();
        assert_eq!(package.manifest.model_name, "d3-mini");
        assert_eq!(package.decision.codes.len(), 255);
        assert_eq!(package.model.text_config.hidden_size, 2560);
        assert_eq!(package.model.vision_config.out_hidden_size, 2560);
    }
}
