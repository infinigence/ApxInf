//! d3-lite package configuration: checkpoint identity, backbone geometry and
//! the FP32 255-code readout contract.
//!
//! This module is the Metal port's own copy of the checkpoint contract. It
//! intentionally does not import the CUDA-only `d3` family: backend-native
//! ports are self-contained (see `doc/model-layer-architecture.md`, "Where
//! backend-native model code lives"). The port runs the d3-lite geometry only;
//! d3-mini and other Qwen3.5 hybrids fail at load time with a named field.

use std::collections::HashSet;
use std::path::Path;

use apxinf_core::{Error, Result};
use serde::Deserialize;

pub const FORMAT_ID: &str = "d3-code-readout-v1";
pub const PROMPT: &str = "d3";
pub const ATTENTION_MODE: &str = "noncausal_full_attention";
pub const POOLING: &str = "last";
pub const READOUT_DTYPE: &str = "float32";
pub const CODES: usize = 255;
/// Longest request this port admits; longer requests fail, never truncate.
pub const MAX_INPUT_TOKENS: usize = 262_144;
pub const BACKBONE_DTYPE: &str = "bfloat16";
/// Image pixel budget the checkpoint's own runtime applies at load
/// (`d3_runtime.load_processor`): it narrows `preprocessor_config.json`'s
/// `longest_edge` to 1.6 MP while keeping `shortest_edge` at 64 K. The port
/// copies that contract so a request takes the same resize as the reference.
pub const IMAGE_MIN_PIXELS: usize = 65_536;
pub const IMAGE_MAX_PIXELS: usize = 1_638_400;
/// The vision tower's axial RoPE base (`vision_config.rope_parameters`).
pub const VISION_ROPE_THETA: f32 = 10_000.0;

/// Immutable numbers every traced computation is keyed by. `Copy` so the
/// compiled callable can capture one copy without sharing mutable state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub k_heads: usize,
    pub v_heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    pub conv_kernel: usize,
    pub full_attention_interval: usize,
    pub eps: f32,
    pub rope_dims: usize,
    pub rope_base: f32,
    pub vocab: usize,
    pub max_positions: usize,
}

impl Geometry {
    pub fn key_dim(self) -> usize {
        self.k_heads * self.k_dim
    }

    pub fn value_dim(self) -> usize {
        self.v_heads * self.v_dim
    }

    pub fn conv_dim(self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }

    pub fn q_width(self) -> usize {
        2 * self.q_heads * self.head_dim
    }

    pub fn kv_width(self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// The checkpoint's schedule: full attention at every `interval`-th layer
    /// (1-based), GatedDeltaNet elsewhere.
    pub fn is_attention(self, layer: usize) -> bool {
        (layer + 1) % self.full_attention_interval == 0
    }
}

/// The exact d3-lite package this port accepts.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub geometry: Geometry,
    pub codes: Vec<String>,
    pub token_ids: Vec<u32>,
    pub temperature: f32,
    pub base_model: String,
    pub revision: String,
    pub model_sha256: String,
    pub vision: VisionContract,
    pub preprocessor: PreprocessorContract,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
}

/// The vision tower's geometry, parsed and validated so the package identity
/// is unambiguous **and** the port can execute `visual.*`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VisionContract {
    pub depth: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub out_hidden: usize,
    pub patch: usize,
    pub temporal_patch: usize,
    pub spatial_merge: usize,
    pub num_position_embeddings: usize,
    pub num_grid_per_side: usize,
    pub rope_theta: f32,
    pub tensors: usize,
}

/// Checkpoint-fixed image canonicalization: resize, normalize and patchify.
/// The runtime owns it because every step is determined by the checkpoint
/// files, not by the caller (see `model-layer-architecture.md`, "VLA input seam").
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreprocessorContract {
    pub min_pixels: usize,
    pub max_pixels: usize,
    /// Resize granularity: `patch * spatial_merge` (32 for d3-lite).
    pub factor: usize,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

impl Config {
    pub fn from_dir(path: &Path) -> Result<Self> {
        let read = |name: &str| {
            let file = path.join(name);
            std::fs::read_to_string(&file)
                .map_err(|error| Error::Other(format!("read {}: {error}", file.display())))
        };
        let checkpoint: CheckpointConfig =
            serde_json::from_str(&read("config.json")?).map_err(|error| {
                Error::Other(format!("parse {}config.json: {error}", path.display()))
            })?;
        let decision: DecisionConfig = serde_json::from_str(&read("decision_config.json")?)
            .map_err(|error| {
                Error::Other(format!(
                    "parse {}decision_config.json: {error}",
                    path.display()
                ))
            })?;
        let manifest: Manifest =
            serde_json::from_str(&read("MODEL_MANIFEST.json")?).map_err(|error| {
                Error::Other(format!(
                    "parse {}MODEL_MANIFEST.json: {error}",
                    path.display()
                ))
            })?;
        let preprocessor: PreprocessorConfig =
            serde_json::from_str(&read("preprocessor_config.json")?).map_err(|error| {
                Error::Other(format!(
                    "parse {}preprocessor_config.json: {error}",
                    path.display()
                ))
            })?;
        let config = Self::compose(checkpoint, decision, manifest, preprocessor)?;
        Ok(config)
    }

    fn compose(
        checkpoint: CheckpointConfig,
        decision: DecisionConfig,
        manifest: Manifest,
        processor_file: PreprocessorConfig,
    ) -> Result<Self> {
        let invalid = |message: String| Error::Other(format!("d3-lite config: {message}"));
        if checkpoint.architectures.as_slice() != ["Qwen3_5Model"]
            || checkpoint.dtype != BACKBONE_DTYPE
            || checkpoint.model_type != "qwen3_5"
            || !checkpoint.tie_word_embeddings
        {
            return Err(invalid(
                "unsupported d3-lite backbone identity or dtype".into(),
            ));
        }
        let text = &checkpoint.text_config;
        let geometry = Geometry {
            hidden: text.hidden_size,
            intermediate: text.intermediate_size,
            layers: text.num_hidden_layers,
            q_heads: text.num_attention_heads,
            kv_heads: text.num_key_value_heads,
            head_dim: text.head_dim,
            k_heads: text.linear_num_key_heads,
            v_heads: text.linear_num_value_heads,
            k_dim: text.linear_key_head_dim,
            v_dim: text.linear_value_head_dim,
            conv_kernel: text.linear_conv_kernel_dim,
            full_attention_interval: text.full_attention_interval,
            eps: text.rms_norm_eps,
            rope_dims: (text.head_dim as f32 * text.rope_parameters.partial_rotary_factor) as usize,
            rope_base: text.rope_parameters.rope_theta,
            vocab: text.vocab_size,
            max_positions: text.max_position_embeddings,
        };
        Self::validate_text(text, geometry)?;
        let vision = VisionContract {
            depth: checkpoint.vision_config.depth,
            hidden: checkpoint.vision_config.hidden_size,
            intermediate: checkpoint.vision_config.intermediate_size,
            heads: checkpoint.vision_config.num_heads,
            head_dim: checkpoint.vision_config.hidden_size / checkpoint.vision_config.num_heads,
            out_hidden: checkpoint.vision_config.out_hidden_size,
            patch: checkpoint.vision_config.patch_size,
            temporal_patch: checkpoint.vision_config.temporal_patch_size,
            spatial_merge: checkpoint.vision_config.spatial_merge_size,
            num_position_embeddings: checkpoint.vision_config.num_position_embeddings,
            // `sqrt(2304) = 48`; the learned table is a 48 x 48 grid.
            num_grid_per_side: (checkpoint.vision_config.num_position_embeddings as f64).sqrt()
                as usize,
            rope_theta: checkpoint.vision_config.rope_parameters.rope_theta,
            // 153 tensors: 12 blocks x 12 + merger 6 + patch embed 2 + pos embed 1.
            tensors: 153,
        };
        Self::validate_vision(&checkpoint.vision_config, vision)?;
        let preprocessor = PreprocessorContract {
            min_pixels: IMAGE_MIN_PIXELS,
            max_pixels: IMAGE_MAX_PIXELS,
            factor: vision.patch * vision.spatial_merge,
            image_mean: float3_array(processor_file.image_mean, "preprocessor image_mean")?,
            image_std: float3_array(processor_file.image_std, "preprocessor image_std")?,
        };
        Self::validate_preprocessor(&preprocessor, &processor_file, vision)?;
        let image_token_id = checkpoint.image_token_id;
        let video_token_id = checkpoint.video_token_id;
        let vision_start_token_id = checkpoint.vision_start_token_id;
        let vision_end_token_id = checkpoint.vision_end_token_id;
        Self::validate_tokens(
            image_token_id,
            video_token_id,
            vision_start_token_id,
            vision_end_token_id,
            geometry,
        )?;
        Self::validate_decision(&decision)?;
        let codes = decision.codes;
        let token_ids = decision.token_ids;
        let temperature = decision.temperature;
        let base_model = decision.base_model.clone();
        let revision = decision.revision.clone();
        Self::validate_manifest(&manifest, &codes, &token_ids, geometry)?;
        Ok(Self {
            geometry,
            codes,
            token_ids,
            temperature,
            base_model,
            revision,
            model_sha256: manifest.identity.model_sha256,
            vision,
            preprocessor,
            image_token_id,
            video_token_id,
            vision_start_token_id,
            vision_end_token_id,
        })
    }

    fn validate_text(text: &TextConfig, geometry: Geometry) -> Result<()> {
        let invalid = |message: String| Error::Other(format!("d3-lite config: {message}"));
        // The d3-lite backbone (vllm-sr/d3-lite): Qwen3.5-0.8B geometry.
        let exact = geometry.hidden == 1_024
            && geometry.intermediate == 3_584
            && geometry.layers == 24
            && geometry.q_heads == 8
            && geometry.kv_heads == 2
            && geometry.head_dim == 256
            && geometry.full_attention_interval == 4
            && geometry.k_heads == 16
            && geometry.v_heads == 16
            && geometry.k_dim == 128
            && geometry.v_dim == 128
            && geometry.conv_kernel == 4
            && geometry.vocab == 248_320
            && geometry.max_positions == 262_144
            && geometry.eps == 1e-6
            && !text.attention_bias
            && text.attn_output_gate
            && text.tie_word_embeddings
            && text.layer_types.len() == geometry.layers;
        if !exact {
            return Err(invalid(
                "unsupported d3-lite text geometry (this Metal port runs the 0.8B d3-lite backbone only)"
                    .into(),
            ));
        }
        for (index, kind) in text.layer_types.iter().enumerate() {
            let expected = if geometry.is_attention(index) {
                "full_attention"
            } else {
                "linear_attention"
            };
            if kind != expected {
                return Err(invalid(format!(
                    "layer_types[{index}] is `{kind}`, expected `{expected}` for the 3 GDN + 1 attention schedule"
                )));
            }
        }
        let rope = &text.rope_parameters;
        if text.partial_rotary_factor != 0.25
            || rope.partial_rotary_factor != 0.25
            || rope.rope_theta != 10_000_000.0
            || !rope.mrope_interleaved
            || rope.mrope_section != [11, 11, 10]
            || geometry.rope_dims != 64
        {
            return Err(invalid(
                "unsupported d3-lite normalization or partial RoPE contract".into(),
            ));
        }
        Ok(())
    }

    fn validate_vision(vision: &VisionConfig, contract: VisionContract) -> Result<()> {
        let invalid = |message: String| Error::Other(format!("d3-lite config: {message}"));
        if vision.depth != 12
            || vision.hidden_size != 768
            || vision.intermediate_size != 3_072
            || vision.num_heads != 12
            || vision.out_hidden_size != 1_024
            || vision.patch_size != 16
            || vision.temporal_patch_size != 2
            || vision.spatial_merge_size != 2
            || vision.num_position_embeddings != 2_304
            || contract.out_hidden != contract.hidden * 4 / 3
            || contract.head_dim != 64
            || contract.num_grid_per_side * contract.num_grid_per_side
                != contract.num_position_embeddings
            || vision.rope_parameters.rope_theta != VISION_ROPE_THETA
            || vision.rope_parameters.rope_type != "axial"
        {
            return Err(invalid(
                "unsupported d3-lite vision tower geometry (qwen3_5_vision depth 12)".into(),
            ));
        }
        Ok(())
    }

    fn validate_preprocessor(
        contract: &PreprocessorContract,
        file: &PreprocessorConfig,
        vision: VisionContract,
    ) -> Result<()> {
        let invalid = |message: String| Error::Other(format!("d3-lite config: {message}"));
        let size = &file.size;
        if file.patch_size != vision.patch
            || file.temporal_patch_size != vision.temporal_patch
            || file.merge_size != vision.spatial_merge
            || size.shortest_edge != Some(IMAGE_MIN_PIXELS)
            || size.longest_edge.is_none_or(|edge| edge < IMAGE_MAX_PIXELS)
            || contract.factor != vision.patch * vision.spatial_merge
            || contract.min_pixels != IMAGE_MIN_PIXELS
            || contract.max_pixels != IMAGE_MAX_PIXELS
        {
            return Err(invalid(
                "unsupported d3-lite image preprocessor contract".into(),
            ));
        }
        for (name, value) in [
            ("image_mean", contract.image_mean),
            ("image_std", contract.image_std),
        ] {
            if value
                .iter()
                .any(|entry| !entry.is_finite() || *entry <= 0.0)
            {
                return Err(invalid(format!(
                    "preprocessor {name} must be finite and positive"
                )));
            }
        }
        if contract.image_mean != [0.5, 0.5, 0.5] || contract.image_std != [0.5, 0.5, 0.5] {
            return Err(invalid(
                "the d3-lite preprocessor normalizes RGB with mean/std 0.5".into(),
            ));
        }
        Ok(())
    }

    /// The multimodal token IDs `config.json` fixes: image placeholders are
    /// replaced by vision features, and `vision_start`/`vision_end` delimit
    /// each image's placeholder run in the rendered chat template.
    fn validate_tokens(
        image: u32,
        video: u32,
        start: u32,
        end: u32,
        geometry: Geometry,
    ) -> Result<()> {
        let invalid = |message: String| Error::Other(format!("d3-lite config: {message}"));
        if image != 248_056 || video != 248_057 || start != 248_053 || end != 248_054 {
            return Err(invalid(
                "unsupported d3-lite multimodal token IDs (image 248056, vision_start 248053, \
                 vision_end 248054)"
                    .into(),
            ));
        }
        for (name, token) in [
            ("image_token_id", image),
            ("video_token_id", video),
            ("vision_start_token_id", start),
            ("vision_end_token_id", end),
        ] {
            if token as usize >= geometry.vocab {
                return Err(invalid(format!("{name} {token} is outside the vocabulary")));
            }
        }
        Ok(())
    }

    fn validate_decision(decision: &DecisionConfig) -> Result<()> {
        let invalid = |message: String| Error::Other(format!("d3-lite config: {message}"));
        if decision.format_version != 1
            || decision.format_id != FORMAT_ID
            || decision.prompt != PROMPT
            || decision.attention_mode != ATTENTION_MODE
            || decision.pooling != POOLING
            || decision.readout_dtype != READOUT_DTYPE
            || decision.max_length.is_some()
            || decision.temperature != 1.0
        {
            return Err(invalid("unsupported d3-lite decision contract".into()));
        }
        if decision.codes.len() != CODES || decision.token_ids.len() != CODES {
            return Err(invalid(format!(
                "the decision config names {} codes / {} token IDs, expected {CODES} each",
                decision.codes.len(),
                decision.token_ids.len()
            )));
        }
        if decision.codes.iter().collect::<HashSet<_>>().len() != CODES
            || decision.token_ids.iter().collect::<HashSet<_>>().len() != CODES
        {
            return Err(invalid(
                "d3-lite decision codes and token IDs must be unique".into(),
            ));
        }
        if decision.codes.iter().any(|code| code.is_empty()) {
            return Err(invalid("d3-lite decision codes must be nonempty".into()));
        }
        Ok(())
    }

    fn validate_manifest(
        manifest: &Manifest,
        codes: &[String],
        token_ids: &[u32],
        geometry: Geometry,
    ) -> Result<()> {
        let invalid = |message: String| Error::Other(format!("d3-lite config: {message}"));
        if manifest.schema != "d3-package-manifest/1"
            || manifest.model_name != "d3-lite"
            || manifest.repo_id != "vllm-sr/d3-lite"
            || manifest.format_id != FORMAT_ID
            || manifest.prompt != PROMPT
            || manifest.attention_mode != ATTENTION_MODE
            || manifest.max_input_tokens.is_some()
            || manifest.identity.model_sha256.len() != 64
            || manifest.parameters.text != 752_393_024
            || manifest.parameters.vision != 100_592_896
            || manifest.parameters.readout != 261_120
            || manifest.parameters.loaded != 853_247_040
        {
            return Err(invalid("unsupported d3-lite package manifest".into()));
        }
        if manifest.parameters.readout != (codes.len() * geometry.hidden) as u64 {
            return Err(invalid(
                "manifest readout parameter count does not match codes x hidden_size".into(),
            ));
        }
        let embedded = &manifest.identity.decision;
        if embedded.format_id != FORMAT_ID
            || embedded.prompt != PROMPT
            || embedded.attention_mode != ATTENTION_MODE
            || embedded.pooling != POOLING
            || embedded.readout_dtype != READOUT_DTYPE
            || embedded.temperature != 1.0
            || embedded.codes.as_slice() != codes
            || embedded.token_ids.as_slice() != token_ids
        {
            return Err(invalid(
                "manifest identity decision differs from decision_config.json".into(),
            ));
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

// ---- serde shapes (extra checkpoint fields are deliberately ignored) ----

#[derive(Clone, Debug, Deserialize)]
struct CheckpointConfig {
    architectures: Vec<String>,
    dtype: String,
    image_token_id: u32,
    model_type: String,
    text_config: TextConfig,
    tie_word_embeddings: bool,
    video_token_id: u32,
    vision_config: VisionConfig,
    vision_end_token_id: u32,
    vision_start_token_id: u32,
}

#[derive(Clone, Debug, Deserialize)]
struct TextConfig {
    attention_bias: bool,
    attn_output_gate: bool,
    full_attention_interval: usize,
    head_dim: usize,
    hidden_size: usize,
    intermediate_size: usize,
    layer_types: Vec<String>,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_value_head_dim: usize,
    max_position_embeddings: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    partial_rotary_factor: f32,
    rms_norm_eps: f32,
    rope_parameters: RopeParameters,
    tie_word_embeddings: bool,
    vocab_size: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct RopeParameters {
    mrope_interleaved: bool,
    mrope_section: [usize; 3],
    partial_rotary_factor: f32,
    rope_theta: f32,
}

#[derive(Clone, Debug, Deserialize)]
struct VisionConfig {
    depth: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_heads: usize,
    num_position_embeddings: usize,
    out_hidden_size: usize,
    patch_size: usize,
    rope_parameters: VisionRopeParameters,
    spatial_merge_size: usize,
    temporal_patch_size: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct VisionRopeParameters {
    rope_theta: f32,
    rope_type: String,
}

#[derive(Clone, Debug, Deserialize)]
struct PreprocessorConfig {
    image_mean: [f32; 3],
    image_std: [f32; 3],
    merge_size: usize,
    patch_size: usize,
    size: PreprocessorSize,
    temporal_patch_size: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct PreprocessorSize {
    longest_edge: Option<usize>,
    shortest_edge: Option<usize>,
}

fn float3_array(values: [f32; 3], name: &str) -> Result<[f32; 3]> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Error::Other(format!(
            "d3-lite config: {name} must be finite"
        )));
    }
    Ok(values)
}

#[derive(Clone, Debug, Deserialize)]
struct DecisionConfig {
    format_version: usize,
    format_id: String,
    prompt: String,
    base_model: String,
    revision: String,
    codes: Vec<String>,
    token_ids: Vec<u32>,
    temperature: f32,
    attention_mode: String,
    pooling: String,
    max_length: Option<usize>,
    readout_dtype: String,
}

#[derive(Clone, Debug, Deserialize)]
struct Manifest {
    schema: String,
    model_name: String,
    repo_id: String,
    format_id: String,
    prompt: String,
    attention_mode: String,
    max_input_tokens: Option<usize>,
    identity: ManifestIdentity,
    parameters: ManifestParameters,
}

#[derive(Clone, Debug, Deserialize)]
struct ManifestIdentity {
    model_sha256: String,
    decision: ManifestDecision,
}

#[derive(Clone, Debug, Deserialize)]
struct ManifestDecision {
    format_id: String,
    prompt: String,
    codes: Vec<String>,
    token_ids: Vec<u32>,
    temperature: f32,
    attention_mode: String,
    pooling: String,
    readout_dtype: String,
}

#[derive(Clone, Debug, Deserialize)]
struct ManifestParameters {
    text: u64,
    vision: u64,
    readout: u64,
    loaded: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn package_json() -> (String, String, String, String) {
        let checkpoint = r#"{
            "architectures": ["Qwen3_5Model"],
            "dtype": "bfloat16",
            "image_token_id": 248056,
            "model_type": "qwen3_5",
            "tie_word_embeddings": true,
            "video_token_id": 248057,
            "vision_end_token_id": 248054,
            "vision_start_token_id": 248053,
            "text_config": {
                "attention_bias": false,
                "attn_output_gate": true,
                "full_attention_interval": 4,
                "head_dim": 256,
                "hidden_size": 1024,
                "intermediate_size": 3584,
                "layer_types": [
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention"
                ],
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 16,
                "linear_value_head_dim": 128,
                "max_position_embeddings": 262144,
                "num_attention_heads": 8,
                "num_hidden_layers": 24,
                "num_key_value_heads": 2,
                "partial_rotary_factor": 0.25,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "mrope_interleaved": true,
                    "mrope_section": [11, 11, 10],
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10000000
                },
                "tie_word_embeddings": true,
                "vocab_size": 248320
            },
            "vision_config": {
                "depth": 12,
                "hidden_size": 768,
                "intermediate_size": 3072,
                "num_heads": 12,
                "num_position_embeddings": 2304,
                "out_hidden_size": 1024,
                "patch_size": 16,
                "rope_parameters": {"rope_theta": 10000.0, "rope_type": "axial"},
                "spatial_merge_size": 2,
                "temporal_patch_size": 2
            }
        }"#;
        let preprocessor = serde_json::json!({
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "merge_size": 2,
            "patch_size": 16,
            "size": {"longest_edge": 16_777_216usize, "shortest_edge": 65_536usize},
            "temporal_patch_size": 2,
        })
        .to_string();
        let codes: Vec<String> = (0..CODES).map(|i| format!("C{i}")).collect();
        let token_ids: Vec<u32> = (0..CODES).map(|i| 1_000 + i as u32).collect();
        let decision = serde_json::json!({
            "format_version": 1,
            "format_id": FORMAT_ID,
            "prompt": PROMPT,
            "base_model": "Qwen/Qwen3.5-0.8B",
            "revision": "2fc06364715b967f1860aea9cf38778875588b17",
            "codes": codes,
            "token_ids": token_ids,
            "temperature": 1.0,
            "attention_mode": ATTENTION_MODE,
            "pooling": POOLING,
            "max_length": null,
            "readout_dtype": READOUT_DTYPE,
        })
        .to_string();
        let manifest = serde_json::json!({
            "schema": "d3-package-manifest/1",
            "model_name": "d3-lite",
            "repo_id": "vllm-sr/d3-lite",
            "format_id": FORMAT_ID,
            "prompt": PROMPT,
            "attention_mode": ATTENTION_MODE,
            "max_input_tokens": null,
            "identity": {
                "model_sha256": "0".repeat(64),
                "decision": serde_json::from_str::<serde_json::Value>(&decision).unwrap()
            },
            "parameters": {"text": 752_393_024u64, "vision": 100_592_896u64,
                            "readout": 261_120u64, "loaded": 853_247_040u64}
        })
        .to_string();
        (checkpoint.into(), decision, manifest, preprocessor)
    }

    fn compose() -> Result<Config> {
        let (checkpoint, decision, manifest, preprocessor) = package_json();
        Config::compose(
            serde_json::from_str(&checkpoint).unwrap(),
            serde_json::from_str(&decision).unwrap(),
            serde_json::from_str(&manifest).unwrap(),
            serde_json::from_str(&preprocessor).unwrap(),
        )
    }

    #[test]
    fn accepts_the_d3_lite_package_contract() {
        let config = compose().unwrap();
        assert_eq!(config.geometry.hidden, 1_024);
        assert_eq!(config.geometry.layers, 24);
        assert_eq!(config.geometry.rope_dims, 64);
        assert_eq!(config.codes.len(), CODES);
        assert_eq!(config.token_ids.len(), CODES);
        assert!(!config.geometry.is_attention(0));
        assert!(config.geometry.is_attention(3));
        assert_eq!(config.vision.depth, 12);
        assert_eq!(config.vision.head_dim, 64);
        assert_eq!(config.vision.num_grid_per_side, 48);
        assert_eq!(config.vision.rope_theta, 10_000.0);
        assert_eq!(config.image_token_id, 248_056);
        assert_eq!(config.vision_start_token_id, 248_053);
        assert_eq!(config.vision_end_token_id, 248_054);
        assert_eq!(config.preprocessor.factor, 32);
        assert_eq!(config.preprocessor.min_pixels, IMAGE_MIN_PIXELS);
        assert_eq!(config.preprocessor.max_pixels, IMAGE_MAX_PIXELS);
        assert_eq!(config.preprocessor.image_mean, [0.5, 0.5, 0.5]);
    }

    #[test]
    fn rejects_the_d3_mini_geometry_with_the_offending_identity() {
        let (checkpoint, decision, manifest, preprocessor) = package_json();
        let mut value: serde_json::Value = serde_json::from_str(&checkpoint).unwrap();
        value["text_config"]["hidden_size"] = serde_json::json!(2560);
        let error = Config::compose(
            serde_json::from_value(value).unwrap(),
            serde_json::from_str(&decision).unwrap(),
            serde_json::from_str(&manifest).unwrap(),
            serde_json::from_str(&preprocessor).unwrap(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("d3-lite text geometry"), "{error}");
    }

    #[test]
    fn rejects_a_wrong_layer_schedule() {
        let (checkpoint, decision, manifest, preprocessor) = package_json();
        let mut value: serde_json::Value = serde_json::from_str(&checkpoint).unwrap();
        value["text_config"]["layer_types"][3] = serde_json::json!("linear_attention");
        let error = Config::compose(
            serde_json::from_value(value).unwrap(),
            serde_json::from_str(&decision).unwrap(),
            serde_json::from_str(&manifest).unwrap(),
            serde_json::from_str(&preprocessor).unwrap(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("layer_types[3]"), "{error}");
    }

    #[test]
    fn rejects_a_non_fp32_readout_and_a_non_unit_temperature() {
        for (field, value, message) in [
            (
                "readout_dtype",
                serde_json::json!("bfloat16"),
                "decision contract",
            ),
            ("temperature", serde_json::json!(0.5), "decision contract"),
            ("pooling", serde_json::json!("mean"), "decision contract"),
        ] {
            let (checkpoint, decision, manifest, preprocessor) = package_json();
            let mut decision: serde_json::Value = serde_json::from_str(&decision).unwrap();
            decision[field] = value;
            let error = Config::compose(
                serde_json::from_str(&checkpoint).unwrap(),
                serde_json::from_value(decision).unwrap(),
                serde_json::from_str(&manifest).unwrap(),
                serde_json::from_str(&preprocessor).unwrap(),
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains(message), "{field}: {error}");
        }
    }

    #[test]
    fn rejects_a_manifest_parameter_mismatch() {
        let (checkpoint, decision, manifest, preprocessor) = package_json();
        let mut manifest: serde_json::Value = serde_json::from_str(&manifest).unwrap();
        manifest["parameters"]["loaded"] = serde_json::json!(1u64);
        let error = Config::compose(
            serde_json::from_str(&checkpoint).unwrap(),
            serde_json::from_str(&decision).unwrap(),
            serde_json::from_value(manifest).unwrap(),
            serde_json::from_str(&preprocessor).unwrap(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("manifest"), "{error}");
    }

    #[test]
    fn rejects_wrong_multimodal_tokens_and_preprocessor_budgets() {
        let (checkpoint, decision, manifest, preprocessor) = package_json();
        let mut checkpoint: serde_json::Value = serde_json::from_str(&checkpoint).unwrap();
        checkpoint["image_token_id"] = serde_json::json!(1);
        let error = Config::compose(
            serde_json::from_value(checkpoint).unwrap(),
            serde_json::from_str(&decision).unwrap(),
            serde_json::from_str(&manifest).unwrap(),
            serde_json::from_str(&preprocessor).unwrap(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("multimodal token IDs"), "{error}");

        let (checkpoint, decision, manifest, preprocessor) = package_json();
        let mut preprocessor: serde_json::Value = serde_json::from_str(&preprocessor).unwrap();
        preprocessor["size"]["shortest_edge"] = serde_json::json!(3_136usize);
        let error = Config::compose(
            serde_json::from_str(&checkpoint).unwrap(),
            serde_json::from_str(&decision).unwrap(),
            serde_json::from_str(&manifest).unwrap(),
            serde_json::from_value(preprocessor).unwrap(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("preprocessor"), "{error}");
    }
}
