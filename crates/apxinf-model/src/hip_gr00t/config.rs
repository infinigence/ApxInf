use std::path::Path;

use apxinf_core::{Error, Result};
use serde_json::{Map, Value};

const MODEL_TYPE: &str = "Gr00tN1d7";
const BF16: &str = "bfloat16";

/// Transformer configuration for the GR00T N1.7 diffusion action head.
#[derive(Clone, Debug, PartialEq)]
pub struct Gr00tDiffusionConfig {
    pub positional_embeddings: Option<String>,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub norm_type: String,
    pub dropout: f32,
    pub final_dropout: bool,
    pub output_dim: usize,
    pub interleave_self_attention: bool,
    pub activation_fn: String,
    pub attention_bias: bool,
    pub upcast_attention: bool,
    pub norm_elementwise_affine: bool,
    pub norm_eps: f32,
    pub max_num_positional_embeddings: usize,
}

impl Default for Gr00tDiffusionConfig {
    fn default() -> Self {
        Self {
            positional_embeddings: None,
            num_layers: 16,
            num_attention_heads: 32,
            attention_head_dim: 48,
            norm_type: "ada_norm".into(),
            dropout: 0.2,
            final_dropout: true,
            output_dim: 1024,
            interleave_self_attention: true,
            // This default comes from DiT.__init__, not BasicTransformerBlock.
            activation_fn: "gelu-approximate".into(),
            attention_bias: true,
            upcast_attention: false,
            norm_elementwise_affine: false,
            norm_eps: 1e-5,
            max_num_positional_embeddings: 512,
        }
    }
}

impl Gr00tDiffusionConfig {
    pub fn inner_dim(&self) -> Result<usize> {
        self.num_attention_heads
            .checked_mul(self.attention_head_dim)
            .ok_or_else(|| Error::Other("GR00T diffusion inner dimension overflow".into()))
    }

    fn update_from_json(&mut self, value: &Value) -> Result<()> {
        let object = object(value, "diffusion_model_cfg")?;
        self.positional_embeddings = optional_string(
            object,
            "positional_embeddings",
            self.positional_embeddings.clone(),
        )?;
        self.num_layers = usize_value(object, "num_layers", self.num_layers)?;
        self.num_attention_heads =
            usize_value(object, "num_attention_heads", self.num_attention_heads)?;
        self.attention_head_dim =
            usize_value(object, "attention_head_dim", self.attention_head_dim)?;
        self.norm_type = string_value(object, "norm_type", &self.norm_type)?;
        self.dropout = f32_value(object, "dropout", self.dropout)?;
        self.final_dropout = bool_value(object, "final_dropout", self.final_dropout)?;
        self.output_dim = usize_value(object, "output_dim", self.output_dim)?;
        self.interleave_self_attention = bool_value(
            object,
            "interleave_self_attention",
            self.interleave_self_attention,
        )?;
        self.activation_fn = string_value(object, "activation_fn", &self.activation_fn)?;
        self.attention_bias = bool_value(object, "attention_bias", self.attention_bias)?;
        self.upcast_attention = bool_value(object, "upcast_attention", self.upcast_attention)?;
        self.norm_elementwise_affine = bool_value(
            object,
            "norm_elementwise_affine",
            self.norm_elementwise_affine,
        )?;
        self.norm_eps = f32_value(object, "norm_eps", self.norm_eps)?;
        self.max_num_positional_embeddings = usize_value(
            object,
            "max_num_positional_embeddings",
            self.max_num_positional_embeddings,
        )?;
        Ok(())
    }
}

/// Optional self-attention stack applied to backbone features before DiT.
#[derive(Clone, Debug, PartialEq)]
pub struct Gr00tVlSelfAttentionConfig {
    pub positional_embeddings: Option<String>,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub dropout: f32,
    pub final_dropout: bool,
    pub activation_fn: String,
    pub attention_bias: bool,
    pub upcast_attention: bool,
}

impl Default for Gr00tVlSelfAttentionConfig {
    fn default() -> Self {
        Self {
            positional_embeddings: None,
            num_layers: 0,
            num_attention_heads: 32,
            attention_head_dim: 64,
            dropout: 0.2,
            final_dropout: true,
            activation_fn: "gelu-approximate".into(),
            attention_bias: true,
            upcast_attention: false,
        }
    }
}

impl Gr00tVlSelfAttentionConfig {
    pub fn hidden_size(&self) -> Result<usize> {
        self.num_attention_heads
            .checked_mul(self.attention_head_dim)
            .ok_or_else(|| Error::Other("GR00T VL self-attention width overflow".into()))
    }

    fn from_json(value: &Value) -> Result<Self> {
        let object = object(value, "vl_self_attention_cfg")?;
        let mut config = Self::default();
        config.positional_embeddings = optional_string(
            object,
            "positional_embeddings",
            config.positional_embeddings.clone(),
        )?;
        config.num_layers = usize_value(object, "num_layers", config.num_layers)?;
        config.num_attention_heads =
            usize_value(object, "num_attention_heads", config.num_attention_heads)?;
        config.attention_head_dim =
            usize_value(object, "attention_head_dim", config.attention_head_dim)?;
        config.dropout = f32_value(object, "dropout", config.dropout)?;
        config.final_dropout = bool_value(object, "final_dropout", config.final_dropout)?;
        config.activation_fn = string_value(object, "activation_fn", &config.activation_fn)?;
        config.attention_bias = bool_value(object, "attention_bias", config.attention_bias)?;
        config.upcast_attention = bool_value(object, "upcast_attention", config.upcast_attention)?;
        Ok(config)
    }
}

/// Inference-relevant GR00T N1.7 model configuration.
///
/// NVIDIA checkpoints omit fields whose values come from Python constructor
/// defaults. Parsing therefore starts from the N1.7 defaults and applies every
/// serialized override. Runtime code must use this parsed value rather than
/// assuming the defaults: the released 3B checkpoint uses a deeper backbone
/// and action head than the source defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct Gr00tConfig {
    pub model_type: String,
    pub model_dtype: String,
    pub model_name: String,
    pub backbone_model_type: String,
    pub backbone_embedding_dim: usize,
    pub select_layer: usize,
    pub reproject_vision: bool,
    pub use_flash_attention: bool,
    pub image_crop_size: Option<[usize; 2]>,
    pub image_target_size: Option<[usize; 2]>,
    pub max_state_dim: usize,
    pub max_action_dim: usize,
    pub action_horizon: usize,
    pub hidden_size: usize,
    pub input_embedding_dim: usize,
    pub state_history_length: usize,
    pub add_pos_embed: bool,
    pub use_vlln: bool,
    pub max_seq_len: usize,
    pub use_alternate_vl_dit: bool,
    pub attend_text_every_n_blocks: usize,
    pub diffusion: Gr00tDiffusionConfig,
    pub use_vl_self_attention: bool,
    pub vl_self_attention: Option<Gr00tVlSelfAttentionConfig>,
    pub num_inference_timesteps: usize,
    pub num_timestep_buckets: usize,
    pub max_num_embodiments: usize,
}

impl Default for Gr00tConfig {
    fn default() -> Self {
        Self {
            model_type: MODEL_TYPE.into(),
            model_dtype: BF16.into(),
            model_name: "nvidia/Cosmos-Reason2-2B".into(),
            backbone_model_type: "qwen".into(),
            backbone_embedding_dim: 2048,
            select_layer: 12,
            reproject_vision: false,
            use_flash_attention: true,
            image_crop_size: Some([230, 230]),
            image_target_size: Some([256, 256]),
            max_state_dim: 132,
            max_action_dim: 132,
            action_horizon: 40,
            hidden_size: 1024,
            input_embedding_dim: 1536,
            state_history_length: 1,
            add_pos_embed: true,
            use_vlln: true,
            max_seq_len: 1024,
            use_alternate_vl_dit: true,
            attend_text_every_n_blocks: 2,
            diffusion: Gr00tDiffusionConfig::default(),
            use_vl_self_attention: false,
            vl_self_attention: None,
            num_inference_timesteps: 4,
            num_timestep_buckets: 1000,
            max_num_embodiments: 32,
        }
    }
}

impl Gr00tConfig {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
        Self::from_json_str(&raw)
    }

    pub fn from_json_str(raw: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(raw)
            .map_err(|error| Error::Other(format!("GR00T config JSON: {error}")))?;
        let object = object(&value, "GR00T config")?;
        let mut config = Self::default();

        config.model_type = string_value(object, "model_type", &config.model_type)?;
        config.model_dtype =
            string_alias_value(object, &["model_dtype", "dtype"], &config.model_dtype)?;
        config.model_name = string_value(object, "model_name", &config.model_name)?;
        config.backbone_model_type =
            string_value(object, "backbone_model_type", &config.backbone_model_type)?;
        config.backbone_embedding_dim = usize_value(
            object,
            "backbone_embedding_dim",
            config.backbone_embedding_dim,
        )?;
        config.select_layer = usize_value(object, "select_layer", config.select_layer)?;
        config.reproject_vision = bool_value(object, "reproject_vision", config.reproject_vision)?;
        config.use_flash_attention =
            bool_value(object, "use_flash_attention", config.use_flash_attention)?;
        config.image_crop_size = optional_pair(object, "image_crop_size", config.image_crop_size)?;
        config.image_target_size =
            optional_pair(object, "image_target_size", config.image_target_size)?;
        config.max_state_dim = usize_value(object, "max_state_dim", config.max_state_dim)?;
        config.max_action_dim = usize_value(object, "max_action_dim", config.max_action_dim)?;
        config.action_horizon = usize_value(object, "action_horizon", config.action_horizon)?;
        config.hidden_size = usize_value(object, "hidden_size", config.hidden_size)?;
        config.input_embedding_dim =
            usize_value(object, "input_embedding_dim", config.input_embedding_dim)?;
        config.state_history_length =
            usize_value(object, "state_history_length", config.state_history_length)?;
        config.add_pos_embed = bool_value(object, "add_pos_embed", config.add_pos_embed)?;
        config.use_vlln = bool_value(object, "use_vlln", config.use_vlln)?;
        config.max_seq_len = usize_value(object, "max_seq_len", config.max_seq_len)?;
        config.use_alternate_vl_dit =
            bool_value(object, "use_alternate_vl_dit", config.use_alternate_vl_dit)?;
        config.attend_text_every_n_blocks = usize_value(
            object,
            "attend_text_every_n_blocks",
            config.attend_text_every_n_blocks,
        )?;
        if let Some(value) = object.get("diffusion_model_cfg") {
            config.diffusion.update_from_json(value)?;
        }
        config.use_vl_self_attention = bool_value(
            object,
            "use_vl_self_attention",
            config.use_vl_self_attention,
        )?;
        if let Some(value) = object.get("vl_self_attention_cfg") {
            config.vl_self_attention = Some(Gr00tVlSelfAttentionConfig::from_json(value)?);
        }
        config.num_inference_timesteps = usize_value(
            object,
            "num_inference_timesteps",
            config.num_inference_timesteps,
        )?;
        config.num_timestep_buckets =
            usize_value(object, "num_timestep_buckets", config.num_timestep_buckets)?;
        config.max_num_embodiments =
            usize_value(object, "max_num_embodiments", config.max_num_embodiments)?;

        config.validate()?;
        Ok(config)
    }

    pub fn state_input_dim(&self) -> Result<usize> {
        self.max_state_dim
            .checked_mul(self.state_history_length)
            .ok_or_else(|| Error::Other("GR00T state input dimension overflow".into()))
    }

    pub fn state_action_sequence_len(&self) -> Result<usize> {
        self.action_horizon
            .checked_add(1)
            .ok_or_else(|| Error::Other("GR00T state/action sequence length overflow".into()))
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_type != MODEL_TYPE {
            return Err(Error::Other(format!(
                "expected GR00T model_type {MODEL_TYPE}, got {}",
                self.model_type
            )));
        }
        if self.model_dtype != BF16 {
            return Err(Error::Other(format!(
                "GR00T N1.7 currently supports bfloat16, got {}",
                self.model_dtype
            )));
        }
        if self.backbone_model_type != "qwen" {
            return Err(Error::Other(format!(
                "GR00T N1.7 expects qwen backbone, got {}",
                self.backbone_model_type
            )));
        }
        let positive = [
            ("backbone_embedding_dim", self.backbone_embedding_dim),
            ("select_layer", self.select_layer),
            ("max_state_dim", self.max_state_dim),
            ("max_action_dim", self.max_action_dim),
            ("action_horizon", self.action_horizon),
            ("hidden_size", self.hidden_size),
            ("input_embedding_dim", self.input_embedding_dim),
            ("state_history_length", self.state_history_length),
            ("max_seq_len", self.max_seq_len),
            ("diffusion.num_layers", self.diffusion.num_layers),
            (
                "diffusion.num_attention_heads",
                self.diffusion.num_attention_heads,
            ),
            (
                "diffusion.attention_head_dim",
                self.diffusion.attention_head_dim,
            ),
            ("num_inference_timesteps", self.num_inference_timesteps),
            ("num_timestep_buckets", self.num_timestep_buckets),
            ("max_num_embodiments", self.max_num_embodiments),
        ];
        if let Some((name, _)) = positive.into_iter().find(|(_, value)| *value == 0) {
            return Err(Error::Other(format!("GR00T {name} must be non-zero")));
        }
        if self.diffusion.inner_dim()? != self.input_embedding_dim {
            return Err(Error::Other(format!(
                "GR00T diffusion width {}x{} does not match input_embedding_dim {}",
                self.diffusion.num_attention_heads,
                self.diffusion.attention_head_dim,
                self.input_embedding_dim
            )));
        }
        if self.diffusion.output_dim != self.hidden_size {
            return Err(Error::Other(format!(
                "GR00T diffusion output_dim {} does not match hidden_size {}",
                self.diffusion.output_dim, self.hidden_size
            )));
        }
        if self.diffusion.norm_type != "ada_norm" {
            return Err(Error::Other(format!(
                "GR00T N1.7 requires ada_norm, got {}",
                self.diffusion.norm_type
            )));
        }
        if self.diffusion.activation_fn != "gelu-approximate" {
            return Err(Error::Other(format!(
                "GR00T N1.7 BF16 runtime requires gelu-approximate, got {}",
                self.diffusion.activation_fn
            )));
        }
        if self.state_action_sequence_len()? > self.max_seq_len {
            return Err(Error::Other(format!(
                "GR00T state/action sequence length {} exceeds max_seq_len {}",
                self.state_action_sequence_len()?,
                self.max_seq_len
            )));
        }
        if self.use_alternate_vl_dit {
            if !self.diffusion.interleave_self_attention {
                return Err(Error::Other(
                    "GR00T AlternateVLDiT requires interleaved self-attention".into(),
                ));
            }
            if self.attend_text_every_n_blocks == 0 {
                return Err(Error::Other(
                    "GR00T attend_text_every_n_blocks must be non-zero".into(),
                ));
            }
        }
        if self.use_vl_self_attention {
            let vl = self.vl_self_attention.as_ref().ok_or_else(|| {
                Error::Other("GR00T enables VL self-attention without vl_self_attention_cfg".into())
            })?;
            if vl.num_layers == 0 {
                return Err(Error::Other(
                    "GR00T VL self-attention must contain at least one layer".into(),
                ));
            }
            if vl.hidden_size()? != self.backbone_embedding_dim {
                return Err(Error::Other(format!(
                    "GR00T VL self-attention width {} does not match backbone width {}",
                    vl.hidden_size()?,
                    self.backbone_embedding_dim
                )));
            }
        }
        Ok(())
    }
}

fn object<'a>(value: &'a Value, name: &str) -> Result<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| Error::Other(format!("{name} must be a JSON object")))
}

fn usize_value(object: &Map<String, Value>, key: &str, default: usize) -> Result<usize> {
    match object.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| Error::Other(format!("GR00T {key} must be a non-negative integer"))),
    }
}

fn f32_value(object: &Map<String, Value>, key: &str, default: f32) -> Result<f32> {
    match object.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_f64()
            .map(|value| value as f32)
            .filter(|value| value.is_finite())
            .ok_or_else(|| Error::Other(format!("GR00T {key} must be finite"))),
    }
}

fn bool_value(object: &Map<String, Value>, key: &str, default: bool) -> Result<bool> {
    match object.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| Error::Other(format!("GR00T {key} must be a boolean"))),
    }
}

fn string_value(object: &Map<String, Value>, key: &str, default: &str) -> Result<String> {
    match object.get(key) {
        None => Ok(default.into()),
        Some(value) => value
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| Error::Other(format!("GR00T {key} must be a string"))),
    }
}

fn string_alias_value(object: &Map<String, Value>, keys: &[&str], default: &str) -> Result<String> {
    for key in keys {
        if object.contains_key(*key) {
            return string_value(object, key, default);
        }
    }
    Ok(default.into())
}

fn optional_string(
    object: &Map<String, Value>,
    key: &str,
    default: Option<String>,
) -> Result<Option<String>> {
    match object.get(key) {
        None => Ok(default),
        Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_str()
            .map(|value| Some(value.to_owned()))
            .ok_or_else(|| Error::Other(format!("GR00T {key} must be a string or null"))),
    }
}

fn optional_pair(
    object: &Map<String, Value>,
    key: &str,
    default: Option<[usize; 2]>,
) -> Result<Option<[usize; 2]>> {
    let Some(value) = object.get(key) else {
        return Ok(default);
    };
    if value.is_null() {
        return Ok(None);
    }
    let values = value
        .as_array()
        .filter(|values| values.len() == 2)
        .ok_or_else(|| Error::Other(format!("GR00T {key} must contain exactly two integers")))?;
    let first = values[0]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| Error::Other(format!("GR00T {key}[0] must be an integer")))?;
    let second = values[1]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| Error::Other(format!("GR00T {key}[1] must be an integer")))?;
    Ok(Some([first, second]))
}

// ── Qwen3-VL backbone config ────────────────────────────────────────
//
// Parsed from the Cosmos-Reason2-2B `config.json` that ships beside the
// checkpoint. Not `apxinf_loader::ModelConfig`: Qwen3-VL's schema is nested
// (`text_config` / `vision_config`) and carries mRoPE sections.


/// Text-stack config for Qwen3-VL. Vision fields live under
/// `Qwen3VLVisionConfig` (added when the vision tower lands in Phase 4).
#[derive(Clone, Debug)]
pub struct Qwen3VLTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// `[T, H, W]` split of the 64 (= head_dim/2) frequency pairs.
    pub mrope_section: [usize; 3],
    pub mrope_interleaved: bool,
}

/// Vision-tower config for Qwen3-VL.
#[derive(Clone, Debug)]
pub struct Qwen3VLVisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub in_channels: usize,
    pub spatial_merge_size: usize,
    pub num_position_embeddings: usize,
    pub out_hidden_size: usize,
    /// Which 3 of the `depth` vision blocks feed the deepstack mergers.
    pub deepstack_visual_indexes: Vec<usize>,
    /// The vision MLP's activation. The port implements
    /// `"gelu_pytorch_tanh"`; [`Qwen3VLConfig::validate_activations`] rejects
    /// anything else rather than silently computing the wrong variant.
    ///
    /// The patch merger has no config entry: `transformers` hardcodes
    /// `nn.GELU()` there, which is the exact error function and not this.
    pub hidden_act: String,
}

impl Qwen3VLVisionConfig {
    pub fn head_dim(&self) -> usize {
        // HF: head_dim = hidden_size // num_heads (= 64 for Qwen3-VL-2B).
        if self.head_dim != 0 {
            self.head_dim
        } else {
            self.hidden_size / self.num_heads
        }
    }
}

/// Full Qwen3-VL config. `vision` is `None` until Phase 4.
#[derive(Clone, Debug)]
pub struct Qwen3VLConfig {
    pub text: Qwen3VLTextConfig,
    pub vision: Qwen3VLVisionConfig,
    pub image_token_id: u32,
}

/// A field that may be absent but must be well-formed when present -- a
/// wrong type, an explicit `null`, or a misspelled sibling key is an error,
/// never a silent default. The activation guard depends on this: with a
/// defaulting parser, `"hidden_act": null` or a misspelled key would parse
/// as the default tanh GELU and pass validation while the checkpoint was
/// configured otherwise.
fn optional<'v>(owner: &'v serde_json::Value, key: &str) -> Option<&'v serde_json::Value> {
    owner.get(key)
}

fn optional_u64(owner: &serde_json::Value, key: &str, default: u64) -> Result<u64> {
    match optional(owner, key) {
        None => Ok(default),
        Some(value) => value.as_u64().ok_or_else(|| {
            Error::Other(format!(
                "qwen3vl config: {key} must be a non-negative integer, got {value}"
            ))
        }),
    }
}

fn optional_f64(owner: &serde_json::Value, key: &str, default: f64) -> Result<f64> {
    match optional(owner, key) {
        None => Ok(default),
        Some(value) => value.as_f64().ok_or_else(|| {
            Error::Other(format!("qwen3vl config: {key} must be a number, got {value}"))
        }),
    }
}

fn optional_str<'v>(
    owner: &'v serde_json::Value,
    key: &str,
    default: &'v str,
) -> Result<&'v str> {
    match optional(owner, key) {
        None => Ok(default),
        Some(value) => value.as_str().ok_or_else(|| {
            Error::Other(format!("qwen3vl config: {key} must be a string, got {value}"))
        }),
    }
}

/// An array entry that the caller has already required to exist.
fn entry_u64(value: &serde_json::Value, name: &str) -> Result<u64> {
    value.as_u64().ok_or_else(|| {
        Error::Other(format!(
            "qwen3vl config: {name} must be a non-negative integer, got {value}"
        ))
    })
}

impl Qwen3VLConfig {
    /// Parse from a HuggingFace `config.json` file.
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| Error::Other(format!("read {}: {e}", path.display())))?;
        Self::from_json_str(&raw)
    }

    pub fn from_json_str(s: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s)
            .map_err(|e| Error::Other(format!("qwen3vl config json: {e}")))?;
        let tc = &v["text_config"];
        if !tc.is_object() {
            return Err(Error::Other("qwen3vl config: missing text_config".into()));
        }

        let rope_scaling = &tc["rope_scaling"];
        let section = rope_scaling["mrope_section"].as_array().ok_or_else(|| {
            Error::Other("qwen3vl config: missing rope_scaling.mrope_section".into())
        })?;
        if section.len() != 3 {
            return Err(Error::Other(format!(
                "qwen3vl config: mrope_section must be [T,H,W] (3 entries), got {}",
                section.len()
            )));
        }
        let mrope_section = [
            entry_u64(&section[0], "mrope_section[0]")? as usize,
            entry_u64(&section[1], "mrope_section[1]")? as usize,
            entry_u64(&section[2], "mrope_section[2]")? as usize,
        ];
        let mrope_interleaved = match optional(rope_scaling, "mrope_interleaved") {
            None => true,
            Some(value) => value.as_bool().ok_or_else(|| {
                Error::Other(format!(
                    "qwen3vl config: mrope_interleaved must be a boolean, got {value}"
                ))
            })?,
        };

        let text = Qwen3VLTextConfig {
            hidden_size: tc["hidden_size"].as_u64().unwrap_or(2048) as usize,
            intermediate_size: tc["intermediate_size"].as_u64().unwrap_or(6144) as usize,
            n_layers: tc["num_hidden_layers"].as_u64().unwrap_or(28) as usize,
            n_heads: tc["num_attention_heads"].as_u64().unwrap_or(16) as usize,
            n_kv_heads: tc["num_key_value_heads"].as_u64().unwrap_or(8) as usize,
            head_dim: tc["head_dim"].as_u64().unwrap_or(128) as usize,
            vocab_size: tc["vocab_size"].as_u64().unwrap_or(151936) as usize,
            max_position_embeddings: tc["max_position_embeddings"].as_u64().unwrap_or(262144)
                as usize,
            rms_norm_eps: optional_f64(tc, "rms_norm_eps", 1e-6)? as f32,
            rope_theta: optional_f64(tc, "rope_theta", 5_000_000.0)? as f32,
            mrope_section,
            mrope_interleaved,
        };

        let vc = &v["vision_config"];
        // A malformed deepstack index would silently vanish through
        // filter_map and leave the mergers reading the default blocks.
        let dvi = match optional(vc, "deepstack_visual_indexes") {
            None => vec![5, 11, 17],
            Some(value) => {
                let list = value.as_array().ok_or_else(|| {
                    Error::Other(format!(
                        "qwen3vl config: deepstack_visual_indexes must be a list, got {value}"
                    ))
                })?;
                let mut indexes = Vec::with_capacity(list.len());
                for (position, entry) in list.iter().enumerate() {
                    indexes.push(
                        entry_u64(entry, &format!("deepstack_visual_indexes[{position}]"))?
                            as usize,
                    );
                }
                indexes
            }
        };
        let vision = Qwen3VLVisionConfig {
            depth: vc["depth"].as_u64().unwrap_or(24) as usize,
            hidden_size: vc["hidden_size"].as_u64().unwrap_or(1024) as usize,
            intermediate_size: vc["intermediate_size"].as_u64().unwrap_or(4096) as usize,
            num_heads: vc["num_heads"].as_u64().unwrap_or(16) as usize,
            head_dim: vc
                .get("head_dim")
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(0),
            patch_size: vc["patch_size"].as_u64().unwrap_or(16) as usize,
            temporal_patch_size: vc["temporal_patch_size"].as_u64().unwrap_or(2) as usize,
            in_channels: vc["in_channels"].as_u64().unwrap_or(3) as usize,
            spatial_merge_size: vc["spatial_merge_size"].as_u64().unwrap_or(2) as usize,
            num_position_embeddings: vc["num_position_embeddings"].as_u64().unwrap_or(2304)
                as usize,
            out_hidden_size: vc["out_hidden_size"].as_u64().unwrap_or(2048) as usize,
            deepstack_visual_indexes: dvi,
            hidden_act: optional_str(vc, "hidden_act", "gelu_pytorch_tanh")?.to_owned(),
        };

        Ok(Qwen3VLConfig {
            text,
            vision,
            image_token_id: v["image_token_id"].as_u64().unwrap_or(151655) as u32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RELEASED_3B_CONFIG: &str = r#"
    {
      "model_type": "Gr00tN1d7",
      "dtype": "bfloat16",
      "model_dtype": "bfloat16",
      "model_name": "nvidia/Cosmos-Reason2-2B",
      "backbone_embedding_dim": 2048,
      "select_layer": 16,
      "max_state_dim": 132,
      "max_action_dim": 132,
      "action_horizon": 40,
      "hidden_size": 1024,
      "max_seq_len": 1024,
      "use_alternate_vl_dit": true,
      "use_vl_self_attention": true,
      "vl_self_attention_cfg": {
        "num_layers": 4,
        "num_attention_heads": 32,
        "attention_head_dim": 64,
        "dropout": 0.2,
        "final_dropout": true,
        "positional_embeddings": null
      },
      "diffusion_model_cfg": {
        "num_layers": 32,
        "num_attention_heads": 32,
        "attention_head_dim": 48,
        "norm_type": "ada_norm",
        "dropout": 0.2,
        "final_dropout": true,
        "output_dim": 1024,
        "interleave_self_attention": true,
        "positional_embeddings": null
      },
      "num_inference_timesteps": 4,
      "num_timestep_buckets": 1000,
      "max_num_embodiments": 32
    }
    "#;

    #[test]
    fn parses_released_3b_overrides_and_constructor_defaults() {
        let config = Gr00tConfig::from_json_str(RELEASED_3B_CONFIG).unwrap();
        assert_eq!(config.select_layer, 16);
        assert_eq!(config.diffusion.num_layers, 32);
        assert_eq!(config.diffusion.activation_fn, "gelu-approximate");
        assert_eq!(config.input_embedding_dim, 1536);
        assert_eq!(config.state_history_length, 1);
        assert_eq!(config.attend_text_every_n_blocks, 2);
        assert_eq!(config.diffusion.inner_dim().unwrap(), 1536);
        assert_eq!(config.vl_self_attention.unwrap().num_layers, 4);
    }

    #[test]
    fn rejects_diffusion_width_mismatch() {
        let raw =
            RELEASED_3B_CONFIG.replace("\"attention_head_dim\": 48", "\"attention_head_dim\": 32");
        let error = Gr00tConfig::from_json_str(&raw).unwrap_err();
        assert!(error
            .to_string()
            .contains("does not match input_embedding_dim"));
    }

    #[test]
    fn rejects_wrong_model_and_dtype() {
        let wrong_model = RELEASED_3B_CONFIG.replace("Gr00tN1d7", "Gr00tN1d6");
        assert!(Gr00tConfig::from_json_str(&wrong_model).is_err());

        let wrong_dtype = RELEASED_3B_CONFIG.replace("bfloat16", "float16");
        assert!(Gr00tConfig::from_json_str(&wrong_dtype).is_err());
    }

    use super::*;

    // Minimal HF-shaped config, matching Qwen3-VL-2B-Instruct.
    const MIN_CONFIG: &str = r#"{
        "image_token_id": 151655,
        "video_token_id": 151656,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "text_config": {
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "rope_theta": 5000000,
            "tie_word_embeddings": true,
            "rope_scaling": {
                "mrope_interleaved": true,
                "mrope_section": [24, 20, 20]
            }
        }
    }"#;

    #[test]
    fn parses_qwen3vl_2b_config() {
        let cfg = Qwen3VLConfig::from_json_str(MIN_CONFIG).unwrap();
        assert_eq!(cfg.text.n_layers, 28);
        assert_eq!(cfg.text.head_dim, 128);
        assert_eq!(cfg.text.n_heads, 16);
        assert_eq!(cfg.text.n_kv_heads, 8);
        assert_eq!(cfg.text.mrope_section, [24, 20, 20]);
        assert!(cfg.text.mrope_interleaved);
        assert_eq!(cfg.text.rope_theta, 5_000_000.0);
        assert_eq!(cfg.image_token_id, 151655);
    }

    /// A field that is absent takes the default; a field that is present but
    /// malformed -- including an explicit `null` -- is an error. The
    /// activation guard is only as good as this: `"hidden_act": null` or a
    /// misspelled key used to parse as the default tanh GELU and pass
    /// validation while the checkpoint was configured otherwise.
    #[test]
    fn present_but_malformed_fields_are_errors_not_defaults() {
        // Absent (no vision_config section at all): the default stands.
        let base = Qwen3VLConfig::from_json_str(MIN_CONFIG).unwrap();
        assert_eq!(base.vision.hidden_act, "gelu_pytorch_tanh");

        for (label, target, patch) in [
            ("null hidden_act", "\"text_config\": {", "\"vision_config\": {\"hidden_act\": null}, \"text_config\": {"),
            ("non-string hidden_act", "\"text_config\": {", "\"vision_config\": {\"hidden_act\": 5}, \"text_config\": {"),
            ("non-numeric rope_theta", "\"rope_theta\": 5000000", "\"rope_theta\": \"big\""),
            ("null rms_norm_eps", "\"rms_norm_eps\": 1e-6", "\"rms_norm_eps\": null"),
            ("null mrope_section entry", "\"mrope_section\": [24, 20, 20]", "\"mrope_section\": [24, null, 20]"),
            ("null mrope_interleaved", "\"mrope_interleaved\": true", "\"mrope_interleaved\": null"),
        ] {
            let raw = MIN_CONFIG.replace(target, patch);
            let error = Qwen3VLConfig::from_json_str(&raw)
                .unwrap_err()
                .to_string();
            assert!(error.contains("must be"), "{label}: {error}");
        }

        let bad_deepstack = MIN_CONFIG.replace(
            "\"text_config\": {",
            "\"vision_config\": {\"deepstack_visual_indexes\": [5, \"x\"]}, \"text_config\": {",
        );
        let error = Qwen3VLConfig::from_json_str(&bad_deepstack)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("deepstack_visual_indexes[1]"),
            "malformed deepstack entry: {error}"
        );
    }
}
